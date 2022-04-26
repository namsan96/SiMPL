import argparse
import importlib

import matplotlib.pyplot as plt
import torch
import wandb

from simpl.alg.spirl import ConstrainedSAC, PriorResidualNormalMLPPolicy
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector
from simpl.nn import itemize
from simpl.rl import MLPQF


def spirl_fine_tune_iter(collector, trainer, *, batch_size, reuse_rate):
    log = {}

    # collect
    with trainer.policy.expl():
        episode = collector.collect_episode(trainer.policy)
    high_episode = episode.as_high_episode()
    trainer.buffer.enqueue(high_episode)
    log['tr_return'] = sum(episode.rewards)

    if trainer.buffer.size < batch_size:
        return log

    # train
    n_step = int(reuse_rate * len(high_episode) / batch_size)
    for _ in range(max(n_step, 1)):
        stat = trainer.step(batch_size)
    log.update(itemize(stat))

    return log


if __name__ == '__main__':
    import_pathes = {
        'maze': 'maze.spirl_fine_tune',
        'kitchen': 'kitchen.spirl_fine_tune',
    }

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=import_pathes.keys())
    parser.add_argument('--gpu', '-g', required=True, type=int)
    parser.add_argument('--spirl-pretrained-path', '-s', required=True)
    
    parser.add_argument('--policy-vis-period', '-f', type=int)
    parser.add_argument('--wandb-project-name', '-p')
    parser.add_argument('--wandb-run-name', '-r')
    args = parser.parse_args()

    module = importlib.import_module(import_pathes[args.domain])
    env, tasks, config, visualize_env = module.env, module.tasks, module.config, module.visualize_env
    
    gpu = args.gpu
    spirl_pretrained_path = args.spirl_pretrained_path
    
    policy_vis_period = args.policy_vis_period or 20
    wandb_project_name = args.wandb_project_name or 'SiMPL'
    wandb_run_name = args.wandb_run_name or args.domain + '.spirl_fine_tune.' + wandb.util.generate_id()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load pre-trained SPiRL
    load = torch.load(spirl_pretrained_path, map_location='cpu')
    horizon = load['horizon']
    high_action_dim = load['z_dim']
    spirl_low_policy = load['spirl_low_policy'].to(gpu).eval().requires_grad_(False)
    spirl_prior_policy = load['spirl_prior_policy'].to(gpu).eval().requires_grad_(False)

    # collector
    spirl_low_policy.explore = False
    collector = LowFixedHierarchicalTimeLimitCollector(env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit'])

    # train on all tasks
    for task_idx, task in enumerate(tasks):
        high_policy = PriorResidualNormalMLPPolicy(spirl_prior_policy, state_dim, high_action_dim, **config['policy'])
        qfs = [MLPQF(state_dim, high_action_dim, **config['qf']) for _ in range(config['n_qf'])]
        buffer = Buffer(state_dim, high_action_dim, config['buffer_size'])
        trainer = ConstrainedSAC(high_policy, spirl_prior_policy, qfs, buffer, **config['constrained_sac']).to(gpu)

        wandb.init(
            project=wandb_project_name, name=wandb_run_name,
            config={**config, 'task_idx': task_idx}
        )

        with env.set_task(task):
            for episode_i in range(1, config['n_episode']+1):
                log = spirl_fine_tune_iter(collector, trainer, **config['train'])
                log['episode_i'] = episode_i
                if episode_i % policy_vis_period == 0:
                    plt.close('all')
                    plt.figure()
                    log['policy_vis'] = visualize_env(plt.gca(), env, list(buffer.episodes)[-20:])
                wandb.log(log)

        wandb.finish()

