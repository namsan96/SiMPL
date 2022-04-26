import argparse
import importlib

import matplotlib.pyplot as plt
import torch
import wandb

from simpl.alg.spirl import ConstrainedSAC, PriorResidualNormalMLPPolicy
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector
from simpl.nn import itemize
from simpl.alg.simpl import ConditionedPolicy, ConditionedQF


def simpl_fine_tune_iter(collector, trainer, *, batch_size, reuse_rate):
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
        'maze': 'maze.simpl_fine_tune',
        'kitchen': 'kitchen.simpl_fine_tune',
    }
    
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=import_pathes.keys())
    parser.add_argument('-g', '--gpu', required=True, type=int)
    parser.add_argument('-m', '--simpl-metatrained-path', required=True)
    parser.add_argument('-s', '--spirl-pretrained-path', required=True)
    
    parser.add_argument('-t', '--policy-vis_period', type=int)
    parser.add_argument('-p', '--wandb-project-name')
    parser.add_argument('-r', '--wandb-run-name')
    args = parser.parse_args()

    module = importlib.import_module(import_pathes[args.domain])
    env, tasks, config, visualize_env = module.env, module.tasks, module.config, module.visualize_env

    gpu = args.gpu
    simpl_metatrained_path = args.simpl_metatrained_path
    spirl_pretrained_path = args.spirl_pretrained_path
    policy_vis_period = args.policy_vis_period or 20
    wandb_project_name = args.wandb_project_name or 'SiMPL'
    wandb_run_name = args.wandb_run_name or args.domain + '.simpl_fine_tune.' + wandb.util.generate_id()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load pre-trained SPiRL
    load = torch.load(spirl_pretrained_path, map_location='cpu')
    horizon = load['horizon']
    high_action_dim = load['z_dim']
    spirl_low_policy = load['spirl_low_policy'].to(gpu).eval().requires_grad_(False)
    spirl_prior_policy = load['spirl_prior_policy'].to(gpu).eval().requires_grad_(False)

    # load meta-trained SiMPL
    load = torch.load(simpl_metatrained_path, map_location='cpu')
    simpl_encoder = load['encoder'].to(gpu)
    simpl_high_policy = load['high_policy'].to(gpu)
    simpl_qfs = [qf.to(gpu) for qf in load['qfs']]
    simpl_alpha = load['policy_post_reg']

    # collector
    spirl_low_policy.explore = False
    collector = LowFixedHierarchicalTimeLimitCollector(env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit'])

    # train on all tasks
    for task_idx, task in enumerate(tasks):
        wandb.init(
            project=wandb_project_name, name=wandb_run_name,
            config={
                **config, 'task_idx': task_idx,
                'spirl_pretrained_path': args.spirl_pretrained_path,
                'simpl_metatrained_path': args.simpl_metatrained_path
            }
        )
        
        with env.set_task(task):
            # collect from prior policy & encode
            prior_episodes = []
            for _ in range(config['n_prior_episode']):
                e = simpl_encoder.encode([], sample=True)
                with simpl_high_policy.expl(), simpl_high_policy.condition(e):
                    episode = collector.collect_episode(simpl_high_policy)
                prior_episodes.append(episode)
            e = simpl_encoder.encode([episode.as_high_episode().as_batch() for episode in prior_episodes], sample=False)

            # ready networks
            high_policy = ConditionedPolicy(simpl_high_policy, e)
            qfs = [ConditionedQF(qf, e) for qf in simpl_qfs]
            buffer = Buffer(state_dim, high_action_dim, config['buffer_size'])
            trainer = ConstrainedSAC(high_policy, spirl_prior_policy, qfs, buffer, init_alpha=simpl_alpha, **config['constrained_sac']).to(gpu)

            for episode_i in range(config['n_prior_episode']+1, config['n_episode']+1):
                log = simpl_fine_tune_iter(collector, trainer, **config['train'])
                log['episode_i'] = episode_i
                if episode_i % policy_vis_period == 0:
                    plt.close('all')
                    plt.figure()
                    log['policy_vis'] = visualize_env(plt.gca(), env, list(buffer.episodes)[-20:])
                wandb.log(log)

        wandb.finish()

