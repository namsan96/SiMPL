import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from simpl.alg.simpl import ContextPriorResidualNormalMLPPolicy, LowFixedGPUWorker, Simpl
from simpl.alg.pearl import SetTransformerEncoder
from simpl.collector import Buffer, LowFixedHierarchicalTimeLimitCollector, ConcurrentCollector
from simpl.nn import itemize
from simpl.rl import MLPQF


def simpl_warm_up_buffer(
    conc_collector, policy, train_tasks, enc_buffers, buffers,
    post_enc_size, batch_size
):

    task_indices = range(len(train_tasks))
    while len(task_indices) > 0:
        for task_idx in task_indices:
            z = encoder.encode([], sample=True)
            with policy.expl(), policy.condition(z):
                conc_collector.submit(train_tasks[task_idx], policy)

        episodes = conc_collector.wait()

        for episode, task_idx in zip(episodes, task_indices):
            enc_buffers[task_idx].enqueue(episode.as_high_episode())

        task_indices = [
            task_idx for task_idx, enc_buffer in enumerate(enc_buffers)
            if enc_buffer.size < post_enc_size
        ]

    task_indices = range(len(train_tasks))
    while len(task_indices) > 0:
        for task_idx in task_indices:
            z = encoder.encode([enc_buffers[task_idx].sample(post_enc_size)], sample=True)
            with policy.expl(), policy.condition(z):
                conc_collector.submit(train_tasks[task_idx], policy)

        episodes = conc_collector.wait()

        for episode, task_idx in zip(episodes, task_indices):
            buffers[task_idx].enqueue(episode.as_high_episode())

        task_indices = [
            task_idx for task_idx, buffer in enumerate(buffers)
            if buffer.size < batch_size
        ]


def simpl_meta_train_iter(
    conc_collector, trainer, train_tasks, *,
    batch_size, reuse_rate,
    n_prior_batch, n_post_batch, prior_enc_size, post_enc_size
):
    log = {}

    # collect
    device = trainer.device
    trainer.policy.to('cpu')

    # - from task prior
    for task in train_tasks:
        z = encoder.encode([], sample=True)
        with trainer.policy.expl(), trainer.policy.condition(z):
            conc_collector.submit(task, trainer.policy)
    prior_episodes = conc_collector.wait()
    prior_high_episodes = [episode.as_high_episode() for episode in prior_episodes]
    for high_episode, enc_buffer, buffer in zip(prior_high_episodes, trainer.enc_buffers, trainer.buffers):
        enc_buffer.enqueue(high_episode)
        buffer.enqueue(high_episode)

    # - from task posterior
    for task, enc_buffer in zip(train_tasks, enc_buffers):
        z = encoder.encode([enc_buffer.sample(post_enc_size)], sample=True)
        with trainer.policy.expl(), trainer.policy.condition(z):
            conc_collector.submit(task, trainer.policy)
    post_episodes = conc_collector.wait()
    post_high_episodes = [episode.as_high_episode() for episode in post_episodes]
    for high_episode, buffer in zip(post_high_episodes, trainer.buffers):
        buffer.enqueue(high_episode)

    tr_returns = [sum(episode.rewards) for episode in post_episodes]
    log.update({
        'avg_tr_return': np.mean(tr_returns),
        'tr_return': {task_idx: tr_return for task_idx, tr_return in enumerate(tr_returns)}
    })

    trainer.policy.to(device)

    # meta train
    n_prior = sum([len(episode) for episode in prior_high_episodes])
    n_post = sum([len(episode) for episode in post_high_episodes])
    n_step = reuse_rate * (n_prior + n_post) / (n_prior_batch + n_post_batch) / batch_size
    for _ in range(max(int(n_step), 1)):
        stat = trainer.step(n_prior_batch, n_post_batch, batch_size, prior_enc_size, post_enc_size)
    log.update(itemize(stat))

    return log


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    import_pathes = {
        'maze_40t': 'maze.simpl_meta_train_40t',
        'maze_20t': 'maze.simpl_meta_train_20t',
        'kitchen': 'kitchen.simpl_meta_train',
    }

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('domain', choices=import_pathes.keys())
    parser.add_argument('-g', '--gpu', required=True, type=int)
    parser.add_argument('-w', '--worker-gpus', required=True, type=int, nargs='+')
    parser.add_argument('-s', '--spirl-pretrained-path', required=True)

    parser.add_argument('-t', '--policy-vis_period', type=int)
    parser.add_argument('-p', '--wandb-project-name')
    parser.add_argument('-r', '--wandb-run-name')
    parser.add_argument('-a', '--save_file_path')
    args = parser.parse_args()

    module = importlib.import_module(import_pathes[args.domain])
    env, train_tasks, config, visualize_env = module.env, module.train_tasks, module.config, module.visualize_env

    gpu = args.gpu
    worker_gpus = args.worker_gpus
    spirl_pretrained_path = args.spirl_pretrained_path 
    policy_vis_period = args.policy_vis_period or 10
    wandb_project_name = args.wandb_project_name or 'SiMPL'
    wandb_run_name = args.wandb_run_name or args.domain + '.simpl_meta_train.' + wandb.util.generate_id()
    save_filepath = args.save_file_path or f'./{wandb_run_name}.pt'

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load pre-trained SPiRL
    load = torch.load(spirl_pretrained_path, map_location='cpu')
    horizon = load['horizon']
    high_action_dim = load['z_dim']
    spirl_low_policy = load['spirl_low_policy'].eval().requires_grad_(False)
    spirl_prior_policy = load['spirl_prior_policy'].eval().requires_grad_(False)

    # collector
    spirl_low_policy.explore = False
    collector = LowFixedHierarchicalTimeLimitCollector(
        env, spirl_low_policy, horizon=horizon, time_limit=config['time_limit']
    )
    conc_collector = ConcurrentCollector([
        LowFixedGPUWorker(collector, gpu)
        for gpu in worker_gpus
    ])

    # ready networks
    encoder = SetTransformerEncoder(state_dim, high_action_dim, config['e_dim'], **config['encoder'])
    high_policy = ContextPriorResidualNormalMLPPolicy(
        spirl_prior_policy, state_dim, high_action_dim, config['e_dim'],
        **config['policy']
    )
    qfs = [MLPQF(state_dim+config['e_dim'], high_action_dim, **config['qf']) for _ in range(config['n_qf'])]

    # ready buffers
    enc_buffers = [
        Buffer(state_dim, high_action_dim, config['enc_buffer_size'])
        for _ in range(len(train_tasks))
    ]
    buffers = [
        Buffer(state_dim, high_action_dim, config['buffer_size'])
        for _ in range(len(train_tasks))
    ]
    simpl_warm_up_buffer(
        conc_collector, high_policy, train_tasks, enc_buffers, buffers,
        config['train']['post_enc_size'], config['train']['batch_size']
    )

    # meta train
    trainer = Simpl(high_policy, spirl_prior_policy, qfs, encoder, enc_buffers, buffers, **config['simpl']).to(gpu)

    wandb.init(
        project=wandb_project_name, name=wandb_run_name,
        config={**config, 'gpu': gpu, 'spirl_pretrained_path': args.spirl_pretrained_path}
    )
    for epoch_i in range(1, config['n_epoch']+1):
        log = simpl_meta_train_iter(conc_collector, trainer, train_tasks, **config['train'])
        log['epoch_i'] = epoch_i
        if epoch_i % policy_vis_period == 0:
            plt.close('all')
            n_row = int(np.ceil(len(train_tasks)/10))
            fig, axes = plt.subplots(n_row, 10, figsize=(20, 2*n_row))
            for task_idx, (task, buffer) in enumerate(zip(train_tasks, buffers)):
                with env.set_task(task):
                    visualize_env(axes[task_idx//10][task_idx%10], env, list(buffer.episodes)[-20:])
            log['policy_vis'] = fig
        wandb.log(log)

    torch.save({
        'encoder': encoder,
        'high_policy': high_policy,
        'qfs': qfs,
        'policy_post_reg': trainer.policy_post_reg().item()
    }, save_filepath)
    conc_collector.close()
