import gym
import simpl.env.maze
from simpl.env.kitchen import KitchenTasks

from .kitchen_vis import draw_kitchen


env = gym.make('simpl-kitchen-v0')
tasks = KitchenTasks.test_tasks
config = dict(
    policy=dict(hidden_dim=128, n_hidden=5),
    qf=dict(hidden_dim=128, n_hidden=5),
    n_qf=2,
    constrained_sac=dict(auto_alpha=True, init_alpha=1,
                         kl_clip=10, target_kl=5, alpha_lr=3e-3),
    buffer_size=20000,
    time_limit=280,
    n_episode=1000,
    train=dict(batch_size=256, reuse_rate=256)
)
visualize_env = draw_kitchen
