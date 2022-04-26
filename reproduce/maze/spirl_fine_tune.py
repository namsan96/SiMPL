import gym
import simpl.env.maze
from simpl.env.maze import Size20Seed0Tasks

from .maze_vis import draw_maze


env = gym.make('simpl-maze-size20-seed0-v0')
tasks = Size20Seed0Tasks.flat_test_tasks
config = dict(
    policy=dict(hidden_dim=256, n_hidden=3),
    qf=dict(hidden_dim=256, n_hidden=3),
    n_qf=2,
    constrained_sac=dict(auto_alpha=True, init_alpha=0.05,
                         kl_clip=10, target_kl=1),
    buffer_size=20000,
    time_limit=2000,
    n_episode=500,
    train=dict(batch_size=256, reuse_rate=256)
)
visualize_env = draw_maze
