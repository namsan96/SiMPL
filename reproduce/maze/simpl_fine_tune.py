import gym
import simpl.env.maze
from simpl.env.maze import Size20Seed0Tasks

from .maze_vis import draw_maze


env = gym.make('simpl-maze-size20-seed0-v0')
tasks = Size20Seed0Tasks.flat_test_tasks
config = dict(
    constrained_sac=dict(auto_alpha=True, kl_clip=5,
                         target_kl=1, increasing_alpha=True),
    buffer_size=20000,
    n_prior_episode=20,
    time_limit=2000,
    n_episode=1000,
    train=dict(batch_size=256, reuse_rate=256)
)
visualize_env = draw_maze
