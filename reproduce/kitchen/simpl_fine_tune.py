import gym
import simpl.env.kitchen
from simpl.env.kitchen import KitchenTasks

from .kitchen_vis import draw_kitchen


env = gym.make('simpl-kitchen-v0')
tasks = KitchenTasks.test_tasks
config = dict(
    constrained_sac=dict(auto_alpha=True, kl_clip=20,
                         target_kl=5, increasing_alpha=True),
    buffer_size=20000,
    n_prior_episode=20,
    time_limit=280,
    n_episode=1000,
    train=dict(batch_size=256, reuse_rate=256)
)
visualize_env = draw_kitchen
