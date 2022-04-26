import gym


gym.register(
    id='simpl-maze-size20-seed0-v0',
    entry_point='simpl.env.maze:MazeEnv',
    kwargs={
        'size':20,
        'seed': 0,
        'reward_type':'sparse',
        'done_on_completed': True,
    }
)
