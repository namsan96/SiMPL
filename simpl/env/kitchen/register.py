import gym


gym.register(
    id='simpl-kitchen-v0',
    entry_point='simpl.env.kitchen:KitchenEnv'
)
