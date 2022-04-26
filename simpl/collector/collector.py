import numpy as np

from .storage import Episode


class TimeLimitCollector:
    def __init__(self, env, time_limit=None):
        self.env = env
        self.time_limit = time_limit if time_limit is not None else np.inf

    def collect_episode(self, actor):
        state, done, t = self.env.reset(), False, 0
        episode = Episode(state)

        while not done and t < self.time_limit:
            action = actor.act(state)
            state, reward, done, info = self.env.step(action)
            data_done = done
            if 'TimeLimit.truncated' in info:
                data_done = not info['TimeLimit.truncated']
            episode.add_step(action, state, reward, data_done, info)
            t += 1
        return episode
