import matplotlib.pyplot as plt
import numpy as np

from simpl.env.kitchen import all_tasks, extract_path


def draw_kitchen(ax, env, episodes):
    ax.plot([1]*len(all_tasks), all_tasks, linewidth=0) # consistent y order
    ax.plot(np.arange(1, 5), env.task.subtasks, linestyle=':', color='red', marker='o')
    
    for episode in episodes:
        path = extract_path(episode.raw_episode)
        if len(path) > 0:
            ax.plot(
                np.arange(1, len(path)+1) + np.random.random()/10,
                path, c='royalblue', marker='o', alpha=0.5
            )
    return ax
