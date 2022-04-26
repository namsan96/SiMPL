'''
Maze layout sampling code from :
https://github.com/rail-berkeley/d4rl/tree/4aff6f8c46f62f9a57f79caa9287efefa45b6688
'''

import numpy as np
from scipy.signal import convolve2d
import re


def compute_sampling_probs(maze_layout, filter, temp):
    probs = convolve2d(maze_layout, filter, 'valid')
    return np.exp(-temp*probs) / np.sum(np.exp(-temp*probs))


def sample_2d(probs, rng):
    flat_probs = probs.flatten()
    sample = rng.choice(np.arange(flat_probs.shape[0]), p=flat_probs)
    sampled_2d = np.zeros_like(flat_probs)
    sampled_2d[sample] = 1
    idxs = np.where(sampled_2d.reshape(probs.shape))
    return idxs[0][0], idxs[1][0]


def place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp):
    """Samples wall such that overlap with other walls is minimized (overlap is determined by temperature).
       Also adds one door per wall."""
    size = maze_layout.shape[0]
    sample_vert_hor = 0 if rng.random() < 0.5 else 1
    sample_len = int(max((max_len_frac-min_len_frac) * size * rng.random() + min_len_frac*size, 3))
    sample_door_offset = rng.choice(np.arange(1, sample_len - 1))

    if sample_vert_hor == 0:
        filter = np.ones((sample_len, 5)) / (5*sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[0]
        sample_pos2 = middle_idxs[1] + 2

        maze_layout[sample_pos1 : sample_pos1 + sample_len, sample_pos2] = 1
        maze_layout[sample_pos1 + sample_door_offset, sample_pos2] = 0
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset - 1, sample_pos2 - 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 + 1] = 1
        maze_layout[sample_pos1 + sample_door_offset + 1, sample_pos2 - 1] = 1
    else:
        filter = np.ones((5, sample_len)) / (5 * sample_len)
        probs = compute_sampling_probs(maze_layout, filter, temp)
        middle_idxs = sample_2d(probs, rng)
        sample_pos1 = middle_idxs[1]
        sample_pos2 = middle_idxs[0] + 2

        maze_layout[sample_pos2, sample_pos1: sample_pos1 + sample_len] = 1
        maze_layout[sample_pos2, sample_pos1 + sample_door_offset] = 0
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset - 1] = 1
        maze_layout[sample_pos2 + 1, sample_pos1 + sample_door_offset + 1] = 1
        maze_layout[sample_pos2 - 1, sample_pos1 + sample_door_offset + 1] = 1
    return maze_layout


def sample_layout(seed=None,
                  size=20,
                  max_len_frac=0.5,
                  min_len_frac=0.3,
                  coverage_frac=0.25,
                  temp=20):
    """
    Generates maze layout with randomly placed walls.
    :param seed: if not None, makes maze layout reproducible
    :param size: number of cells per side in maze
    :param max_len_frac: maximum length of walls, as fraction of total maze side length
    :param min_len_frac: minimum length of walls, as fraction of total maze side length
    :param coverage_frac: fraction of cells that is covered with walls in randomly generated layout
    :param temp: controls overlap of walls in maze, the higher the temp the less the overlap of walls
    :return: layout matrix (where 1 indicates wall, 0 indicates free space)
    """
    rng = np.random.default_rng(seed=seed)
    maze_layout = np.zeros((size, size))

    while np.mean(maze_layout) < coverage_frac:
        maze_layout = place_wall(maze_layout, rng, min_len_frac, max_len_frac, temp)

    return maze_layout


def layout2str(layout):
    """Transfers a layout matrix to string format that is used by MazeEnv class."""
    h, w = layout.shape
    padded_layout = np.ones((h+2, w+2))
    padded_layout[1:-1, 1:-1] = layout
    output_str = ""
    for row in padded_layout:
        for cell in row:
            output_str += "O" if cell == 0 else "#"
        output_str += "\\"
    output_str = output_str[:-1]    # remove last line break
    output_str = re.sub("O", "G", output_str, count=1)   # add goal at random position
    return output_str


def rand_layout(seed=None, **kwargs):
    """Generates random layout with specified params (see 'sample_layout' function)."""
    rand_layout = sample_layout(seed, **kwargs)
    layout_str = layout2str(rand_layout)
    return layout_str
