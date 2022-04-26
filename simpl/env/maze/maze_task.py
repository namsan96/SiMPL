from pathlib import Path

from .maze import MazeTask

    
def load_goals(file_name):
    with open(Path(__file__).resolve().parent / 'asset' / (file_name + '.csv')) as f:
        f.readline()
        lines = f.readlines()
    return [[float(c) for c in l.split(',')] for l in lines]


size20_init_loc = [10, 10]
flat_40t_train_goals = load_goals('flat_40t_train_goals')
flat_20t_train_goals = load_goals('flat_20t_train_goals')
flat_10t_train_goals = load_goals('flat_10t_train_goals')
flat_test_goals = load_goals('flat_test_goals')


class Size20Seed0Tasks:
    flat_40t_train_tasks = [
        MazeTask(size20_init_loc, goal_loc)
        for goal_loc in flat_40t_train_goals
    ]
    flat_20t_train_tasks = [
        MazeTask(size20_init_loc, goal_loc)
        for goal_loc in flat_20t_train_goals
    ]
    flat_10t_train_tasks = [
        MazeTask(size20_init_loc, goal_loc)
        for goal_loc in flat_10t_train_goals
    ]
    flat_test_tasks = [
        MazeTask(size20_init_loc, goal_loc)
        for goal_loc in flat_test_goals
    ]
