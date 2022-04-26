from pathlib import Path

from .kitchen import all_tasks, KitchenTask

    
def load_tasks(file_name):
    with open(Path(__file__).resolve().parent / 'asset' / (file_name + '.csv')) as f:
        f.readline()
        lines = f.readlines()
    return [KitchenTask([all_tasks[int(c)] for c in l.split(',')]) for l in lines]

train_tasks = load_tasks('train_tasks')
test_tasks = load_tasks('test_tasks')


class KitchenTasks:
    train_tasks = train_tasks
    test_tasks = test_tasks
