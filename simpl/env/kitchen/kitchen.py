from collections import defaultdict
from contextlib import contextmanager

from dm_control.mujoco import engine
import mujoco_py
import numpy as np
from d4rl.kitchen.kitchen_envs import KitchenBase, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH
from d4rl.kitchen.adept_envs import mujoco_env


mujoco_env.USE_DM_CONTROL = False
all_tasks = ['bottom burner', 'top burner', 'light switch', 'slide cabinet', 'hinge cabinet', 'microwave', 'kettle']


class KitchenTask:
    def __init__(self, subtasks):
        for subtask in subtasks:
            if subtask not in all_tasks:
                raise ValueError(f'{subtask} is not valid subtask')
        self.subtasks = subtasks

    def __repr__(self):
        return f"MTKitchenTask({' -> '.join(self.subtasks)})"


class KitchenEnv(KitchenBase):
    render_width = 400
    render_height = 400
    render_device = -1

    def __init__(self, *args, **kwargs):
        self.TASK_ELEMENTS = ['top burner']  # for initialization
        super().__init__(*args, **kwargs)
        
        self.task = None
        self.TASK_ELEMENTS = None
    
    @contextmanager
    def set_task(self, task):
        if type(task) != KitchenTask:
            raise TypeError(f'task should be KitchenTask but {type(task)} is given')

        prev_task = self.task
        prev_task_elements = self.TASK_ELEMENTS
        self.task = task
        self.TASK_ELEMENTS = task.subtasks
        yield
        self.task = prev_task
        self.TASK_ELEMENTS = prev_task_elements
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        self.render_width = width
        self.render_height = height
        self.render_device = device
        self.metadata['video.frames_per_second'] = fps
        self.metadata['video.frame_drop'] = frame_drop

    def _get_task_goal(self, task=None):
        if task is None:
            task = ['microwave', 'kettle', 'bottom burner', 'light switch']
        new_goal = np.zeros_like(self.goal)
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal
        return new_goal

    def compute_reward(self, obs_dict):
        reward_dict = {}
        
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = self._get_task_goal(task=self.TASK_ELEMENTS)
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] -
                next_goal[element_idx])
            complete = distance < BONUS_THRESH
            if complete and all_completed_so_far:
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        for completion in completions:
            self.tasks_to_complete.remove(completion)
        reward = float(len(completions))
        return reward
    
    def reset_model(self):
        ret = super().reset_model()
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return ret

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        if not self.initializing:
            a = self.act_mid + a * self.act_amp

        self.robot.step(self, a, step_duration=self.skip * self.model.opt.timestep)

        obs = self._get_obs()
        reward = self.compute_reward(self.obs_dict)
        done = not self.tasks_to_complete
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
        }
        return obs, reward, done, env_info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            if not hasattr(self, 'viewer'):
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=self.render_device)
                
                self.viewer.cam.lookat[0] = -0.2
                self.viewer.cam.lookat[1] = .5
                self.viewer.cam.lookat[2] = 2.
                self.viewer.cam.distance = 2.2
                self.viewer.cam.azimuth = 70
                self.viewer.cam.elevation = -35
            self.viewer.render(self.render_width, self.render_height)
            return self.viewer.read_pixels(self.render_width, self.render_height, depth=False)[::-1, :, :]
        else:
            raise NotImplementedError

            
def extract_path(episode):
    obj_qps = np.array([info['obs_dict']['obj_qp'] for info in episode.infos])

    completes = {}
    for task in all_tasks:
        ds = np.linalg.norm(
            obj_qps[:, OBS_ELEMENT_INDICES[task]-9] - OBS_ELEMENT_GOALS[task][None, :],
            axis=-1
        )
        completes[task] = ds < BONUS_THRESH

    path = []
    undones = set(all_tasks.copy())
    for t in range(len(episode)):
        new_task = None
        for task in undones:
            if completes[task][t]:
                new_task = task
                break
        if new_task is not None:
            undones.remove(new_task)
            path.append(new_task)
    return path
