from contextlib import contextmanager

from d4rl.pointmaze import MazeEnv
import gym
import mujoco_py
import numpy as np


from .maze_layout import rand_layout


init_loc_noise = 0.1
complete_threshold = 1.0


class MazeTask:
    def __init__(self, init_loc, goal_loc):
        self.init_loc = np.array(init_loc, dtype=np.float32)
        self.goal_loc = np.array(goal_loc, dtype=np.float32)

    def __repr__(self):
        return f'MTMazeTask(start:{self.init_loc}+-{init_loc_noise}, end: {self.goal_loc})'


class MazeEnv(MazeEnv):
    reward_types = ['sparse', 'dense']
    
    render_width = 100
    render_height = 100
    render_device = -1

    def __init__(self, size, seed, reward_type, done_on_completed):
        if reward_type not in self.reward_types:
            raise f'reward_type should be one of {reward_types}, but {reward_type} is given'
        
        self.maze_size = size
        self.maze_spec = rand_layout(size=size, seed=seed)
        
        # for initialization
        self.task = MazeTask([0, 0], [0, 0])
        self.done_on_completed = False
        
        super().__init__(self.maze_spec, reward_type, reset_target=False)
        
        self.task = None
        self.done_on_completed = done_on_completed
        
        gym.utils.EzPickle.__init__(self, size, seed, reward_type, done_on_completed)
        
    @contextmanager
    def set_task(self, task):
        if type(task) != MazeTask:
            raise TypeError(f'task should be MazeTask but {type(task)} is given')

        prev_task = self.task
        self.task = task
        self.set_target(task.goal_loc)
        yield
        self.task = prev_task
        
    def set_render_options(self, width, height, device, fps=30, frame_drop=1):
        self.render_width = width
        self.render_height = height
        self.render_device = device
        self.metadata['video.frames_per_second'] = fps
        self.metadata['video.frame_drop'] = frame_drop

    def reset_model(self):
        if self.task is None:
            raise RuntimeError('task is not set')
        init_loc = self.task.init_loc
        qpos = init_loc + self.np_random.uniform(low=-init_loc_noise, high=init_loc_noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        if self.task is None:
            raise RuntimeError('task is not set')
        action = np.clip(action, -1.0, 1.0)
        self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        
        ob = self._get_obs()
        goal_dist = np.linalg.norm(ob[0:2] - self._target)
        completed = (goal_dist <= complete_threshold)
        done = self.done_on_completed and completed
        
        if self.reward_type == 'sparse':
            reward = float(completed)
        elif self.reward_type == 'dense':
            reward = np.exp(-goal_dist)
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)

        return ob, reward, done, {}

    def render(self, mode='rgb_array'):
        return super().render(mode, self.render_width, self.render_height)
        
    def _get_viewer(self, mode):
        if self._viewers.get(mode) is None and mode in ['rgb_array', 'depth_array']:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=self.render_device)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return super()._get_viewer(mode)
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] += 0.5
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0
            
    
class AgentCentricMazeEnv(MazeEnv):
    n_frame = 2
    agent_centric_res = 32
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.img_vec_q = deque(maxlen=self.n_frame)
        self.obs_layout = {
            'state': slice(0, self.observation_space.shape[0]),
            'image': slice(self.observation_space.shape[0], None),
        }
    
    @contextmanager
    def agent_centric_render(self):
        prev_type = self.viewer.cam.type
        prev_distance = self.viewer.cam.distance
        
        self.viewer.cam.type = mujoco_py.generated.const.CAMERA_TRACKING
        self.viewer.cam.distance = 5.0
        
        yield
        
        self.viewer.cam.type = prev_type
        self.viewer.cam.distance = prev_distance
        
    def _get_obs(self):
        state = super()._get_obs()
        
        with self.agent_centric_render():
            img = self.sim.render(self.render_width, self.render_height, device_id=self.render_device)
        
        img_vec = (img.transpose(2, 0, 1) / 255 * 2 - 1).flatten()
        while len(self.img_vec_q) < self.n_frame-1:
            self.img_vec_q.append(img_vec)
        self.img_vec_q.append(img_vec)
        return np.concatenate([state] + list(self.img_vec_q))
