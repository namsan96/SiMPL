from contextlib import contextmanager

from gym.wrappers.monitor import video_recorder


@contextmanager
def record(env, video_path):    
    global t
    recorder = video_recorder.VideoRecorder(env, video_path)
    env_reset = env.reset
    env_step = env.step
    t = 0
    
    def reset():
        global t
        ret = env_reset()
        recorder.capture_frame()
        t += 1
        return ret
    
    def step(action):
        global t
        ret = env_step(action)
        frame_drop = env.metadata.get('video.frame_drop')
        if frame_drop is None or t % frame_drop == 0:
            recorder.capture_frame()
        t += 1
        return ret
    
    env.reset = reset
    env.step = step
    
    yield 
    
    recorder.close()
    env.reset = env_reset
    env.step = env_step
