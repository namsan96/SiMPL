import torch.multiprocessing as mp


class ConcurrentCollector:
    def __init__(self, workers):
        self.work_queue = mp.Queue(1) 
        self.received_queue = mp.Queue(1)
        self.result_queue = mp.Queue()
        
        self.processes = []
        for worker in workers:
            queues = (self.work_queue, self.received_queue, self.result_queue)
            p = mp.Process(target=worker, args=queues)
            p.start()
            self.processes.append(p)
        self.work_i = 0
    
    def submit(self, *args, **kwargs):
        msg = (self.work_i, args, kwargs)
        self.work_queue.put(msg)
        self.received_queue.get()
        
        self.work_i += 1
    
    def wait(self):
        episodes = [None]*self.work_i
        while self.work_i > 0:
            msg = self.result_queue.get()
            work_i, episode = msg
            episodes[work_i] = episode
            self.work_i -= 1
        return episodes
    
    def close(self):
        for _ in range(len(self.processes)):
            self.work_queue.put(False)
        for process in self.processes:
            process.join()

            
class BaseWorker:
    def collect_episode(self):
        raise NotImplementedError
    
    def __call__(self, work_queue, received_queue, result_queue):
        while True:
            msg = work_queue.get()
            
            if msg is False:
                break

            received_queue.put(True)            
            work_i, args, kwargs = msg
            episode = self.collect_episode(*args, **kwargs)
            
            msg = (work_i, episode)
            result_queue.put(msg)

            
class GPUWorker(BaseWorker):
    def __init__(self, collector, gpu):
        self.collector = collector
        self.gpu = gpu

    def collect_episode(self, task, policy):
        policy.to(self.gpu)
        with self.collector.env.set_task(task):
            episode = self.collector.collect_episode(policy)
        return episode