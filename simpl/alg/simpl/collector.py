from simpl.collector import GPUWorker


class LowFixedGPUWorker(GPUWorker):
    def __call__(self, work_queue, received_queue, result_queue):
        self.collector.low_actor.to(self.gpu)
        return super().__call__(work_queue, received_queue, result_queue)
