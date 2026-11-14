import cv2

from iwmymdosr.dataset.datasets.objects import Meta
from iwmymdosr.iwmymdosr.runners.base_runner import BaseRunner
from iwmymdosr.model.model import Iwmymdosr
import torch as t
import matplotlib.pyplot as plt

class DebugRunner(BaseRunner):
    def __init__(self, model: Iwmymdosr, meta: Meta, device, data: t.utils.data.DataLoader):
        super().__init__(model, meta, device)
        self.loader = data
        self.data = iter(data)
        self.current_batch = None
        self.frame_idx = 0

    def step(self):
        heatmap = super().step()
        if heatmap is None:
            return
        plt.subplot(1, 4, 1)
        plt.imshow(heatmap[0].cpu().numpy(), vmin=0.0, vmax=1.0) # type: ignore
        plt.title("result")

        for ind, frame in enumerate(self.queue, 2):
            plt.subplot(1, 4, ind)
            plt.imshow(frame, cmap="gray") # type: ignore
            plt.title(f"frame #{ind-1}")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1.0)
        plt.close()

    def get_current_resolution(self) -> tuple[int, int]:
        return (self.meta.resolution[0], self.meta.resolution[1])

    def get_frame(self):
        if self.current_batch is None or self.frame_idx >= self.current_batch.shape[0]:
            try:
                batch_x = next(self.data)[0]
            except StopIteration:
                self.data = iter(self.loader)
                batch_x = next(self.data)[0]

            self.current_batch = batch_x.reshape(-1, *batch_x.shape[-2:])
            self.frame_idx = 0

        frame = self.current_batch[self.frame_idx]
        self.frame_idx += 1
        bgr = cv2.cvtColor(frame.cpu().numpy(), cv2.COLOR_GRAY2BGR) * 255
        return bgr
    
    def move_cursor(self, d):
        return

    def get_mouse_position(self):
        return (0,0)