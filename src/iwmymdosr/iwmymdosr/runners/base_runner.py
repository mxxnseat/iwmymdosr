from abc import abstractmethod
from collections import deque

import cv2
import numpy as np
from numpy.typing import NDArray
import torch as t
import time

from iwmymdosr.dataset.datasets.objects import Meta
from iwmymdosr.dataset.datasets.preprocess import process_frame, sample
from iwmymdosr.dataset.datasets.shard_loader import ShardLoader
from iwmymdosr.iwmymdosr.utils import denormalize_output
from iwmymdosr.model.evaluate import evaluate
from iwmymdosr.model.model import Iwmymdosr

Frame = NDArray[np.uint8]

class BaseRunner:
    def __init__(self, model: Iwmymdosr, meta: Meta, device):
        self.device = device
        self.meta = meta
        self.model = model
        self.queue = deque(maxlen=self.meta.frames_number)

    def step(self):
        frame = self.get_frame()
        grayscaled_frame, _ = process_frame(frame, self.meta.resolution)
        inp = sample(t.from_numpy(grayscaled_frame))
        self.queue.append(inp)

        if len(self.queue) < self.meta.frames_number:
            return None
        
        stack = t.from_numpy(np.stack(self.queue)).to(self.device)
        heatmap = evaluate(self.model, stack.unsqueeze(0))
        x, y, has_target = denormalize_output(heatmap, self.get_current_resolution(), self.meta, target_thr=0.2)
        if not has_target:
            return heatmap
        self.move_cursor((x,y))
        return heatmap

    def run(self):
        while True:
            self.step()

    @abstractmethod
    def get_current_resolution(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_frame(self)-> Frame | None:  ...

    @abstractmethod
    def move_cursor(self, d): ...

    @abstractmethod
    def get_mouse_position(self) -> tuple[int, int]: ...
