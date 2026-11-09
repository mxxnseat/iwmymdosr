from abc import abstractmethod
from collections import deque

import cv2
import numpy as np
from numpy.typing import NDArray
import torch as t
import time

from cordconv2d.dataset.datasets.objects import Meta
from cordconv2d.dataset.datasets.preprocess import process_frame, sample
from cordconv2d.dataset.datasets.shard_loader import ShardLoader
from cordconv2d.iwmymdosr.utils import denormalize_output
from cordconv2d.model.evaluate import evaluate
from cordconv2d.model.model import Iwmymdosr


Frame = NDArray[np.uint8]

class BaseRunner:
    def __init__(self, model: Iwmymdosr, meta: Meta, device):
        self.device = device
        self.meta = meta
        self.model = model
        self.queue = deque(maxlen=self.meta.frames_n)

    def step(self):
        frame = np.asarray(self.get_frame())
        grayscaled_frame, _ = process_frame(frame, self.meta.resolution)
        H, W, _ = frame.shape
        inp = sample(t.from_numpy(grayscaled_frame))
        inp = inp.to(self.device)
        dx_dy = evaluate(self.model, inp.unsqueeze(0).unsqueeze(0))
        dx_dy = dx_dy.squeeze(0).tolist()
        d = denormalize_output((dx_dy[0], dx_dy[1]), (W, H), self.meta)
        self.move_cursor(d)

    def run(self):
        while True:
            self.step()

    @abstractmethod
    def get_frame(self)-> Frame | None:  ...

    @abstractmethod
    def move_cursor(self, d): ...

    @abstractmethod
    def get_mouse_position(self) -> tuple[int, int]: ...
