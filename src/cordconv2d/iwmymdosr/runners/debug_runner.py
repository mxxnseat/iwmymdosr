from collections import deque
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch as t

from cordconv2d.dataset.datasets.objects import Meta
from cordconv2d.dataset.datasets.preprocess import process_frame, sample
from cordconv2d.dataset.datasets.shard_loader import ShardLoader
from cordconv2d.iwmymdosr.utils import denormalize_output
from cordconv2d.model.evaluate import evaluate
from cordconv2d.model.model import Iwmymdosr

class DebugRunner:
    def __init__(self, model: Iwmymdosr, meta: Meta, device, shard_loader: ShardLoader, debug_resolution: tuple[int, int], debug_dir: PathLike[str] | str):
        self.debug_resolution = debug_resolution
        self.debug_dir = Path(debug_dir)
        
        self.shard_loader = shard_loader
        self.device = device
        self.meta = meta
        self.model = model
        self.queue = deque(maxlen=self.meta.frames_n)

        self.batch = self.load_shard()
        self.virtual_cursor = (60, 60)

        self.cursor_history: list[tuple[int, int]] = [self.virtual_cursor]

    def step(self):
        shard = next(self.batch)
        frames = shard["frames"][20] # torch.Size([3, 72, 128])
        last_frame = frames[-1]
        for frame_t in frames:
            frame = np.asarray(frame_t)
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            grayscaled_frame, _ = process_frame(bgr, self.meta.resolution)
            current_mouse_position = self.get_mouse_position()
            self.queue.append([t.from_numpy(grayscaled_frame), t.tensor([current_mouse_position[0], current_mouse_position[1]])])
            if len(self.queue) == self.meta.frames_n:
                self.run()
                break
        self.output(last_frame)

    def run(self):
        frames = t.stack([item[0] for item in self.queue])
        cursors = t.stack([item[1] for item in self.queue])
        inp = sample(frames, cursors, 4.0, self.meta.resolution)
        inp = inp.to(self.device)
        dx_dy, _ = evaluate(self.model, inp.unsqueeze(0))
        dx_dy = dx_dy.squeeze(0).tolist()
        d = denormalize_output((dx_dy[0], dx_dy[1]), (self.debug_resolution[0], self.debug_resolution[1]), self.meta)
        self.move_cursor(d)

    def output(self, circle):
        image = cv2.cvtColor(np.asarray(circle), cv2.COLOR_GRAY2BGR)
        s = 1
        for cx, cy in self.cursor_history:
            cv2.rectangle(image, (int(cx-s), int(cy-s)), (int(cx+s), int(cy+s)), (0, 0, 255), -1)

        cv2.imwrite(self.debug_dir / "test.jpg", image)

    def load_shard(self):
        return self.shard_loader.load()

    def move_cursor(self, d):
        dx, dy = d
        vc = (self.virtual_cursor[0] + dx, self.virtual_cursor[1] + dy)

        self.virtual_cursor = vc
        self.cursor_history.append(self.virtual_cursor)

    def get_mouse_position(self):
        return self.virtual_cursor