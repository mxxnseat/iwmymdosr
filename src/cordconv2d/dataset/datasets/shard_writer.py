from pathlib import Path

import numpy as np
import torch as t

from cordconv2d.dataset.datasets.utils import write_dataset_meta
from cordconv2d.utils.main import RunningStd

class ShardWriter:
    """Buffers samples and flushes a shard file every `samples_per_shard`."""

    def __init__(self, out_dir: Path, samples_per_shard: int, meta: dict):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.meta = meta
        self.counts: dict[str, int] = {}
        self.shard_id = 0
        self._reset()

        self.running_std = RunningStd(n_features=2)

    def _reset(self):
        self.frames, self.targets = [], []

    def add(self, frames: np.ndarray, target: np.ndarray):
        self.frames.append(frames)
        self.targets.append(target)
        if len(self.frames) >= self.samples_per_shard:
            self.flush()

    def flush(self):
        if not self.frames:
            return
        name = f"shard_{self.shard_id:05d}.pt"
        target_stack = np.stack(self.targets)
        t.save(
            {
                "frames": t.from_numpy(np.stack(self.frames)),
                "target": t.from_numpy(target_stack),
            },
            self.out_dir / name,
        )
        self.counts[name] = len(self.frames)
        self.shard_id += 1
        self._reset()

    def close(self):
        self.flush()
        write_dataset_meta(self.out_dir, {
            **self.meta,
            "counts": self.counts
        })
