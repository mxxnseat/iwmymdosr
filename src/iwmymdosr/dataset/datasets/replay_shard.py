from os import PathLike
from pathlib import Path
import random

import torch as t

from iwmymdosr.dataset.datasets.preprocess import sample
from iwmymdosr.dataset.datasets.utils import read_dataset_meta

class ReplayShards(t.utils.data.IterableDataset):
    def __init__(self, shard_dir: PathLike[str] | str, limit = None, shuffle: bool = True,
                 sigma: float = 4.0, seed: int = 0):
        self.dir = Path(shard_dir)
        self.meta = read_dataset_meta(self.dir)
        self.files = sorted(self.meta.counts)
        self.shuffle = shuffle
        self.sigma = sigma
        self.seed = seed
        self.epoch = 0
        self.limit = limit

        if not self.limit is None:
            self.files = self.files[:self.limit]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return sum(self.meta.counts.values())

    def get_heatmap(self, points, H, W, dt, sigma_t=1.5, sigma=4.0):
        # points: (B, N, 2) as (y, x);  dt: (B, N)
        B, N = points.shape[:2]
        dev = points.device

        ys = t.arange(H, dtype=t.float32, device=dev).view(1, 1, H, 1)
        xs = t.arange(W, dtype=t.float32, device=dev).view(1, 1, 1, W)
        y0 = points[..., 1].float().view(B, N, 1, 1)
        x0 = points[..., 0].float().view(B, N, 1, 1)

        blob = t.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma ** 2))   # (B, N, H, W)

        dt = dt.float()
        amp = t.exp(-(dt ** 2) / (2 * sigma_t ** 2))      # (B, N)
        blob = blob * amp.view(B, N, 1, 1)

        return blob.max(dim=1).values                                          # (B, H, W)

    def __iter__(self):
        info = t.utils.data.get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)

        files = list(self.files)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(files)
        files = files[wid::nworkers]

        for f in files:
            shard = t.load(self.dir / f, mmap=True)
            for row in range(len(shard["target"])):
                p = shard["target"][row]
                if p.size(0) != 0:
                    heatmap = self.get_heatmap(
                        p[:, :2].unsqueeze(0),
                        self.meta.resolution[1],
                        self.meta.resolution[0],
                        p[..., 2].unsqueeze(0),
                        sigma_t=5.0
                    )[0]
                else:
                    heatmap = t.zeros((self.meta.resolution[1], self.meta.resolution[0]), dtype=t.float32)

                inp = sample(shard["frames"][row])
                yield inp, heatmap