from os import PathLike
from pathlib import Path
import random

import torch as t

from cordconv2d.dataset.datasets.preprocess import sample
from cordconv2d.dataset.datasets.utils import read_dataset_meta

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

    def _sample(self, shard, row):
        inp = sample(shard["frames"][row])
        return inp, shard["target"][row]

    def __iter__(self):
        info = t.utils.data.get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)

        files = list(self.files)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(files)
        files = files[wid::nworkers]

        for f in files:
            shard = t.load(self.dir / f, mmap=True)
            shard["target"][:, :2].float().div_(t.tensor([self.meta.resolution[0], self.meta.resolution[1]]))
            for r in range(len(shard["target"])):
                yield self._sample(shard, r)
