from os import PathLike
from pathlib import Path
import torch as t

class ShardLoader:
    def __init__(self, shard_dir: PathLike[str] | str):
        self.shard_dir = Path(shard_dir)

    def load(self):
        for shard in self.shard_dir.glob("*.pt"):
            yield t.load(shard, mmap=True)
