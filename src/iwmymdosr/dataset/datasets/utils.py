import json
from os import PathLike
from pathlib import Path
from dataclasses import asdict

import numpy as np
from iwmymdosr.dataset.datasets.objects import Meta

META_FILENAME = "index.json"

def write_dataset_meta(out_dir: PathLike[str] | str, meta: dict):
    out_dir = Path(out_dir)

    (out_dir / META_FILENAME).write_text(json.dumps(asdict(Meta(**meta))))

def read_dataset_meta(dir: PathLike[str] | str):
    dir = Path(dir)
    meta = json.loads((dir / META_FILENAME).read_text())

    return Meta(frames_n=meta["frames_n"],
                resolution=meta["resolution"],
                counts=meta["counts"]
        )