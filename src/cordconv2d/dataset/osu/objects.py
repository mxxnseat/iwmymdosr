from dataclasses import dataclass

from numpy import ndarray

@dataclass
class HitObject:
    x: int
    y: int
    radius: float
    time: int
    type: int
    is_circle: bool
    is_slider: bool
    is_spinner: bool

@dataclass
class MapDifficulty:
    cs: float
    ar: float
    leniency: float
    od: float

@dataclass
class MapGeneral:
    audio_filename: str

@dataclass
class Transform:
    s: float
    ox: float
    oy: float

@dataclass
class Replay:
    replay_data: ndarray
    mods: int