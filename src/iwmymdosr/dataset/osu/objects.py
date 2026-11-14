from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from numpy import ndarray

class ISlider:
    end_time: int

    @abstractmethod
    def ball_pos(self, t) -> tuple[int,int]:...

    @abstractmethod
    def set_stack_offset(self, head_x, head_y): ...

@dataclass
class HitObject:
    x: int
    y: int
    radius: float
    time: int
    type: int
    slider: ISlider | None
    is_circle: bool
    is_slider: bool
    is_spinner: bool

@dataclass
class MapDifficulty:
    cs: float
    ar: float
    leniency: float
    od: float
    slider_multiplier: float

@dataclass
class SliderProperties:
    curve_type: Literal["B", "L", "C", "P"]
    curve_points: list[tuple[int, int]]
    slides: int
    length: float

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