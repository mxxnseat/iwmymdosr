from dataclasses import dataclass

@dataclass
class Meta:
    frames_n: int
    resolution: tuple[int, int]
    counts: dict[str, int]