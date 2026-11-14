from dataclasses import dataclass

@dataclass
class Meta:
    frames_number: int
    resolution: tuple[int, int]
    counts: dict[str, int]