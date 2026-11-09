from cordconv2d.dataset.datasets.objects import Meta
from cordconv2d.dataset.osu.utils import derive_transform, osu_to_px, px_to_osu

def denormalize_output(dx_dy: tuple[int, int], current_resolution: tuple[int, int], meta: Meta):
    dx, dy = dx_dy
    return dx * current_resolution[0], dy * current_resolution[1]
