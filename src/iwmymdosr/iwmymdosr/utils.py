import torch as t

from iwmymdosr.dataset.datasets.objects import Meta
from iwmymdosr.dataset.osu.utils import derive_transform, osu_to_px, px_to_osu
import matplotlib.pyplot as plt

def denormalize_output(heatmap, current_resolution: tuple[int, int], meta: Meta, target_thr = 0.5):
    r = current_resolution[0]/meta.resolution[0]
    B, H, W = heatmap.shape
    peak, idx = heatmap.view(B, -1).max(dim=1)
    y = idx // W
    x = idx % W
    has_target = peak > target_thr
    return x * r, y * r, has_target
