import torch as t
import cv2

def get_heatmaps(cursors: t.Tensor, sigma: float, resolution: tuple[int, int]) -> t.Tensor:
    W, H = resolution
    xs = t.arange(W, dtype=t.float32)
    ys = t.arange(H, dtype=t.float32)
    two_s2 = 2 * sigma ** 2
    gx = t.exp(-((xs[None, :] - cursors[:, 0:1]) ** 2) / two_s2)
    gy = t.exp(-((ys[None, :] - cursors[:, 1:2]) ** 2) / two_s2)
    return gy[:, :, None] * gx[:, None, :]

def process_frame(frame, resolution: tuple[int, int]):
    W, H = resolution
    resized_frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    grayscaled_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return grayscaled_frame, resized_frame

def sample(frames):
    frames = frames.float().div_(255.0)
    return frames