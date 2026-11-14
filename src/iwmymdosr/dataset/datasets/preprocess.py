import torch as t
import cv2

def process_frame(frame, resolution: tuple[int, int]):
    W, H = resolution
    resized_frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    grayscaled_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return grayscaled_frame, resized_frame

def sample(frames):
    frames = frames.float().div_(255.0)
    return frames