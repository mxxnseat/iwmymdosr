import time
from functools import wraps
import numpy as np
import torch as t

def log(where, what):
    print(f"{where.__name__}: {what}")

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.4f} s")
    return wrapper

class RunningStd:
    def __init__(self, n_features=None):
        self.n = 0
        self.running_mean = 0.0 if n_features is None else np.zeros(n_features)
        self.M2 = 0.0 if n_features is None else np.zeros(n_features)

    def update(self, batch, axis=0):
        batch = np.asarray(batch, dtype=np.float64)
        n_b = batch.shape[axis]
        mean_b = batch.mean(axis=axis)
        M2_b = ((batch - mean_b) ** 2).sum(axis=axis)

        delta = mean_b - self.running_mean
        n_total = self.n + n_b

        self.running_mean = self.running_mean + delta * n_b / n_total
        self.M2 = self.M2 + M2_b + delta**2 * self.n * n_b / n_total
        self.n = n_total

    def var(self, ddof=0):
        return self.M2 / (self.n - ddof)

    def mean(self):
        return self.running_mean
    
    def std(self, ddof=0):
        return np.sqrt(self.var(ddof))