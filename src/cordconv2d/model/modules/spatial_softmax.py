import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, temperature=1.0, learnable_temp=True):
        super().__init__()
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", xs.reshape(1, 1, -1))
        self.register_buffer("pos_y", ys.reshape(1, 1, -1))
        t = torch.tensor(float(temperature))
        self.temperature = nn.Parameter(t) if learnable_temp else t

    def forward(self, x):
        B, C, H, W = x.shape
        attn = F.softmax(x.reshape(B, C, H * W) / self.temperature, dim=-1)
        ex = (attn * self.pos_x).sum(-1)        # type: ignore
        ey = (attn * self.pos_y).sum(-1)        # type: ignore
        return torch.stack([ex, ey], dim=-1)