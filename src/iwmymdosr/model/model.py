import torch as t

from iwmymdosr.model.modules.cord_conv_2d import CoordConv2d
from iwmymdosr.model.modules.spatial_softmax import SpatialSoftmax

def block(in_feat, out_f, kernel, stride=1, padding=0, bias=True):
    return t.nn.Sequential(
        t.nn.Conv2d(in_feat, out_f, kernel, stride=stride, padding=padding, bias=bias),
        t.nn.BatchNorm2d(out_f),
        t.nn.ReLU(),
    )

class Iwmymdosr(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = t.nn.Sequential(
            CoordConv2d(1, 32, 3, padding=1, bias=False), t.nn.BatchNorm2d(32), t.nn.ReLU(),
            block(32, 32, kernel=3, padding=1, bias=False),
            block(32, 64, kernel=3, stride=2, padding=1, bias=False),
            block(64, 64, kernel=3, padding=1, bias=False),
            block(64, 64, kernel=3, stride=2, padding=1, bias=False),
            block(64, 64, kernel=3, padding=1, bias=False),
        )
        self.head = t.nn.Sequential(
            SpatialSoftmax(18, 32),
            t.nn.Flatten(),
            t.nn.Linear(128, 256), t.nn.ReLU(),
            t.nn.Dropout(p=0.3),
            t.nn.Linear(256, 2),
        )

    def forward(self, inp):
        conv_out = self.conv(inp)
        out = self.head(conv_out)

        return out
