import torch as t

def block(in_feat, out_f, kernel=3, stride=1, padding=1, bias=True):
    return t.nn.Sequential(
        t.nn.Conv2d(in_feat, out_f, kernel, stride=stride, padding=padding, bias=bias),
        t.nn.BatchNorm2d(out_f, bias=not bias),
        t.nn.ReLU(),
    )

class Iwmymdosr(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = t.nn.Sequential(block(3, 32, bias=False), block(32, 32, bias=False))
        self.enc2 = t.nn.Sequential(block(32, 64, stride=2, bias=False), block(64, 64, bias=False))
        self.enc3 = t.nn.Sequential(block(64, 128, stride=2, bias=False), block(128, 128, bias=False))
        self.bott = t.nn.Sequential(block(128, 128, stride=2, bias=False), block(128, 128, bias=False))

        self.up = t.nn.Upsample(scale_factor=2)

        self.dec3 = t.nn.Sequential(block(128 + 128, 128, bias=False), block(128, 64, bias=False))
        self.dec2 = t.nn.Sequential(block(64 + 64, 64, bias=False), block(64, 32, bias=False))
        self.dec1 = t.nn.Sequential(block(32 + 32, 32, bias=False), block(32, 32, bias=False))

        self.head = t.nn.Conv2d(32, 1, 1)

    def forward(self, inp):
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bott(e3)
        d3 = self.dec3(t.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(t.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(t.cat([self.up(d2), e1], dim=1))

        return self.head(d1).squeeze(1)
