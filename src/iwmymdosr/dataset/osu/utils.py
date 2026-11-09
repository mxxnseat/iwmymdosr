from iwmymdosr.dataset.osu.objects import Transform

def derive_transform(W, H):
    s = H / 480
    return Transform(s, (W - 512 * s) / 2, (H - 384 * s) / 2 + 8 * s)

def osu_to_px(x, y, T: Transform):
    return T.s * x + T.ox, T.s * y + T.oy

def px_to_osu(px, py, T: Transform):                  # T = Transform(s, ox, oy) from calibrate()
    return (px - T.ox) / T.s, (py - T.oy) / T.s