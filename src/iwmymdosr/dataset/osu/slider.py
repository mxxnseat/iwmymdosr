import math

from iwmymdosr.dataset.osu.objects import ISlider


def lerp(a, b, t): return (a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)

def bezier_point(pts, t):
    pts = list(pts)
    while len(pts) > 1:
        pts = [lerp(pts[i], pts[i+1], t) for i in range(len(pts)-1)]
    return pts[0]

def sample_bezier(pts, n=64):
    # split at duplicate points (red anchors)
    segs, cur = [], [pts[0]]
    for p in pts[1:]:
        cur.append(p)
        if p == cur[-2]:
            segs.append(cur[:-1]); cur = [p]
    segs.append(cur)
    out = []
    for s in segs:
        if len(s) < 2: continue
        out += [bezier_point(s, i/n) for i in range(n+1)]
    return out

def sample_perfect(pts, length, n=200):
    A, B, C = pts
    d = 2*(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    if abs(d) < 1e-6:
        return None  # collinear -> fall back
    a2, b2, c2 = sum(v*v for v in A), sum(v*v for v in B), sum(v*v for v in C)
    cx = (a2*(B[1]-C[1]) + b2*(C[1]-A[1]) + c2*(A[1]-B[1])) / d
    cy = (a2*(C[0]-B[0]) + b2*(A[0]-C[0]) + c2*(B[0]-A[0])) / d
    r = math.hypot(A[0]-cx, A[1]-cy)
    cross = (B[0]-A[0])*(C[1]-B[1]) - (B[1]-A[1])*(C[0]-B[0])
    direction = 1 if cross > 0 else -1
    a0 = math.atan2(A[1]-cy, A[0]-cx)
    total = direction * length / r          # use pixelLength, not point C
    return [(cx + r*math.cos(a0 + total*i/n), cy + r*math.sin(a0 + total*i/n))
            for i in range(n+1)]

def build_path(curve_type, pts, length):
    if curve_type == 'P' and len(pts) == 3:
        path = sample_perfect(pts, length)
        if path: return path
    if curve_type == 'L':
        path = []
        for i in range(len(pts)-1):
            path += [lerp(pts[i], pts[i+1], k/32) for k in range(33)]
    else:
        path = sample_bezier(pts)
    return path

class Slider(ISlider):
    def __init__(self, start_time, curve_type, pts, slides, length,
                 slider_multiplier, beat_length, sv_multiplier=1.0):
        super().__init__()
        self.start, self.slides, self.length = start_time, slides, length
        self.slide_dur = length / (100*slider_multiplier*sv_multiplier) * beat_length
        raw = build_path(curve_type, pts, length)

        self.end_time = start_time + self.slide_dur * slides
        self.head = pts[0]
        self.offset = (0.0, 0.0)      # stack offset, set after stacking

        
        # cumulative arc length
        self.path, self.cum = [raw[0]], [0.0]
        for p in raw[1:]:
            d = math.dist(p, self.path[-1])
            if d == 0: continue
            self.path.append(p); self.cum.append(self.cum[-1] + d)
        # extend if path is shorter than pixelLength (tangent extension)
        if self.cum[-1] < length and len(self.path) > 1:
            a, b = self.path[-2], self.path[-1]
            dx, dy = (b[0]-a[0]), (b[1]-a[1]); n = math.hypot(dx, dy)
            extra = length - self.cum[-1]
            self.path.append((b[0]+dx/n*extra, b[1]+dy/n*extra))
            self.cum.append(length)

    def pos_at_distance(self, d):
        d = max(0, min(d, self.cum[-1], self.length))
        lo, hi = 0, len(self.cum)-1
        while lo < hi - 1:
            mid = (lo+hi)//2
            if self.cum[mid] <= d: lo = mid
            else: hi = mid
        seg = self.cum[hi] - self.cum[lo]
        t = 0 if seg == 0 else (d - self.cum[lo]) / seg
        return lerp(self.path[lo], self.path[hi], t)

    def set_stack_offset(self, head_x, head_y):
        """Pass the head position after stacking."""
        self.offset = (head_x - self.head[0], head_y - self.head[1])

    def ball_pos(self, t):
        p = (t - self.start) / self.slide_dur
        p = max(0, min(p, self.slides))
        idx = min(int(p), self.slides - 1)
        frac = p - idx
        if idx % 2 == 1: frac = 1 - frac
        x, y = self.pos_at_distance(frac * self.length)
        return x + self.offset[0], y + self.offset[1]