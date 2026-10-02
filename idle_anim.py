#!/usr/bin/env python3
"""
Fruity Limiter (FL Studio) -- idle / "disabled in mixer" visual animation.

Usage:
    python3 idle_anim.py out.png [seconds_t]      # single frame at time t
    python3 idle_anim.py out.gif --gif [secs]     # animated GIF (needs Pillow)
"""
import math
import sys

import numpy as np

# ---------------------------------------------------------------- constants
W_DEFAULT, H_DEFAULT = 597, 160      # Visual control size from the form resource
SLIDER_DEFAULT = 50                  # SpeedSlider default (range 0..100)

PURPLE = (0x81, 0x3C, 0x5D)          
GREEN = (0x25, 0x7F, 0x52)           
WHITE = (255, 255, 255)
BG_TOP = (0x31, 0x2F, 0x26)          
BG_BOT = (0x44, 0x41, 0x36)          


def speed(slider=SLIDER_DEFAULT):
    """Phase units per millisecond."""
    return (slider * 0.1 + 1.0) * 0.07


# ---------------------------------------------------------------- generator
def make_columns(phase, width):
 
    n = width + 1
    i = np.arange(n, dtype=np.float64)

    # slow "breathing" envelope, period 2*pi/0.00106723586 ~ 5887 phase units
    env = ((math.sin(phase * 0.0010672358591248667) + 1.0) * 0.6 + 0.8) / (width * 2) * 2.0
    f = ((width + 21) - i) * env                      # fVar7, float in the original
    f = f.astype(np.float32).astype(np.float64)
    d = phase + i

    out = np.empty((n, 6), dtype=np.float64)
    out[:, 0] = (np.sin(d * 0.01 + math.pi / 2) + 1.1) * f                          # purple area
    out[:, 1] = (np.sin(d * (1 / 70) + math.pi / 3) + 1.2) * f * 1.5                # green area
    out[:, 2] = (np.sin(d * (1 / 140) + 2 * math.pi / 3) + 1.0) * f * f * 1.1       # white line
    out[:, 3] = out[:, 2] - 0.24
    out[:, 4] = (np.sin(d * (1 / 140) + 2.362099739541198) + 1.0) * f * f * 1.9     # optional 3rd band
    out[:, 5] = out[:, 2] - 0.41
    return out.astype(np.float32)


def vol_to_y(v, h):
    return h - (np.log(2.0 * v + 1.0) / math.log(3.0)) * h * 0.5 - 0.5


# ---------------------------------------------------------------- rasteriser
def _lut(gamma):
    return np.rint(256.0 * (np.arange(257) / 256.0) ** gamma).astype(np.int32)


LUT0, LUT1 = _lut(1.0), _lut(5.0 / 6.0)


def _round(x):          # Delphi Round(): banker's rounding
    return int(np.rint(x))


def _area_column(img, x, y, yprev, color, h):
    lo, hi = (y, yprev) if y <= yprev else (yprev, y)
    r = max(_round(lo - 0.5), 0)          # original clamps to the clip rect *before* seeding the ramp
    bottom = h - 1
    if r > bottom:
        return
    step = min(1.0 / ((hi - lo) + 1.0), 1.0) * 256.0
    t = ((r - lo) + 1.0) * step
    if t < 0.0:
        t += step
        r += 1
    for row in range(r, bottom + 1):
        t = min(t, 256.0)
        a = LUT0[_round(t)]
        for c in range(3):
            img[row, x, c] = min(255, img[row, x, c] + ((color[c] * a) >> 8))
        t += step


def _band_column(img, x, y2, y2p, y3, y3p, color, alpha, h):
    tlo, thi = (y2, y2p) if y2 <= y2p else (y2p, y2)
    blo, bhi = (y3, y3p) if y3 <= y3p else (y3p, y3)
    r1 = max(_round(tlo - 0.5), 0)
    r2 = min(_round(bhi + 0.5), h - 1)
    up = 256.0 / ((thi - tlo) + 1.0)
    down = 1.0 / ((blo - bhi) - 1.0)
    t_up = ((r1 - tlo) + 1.0) * up
    t_dn = ((r1 - blo) * down + 1.0)
    for row in range(r1, r2 + 1):
        cov = min(t_up, 256.0, t_dn * 256.0)
        cov = max(cov, 0.0)
        a = (LUT1[_round(cov)] * alpha) >> 8
        for c in range(3):
            d = img[row, x, c]
            img[row, x, c] = d + (((color[c] - int(d)) * a) >> 8)
        t_up += up
        t_dn += down


def render(cols, width=W_DEFAULT, height=H_DEFAULT):
    img = np.zeros((height, width, 3), dtype=np.int32)
    g = np.linspace(0, 1, height)[:, None]
    img[:] = (np.array(BG_TOP) * (1 - g) + np.array(BG_BOT) * g).astype(np.int32)[:, None, :]

    # ring-buffer read order: column x shows data index x-1, x=0 shows an empty slot
    def val(k, x):
        return 0.0 if x - 1 < 0 else float(cols[x - 1, k])

    ys = {k: np.array([vol_to_y(val(k, x), height) for x in range(width + 1)]) for k in range(6)}

    # additive pass: channel 0 (purple) and channel 1 (green) filled areas.
    # Original iterates x = W..0 and uses the right neighbour as "previous"; x == W is clipped.
    for k, color in ((0, PURPLE), (1, GREEN)):
        yy = ys[k] + 0.5
        for x in range(width - 1, -1, -1):
            _area_column(img, x, yy[x], yy[x + 1], color, height)

    # white band between curve 2 and curve 3, alpha 200
    for x in range(width - 1, -1, -1):
        _band_column(img, x, ys[2][x], ys[2][x + 1], ys[3][x], ys[3][x + 1], WHITE, 200, height)

    return img.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------- driver
def frame_at(t_seconds, slider=SLIDER_DEFAULT, width=W_DEFAULT, height=H_DEFAULT):
    phase = t_seconds * 1000.0 * speed(slider)
    return render(make_columns(phase, width), width, height)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "idle.png"
    from PIL import Image
    if "--gif" in sys.argv:
        secs = float(sys.argv[sys.argv.index("--gif") + 1]) if len(sys.argv) > sys.argv.index("--gif") + 1 else 4
        frames = [Image.fromarray(frame_at(k / 20.0)) for k in range(int(secs * 20))]
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
    else:
        t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
        Image.fromarray(frame_at(t)).save(out)
