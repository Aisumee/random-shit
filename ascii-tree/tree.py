#!/usr/bin/env python3
"""A rotating 3D Christmas tree drawn with ASCII letters.

    pip install numpy
    python3 tree.py          # Ctrl+C to quit
"""
import shutil
import string
import sys
import time

import numpy as np

rng = np.random.default_rng(2026)
LETTERS = np.array(list(string.ascii_letters + "#%&@$*+"))

Y0, Y1 = 0.40, 2.70          # foliage from Y0 to Y1
TIERS = 6                    # branch layers
N_LEAF, N_LIGHTS = 9000, 60


def envelope(y):
    f = np.clip((y - Y0) / (Y1 - Y0), 0, 1)
    tier = (f * TIERS) % 1.0
    return (0.04 + 0.92 * (1 - f) ** 1.1) * (0.78 + 0.28 * tier)


class Parts:
    def __init__(s):
        s.P, s.C, s.ch = [], [], []

    def add(s, pts, col, chars=None):
        s.P.append(pts)
        s.C.append(col)
        s.ch.append(chars if chars is not None else rng.choice(LETTERS, len(pts)))

    def done(s):
        s.P, s.C, s.ch = np.vstack(s.P), np.vstack(s.C), np.concatenate(s.ch)
        return s


def build():
    parts = Parts()

    # foliage: points packed toward the outer skin of each tier
    y = rng.uniform(Y0, Y1, N_LEAF)
    a = rng.uniform(0, 2 * np.pi, N_LEAF)
    depth = rng.random(N_LEAF) ** 0.35
    r = envelope(y) * depth
    pts = np.c_[r * np.cos(a), y, r * np.sin(a)]
    f = (y - Y0) / (Y1 - Y0)
    tier = (f * TIERS) % 1.0
    snowy = (depth > 0.8) & (rng.random(N_LEAF) < 0.1 + 0.75 * f ** 1.4 + 0.35 * (tier > 0.75))
    green = np.c_[0.10 + 0.10 * rng.random(N_LEAF), 0.35 + 0.35 * rng.random(N_LEAF), 0.16 + 0.12 * rng.random(N_LEAF)]
    col = np.where(snowy[:, None], [0.92, 0.95, 1.0], green)
    parts.add(pts, col)

    # trunk
    n = 500
    a, y = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, Y0 + 0.1, n)
    parts.add(np.c_[0.11 * np.cos(a), y, 0.11 * np.sin(a)], np.tile([0.45, 0.28, 0.14], (n, 1)))

    # ground: concentric rings of dashes
    rings = []
    for rad in np.linspace(0.3, 2.1, 18):
        m = int(rad * 70)
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        rings.append(np.c_[rad * np.cos(a), np.zeros(m), rad * np.sin(a)])
    g = np.vstack(rings)
    parts.add(g, np.tile([0.32, 0.34, 0.40], (len(g), 1)), np.full(len(g), "-"))
    return parts.done(), np.flatnonzero(depth > 0.9)[:: max(1, int((depth > 0.9).sum() / N_LIGHTS))][:N_LIGHTS]


def project(P, angle, cols, rows):
    c, s = np.cos(angle), np.sin(angle)
    x = P[:, 0] * c - P[:, 2] * s
    z = P[:, 0] * s + P[:, 2] * c
    y = P[:, 1] - 1.2
    tilt = 0.32                                   # camera looks slightly down
    y, z = y * np.cos(tilt) - z * np.sin(tilt), y * np.sin(tilt) + z * np.cos(tilt)
    persp = 4.2 / (4.2 + z)
    scale = min(rows / 3.3, cols / 6.0)
    sx = (cols / 2 + x * persp * scale * 2.0).astype(int)   # x2: chars are tall
    sy = (rows * 0.46 - y * persp * scale).astype(int)
    return sx, sy, z


def render(parts, lights, t, cols, rows, stars):
    angle = t * 0.45
    sx, sy, z = project(parts.P, angle, cols, rows)
    shade = np.clip(1.15 - 0.45 * (z + 1.0), 0.25, 1.2)[:, None]
    col = np.clip(parts.C * shade, 0, 1)
    ch = parts.ch.copy()

    lit = min(len(lights), max(0, int((t - 3.0) * 12)))
    if lit:
        idx = lights[:lit]
        palette = np.array([[1, .25, .3], [1, .85, .25], [.3, .7, 1], [1, .45, .85], [1, 1, 1]])
        blink = (np.sin(t * 5 + idx * 1.7) > -0.3)[:, None]
        col[idx] = np.where(blink, palette[idx % 5], col[idx] * 0.4)
        ch[idx] = "o"

    ok = (sx >= 0) & (sx < cols) & (sy >= 0) & (sy < rows)
    order = np.argsort(-z[ok])                     # far first, near overwrite
    cell = (sy[ok] * cols + sx[ok])[order]
    out_ch = np.full(rows * cols, " ", dtype="<U1")
    out_col = np.zeros((rows * cols, 3))
    out_ch[stars] = "."
    out_col[stars] = 0.55 + 0.45 * np.sin(t * 2 + stars)[:, None]
    out_ch[cell] = ch[ok][order]
    out_col[cell] = col[ok][order]

    rgb = (out_col * 255).astype(int)
    lines = []
    for r in range(rows):
        row = []
        for i in range(r * cols, (r + 1) * cols):
            if out_ch[i] == " ":
                row.append(" ")
            else:
                row.append("\033[38;2;%d;%d;%dm%s" % (*rgb[i], out_ch[i]))
        lines.append("".join(row) + "\033[0m")
    hud = "rotation %06.2f°   lights %3d   letters %s   t %05.2fs" % (
        np.degrees(angle) % 360, lit, f"{int(ok.sum()):,}", t)
    return "\n".join(lines) + "\n\033[90m" + hud.ljust(cols)[:cols] + "\033[0m"


def main():
    parts, lights = build()
    sys.stdout.write("\033[?25l\033[2J")
    start, size, stars = time.time(), None, None
    try:
        while True:
            cols, rows = shutil.get_terminal_size()
            rows -= 1
            if size != (cols, rows):
                size = (cols, rows)
                stars = rng.choice(cols * rows // 2, max(1, cols * rows // 60), replace=False)
            t = time.time() - start
            sys.stdout.write("\033[H" + render(parts, lights, t, cols, rows, stars))
            sys.stdout.flush()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[0m\033[?25h\n")


if __name__ == "__main__":
    main()
