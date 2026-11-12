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

Y0, Y1 = 0.45, 2.75          # foliage from Y0 to Y1
TIERS = 5                    # branch layers
N_LEAF, N_LIGHTS, N_BALLS = 30000, 170, 26
LEAF, TRUNK, GARLAND, BALL, GIFT, SKIRT, LIGHT = range(7)

# timeline (seconds)
GROW, GARLANDS, BALLS, LIGHTS, STAR, GIFTS = 5.0, 4.5, 5.5, 5.0, 8.5, 8.8


def envelope(y):
    f = np.clip((y - Y0) / (Y1 - Y0), 0, 1)
    tier = (f * TIERS) % 1.0
    return (0.03 + 1.02 * (1 - f) ** 0.95) * (0.86 + 0.16 * (1 - tier))


class Parts:
    def __init__(s):
        s.P, s.C, s.ch, s.kind, s.key = [], [], [], [], []

    def add(s, pts, col, kind, key=0.0, chars=None):
        n = len(pts)
        s.P.append(pts)
        s.C.append(np.broadcast_to(np.asarray(col, float), (n, 3)))
        s.ch.append(chars if chars is not None else rng.choice(LETTERS, n))
        s.kind.append(np.full(n, kind))
        s.key.append(np.broadcast_to(np.asarray(key, float), (n,)))

    def done(s):
        s.P, s.C = np.vstack(s.P), np.vstack(s.C)
        s.ch, s.kind, s.key = np.concatenate(s.ch), np.concatenate(s.kind), np.concatenate(s.key)
        return s


def ring(y, a, grow=1.0):
    r = envelope(y) * grow
    return np.c_[r * np.cos(a), y, r * np.sin(a)]


def build():
    parts = Parts()

    # foliage: points packed toward the outer skin
    y = rng.uniform(Y0, Y1, N_LEAF)
    d = rng.random(N_LEAF) ** 0.3
    green = np.c_[0.07 + 0.12 * rng.random(N_LEAF), 0.30 + 0.32 * rng.random(N_LEAF), 0.13 + 0.13 * rng.random(N_LEAF)]
    parts.add(ring(y, rng.uniform(0, 2 * np.pi, N_LEAF), d), green, LEAF, y)

    # trunk
    n = 600
    a, y = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, Y0 + 0.1, n)
    parts.add(np.c_[0.12 * np.cos(a), y, 0.12 * np.sin(a)], [0.42, 0.26, 0.13], TRUNK, y)

    # three gold garlands spiralling from the top down
    for k in range(3):
        s = np.sort(rng.random(2000))
        y = Y1 - 0.25 - s * (Y1 - Y0 - 0.35) + rng.normal(0, 0.022, 2000)
        pts = ring(y, s * 3.2 * 2 * np.pi + k * 2.1, 1.03)
        parts.add(pts, [1.0, 0.78, 0.30], GARLAND, s)

    # baubles: small spheres stuck to the surface
    for b in range(N_BALLS):
        y = rng.uniform(Y0 + 0.15, Y1 - 0.4)
        c = ring(np.array([y]), np.array([rng.uniform(0, 2 * np.pi)]), 1.02)[0]
        n = rng.normal(size=(70, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        base = np.array([1.0, 0.72, 0.2]) if rng.random() < 0.3 else np.array([0.86, 0.07, 0.11])
        lit = 0.45 + 0.75 * np.clip(n @ np.array([-0.4, 0.6, -0.7]), 0, 1)
        parts.add(c + 0.075 * n, base * lit[:, None], BALL, b, rng.choice(list("@O0Q8%"), 70))

    # fairy lights, switched on one by one
    y = rng.uniform(Y0 + 0.05, Y1 - 0.15, N_LIGHTS)
    warm = np.array([[1, .85, .45], [1, 1, .9], [1, .55, .2]])[rng.choice(3, N_LIGHTS, p=[.6, .3, .1])]
    parts.add(ring(y, rng.uniform(0, 2 * np.pi, N_LIGHTS), 1.04), warm, LIGHT, np.arange(N_LIGHTS), np.full(N_LIGHTS, "o"))

    # presents with gold ribbons
    for x, z, w, h, dp, col in [(-0.90, 0.35, .34, .34, .32, [.8, .06, .1]), (0.95, 0.05, .38, .28, .32, [.1, .25, .85]),
                                (-0.25, 0.95, .30, .24, .30, [.8, .06, .1]), (0.60, 0.80, .26, .34, .26, [.1, .25, .85])]:
        u = rng.uniform(-0.5, 0.5, (2600, 3))
        u[np.arange(2600), rng.integers(0, 3, 2600)] = rng.choice([-0.5, 0.5], 2600)   # push onto a face
        ribbon = (np.abs(u[:, 0]) < 0.07) | (np.abs(u[:, 2]) < 0.07)
        pts = np.c_[x + u[:, 0] * w, (u[:, 1] + 0.5) * h, z + u[:, 2] * dp]
        parts.add(pts, np.where(ribbon[:, None], [1, .8, .3], col), GIFT)

    # tree skirt
    n = 4000
    r, a = np.sqrt(rng.uniform(0.15 ** 2, 1.15 ** 2, n)), rng.uniform(0, 2 * np.pi, n)
    cream = np.c_[0.80 + 0.12 * rng.random(n), 0.72 + 0.1 * rng.random(n), 0.56 + 0.1 * rng.random(n)]
    parts.add(np.c_[r * np.cos(a), np.full(n, 0.015), r * np.sin(a)], cream, SKIRT)
    return parts.done()


def build_ground():
    gx, gz = np.meshgrid(np.arange(-5, 5, 0.07), np.arange(-3.0, 1.1, 0.07))
    return np.c_[gx.ravel(), np.zeros(gx.size), gz.ravel()]


def project(P, angle, cols, rows):
    c, s = np.cos(angle), np.sin(angle)
    x = P[:, 0] * c - P[:, 2] * s
    z = P[:, 0] * s + P[:, 2] * c
    y = P[:, 1] - 1.3
    tilt = 0.28                                   # camera looks slightly down
    y, z = y * np.cos(tilt) + z * np.sin(tilt), z * np.cos(tilt) - y * np.sin(tilt)
    persp = 4.6 / (4.6 + z)
    scale = min(rows / 4.1, cols / 7.6)
    sx = (cols / 2 + x * persp * scale * 2.0).astype(int)   # x2: chars are tall
    sy = (rows * 0.54 - y * persp * scale).astype(int)
    return sx, sy, z


def visible(parts, t):
    k, key, y = parts.kind, parts.key, parts.P[:, 1]
    grow_y = -0.1 + (Y1 + 0.45) * min(1.0, t / GROW)
    mask = np.select(
        [(k == LEAF) | (k == TRUNK), k == GARLAND, k == BALL, k == LIGHT, (k == GIFT) | (k == SKIRT)],
        [y < grow_y, key < (t - GARLANDS) / 3, key < (t - BALLS) * 5, key < (t - LIGHTS) * 36,
         np.full(len(k), t > GIFTS)], False)
    return mask, grow_y


def draw_star(out_ch, out_col, angle, t, cols, rows):
    (tx,), (ty,), _ = project(np.array([[0, Y1 + 0.12, 0]]), angle, cols, rows)
    grow = min(1.0, (t - STAR) / 0.8)
    for i in range(12):
        a = i * np.pi / 6 + t * 0.15
        ch = "-/|\\"[int(round((a % np.pi) / (np.pi / 4))) % 4]
        for d in np.arange(2.5, (3.5 if i % 2 else 6) * grow, 0.6):
            x, y = int(tx + np.cos(a) * d * 2), int(ty - np.sin(a) * d)
            if 0 <= x < cols and 0 <= y < rows:
                out_ch[y * cols + x], out_col[y * cols + x] = ch, np.array([1, .8, .35]) * (1 - d / 10)
    for dx, dy in [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        x, y = tx + dx, ty + dy
        if 0 <= x < cols and 0 <= y < rows:
            out_ch[y * cols + x], out_col[y * cols + x] = "#", [1, .92, .5]


def render(parts, ground, t, cols, rows, stars):
    angle = t * 0.32
    vis, grow_y = visible(parts, t)
    P, col, ch, kind, key = parts.P[vis], parts.C[vis].copy(), parts.ch[vis], parts.kind[vis], parts.key[vis]

    # fresh growth is frosted white, so the tree looks snow-capped while it grows
    frost = np.clip(1 - (grow_y - P[:, 1]) / 0.35, 0, 1)[:, None] * (kind == LEAF)[:, None]
    col = col * (1 - frost) + 0.95 * frost
    on = kind == LIGHT
    col[on] *= (0.65 + 0.35 * np.sin(t * 3 + key[on] * 2.3))[:, None]

    sx, sy, z = project(P, angle, cols, rows)
    col = col * np.clip(1.15 - 0.42 * (z + 1.0), 0.3, 1.15)[:, None]
    col[on] = np.maximum(col[on], 0.5)

    gx, gy, gz = project(ground, 0.0, cols, rows)
    fade = np.clip(1 - np.hypot(ground[:, 0], ground[:, 2] + 0.3) / 5, 0, 1) ** 1.5 * 0.9
    sx, sy, z = np.r_[sx, gx], np.r_[sy, gy], np.r_[z, gz + 0.02]
    col = np.vstack([col, np.c_[fade * 0.8, fade * 0.85, fade]])
    ch = np.r_[ch, np.full(len(ground), "-")]

    ok = (sx >= 0) & (sx < cols) & (sy >= 0) & (sy < rows)
    order = np.argsort(-z[ok])                     # far first, near overwrite
    cell = (sy[ok] * cols + sx[ok])[order]
    out_ch = np.full(rows * cols, " ", dtype="<U1")
    out_col = np.zeros((rows * cols, 3))
    out_ch[stars] = "."
    out_col[stars] = (0.45 + 0.4 * np.sin(t * 2 + stars))[:, None]
    out_ch[cell] = ch[ok][order]
    out_col[cell] = np.clip(col[ok][order], 0, 1)
    is_tree = np.zeros(rows * cols, bool)
    is_tree[cell] = (np.arange(len(sx)) < len(P))[ok][order]
    if t > STAR:
        draw_star(out_ch, out_col, angle, t, cols, rows)

    rgb = (np.clip(out_col, 0, 1) * 255).astype(int)
    lines = []
    for r in range(rows):
        row = []
        for i in range(r * cols, (r + 1) * cols):
            row.append(" " if out_ch[i] == " " else "\033[38;2;%d;%d;%dm%s" % (*rgb[i], out_ch[i]))
        lines.append("".join(row) + "\033[0m")
    hud = "rotation %06.2f°   lights %3d   letters %s   t %05.2fs" % (
        np.degrees(angle) % 360, int(on.sum()), f"{int(is_tree.sum()):,}", t)
    return "\n".join(lines) + "\n\033[90m" + hud.ljust(cols)[:cols] + "\033[0m"


def main():
    parts, ground = build(), build_ground()
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
            sys.stdout.write("\033[H" + render(parts, ground, t, cols, rows, stars))
            sys.stdout.flush()
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[0m\033[?25h\n")


if __name__ == "__main__":
    main()
