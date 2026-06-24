"""Drawing primitives for synthetic 64x64 ARC-AGI-3-style frames (palette-index grids 0..15).

Frames are int8 grids of palette indices — exactly what `perception.to_grid` returns for real
frames. The classifier consumes a 16-channel ONE-HOT of these indices, so colour *identity* carries
no signal beyond "which channel"; that is why the generators randomize every role→colour assignment
uniformly over all 16 indices (`roles`) — it forces the model to learn genre STRUCTURE, not palette.
v2: structures matched to the real HOLDOUT frames (solid interiors, aim-lines, shape-pair grids,
checkerboard consoles, two-tone symmetry) + heavy domain randomization.
"""
from __future__ import annotations

import numpy as np

SIZE = 64


def roles(rng, k: int) -> list[int]:
    """k distinct palette indices chosen uniformly from all 16 — the core domain-randomization
    lever. One-hot input means colour identity is arbitrary, so randomizing roles kills colour cues."""
    return [int(x) for x in rng.choice(16, size=min(k, 16), replace=False)]


def new_board(rng, bg, border=None):
    """Board filled with bg, optional N-pixel border frame. Returns (grid, (r0,c0,r1,c1) interior)."""
    grid = np.full((SIZE, SIZE), bg, dtype=np.int8)
    r0 = c0 = 0; r1 = c1 = SIZE - 1
    if border is not None:
        w = int(rng.integers(2, 5))
        grid[:w, :] = border; grid[-w:, :] = border; grid[:, :w] = border; grid[:, -w:] = border
        r0 = c0 = w; r1 = c1 = SIZE - 1 - w
    return grid, (r0, c0, r1, c1)


def _clampbox(bounds, h, w):
    r0, c0, r1, c1 = bounds
    return r0, c0, max(r0, r1 - h), max(c0, c1 - w)


def fill_rect(grid, r0, c0, r1, c1, color):
    grid[r0:r1 + 1, c0:c1 + 1] = color


def solid_interior(grid, rng, bounds, color, margin=None):
    """A large solid interior region covering most of the board (ls20's green interior)."""
    r0, c0, r1, c1 = bounds
    m = margin if margin is not None else int(rng.integers(6, 14))
    ir0, ic0, ir1, ic1 = r0 + m, c0 + m, r1 - m, c1 - int(rng.integers(2, m + 1))
    grid[ir0:ir1 + 1, ic0:ic1 + 1] = color
    # carve a couple of corridors (bg pokes through) so it reads as rooms, not a solid block
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:
            rr = int(rng.integers(ir0, ir1)); grid[rr:rr + int(rng.integers(2, 5)), ic0:ic1 + 1] = grid[r0, c0]
        else:
            cc = int(rng.integers(ic0, ic1)); grid[ir0:ir1 + 1, cc:cc + int(rng.integers(2, 5))] = grid[r0, c0]
    return ir0, ic0, ir1, ic1


def place_block(grid, rng, bounds, color, size=None):
    s = size or int(rng.integers(3, 7))
    r0, c0, rr, cc = _clampbox(bounds, s, s)
    r = int(rng.integers(r0, rr + 1)); c = int(rng.integers(c0, cc + 1))
    grid[r:r + s, c:c + s] = color
    return r, c


def place_bar(grid, rng, bounds, color, horizontal=None):
    """An elongated bar block (wa30's red bar / pushable)."""
    r0, c0, r1, c1 = bounds
    horizontal = rng.random() < 0.5 if horizontal is None else horizontal
    if horizontal:
        L = int(rng.integers(8, 18)); h = int(rng.integers(2, 5))
        r = int(rng.integers(r0, r1 - h)); c = int(rng.integers(c0, c1 - L))
        grid[r:r + h, c:c + L] = color
    else:
        L = int(rng.integers(8, 18)); w = int(rng.integers(2, 5))
        r = int(rng.integers(r0, r1 - L)); c = int(rng.integers(c0, c1 - w))
        grid[r:r + L, c:c + w] = color


def place_bordered_square(grid, rng, bounds, border, center, size=None):
    s = size or int(rng.integers(5, 8))
    r0, c0, rr, cc = _clampbox(bounds, s, s)
    r = int(rng.integers(r0, rr + 1)); c = int(rng.integers(c0, cc + 1))
    grid[r:r + s, c:c + s] = border
    grid[r + 1:r + s - 1, c + 1:c + s - 1] = center
    return r, c


def ring_target(grid, rng, bounds, ring, center, size=None):
    """A ringed circle target (su15 / aiming goal)."""
    s = size or int(rng.integers(5, 9))
    r0, c0, rr, cc = _clampbox(bounds, s, s)
    r = int(rng.integers(r0, rr + 1)); c = int(rng.integers(c0, cc + 1))
    yy, xx = np.ogrid[:s, :s]
    d = (yy - (s - 1) / 2) ** 2 + (xx - (s - 1) / 2) ** 2
    sub = grid[r:r + s, c:c + s]
    sub[d <= (s / 2) ** 2] = ring
    sub[d <= (s / 2 - 1.5) ** 2] = center
    return r + s // 2, c + s // 2


def aim_line(grid, p0, p1, color):
    """A dotted line between two points (su15 aim-line)."""
    r0, c0 = p0; r1, c1 = p1
    n = max(abs(r1 - r0), abs(c1 - c0)) + 1
    for i in range(0, n, 2):  # dotted
        t = i / max(n - 1, 1)
        rr = int(round(r0 + (r1 - r0) * t)); cc = int(round(c0 + (c1 - c0) * t))
        if 0 <= rr < SIZE and 0 <= cc < SIZE:
            grid[rr, cc] = color


def place_cross(grid, rng, bounds, color, thick=2):
    r0, c0, r1, c1 = bounds
    cr = int(rng.integers(r0 + 6, r1 - 6)); cc = int(rng.integers(c0 + 6, c1 - 6))
    half = int(rng.integers(10, 22))
    grid[cr:cr + thick, max(c0, cc - half):min(c1, cc + half)] = color
    grid[max(r0, cr - half):min(r1, cr + half), cc:cc + thick] = color
    return cr, cc


def place_avatar(grid, rng, bounds, color):
    r0, c0, r1, c1 = bounds
    r = int(rng.integers(r0 + 2, max(r0 + 3, r1 - 4))); c = int(rng.integers(c0 + 2, max(c0 + 3, c1 - 4)))
    if rng.random() < 0.5:
        grid[r + 1, c:c + 3] = color; grid[r:r + 3, c + 1] = color  # plus
    else:
        grid[r:r + 3, c:c + 3] = color  # block
    return r, c


def scatter_items(grid, rng, bounds, color, n):
    return [place_block(grid, rng, bounds, color, size=int(rng.integers(2, 4))) for _ in range(n)]


def shape_icon(grid, rng, r, c, size, color):
    """A small distinct glyph (random connected pixel pattern) — tr87 rule-icons / abstract shapes."""
    pat = rng.random((size, size)) < 0.5
    pat[size // 2, :] = True  # keep it connected-ish
    grid[r:r + size, c:c + size][pat] = color


def shape_pair_grid(grid, rng, bounds, frame, a, b):
    """A grid of paired shape-icons 'A -> B' (tr87 rule-induction look)."""
    r0, c0, r1, c1 = bounds
    sz = int(rng.integers(4, 6)); gap = 3
    cellw = sz * 2 + gap + 2
    rows = int(rng.integers(2, 4)); cols = int(rng.integers(2, 4))
    rr = r0 + 2
    for _ in range(rows):
        ccol = c0 + 2
        for _ in range(cols):
            if ccol + cellw > c1 or rr + sz > r1:
                break
            grid[rr:rr + sz, ccol:ccol + sz] = frame
            shape_icon(grid, rng, rr, ccol, sz, a)
            grid[rr:rr + sz, ccol + sz + gap:ccol + 2 * sz + gap] = frame
            shape_icon(grid, rng, rr, ccol + sz + gap, sz, b)
            ccol += cellw
        rr += sz + gap + 1


def checkerboard(grid, r0, c0, r1, c1, ca, cb, cell=4):
    for i, r in enumerate(range(r0, r1, cell)):
        for j, c in enumerate(range(c0, c1, cell)):
            grid[r:r + cell, c:c + cell] = ca if (i + j) % 2 == 0 else cb


def maze_walls(grid, rng, bounds, color, n=None):
    r0, c0, r1, c1 = bounds
    for _ in range(n or int(rng.integers(3, 7))):
        if rng.random() < 0.5:
            r = int(rng.integers(r0, r1)); cs = int(rng.integers(c0, c1 - 8))
            grid[r:r + 2, cs:cs + int(rng.integers(8, 24))] = color
        else:
            c = int(rng.integers(c0, c1)); rs = int(rng.integers(r0, r1 - 8))
            grid[rs:rs + int(rng.integers(8, 24)), c:c + 2] = color


def control_bar(grid, rng, color, tick=None):
    """A bottom HUD/progress/control bar, optionally with tick marks (ls20/tn36)."""
    frac = rng.random()
    h = int(rng.integers(2, 5))
    grid[-h:, :int(SIZE * frac) if rng.random() < 0.5 else SIZE] = color
    if tick is not None:
        for c in range(2, SIZE - 2, 5):
            grid[-h:, c:c + 1] = tick


def divider(grid, rng, bounds, color, vertical=None):
    r0, c0, r1, c1 = bounds
    vertical = rng.random() < 0.5 if vertical is None else vertical
    if vertical:
        c = (c0 + c1) // 2 + int(rng.integers(-4, 5))
        grid[r0:r1 + 1, c:c + 2] = color
        return ("v", c)
    r = (r0 + r1) // 2 + int(rng.integers(-4, 5))
    grid[r:r + 2, c0:c1 + 1] = color
    return ("h", r)
