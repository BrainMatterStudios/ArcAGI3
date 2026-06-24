"""Per-genre synthetic frame generators (v2 — structures matched to real HOLDOUT frames).

Each returns (grid, genre, agent_pos|None, goal). Colours for every role are drawn uniformly at
random from the 16 palette indices (canvas.roles) — heavy domain randomization so the classifier
learns genre STRUCTURE, not palette. Where a real genre has visual variety (AIM: crosshair vs
aim-line; MATCH: colour-sequence vs shape-pair grid; CLICK: flat-regions vs checkerboard console),
the generator samples among sub-styles.
"""
from __future__ import annotations

import numpy as np

from . import canvas as cv

GENRES = ["NAVIGATE", "COLLECT", "PUSH", "AIM", "MATCH", "SYMMETRY", "CLICK"]


def _board(rng, ncolors):
    cols = cv.roles(rng, ncolors)
    bg = cols[0]
    border = cols[1] if rng.random() < 0.75 else None
    grid, b = cv.new_board(rng, bg, border)
    return grid, b, bg, cols


def gen_navigate(rng):
    # ls20: large solid interior region + bordered target box(es) + small avatar + ticked bar
    grid, b, bg, cols = _board(rng, 7)
    interior, target_b, target_c, av, tick = cols[2], cols[3], cols[4], cols[5], cols[6]
    inner = cv.solid_interior(grid, rng, b, interior)
    for _ in range(int(rng.integers(1, 3))):
        cv.place_bordered_square(grid, rng, b, target_b, target_c, size=int(rng.integers(6, 10)))
    apos = cv.place_avatar(grid, rng, inner, av)
    if rng.random() < 0.6:
        cv.control_bar(grid, rng, cols[3], tick=tick if rng.random() < 0.6 else None)
    return grid, "NAVIGATE", apos, "reach the target"


def gen_collect(rng):
    grid, b, bg, cols = _board(rng, 5)
    item, av = cols[2], cols[3]
    cv.scatter_items(grid, rng, b, item, n=int(rng.integers(6, 14)))
    apos = cv.place_avatar(grid, rng, b, av)
    if rng.random() < 0.4:
        cv.control_bar(grid, rng, cols[4])
    return grid, "COLLECT", apos, "collect all items"


def gen_push(rng):
    # wa30: solid bg + scattered bordered markers + elongated bar block(s) + small avatar
    grid, b, bg, cols = _board(rng, 6)
    marker, block, av = cols[2], cols[3], cols[4]
    n = int(rng.integers(2, 5))
    for _ in range(n):
        cv.place_bordered_square(grid, rng, b, marker, bg, size=int(rng.integers(5, 7)))
    for _ in range(int(rng.integers(1, 3))):
        cv.place_bar(grid, rng, b, block)
    apos = cv.place_avatar(grid, rng, b, av)
    return grid, "PUSH", apos, "push blocks onto markers"


def gen_aim(rng):
    grid, b, bg, cols = _board(rng, 6)
    if rng.random() < 0.5:
        # re86 style: full crosshair(s) + scattered bordered targets
        for _ in range(int(rng.integers(1, 3))):
            cv.place_cross(grid, rng, b, cols[2])
        for _ in range(int(rng.integers(3, 8))):
            cv.place_bordered_square(grid, rng, b, cols[3], cols[4], size=int(rng.integers(3, 6)))
    else:
        # su15 style: a cursor (+) with a dotted aim-line to a ringed target
        r0, c0, r1, c1 = b
        cur = cv.place_avatar(grid, rng, (r1 - 14, c0, r1, c1), cols[2])
        tgt = cv.ring_target(grid, rng, (r0, c0, r0 + 16, c1), cols[3], cols[4], size=int(rng.integers(6, 10)))
        cv.aim_line(grid, (cur[0], cur[1]), tgt, cols[5])
        for _ in range(int(rng.integers(2, 5))):
            cv.place_block(grid, rng, b, cols[5], size=2)
    return grid, "AIM", None, "aim at the target"


def gen_match(rng):
    grid, b, bg, cols = _board(rng, 8)
    r0, c0, r1, c1 = b
    if rng.random() < 0.5:
        # sk48 style: a reference sequence of coloured blocks + a workspace sequence, divider between
        dv = cv.divider(grid, rng, b, cols[2], vertical=False)
        split = dv[1]
        palette = cols[3:8]
        seq = [int(rng.choice(palette)) for _ in range(int(rng.integers(3, 6)))]
        x = c0 + 4
        for col in seq:
            cv.fill_rect(grid, r0 + 3, x, r0 + 7, x + 4, col); x += 7
        x = c0 + 4
        for col in rng.permutation(seq):
            cv.fill_rect(grid, split + 6, x, split + 10, x + 4, int(col)); x += 7
        apos = cv.place_avatar(grid, rng, (split + 2, c0, r1, c1), cols[2])
        return grid, "MATCH", apos, "arrange to match the reference"
    # tr87 style: a grid of paired shape-icons (input -> output rules)
    cv.shape_pair_grid(grid, rng, b, frame=cols[2], a=cols[3], b=cols[4])
    if rng.random() < 0.5:
        cv.fill_rect(grid, (r0 + r1) // 2, c0, (r0 + r1) // 2 + 1, c1, cols[5])  # section divider
    return grid, "MATCH", None, "apply the transformation rule"


def gen_symmetry(rng):
    # m0r0: two-tone vertical halves + a blocky maze/loop shape spanning + paired blocks
    grid, b, bg, cols = _board(rng, 6)
    r0, c0, r1, c1 = b
    mid = (c0 + c1) // 2
    grid[r0:r1 + 1, mid:c1 + 1] = cols[2]  # second-tone right half
    shape = cols[3]
    cx, cy = (c0 + c1) // 2, (r0 + r1) // 2
    for _ in range(int(rng.integers(8, 16))):  # a blocky connected loop-ish shape near the centre
        ang = rng.random() * 6.28; rad = rng.integers(6, 18)
        rr = int(cy + rad * np.sin(ang)); ccx = int(cx + rad * np.cos(ang)); s = int(rng.integers(3, 6))
        if r0 <= rr < r1 - s and c0 <= ccx < c1 - s:
            grid[rr:rr + s, ccx:ccx + s] = shape
    pair = cols[4]
    cv.place_block(grid, rng, (r0, c0, r1, mid), pair, size=int(rng.integers(3, 6)))
    cv.place_block(grid, rng, (r0, mid, r1, c1), pair, size=int(rng.integers(3, 6)))
    return grid, "SYMMETRY", None, "make the halves symmetric"


def gen_click(rng):
    grid, b, bg, cols = _board(rng, 6)
    r0, c0, r1, c1 = b
    if rng.random() < 0.5:
        # tn36 style: a checkerboard console + control bar at the bottom
        m = int(rng.integers(6, 12))
        cv.checkerboard(grid, r0 + 2, c0 + 2, r1 - m, c1 - 2, cols[2], cols[3], cell=int(rng.integers(3, 5)))
        for _ in range(int(rng.integers(1, 4))):
            cv.place_block(grid, rng, (r0 + 2, c0 + 2, r1 - m, c1 - 2), cols[4], size=2)
        cv.control_bar(grid, rng, cols[5], tick=cols[4])
    else:
        # vc33 style: large flat colour regions + a few small marks, no avatar
        split = int(rng.integers(c0 + 12, c1 - 12))
        grid[r0:r1 + 1, split:c1 + 1] = cols[2]
        if rng.random() < 0.6:
            rr = int(rng.integers(r0 + 8, r1 - 8)); grid[rr:rr + int(rng.integers(3, 6)), c0:c1 + 1] = cols[3]
        for _ in range(int(rng.integers(1, 4))):
            cv.place_block(grid, rng, b, cols[4], size=int(rng.integers(2, 4)))
    return grid, "CLICK", None, "click the special element"


_GENERATORS = {
    "NAVIGATE": gen_navigate, "COLLECT": gen_collect, "PUSH": gen_push, "AIM": gen_aim,
    "MATCH": gen_match, "SYMMETRY": gen_symmetry, "CLICK": gen_click,
}


def generate_one(rng, genre: str):
    return _GENERATORS[genre](rng)
