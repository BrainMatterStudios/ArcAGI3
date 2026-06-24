"""Per-genre synthetic frame generators. Each returns (grid, genre, agent_pos|None, goal).

Genres mirror the public ARC-AGI-3 families (Exp 48/49/51): NAVIGATE (ls20), COLLECT (collect),
PUSH (wa30), AIM (su15/re86), MATCH (sk48/tr87), SYMMETRY (m0r0), CLICK (vc33/tn36). The layouts
aim for the real games' visual idiom so a classifier trained here transfers to real frames — the
exact thing the Phase-1 gate measures.
"""
from __future__ import annotations

import numpy as np

from . import canvas as cv

GENRES = ["NAVIGATE", "COLLECT", "PUSH", "AIM", "MATCH", "SYMMETRY", "CLICK"]


def _pick(rng, pool, exclude):
    return int(rng.choice([c for c in pool if c not in exclude]))


def gen_navigate(rng):
    grid, bg, b = cv.new_board(rng)
    used = {bg}
    wall = _pick(rng, [cv.GREEN, cv.GRAY, cv.BROWN, cv.OLIVE], used); used.add(wall)
    cv.maze_walls(grid, rng, b, wall)
    tgt = _pick(rng, cv.OBJ_COLORS, used); used.add(tgt)
    for _ in range(int(rng.integers(1, 3))):
        cv.place_bordered_square(grid, rng, b, _pick(rng, cv.BORDER_COLORS, used), tgt, size=int(rng.integers(5, 8)))
    av = _pick(rng, [cv.BLACK, cv.TEAL, cv.RED, cv.MAGENTA], used)
    apos = cv.place_avatar(grid, rng, b, av)
    if rng.random() < 0.5:
        cv.progress_bar(grid, rng)
    return grid, "NAVIGATE", apos, "reach the target markers"


def gen_collect(rng):
    grid, bg, b = cv.new_board(rng)
    used = {bg}
    item = _pick(rng, cv.OBJ_COLORS, used); used.add(item)
    cv.scatter_items(grid, rng, b, item, n=int(rng.integers(5, 12)))
    av = _pick(rng, [cv.BLACK, cv.TEAL, cv.RED, cv.WHITE], used)
    apos = cv.place_avatar(grid, rng, b, av)
    if rng.random() < 0.4:
        cv.progress_bar(grid, rng)
    return grid, "COLLECT", apos, "collect all items"


def gen_push(rng):
    grid, bg, b = cv.new_board(rng)
    used = {bg}
    marker = _pick(rng, cv.BORDER_COLORS, used); used.add(marker)
    block = _pick(rng, [cv.RED, cv.MAROON, cv.ORANGE, cv.WHITE], used); used.add(block)
    n = int(rng.integers(1, 4))
    for _ in range(n):  # target markers (bordered squares)
        cv.place_bordered_square(grid, rng, b, marker, bg, size=int(rng.integers(5, 7)))
    for _ in range(n):  # pushable blocks (solid)
        cv.place_block(grid, rng, b, block, size=int(rng.integers(4, 7)))
    av = _pick(rng, [cv.BLACK, cv.TEAL, cv.WHITE], used)
    apos = cv.place_avatar(grid, rng, b, av)
    return grid, "PUSH", apos, "push blocks onto markers"


def gen_aim(rng):
    grid, bg, b = cv.new_board(rng)
    used = {bg}
    # one or two crosshairs (the "cursor"), scattered target squares
    for _ in range(int(rng.integers(1, 3))):
        cv.place_cross(grid, rng, b, _pick(rng, [cv.GRAY, cv.MAROON, cv.WHITE, cv.RED], used))
    tgt_border = _pick(rng, cv.BORDER_COLORS, used); used.add(tgt_border)
    tgt_center = _pick(rng, cv.OBJ_COLORS, used)
    for _ in range(int(rng.integers(3, 8))):
        cv.place_bordered_square(grid, rng, b, tgt_border, tgt_center, size=int(rng.integers(4, 6)))
    return grid, "AIM", None, "aim the crosshair at targets"


def gen_match(rng):
    grid, bg, b = cv.new_board(rng)
    used = {bg}
    dv = cv.divider(grid, rng, b, _pick(rng, [cv.RED, cv.GRAY, cv.WHITE, cv.ORANGE], used), vertical=False)
    used.add(grid[dv[1], (b[1] + b[3]) // 2])
    r0, c0, r1, c1 = b
    split = dv[1]
    # a reference row of coloured blocks (top) and a workspace row (bottom) — arrange to match
    cols = [_pick(rng, cv.OBJ_COLORS, used) for _ in range(int(rng.integers(3, 6)))]
    x = c0 + 4
    for col in cols:
        cv.fill_rect(grid, r0 + 3, x, r0 + 7, x + 4, col); x += 7
    x = c0 + 4
    for col in rng.permutation(cols):
        cv.fill_rect(grid, split + 6, x, split + 10, x + 4, int(col)); x += 7
    apos = cv.place_avatar(grid, rng, (split + 2, c0, r1, c1), _pick(rng, [cv.BLACK, cv.MAGENTA], used))
    return grid, "MATCH", apos, "arrange to match the reference"


def gen_symmetry(rng):
    grid, bg, b = cv.new_board(rng)
    r0, c0, r1, c1 = b
    used = {bg}
    mid = (c0 + c1) // 2
    bg2 = _pick(rng, cv.BG_COLORS, used); used.add(bg2)
    grid[r0:r1 + 1, mid:c1 + 1] = bg2  # two-tone halves (m0r0 style)
    shape = _pick(rng, [cv.GRAY, cv.WHITE, cv.BROWN], used); used.add(shape)
    # a blocky shape on the left; partially (not perfectly) mirrored on the right
    for _ in range(int(rng.integers(4, 8))):
        s = int(rng.integers(3, 6))
        rr = int(rng.integers(r0 + 2, r1 - s)); ccl = int(rng.integers(c0 + 2, mid - s - 1))
        grid[rr:rr + s, ccl:ccl + s] = shape
        if rng.random() < 0.6:  # imperfect mirror
            ccr = (c1 - (ccl - c0)) - s
            grid[rr:rr + s, max(mid, ccr):min(c1, ccr + s)] = shape
    pair = _pick(rng, cv.OBJ_COLORS, used)
    cv.place_block(grid, rng, (r0, c0, r1, mid), pair, size=4)
    cv.place_block(grid, rng, (r0, mid, r1, c1), pair, size=4)
    return grid, "SYMMETRY", None, "make the halves symmetric"


def gen_click(rng):
    grid, bg, b = cv.new_board(rng)
    r0, c0, r1, c1 = b
    used = {bg}
    # large flat colour regions (vc33 style) + a few small special marks, no avatar
    split = int(rng.integers(c0 + 12, c1 - 12))
    grid[r0:r1 + 1, split:c1 + 1] = _pick(rng, [cv.BLACK, cv.GREEN, cv.GRAY], used)
    if rng.random() < 0.6:
        rr = int(rng.integers(r0 + 8, r1 - 8))
        grid[rr:rr + int(rng.integers(3, 6)), c0:c1 + 1] = _pick(rng, [cv.GRAY, cv.WHITE], used)
    mark = _pick(rng, cv.OBJ_COLORS, used)
    for _ in range(int(rng.integers(1, 4))):
        cv.place_block(grid, rng, b, mark, size=int(rng.integers(2, 4)))
    return grid, "CLICK", None, "click the special element"


_GENERATORS = {
    "NAVIGATE": gen_navigate, "COLLECT": gen_collect, "PUSH": gen_push, "AIM": gen_aim,
    "MATCH": gen_match, "SYMMETRY": gen_symmetry, "CLICK": gen_click,
}


def generate_one(rng, genre: str):
    return _GENERATORS[genre](rng)
