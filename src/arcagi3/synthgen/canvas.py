"""Drawing primitives for synthetic 64x64 ARC-AGI-3-style frames (palette-index grids 0..15).

Frames are int8 grids whose values are palette indices, exactly the representation
`perception.to_grid` returns for real frames (verified: 64x64 int8, values 0..15). The genre
generators (genres.py) compose these primitives into game-family layouts; visual fidelity to the
real games is the whole point — bordered boards, blocky connected objects, edge HUD bars.
"""
from __future__ import annotations

import numpy as np

SIZE = 64
# Palette indices (mirror vlm_prior_probe.PAL ordering). Used to pick plausible colours per genre.
BLACK, BLUE, RED, GREEN, YELLOW, GRAY, MAGENTA, ORANGE = 0, 1, 2, 3, 4, 5, 6, 7
LIGHTBLUE, MAROON, BROWN, PURPLE, TEAL, OLIVE, PERIWINKLE, WHITE = 8, 9, 10, 11, 12, 13, 14, 15

# Colours that read as "background fill" in the real games (large flat regions).
BG_COLORS = [GRAY, YELLOW, GREEN, BLUE, BLACK, BROWN, WHITE]
# Colours that read as "object / marker" (small salient blocks).
OBJ_COLORS = [MAROON, RED, ORANGE, MAGENTA, TEAL, PURPLE, LIGHTBLUE, OLIVE, WHITE, YELLOW]
BORDER_COLORS = [YELLOW, MAGENTA, PURPLE, ORANGE, RED, GRAY]


def new_board(rng) -> tuple[np.ndarray, int, tuple[int, int, int, int]]:
    """Blank board filled with a bg colour, optional N-pixel border frame. Returns
    (grid, bg_color, (r0, c0, r1, c1) interior bounds inclusive)."""
    bg = int(rng.choice(BG_COLORS))
    grid = np.full((SIZE, SIZE), bg, dtype=np.int8)
    r0 = c0 = 0
    r1 = c1 = SIZE - 1
    if rng.random() < 0.7:  # most real boards have a coloured border frame
        b = int(rng.choice([c for c in BORDER_COLORS if c != bg]))
        w = int(rng.integers(2, 5))
        grid[:w, :] = b; grid[-w:, :] = b; grid[:, :w] = b; grid[:, -w:] = b
        r0 = c0 = w; r1 = c1 = SIZE - 1 - w
    return grid, bg, (r0, c0, r1, c1)


def _clampbox(bounds, h, w):
    r0, c0, r1, c1 = bounds
    return r0, c0, max(r0, r1 - h), max(c0, c1 - w)


def fill_rect(grid, r0, c0, r1, c1, color):
    grid[r0:r1 + 1, c0:c1 + 1] = color


def place_block(grid, rng, bounds, color, size=None) -> tuple[int, int]:
    """A solid square block. Returns its top-left (r, c)."""
    s = size or int(rng.integers(3, 7))
    r0, c0, rr, cc = _clampbox(bounds, s, s)
    r = int(rng.integers(r0, rr + 1)); c = int(rng.integers(c0, cc + 1))
    grid[r:r + s, c:c + s] = color
    return r, c


def place_bordered_square(grid, rng, bounds, border, center, size=None) -> tuple[int, int]:
    """A target-marker: a bordered square with a different-coloured centre (re86/wa30/collect style)."""
    s = size or int(rng.integers(5, 8))
    r0, c0, rr, cc = _clampbox(bounds, s, s)
    r = int(rng.integers(r0, rr + 1)); c = int(rng.integers(c0, cc + 1))
    grid[r:r + s, c:c + s] = border
    grid[r + 1:r + s - 1, c + 1:c + s - 1] = center
    return r, c


def place_cross(grid, rng, bounds, color, thick=2) -> tuple[int, int]:
    """A crosshair: full-ish horizontal + vertical bar through a centre (aim games)."""
    r0, c0, r1, c1 = bounds
    cr = int(rng.integers(r0 + 6, r1 - 6)); cc = int(rng.integers(c0 + 6, c1 - 6))
    half = int(rng.integers(10, 22))
    grid[cr:cr + thick, max(c0, cc - half):min(c1, cc + half)] = color
    grid[max(r0, cr - half):min(r1, cr + half), cc:cc + thick] = color
    return cr, cc


def place_avatar(grid, rng, bounds, color) -> tuple[int, int]:
    """A small distinctive avatar — a plus sign or a 3x3 block (ls20/collect style)."""
    r0, c0, r1, c1 = bounds
    r = int(rng.integers(r0 + 2, r1 - 4)); c = int(rng.integers(c0 + 2, c1 - 4))
    if rng.random() < 0.5:  # plus
        grid[r + 1, c:c + 3] = color
        grid[r:r + 3, c + 1] = color
    else:                   # block
        grid[r:r + 3, c:c + 3] = color
    return r, c


def scatter_items(grid, rng, bounds, color, n) -> list[tuple[int, int]]:
    """n small scattered blocks (collect items)."""
    out = []
    for _ in range(n):
        out.append(place_block(grid, rng, bounds, color, size=int(rng.integers(2, 4))))
    return out


def maze_walls(grid, rng, bounds, color, n=None):
    """A few axis-aligned wall segments (navigation maze structure)."""
    r0, c0, r1, c1 = bounds
    for _ in range(n or int(rng.integers(3, 7))):
        if rng.random() < 0.5:  # horizontal
            r = int(rng.integers(r0, r1)); cs = int(rng.integers(c0, c1 - 8))
            grid[r:r + 2, cs:cs + int(rng.integers(8, 24))] = color
        else:                   # vertical
            c = int(rng.integers(c0, c1)); rs = int(rng.integers(r0, r1 - 8))
            grid[rs:rs + int(rng.integers(8, 24)), c:c + 2] = color


def progress_bar(grid, rng, color=None):
    """An edge HUD/progress bar (the elements that broke earlier explorers; present in many games)."""
    color = color if color is not None else int(rng.choice([GRAY, WHITE, LIGHTBLUE, RED]))
    frac = rng.random()
    if rng.random() < 0.5:  # bottom row band
        h = int(rng.integers(2, 4))
        grid[-h:, :int(SIZE * frac)] = color
    else:                   # left column band
        w = int(rng.integers(2, 4))
        grid[:int(SIZE * frac), :w] = color


def divider(grid, rng, bounds, color, vertical=None):
    """A line splitting the board into two regions (match workspace/reference; symmetry axis)."""
    r0, c0, r1, c1 = bounds
    vertical = rng.random() < 0.5 if vertical is None else vertical
    if vertical:
        c = (c0 + c1) // 2 + int(rng.integers(-4, 5))
        grid[r0:r1 + 1, c:c + 2] = color
        return ("v", c)
    r = (r0 + r1) // 2 + int(rng.integers(-4, 5))
    grid[r:r + 2, c0:c1 + 1] = color
    return ("h", r)
