"""re86 SOURCE-FREE paint-to-stencil solver (frame perception only — no engine truth). Pieces = large
single-color movable blobs; target cells = small same-color regions framed by color 4 (the stencil); active
piece = the blob containing the color-0 cursor. Heuristic: for each target color, cycle (ACTION5) until that
color's piece is active, then drag it toward the target-color centroid (its trail paints the stencil)."""
from __future__ import annotations
from collections import Counter
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

FRAME_COLOR = 4   # stencil frame ("don't care")
CURSOR = 0        # active-piece marker


def _frame_adjacent(grid):
    """boolean mask: cells within 1 of a FRAME_COLOR(4) pixel (target centers are surrounded by the frame)."""
    fm = (grid == FRAME_COLOR)
    adj = np.zeros_like(fm)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            adj |= np.roll(np.roll(fm, dr, axis=0), dc, axis=1)
    return adj


def perceive(grid):
    """pieces = colored pixels NOT frame-adjacent (the movable shapes); targets = colored pixels frame-adjacent
    (the stencil centers). Pieces and targets share colors, so we split by frame-adjacency, not size."""
    bg = P.detect_background(grid)
    adj = _frame_adjacent(grid)
    rows = np.arange(grid.shape[0])[:, None] * np.ones((1, grid.shape[1]))
    pieces = {}   # color -> (cy, cx)
    targets = {}  # color -> (cy, cx) centroid of target cells
    for c in range(16):
        if c in (bg, FRAME_COLOR, CURSOR):
            continue
        cmask = (grid == c) & (rows < 60)      # exclude the bottom HUD bar
        if not cmask.any():
            continue
        tmask = cmask & adj
        pmask = cmask & ~adj
        if tmask.sum() > 0:
            ys, xs = np.where(tmask); targets[c] = (ys.mean(), xs.mean())
        if pmask.sum() >= 8:
            ys, xs = np.where(pmask); pieces[c] = (ys.mean(), xs.mean())
    return pieces, targets


def active_color(grid):
    """the active piece carries the color-0 cursor -> the perceived piece whose centroid is nearest the cursor
    (uses perceive() pieces so cross-shaped pieces with small arms + the HUD bar are handled correctly)."""
    ys, xs = np.where(grid == CURSOR)
    if len(ys) == 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    pieces, _ = perceive(grid)
    if not pieces:
        return None
    return min(pieces, key=lambda c: abs(pieces[c][0] - cy) + abs(pieces[c][1] - cx))


def solve(verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("re86"))
    env = client.make(game_id=gid, scorecard_id="re86-sf")
    obs = env.reset()

    def grid():
        return P.to_grid(obs.frame)
    def step(a):
        nonlocal obs
        obs = env.step(GameAction.from_id(a))

    pieces, targets = perceive(grid())
    if verbose:
        print(f"pieces={ {c: (round(v[0]), round(v[1])) for c, v in pieces.items()} } "
              f"targets={ {c: (round(v[0]), round(v[1])) for c, v in targets.items()} }")
    order = [c for c in targets if c in pieces]
    if not order:
        print("  no paintable (piece,target) color pairs -> abstain"); return 0
    def cursor():
        ys, xs = np.where(grid() == CURSOR)
        return (ys.mean(), xs.mean()) if len(ys) else None

    lv = int(obs.levels_completed or 0); n = 0
    for color in order:
        ty, tx = targets[color]
        # select this color's piece via ACTION5 cycling
        for _ in range(5):
            if active_color(grid()) == color:
                break
            step(5); n += 1
        # drag the active piece's HEAD (the color-0 cursor) toward the target; the trail paints it
        for _ in range(50):
            cur = cursor()
            if cur is None:
                break
            dy, dx = ty - cur[0], tx - cur[1]
            if abs(dx) + abs(dy) < 2:
                break
            a = (4 if dx > 0 else 3) if abs(dx) >= abs(dy) else (2 if dy > 0 else 1)
            step(a); n += 1
            if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
                print(f"  *** re86 L0 SOLVED SOURCE-FREE ({n} actions) ***"); return int(obs.levels_completed or 0)
            if obs.state == GameState.GAME_OVER:
                print("  GAME_OVER"); return 0
    print(f"  did not solve in {n} actions (levels {obs.levels_completed})"); return int(obs.levels_completed or 0)


if __name__ == "__main__":
    solve()
