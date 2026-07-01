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


def perceive(grid):
    bg = P.detect_background(grid)
    comps = P.connected_components(grid, background=bg)
    pieces = {}   # color -> (centroid, size)  (large single-color blobs)
    targets = {}  # color -> list of cell centroids (small framed markers)
    # a cell is a "target" if it is colored (not bg/4/0) and adjacent to a frame-color(4) pixel
    frame_cells = set()
    ys, xs = np.where(grid == FRAME_COLOR)
    for r, c in zip(ys.tolist(), xs.tolist()):
        frame_cells.add((r, c))
    for o in comps:
        if o.color in (bg, FRAME_COLOR, CURSOR):
            continue
        if o.size >= 20:                       # large blob = a movable piece
            if o.color not in pieces or o.size > pieces[o.color][1]:
                pieces[o.color] = (o.centroid, o.size)
        else:                                   # small region: target if near a frame
            r0, c0, r1, c1 = o.bbox
            near_frame = any((r, c) in frame_cells
                             for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2))
            if near_frame:
                targets.setdefault(o.color, []).append(o.centroid)
    return pieces, targets


def active_color(grid):
    """the piece currently carrying the color-0 cursor -> the dominant non-0 color touching a 0 pixel."""
    bg = P.detect_background(grid)
    for o in P.connected_components(grid, background=bg):
        px_colors = None
    # find components that contain BOTH color-0 and a piece color: the active blob renders 0 inside it.
    ys, xs = np.where(grid == CURSOR)
    if len(ys) == 0:
        return None
    cy, cx = int(ys.mean()), int(xs.mean())
    # nearest large-blob color to the cursor
    best = None
    for o in P.connected_components(grid, background=bg):
        if o.color in (bg, FRAME_COLOR, CURSOR) or o.size < 20:
            continue
        d = abs(o.centroid[0] - cy) + abs(o.centroid[1] - cx)
        if best is None or d < best[0]:
            best = (d, o.color)
    return best[1] if best else None


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
        print(f"pieces={ {c: (round(v[0][0]), round(v[0][1])) for c, v in pieces.items()} } "
              f"target colors={ {c: len(ps) for c, ps in targets.items()} }")
    order = [c for c in targets if c in pieces]
    if not order:
        print("  no paintable (piece,target) color pairs -> abstain"); return 0
    lv = int(obs.levels_completed or 0); n = 0
    for color in order:
        tc = targets[color]
        tx = np.mean([p[1] for p in tc]); ty = np.mean([p[0] for p in tc])
        # select this color's piece via ACTION5 cycling
        for _ in range(5):
            if active_color(grid()) == color:
                break
            step(5); n += 1
        # drag toward the target centroid
        for _ in range(50):
            g = grid(); pieces2, _ = perceive(g)
            if color not in pieces2:
                break
            (pcy, pcx), _ = pieces2[color]
            dx, dy = tx - pcx, ty - pcy
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
