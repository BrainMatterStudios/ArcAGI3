"""dc22 SOURCE-FREE reach-cell + bridge-panel solver. The avatar + goal live in disconnected floor clusters;
clicking the panel buttons toggles linked bridges (intangible = walkable) in pairs, so the solution INTERLEAVES
clicks and movement. Joint search over (avatar frame-position x frame-state) with actions = move(1-4) + click
each perceived button; BFS via reset+replay with state dedup; execute the found plan.

Frame perception only: the frame IS the rendered display, so a button is clicked at its frame centroid (no
camera inversion). Avatar = the small distinct movable object; buttons = compact colored regions in the panel.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

AVATAR_COLOR = 14
GOAL_COLOR = 11


def _avatar_pos(grid):
    ys, xs = np.where(grid == AVATAR_COLOR)
    return (int(round(xs.mean())), int(round(ys.mean()))) if len(ys) else None


def _click_targets(grid):
    """candidate buttons: compact colored regions that are not avatar/goal/background, right of the play area."""
    bg = P.detect_background(grid)
    out = []
    for o in P.connected_components(grid, background=bg):
        if o.color in (bg, AVATAR_COLOR, GOAL_COLOR):
            continue
        r0, c0, r1, c1 = o.bbox
        if o.size >= 6 and c0 >= 30:          # panel region (right of the play area)
            # click a solid interior cell (its centroid rounds onto a non-bg pixel)
            cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
            if grid[cy, cx] == o.color:
                out.append((cx, cy))
    return sorted(set(out))


def solve(verbose=True, max_nodes=6000):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("dc22"))
    env = client.make(game_id=gid, scorecard_id="dc22-solve")
    obs = env.reset()

    grid0 = P.to_grid(obs.frame)
    clicks = _click_targets(grid0)
    if verbose:
        print(f"avatar@{_avatar_pos(grid0)} click-buttons={clicks}")
    actions = [("S", 1), ("S", 2), ("S", 3), ("S", 4)] + [("C", cx, cy) for (cx, cy) in clicks]

    def apply(tok):
        nonlocal obs
        if tok[0] == "C":
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        else:
            obs = env.step(GameAction.from_id(tok[1]))

    def replay(seq):
        nonlocal obs
        obs = env.reset()
        for tok in seq:
            apply(tok)
            if obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1:
                return True
        return False

    def state():
        # dedup on avatar position + the STRUCTURAL frame (mask the play area's HUD counter row/col so the
        # per-step countdown doesn't make every state look new). Hash rows 0-55, cols 0-55 (excludes HUD edges).
        g = P.to_grid(obs.frame)
        return (_avatar_pos(g), hash(g[:56, :56].tobytes()))

    replay([])
    seen = {state()}
    q = deque([[]]); sol = None; nodes = 0
    while q and sol is None and nodes < max_nodes:
        seq = q.popleft()
        for a in actions:
            replay(seq); apply(a); nodes += 1
            if obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1:
                sol = seq + [a]; break
            st = state()
            if st not in seen:
                seen.add(st); q.append(seq + [a])
    if verbose:
        print(f"joint search: nodes={nodes} states={len(seen)} -> plan {len(sol) if sol else None}")
    if sol is None:
        print("  no plan found"); return 0
    replay(sol)
    if int(obs.levels_completed or 0) >= 1 or obs.state == GameState.WIN:
        print(f"  *** dc22 L0 SOLVED SOURCE-FREE (interleaved click+move, {len(sol)} actions) ***")
        return int(obs.levels_completed or 0)
    return 0


if __name__ == "__main__":
    solve()
