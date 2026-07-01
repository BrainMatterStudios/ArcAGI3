"""sc25 glyph-cast + reach solver (fourth archetype). L0: draw the diamond glyph (sieesc_chwjgc) to shrink
the avatar 4x4 -> 2x2, then navigate the small avatar through the maze to the goal. Mechanic RE'd + validated
against the engine (prime click + 4 diamond-slot clicks casts; avatar shrinks; moves +-2).

Uses engine truth for perception to validate the model+planner first; source-free perception is the follow-up.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

# sieesc diamond glyph cells (True) in the 3x3 slot grid; slots at (24+5j, 49+5i), center (26+5j, 51+5i)
DIAMOND = [(0, 1), (1, 0), (1, 2), (2, 1)]


def solve(verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("sc25"))
    env = client.make(game_id=gid, scorecard_id="sc25-solve")
    obs = env.reset()
    g = env._game; lvl = g.current_level

    def spr(name):
        r = [s for s in lvl.get_sprites() if s.name == name]
        return r[0] if r else None
    def click(x, y):
        nonlocal obs
        obs = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

    # --- STAGE 1: draw the diamond glyph -> cast (shrink avatar) ---
    si = [s for s in lvl.get_sprites() if s.name and s.name.startswith("sptivk-sieesc")][0]
    click(si.x + 1, si.y + 1)                 # prime / select the spell (first click is consumed)
    for (i, j) in DIAMOND:
        click(26 + 5 * j, 51 + 5 * i)
    av = spr("pluyoo")
    if verbose:
        print(f"after cast: avatar {(av.x, av.y, av.width, av.height)} (shrunk if 2x2)")

    # --- STAGE 2: navigate the (now small) avatar to the goal ---
    go = spr("exydhv")
    w, h = av.width, av.height
    step = 2 if w <= 2 else 4
    wall = set()
    for s in lvl.get_sprites():
        if getattr(s, "is_collidable", False) and s.name not in ("pluyoo", "exydhv"):
            px = np.asarray(s.pixels)
            for i in range(px.shape[0]):
                for j in range(px.shape[1]):
                    if px[i, j] != -1:
                        wall.add((s.x + j, s.y + i))

    def free(x, y):
        return all(0 <= x < 64 and 0 <= y < 64 and (x + a, y + b) not in wall
                   for a in range(w) for b in range(h))
    def at_goal(x, y):
        return x < go.x + go.width and x + w > go.x and y < go.y + go.height and y + h > go.y

    DIRS = {1: (0, -step), 2: (0, step), 3: (-step, 0), 4: (step, 0)}
    start = (av.x, av.y)
    seen = {start}; q = deque([(start, [])]); plan = None
    while q and plan is None:
        (x, y), path = q.popleft()
        for a, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if at_goal(nx, ny):
                plan = path + [a]; break
            if (nx, ny) not in seen and free(nx, ny):
                seen.add((nx, ny)); q.append(((nx, ny), path + [a]))
    if plan is None:
        print(f"  no nav plan at {w}x{h} (reached {len(seen)} cells) -> may need mid-path scale toggle")
        return 0
    if verbose:
        print(f"  nav plan: {len(plan)} moves")
    lv = int(obs.levels_completed or 0)
    for a in plan:
        obs = env.step(GameAction.from_id(a))
        if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
            print("  *** sc25 L0 SOLVED (glyph-cast + navigate) ***"); return int(obs.levels_completed or 0)
        if obs.state == GameState.GAME_OVER:
            print("  GAME_OVER (budget/desync)"); break
    print(f"  did not solve (levels {obs.levels_completed})"); return int(obs.levels_completed or 0)


if __name__ == "__main__":
    solve()
