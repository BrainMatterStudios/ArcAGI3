"""dc22 reach-cell solver. L0: the buezna panel walls don't block the play area, so the avatar (2x2, moves
+-2) just needs to navigate the maze to the goal. BFS shortest path on the walkable region (avoid walls AND
stay on the vcha floor), execute. Validated against the engine."""
from __future__ import annotations
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState


def solve(verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("dc22"))
    env = client.make(game_id=gid, scorecard_id="dc22-solve")
    obs = env.reset()
    g = env._game; lvl = g.current_level
    av = lvl.get_sprites_by_tag("jfva")[0]; go = lvl.get_sprites_by_tag("goknoi")[0]
    STEP = 2
    # walkable = the engine's own floor-tile check sxnzvaqltp (avatar must land on a floor tile)
    def free(x, y):
        if not (0 <= x < 64 and 0 <= y < 64):
            return False
        try:
            return g.sxnzvaqltp(x, y, av) is not None
        except Exception:
            return False

    DIRS = {1: (0, -STEP), 2: (0, STEP), 3: (-STEP, 0), 4: (STEP, 0)}
    start = (av.x, av.y); seen = {start}; q = deque([(start, [])]); plan = None
    while q and plan is None:
        (x, y), path = q.popleft()
        for a, (dx, dy) in DIRS.items():
            nx, ny = x + dx, y + dy
            if (nx, ny) == (go.x, go.y):
                plan = path + [a]; break
            if (nx, ny) not in seen and free(nx, ny):
                seen.add((nx, ny)); q.append(((nx, ny), path + [a]))
    if plan is None:
        print("  no path found"); return 0
    floor = seen
    if verbose:
        print(f"  nav plan: {len(plan)} moves (floor cells={len(floor)})")
    lv = int(obs.levels_completed or 0)
    for a in plan:
        obs = env.step(GameAction.from_id(a))
        if int(obs.levels_completed or 0) > lv or obs.state == GameState.WIN:
            print("  *** dc22 L0 SOLVED (reach-cell BFS) ***"); return int(obs.levels_completed or 0)
        if obs.state == GameState.GAME_OVER:
            print("  GAME_OVER (step limit/desync)"); break
    print(f"  did not solve (levels {obs.levels_completed}, avatar now @({av.x},{av.y}))"); return int(obs.levels_completed or 0)


if __name__ == "__main__":
    solve()
