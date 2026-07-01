"""GENERAL joint-search solver: BFS over the action space (moves 1-4 + ACTION5 + clicks at perceived panel
buttons) via reset+replay with HUD-masked frame-state dedup, until a level-up. Finds the win without needing
to identify avatar/goal/mechanic -- it just searches the reachable state graph for a rewarding sequence. This
is the dc22 technique generalized; it fits state-space puzzles like g50t's clone-recorder (moves + record).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 python .../general_search.py <game> [max_nodes]
"""
from __future__ import annotations
import sys
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P


def _click_targets(grid):
    bg = P.detect_background(grid); out = []
    for o in P.connected_components(grid, background=bg):
        if o.color == bg or o.size < 6:
            continue
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        if 0 <= cy < 64 and 0 <= cx < 64 and grid[cy, cx] == o.color and cx >= 30:
            out.append((cx, cy))
    return sorted(set(out))[:4]


def solve(game, max_nodes=20000, verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"gs-{game}")
    obs = env.reset()
    avail = list(obs.available_actions or [])
    grid0 = P.to_grid(obs.frame)
    actions = [("S", a) for a in avail if a in (1, 2, 3, 4, 5)]
    if 6 in avail:
        actions += [("C", cx, cy) for (cx, cy) in _click_targets(grid0)]
    if verbose:
        print(f"{game}: actions={actions}")

    def apply(tok):
        nonlocal obs
        if tok[0] == "C":
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        else:
            obs = env.step(GameAction.from_id(tok[1]))

    def won():
        return obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1

    def replay(seq):
        nonlocal obs
        obs = env.reset()
        for tok in seq:
            apply(tok)
            if won():
                return True
        return False

    def state():
        g = P.to_grid(obs.frame)
        return hash(g[:56, :56].tobytes())   # HUD-masked structural state

    replay([]); seen = {state()}; q = deque([[]]); sol = None; nodes = 0
    while q and sol is None and nodes < max_nodes:
        seq = q.popleft()
        for a in actions:
            replay(seq); apply(a); nodes += 1
            if won():
                sol = seq + [a]; break
            st = state()
            if st not in seen:
                seen.add(st); q.append(seq + [a])
    if verbose:
        print(f"  search: nodes={nodes} states={len(seen)} -> {'SOLVED len '+str(len(sol)) if sol else 'no plan'}")
    if sol:
        replay(sol)
        print(f"  *** {game} SOLVED by general search ({len(sol)} actions) ***")
        return int(obs.levels_completed or 0)
    return 0


if __name__ == "__main__":
    solve(sys.argv[1] if len(sys.argv) > 1 else "g50t",
          int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
