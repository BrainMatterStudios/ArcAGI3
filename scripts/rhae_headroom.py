"""RHAE HEADROOM ORACLE (asset A3, never built until now). Per dev game: drive banked SalienceExplorer,
capture the learned state graph + the temporally-ordered reward transitions, and compare ACTUAL
actions-to-each-level against the GEODESIC (BFS-shortest path in the learned graph from the level's start
node to the reward-source node). headroom = actual/geodesic; headroom^2 = the RHAE efficiency multiplier a
perfect geodesic-replay agent could recover on that level. Aggregate median headroom answers the strategic
question: does within-game efficiency have REAL upside (big headroom) or is the explorer already near-optimal
(headroom ~1, => efficiency levers are inert, explaining Kaggle-inert transfer/CAI)?

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/rhae_headroom.py [budget] [game,game,...]
"""
from __future__ import annotations
import logging, sys
from collections import deque
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
logging.basicConfig(level=logging.ERROR)

# dev games that complete >=1 level (from the campaign baseline); skip pure 0-level walls.
DEFAULT = ["vc33","cd82","tu93","lp85","lf52","ar25","sp80","su15","m0r0","tr87","ls20"]


def geodesic(nodes, start, goal):
    """BFS hop-distance over the learned graph edges (each hop = 1 action)."""
    if start == goal:
        return 0
    if start not in nodes:
        return None
    seen = {start}; q = deque([(start, 0)])
    while q:
        k, d = q.popleft()
        node = nodes.get(k)
        if not node:
            continue
        for a, (nk, _r) in node.edges.items():
            if nk == goal:
                return d + 1
            if nk not in seen:
                seen.add(nk); q.append((nk, d + 1))
    return None  # goal unreachable in the learned graph


def run(game, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("rhae"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"rhae-{game}")
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset(); n = 0; prev_levels = 0
    marks = []  # (action_count, source_key, source_action)
    while n < budget:
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame), obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        src_key, src_action = pol.prev_key, pol.prev_action   # set inside decide for this step
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > prev_levels and src_key is not None and src_action is not None:
            marks.append((n, src_key, src_action))
            prev_levels = lv
    # compute geodesic vs actual per completed level
    rows = []
    start = pol.root_key
    prev_n = 0
    for i, (an, src, act) in enumerate(marks):
        g = geodesic(pol.nodes, start, src)
        actual = an - prev_n
        rows.append((i + 1, actual, g))
        # next level starts at the reward edge's target
        nxt = pol.nodes.get(src)
        start = nxt.edges.get(act, (start, 0))[0] if nxt else start
        prev_n = an
    return rows, len(pol.nodes)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT
    print(f"RHAE headroom oracle (banked salience, budget {budget}). headroom = actual/geodesic; "
          f"headroom^2 = recoverable efficiency.\n")
    print(f"{'game':>6} {'lvl':>3} {'actual':>8} {'geodesic':>9} {'headroom':>9}")
    all_ratios = []
    for g in games:
        rows, nnodes = run(g, budget)
        if not rows:
            print(f"{g:>6}  (0 levels completed @ budget)")
            continue
        for (lv, actual, geo) in rows:
            ratio = (actual / geo) if geo else None
            if ratio:
                all_ratios.append(ratio)
            rs = f"{ratio:8.1f}x" if ratio else "   (unreach)"
            print(f"{g:>6} {lv:>3} {actual:>8} {str(geo):>9} {rs:>9}")
    if all_ratios:
        med = float(np.median(all_ratios)); mx = max(all_ratios)
        print(f"\n== median headroom = {med:.1f}x  (median recoverable efficiency ~ {med**2:.0f}x);  "
              f"max = {mx:.1f}x;  n_levels = {len(all_ratios)}")
        print("Interpretation: headroom ~1-2x => explorer near-optimal, efficiency levers INERT (explains "
              "Kaggle-inert transfer/CAI). headroom >>1 => real efficiency upside (geodesic-replay worth building).")


if __name__ == "__main__":
    main()
