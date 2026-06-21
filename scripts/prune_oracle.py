# scripts/prune_oracle.py
"""Phase 0a prune-capability oracle probe (live driver).

Runs banked v6 (SalienceExplorer trust=3, border_mask=2) on real public games, captures a
per-step trace, reconstructs the level graph, and reports per completed level:
  Tier-1 prune ceiling (states + actions) and Tier-2 leave-one-game-out AUC vs shuffle.

Usage:
  ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/prune_oracle.py \
      [budget] [highbudget] [game_prefixes...]
Defaults: budget=40000 highbudget=150000 games=tu93,vc33,m0r0,ls20,lp85,cd82
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3 import prune_analysis as A  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("oracle"))


def run_and_capture(prefix, budget):
    """Run v6 on one real game; return (steps, grid_by_key, colors_by_key, best_level, gid, bg)."""
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    card = client.open_scorecard(tags=["prune-oracle"])
    env = client.make(game_id=gid, scorecard_id=card)
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset()
    steps, grid_by_key, colors_by_key = [], {}, {}
    n, best = 0, 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        lv_before = int(obs.levels_completed or 0)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         lv_before, list(obs.available_actions or []))
        fk = pol.prev_key  # state acted from (set inside decide)
        tier = 0
        if fk is not None and fk in pol.nodes and tok != A.RESET:
            tier = int(pol.nodes[fk].tier.get(tok, 0))
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        lv_after = int(obs.levels_completed or 0)
        if fk is not None:
            steps.append(A.Step(n, lv_before, fk, tok, float(lv_after - lv_before), tier))
            if fk not in grid_by_key:
                grid_by_key[fk] = grid
                colors_by_key[fk] = {int(c) for c in np.unique(grid)} - {pol.bg or 0}
        best = max(best, lv_after)
        n += 1
    return steps, grid_by_key, colors_by_key, best, gid, (pol.bg or 0)


def discovery_meta(steps, colors_by_key):
    """discovery_tier[to_key] and parent_colors[to_key] from the discovering edge."""
    disc_tier, parent_colors = {}, {}
    for i in range(len(steps) - 1):
        s = steps[i]
        if s.action == A.RESET or s.reward > 0:
            continue
        to_key = steps[i + 1].from_key
        if to_key not in disc_tier:
            disc_tier[to_key] = s.tier
            parent_colors[to_key] = colors_by_key.get(s.from_key, set())
    return disc_tier, parent_colors


def analyze_game(prefix, budget, highbudget):
    steps, grids, colors, best, gid, bg = run_and_capture(prefix, budget)
    edges, first_seen = A.build_edges(steps)
    segs = A.segment_levels(steps, first_seen)
    # If the deepest target level wasn't completed, retry once at high budget.
    if not segs and highbudget > budget:
        print(f"  {gid}: no completed level @ {budget}; retry @ {highbudget}", flush=True)
        steps, grids, colors, best, gid, bg = run_and_capture(prefix, highbudget)
        edges, first_seen = A.build_edges(steps)
        segs = A.segment_levels(steps, first_seen)
    disc_tier, parent_colors = discovery_meta(steps, colors)

    per_level, game_X, game_y = {}, [], []
    for lvl, seg in sorted(segs.items()):
        # Phase 0a': allow reset-roots as entries (the first frame can be a disconnected
        # intro state) and take the shortest path from any entry.
        entries = A.level_entries(steps, seg)
        path, entry = A.best_path(edges, entries, seg.target_key, seg.member_keys)
        cl = A.ceilings(seg, path)
        per_level[lvl] = cl
        if path is None:
            continue
        # Fairer label: union of near-optimal paths (slack=2), not the single shortest path.
        on = A.near_optimal_states(edges, entry, seg.target_key, seg.member_keys, slack=2)
        for k in seg.member_keys:
            if k not in grids:
                continue
            f = A.state_features(grids[k], background=bg,
                                 discovery_tier=disc_tier.get(k, 0),
                                 parent_colors=parent_colors.get(k))
            game_X.append(A.feature_vector(f))
            game_y.append(1 if k in on else 0)
    return {"gid": gid, "best_level": best, "per_level": per_level,
            "X": np.array(game_X) if game_X else None,
            "y": np.array(game_y) if game_y else None}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    highbudget = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
    games = sys.argv[3:] if len(sys.argv) > 3 else ["tu93", "vc33", "m0r0", "ls20", "lp85", "cd82"]
    t0 = time.time()
    results, per_game = {}, {}
    for g in games:
        try:
            r = analyze_game(g, budget, highbudget)
        except StopIteration:
            print(f"  {g}: NOT IN DEV SET", flush=True)
            continue
        results[g] = {"gid": r["gid"], "best_level": r["best_level"], "per_level": r["per_level"]}
        for lvl, cl in r["per_level"].items():
            print(f"  [{g} L{lvl}] ceiling_states={cl['ceiling_states']} "
                  f"ceiling_actions={cl['ceiling_actions']} disc={cl['discovered_states']} "
                  f"acts={cl['actual_actions']} reachable={cl['reachable']}", flush=True)
        if r["X"] is not None and r["y"] is not None and r["y"].sum() > 0:
            per_game[g] = (r["X"], r["y"])
    enough = len(per_game) >= 2
    auc = A.logo_auc(per_game) if enough else float("nan")
    sh = A.shuffle_auc(per_game) if enough else float("nan")
    auc_mlp = A.logo_auc_mlp(per_game) if enough else float("nan")
    per_lin = A.logo_auc_per_game(per_game) if enough else {}
    per_pos = {g: int(y.sum()) for g, (_X, y) in per_game.items()}
    print(f"\n== Tier-2 leave-one-game-out: logistic AUC={auc:.3f}  MLP AUC={auc_mlp:.3f}  "
          f"shuffle={sh:.3f}  (games={list(per_game)})", flush=True)
    print(f"== per-held-game logistic AUC: "
          + ", ".join(f"{g}={v:.3f}(+{per_pos[g]})" for g, v in per_lin.items()), flush=True)
    best_auc = max([a for a in (auc, auc_mlp) if a == a], default=float("nan"))
    verdict = ("BUILD" if (best_auc >= 0.65 and best_auc >= sh + 0.10)
               else "KILL/AMBER")
    print(f"== VERDICT: best AUC={best_auc:.3f} vs bar 0.65 & shuffle+0.10={sh+0.10:.3f} "
          f"=> {verdict}", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)
    out = "/tmp/prune_oracle.json"
    with open(out, "w") as f:
        json.dump({"auc": auc, "auc_mlp": auc_mlp, "shuffle": sh,
                   "per_game_auc": per_lin, "per_game_pos": per_pos,
                   "results": results}, f, indent=2, default=str)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
