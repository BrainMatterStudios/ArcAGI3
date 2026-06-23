"""Exp 51 — Pipeline-phase diagnostic across the HOLDOUT set: where does each game die?

The solvability audit (Exp 50) proved search/planning are not the wall; the residual is the
perception front-end + induction ontology. This generalizes the ls20 A/C probe (Exp 50b) across
the whole held-out set to MEASURE the decomposition: for each game, run the current
DiscoveryExplorer with phase/plan instrumentation and record the death point —

  PROBE_MOVEMENT-dominant, no deltas   -> FRONT-END: no directional avatar (wrong family)
  reaches INDUCE/PLAN, no plan         -> ONTOLOGY: avatar found but no candidate/plan (DSL gap)
  reaches EXECUTE, 0 levels            -> TERMINAL: plans+executes but goal mis-modelled (e.g. ls20 moving goal)
  clears levels                        -> on-family (collect)

The split FRONT-END vs ONTOLOGY/TERMINAL tells us whether progress needs a family-detection
front-end (breadth) or a richer induction ontology (depth) — and, run frozen across games,
whether one fix would generalize or each game needs bespoke work (the reusability question).

Source is grader-only (A_h via bakeoff). Live API (NORMAL); modest per-game budget.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/pipeline_diagnostic.py [budget] [games...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.discovery_bakeoff import make_game
from arcagi3 import perception as P
from arcagi3.discovery_explorer import DiscoveryExplorer
from arcengine import GameAction, GameState

HOLDOUT = ["collect", "ls20", "su15", "m0r0", "tr87", "re86", "sk48", "wa30", "tn36"]


def classify(levels, avatar_found, reached_execute, plans_built):
    if levels > 0:
        return "CLEARS (on-family)"
    if not avatar_found:
        return "FRONT-END (no directional avatar)"
    if not reached_execute:
        return "ONTOLOGY (avatar ok; no candidate/plan)"
    return "TERMINAL (plans+executes; goal mis-modelled)"


def run_game(game: str, budget: int):
    eng = DiscoveryExplorer(seed=0, planner_backend="painted_set")
    eng.reset_all()
    try:
        env = make_game(game)
        obs = env.reset()
    except Exception as e:  # noqa: BLE001
        return {"game": game, "error": str(e)[:80]}
    cur_level = int(obs.levels_completed or 0)
    n = 0
    phase_counts: dict[str, int] = {}
    goals: set = set()
    reached_execute = False
    while n < budget and cur_level < 2:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        try:
            token = eng.decide(
                grid=grid,
                gstate_terminal=(obs.state == GameState.GAME_OVER),
                gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                levels=cur_level,
                available=list(obs.available_actions or []),
            )
        except Exception as e:  # noqa: BLE001
            return {"game": game, "error": f"decide: {str(e)[:70]}", "n": n}
        ph = getattr(eng, "_phase", "?")
        phase_counts[ph] = phase_counts.get(ph, 0) + 1
        if ph == "EXECUTE" and getattr(eng, "_plan", None):
            reached_execute = True
            if getattr(eng, "_goal_pos", None):
                goals.add(eng._goal_pos)
        try:
            if token[0] == "reset":
                obs = env.reset()
            elif token[0] == "S":
                obs = env.step(GameAction.from_id(token[1])); n += 1
            else:
                obs = env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]}); n += 1
        except Exception as e:  # noqa: BLE001
            return {"game": game, "error": f"step: {str(e)[:70]}", "n": n}
        if obs is None:
            # local wrapper swallows some game-side errors (e.g. a click game rejecting a
            # coordinate-less ACTION6) and returns None — the loop emitted an action the game's
            # family doesn't accept. Treat as a front-end / family mismatch and stop.
            avatar_found = bool(getattr(eng, "_deltas", {}))
            return {"game": game, "levels": cur_level, "actions": n, "avatar": avatar_found,
                    "n_deltas": len(getattr(eng, "_deltas", {})),
                    "paint": sorted(getattr(eng, "_paint_colors", set())),
                    "reached_execute": reached_execute, "goals_tried": len(goals),
                    "death": "FRONT-END (env rejected action — click/non-avatar family)",
                    "phases": phase_counts}
        new_level = int(obs.levels_completed or 0)
        if new_level > cur_level:
            cur_level = new_level

    avatar_found = bool(getattr(eng, "_deltas", {}))
    return {
        "game": game, "levels": cur_level, "actions": n,
        "avatar": avatar_found, "n_deltas": len(getattr(eng, "_deltas", {})),
        "paint": sorted(getattr(eng, "_paint_colors", set())),
        "reached_execute": reached_execute, "goals_tried": len(goals),
        "death": classify(cur_level, avatar_found, reached_execute, len(goals)),
        "phases": phase_counts,
    }


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    games = sys.argv[2:] if len(sys.argv) > 2 else HOLDOUT
    print(f"=== Exp 51 pipeline diagnostic — budget={budget}/game ===\n", flush=True)
    rows = []
    for g in games:
        print(f"[{g}] running...", flush=True)
        r = run_game(g, budget)
        rows.append(r)
        if "error" in r:
            print(f"   ERROR: {r['error']}", flush=True)
        else:
            print(f"   levels={r['levels']} avatar={r['avatar']}(d={r['n_deltas']}) "
                  f"paint={r['paint']} execute={r['reached_execute']} goals={r['goals_tried']} "
                  f"-> {r['death']}", flush=True)

    print("\n" + "-" * 96)
    print(f"{'game':9}{'levels':7}{'avatar':8}{'execute':9}{'goals':7}death-point")
    print("-" * 96)
    for r in rows:
        if "error" in r:
            print(f"{r['game']:9}{'ERR':7}{'-':8}{'-':9}{'-':7}{r['error']}")
        else:
            print(f"{r['game']:9}{r['levels']:<7}{str(r['avatar']):8}"
                  f"{str(r['reached_execute']):9}{r['goals_tried']:<7}{r['death']}")
    # decomposition summary
    ok = [r for r in rows if "error" not in r]
    front = sum(1 for r in ok if r["death"].startswith("FRONT-END"))
    onto = sum(1 for r in ok if r["death"].startswith("ONTOLOGY"))
    term = sum(1 for r in ok if r["death"].startswith("TERMINAL"))
    clears = sum(1 for r in ok if r["death"].startswith("CLEARS"))
    print("-" * 96)
    print(f"decomposition: FRONT-END={front}  ONTOLOGY={onto}  TERMINAL={term}  CLEARS={clears}  "
          f"(of {len(ok)} run, {len(rows)-len(ok)} errored)")


if __name__ == "__main__":
    main()
