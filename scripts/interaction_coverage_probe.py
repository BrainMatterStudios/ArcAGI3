"""Gate 0: does object-curiosity reach MORE distinct interaction signatures than baseline?"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer
from arcagi3.object_interaction import NONE_SIGNATURE, interaction_signature
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game

logging.basicConfig(level=logging.ERROR)


def compare_rows(rows):
    return {
        "more_coverage": sum(1 for _g, base, cur in rows if cur > base),
        "worse_coverage": sum(1 for _g, base, cur in rows if cur < base),
    }


def coverage(game_id: str, budget: int, enable_curiosity: bool, seed: int = 0) -> dict:
    eng = ObjectCuriosityExplorer(seed=seed, enable_object_curiosity=enable_curiosity)
    eng.reset_all()
    env = make_game(game_id)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    distinct = set()
    prev_grid = None
    prev_tok = None
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        # the transition prev_grid -> grid was caused by prev_tok (taken last iteration)
        if prev_grid is not None and prev_tok is not None and prev_tok[0] in ("S", "C"):
            sig = interaction_signature(prev_grid, prev_tok, grid, eng.bg)
            if sig != NONE_SIGNATURE:
                distinct.add(sig)
        tok = eng.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level, available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset(); prev_grid = None; prev_tok = None
        elif tok[0] == "S":
            prev_grid = grid; prev_tok = tok
            obs = env.step(GameAction.from_id(tok[1])); actions += 1
        else:
            prev_grid = grid; prev_tok = tok
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]}); actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    return {"game": game_id, "levels": level, "actions": actions,
            "distinct_signatures": len(distinct)}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["ls20", "re86"]
    rows = []
    for game in games:
        base = coverage(game, budget, enable_curiosity=False)
        cur = coverage(game, budget, enable_curiosity=True)
        rows.append((game, base["distinct_signatures"], cur["distinct_signatures"]))
        print(f"[{game:8}] base_distinct={base['distinct_signatures']} "
              f"curiosity_distinct={cur['distinct_signatures']} "
              f"(levels {base['levels']}->{cur['levels']})", flush=True)
    summary = compare_rows(rows)
    print("-" * 80, flush=True)
    print(f"more_coverage={summary['more_coverage']} worse_coverage={summary['worse_coverage']}",
          flush=True)
    print("GATE 0: " + ("PASS" if summary["more_coverage"] >= 1 else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
