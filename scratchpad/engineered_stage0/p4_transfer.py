"""Stage-0 Pillar 4 — TransferExplorer resurrection.

Re-runs the exact historical configuration of the 2026-06-29 goose bakeoff
(memory arcagi3-stochasticgoose-facts): TransferExplorer(DENSE) — DENSE =
trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256 —
seed 0, budget 4000 actions, on the same 8 dev games:
    tu93 lp85 vc33 cd82 ar25 lf52 su15 m0r0
Historical reference: TOTAL L16 across the 8 (goose scored L2). Per-game detail
from docs/superpowers/specs/2026-06-28-stochastic-goose-RESULT.md: tu93 -> L4
(level-up at action ~431).

Parity criterion: total levels >= 13 (>= ~80% of the recorded 16, allowing for
engine-version stochasticity) and tu93 >= 3.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import os
os.environ.setdefault("ARC_API_KEY", "local-dev")
logging.basicConfig(level=logging.ERROR)

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer

DENSE = dict(trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256)
GAMES = ["tu93", "lp85", "vc33", "cd82", "ar25", "lf52", "su15", "m0r0"]
BUDGET = 4000
REFERENCE_TOTAL = 16
REFERENCE_TU93 = 4


def maxlevel(game_stem: str, budget: int = BUDGET):
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir="environment_files",
                    logger=logging.getLogger("s0p4"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game_stem))
    env = client.make(game_id=gid, scorecard_id=f"s0p4-{game_stem}")
    pol = TransferExplorer(seed=0, **DENSE)
    obs = env.reset()
    n, best, first_up = 0, 0, None
    while n < budget:
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame),
                         obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > best and first_up is None:
            first_up = n
        best = max(best, lv)
    return best, first_up, n


def main() -> int:
    out, total = {}, 0
    t0 = time.time()
    print(f"{'game':6} {'levels':>6} {'first_up@':>9} {'actions':>8}")
    for g in GAMES:
        t1 = time.time()
        best, first_up, n = maxlevel(g)
        total += best
        out[g] = {"levels": best, "first_levelup_at": first_up, "actions": n,
                  "elapsed_s": round(time.time() - t1, 1)}
        print(f"{g:6} {best:>6} {str(first_up):>9} {n:>8}   ({out[g]['elapsed_s']}s)", flush=True)
    ok = total >= 13 and out["tu93"]["levels"] >= 3
    print(f"\nTOTAL {total} (reference {REFERENCE_TOTAL}; tu93 ref {REFERENCE_TU93})")
    print(f"P4 VERDICT: {'PASS' if ok else 'FAIL'}  elapsed {time.time()-t0:.0f}s")
    json.dump({"per_game": out, "total": total, "reference_total": REFERENCE_TOTAL},
              open("scratchpad/engineered_stage0/p4_transfer.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
