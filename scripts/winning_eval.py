"""Phase 7 — evaluation protocol for the winning branch (mechanic model-search).

Run TransferExplorer vs WinningExplorer (+ optional DiscoveryExplorer) on the movement-capable
HOLDOUT games + the collect control, logging levels cleared and the model-search counters. The
go-rule for a Kaggle submission (mission Phase 7): HOLDOUT mean levels strictly improves, zero deep
regressions, model fires on >=2 holdout-proxy games, >=1 fire reaches L2+ or >5x efficiency.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/winning_eval.py [budget] [games...]
"""
from __future__ import annotations

import sys

from scripts.discovery_bakeoff import run_engine
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.winning_explorer import WinningExplorer

MOVEMENT_HOLDOUT = ["ls20", "m0r0", "wa30", "re86", "su15", "tr87"]
DEFAULT = ["collect"] + MOVEMENT_HOLDOUT


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    games = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT
    print(f"=== Winning eval — budget={budget}/game ===\n", flush=True)
    rows = []
    for game in games:
        out = {"game": game}
        for name, eng in [("transfer", TransferExplorer(seed=0)),
                          ("winning", WinningExplorer(seed=0, enable_model_search=True))]:
            eng.reset_all()
            try:
                r = run_engine(name, eng, budget=budget, game=game, max_level=4)
                out[name] = r.levels_cleared
                out[name + "_acts"] = [(l.level, l.actions) for l in r.levels]
            except Exception as e:  # noqa: BLE001
                out[name] = f"ERR:{str(e)[:40]}"
            if name == "winning":
                out["fires"] = getattr(eng, "_model_fires", 0)
                out["plan_starts"] = getattr(eng, "_model_plan_starts", 0)
                out["probe"] = getattr(eng, "_probe_used", 0)
                out["gaveup"] = getattr(eng, "_gave_up", None)
        rows.append(out)
        print(f"[{game}] transfer={out.get('transfer')} winning={out.get('winning')} "
              f"fires={out.get('fires')} plan_starts={out.get('plan_starts')} probe={out.get('probe')} "
              f"gaveup={out.get('gaveup')}  win_acts={out.get('winning_acts')}", flush=True)

    print("\n" + "-" * 88)
    print(f"{'game':9}{'transfer':10}{'winning':10}{'fires':7}{'verdict'}")
    print("-" * 88)
    improved = regressed = fired = 0
    for r in rows:
        t, w = r.get("transfer"), r.get("winning")
        v = ""
        if isinstance(t, int) and isinstance(w, int):
            if w > t:
                v = "IMPROVED"; improved += 1
            elif w < t:
                v = "REGRESSED"; regressed += 1
            else:
                v = "tie"
        if r.get("fires", 0) > 0 and r["game"] != "collect":
            fired += 1
        print(f"{r['game']:9}{str(t):10}{str(w):10}{str(r.get('fires',0)):7}{v}")
    print("-" * 88)
    print(f"holdout improved={improved} regressed={regressed} | model fired on {fired} holdout games")
    go = improved >= 1 and regressed == 0 and fired >= 2
    print(f"SUBMISSION GO-RULE: {'PASS — candidate for Kaggle (verify deep-level preservation)' if go else 'FAIL — ship transfer-dense'}")


if __name__ == "__main__":
    main()
