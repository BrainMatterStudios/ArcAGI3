"""Offline end-to-end validation of the CONCURRENT shadow through a real TAAF Benchmark (no GPU).
A scripted solver wins tu93 L1 WASTEFULLY (replays the recorded 171-action duck history); the shadow
must open a fresh play on the same card, replay the 18-action winning attempt, and max-over-plays
must then bank the efficient score. Proves the whole run_with_shadow plumbing minus the live gateway."""
import asyncio, json, sys
from dataclasses import dataclass, field
from pathlib import Path

TAAF = "/Users/ahmed/Documents/ArcAGI3/submission/_adopt/taaf-src/src/tufa-arc-agi-framework/src"
HERE = "/Users/ahmed/Documents/ArcAGI3/scratchpad/ewm_pathb"
for p in (TAAF, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import arc_agi, arcengine
from taaf.benchmark import Benchmark
from taaf.game_api import GameAPI, ArcadeSpec
from taaf.solver import Solver
from shadow_run import run_with_shadow

ENV = "/Users/ahmed/Documents/ArcAGI3/environment_files"
BJ = "/Users/ahmed/Documents/ArcAGI3/submission/_adopt/taaf-src/src/ARC3-Inference/runs/20260703_090517/benchmark.json"
run = next(r for r in json.load(open(BJ))["game_runs"] if r["game_id"].startswith("tu93") and r["levels_completed"] >= 1)
HISTORY = [{"id": h["action"]["id"], "data": dict(h["action"].get("data") or {})} for h in run["history"]]
print(f"scripted solver will replay {len(HISTORY)} recorded actions (wins tu93 L1 wastefully)")


@dataclass
class SolverReplayHistory(Solver):
    """Wins each game exactly as the duck did — wastefully — by replaying a recorded history."""
    label: str = "ReplayHistory"
    history: list = field(default_factory=list)

    async def _run_games(self, games):
        for game in games:
            if game.game_run is None or game.game_run.state != "playing":
                continue
            for a in self.history:
                if game.game_run.state != "playing":
                    break
                try:
                    game.execute_action(arcengine.ActionInput(id=arcengine.GameAction[a["id"]], data=dict(a["data"])))
                except Exception:
                    break
                await asyncio.sleep(0)
            if game.game_run is not None and game.game_run.state == "playing":
                game.finish_game()


spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV)
bm = Benchmark(games=[GameAPI(env_name="tu93", arcade_spec=spec)],
               solver=SolverReplayHistory(history=HISTORY), n_passes=1)

async def main():
    await run_with_shadow(bm)

print("running scripted duck + concurrent shadow (OFFLINE)...", flush=True)
asyncio.run(main())

# group game_runs by game_id -> max-over-plays
by_game = {}
for gr in bm.game_runs:
    by_game.setdefault(gr.game_id, []).append(gr)
print("\n=== plays per game (game_id -> [(state, levels, final_score)]) ===")
for gid, runs in by_game.items():
    print(f"  {gid}: " + " | ".join(f"{r.state} lvls={r.levels_completed} score={r.final_score}" for r in runs))
    scores = [r.final_score or 0.0 for r in runs]
    banked = max(scores)
    pass0 = runs[0].final_score or 0.0
    print(f"  -> plays={len(runs)}  pass0_score={pass0:.2f}  MAX-OVER-PLAYS(banked)={banked:.2f}")
    ok = len(runs) >= 2 and banked > pass0 + 0.01
    print("  CONCURRENT SHADOW VALIDATED: shadow play banked a higher score via max-over-plays"
          if ok else "  FAILED: shadow did not add a higher-scoring play")
