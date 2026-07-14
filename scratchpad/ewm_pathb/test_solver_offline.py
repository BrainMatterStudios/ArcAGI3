"""Offline validation: TAAF Benchmark + SolverEWM on a dev game, brain=MLX. Validates the FULL
integration plumbing (agent -> server_taaf -> game.execute_action -> TAAF scoring). The 7B won't
solve; success = it RUNS end-to-end and TAAF produces a score without crashing."""
import asyncio, sys
from pathlib import Path

HERE = Path("/Users/ahmed/Documents/ArcAGI3/scratchpad/ewm_pathb")
TAAF = Path("/Users/ahmed/Documents/ArcAGI3/submission/_adopt/taaf-src/src/tufa-arc-agi-framework/src")
for p in (str(TAAF), str(HERE), str(HERE / "src" / "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

import arc_agi
from taaf.benchmark import Benchmark
from taaf.game_api import GameAPI, ArcadeSpec
from ewm_solver import SolverEWM

ENV = "/Users/ahmed/Documents/ArcAGI3/environment_files"
spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV)
game = GameAPI(env_name="tu93", arcade_spec=spec)
solver = SolverEWM(
    base_url="http://127.0.0.1:8081/v1",
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    per_game_seconds=120.0, max_turns=12, max_ctx_chars=20000, max_tokens=1024,
    work_root=str(HERE / "_solver_test"), port_base=8930,
)
bm = Benchmark(games=[game], solver=solver, n_passes=1)

async def main():
    await bm.run()

print("running SolverEWM(MLX-7B) on tu93 offline via TAAF (~2min box)...", flush=True)
asyncio.run(main())
gr = bm.game_runs[0]
print("RESULT: state=", gr.state, "| final_score=", gr.final_score,
      "| levels_completed=", gr.levels_completed,
      "| actions_per_level=", list(gr.actions_per_level))
print("INTEGRATION OK: TAAF scored a SolverEWM run end-to-end." if gr.final_score is not None
      else "INTEGRATION INCOMPLETE: no final_score")
