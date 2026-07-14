"""ewm_solver.py — a TAAF Solver that plays each game with the verification-by-execution loop.

Per game (already start_game()'d by Benchmark): stand up an in-process server wrapping the TAAF
game (server_taaf.serve_game), lay down a fresh EWM workspace, and run ewm_agent.py (driven by a
local vLLM/MLX model) against it with a per-game time-box. Every action the agent takes flows
through game.execute_action -> TAAF scores it. Respects soft_end_time + cancellation.

Plug into the duck's benchmark by setting `bm.solver = SolverEWM(base_url=..., model=..., ...)`.
"""
from __future__ import annotations
import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import taaf.game
from taaf.solver import Solver

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"


def _finish_remaining(games) -> None:
    for g in games:
        if g.game_run is not None and g.game_run.final_score is None:
            try:
                g.finish_game()
            except Exception:
                pass


@dataclass
class SolverEWM(Solver):
    label: str = "SolverEWM"
    base_url: str = "http://127.0.0.1:1234/v1"   # vLLM (Qwen) or MLX endpoint
    model: str = "qwen"
    per_game_seconds: float = 600.0              # per-game time-box
    max_turns: int = 160
    max_ctx_chars: int = 110_000
    obs_cap: int = 1800
    max_tokens: int = 4096
    port_base: int = 8890
    work_root: Path | None = field(default=None)
    python: str = field(default_factory=lambda: sys.executable)

    def _setup(self) -> None:
        self._work = Path(self.work_root) if self.work_root else (_HERE / "runs_solver")
        self._work.mkdir(parents=True, exist_ok=True)

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        try:
            # sequential: one game at a time shares the single GPU/model + a distinct port
            for i, game in enumerate(games):
                if self.soft_end_time is not None and datetime.now(timezone.utc) >= self.soft_end_time:
                    if game.game_run is not None and game.game_run.state == "playing":
                        game.finish_game()
                    continue
                await self._play_one(game, self.port_base + (i % 20))
        except asyncio.CancelledError:
            _finish_remaining(games)
            raise

    async def _play_one(self, game: taaf.game.Game, port: int) -> None:
        run = game.game_run
        if run is None or run.state != "playing":
            return
        from server_taaf import serve_game  # lazy: caller sets the taaf-src import path
        ws = self._work / f"game_{game.game_id or port}_{int(port)}"
        if ws.exists():
            shutil.rmtree(ws)
        shutil.copytree(_SRC / "agent" / "workspace_init", ws)
        for name, tgt in [("g", "client/client.py"), ("verify", "verify_world_model.py"),
                          ("verifyplan", "verify_main_planner.py"), ("plan", "run_main_planner.py")]:
            (ws / name).write_text(
                f'#!/bin/bash\nexport GAME_SERVER_URL="http://127.0.0.1:{port}"\n'
                f'exec "{self.python}" "$(dirname "$0")/{tgt}" "$@"\n')
            os.chmod(ws / name, 0o755)

        srv = serve_game(game, port=port)
        tokens = 0
        proc = None
        try:
            env = {**os.environ, "GAME_SERVER_URL": f"http://127.0.0.1:{port}"}
            await asyncio.to_thread(subprocess.run,
                [self.python, str(ws / "client/client.py"), "start", "target"],
                cwd=str(ws), env=env, check=True, capture_output=True)
            proc = await asyncio.create_subprocess_exec(
                self.python, str(_HERE / "ewm_agent.py"),
                "--workspace", str(ws), "--base-url", self.base_url, "--model", self.model,
                "--port", str(port), "--max-turns", str(self.max_turns),
                "--max-ctx-chars", str(self.max_ctx_chars), "--obs-cap", str(self.obs_cap),
                "--max-tokens", str(self.max_tokens),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.per_game_seconds)
                for line in reversed((out or b"").decode(errors="ignore").splitlines()):
                    if line.strip().startswith("{"):
                        try:
                            tokens = int(json.loads(line).get("completion_tokens", 0))
                            break
                        except Exception:
                            pass
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 10)
                except Exception:
                    proc.kill()
        except asyncio.CancelledError:
            if proc is not None:
                proc.kill()
            raise
        finally:
            srv.stop()
            if game.game_run is not None and game.game_run.state == "playing":
                try:
                    game.finish_game(generated_tokens=tokens)
                except Exception:
                    pass
