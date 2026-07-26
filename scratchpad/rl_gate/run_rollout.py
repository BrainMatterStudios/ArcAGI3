"""run_rollout.py — one duck episode, offline, through the capture proxy.

Drives ONE dev game through the REAL duck harness (TAAF GameAPI + the
HarnessSolver session loop + ToolAgent), pointed at `--upstream` via an
in-process capture proxy. Writes a per-episode workdir:

    <workdir>/trace.jsonl     every /chat/completions pair (via capture_proxy)
    <workdir>/manifest.json   {game, levels_completed, n_turns, tokens, wall_s, trace_path}
    <workdir>/artifacts/...   duck runtime state + viewer payloads
    <workdir>/transcripts/... duck transcript

Reward = game.game_run.levels_completed (taaf.game.GameRun bookkeeping —
incremented by Game.execute_action when raw.levels_completed rises).

Usage:
  .venv/bin/python scratchpad/rl_gate/run_rollout.py --game tu93 \
      --upstream http://127.0.0.1:8000/v1 --max-actions 5 --multimodal
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TAAF_INFERENCE = REPO / "submission/_adopt/taaf-src/src/ARC3-Inference"
TAAF_SRC = REPO / "submission/_adopt/taaf-src/src/tufa-arc-agi-framework/src"
ENV_DIR = REPO / "environment_files"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True, help="dev game id, e.g. tu93")
    ap.add_argument("--upstream", default=os.environ.get("LOCAL_ANALYZER_BASE_URL", ""),
                    help="real model endpoint incl. /v1 (proxied)")
    ap.add_argument("--workdir", default="", help="episode dir (default episodes/<game>_<ts>)")
    ap.add_argument("--model-id", default="duck-model", help="model id sent to the endpoint")
    ap.add_argument("--rollout-id", default="")
    ap.add_argument("--max-actions", type=int, default=20)
    ap.add_argument("--max-runtime-s", type=float, default=600.0)
    ap.add_argument("--analyzer-timeout", type=float, default=120.0)
    ap.add_argument("--multimodal", action="store_true",
                    help="set MULTIMODAL_CONTEXT=current_grid so requests carry grid images")
    ap.add_argument("--env-dir", default=str(ENV_DIR),
                    help="environment_files dir (default: repo dev games; point at "
                         "scratchpad/arc_interactive_upstream/environment_files for the "
                         "arc-interactive holdout)")
    args = ap.parse_args()
    if not args.upstream:
        ap.error("--upstream (or LOCAL_ANALYZER_BASE_URL) is required")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    workdir = Path(args.workdir) if args.workdir else HERE / "episodes" / f"{args.game}_{stamp}"
    workdir.mkdir(parents=True, exist_ok=True)
    rollout_id = args.rollout_id or f"{args.game}_{stamp}"
    trace_path = workdir / "trace.jsonl"

    # 1. capture proxy between duck and model
    sys.path.insert(0, str(HERE))
    from capture_proxy import start_proxy
    proxy = start_proxy(args.upstream, trace_path, rollout_id)
    print(f"[rollout] proxy {proxy.base_url} -> {args.upstream}", flush=True)

    # 2. env BEFORE importing the duck (tool_agent reads env at import/init time)
    os.environ["LOCAL_ANALYZER_BASE_URL"] = proxy.base_url
    os.environ["OPENAI_BASE_URL"] = proxy.base_url
    os.environ["LOCAL_ANALYZER_MODEL_ID"] = args.model_id
    if args.multimodal:
        os.environ["MULTIMODAL_CONTEXT"] = "current_grid"

    for p in (str(TAAF_INFERENCE), str(TAAF_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from taaf.game import RunSession
    from taaf.game_api import ArcadeSpec, GameAPI
    from inference.framework import solver as duck_solver

    # 3. real TAAF game, offline arcade
    session_res = RunSession()
    game = GameAPI(env_name=args.game,
                   arcade_spec=ArcadeSpec(environments_dir=args.env_dir))
    game.start_game(session_res)
    print(f"[rollout] game {game.game_id} started: {game.number_of_levels} levels", flush=True)

    # 4. real duck session loop (mirrors HarnessSolver._play_one, minus the
    #    geodesic postpass — rollouts must contain only the policy's own actions)
    solver = duck_solver.HarnessSolver(
        label="rl-gate-rollout",
        model=args.model_id,
        analyzer_timeout=args.analyzer_timeout,
        max_actions_per_game=args.max_actions,
        max_runtime_s_per_game=args.max_runtime_s,
        concurrency=1,
    )
    solver.job_dir = workdir
    analyzer = solver._make_analyzer(game, 0)
    harness_session = duck_solver._HarnessGameSession(
        solver=solver,
        game=game,
        analyzer=analyzer,
        game_index=0,
        pass_index=0,
        state_path=workdir / "artifacts" / "runtime_state.json",
        transcript_path=workdir / "transcripts" / f"{args.game}.txt",
        analysis_html_relpath=f"solver_analysis/{args.game}.html",
        stop_event=threading.Event(),
        viewer_data_path=workdir / "artifacts" / "viewer_data.json",
    )
    t0 = time.time()
    try:
        harness_session.play()  # finishes the game in its finally block
    finally:
        wall_s = time.time() - t0
        proxy.stop()
        try:
            session_res.close()
        except Exception:
            pass

    # 5. manifest
    run = game.game_run
    n_requests = sum(1 for _ in open(trace_path)) if trace_path.exists() else 0
    manifest = {
        "schema": "rl_gate.manifest.v1",
        "rollout_id": rollout_id,
        "game": args.game,
        "game_id": game.game_id,
        "levels_completed": int(run.levels_completed if run else 0),
        "number_of_levels": int(game.number_of_levels),
        "reward": int(run.levels_completed if run else 0),  # episode reward for GRPO
        "state": run.state if run else "unknown",
        "final_score": run.final_score if run else None,
        "n_turns": int(harness_session.analysis_step),
        "n_actions": int(harness_session.action_count),
        "n_model_requests": n_requests,
        "tokens_total": int(getattr(analyzer, "total_tokens", 0)),
        "tokens_generated": int(getattr(analyzer, "generated_tokens", 0)),
        "wall_s": round(wall_s, 1),
        "multimodal": bool(args.multimodal),
        "trace_path": str(trace_path),
        "solver_note": run.solver_note if run else None,
    }
    (workdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[rollout] done: reward={manifest['reward']} turns={manifest['n_turns']} "
          f"actions={manifest['n_actions']} requests={n_requests} wall={manifest['wall_s']}s")
    print(f"[rollout] manifest: {workdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
