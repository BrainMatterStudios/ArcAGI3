#!/usr/bin/env python3
"""ab_driver.py — paired offline A/B over the pinned holdout (arms x games x rollouts).

The August measurement instrument (protocol: docs/A1-PROTOCOL-2026-08.md).
Each (game, rollout) pair runs EVERY arm back-to-back against the same vLLM
endpoint, so per-game deltas are paired. Prompt-side arms differ only in env
toggles; checkpoint arms differ by SERVER (run this driver once per served
model). Every run uses the SCORED dataset tree (TAAF_ROOT) — never the
drifted local one — and the ledger/doctrine patches applied-but-toggled, so
the shipped combination is exactly what was measured.

Arms syntax:  name=ENV1:v,ENV2:v  (no '=' -> all toggles off = base arm)

Example (doctrine A/B, D1+D2 vs base, 2 paired rollouts):
  .venv/bin/python scratchpad/rl_gate/ab_driver.py \
      --games ff01,sy01,sq01,cs01,cx01,dm01,lo01,sc01,ic01,fs03,sk01,bd01,tp02 \
      --arms "base" "doctrine=DOCTRINE_FIELDGUIDE:1,DOCTRINE_PLAYBOOK:1" \
      --upstream http://127.0.0.1:1234/v1 --rollouts 2 --tag aug_w1
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PY = str(REPO / ".venv/bin/python")
SCORED_REF = "scratchpad/taaf_scored_ref"
HARD_TIMEOUT_S = 2700  # 45 min/game box hard kill (audit: bound by wall-clock)


def parse_arm(spec: str) -> tuple[str, dict]:
    if "=" not in spec:
        return spec, {}
    name, envs = spec.split("=", 1)
    env = {}
    for pair in envs.split(","):
        k, _, v = pair.partition(":")
        env[k.strip()] = v.strip() or "1"
    return name, env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--tag", default="ab")
    ap.add_argument("--env-dir", default="scratchpad/holdout_arcint/environment_files")
    ap.add_argument("--max-runtime-s", type=float, default=2400.0)
    ap.add_argument("--max-actions", type=int, default=400)
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    arms = [parse_arm(s) for s in args.arms]
    results: dict = {}

    for r in range(args.rollouts):
        for g in games:
            for arm_name, arm_env in arms:
                wd = HERE / "episodes" / f"ab_{args.tag}_{arm_name}_{g}_r{r}"
                env = dict(os.environ,
                           TAAF_ROOT=SCORED_REF,
                           APPLY_DUCK_FIXES="1",
                           APPLY_LEDGER_PATCH="1",
                           APPLY_DOCTRINE_PATCH="1",
                           **arm_env)
                cmd = [PY, str(HERE / "run_rollout.py"), "--game", g,
                       "--upstream", args.upstream, "--workdir", str(wd),
                       "--max-actions", str(args.max_actions),
                       "--max-runtime-s", str(args.max_runtime_s),
                       "--multimodal", "--env-dir", args.env_dir,
                       "--rollout-id", f"{args.tag}-{arm_name}-{g}-r{r}"]
                print(f"=== [{time.strftime('%H:%M:%S')}] {arm_name} / {g} / r{r} ===",
                      flush=True)
                p = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                     start_new_session=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
                try:
                    p.communicate(timeout=HARD_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    print(f"  HARD TIMEOUT — killing group", flush=True)
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    p.wait()
                manifest = wd / "manifest.json"
                levels = None
                if manifest.exists():
                    m = json.loads(manifest.read_text())
                    levels = m.get("levels_completed")
                results.setdefault(g, {}).setdefault(arm_name, []).append(levels)
                print(f"  -> levels_completed={levels}", flush=True)

    # paired per-game summary vs the FIRST arm
    base_name = arms[0][0]
    summary = {"tag": args.tag, "base_arm": base_name, "games": {}}
    for g, per_arm in results.items():
        base_runs = [v for v in per_arm.get(base_name, []) if v is not None]
        row = {"base_levels": base_runs}
        for arm_name, _ in arms[1:]:
            runs = [v for v in per_arm.get(arm_name, []) if v is not None]
            row[arm_name] = {"levels": runs,
                             "paired_delta_sum": (sum(runs) - sum(base_runs))
                             if runs and base_runs else None}
        summary["games"][g] = row
    out = HERE / "episodes" / f"ab_{args.tag}_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[ab] summary -> {out}")
    for g, row in summary["games"].items():
        print(f"[ab] {g}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
