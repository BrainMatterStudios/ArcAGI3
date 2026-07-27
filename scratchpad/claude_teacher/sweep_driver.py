"""sweep_driver.py — resumable Claude-teacher sweep over the 25 official dev games.

Sequentially rolls each game through the local claude_shim (subscription-backed)
via the standard rl_gate capture path. A game with a manifest.json in its
workdir is DONE and is skipped, so the driver can be killed and re-run at any
time (usage-limit sleeps happen inside the shim; this driver just waits).

Usage:
    .venv/bin/python scratchpad/claude_teacher/sweep_driver.py \
        [--games cd82,ft09,...] [--upstream http://127.0.0.1:8114/v1] \
        [--max-actions 120] [--max-runtime-s 5400]

Output: scratchpad/claude_corpus/rollouts/claude_<game>/ (trace.jsonl etc.)
The corpus root carries DO_NOT_TRAIN.md — see it before touching the data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "bin" / "python"
ENV_DIR = REPO / "submission/devkit_ds/lib/environment_files"
CORPUS = REPO / "scratchpad/claude_corpus"

# gap-prioritized: games with the LEAST K3 win coverage first (19 of 25 dev
# games have ZERO K3 win samples — sft_data/stats.json, 2026-07-28), so if the
# usage window runs dry mid-sweep the highest-marginal-value games are banked
DEV_GAMES = ["ar25", "bp35", "cn04", "dc22", "g50t", "ka59", "lf52", "ls20",
             "m0r0", "r11l", "re86", "s5i5", "sc25", "sk48", "sp80", "tn36",
             "tr87", "tu93", "wa30", "lp85", "ft09", "cd82", "su15", "vc33", "sb26"]


def shim_alive(upstream: str) -> bool:
    try:
        with urllib.request.urlopen(f"{upstream}/models", timeout=5) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default=",".join(DEV_GAMES))
    ap.add_argument("--upstream", default="http://127.0.0.1:8114/v1")
    ap.add_argument("--max-actions", type=int, default=120)
    ap.add_argument("--max-runtime-s", type=int, default=5400)
    ap.add_argument("--analyzer-timeout", type=int, default=300)
    args = ap.parse_args()

    assert (CORPUS / "DO_NOT_TRAIN.md").exists(), "segregation marker missing"
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    print(f"[sweep] {len(games)} games; upstream {args.upstream}", flush=True)

    done, failed = [], []
    for game in games:
        workdir = CORPUS / "rollouts" / f"claude_{game}"
        if (workdir / "manifest.json").exists():
            print(f"[sweep] {game}: already done, skip", flush=True)
            done.append(game)
            continue
        if not shim_alive(args.upstream):
            print("[sweep] shim not reachable — start claude_shim.py first", flush=True)
            return 2
        if workdir.exists():
            # partial run from a previous kill — recapture from scratch
            subprocess.run(["rm", "-rf", str(workdir)], check=True)
        print(f"[sweep] {game}: starting at {time.strftime('%F %T')}", flush=True)
        cmd = [str(PY), str(REPO / "scratchpad/rl_gate/run_rollout.py"),
               "--game", game, "--upstream", args.upstream,
               "--workdir", str(workdir),
               "--max-actions", str(args.max_actions),
               "--max-runtime-s", str(args.max_runtime_s),
               "--analyzer-timeout", str(args.analyzer_timeout),
               "--multimodal", "--rollout-id", f"claude_teacher_{game}",
               "--env-dir", str(ENV_DIR)]
        env_run = dict(TAAF_ROOT="scratchpad/taaf_scored_ref")
        import os
        proc = subprocess.run(cmd, cwd=REPO, env={**os.environ, **env_run},
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[sweep] {game}: FAILED rc={proc.returncode}\n{proc.stderr[-800:]}",
                  flush=True)
            failed.append(game)
            continue
        manifest = json.loads((workdir / "manifest.json").read_text())
        print(f"[sweep] {game}: done — levels={manifest.get('levels_completed')} "
              f"actions={manifest.get('n_actions')} turns={manifest.get('n_turns')}",
              flush=True)
        done.append(game)

    print(f"[sweep] finished: {len(done)} done, {len(failed)} failed "
          f"({','.join(failed) or 'none'})", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
