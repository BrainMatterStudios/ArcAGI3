#!/usr/bin/env python3
"""Hardened K3 sweep driver: each rollout runs in its own subprocess with a HARD
timeout (the 2026-07-23 stall: both workers blocked forever inside a hung upstream
HTTP call — the proxy forwards with no timeout, and the in-run wall check only fires
between turns). On expiry the whole process group is killed and the sweep moves on."""
import os, signal, subprocess, sys, time
GAMES = sys.argv[1].split(",")
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"
HARD_TIMEOUT = 4500  # box 3600s + 15 min grace
PY = "/Users/ahmed/Documents/ArcAGI3/.venv/bin/python"
ROOT = "/Users/ahmed/Documents/ArcAGI3"
for g in GAMES:
    wd = f"{ROOT}/scratchpad/rl_gate/episodes/k3_sweep_{g}_{TAG}"
    print(f"=== [{time.strftime('%H:%M', time.gmtime())}] K3 on {g} ===", flush=True)
    p = subprocess.Popen(
        [PY, f"{ROOT}/scratchpad/rl_gate/run_rollout.py", "--game", g,
         "--upstream", "https://openrouter.ai/api/v1",
         "--model-id", "moonshotai/kimi-k3", "--workdir", wd,
         "--max-actions", "300", "--max-runtime-s", "3600",
         "--multimodal", "--rollout-id", f"k3-sw-{g}-{TAG}"],
        cwd=ROOT, start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=HARD_TIMEOUT)
        for line in out.splitlines():
            if "finished" in line or "done:" in line:
                print("  " + line.strip()[:160], flush=True)
    except subprocess.TimeoutExpired:
        print(f"  HARD TIMEOUT at {HARD_TIMEOUT}s — killing process group, moving on", flush=True)
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        p.wait()
print("=== sweep worker done ===", flush=True)
