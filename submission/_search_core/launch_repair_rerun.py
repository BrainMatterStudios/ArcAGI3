"""Detached launcher for the repair-verification falsifier re-run
(regressed + near-miss set, budget 2700 s/game, 3 jobs => 2 waves ~1.5 h).

Usage: .venv/bin/python submission/_search_core/launch_repair_rerun.py
Appends stdout+stderr to results/repair_rerun.log; survives session end
(start_new_session=True)."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOG = os.path.join(HERE, "results", "repair_rerun.log")
GAMES = "sb26,su15,r11l,ls20,vc33,cd82"

os.makedirs(os.path.dirname(LOG), exist_ok=True)
cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
       os.path.join(HERE, "run_falsifier.py"),
       "--budget", "2700", "--jobs", "3", "--games", GAMES,
       "--tag", "repair_rerun"]
with open(LOG, "a") as log:
    log.write(f"\n===== launch: {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=ROOT)
print(f"launched pid {p.pid} -> {LOG}")
