"""Detached launcher for the specialist-tier falsifier re-run
(the 4 specialist games + tu93 sanity, budget 2700 s/game, 3 jobs).

Usage: .venv/bin/python submission/_search_core/launch_specialist_rerun.py
Appends stdout+stderr to results/specialist_rerun.log; survives session end
(start_new_session=True)."""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LOG = os.path.join(HERE, "results", "specialist_rerun.log")
GAMES = "ft09,tn36,sc25,wa30,tu93"

os.makedirs(os.path.dirname(LOG), exist_ok=True)
cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
       os.path.join(HERE, "run_falsifier.py"),
       "--budget", "2700", "--jobs", "3", "--games", GAMES,
       "--tag", "specialist_rerun"]
with open(LOG, "a") as log:
    log.write(f"\n===== launch: {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=ROOT)
print(f"launched pid {p.pid} -> {LOG}")
