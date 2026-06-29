"""Generate submission/notebook.ipynb — self-contained one-click ARC-AGI-3 submission.

Embeds the arcagi3 core package (base64) directly in the notebook, written to
/kaggle/working/arcagi3 at runtime, so the agent has NO external dataset/path dependency
(the earlier dataset-packaging route ERRORed). Replicates the official sample harness
(offline wheels, gateway-served games, official framework run). Run:
  python submission/build_notebook.py
"""

import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parent / "src" / "arcagi3"
MY_AGENT = (HERE / "my_agent.py").read_text()

# Core modules the agent needs at eval (skip runner/validate_online/games — unused there).
CORE = ["__init__.py", "perception.py", "world_model.py", "movement.py", "agent.py",
        "policy.py", "spatial.py", "salience_explorer.py",
        "transfer_explorer.py",  # fallback (v13): SalienceExplorer + within-game reward transfer
        "cai_prune_explorer.py",       # no-op-click pruning lever
        "transfer_cai_explorer.py",    # transfer + CAI-prune combo
        "relational_explorer.py",          # portfolio strategy: relational coverage
        "transfer_relational_explorer.py", # portfolio strategy: transfer + relational
        "portfolio_policy.py",         # PRIMARY: multi-strategy portfolio (max-over-plays, strict-superset)
        "online_model.py", "online_explorer.py"]  # Phase A (opt-in via ARCAGI3_ONLINE=1)
PKG = {name: base64.b64encode((SRC / name).read_bytes()).decode() for name in CORE}

INSTALL = """\
# Install the ARC-AGI-3 toolkit + engine offline from the competition wheels.
!pip install --no-index --find-links \\
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
    arc-agi python-dotenv
"""

PROBE = """\
# Phase A.0 probe: record torch/GPU availability on Kaggle (go/no-go for online learning).
# Runs UNCONDITIONALLY (interactive run too) because code-competition RERUN logs are hidden
# — the interactive kernel log IS readable via `kaggle kernels output`, and the interactive
# GPU image is a strong proxy for the rerun image. Fully try/excepted; never crashes the run.
import os as _os
print('=== ARCAGI3_EVAL_PROBE_BEGIN ===', flush=True)
print('is_rerun', bool(_os.getenv('KAGGLE_IS_COMPETITION_RERUN')), flush=True)
try:
    import numpy as _np; print('numpy', _np.__version__, flush=True)
except Exception as _e:
    print('numpy import FAILED', repr(_e), flush=True)
try:
    import torch as _t
    print('torch', _t.__version__, flush=True)
    print('cuda_available', _t.cuda.is_available(), flush=True)
    print('device_count', _t.cuda.device_count(), flush=True)
    for _i in range(_t.cuda.device_count()):
        _p = _t.cuda.get_device_properties(_i)
        print(f'gpu{_i}', _p.name, round(_p.total_memory/1e9, 2), 'GB', flush=True)
except Exception as _e:
    print('torch import FAILED', repr(_e), flush=True)
try:
    _w = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'
    print('torch_wheels', [f for f in _os.listdir(_w) if 'torch' in f.lower()], flush=True)
except Exception as _e:
    print('wheels listdir FAILED', repr(_e), flush=True)
print('=== ARCAGI3_EVAL_PROBE_END ===', flush=True)
"""

WRITE_PKG = (
    "# Write the self-contained arcagi3 package to /kaggle/working/arcagi3 (no dataset dep).\n"
    "import base64, os, pathlib\n"
    "os.makedirs('/kaggle/working/arcagi3', exist_ok=True)\n"
    "PKG = " + repr(PKG) + "\n"
    "for _name, _b in PKG.items():\n"
    "    pathlib.Path('/kaggle/working/arcagi3', _name).write_bytes(base64.b64decode(_b))\n"
    "print('wrote arcagi3 package:', sorted(PKG))\n"
)

WRITE_AGENT = "%%writefile /kaggle/working/my_agent.py\n" + MY_AGENT

RUN = """\
import os

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # 1) wait for the gateway that serves the private games
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
          --retry-max-time 600 http://gateway:8001/api/games

    # 2) copy the official agents framework to a writable location
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
           /kaggle/working/ARC-AGI-3-Agents

    # 3) drop our agent into the framework templates (it imports arcagi3 from /kaggle/working)
    !cp /kaggle/working/my_agent.py \\
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

    # 4) minimal agents/__init__.py: register only what we need (avoid heavy template imports)
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write('''from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
    "myagent": MyAgent,
}
''')

    # 5) .env pointing the framework at the gateway (online mode, no local env files)
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write('''SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
ENVIRONMENTS_DIR=
RECORDINGS_DIR=/kaggle/working/server_recording
''')

    # 6) play all games; the gateway records the scorecard -> submission
    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent
"""

DUMMY = """\
import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    import pandas as pd
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 0]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
    submission.head()
"""

MD = """\
# ARC-AGI-3 — Hybrid Explorer Agent (self-contained)

General, training-free interactive agent (MIT-0): perception (object segmentation,
counter/distractor masking) -> state-transition graph exploration -> motion-model avatar
navigation -> 5-tier click salience. The arcagi3 package is embedded in this notebook
(written to /kaggle/working) so there is no external dependency; the agent is fail-safe
(random fallback) so it always acts.
"""


def cell_code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def cell_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {
    "cells": [cell_md(MD), cell_code(INSTALL), cell_code(PROBE), cell_code(WRITE_PKG),
              cell_code(WRITE_AGENT), cell_code(RUN), cell_code(DUMMY)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = HERE / "notebook.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size} bytes, {len(nb['cells'])} cells, "
      f"{len(PKG)} embedded modules)")
