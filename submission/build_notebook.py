"""Generate submission/notebook.ipynb — the one-click Kaggle submission notebook.

Embeds the verified submission/my_agent.py and replicates the official harness from the
"ARC3 Sample Submission" notebook (offline wheel install, gateway-served games, official
ARC-AGI-3-Agents framework run). Run:  python submission/build_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
MY_AGENT = (HERE / "my_agent.py").read_text()

INSTALL = """\
# Install the ARC-AGI-3 toolkit + engine offline from the competition wheels.
!pip install --no-index --find-links \\
    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
    arc-agi python-dotenv
"""

WRITE_AGENT = "%%writefile /kaggle/working/my_agent.py\n" + MY_AGENT

# Harness: at competition rerun, the gateway serves the private games. We run our agent
# through the official framework against the gateway (ONLINE mode). The arcagi3 package
# must be attached to this notebook as the Kaggle dataset "arcagi3-agent"; my_agent.py
# adds /kaggle/input/arcagi3-agent[/src] to sys.path automatically.
RUN = """\
import os, shutil, glob

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # 1) Wait for the gateway that serves the private games.
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
          --retry-max-time 600 http://gateway:8001/api/games

    # 2) Copy the official agents framework to a writable location.
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
           /kaggle/working/ARC-AGI-3-Agents

    # 3) Drop our agent + the arcagi3 package next to the framework templates so it
    #    imports cleanly even if the dataset attach path differs.
    !cp /kaggle/working/my_agent.py \\
        /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    for cand in ['/kaggle/input/arcagi3-agent/src/arcagi3',
                 '/kaggle/input/arcagi3-agent/arcagi3',
                 '/kaggle/input/arcagi3/arcagi3']:
        if os.path.isdir(cand):
            dst = '/kaggle/working/ARC-AGI-3-Agents/agents/templates/arcagi3'
            if not os.path.isdir(dst):
                shutil.copytree(cand, dst)
            break

    # 4) Minimal __init__.py: register only what we need (avoid heavy template imports).
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

    # 5) .env pointing the framework at the gateway (ONLINE mode, no local env files).
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

    # 6) Play all games. The gateway records the scorecard -> submission.
    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent
"""

DUMMY = """\
import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Local/commit run: emit a placeholder submission so the notebook saves cleanly.
    import pandas as pd
    submission = pd.DataFrame(
        data=[['1_0', '1', True, 0]],
        columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
    submission.head()
"""

MD = """\
# ARC-AGI-3 — Hybrid Explorer Agent

General, training-free interactive agent (MIT-0). Motion model (avatar + per-action
displacement) + coordinate navigation to candidate goals, with graph-based
exploration/exploitation fallback. Runs through the official ARC-AGI-3-Agents framework
against the gateway-served games.

**Setup:** attach the `arcagi3-agent` dataset (this repo's `src/`) to the notebook.
"""


def cell_code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


def cell_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {
    "cells": [
        cell_md(MD),
        cell_code(INSTALL),
        cell_code(WRITE_AGENT),
        cell_code(RUN),
        cell_code(DUMMY),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "notebook.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size} bytes, {len(nb['cells'])} cells)")
