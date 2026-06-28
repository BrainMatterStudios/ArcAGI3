"""Build a Kaggle T4 DEV-MEASUREMENT kernel: dataset (arcagi3 package + dev games) + notebook
that runs the rewardrl-vs-transfer A/B at scale on a real Tesla T4. NOT a competition submission
— a measurement run to settle whether the reward-RL derail is a CPU/scale artifact.

Run:  python submission/devkit_build.py
Then: kaggle datasets create -p submission/devkit_ds --dir-mode zip   # (or: datasets version -p ... -m msg)
      kaggle kernels push -p submission/devkit_nb
      kaggle kernels status ahmedmobasher86/arcagi3-rewardrl-t4
      kaggle kernels output ahmedmobasher86/arcagi3-rewardrl-t4 -p <dir>
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SRC_PKG = ROOT / "src" / "arcagi3"
ENV = ROOT / "environment_files"
OWNER = "ahmedmobasher86"

# ---- dataset payload: lib/arcagi3 + lib/environment_files (--dir-mode zip strips one level) ----
DS = HERE / "devkit_ds"
if DS.exists():
    shutil.rmtree(DS)
(DS / "lib" / "arcagi3").mkdir(parents=True)
for p in SRC_PKG.rglob("*.py"):
    dst = DS / "lib" / "arcagi3" / p.relative_to(SRC_PKG)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
shutil.copytree(ENV, DS / "lib" / "environment_files")
(DS / "dataset-metadata.json").write_text(json.dumps({
    "title": "arcagi3-devkit", "id": f"{OWNER}/arcagi3-devkit",
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))

# ---- notebook ----
INSTALL = (
    "!pip install --no-index --find-links "
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels "
    "arc-agi python-dotenv\n"
)

AB = r'''
import os, sys, time, glob, logging
os.environ["ARC_API_KEY"] = "local-dev"
logging.basicConfig(level=logging.ERROR)
# Discover where the dataset placed the package + games (zip-mode strip is path-dependent).
# Kaggle mounts datasets under /kaggle/input/datasets/<owner>/<slug>/ (newer layout) — discover it.
_inits = glob.glob("/kaggle/input/**/arcagi3/__init__.py", recursive=True)
_inits = [p for p in _inits if "environment_files" not in p]
_root = os.path.dirname(os.path.dirname(_inits[0]))   # dir containing arcagi3/ + environment_files/
sys.path.insert(0, _root)
GAMES_DIR = os.path.join(_root, "environment_files")
assert os.path.isdir(GAMES_DIR), (_root, os.listdir(_root))
print("PKG_ROOT", _root, "GAMES_DIR", GAMES_DIR, "n_games", len(os.listdir(GAMES_DIR)), flush=True)

print("=== GPU CHECK ===", flush=True)
GPU = False
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
          "device_count", torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available():
        x = torch.ones(8, device="cuda")
        GPU = float((x + x).sum().item()) == 16.0
        print("gpu0", torch.cuda.get_device_properties(0).name, "real_op_ok", GPU, flush=True)
except Exception as e:
    print("torch/gpu FAILED", repr(e), flush=True)
print("GPU_USABLE", GPU, flush=True)

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.learned_explorer import LearnedExplorer
from arcagi3.reward_rl_learner import RewardRLLearner

DENSE = dict(trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256)

def mk(name):
    if name == "transfer":
        return TransferExplorer(seed=0, **DENSE)
    # require_gpu=True -> the T4 fail-safe must pass for the learner to engage (else == transfer)
    return LearnedExplorer(seed=0, enable_learn=True, learner=RewardRLLearner(),
                           require_gpu=True, learn_mode="propose", **DENSE)

def run(game, name, budget):
    env = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR,
                 logger=logging.getLogger("x")).make(game_id=game, scorecard_id="s")
    pol = mk(name); obs = env.reset(); a = 0; lv = 0
    while a < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, gstate_terminal=(obs.state == GameState.GAME_OVER),
                         gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        a += 1; lv = int(obs.levels_completed or 0)
    return lv

BUDGET = 20000
GAMES = ["tu93", "lp85", "cd82", "vc33", "ar25", "lf52"]
print("=== AB_BEGIN budget", BUDGET, "===", flush=True)
tt = tr = 0
for g in GAMES:
    t0 = time.time(); tl = run(g, "transfer", BUDGET); t1 = time.time()
    rl = run(g, "rewardrl", BUDGET); t2 = time.time()
    tt += tl; tr += rl
    tag = "WIN" if rl > tl else ("REGRESS" if rl < tl else "tie")
    print(f"AB {g} transfer L{tl} ({t1-t0:.0f}s)  rewardrl L{rl} ({t2-t1:.0f}s)  {tag}", flush=True)
print(f"AB_TOTAL transfer L{tt}  rewardrl L{tr}", flush=True)
print("=== AB_END ===", flush=True)

import pandas as pd
pd.DataFrame([["1_0", "1", True, 0]],
             columns=["row_id", "game_id", "end_of_game", "score"]
             ).to_parquet("/kaggle/working/submission.parquet", index=False)
'''

def cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

nb = {"cells": [cell(INSTALL), cell(AB)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

NB = HERE / "devkit_nb"
NB.mkdir(exist_ok=True)
(NB / "notebook.ipynb").write_text(json.dumps(nb, indent=1))
(NB / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{OWNER}/arcagi3-rewardrl-t4",
    "title": "arcagi3-rewardrl-t4",
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [f"{OWNER}/arcagi3-devkit"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": [],
}, indent=2))

print("built", DS, "and", NB)
print("dataset files:", len(list((DS / "lib" / "arcagi3").rglob("*.py"))), "py +",
      len(list((DS / "lib" / "environment_files").rglob("metadata.json"))), "games")
