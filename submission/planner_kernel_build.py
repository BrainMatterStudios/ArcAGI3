"""Build the Kaggle T4 PLANNER-FEASIBILITY kernel: attaches Qwen2.5-VL-7B (official Kaggle model) +
the devkit dataset (arcagi3 + dev games), runs the interactive VLM-planner loop on legible games on a
real T4. Decisive test: can a 7B (T4-size) VLM PLAN where the 3B collapsed to frame-blind output?
Dev kernel (internet ON for pip; this is feasibility, not the offline submission).

Run:  python submission/planner_kernel_build.py
      kaggle kernels push -p submission/planner_nb --accelerator NvidiaTeslaT4
      kaggle kernels status ahmedmobasher86/arcagi3-vlm-planner-t4
      kaggle kernels output ahmedmobasher86/arcagi3-vlm-planner-t4 -p <dir>
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
OWNER = "ahmedmobasher86"

INSTALL = (
    "!pip install -q --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv\n"
    "!pip install -q transformers qwen-vl-utils accelerate\n"
    "!pip install -q 'pillow==11.3.0'\n"   # transformers upgrades pillow to 12.x -> breaks matplotlib/torchvision (_Ink)
    "print('deps installed', flush=True)\n"
)

BODY = r'''
import os, sys, glob, json, re, time, logging
os.environ["ARC_API_KEY"] = "local-dev"
logging.basicConfig(level=logging.ERROR)
import numpy as np, torch

print("=== GPU CHECK ===", flush=True)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "-", flush=True)

# locate arcagi3 + games (devkit dataset) and the Qwen model dir
_init = glob.glob("/kaggle/input/**/arcagi3/__init__.py", recursive=True)
ROOT = os.path.dirname(os.path.dirname(_init[0])); sys.path.insert(0, ROOT)
GAMES = os.path.join(ROOT, "environment_files")
_cfg = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "arcagi3" not in p]
MODELDIR = None
for p in _cfg:
    d = os.path.dirname(p)
    if glob.glob(os.path.join(d, "*.safetensors")):
        MODELDIR = d; break
print("ROOT", ROOT, "GAMES_ok", os.path.isdir(GAMES), "MODELDIR", MODELDIR, flush=True)

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
PAL=["#000000","#0074D9","#FF4136","#2ECC40","#FFDC00","#AAAAAA","#F012BE","#FF851B","#7FDBFF","#870C25","#555555","#FFFFFF","#39CCCC","#01FF70","#85144b","#B10DC9"]
cmap=ListedColormap(PAL)

print("loading 7B VLM (HF, internet on) ...", flush=True); t0=time.time()
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
HFID="Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(HFID, torch_dtype=torch.float16, device_map="auto")  # shard across both T4s
proc = AutoProcessor.from_pretrained(HFID)
print(f"VLM loaded in {time.time()-t0:.0f}s", flush=True)

def render(grid, path):
    fig,ax=plt.subplots(figsize=(5,5)); ax.imshow(grid,cmap=cmap,vmin=0,vmax=15)
    ax.set_xticks(range(0,64,8)); ax.set_yticks(range(0,64,8)); ax.grid(True,alpha=0.3,lw=0.3)
    ax.set_title("frame (x=col,y=row 0-63)",fontsize=8); plt.tight_layout(); plt.savefig(path,dpi=70); plt.close()

def ask(img, prompt):
    msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    inp=proc(text=[text],images=imgs,videos=vids,padding=True,return_tensors="pt").to("cuda:0")
    with torch.no_grad(): out=model.generate(**inp,max_new_tokens=300,do_sample=False)
    return proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0]

GOALS={"su15":"move the magenta/purple piece onto the dark-red circular target",
       "lp85":"click the cell whose colour completes the pattern shown around the border",
       "cd82":"rotate/move the magenta piece so it matches the target configuration in the top green bar"}

def planner(g, budget=80, max_rounds=20):
    env=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=GAMES,logger=logging.getLogger("x")).make(game_id=g,scorecard_id="s")
    obs=env.reset(); a=0; rounds=0; goal=GOALS.get(g,"make the score increase")
    PROMPT=(f"Interactive 64x64 grid puzzle (x=col 0-63, y=row 0-63). You act by CLICKING a cell (x,y). "
            f"GOAL: {goal}. In ONE sentence say where the player piece and the target are, then a line "
            f"CLICKS=[[x,y],...] with 2-4 distinct valid coords (0-63) toward the goal.")
    last=""
    while a<budget and rounds<max_rounds:
        if obs.state==GameState.WIN: break
        if obs.state in (GameState.GAME_OVER,GameState.NOT_PLAYED) or len(np.asarray(obs.frame).ravel())==0:
            obs=env.reset(); a+=1; continue
        grid=P.to_grid(obs.frame); img="/kaggle/working/_cur.png"; render(grid,img)
        resp=ask(img,PROMPT); rounds+=1; last=resp[:120]
        mm=re.search(r"CLICKS=\s*(\[\s*\[.*?\]\s*\])",resp,re.S) or re.search(r"(\[\s*\[.*?\]\s*\])",resp,re.S)
        clicks=[]
        if mm:
            try: clicks=[(int(x),int(y)) for x,y in json.loads(mm.group(1)) if 0<=int(x)<64 and 0<=int(y)<64][:4]
            except Exception: clicks=[]
        print(f"  [{g}] round{rounds} a{a}: {clicks if clicks else repr(resp[:70])}", flush=True)
        if not clicks: break
        for (x,y) in clicks:
            obs=env.step(GameAction.ACTION6,data={"x":int(x),"y":int(y)}); a+=1
            if int(obs.levels_completed or 0)>0:
                print(f"  [{g}] LEVEL UP at action {a} (VLM-planned)!  blind baseline ~368", flush=True); return a
            if obs.state!=GameState.NOT_FINISHED: break
    print(f"  [{g}] no levelup in {a} actions/{rounds} rounds. last={last!r}", flush=True); return None

print("=== PLANNER FEASIBILITY (7B on T4) ===", flush=True)
for g in ["su15","lp85","cd82"]:
    t=time.time(); r=planner(g); print(f"RESULT {g}: {'L0 in '+str(r) if r else 'no levelup'} ({time.time()-t:.0f}s)", flush=True)
print("=== DONE ===", flush=True)

import pandas as pd
pd.DataFrame([["1_0","1",True,0]],columns=["row_id","game_id","end_of_game","score"]).to_parquet("/kaggle/working/submission.parquet",index=False)
'''

def cell(src):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}

nb={"cells":[cell(INSTALL),cell(BODY)],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python","version":"3.12"}},
    "nbformat":4,"nbformat_minor":5}

NB=HERE/"planner_nb"; NB.mkdir(exist_ok=True)
(NB/"notebook.ipynb").write_text(json.dumps(nb,indent=1))
(NB/"kernel-metadata.json").write_text(json.dumps({
    "id": f"{OWNER}/arcagi3-vlm-planner-t4",
    "title": "arcagi3-vlm-planner-t4",
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [f"{OWNER}/arcagi3-devkit"],
    "model_sources": ["qwen-lm/qwen2.5-vl/transformers/7b-instruct/2"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
}, indent=2))
print("built", NB)
