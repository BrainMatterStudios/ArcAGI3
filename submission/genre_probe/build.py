"""Build the Kaggle genre-VLM-probe notebook (Direction #2: does a 32B VLM clear ~75% genre on T4x2?).

Generates notebook.ipynb (single code cell) + kernel-metadata.json. Push with:
  kaggle kernels push -p submission/genre_probe/kernel --accelerator NvidiaTeslaT4
Then read results with: kaggle kernels output ahmedmobasher86/arcagi3-genre-vlm-probe -p /tmp/out
"""
import json
from pathlib import Path

CODE = r'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "accelerate", "bitsandbytes", "qwen-vl-utils"], check=False)

import io, re, time
import numpy as np
import torch
from PIL import Image

PAL = np.array([
    [0,0,0],[0,116,217],[255,65,54],[46,204,64],[255,220,0],[170,170,170],[240,18,190],[255,133,27],
    [127,219,255],[135,12,37],[100,70,30],[200,100,200],[60,180,180],[180,180,60],[120,120,255],[255,255,255]],
    dtype=np.uint8)
GENRES = ["NAVIGATE","COLLECT","PUSH","AIM","MATCH","SYMMETRY","CLICK"]
GAME_GENRE = {
 "ls20":"NAVIGATE","su15":"AIM","m0r0":"SYMMETRY","sk48":"MATCH","wa30":"PUSH","re86":"AIM","tn36":"CLICK",
 "tr87":"MATCH","vc33":"CLICK","cd82":"MATCH","sc25":"MATCH","lp85":"CLICK","lf52":"MATCH","tu93":"NAVIGATE",
 "ar25":"MATCH","sp80":"NAVIGATE","bp35":"NAVIGATE","cn04":"MATCH","dc22":"MATCH","ft09":"MATCH","g50t":"NAVIGATE",
 "ka59":"MATCH","r11l":"AIM","s5i5":"AIM","sb26":"MATCH"}
ACCEPT = {
 "re86":{"AIM","MATCH"},"sk48":{"MATCH","AIM"},"tr87":{"MATCH","CLICK"},"cd82":{"MATCH","PUSH"},"sc25":{"MATCH","PUSH"},
 "ar25":{"MATCH","PUSH"},"lp85":{"CLICK","MATCH"},"lf52":{"CLICK","MATCH","PUSH"},"tu93":{"NAVIGATE","COLLECT"},
 "sp80":{"NAVIGATE","AIM"},"bp35":{"NAVIGATE","COLLECT","PUSH"},"cn04":{"MATCH","PUSH"},"dc22":{"MATCH","CLICK"},
 "ka59":{"MATCH","PUSH","NAVIGATE"},"s5i5":{"AIM","COLLECT","PUSH"}}
PROMPT = (
 "This is the start screen of a 64x64 grid puzzle game. Classify its GENRE as ONE label:\n"
 "  NAVIGATE = a small avatar (plus/dot) to move to a target marker inside a large region\n"
 "  COLLECT  = an avatar gathering many scattered small items\n"
 "  PUSH     = push/carry blocks (often a bar) onto bordered marker squares\n"
 "  AIM      = a crosshair or dotted aim-LINE pointing a cursor at ringed/bordered targets\n"
 "  MATCH    = arrange/align pieces to a shown reference sequence, OR a grid of icon pairs\n"
 "  SYMMETRY = the board split into two halves that should mirror each other\n"
 "  CLICK    = large flat regions or a checkerboard console; NO moving avatar\n"
 "First reason about the key visual structure, then answer. Format EXACTLY:\n"
 "REASON: <one line>\nGENRE: <one label>")

print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
import glob, os
print("INPUT TREE:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "NO /kaggle/input", flush=True)
cands = glob.glob("/kaggle/input/**/all25_frames.npz", recursive=True)
print("found npz:", cands, flush=True)
assert cands, "dataset all25_frames.npz not mounted"
d = np.load(cands[0])
X = d["X"]; games = np.array([str(g) for g in d["games"]])
first = {g: X[np.where(games == g)[0][0]] for g in sorted(set(games))}

def render(g, scale=6):
    rgb = PAL[np.clip(g, 0, 15)]
    return Image.fromarray(np.kron(rgb, np.ones((scale, scale, 1), np.uint8)))

from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLM
except Exception:
    from transformers import Qwen2VLForConditionalGeneration as VLM

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
model = proc = used = None
for MODEL in ["Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
    try:
        print("loading", MODEL, "...", flush=True)
        model = VLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
        proc = AutoProcessor.from_pretrained(MODEL)
        used = MODEL
        print("LOADED", MODEL, flush=True)
        break
    except Exception as e:
        print("FAILED", MODEL, "->", repr(e)[:300], flush=True)

def classify(img):
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

def parse(t):
    m = re.search(r"GENRE:\s*([A-Z]+)", t.upper())
    if m and m.group(1) in GENRES:
        return m.group(1)
    for g in GENRES:
        if g in t.upper():
            return g
    return "?"

print("\n=== genre probe — model =", used, "===", flush=True)
hits = tot = 0
for gm in sorted(first):
    t0 = time.time()
    try:
        resp = classify(render(first[gm]))
    except Exception as e:
        resp = "ERR " + repr(e)[:120]
    g = parse(resp); accept = ACCEPT.get(gm, {GAME_GENRE[gm]}); ok = g in accept
    hits += ok; tot += 1
    print(f"{gm:7}{GAME_GENRE[gm]:10}-> {g:10}{'HIT' if ok else 'miss'}  {time.time()-t0:.1f}s", flush=True)
print(f"\nGENRE acc ({used}): {hits}/{tot} = {hits/tot:.0%}   (7B-CoT-local 60% / gate 75% / strong-VLM 83%)", flush=True)
'''


def main():
    root = Path(__file__).resolve().parent
    kdir = root / "kernel"
    kdir.mkdir(exist_ok=True)
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": [l + "\n" for l in CODE.strip("\n").split("\n")]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (kdir / "notebook.ipynb").write_text(json.dumps(nb))
    meta = {
        "id": "ahmedmobasher86/arc-agi-3-genre-vlm-probe",
        "title": "arc-agi-3-genre-vlm-probe",
        "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["ahmedmobasher86/arcagi3-genre-frames"],
        "competition_sources": [], "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", kdir / "notebook.ipynb", "and kernel-metadata.json")


if __name__ == "__main__":
    main()
