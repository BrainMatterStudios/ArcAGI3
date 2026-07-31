#!/usr/bin/env python3
"""Build submission/_duck_sft/duck-sft.ipynb — fine-tuned SFT adapter deployment notebook.

Transforms submission/_duck_base/duck-base.ipynb to perform an IN-SUBPROCESS merge of
the valid run-8 SFT adapter (ahmedmobasher86/arc3-sft-k3-ckpts v2) into Qwen3.6-27B.
Executing the merge in an isolated subprocess guarantees 100% GPU VRAM reclamation
by the OS before vLLM initializes, preventing CUDA OOM crashes.
"""
import json
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")
BASE_NB = REPO / "submission/_duck_base/duck-base.ipynb"
OUT = REPO / "submission/_duck_sft"

nb = json.loads(BASE_NB.read_text())

# Code cell executing merge in an ISOLATED Python subprocess to prevent CUDA VRAM leak
MERGE_CELL_SOURCE = [
    "# In-kernel merge executed inside an ISOLATED SUBPROCESS to ensure complete VRAM reclamation\n",
    "import os, subprocess, sys\n",
    "print('[duck-sft] checking in-kernel merge requirement...', flush=True)\n",
    "\n",
    "if TRUE_SUBMISSION or os.environ.get('FORCE_MERGE') == '1':\n",
    "    merge_script = '''\n",
    "import glob, json, os, shutil, sys, time, gc\n",
    "import torch\n",
    "CORPUS = os.path.dirname(sorted(glob.glob('/kaggle/input/**/train.jsonl', recursive=True))[0])\n",
    "sys.path.insert(0, CORPUS)\n",
    "from sft_common import dequantize_fp8_inplace, strip_quantization_runtime\n",
    "MODEL = next(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)\n",
    "             if 'tokenizer_bundle' not in p and json.load(open(p)).get('model_type') == 'qwen3_5')\n",
    "_ckpt_candidates = sorted(glob.glob('/kaggle/input/**/sft_out/checkpoint-*', recursive=True)) \\\n",
    "                 + sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/sft_adapter/adapter_config.json', recursive=True))\n",
    "CKPT = next((p for p in _ckpt_candidates if 'sft_adapter' in p or 'checkpoint-8' in p), _ckpt_candidates[-1])\n",
    "MERGED = '/tmp/merged_sft'\n",
    "print(f'[duck-sft-subprocess] base model: {MODEL} | adapter ckpt: {CKPT}', flush=True)\n",
    "\n",
    "from transformers import AutoModelForImageTextToText, AutoProcessor\n",
    "from peft import PeftModel\n",
    "t0 = time.time()\n",
    "model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0})\n",
    "for a in ('quantization_config', '_pre_quantization_dtype'):\n",
    "    if hasattr(model.config, a):\n",
    "        try: setattr(model.config, a, None)\n",
    "        except Exception: pass\n",
    "model.is_quantized = False\n",
    "if hasattr(model, 'hf_quantizer'): model.hf_quantizer = None\n",
    "dequantize_fp8_inplace(model)\n",
    "strip_quantization_runtime(model)\n",
    "pmodel = PeftModel.from_pretrained(model, CKPT, is_trainable=False)\n",
    "merged = pmodel.merge_and_unload()\n",
    "merged.config.torch_dtype = torch.bfloat16\n",
    "merged.config.use_cache = True\n",
    "# save_original_format defaults True in transformers 5.14.0.dev0 and runs\n",
    "# revert_weight_conversion(), where an op is None ->\n",
    "#   AttributeError: 'NoneType' object has no attribute 'convert'\n",
    "# (caught by serve-verify-k3 v6). vLLM reads HF format, so skip the revert.\n",
    "for _a in ('hf_quantizer', '_hf_peft_config_loaded'):\n",
    "    if hasattr(merged, _a):\n",
    "        try: setattr(merged, _a, None)\n",
    "        except Exception: pass\n",
    "merged.is_quantized = False\n",
    "try:\n",
    "    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size='4GB',\n",
    "                           save_original_format=False)\n",
    "except TypeError:\n",
    "    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size='4GB')\n",
    "try: proc = AutoProcessor.from_pretrained(MODEL)\n",
    "except Exception: proc = AutoProcessor.from_pretrained(os.path.join(CORPUS, 'tokenizer_bundle'))\n",
    "proc.save_pretrained(MERGED)\n",
    "for extra in ('chat_template.jinja', 'chat_template.json'):\n",
    "    src = os.path.join(MODEL, extra)\n",
    "    if os.path.exists(src) and not os.path.exists(os.path.join(MERGED, extra)):\n",
    "        shutil.copy(src, MERGED)\n",
    "cfg = json.load(open(os.path.join(MERGED, 'config.json')))\n",
    "for k in ('quantization_config', '_pre_quantization_dtype', 'compression_config'): cfg.pop(k, None)\n",
    "json.dump(cfg, open(os.path.join(MERGED, 'config.json'), 'w'), indent=2)\n",
    "print(f'[duck-sft-subprocess] successfully merged adapter to {MERGED} in {time.time()-t0:.0f}s', flush=True)\n",
    "'''\n",
    "    # Run merge in isolated subprocess -> OS reclaims 100% VRAM on exit\n",
    "    sub_res = subprocess.run([sys.executable, '-c', merge_script], capture_output=False)\n",
    "    print(f'[duck-sft] merge subprocess exited with code {sub_res.returncode}', flush=True)\n",
    "\n",
    "    # A failed merge used to be silent: /tmp/merged_sft simply would not exist, the\n",
    "    # setup command would fall back to the base snapshot, and the run would score as\n",
    "    # plain base duck under a fine-tuned label. Fail loudly instead.\n",
    "    if sub_res.returncode != 0:\n",
    "        raise RuntimeError(f'[duck-sft] merge subprocess FAILED (rc={sub_res.returncode})')\n",
    "    _need = ['config.json', 'model.safetensors.index.json']\n",
    "    _missing = [f for f in _need if not os.path.exists(os.path.join('/tmp/merged_sft', f))]\n",
    "    if _missing:\n",
    "        raise RuntimeError(f'[duck-sft] merged model incomplete, missing {_missing}')\n",
    "    print('[duck-sft] MERGE VERIFIED — /tmp/merged_sft complete', flush=True)\n"
]

# Insert merge cell right before vLLM setup/launch
new_cells = []
merge_inserted = False
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if c["cell_type"] == "code" and "setup_commands.json" in src and not merge_inserted:
        new_cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": MERGE_CELL_SOURCE
        })
        merge_inserted = True
    
    # Point the setup command's vLLM --model at the merged tree.
    #
    # The previous version did:
    #     command.replace("driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot", "/tmp/merged_sft")
    # That concatenated string does not occur in setup_commands.json — owner and slug are
    # separate variables (MODEL_OWNER / MODEL_SLUG) and the path is built by
    # resolve_kaggle_dataset_path(). The replace was a no-op, so vLLM served the BASE
    # snapshot while the submission was labelled fine-tuned (2026-07-31, score 0.95 =
    # base draw). Anchor on the real assignment instead, and fail the build if it moves.
    if c["cell_type"] == "code" and "setup_commands.json" in src:
        MODEL_PATH_ANCHOR = "MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)"
        MODEL_PATH_OVERRIDE = (
            "MODEL_PATH = Path('/tmp/merged_sft') if Path('/tmp/merged_sft').exists() "
            "else resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)\n"
            "print(f'[duck-sft] vLLM will serve: {MODEL_PATH}', flush=True)\n"
            "assert str(MODEL_PATH) == '/tmp/merged_sft', (\n"
            "    f'[duck-sft] REFUSING to serve {MODEL_PATH}: the merged adapter is not "
            "present, so this run would score as base duck under a fine-tuned label')"
        )
        src_mod = src.replace(
            'subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)',
            "if MODEL_PATH_ANCHOR not in command:\n"
            "            raise RuntimeError('[duck-sft] MODEL_PATH anchor missing from setup command '\n"
            "                               '— upstream moved; refusing to serve an unswapped model')\n"
            "        cmd_mod = command.replace(MODEL_PATH_ANCHOR, MODEL_PATH_OVERRIDE)\n"
            "        subprocess.run(cmd_mod, shell=True, check=True, cwd=WORKING_DIR, env=env)"
        )
        # Define the anchor constants in the cell, above the setup loop.
        src_mod = (
            f"MODEL_PATH_ANCHOR = {MODEL_PATH_ANCHOR!r}\n"
            f"MODEL_PATH_OVERRIDE = {MODEL_PATH_OVERRIDE!r}\n"
            + src_mod
        )
        new_cells.append({
            "cell_type": "code",
            "metadata": c.get("metadata", {}),
            "execution_count": None,
            "outputs": [],
            "source": src_mod.splitlines(keepends=True)
        })
        continue

    new_cells.append(c)

nb["cells"] = new_cells

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "duck-sft.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "ahmedmobasher86/arc-agi-3-duck-sft",
    "title": "arc-agi-3-duck-sft",
    "code_file": "duck-sft.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": [
        "driessmit1/arc3-vllm-h100-wheelhouse-v3",
        "ahmedmobasher86/taaf-src-hybrid",
        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
        "ahmedmobasher86/arc3-sft-k3-ckpts",
        "ahmedmobasher86/arc3-sft-k3-corpus"
    ],
    "competition_sources": [
        "arc-prize-2026-arc-agi-3"
    ],
    "kernel_sources": [],
    "model_sources": []
}

(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"wrote {OUT/'duck-sft.ipynb'} ({len(nb['cells'])} cells) with SUBPROCESS merge cell successfully!")
