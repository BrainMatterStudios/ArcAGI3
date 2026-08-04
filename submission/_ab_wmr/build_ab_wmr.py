#!/usr/bin/env python3
"""Build submission/_ab_wmr/ab-wmr.ipynb — ROUND 4: the BEHAVIORAL ADAPTER
EVAL. The served WEIGHTS are the single variable; the duck config (v7 pins) is
identical in both arms. Round 1 (WMR trio) is preserved at commit 3c21726;
round 2 (graph/compact) at ae1286b; round 3 (playbook) at 28e253a, artifacts
in docs/test-artifacts-2026-08-02/ (ab_round3_result.json + ab_round3_kernel/).

NOT a competition submission. A plain GPU commit kernel on the _rig mechanism:
duck-base notebook, serve forced, run cell replaced by the A/B wave driver
(ab_wave_driver.py). Waves M,B,B,M:
    M = vLLM serves /tmp/merged_sft = base + sft-synth-v1 checkpoint-10 LoRA
    B = vLLM serves the base FP8 snapshot
The merge runs in a NEW notebook cell inserted before the serve cell, an
adapted copy of the PROVEN duck-sft v4 recipe (sub 55160933: subprocess
isolation for VRAM reclamation, dequantize_fp8_inplace +
strip_quantization_runtime from the corpus's sft_common, PeftModel
merge_and_unload, save_original_format=False, processor + chat template copy,
quantization keys stripped from the merged config). The serve cell gets the
duck-sft MODEL_PATH_ANCHOR override so the FIRST server start (wave 0 = M)
already serves the merged tree; the driver restarts vLLM at every arm switch
and re-runs the full serving assert with per-arm model-identity checks.

Adapter source: kernel output of ahmedmobasher86/arc-agi-3-sft-synth-v1
(kernel_sources; v4 run proved it mounts at /kaggle/input/arc-agi-3-sft-synth-v1/
and the checkpoint-10 asserts passed). sft_common comes from
ahmedmobasher86/arc3-corpus-synth-v1 (dataset_sources; contains
dequantize_fp8_inplace/strip_quantization_runtime, byte-identical to the k3
corpus copy the duck-sft merge used). The merge subprocess additionally
prepends the arc3-deps-prep kernel output's deps/ dir to sys.path
(transformers-main 5.14.0.dev0 + peft 0.19.1, torch stripped) because the
stock image transformers does not know model_type qwen3_5 — the exact
env-prep sft-synth-v1 itself trained with. vLLM is unaffected: it runs from
the wheelhouse's vllm-site-packages, which served qwen3_5 in rounds 1-3 and
served a merged tree in duck-sft v4 (sub 55160933).

Every arm gets the IDENTICAL patch layer the submitted v6 kernel installs:
apply_all() from the CURRENT duck_patches.py (now through patch14: playbook +
antifreeze, the HUD rotation fix, marker forwarding, grid-burner default OFF,
the patch6 no-__file__ fix). Expected SKIPs on this bundle, positively
asserted by the apply gate:
  * patch9 hud-sandbox — the scored bundle's sandbox bootstrap has no
    state_hash/diff_frames; patch9 detects that and declines.
  * patch6 tool_agent_analyze — the arcagi3-agent dataset is deliberately NOT
    attached to this kernel (its heuristic prober would own the first 200
    actions of every game and confound both arms against rounds 1-2).
Anything else SKIP/FAIL/REVIEW aborts before a GPU-minute is spent on games.

Round-4 instrumentation (see ab_wave_driver.py): all round-3 row metrics
(fixed gen_tokens accounting, HUD mask stats, watchdog/replay/antifreeze),
plus per-wave serve records (which weights, restart wall time), the per-arm
serving assert (merged-tree identity for M, base-snapshot identity for B) and
the cross-arm temperature-0 logprob fingerprint gate (identical fingerprints
=> silent no-op merge => abort). Pre-registered reading baked into the result
JSON header (primary = paired per-game levels + event texture; scores are
noise at 2 waves/arm; neither outcome ships anything by itself).

Output artifact: /kaggle/working/ab_result.json (written after every wave and
on failure) + a compact summary table at the end of the log.

Usage:
    .venv/bin/python submission/_ab_wmr/build_ab_wmr.py
    kaggle kernels push -p submission/_ab_wmr --accelerator NvidiaRtxPro6000
(the accelerator flag is required — metadata machine_shape does NOT select
the RTX Pro 6000; July-proven on the sft kernels)
Env overrides at build time (baked into nothing — the driver reads them at
RUN time, so on Kaggle the defaults in ab_wave_driver.py apply):
    AB_GAMES, AB_WAVES, AB_BUDGET, AB_DEADLINE_S  (see ab_wave_driver.py)
"""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
PATCHES = REPO / "submission/_duck_patched/duck_patches.py"
PROBE = REPO / "submission/_rig/behav_probe.py"
DRIVER = Path(__file__).parent / "ab_wave_driver.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "ab-wmr.ipynb"
SLUG = "arc-agi-3-ab-wmr"

HOOK_MARKER = "Make one-off changes to `bm`, `bm.games`, or `bm.solver` here"
SERVE_GUARD = "if TRUE_SUBMISSION:  # serving Qwen needs the eval GPU; the CPU-safe commit skips it"
RUN_MARKER = "# Build the live competition game list from the gateway's available environments."

# ---------------------------------------------------------------------------
# ROUND 4: in-kernel merge (adapted from the PROVEN duck-sft v4 recipe,
# submission/_duck_sft/build_duck_sft.py, sub 55160933) + the duck-sft
# MODEL_PATH override so the FIRST vLLM start serves the merged tree.
# ---------------------------------------------------------------------------
SETUP_RUN_LINE = "subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)"
MODEL_PATH_ANCHOR = "MODEL_PATH = resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)"
MODEL_PATH_OVERRIDE = (
    "MODEL_PATH = Path('/tmp/merged_sft') if Path('/tmp/merged_sft').exists() "
    "else resolve_kaggle_dataset_path(MODEL_OWNER, MODEL_SLUG)\n"
    "print(f'[ab-merge] vLLM will serve: {MODEL_PATH}', flush=True)\n"
    "assert str(MODEL_PATH) == '/tmp/merged_sft', (\n"
    "    f'[ab-merge] REFUSING to serve {MODEL_PATH}: the merged adapter is not "
    "present, so wave 0 (arm M) would run base weights under a merged label')"
)

# The merge itself, run via `python -c` in an ISOLATED subprocess so the OS
# reclaims 100% of the GPU VRAM before vLLM initializes. Byte-for-byte the
# duck-sft v4 recipe except: checkpoint selection pinned to the sft-synth
# kernel's sft_out/checkpoint-10, and the corpus glob pinned to the synth
# corpus (which carries the same sft_common as the k3 corpus).
MERGE_INNER = '''\
import glob, json, os, shutil, sys, time
# OFFLINE env-prep (sft-synth lineage): the kernel image's stock transformers
# does not know model_type qwen3_5 (v4 abort: KeyError 'qwen3_5' at
# AutoConfig). The arc3-deps-prep kernel output bundles transformers-main
# (5.14.0.dev0) + peft 0.19.1 with torch/nvidia/triton stripped, so the
# image's Blackwell torch is kept — exactly the path sft-synth-v1 trained
# with. Internet is off; this is the only offline route to a new-enough
# transformers. Scoped to THIS subprocess: the notebook process and vLLM
# (which runs from the wheelhouse's own site-packages) are untouched.
_deps = sorted(p for p in glob.glob('/kaggle/input/**/deps', recursive=True)
               if 'deps-prep' in p)
assert _deps, 'arc3-deps-prep deps dir not found under /kaggle/input'
sys.path.insert(0, _deps[0])
import torch
import transformers
print(f'[ab-merge-subprocess] deps on path: {_deps[0]} | '
      f'transformers {transformers.__version__} | torch {torch.__version__}', flush=True)
assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (5, 6), \\
    f'qwen3_5 needs transformers>=5.6.2, got {transformers.__version__}'
CORPUS = next(os.path.dirname(p)
              for p in sorted(glob.glob('/kaggle/input/**/train.jsonl', recursive=True))
              if 'synth' in p)
sys.path.insert(0, CORPUS)
from sft_common import dequantize_fp8_inplace, strip_quantization_runtime
MODEL = next(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
             if 'tokenizer_bundle' not in p and json.load(open(p)).get('model_type') == 'qwen3_5')
_ckpts = sorted(glob.glob('/kaggle/input/**/sft_out/checkpoint-10', recursive=True))
assert len(_ckpts) == 1, f'expected exactly one sft_out/checkpoint-10 under /kaggle/input, got {_ckpts}'
CKPT = _ckpts[0]
assert 'sft-synth' in CKPT, f'checkpoint-10 is not from the sft-synth kernel output: {CKPT}'
MERGED = '/tmp/merged_sft'
print(f'[ab-merge-subprocess] base model: {MODEL} | adapter ckpt: {CKPT}', flush=True)

from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0})
for a in ('quantization_config', '_pre_quantization_dtype'):
    if hasattr(model.config, a):
        try: setattr(model.config, a, None)
        except Exception: pass
model.is_quantized = False
if hasattr(model, 'hf_quantizer'): model.hf_quantizer = None
dequantize_fp8_inplace(model)
strip_quantization_runtime(model)
pmodel = PeftModel.from_pretrained(model, CKPT, is_trainable=False)
merged = pmodel.merge_and_unload()
merged.config.torch_dtype = torch.bfloat16
merged.config.use_cache = True
# save_original_format defaults True in transformers 5.14.0.dev0 and runs
# revert_weight_conversion(), where an op is None ->
#   AttributeError: 'NoneType' object has no attribute 'convert'
# (caught by serve-verify-k3 v6). vLLM reads HF format, so skip the revert.
for _a in ('hf_quantizer', '_hf_peft_config_loaded'):
    if hasattr(merged, _a):
        try: setattr(merged, _a, None)
        except Exception: pass
merged.is_quantized = False
try:
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size='4GB',
                           save_original_format=False)
except TypeError:
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size='4GB')
try: proc = AutoProcessor.from_pretrained(MODEL)
except Exception: proc = AutoProcessor.from_pretrained(os.path.join(CORPUS, 'tokenizer_bundle'))
proc.save_pretrained(MERGED)
for extra in ('chat_template.jinja', 'chat_template.json'):
    src = os.path.join(MODEL, extra)
    if os.path.exists(src) and not os.path.exists(os.path.join(MERGED, extra)):
        shutil.copy(src, MERGED)
cfg = json.load(open(os.path.join(MERGED, 'config.json')))
for k in ('quantization_config', '_pre_quantization_dtype', 'compression_config'): cfg.pop(k, None)
json.dump(cfg, open(os.path.join(MERGED, 'config.json'), 'w'), indent=2)
print(f'[ab-merge-subprocess] successfully merged adapter to {MERGED} in {time.time()-t0:.0f}s', flush=True)
'''


def merge_cell() -> str:
    return (
        "# ============================================================================\n"
        "# ROUND 4 in-kernel MERGE: base + sft-synth-v1 checkpoint-10 -> /tmp/merged_sft\n"
        "# Adapted from the PROVEN duck-sft v4 recipe (sub 55160933): the merge runs in\n"
        "# an ISOLATED SUBPROCESS so the OS reclaims 100% of the GPU VRAM before vLLM\n"
        "# initializes. Runs UNCONDITIONALLY: this commit kernel IS the experiment, and\n"
        "# a missing merge must fail loudly (a builder is not a build).\n"
        "# ============================================================================\n"
        "import subprocess\n"
        f"_ab_merge_script = {MERGE_INNER!r}\n"
        "print('[ab-merge] running in-kernel merge (isolated subprocess)...', flush=True)\n"
        "_ab_merge_res = subprocess.run([sys.executable, '-c', _ab_merge_script], capture_output=False)\n"
        "print(f'[ab-merge] merge subprocess exited with code {_ab_merge_res.returncode}', flush=True)\n"
        "if _ab_merge_res.returncode != 0:\n"
        "    raise RuntimeError(f'[ab-merge] merge subprocess FAILED (rc={_ab_merge_res.returncode})')\n"
        "_ab_need = ['config.json', 'model.safetensors.index.json']\n"
        "_ab_missing = [f for f in _ab_need if not os.path.exists(os.path.join('/tmp/merged_sft', f))]\n"
        "if _ab_missing:\n"
        "    raise RuntimeError(f'[ab-merge] merged model incomplete, missing {_ab_missing}')\n"
        "_ab_cfg = json.loads(Path('/tmp/merged_sft/config.json').read_text())\n"
        "assert 'quantization_config' not in _ab_cfg, '[ab-merge] merged config still quantized'\n"
        "print('[ab-merge] MERGE VERIFIED — /tmp/merged_sft complete and quantization-free', flush=True)\n"
    )

APPLY_BLOCK = '''
# --- full v6 patch application (ALL arms, identical) ----------------------------
# Round 3 installs the SAME layer the submitted v6 kernel installs: apply_all()
# (now through patch14). Arms differ ONLY in env pins the driver sets per wave
# (each patch reads its switch at call time). Hard gate (rig pack law: an
# unpatched arm comparison is meaningless): any FAIL/REVIEW aborts, and SKIP is
# allowed ONLY for the two lines this bundle is EXPECTED to skip —
#   patch9: the scored bundle's sandbox bootstrap lacks state_hash/diff_frames
#           (patch9 self-detects the round-1 measured defect and declines);
#   patch6: arcagi3-agent is deliberately not attached (its prober would own
#           the first 200 actions and confound both arms vs rounds 1-2).
# Both are asserted POSITIVELY: if either unexpectedly applied, the bundle
# under test is not the one this experiment was designed for — abort.
_ab_patch_results = apply_all()
_AB_EXPECTED_SKIPS = ("patch9 hud-sandbox:", "patch6 tool_agent_analyze:")
_ab_bad = [line for line in _ab_patch_results
           if "FAIL" in line or "REVIEW" in line
           or ("SKIP" in line and not line.startswith(_AB_EXPECTED_SKIPS))]
if _ab_bad:
    raise RuntimeError(f"[ab] patch layer did not fully apply: {_ab_bad}")
for _prefix in _AB_EXPECTED_SKIPS:
    _line = next((l for l in _ab_patch_results if l.startswith(_prefix)), "")
    if "SKIP" not in _line:
        raise RuntimeError(
            f"[ab] expected {_prefix} SKIP on this bundle, got: {_line!r}")
print("[ab] patch layer = v6 apply_all(); expected SKIPs verified "
      "(patch9 sandbox-defect decline, patch6 prober not mounted)", flush=True)
# Belt and suspenders: prove the sandbox is ALIVE after patching. patch9 (had
# it been applied) kills every sandbox subprocess on this bundle; a dead
# sandbox turns the whole A/B into two zero-action arms.
from inference.agent import python_tool_sandbox as _ab_ptx
_ab_sbx = _ab_ptx.run_sandboxed_python(
    code="print('sandbox-alive')", timeout_seconds=20,
    initial_state={"current_frame": [[0]], "valid_actions": [], "history": []},
    action_handler=lambda actions: {"result": [], "state": {}})
if "sandbox-alive" not in str(_ab_sbx.get("stdout", "")):
    raise RuntimeError(f"[ab] python sandbox is DEAD after patching: {_ab_sbx}")
print("[ab] sandbox liveness: OK", flush=True)
'''

PROBE_BLOCK = '''
if not install():
    raise RuntimeError("[ab] behavioural probe failed to install")
behav_report = report


def behav_raw():
    """Raw cumulative probe counters (diffable per wave offline)."""
    return {stem: {k: v for k, v in s.items() if not k.startswith("_")}
            for stem, s in _G.items()}


print("[ab] behavioural probe ACTIVE (identical in both arms)", flush=True)
'''


def hook_cell() -> str:
    return (
        "# ============================================================================\n"
        "# A/B patch layer. Inlined from submission/_duck_patched/duck_patches.py and\n"
        "# submission/_rig/behav_probe.py by build_ab_wmr.py — edit those and rebuild.\n"
        "# ============================================================================\n"
        f"{PATCHES.read_text()}\n"
        f"{APPLY_BLOCK}\n"
        "# --- behavioural probe: identical in every arm, observes only ---\n"
        f"{PROBE.read_text()}\n"
        f"{PROBE_BLOCK}"
    )


def run_cell() -> str:
    return (
        "# rig-style A/B run. Replaces duck-base's submission cell (its gateway poll\n"
        "# cannot succeed in a commit run). NOT a submission; no submission.parquet.\n"
        "print((BUNDLE_DIR / \"preamble.txt\").read_text())\n"
        "os.environ.setdefault(\"RECORDINGS_DIR\", str(WORKING_DIR / \"server_recording\"))\n"
        "\n"
        f"{DRIVER.read_text()}\n"
        "\n"
        "_ab_result = await ab_main(bm=bm, target=target, working_dir=WORKING_DIR,\n"
        "                           notebook_start=NOTEBOOK_START_EPOCH,\n"
        "                           behav_report=behav_report, behav_raw=behav_raw)\n"
    )


def main() -> None:
    # The hook/run/merge cells must be valid python on their own (they contain
    # inlined modules) — catch syntax breakage at build time, not on the GPU.
    ast.parse(hook_cell())
    ast.parse(MERGE_INNER)          # the subprocess merge script
    merge_src = merge_cell()
    ast.parse(merge_src)
    ast.parse(MODEL_PATH_OVERRIDE)  # the setup-command splice must parse too
    run_src = run_cell()
    ast.parse(run_src.replace("await ab_main", "_ = ab_main"))  # top-level await is notebook-only

    nb = json.loads(BASE.read_text())
    seen = {"serve": False, "hook": False, "run": False, "merge": False,
            "model_override": False}
    cells = []
    originals = []  # original cell for each emitted cell (None = inserted)
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code":
            if SERVE_GUARD in src:
                # Insert the merge cell BEFORE the serve cell so /tmp/merged_sft
                # exists when the setup command starts vLLM (wave 0 = arm M).
                cells.append({
                    "cell_type": "code", "metadata": {}, "execution_count": None,
                    "outputs": [], "source": merge_src.splitlines(keepends=True),
                })
                originals.append(None)
                seen["merge"] = True
                src = src.replace(
                    SERVE_GUARD,
                    "if True:  # ab-wmr: force the serve — a commit run must serve Qwen")
                seen["serve"] = True
                # duck-sft MODEL_PATH override: point the setup command's vLLM
                # --model at the merged tree, refusing to serve base under an
                # M label (the 2026-07-31 silent-swap failure class).
                if SETUP_RUN_LINE not in src:
                    raise SystemExit("setup-command run line moved — cannot splice "
                                     "the MODEL_PATH override")
                src = src.replace(
                    SETUP_RUN_LINE,
                    "if MODEL_PATH_ANCHOR not in command:\n"
                    "            raise RuntimeError('[ab-merge] MODEL_PATH anchor missing from "
                    "setup command — upstream moved; refusing to serve an unswapped model')\n"
                    "        cmd_mod = command.replace(MODEL_PATH_ANCHOR, MODEL_PATH_OVERRIDE)\n"
                    "        subprocess.run(cmd_mod, shell=True, check=True, cwd=WORKING_DIR, env=env)")
                src = (
                    f"MODEL_PATH_ANCHOR = {MODEL_PATH_ANCHOR!r}\n"
                    f"MODEL_PATH_OVERRIDE = {MODEL_PATH_OVERRIDE!r}\n"
                    + src
                )
                ast.parse(src)  # the spliced serve cell must still parse
                seen["model_override"] = True
            elif HOOK_MARKER in src:
                src = hook_cell()
                seen["hook"] = True
            elif RUN_MARKER in src:
                src = run_src
                seen["run"] = True
        out = dict(cell)
        out["source"] = src.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            out["execution_count"] = None
            out["outputs"] = []
        cells.append(out)
        originals.append(cell)

    missing = [k for k, v in seen.items() if not v]
    if missing:
        raise SystemExit(f"never found anchors: {missing}")
    for i, (before, after) in enumerate(zip(originals, cells)):
        if before is None:
            continue  # inserted merge cell, built with the standard keys
        lost = set(before) - set(after)
        if lost:
            raise SystemExit(f"cell {i} lost keys {sorted(lost)} — nbconvert will reject this")

    nb["cells"] = cells
    OUT.write_text(json.dumps(nb, indent=1))
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{SLUG}",
        "title": SLUG,
        "code_file": OUT.name,
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
            # round 4: sft_common (dequantize/strip helpers) for the merge
            "ahmedmobasher86/arc3-corpus-synth-v1",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        # round 4: the adapter — sft-synth-v1 kernel OUTPUT (sft_out/checkpoint-10)
        # + arc3-deps-prep — offline transformers-main/peft for the merge
        # subprocess (v4 abort: stock image transformers lacks qwen3_5)
        "kernel_sources": ["ahmedmobasher86/arc-agi-3-sft-synth-v1",
                           "ahmedmobasher86/arc3-deps-prep"],
        "model_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {OUT} ({len(nb['cells'])} cells, anchors {sorted(seen)})")
    print(f"push with: kaggle kernels push -p {OUT_DIR}")


if __name__ == "__main__":
    main()
