#!/usr/bin/env python3
"""Build submission/_ab_wmr/ab-wmr.ipynb — ROUND 5: the K3 RUN-8 ADAPTER
BEHAVIORAL EVAL. The served WEIGHTS are the single variable; the duck config
(v7 pins) is identical in both arms. Round 1 (WMR trio) is preserved at commit
3c21726; round 2 (graph/compact) at ae1286b; round 3 (playbook) at 28e253a;
round 4 (synth adapter — FAILED behaviorally by under-deliberating, ~30%
gen_tokens drop) at 6cde0c8, result in
docs/test-artifacts-2026-08-02/ADAPTER-BEHAVIORAL-EVAL-2026-08-04.md.

NOT a competition submission. A plain GPU commit kernel on the _rig mechanism:
duck-base notebook, serve forced, run cell replaced by the A/B wave driver
(ab_wave_driver.py). Waves M,B,B,M:
    M = vLLM serves /tmp/merged_sft = base + K3 RUN-8 checkpoint-8 LoRA
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

WHY THIS CHECKPOINT (run-8 sft_out/checkpoint-8) — the pre-registered choice:
  * Run 8 is the FIRST VALID SFT run (runs 1-5 trained on a corrupted base —
    FP8 scales stripped unapplied; their poisoned ckpts-dataset v1 is fully
    superseded by the 2026-07-26 v2 upload, which carries ONLY run-8
    artifacts: sft_adapter/ (step-15) + sft_out/checkpoint-8/).
  * docs/A1-PROTOCOL-2026-08.md §3 pre-registers the checkpoint arms as
    base / checkpoint-8 / sft_adapter(step-15), with ties broken TOWARD
    ckpt-8 (OOD-peaks-early law). It names no other checkpoint.
  * Gate 0 (serve-verify-k3 v9, 2026-08-01) PASSED for checkpoint-8
    SPECIFICALLY: merged NLL gain +13.51% (bar >= 2%), merge retained 98.2%
    of the adapter's fit. The sft_adapter step-15 arm never ran its gate, so
    checkpoint-8 is the only serve-verified artifact.
  * Run-8 lineage (downloaded + verified 2026-08-04; corpus version
    CORRECTED 2026-08-04): trained on arc3-sft-k3-corpus **v1** (uploaded
    2026-07-24, the only version of that slug when run 8 trained; the row
    counts 392 train / 43 val match v1 exactly — corpus_v3 is 379/56, so the
    earlier "corpus_v3" label here was wrong). Kimi-K3 teacher, long-form
    deliberation targets, VAL target-loss base 0.7781 -> tuned 0.6807
    (-12.5%). v1 uses a ROW-level split over 20 of 25 dev games — hence the
    round-5 panel decontamination (vc33 out, dc22 in; see
    ab_wave_driver.py AB_PREREGISTERED_READING["contamination"]).
    Behaviorally NEVER validated — its one submission (55160933) errored. Round 4's synth adapter failed by learning a SHORT
    style; run-8's data has the right style, so this is the cleanest test of
    whether deliberation-shaped SFT moves play at 27B.
The merge cell hard-asserts the checkpoint identity at run time
(trainer_state global_step/max_steps/step-8 loss + adapter byte size, all
recorded from the live dataset listing 2026-08-04) — kernels always mount
the LATEST dataset version, so a surprise re-upload fails loudly instead of
silently swapping the artifact.

Adapter source: dataset ahmedmobasher86/arc3-sft-k3-ckpts (dataset_sources;
current version = the 2026-07-26 v2 run-8 upload, file listing verified via
the kaggle CLI 2026-08-04: sft_out/checkpoint-8/adapter_model.safetensors,
467062560 bytes). sft_common comes from ahmedmobasher86/arc3-sft-k3-corpus
(dataset_sources; the EXACT corpus run-8 trained on and the same
dequantize_fp8_inplace/strip_quantization_runtime pair the proven duck-sft v4
merge imported). The merge subprocess additionally
prepends the arc3-deps-prep kernel output's deps/ dir to sys.path
(transformers-main 5.14.0.dev0 + peft 0.19.1, torch stripped) because the
stock image transformers does not know model_type qwen3_5 — the exact
env-prep run-8 itself trained with (its log: deps on path -> transformers
5.14.0.dev0). vLLM is unaffected: it runs from the wheelhouse's
vllm-site-packages, which served qwen3_5 in rounds 1-4 and served a merged
tree in duck-sft v4 (sub 55160933) and ab-wmr v5 (round 4, 4.39h COMPLETE).

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

# Path-resolution block for the merge: CORPUS (sft_common), MODEL (base FP8
# snapshot) and CKPT (the run-8 checkpoint-8 adapter, hard identity asserts).
# Kept as its own constant so dry_run.py can exec it verbatim against a mock
# /kaggle/input built from the REAL dataset file listing (and against
# tampered trees, proving every abort path) without a GPU.
ADAPTER_SELECT = '''\
# --- path resolution: corpus, base model, and the RUN-8 CHECKPOINT-8 adapter --
_corp = sorted(os.path.dirname(p)
               for p in glob.glob('/kaggle/input/**/train.jsonl', recursive=True)
               if 'k3-corpus' in p)
assert len(_corp) == 1, f'expected exactly one k3-corpus train.jsonl under /kaggle/input, got {_corp}'
CORPUS = _corp[0]
MODEL = next(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
             if 'tokenizer_bundle' not in p and json.load(open(p)).get('model_type') == 'qwen3_5')
# Adapter identity chain (values recorded from the LIVE arc3-sft-k3-ckpts
# listing + downloaded trainer_state, 2026-08-04). The dataset's current
# version is the 2026-07-26 v2 upload = run 8, the FIRST VALID run (runs 1-5
# trained on a corrupted base; no artifact of theirs exists in this version).
# Kernels mount the LATEST dataset version, so these asserts turn any future
# re-upload into a loud abort instead of a silent artifact swap.
_ckpts = sorted(glob.glob('/kaggle/input/**/sft_out/checkpoint-8', recursive=True))
assert len(_ckpts) == 1, f'expected exactly one sft_out/checkpoint-8 under /kaggle/input, got {_ckpts}'
CKPT = _ckpts[0]
assert 'k3-ckpts' in CKPT, f'checkpoint-8 is not from the arc3-sft-k3-ckpts dataset: {CKPT}'
_ts = json.load(open(os.path.join(CKPT, 'trainer_state.json')))
assert _ts.get('global_step') == 8 and _ts.get('max_steps') == 15, (
    f"not run-8 checkpoint-8: global_step={_ts.get('global_step')} max_steps={_ts.get('max_steps')}")
_loss8 = next((h['loss'] for h in _ts.get('log_history', []) if h.get('step') == 8), None)
assert _loss8 is not None and abs(_loss8 - 0.7335078716278076) < 1e-9, (
    f'checkpoint-8 trainer_state does not match the verified run-8 record (step-8 loss {_loss8!r})')
_adapter_bytes = os.path.getsize(os.path.join(CKPT, 'adapter_model.safetensors'))
assert _adapter_bytes == 467062560, (
    f'adapter_model.safetensors is {_adapter_bytes} bytes, run-8 recorded 467062560')
'''

# The merge itself, run via `python -c` in an ISOLATED subprocess so the OS
# reclaims 100% of the GPU VRAM before vLLM initializes. Byte-for-byte the
# duck-sft v4 recipe except: adapter selection pinned to run-8's
# sft_out/checkpoint-8 with the identity chain above, and the corpus glob
# pinned to arc3-sft-k3-corpus (the corpus run-8 trained on — same
# sft_common the duck-sft v4 merge imported).
MERGE_INNER = '''\
import glob, json, os, shutil, sys, time
# OFFLINE env-prep (run-8 lineage): the kernel image's stock transformers
# does not know model_type qwen3_5 (round-4 v4 abort: KeyError 'qwen3_5' at
# AutoConfig). The arc3-deps-prep kernel output bundles transformers-main
# (5.14.0.dev0) + peft 0.19.1 with torch/nvidia/triton stripped, so the
# image's Blackwell torch is kept — exactly the deps run-8 itself trained
# with (its log: 'deps on path ... arc3-deps-prep/deps', transformers
# 5.14.0.dev0). Internet is off; this is the only offline route to a
# new-enough transformers. Scoped to THIS subprocess: the notebook process
# and vLLM (which runs from the wheelhouse's own site-packages) are untouched.
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
''' + ADAPTER_SELECT + '''\
sys.path.insert(0, CORPUS)
from sft_common import dequantize_fp8_inplace, strip_quantization_runtime
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
        "# ROUND 5 in-kernel MERGE: base + K3 RUN-8 checkpoint-8 -> /tmp/merged_sft\n"
        "# (arc3-sft-k3-ckpts v2 upload; identity hard-asserted in the subprocess:\n"
        "# trainer_state global_step 8 / max_steps 15 / step-8 loss + adapter bytes).\n"
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
    ast.parse(ADAPTER_SELECT)       # the path-resolution block on its own
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
            # round 5: the ADAPTER — run-8 sft_out/checkpoint-8 (v2 upload;
            # identity hard-asserted in the merge subprocess)
            "ahmedmobasher86/arc3-sft-k3-ckpts",
            # round 5: sft_common (dequantize/strip helpers) + tokenizer_bundle
            # fallback — the corpus run-8 trained on (duck-sft v4's source set)
            "ahmedmobasher86/arc3-sft-k3-corpus",
        ],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        # arc3-deps-prep — offline transformers-main/peft for the merge
        # subprocess (round-4 abort: stock image transformers lacks qwen3_5).
        # The sft-synth kernel source is REMOVED so no second sft_out tree can
        # mount and collide with the checkpoint-8 glob.
        "kernel_sources": ["ahmedmobasher86/arc3-deps-prep"],
        "model_sources": [],
    }, indent=2) + "\n")
    print(f"wrote {OUT} ({len(nb['cells'])} cells, anchors {sorted(seen)})")
    print(f"push with: kaggle kernels push -p {OUT_DIR}")


if __name__ == "__main__":
    main()
