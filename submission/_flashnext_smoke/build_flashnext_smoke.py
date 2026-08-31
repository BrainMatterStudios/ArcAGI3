#!/usr/bin/env python3
"""build_flashnext_smoke.py — THE MODEL-SWAP READ (2026-08-30).

STOCK duck harness (anim-20260807 bundle, the bytes that flew 1.55) on all 25
public ARC-AGI-3 games, served by sonpham's Qwen3.8-Flash-Next NVFP4 package
instead of the Qwen3.8-27B-FP8 — the decisive test of the MODEL axis on the
scored GPU class.

Fusion of two PROVEN building blocks:
  * submission/_flashnext_gate/build_flashnext_gate.py — ASSEMBLE (runtime
    tarball + cu13 nvcc wiring + symlink-union model view) and BOOT (their
    launch argv verbatim, VLLM_PLE_CPU_OFFLOAD=1). This exact path BOOTED on
    Kaggle v4 in 740 s on rung "gcp_exact" (results_v4/). Reused VERBATIM via
    runpy — minus the load-generator/battery phases.
  * submission/_tp_smoke/build_tp_smoke.py — the v12-notebook-derived game
    cells: GAMES_25 from environment_files, bm.run at 7,920 s/game with the
    serialized solver's concurrency 28, per-phase telemetry reading vLLM
    /metrics, per-game results JSON. Minus every graft: the harness is STOCK.

What changes vs the duck38-v12 base notebook:
  * the 27B Kaggle Model is NOT attached; the v12 model-mount assert cell and
    the setup_commands vLLM-boot cell are REPLACED by the flashnext
    config/assemble/boot cells (bundle discovery + sys.path + deploy pkl cells
    kept verbatim).
  * duck analyzer env is exported by the notebook itself (reproducing the 27B
    bundle's setup_env exactly) with base URL/model pointed at the Flash-Next
    server, LOCAL_ANALYZER_CONTEXT_WINDOW=24576 and LOCAL_ANALYZER_MAX_OUTPUT=
    4096 (their server is --max-model-len 32768; the duck default 32768 window
    sends no max_tokens — 24576+4096 keeps prompt+completion under the cap).
    NOT exported: PYTHONPATH (the cu130 serving runtime must not shadow the
    notebook's harness deps; the harness talks HTTP via `requests`).
  * boot attestation asserts the Flash-Next NVFP4 signature
    (architectures Qwen4ExpForConditionalGeneration / model_type qwen4_exp,
    206 shards > 186 GB) instead of the 27B discriminators.

ONE phase: ("flashnext", GAMES_25, 7920). soft_end = NOTEBOOK_START_EPOCH +
11,400 s (assemble+boot ~25 min + one 2.2 h wave + teardown <= 3.2 h).

Pre-registered read (README.md): baseline = pooled stock 27B (3 kernels:
levels/game 1.00 / 1.04 / 0.84-0.88, zero-level 8-9/25, mean local score
3.5-4.9). PASS = levels/game >= 1.3 OR zero-level <= 6 with levels >= 1.0;
INCONCLUSIVE = levels 0.9-1.3; FAIL = levels < 0.9. Also record per-game
actions (expect MORE actions/game from the faster serving: gate conc-28 was
412 vs 297 gen tok/s).

Usage:
  .venv/bin/python submission/_flashnext_smoke/build_flashnext_smoke.py
  .venv/bin/python submission/_flashnext_smoke/validate_flashnext_smoke.py
  # push (ONLY on Ahmed's go):
  cd submission/_flashnext_smoke && python3 -m kaggle kernels push -p .
"""
import hashlib
import json
import runpy
from pathlib import Path

HERE = Path(__file__).parent
SUB = HERE.parent
ROOT = SUB.parent
BASE_NB = SUB / "_duck38_v12" / "arc3-duck38-v12.ipynb"
GATE_BUILD = SUB / "_flashnext_gate" / "build_flashnext_gate.py"
KERNEL_SLUG = "arc3-flashnext-smoke"

PER_GAME_S = 7920
SOFT_END_S = 18600          # boot (~1,000 s) + 2 x 7,920 s phases + slack

# ---- markers into the v12 base notebook ------------------------------------
MARK_IMPORTS = "NOTEBOOK_START_EPOCH = time.time()"
MARK_WHEELS = "arc_agi_3_wheels"
MARK_CONFIG = "# Qwen3.8 / Kaggle input configuration"
MARK_AUDIT = "# Audit the attached inputs"
MARK_SETUP = "_patch_qwen38_setup_commands"
MARK_ATTEST_CELL = "# Boot attestation"
MARK_SMOKE = "# Smoke/eval hook:"
MARK_RUN = "run_context = contextlib.nullcontext()"

GRAFT_SRC_DIR = SUB / "_throughput_v1"
GRAFT_FILES = ["graft_throughput.py", "graft_control.py", "frontier_explorer.py",
               "graft_explore.py", "graft_emission.py", "graft_economy.py"]

GRAFT_CELL_HEAD = r'''# ==== graft install (tuned phase machinery; every flag phase-controlled) ====
# All grafts install ONCE; behaviour is env-gated per phase (TP*_ENABLE).
# Defaults here are ALL OFF — the stock phase runs first.
for _flag in ("TP_ENABLE", "TP2_ENABLE", "TP4_ENABLE", "TP5_ENABLE", "TP6_ENABLE"):
    os.environ[_flag] = "0"
os.environ["TP5_WM_FROM_REASONING"] = "1"
os.environ["TP5_ACT_FLOOR"] = "3"
os.environ["TP4_STALL_T3"] = "30"
os.environ["TP4_BUDGET"] = "800"
os.environ["TP2_STALL_T2"] = "30"
'''

GRAFT_CELL_TAIL = r'''
_G_DIR = WORKING_DIR / "tuned_bundle"
_G_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in _GRAFT_SOURCES.items():
    (_G_DIR / _name).write_text(_src, encoding="utf-8")
if str(_G_DIR) not in sys.path:
    sys.path.insert(0, str(_G_DIR))
import importlib as _il
_tp = _il.import_module("graft_throughput"); _st = _tp.install()
assert _st == "throughput: OK", _st
_tc = _il.import_module("graft_control"); _st = _tc.install()
assert _st == "control: OK", _st
_te = _il.import_module("graft_explore"); _st = _te.install()
assert _st == "explore: OK", _st
_tm = _il.import_module("graft_emission"); _st = _tm.install()
assert _st == "emission: OK", _st
_t6 = _il.import_module("graft_economy"); _st = _t6.install()
assert _st == "economy: OK", _st
print("[tuned] all grafts installed; enabled:",
      {m.__name__.split("_")[1]: m.enabled() for m in (_tp, _tc, _te, _tm, _t6)})
'''

MARK_ATTEST = "attest: OK"

RUN_BLOCK_OLD = """    try:
        await bm.run(
            soft_end_time=soft_end,
            runtime_environment=target,
            minimal_diagnostics=run_as_submission,
        )
"""

RUN_BLOCK_NEW = '''    try:
        # flashnext smoke: ONE phase — the STOCK duck against the Flash-Next
        # server. The scored-rerun path takes exactly one pass with the
        # competition games (and would need the 27B kernel, not this one).
        for _phase_name, _phase_games, _phase_cap, _phase_env in SMOKE_PHASES:
            for _k, _v in _phase_env.items():
                os.environ[_k] = _v
            if not run_as_submission:
                FN_ALL_RUNS.extend(bm.game_runs)
                bm.game_runs = []
                bm.games = [GameAPI(env_name=_n, arcade_spec=_spec) for _n in _phase_games]
                bm.n_passes = 1
                bm.game_weights = None
                bm.label = "flashnext-smoke-" + _phase_name
                bm.solver.max_runtime_s_per_game = float(_phase_cap)
                print(f"=== PHASE {_phase_name}: {len(_phase_games)} games env={_phase_env} "
                      f"per_game_cap={_phase_cap}s concurrency={bm.solver.concurrency} "
                      f"model={os.environ.get('INFERENCE_ANALYZER_MODEL')} ===", flush=True)
            _fn_phase_begin(_phase_name)
            try:
                await bm.run(
                    soft_end_time=soft_end,
                    runtime_environment=target,
                    minimal_diagnostics=run_as_submission,
                )
            except Exception as _phase_exc:  # noqa: BLE001
                import traceback

                FN_PHASE_ERRORS.append(f"{_phase_name}: {type(_phase_exc).__name__}: {_phase_exc}")
                print(f"PHASE {_phase_name} RAISED {type(_phase_exc).__name__}: {_phase_exc}",
                      flush=True)
                traceback.print_exc()
            try:
                _fn_phase_end(_phase_name, list(bm.game_runs))
            except Exception:  # noqa: BLE001
                pass
            if run_as_submission:
                break
'''

GPU_ASSERT_CELL = r'''# Fail-fast GPU assert: metadata machine_shape + --accelerator alone can still
# bind P100; the competition source attachment is the real RTX Pro 6000 gate.
import subprocess as _sp

_gpu = _sp.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("boot gpu:", (_gpu.stdout or "").strip() or (_gpu.stderr or "").strip())
_gpu_name = (_gpu.stdout or "").upper()
assert "RTX" in _gpu_name and "6000" in _gpu_name, (
    f"GPU misbind — expected RTX Pro 6000, got: {_gpu.stdout!r} {_gpu.stderr!r}"
)
'''

MD_HEADER = """\
# arc3-flashnext-smoke — STOCK duck × Flash-Next NVFP4 (the model-swap read)

One phase, all 25 public games, eval geometry (7,920 s/game, concurrency 28
from the serialized solver), served by sonpham's **Qwen3.8-Flash-Next NVFP4**
package on ONE RTX Pro 6000 (gate-proven boot: 740 s on rung `gcp_exact`,
`submission/_flashnext_gate/results_v4/`). The harness is the **stock duck**
(anim-20260807 bundle): no grafts, `ONLY_RESET_LEVELS=true`, stock sampling
(temp 0.6 / top-p 0.95 / top-k 20).

**Deviation vs the 27B baseline:** `LOCAL_ANALYZER_CONTEXT_WINDOW=24576` +
`LOCAL_ANALYZER_MAX_OUTPUT=4096`, because their server serves
`--max-model-len 32768` (the 27B ran a 65536-ctx server with a 32768 window
and no max_tokens). Fair enough: the stock 27B's effective history is 4-9
turns anyway.

**Pre-registered read** — baseline = pooled stock 27B (3 kernels: levels/game
1.00 / 1.04 / 0.84-0.88, zero-level 8-9/25, mean local score 3.5-4.9):

* **PASS** = flashnext levels/game >= 1.3 OR zero-level <= 6 with levels >= 1.0
* **INCONCLUSIVE** = levels/game 0.9-1.3
* **FAIL** = levels/game < 0.9

Also record per-game actions (expect MORE actions/game from the faster
serving: gate conc-28 measured 412 vs the 27B's 297 gen tok/s aggregate) and
the server queue (28 workers vs `--max-num-seqs 22` — requests queue by
design; running/waiting/kv sampled every 60 s).
"""

CELL_CONFIG = r'''# ================= flashnext-smoke config — bundle, sys.path, results sink =================
# REPLACES the v12 "Qwen3.8 / Kaggle input configuration" cell. The 27B Kaggle
# Model is deliberately NOT attached: the model axis IS the experiment. The
# serving stack (sonpham's Flash-Next NVFP4 package) is assembled below.
import re
import shutil
import signal
import traceback
import urllib.request

WORKING_DIR = Path(os.getenv("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working")).resolve()
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"
SOFT_DEADLINE_BUFFER_S = 600.0
DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"

# Keep the whole run offline. vLLM/Transformers must use the mounted files only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
os.environ["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)


def _find_taaf_bundle() -> Path:
    explicit = os.getenv("TAAF_KAGGLE_BUNDLE_DIR", "").strip()
    # v2: sonpham's flashnext datasets ALSO contain a taaf-kaggle-bundle.json
    # (their apex fork). v1 rglobbed /kaggle/input and picked theirs first ->
    # their solver source + our pickle -> AttributeError hard_noop_guard.
    # Pin to the anim-20260807 bundle (the bytes the pickle was built from).
    for pinned in (Path("/kaggle/input/taaf-kaggle-source-anim-20260807-anim"),
                   Path("/kaggle/input/datasets/jakobbrggen/taaf-kaggle-source-anim-20260807-anim")):
        if (pinned / DATASET_BUNDLE_MARKER).is_file():
            return pinned
    if explicit and (Path(explicit) / DATASET_BUNDLE_MARKER).is_file():
        return Path(explicit)
    for root in [Path("/kaggle/input/datasets"), Path("/kaggle/input"), Path.cwd()]:
        if root.exists():
            for marker in root.rglob(DATASET_BUNDLE_MARKER):
                return marker.parent
    raise RuntimeError("Could not find TAAF Kaggle source bundle dataset.")


BUNDLE_DIR = _find_taaf_bundle()
TAAF_BUNDLE_DIR = BUNDLE_DIR    # survives the gate assemble cell, which REBINDS BUNDLE_DIR
os.environ["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
print(f"TAAF source bundle: {BUNDLE_DIR}")


# Make bundled TAAF repos importable for this notebook and child Python
# processes (verbatim from the v12 setup cell — the part we keep).
def _source_path_entries(bundle_dir: Path) -> list[Path]:
    src_root = bundle_dir / "src"
    if not src_root.is_dir():
        return []
    entries: list[Path] = []
    for repo in sorted(src_root.iterdir(), reverse=True):
        if not repo.is_dir():
            continue
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
if source_entries:
    import sysconfig

    pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
    pth_path.write_text("".join(f"{entry}\n" for entry in source_entries), encoding="utf-8")
    print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)", flush=True)

# ---- boot-results sink (flashnext-gate pattern): partial state always lands on disk ----
RESULTS_PATH = WORKING_DIR / "flashnext_smoke_boot.json"
RESULTS = {
    "meta": {"kernel": "arc3-flashnext-smoke",
             "started_utc": datetime.utcnow().isoformat() + "Z"},
    "boots": {},
    "phases": {},
    "verdicts": {},
}


def elapsed_min():
    return (time.time() - NOTEBOOK_START_EPOCH) / 60.0


def save_results():
    tmp = RESULTS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(RESULTS, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(RESULTS_PATH)


def host_snapshot(tag):
    snap = {}
    try:
        mem = dict((ln.split(":", 1)[0], ln.split(":", 1)[1].strip())
                   for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
        snap["mem_total"] = mem.get("MemTotal")
        snap["mem_available"] = mem.get("MemAvailable")
    except Exception as exc:
        snap["meminfo_error"] = repr(exc)[:120]
    for mount in ("/kaggle/working", "/kaggle/tmp", "/tmp", "/"):
        try:
            usage = shutil.disk_usage(mount)
            snap[mount] = f"free {usage.free / 1e9:.1f} / total {usage.total / 1e9:.1f} GB"
        except Exception:
            snap[mount] = "n/a"
    RESULTS["meta"].setdefault("host", {})[tag] = snap
    return snap


print("flashnext-smoke: host snapshot", json.dumps(host_snapshot("start"), indent=1), flush=True)
save_results()
'''

CELL_AUDIT = r'''# Audit the attached inputs that matter for this run. NOTE: no 27B model
# mount — the Kaggle models input dir is expected to be ABSENT.
INPUT_ROOT = Path("/kaggle/input")
print("=== TAAF bundle ===")
print(BUNDLE_DIR, "exists:", BUNDLE_DIR.exists())
for _needle in ("serving-part-000", "serving-part-001", "serving-part-002"):
    _hits = sorted({str(p) for p in INPUT_ROOT.rglob(_needle) if p.is_dir()})
    print(f"=== {_needle} ===", _hits)
_tarballs = sorted({str(p) for p in INPUT_ROOT.rglob("flashnext-gcp-container-site-packages.tar.zst")})
print("=== runtime tarball ===", _tarballs)
_wheels = [str(p) for p in (INPUT_ROOT / "arc3-qwen36-runtime-wheels",
                            INPUT_ROOT / "datasets" / "jcole75" / "arc3-qwen36-runtime-wheels")
           if p.is_dir()]
print("=== cu13 wheelhouse ===", _wheels)
_envdirs = [str(p) for p in (
    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
    Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files")) if p.is_dir()]
print("=== competition environment_files ===", _envdirs)
print("=== input models dir (should NOT exist) ===", (INPUT_ROOT / "models").exists())
_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print((_smi.stdout or "").strip()[:900], flush=True)
'''

CELL_DUCK_ENV = r'''# ============ duck analyzer env — the ONLY wiring change vs the 27B kernel ============
# v3: the gate ASSEMBLE cell rebinds BUNDLE_DIR to sonpham's source-bundle
# (their June-fork tree, which carries its own solver pkls WITHOUT
# the anim fields — v2 loaded it and died on hard_noop_guard). Re-pin the
# TAAF bundle before the deploy cell reads any pkl, and drop their tree from
# sys.path + sys.modules so the anim classes win.
BUNDLE_DIR = TAAF_BUNDLE_DIR
_sonpham_paths = [p for p in sys.path if "source-bundle" in p or "sonphamorg" in p]
for _p in _sonpham_paths:
    sys.path.remove(_p)
for _m, _mod in list(sys.modules.items()):
    if _m.split(".")[0] in ("inference", "taaf"):
        _f = getattr(_mod, "__file__", "") or ""
        if "source-bundle" in _f or "sonphamorg" in _f:
            del sys.modules[_m]
print("duck-env: BUNDLE_DIR re-pinned to", BUNDLE_DIR, "| pruned", len(_sonpham_paths), "paths")
assert (BUNDLE_DIR / "taaf-kaggle-bundle.json").is_file()
assert "anim-20260807" in str(BUNDLE_DIR), f"re-pin failed: {BUNDLE_DIR}"
# Reproduces the 27B bundle's exported setup_env EXACTLY (same keys,
# same stock sampling: temp 0.6 / top-p 0.95 / top-k 20 / thinking on) except:
#   * base URL + model id -> the Flash-Next server booted above (port 1234,
#     served name from their FLASH_SERVE_FLAGS);
#   * LOCAL_ANALYZER_CONTEXT_WINDOW 24576 + LOCAL_ANALYZER_MAX_OUTPUT 4096:
#     their server is --max-model-len 32768; the duck's default 32768 window
#     with no max_tokens could push prompt+completion past the server cap.
#     (The 27B baseline ran a 32768 window on a 65536-ctx server. Noted as a
#     deviation in the README; the stock 27B effective history is 4-9 turns.)
#   * PYTHONPATH NOT exported: the cu130 serving runtime must not shadow the
#     notebook's harness deps. The harness talks plain HTTP via `requests`;
#     only the vLLM server subprocess sees the extracted site-packages.
# These exports MUST land before the deploy pkls are loaded two cells down:
# tool_agent reads CONTEXT_WINDOW/MAX_OUTPUT at import time.
if BOOT_TAG is None:
    raise RuntimeError("no Flash-Next server — cannot wire the analyzer")
_booted_flags = CURRENT_SERVER["flags"]
_booted_ctx = int(_booted_flags[_booted_flags.index("--max-model-len") + 1])
if _booted_ctx >= 32768:
    _ctx_window, _max_out = "24576", "4096"
else:
    # reduced rung (16k server): shrink the window proportionally
    _ctx_window, _max_out = "12288", "2048"

duck_env = {
    "USE_TF": "0",
    "TRANSFORMERS_NO_TF": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "LOCAL_ANALYZER_BASE_URL": VLLM_API,
    "OPENAI_BASE_URL": VLLM_API,
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": QWEN_SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": QWEN_SERVED_MODEL_NAME,
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": _ctx_window,
    "LOCAL_ANALYZER_MAX_OUTPUT": _max_out,
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}
os.environ.update(duck_env)
SETUP_ENV_PATH.write_text(json.dumps(duck_env, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_shell_commands(filename, *, label, check):
    # Teardown shim: the v12 run cell calls this with teardown_commands.json,
    # which targets the 27B bundle's own vLLM pid file. This kernel's server
    # is ours — stop it directly and flush the results sink.
    print(f"taaf.kaggle: {label} ({filename}) -> flashnext stop_server", flush=True)
    try:
        stop_server(label)
    except Exception as exc:  # noqa: BLE001
        print("flashnext-smoke: teardown stop failed:", repr(exc)[:300], flush=True)
    save_results()


assert os.environ.get("INFERENCE_ANALYZER_MODEL") == QWEN_SERVED_MODEL_NAME
RESULTS["meta"]["duck_env"] = {k: v for k, v in duck_env.items()
                               if k.startswith(("LOCAL_", "INFERENCE_", "OPENAI_", "MULTIMODAL_"))}
save_results()
print("\n✅ duck analyzer wired to Flash-Next")
print("Analyzer endpoint:", os.environ["LOCAL_ANALYZER_BASE_URL"])
print("Analyzer model:   ", os.environ["INFERENCE_ANALYZER_MODEL"])
print("Context window:   ", _ctx_window, "| max output:", _max_out,
      "| server max-model-len:", _booted_ctx)
'''

CELL_ATTEST = r'''# Boot attestation (flashnext variant of doctrine v2, 2026-08-17): the
# assembled model view must be the RadixArk Flash-Next NVFP4 package and the
# LIVE server must answer through the duck's analyzer endpoint. Discriminators
# from the model card + the gate v4 Kaggle run: architectures
# Qwen4ExpForConditionalGeneration, model_type qwen4_exp, 206 safetensors
# > 186 GB (NVFP4 experts + BF16-converted PLE tables; the PLE conversion is
# hash-verified against FLASHNEXT_GCP_MODEL_INFO.json in the assemble cell).
# A wrong view must DIE here, before any game action is spent.
import hashlib as _hashlib
import urllib.request as _rq

_cfg_path = QWEN_MODEL_PATH / "config.json"
_cfg_raw = _cfg_path.read_bytes()
_cfg = json.loads(_cfg_raw)
assert _cfg.get("architectures") == ["Qwen4ExpForConditionalGeneration"], (
    f"attest FAIL: architectures {_cfg.get('architectures')}")
assert _cfg.get("model_type") == "qwen4_exp", (
    f"attest FAIL: model_type {_cfg.get('model_type')}")
print("attest: config sha256", _hashlib.sha256(_cfg_raw).hexdigest())

_shards = sorted(QWEN_MODEL_PATH.glob("*.safetensors"))
_total = sum(p.stat().st_size for p in _shards)
print(f"attest: {len(_shards)} shards, {_total} bytes total")
assert len(_shards) == 206, f"attest FAIL: shard count {len(_shards)} != 206"
assert _total > 186_000_000_000, f"attest FAIL: total shard bytes {_total} too small"

# Greedy decode fingerprint through the analyzer endpoint — logged (not
# asserted) for cross-run comparison, and proves the served model answers.
_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
if not _base.endswith("/v1"):
    _base += "/v1"
_body = json.dumps({
    "model": os.environ["INFERENCE_ANALYZER_MODEL"],
    "messages": [{"role": "user", "content": "Reply with exactly the sum of 17 and 25, then the word quack."}],
    "temperature": 0.0,
    "max_tokens": 48,
    "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
}).encode()
_req = _rq.Request(_base + "/chat/completions", data=_body, headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer " + (os.environ.get("LOCAL_ANALYZER_API_KEY") or "EMPTY"),
})
with _rq.urlopen(_req, timeout=300) as _resp:
    _reply = json.loads(_resp.read())["choices"][0]["message"].get("content") or ""
print("attest: decode fingerprint", repr(_reply)[:160])
print("attest: decode sha256", _hashlib.sha256(_reply.encode()).hexdigest())
print("attest: OK — Flash-Next NVFP4 signature + live analyzer endpoint verified before any game")
'''

SERVE_LIB_HEAD = r'''# ================= serve library — helpers + their launch argv (gate-proven) =================
# Minimal slice of the arc3-flashnext-gate serve chain: constants, HTTP/log/GPU
# helpers, and the server lifecycle (their serving_env + argv verbatim). The
# gate's load-generator/battery phases are NOT included — the load here is the
# real duck harness.
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 1234
VLLM_ROOT = f"http://{VLLM_HOST}:{VLLM_PORT}"
VLLM_API = VLLM_ROOT + "/v1"
VLLM_MAX_MODEL_LEN = 32768   # their launch_server value

CURRENT_SERVER = {"proc": None, "log": str(WORKING_DIR / "vllm-gcp_exact.log"),
                  "tag": "none", "flags": []}


def http_json(url, payload=None, timeout=120):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def server_alive(timeout=5):
    try:
        http_json(VLLM_API + "/models", timeout=timeout)
        return True
    except Exception:
        return False


def vllm_procs():
    out = subprocess.run(["pgrep", "-f", "vllm.entrypoints"], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        util, mem = out.stdout.strip().splitlines()[0].split(",")
        return {"util_pct": int(util.strip()), "mem_mib": int(mem.strip())}
    except Exception:
        return None


def tail_log_lines(path, max_bytes=524288):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace").splitlines()


'''


def _extract(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def _smoke_cell(games: list[str]) -> str:
    games_repr = json.dumps(games, indent=4)
    return f'''# Smoke/eval hook: THE MODEL-SWAP READ. A NORMAL COMMIT runs the STOCK duck
# on all 25 public games against the Flash-Next server, eval geometry
# ({PER_GAME_S} s per game, concurrency 28 from the serialized solver). One
# phase, no grafts, ONLY_RESET_LEVELS=true, stock sampling. The scored rerun
# path (KAGGLE_IS_COMPETITION_RERUN) never enters this branch.
GAMES_25 = {games_repr}
SMOKE_PHASES = [
    ("stock", GAMES_25, {PER_GAME_S}, {{"TP_ENABLE": "0", "TP2_ENABLE": "0", "TP4_ENABLE": "0",
                                        "TP5_ENABLE": "0", "TP6_ENABLE": "0"}}),
    # tuned = neutral Pack-1 base + emission (act-floor, wm-from-reasoning) +
    # explorer fallback + action-economy prompt. Composed arm: the goal is a
    # >=3-LB submission today, attribution later.
    ("tuned", GAMES_25, {PER_GAME_S}, {{"TP_ENABLE": "1", "TP_TRIM_LOW_WATER": "1.0",
                                        "TP_CONTEXT_WINDOW": "0", "TP_YIELD_SECONDS": "-1",
                                        "TP_TOOL_STEPS": "-1", "TP_BATCH_CAP": "0",
                                        "TP_KEEP_NOTES_ON_GAME_OVER": "1",
                                        "TP2_ENABLE": "0", "TP4_ENABLE": "1",
                                        "TP5_ENABLE": "1", "TP6_ENABLE": "1"}}),
]
FN_PHASE_ERRORS = []
FN_ALL_RUNS = []

if not run_as_submission:
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    def _resolve_env_dir():
        candidates = [
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
            Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
        ]
        for cand in candidates:
            if cand.is_dir():
                return str(cand)
        for hit in Path("/kaggle/input").rglob("environment_files"):
            if hit.is_dir():
                return str(hit)
        raise RuntimeError("environment_files dir not found in /kaggle/input")

    _env_dir = _resolve_env_dir()
    _spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_env_dir)
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in SMOKE_PHASES[0][1]]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = "flashnext-smoke-" + SMOKE_PHASES[0][0]
    bm.solver.max_runtime_s_per_game = float(SMOKE_PHASES[0][2])
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds={SOFT_END_S})
    print(f"smoke hook: phases={{[(p[0], len(p[1]), p[2]) for p in SMOKE_PHASES]}} "
          f"env_dir={{_env_dir}} concurrency={{bm.solver.concurrency}} "
          f"per_game_cap={{bm.solver.max_runtime_s_per_game}}s soft_end={{soft_end}}")
else:
    print("scored rerun: smoke hook inert — full competition games")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''


TELEMETRY_CELL = r'''# ---- flashnext smoke telemetry: THE READ. Per-game actions / levels / score /
# turns / tokens from the framework mirror + the ToolAgent session counters,
# vLLM /metrics deltas (prompt + generation tokens, prefix-cache queries/hits,
# preemptions), and a 60 s queue sampler: 28 client workers vs their
# --max-num-seqs 22 means requests queue by design — running/waiting/kv are
# part of the record.
import json as _tel_json
import re as _tel_re
import threading as _tel_threading
import time as _tel_time
import urllib.request as _tel_rq

from inference.framework import solver as _solver_mod

_tel_lock = _tel_threading.Lock()
_TEL_PATH = WORKING_DIR / "flashnext_smoke_results.json"
FN_SESSIONS = {}      # phase -> {game_id: {...}} filled at session exit
FN_PHASES = []        # ordered phase records
_FN_CURRENT = {"phase": None, "t0": None, "metrics0": None,
               "sampler_stop": None, "queue_samples": None}


def _metrics_url():
    base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/metrics"


_COUNTER_KEYS = (
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:request_success_total", "vllm:num_preemptions_total",
)
_GAUGE_KEYS = (
    "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def _fn_scrape(keys):
    out = {}
    try:
        with _tel_rq.urlopen(_metrics_url(), timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        for key in keys:
            # exact metric name only ("...waiting" must not match "...waiting_by_reason")
            if line.startswith(key + "{") or line.startswith(key + " "):
                m = _tel_re.match(r"^\S+(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
                if m:
                    try:
                        out[key] = out.get(key, 0.0) + float(m.group(1))
                    except ValueError:
                        pass
    return out


def _fn_phase_begin(phase):
    stop = _tel_threading.Event()
    samples = []

    def _queue_sampler():
        while not stop.is_set():
            g = _fn_scrape(_GAUGE_KEYS)
            if "error" not in g:
                g["t"] = round(_tel_time.time(), 1)
                samples.append(g)
            stop.wait(60)

    with _tel_lock:
        _FN_CURRENT.update({"phase": phase, "t0": _tel_time.monotonic(),
                            "metrics0": _fn_scrape(_COUNTER_KEYS),
                            "sampler_stop": stop, "queue_samples": samples})
        FN_SESSIONS.setdefault(phase, {})
    _tel_threading.Thread(target=_queue_sampler, daemon=True).start()
    print(f"[fn-tel] phase {phase} begin metrics0={_FN_CURRENT['metrics0']}", flush=True)


def _fn_phase_end(phase, game_runs):
    stop = _FN_CURRENT.get("sampler_stop")
    if stop is not None:
        stop.set()
    m1 = _fn_scrape(_COUNTER_KEYS)
    m0 = _FN_CURRENT.get("metrics0") or {}
    delta = {k: (m1.get(k, 0.0) - m0.get(k, 0.0)) for k in _COUNTER_KEYS if k in m1}
    wall = _tel_time.monotonic() - (_FN_CURRENT.get("t0") or _tel_time.monotonic())
    games = []
    for game_run in game_runs:
        apl = list(game_run.actions_per_level or [])
        actions = sum(apl) if apl else len(game_run.history)
        sess = FN_SESSIONS.get(phase, {}).get(game_run.game_id, {})
        games.append({
            "game_id": game_run.game_id,
            "state": game_run.state,
            "levels_completed": game_run.levels_completed,
            "number_of_levels": game_run.number_of_levels,
            "final_score": game_run.final_score,
            "actions": actions,
            "actions_per_level": apl,
            "base_actions_per_level": list(game_run.base_actions_per_level or []),
            "wallclock_s": game_run.final_wallclock_seconds,
            "solver_note": game_run.solver_note,
            "turns": sess.get("turns"),
            "session_total_tokens": sess.get("total_tokens"),
            "session_generated_tokens": sess.get("generated_tokens"),
            "history_messages_at_exit": sess.get("history_messages"),
            "context_budget_tokens": sess.get("context_budget_tokens"),
            "yield_seconds": sess.get("yield_seconds"),
            "tool_steps": sess.get("tool_steps"),
        })
    n = max(1, len(games))
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    hits = delta.get("vllm:prefix_cache_hits_total", 0.0)
    gen = delta.get("vllm:generation_tokens_total", 0.0)
    prompt = delta.get("vllm:prompt_tokens_total", 0.0)
    q = list(_FN_CURRENT.get("queue_samples") or [])
    running = [s["vllm:num_requests_running"] for s in q if "vllm:num_requests_running" in s]
    waiting = [s["vllm:num_requests_waiting"] for s in q if "vllm:num_requests_waiting" in s]
    kv = [s["vllm:kv_cache_usage_perc"] for s in q if "vllm:kv_cache_usage_perc" in s]
    rec = {
        "phase": phase,
        "harness": "stock-duck",
        "server_tag": CURRENT_SERVER["tag"],
        "analyzer_env": {k: v for k, v in os.environ.items()
                         if k.startswith(("LOCAL_ANALYZER", "INFERENCE_ANALYZER", "MULTIMODAL"))},
        "wall_s": round(wall, 1),
        "games": games,
        "n_games": len(games),
        "mean_actions": round(sum(g["actions"] for g in games) / n, 2),
        "mean_levels": round(sum(g["levels_completed"] for g in games) / n, 3),
        "mean_score": round(sum(float(g["final_score"] or 0.0) for g in games) / n, 4),
        "zero_level_games": sum(1 for g in games if g["levels_completed"] == 0),
        "mean_turns": round(sum(float(g["turns"] or 0) for g in games) / n, 1),
        "metrics_delta": delta,
        "prefix_hit_rate": round(hits / queries, 4) if queries else None,
        "prefill_per_gen_token": round(prompt / gen, 2) if gen else None,
        "gen_tok_s": round(gen / wall, 1) if wall else None,
        "queue": {
            "n_samples": len(q),
            "running_mean": round(sum(running) / len(running), 1) if running else None,
            "running_max": max(running) if running else None,
            "waiting_mean": round(sum(waiting) / len(waiting), 1) if waiting else None,
            "waiting_max": max(waiting) if waiting else None,
            "kv_usage_mean": round(sum(kv) / len(kv), 3) if kv else None,
            "kv_usage_max": round(max(kv), 3) if kv else None,
        },
    }
    with _tel_lock:
        FN_PHASES.append(rec)
    try:
        _TEL_PATH.write_text(_tel_json.dumps(
            {"phases": FN_PHASES, "phase_errors": FN_PHASE_ERRORS}, indent=1, default=str),
            encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(f"[fn-tel] phase {phase} end: games={rec['n_games']} mean_actions={rec['mean_actions']} "
          f"mean_levels={rec['mean_levels']} mean_score={rec['mean_score']} "
          f"zero_level={rec['zero_level_games']} turns={rec['mean_turns']} "
          f"prefix_hit={rec['prefix_hit_rate']} prefill/gen={rec['prefill_per_gen_token']} "
          f"gen_tok_s={rec['gen_tok_s']} queue={rec['queue']} wall={rec['wall_s']}s", flush=True)


# Session exit seam: record turns + token counters per game for the phase.
_inner_play = _solver_mod._HarnessGameSession.play


def _tel_play(self):
    try:
        return _inner_play(self)
    finally:
        try:
            gid = getattr(getattr(self.game, "game_run", None), "game_id", "?")
            an = self.analyzer
            rec = {
                "turns": int(getattr(self, "analysis_step", 0) or 0),
                "total_tokens": int(getattr(an, "_session_total_tokens", 0) or 0),
                "generated_tokens": int(getattr(an, "_session_generated_tokens", 0) or 0),
                "history_messages": len(getattr(an, "_history_messages", []) or []),
                "context_budget_tokens": getattr(an, "_context_budget_tokens", None),
                "yield_seconds": getattr(an, "_yield_seconds", None),
                "tool_steps": getattr(an, "_tool_steps", None),
            }
            with _tel_lock:
                FN_SESSIONS.setdefault(_FN_CURRENT.get("phase") or "?", {})[gid] = rec
        except Exception:  # noqa: BLE001
            pass


if not getattr(_solver_mod._HarnessGameSession.play, "_fn_tel", False):
    _tel_play._fn_tel = True
    _solver_mod._HarnessGameSession.play = _tel_play

print("[fn-tel] installed; metrics url =", _metrics_url(), flush=True)
'''

REPORT_CELL = r'''# ---- flashnext smoke final report (grep for FLASHNEXT SMOKE / READ) ----
BASELINE_27B_STOCK = {
    "source": "pooled stock phases of arc3-tp-smoke / arc3-tp1b-smoke / arc3-tp1c-smoke "
              "(27B-FP8, same GPU class, same geometry, 2026-08-29)",
    "levels_per_game": "1.00 / 1.04 / 0.84-0.88",
    "zero_level_games": "8-9 of 25",
    "mean_local_score": "3.5-4.9",
}
print("=" * 78)
print("FLASHNEXT SMOKE RESULTS — STOCK duck x Flash-Next NVFP4 (model-swap read)")
print("boot:", RESULTS["verdicts"].get("boot"))
print("27B stock baseline:", BASELINE_27B_STOCK)
by = {p["phase"]: p for p in FN_PHASES}
p = by.get("flashnext")
verdict = "UNREADABLE"
detail = ""
if not p or not p["n_games"]:
    print("PHASE flashnext: MISSING")
else:
    print(f"PHASE flashnext: games={p['n_games']} mean_actions={p['mean_actions']} "
          f"mean_levels={p['mean_levels']} mean_score={p['mean_score']} "
          f"zero_level={p['zero_level_games']} mean_turns={p['mean_turns']} "
          f"prefix_hit={p['prefix_hit_rate']} prefill/gen={p['prefill_per_gen_token']} "
          f"gen_tok_s={p['gen_tok_s']} wall={p['wall_s']}s")
    print(f"queue: {p['queue']}")
    for g in sorted(p["games"], key=lambda g: g["game_id"]):
        print(f"  {g['game_id']}: levels={g['levels_completed']}/{g['number_of_levels']} "
              f"score={g['final_score']} actions={g['actions']} turns={g['turns']} "
              f"gen_tok={g['session_generated_tokens']} state={g['state']}")
    lev = p["mean_levels"]
    zero = p["zero_level_games"]
    detail = (f"levels {lev:.3f} (27B stock 1.00/1.04/0.84-0.88) zero_level {zero}/25 "
              f"(27B 8-9/25) actions {p['mean_actions']} score {p['mean_score']}")
    # Pre-registered rules (README.md, fixed before flight):
    if lev >= 1.3 or (zero <= 6 and lev >= 1.0):
        verdict = "PASS"
    elif lev < 0.9:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
print(f"FLASHNEXT SMOKE READ: {verdict} ({detail}) phase_errors={FN_PHASE_ERRORS}")
results = {
    "kernel": "arc3-flashnext-smoke",
    "harness": "stock-duck",
    "boot": RESULTS["verdicts"].get("boot"),
    "baseline_27b_stock": BASELINE_27B_STOCK,
    "phases": FN_PHASES,
    "phase_errors": FN_PHASE_ERRORS,
    "verdict": verdict,
    "detail": detail,
}
(WORKING_DIR / "flashnext_smoke_results.json").write_text(
    json.dumps(results, indent=1, default=str), encoding="utf-8")
print("wrote", WORKING_DIR / "flashnext_smoke_results.json")
'''

BOOT_RAISE_SUFFIX = '''

# The smoke is unreadable without a live server — die loudly here rather than
# spend three hours playing games against a dead endpoint.
if BOOT_TAG is None:
    raise RuntimeError("Flash-Next server did not boot on any rung — smoke aborted")
'''


def public_games() -> list[str]:
    env_root = ROOT / "environment_files"
    names = []
    for game_dir in sorted(p for p in env_root.iterdir() if p.is_dir()):
        hashes = sorted(p.name for p in game_dir.iterdir() if p.is_dir())
        assert len(hashes) == 1, (game_dir, hashes)
        names.append(f"{game_dir.name}-{hashes[0]}")
    assert len(names) == 25, len(names)
    return names


def build_notebook() -> dict:
    gate = runpy.run_path(str(GATE_BUILD))
    nb = json.loads(BASE_NB.read_text())

    # ---- gate pieces, VERBATIM (assemble + serve flags + server lifecycle + boot) ----
    assemble_cell = (gate["CELL_ASSEMBLE"]
                     .replace("@@RUNTIME_SHA@@", gate["EXPECTED_RUNTIME_SHA256"])
                     .replace("@@ZSTD_SHA@@", gate["EXPECTED_ZSTD_SHA256"])
                     .replace("@@VERSIONS_LINE@@", gate["EXPECTED_VERSIONS_LINE"])
                     .replace("@@RUNTIME_ARCHIVE@@", gate["RUNTIME_ARCHIVE"])
                     .replace("@@MODEL_ID@@", gate["MODEL_ID"])
                     .replace("@@MODEL_REVISION@@", gate["MODEL_REVISION"])
                     .replace("@@EXPECTED_SAFETENSOR_FILES@@", str(gate["EXPECTED_SAFETENSOR_FILES"]))
                     .replace("@@EXPECTED_PAYLOAD_MIN_BYTES@@", str(gate["EXPECTED_PAYLOAD_MIN_BYTES"])))
    assert "@@" not in assemble_cell
    flags_block = _extract(gate["LIB_CONSTANTS"],
                           "# ---- their launch_server argv VERBATIM", "BASELINE_27B =")
    serve_cell = SERVE_LIB_HEAD + flags_block + "\n" + gate["LIB_SERVER"]
    boot_cell = gate["CELL_BOOT"] + BOOT_RAISE_SUFFIX
    # gate prints say "flashnext-gate:"; this kernel is the smoke
    assemble_cell = assemble_cell.replace("flashnext-gate:", "flashnext-smoke:")
    serve_cell = serve_cell.replace("flashnext-gate:", "flashnext-smoke:")
    boot_cell = boot_cell.replace("flashnext-gate:", "flashnext-smoke:")

    games = public_games()

    def idx_of(marker: str) -> int:
        hits = [i for i, c in enumerate(nb["cells"])
                if c["cell_type"] == "code" and marker in "".join(c["source"])]
        assert len(hits) == 1, (marker, len(hits))
        return hits[0]

    def code_cell(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": text.splitlines(keepends=True)}

    def replace_cell(marker: str, text: str) -> None:
        nb["cells"][idx_of(marker)]["source"] = text.splitlines(keepends=True)

    # 0) markdown header
    assert nb["cells"][0]["cell_type"] == "markdown"
    nb["cells"][0]["source"] = MD_HEADER.splitlines(keepends=True)

    # 1) fail-fast GPU assert straight after the imports cell
    nb["cells"].insert(idx_of(MARK_IMPORTS) + 1, code_cell(GPU_ASSERT_CELL))

    # 2) v12 model/config cell -> flashnext config (bundle + sys.path + sink)
    replace_cell(MARK_CONFIG, CELL_CONFIG)

    # 3) v12 input audit -> flashnext mounts audit
    replace_cell(MARK_AUDIT, CELL_AUDIT)

    # 4) v12 setup_commands vLLM boot -> gate ASSEMBLE; then serve lib + BOOT +
    #    duck env exports inserted right after (before the attestation cell)
    setup_idx = idx_of(MARK_SETUP)
    nb["cells"][setup_idx]["source"] = assemble_cell.splitlines(keepends=True)
    nb["cells"].insert(setup_idx + 1, code_cell(serve_cell))
    nb["cells"].insert(setup_idx + 2, code_cell(boot_cell))
    nb["cells"].insert(setup_idx + 3, code_cell(CELL_DUCK_ENV))

    # 5) 27B attestation -> flashnext attestation
    replace_cell(MARK_ATTEST_CELL, CELL_ATTEST)

    # 6) smoke hook -> single flashnext phase
    replace_cell(MARK_SMOKE, _smoke_cell(games))

    # 7) telemetry before the run cell; run block swapped; report appended
    run_idx = idx_of(MARK_RUN)
    run_src = "".join(nb["cells"][run_idx]["source"])
    assert run_src.count(RUN_BLOCK_OLD) == 1, "base run cell drifted"
    run_src = run_src.replace(RUN_BLOCK_OLD, RUN_BLOCK_NEW)
    nb["cells"][run_idx]["source"] = run_src.splitlines(keepends=True)
    graft_sources = {name: (GRAFT_SRC_DIR / name).read_text() for name in GRAFT_FILES}
    for name, text in graft_sources.items():
        assert "def install() -> str:" in text or "class FrontierExplorer" in text, name
    graft_lines = ["_GRAFT_SOURCES = {\n"]
    for name in GRAFT_FILES:
        graft_lines.append(f"    {name!r}: {graft_sources[name]!r},\n")
    graft_lines.append("}\n")
    nb["cells"].insert(run_idx, code_cell(GRAFT_CELL_HEAD + "".join(graft_lines) + GRAFT_CELL_TAIL))
    run_idx = idx_of(MARK_RUN)
    nb["cells"].insert(run_idx, code_cell(TELEMETRY_CELL))
    nb["cells"].insert(idx_of(MARK_RUN) + 1, code_cell(REPORT_CELL))

    # ---- invariants ----------------------------------------------------------
    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    # ordering: GPU gate -> assemble/boot -> analyzer env -> attest -> games
    assert joined.index("GPU misbind") < joined.index("Phase 1 — ASSEMBLE")
    assert joined.index("Phase 1 — ASSEMBLE") < joined.index("Phase 2 — BOOT")
    assert joined.index("Phase 2 — BOOT") < joined.index("duck analyzer env")
    assert joined.index("duck analyzer env") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index("SMOKE_PHASES = [")
    assert joined.index("[fn-tel] installed") < joined.index(MARK_RUN)
    assert joined.index(MARK_RUN) < joined.index("FLASHNEXT SMOKE RESULTS")
    # the serve chain is theirs, verbatim
    for needed in ('"VLLM_PLE_CPU_OFFLOAD": "1"', '"VLLM_PLE_OFFLOAD_READY_TIMEOUT": "1800"',
                   '"--tensor-parallel-size", "1"', '"--gpu-memory-utilization", "0.96"',
                   '"--max-model-len", "32768"', '"--max-num-seqs", "22"',
                   '"--max-num-batched-tokens", "6144"', '"--tool-call-parser", "qwen3_xml"',
                   '"--enable-prefix-caching"', '"--no-enable-flashinfer-autotune"',
                   gate["EXPECTED_RUNTIME_SHA256"], gate["EXPECTED_ZSTD_SHA256"],
                   gate["EXPECTED_VERSIONS_LINE"], gate["MODEL_ID"], gate["MODEL_REVISION"],
                   "serving-part-000", "source-bundle/zstd", "FLASHNEXT_GCP_MODEL_INFO.json",
                   "ple-bf16-conversion.json", "arc3-qwen36-runtime-wheels",
                   "Qwen4ExpForConditionalGeneration", "qwen4_exp"):
        assert needed in joined, f"missing from kernel: {needed}"
    # the harness is stock and the 27B is gone
    # (v5) the tuned phase deliberately embeds the grafts; the old stock-only
    # purity ban is replaced by phase-gating asserts.
    assert '("stock", GAMES_25' in joined and '("tuned", GAMES_25' in joined
    assert '"TP_ENABLE": "0"' in joined and '"TP5_ENABLE": "1"' in joined
    # stock analyzer knobs + the two deviations, exported before the deploy pkls
    for needed in ('"LOCAL_ANALYZER_TEMPERATURE": "0.6"', '"LOCAL_ANALYZER_TOP_P": "0.95"',
                   '"LOCAL_ANALYZER_TOP_K": "20"', '"LOCAL_ANALYZER_ENABLE_THINKING": "true"',
                   '"LOCAL_ANALYZER_PROVIDER": "vllm"', '"LOCAL_ANALYZER_YIELD_SECONDS": "60"',
                   '"LOCAL_ANALYZER_TOOL_STEPS": "0"', '"24576", "4096"',
                   'os.environ["ONLY_RESET_LEVELS"] = "true"'):
        assert needed in joined, f"missing analyzer wiring: {needed}"
    assert joined.index('"LOCAL_ANALYZER_TEMPERATURE": "0.6"') < joined.index("benchmark_initial")
    # scored-rerun branch intact; one phase; every game exactly once
    assert "bm.games = _competition_games()" in joined
    assert joined.count(", GAMES_25, ") == 2
    for name in games:
        assert joined.count(name) == 1, name
    assert "vllm:prefix_cache_hits_total" in joined and "vllm:num_requests_waiting" in joined
    return nb


def main() -> None:
    nb = build_notebook()
    nb_path = HERE / f"{KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "machine_shape": "NvidiaRtxPro6000",
        "dataset_sources": [
            "jcole75/arc3-qwen36-runtime-wheels",
            "sonphamorg/arc3-flashnext-serving-part-a-v1",
            "sonphamorg/arc3-flashnext-serving-part-b-v1",
            "sonphamorg/arc3-flashnext-serving-part-c-v1",
            "sonphamorg/arc3-flashnext-gcp-runtime-exact-v1",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
        ],
        "kernel_sources": [],
        # THE RTX Pro 6000 GATE (08-22 push lesson): the competition source is
        # what admits the kernel to the scored GPU pool.
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        # NO model_sources: the 27B mount is deliberately absent — the model
        # axis is the experiment; flashnext ships as the datasets above.
        "model_sources": [],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest()[:16])
    print("cells:", len(nb["cells"]), "| notebook:", nb_path, "|", nb_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
