#!/usr/bin/env python3
"""build_sc25_thresh.py — SC25 TOKEN-THRESHOLD A/B (55k-vs-110k), zero slots.

MOTIVATION (docs/RESEARCH-2026-08-22-slotmath-and-top3.md, Addendum 2): sc25
is TOKEN-graded, not config-graded — every observed failure sat <63k generated
tokens/session, both successes >=70k. Hypothesis: sc25-class games flip from
~0% to 30-50% success once a session passes ~70k generated tokens. If real,
the production lever is BUDGET ALLOCATION (triage + concurrency shaping),
priced ~+0.57 per rescued sc25-class game. This kernel is the family's direct
falsifier.

DESIGN — one kernel (ahmedmobasher86/arc3-sc25-thresh), duck38-v12 bundle
(jakobbrggen/taaf-kaggle-source-anim-20260807-anim) + EFFORT_MEDIUM=1 /
EFFORT_DEAD_RETRY=0 (the validated baseline: effort smoke 13/13 offline +
live smoke, dead-decode 0/212) on BOTH arms, sequential on one Qwen3.8-FP8
vLLM serve:

  ARM A (starved): 6 offline sc25-635fd71a sessions CONCURRENTLY (conc 6),
      token-boxed at ~55k/session.
  ARM B (rich):    6 sc25 sessions at conc 2 in three sequential waves of 2,
      token-boxed at >=110k/session.

THROUGHPUT CALIBRATION (same-rig measurement, NOT the tasking's assumption):
the dc22-ab kernel (2026-08-22, RTX Pro 6000, same serve chain, both bundles)
delivered a measured ~1430 tok/min/session at conc 8 over a 60-min box
(A: 78.7-90.8k, B: 78.6-93.0k per session; boot only 6.4 min). At conc 6 a
65-min clock box would therefore land ~110k tokens/session — NOT the 50-60k
the tasking predicted (that arithmetic came from the conc-28 132-min wave's
460 tok/min and the ~1000 tok/min screens on other loads). A pure clock box
would null the contrast mechanically. Fix: BOTH arms are TOKEN-boxed via a
graceful early stop (solver._stop_event — the same seam the soft-end
cancellation uses; sessions end state "cancelled" with fully readable rows,
proven on dc22-ab):
  ARM A: clock backstop 60 min, early stop when MEAN tokens/session >= 56k
         => predicted 50-62k/session, wall ~32-36 min at ~1500-1800
         tok/min/session (conc-6 interpolation of the conc-8 measurement).
  ARM B: 75-min wave boxes, early stop when EVERY session >= 112k or is
         terminal => predicted 105-125k/session, wall ~45-75 min/wave at
         ~1500-2500 tok/min/session (conc-2 is at or above the conc-8 rate).
Budget: boot ~7-10 min + A ~35 + 3 waves x ~55-77 + slack => ~3.6-4.7h,
hard planning cap 5h (waves are skipped LOUDLY if the cap would be crossed).
This exceeds the tasking's 3.3h estimate because its "3x35-min waves"
arithmetic cannot deliver >=100k tokens/session at any measured rate; tokens
are the experiment, so the wall stretches. Trimming sessions would not
shorten waves, so the 6v6 geometry is kept.

PRE-REGISTERED READING (from the tasking):
  REAL  = B reaches L1+ on >=3/6 (>=3/5 if a wave is lost) while A <=1/6
          => threshold real; production lever = budget allocation, ~+0.57
          per rescued sc25-class game.
  DEAD  = A ~ B (B_L1 <= A_L1 + 1) => the threshold hypothesis dies.
  else INCONCLUSIVE.
  VALIDITY: A sessions must land <=63k tokens (the observed failure band);
  an A session >70k is contamination. B sessions must land >=90k (target
  110k+); a B session <70k never tested the hypothesis. A read where the
  token geometry failed is declared VOID/degraded, not spun.

NOTE on pickle: deploy_target.pkl / benchmark_initial.pkl are the audited
bundle format every scored kernel uses, loaded from our own attested dataset
— same load path as dc22-ab and all duck38-v12 arms.

Usage:
  python3 submission/_sc25_thresh/build_sc25_thresh.py
  cd submission/_sc25_thresh && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
KERNEL_SLUG = "arc3-sc25-thresh"
GRAFT_PY = HERE.parent / "_effort_medium" / "graft_effort.py"

ANIM_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
MODEL_SOURCE = "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
COMPETITION = "arc-prize-2026-arc-agi-3"

MD_INTRO = """\
# arc3-sc25-thresh — SC25 token-threshold A/B (55k vs 110k)

**Not a submission candidate.** Direct falsifier for the token-budget-
reallocation family: do sc25-class games flip from ~0% to 30-50% success once
a session passes ~70k generated tokens? ARM A = 6 sc25 sessions starved at
~55k tokens (conc 6), ARM B = 6 sc25 sessions fed >=110k tokens (conc 2,
three waves), same duck38-v12 bundle + EFFORT_MEDIUM=1 baseline, same
Qwen3.8-FP8 serve, same GPU session. Token-boxed via graceful early stop.

Pre-registered: B >=3/6 at L1+ while A <=1/6 -> threshold REAL (lever =
budget allocation, ~+0.57/rescued game); A ~ B -> hypothesis dies.
"""

CELL_ENV = r'''# 1. Environment, submission guard, P100 fail-fast.
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
assert not TRUE_SUBMISSION, "arc3-sc25-thresh is a falsifier kernel, never a submission candidate"
NOTEBOOK_START_EPOCH = time.time()

os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"
os.environ["ONLY_RESET_LEVELS"] = "true"
# The validated effort baseline rides BOTH arms (single shared setting, not a
# variable of this A/B): reasoning_effort=medium, retry hardening OFF.
os.environ["EFFORT_MEDIUM"] = "1"
os.environ["EFFORT_DEAD_RETRY"] = "0"

cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# P100 fail-fast: metadata machine_shape + --accelerator alone can still bind
# P100; the competition source is the real RTX Pro 6000 gate. Die at 5s, not
# after boot.
_gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
_gpu_names = [line.strip() for line in _gpu.stdout.splitlines() if line.strip()]
assert _gpu_names and all("rtx pro 6000" in name.lower() for name in _gpu_names), (
    f"FAIL-FAST: expected RTX Pro 6000, got {_gpu_names!r} (rc={_gpu.returncode}, err={_gpu.stderr.strip()!r})"
)
print(f"sc25-thresh: GPU OK: {_gpu_names}")
'''

CELL_WHEELS = r'''# 2. Install the ARC runtime from the offline competition wheelhouse.
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-index",
        "--no-warn-conflicts", "--disable-pip-version-check", "--find-links",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)
print("sc25-thresh: arc-agi installed from competition wheels")
'''

CELL_MOUNTS = r'''# 3. Resolve the Qwen3.8 model mount and the duck38-v12 (anim) bundle.
ANIM_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"  # duck38-v12 bundle, BOTH arms
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"

QWEN_MODEL_OWNER = "foysalemonshanto"
QWEN_MODEL_SLUG = "qwen3-8-27b-fp8-repacked-v1"
QWEN_MODEL_REF = f"{QWEN_MODEL_OWNER}/{QWEN_MODEL_SLUG}"
QWEN_MODEL_FRAMEWORK = "pytorch"
QWEN_MODEL_VARIATION = "hf-fp8"
QWEN_MODEL_VERSION = "1"
QWEN_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"
DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"

# Serve-chain fingerprints (asserted identical June-vs-anim on dc22-ab live):
EXPECTED_SETUP_MD5 = "99e4b35dce683cea36db30e1dc6ef880"
EXPECTED_TEARDOWN_MD5 = "3eca3c6fc3e64bbc18cc6d396ae7ff9a"


def _model_mount_candidates() -> list:
    roots = [Path("/kaggle/input/models") / QWEN_MODEL_OWNER / QWEN_MODEL_SLUG,
             Path("/kaggle/input/models") / QWEN_MODEL_SLUG,
             Path("/kaggle/input") / QWEN_MODEL_SLUG]
    frameworks = [QWEN_MODEL_FRAMEWORK, QWEN_MODEL_FRAMEWORK.capitalize(), "PyTorch"]
    seen, out = set(), []
    for root in roots:
        for framework in frameworks:
            candidate = root / framework / QWEN_MODEL_VARIATION / QWEN_MODEL_VERSION
            if str(candidate) not in seen:
                seen.add(str(candidate))
                out.append(candidate)
    return out


_qwen_candidates = _model_mount_candidates()
QWEN_MODEL_PATH = next((c for c in _qwen_candidates if c.is_dir()), None)
if QWEN_MODEL_PATH is None:
    raise FileNotFoundError(
        "Qwen3.8 Kaggle Model is not attached. Tried:\n  " + "\n  ".join(str(c) for c in _qwen_candidates)
    )
_qwen_required = ["config.json", "model.safetensors.index.json", "tokenizer.json",
                  "tokenizer_config.json", "chat_template.jinja"]
_qwen_missing = [n for n in _qwen_required if not (QWEN_MODEL_PATH / n).is_file()]
if _qwen_missing:
    raise FileNotFoundError(f"Qwen3.8 mount {QWEN_MODEL_PATH} incomplete; missing: {', '.join(_qwen_missing)}")
_qwen_shards = sorted(QWEN_MODEL_PATH.glob("*.safetensors"))
if len(_qwen_shards) != 18:
    raise RuntimeError(f"Unexpected Qwen3.8 layout: {len(_qwen_shards)} safetensors files, expected 18.")
print(f"sc25-thresh: qwen3.8 model = {QWEN_MODEL_PATH} ({len(_qwen_shards)} shards)")


def _dataset_mount_candidates(ref: str) -> list:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _resolve_bundle(ref: str) -> Path:
    for cand in _dataset_mount_candidates(ref):
        if (cand / DATASET_BUNDLE_MARKER).is_file():
            return cand
    raise FileNotFoundError(f"bundle {ref} not mounted with marker; tried {_dataset_mount_candidates(ref)}")


ANIM_BUNDLE_DIR = _resolve_bundle(ANIM_REF)
print(f"sc25-thresh: duck38-v12 bundle = {ANIM_BUNDLE_DIR}")
print("sc25-thresh: bundle marker = "
      + json.dumps(json.loads((ANIM_BUNDLE_DIR / DATASET_BUNDLE_MARKER).read_text())))
_setup_md5 = hashlib.md5((ANIM_BUNDLE_DIR / "setup_commands.json").read_bytes()).hexdigest()
_teardown_md5 = hashlib.md5((ANIM_BUNDLE_DIR / "teardown_commands.json").read_bytes()).hexdigest()
print(f"sc25-thresh: setup md5={_setup_md5} teardown md5={_teardown_md5}")
assert _setup_md5 == EXPECTED_SETUP_MD5, "setup_commands.json drifted from the audited serve chain"
assert _teardown_md5 == EXPECTED_TEARDOWN_MD5, "teardown_commands.json drifted"

_wheelhouse_dir = next((c for c in _dataset_mount_candidates(WHEELHOUSE_REF) if c.exists()), None)
assert _wheelhouse_dir is not None, "vLLM wheelhouse dataset not mounted"

kaggle_input_paths = {
    ANIM_REF: str(ANIM_BUNDLE_DIR),
    WHEELHOUSE_REF: str(_wheelhouse_dir),
    QWEN_MODEL_REF: str(QWEN_MODEL_PATH),
}
setup_env = {
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps([ANIM_REF, WHEELHOUSE_REF]),
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps([]),
    "TAAF_QWEN_MODEL_PATH": str(QWEN_MODEL_PATH),
    "TAAF_QWEN_SERVED_MODEL_NAME": QWEN_SERVED_MODEL_NAME,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\n")
print(f"sc25-thresh: input paths = {setup_env['TAAF_KAGGLE_INPUT_PATHS']}")
'''

CELL_SERVE = r'''# 4. Patch the bundled setup's model identity to Qwen3.8 and boot vLLM ONCE.
# (Identical mechanism to pack-v22 / duck38-v12 / dc22-ab.) The blob pins
# LOCAL_ANALYZER_TEMPERATURE=0.6 and MULTIMODAL_UPSCALE=4 — both arms inherit.
import re as _re


def _command_env() -> dict:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(ANIM_BUNDLE_DIR)
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
    env.update({str(k): str(v) for k, v in json.loads(SETUP_ENV_PATH.read_text()).items()})
    return env


def _replace_python_assignment(command: str, variable_name: str, value: str):
    pattern = rf"(?m)^{_re.escape(variable_name)}\s*=\s*(['\"])[^\r\n]*?\1\s*$"
    return _re.subn(pattern, f"{variable_name} = {value!r}", command, count=1)


def _patch_qwen38_setup_commands(commands: list) -> list:
    replacements = {
        "MODEL_OWNER": QWEN_MODEL_OWNER,
        "MODEL_SLUG": QWEN_MODEL_SLUG,
        "SERVED_MODEL_NAME": QWEN_SERVED_MODEL_NAME,
    }
    counts = {name: 0 for name in replacements}
    patched = []
    for raw in commands:
        command = str(raw)
        for name, value in replacements.items():
            command, n = _replace_python_assignment(command, name, value)
            counts[name] += n
        if "def vllm_env()" in command and "'VLLM_NO_USAGE_STATS': '1'," in command:
            command = command.replace(
                "'VLLM_NO_USAGE_STATS': '1',",
                "'VLLM_NO_USAGE_STATS': '1',\n"
                "            'HF_HUB_OFFLINE': '1',\n"
                "            'TRANSFORMERS_OFFLINE': '1',",
                1,
            )
        patched.append(command)
    missing = [name for name, n in counts.items() if n == 0]
    if missing:
        raise RuntimeError("Could not repoint the bundled setup at Qwen3.8; missing: " + ", ".join(missing))
    print(f"sc25-thresh: qwen3.8 setup patch = {counts}")
    return patched


env = _command_env()
_setup_commands = _patch_qwen38_setup_commands(json.loads((ANIM_BUNDLE_DIR / "setup_commands.json").read_text()))
for command in _setup_commands:
    print(f"sc25-thresh: setup command: {command[:160]}...", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    env = _command_env()
    os.environ.update(env)

_served = os.environ.get("INFERENCE_ANALYZER_MODEL", "")
if _served != QWEN_SERVED_MODEL_NAME:
    raise RuntimeError(f"setup completed but analyzer model id is {_served!r}, expected {QWEN_SERVED_MODEL_NAME!r}")
print(f"SC25THRESH_MODEL_PIN SERVED_MODEL_NAME={_served} MODEL_PATH={QWEN_MODEL_PATH}")
print(f"SC25THRESH_CONFIG temp={os.environ.get('LOCAL_ANALYZER_TEMPERATURE')} "
      f"upscale={os.environ.get('MULTIMODAL_UPSCALE')} "
      f"multimodal={os.environ.get('MULTIMODAL_CONTEXT')}")
assert os.environ.get("LOCAL_ANALYZER_TEMPERATURE") == "0.6", "config drift: temperature != 0.6"
assert os.environ.get("MULTIMODAL_UPSCALE") == "4", "config drift: upscale != 4"
'''

CELL_ATTEST = r'''# 5. Boot attestation (doctrine v2, verbatim): the mounted weights must be
# the OFFICIAL Qwen3.8-FP8. A wrong mount must DIE here.
import hashlib as _hashlib
import urllib.request as _rq

_cfg_path = QWEN_MODEL_PATH / "config.json"
_cfg_raw = _cfg_path.read_bytes()
_cfg = json.loads(_cfg_raw)
_q = _cfg.get("quantization_config") or {}
assert _cfg.get("architectures") == ["Qwen3_5ForConditionalGeneration"], (
    f"attest FAIL: architectures {_cfg.get('architectures')}")
assert _q.get("quant_method") == "fp8" and _q.get("fmt") == "e4m3", (
    f"attest FAIL: quantization_config is not official fp8/e4m3: {_q}")
assert _cfg.get("transformers_version") == "5.8.0.dev0", (
    f"attest FAIL: transformers_version {_cfg.get('transformers_version')} "
    "(vrfai 3.6 stamps 5.6.2)")
print("attest: config sha256", _hashlib.sha256(_cfg_raw).hexdigest())

_idx_path = QWEN_MODEL_PATH / "model.safetensors.index.json"
if _idx_path.is_file():
    print("attest: index sha256", _hashlib.sha256(_idx_path.read_bytes()).hexdigest())
_shards = sorted(QWEN_MODEL_PATH.glob("*.safetensors"))
assert _shards, "attest FAIL: no safetensors shards at model path"
_total = sum(p.stat().st_size for p in _shards)
print(f"attest: {len(_shards)} shards, {_total} bytes total")
assert _total > 25_000_000_000, f"attest FAIL: total shard bytes {_total} too small for 27B FP8"
_h = _hashlib.sha256()
with open(_shards[0], "rb") as _f:
    _h.update(_f.read(1 << 20))
print("attest: first-shard-1MiB sha256", _h.hexdigest())

_base = (os.environ.get("LOCAL_ANALYZER_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
if not _base.endswith("/v1"):
    _base += "/v1"
_body = json.dumps({
    "model": QWEN_SERVED_MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly the sum of 17 and 25, then the word quack."}],
    "temperature": 0.0,
    "max_tokens": 48,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
_req = _rq.Request(_base + "/chat/completions", data=_body, headers={
    "Content-Type": "application/json",
    "Authorization": "Bearer " + (os.environ.get("LOCAL_ANALYZER_API_KEY") or "EMPTY"),
})
with _rq.urlopen(_req, timeout=180) as _resp:
    _reply = json.loads(_resp.read())["choices"][0]["message"].get("content") or ""
print("attest: decode fingerprint", repr(_reply)[:160])
print("attest: decode sha256", _hashlib.sha256(_reply.encode()).hexdigest())
print("attest: OK — official Qwen3.8-FP8 signature verified before any game")
'''

# The per-arm runner: executed as a subprocess (dc22-ab chassis + effort graft
# + token-boxed early stop). Embedded into the notebook via repr().
ARM_RUNNER = r'''"""sc25_arm_runner.py — one token-boxed batch of concurrent sc25 sessions.

argv: bundle_dir tag out_json box_minutes n_sessions stop_mode stop_tokens
  stop_mode "mean": graceful early stop when MEAN generated tokens/session
      >= stop_tokens (the STARVED arm — lands every session in a tight band
      around the target).
  stop_mode "min": graceful early stop when EVERY session has generated
      >= stop_tokens or has left the "playing" state (the RICH arm).
Early stop = solver._stop_event.set() — the exact seam bm.run's own soft-end
cancellation uses (solver.py:1260 "cancelled"); rows stay fully readable
(proven live on dc22-ab). Per-session generated tokens = sum over the
benchmark history (h.generated_tokens), the same read dc22-ab shipped.
Installs the EFFORT_MEDIUM graft (validated baseline) before any game and
dies if it does not report OK.
"""
import asyncio
import importlib.util
import json
import os
import statistics
import sys
import pickle
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path

BUNDLE_DIR = Path(sys.argv[1])
TAG = sys.argv[2]
OUT = Path(sys.argv[3])
BOX_MINUTES = float(sys.argv[4])
N_SESSIONS = int(sys.argv[5])
STOP_MODE = sys.argv[6]
STOP_TOKENS = int(sys.argv[7])
assert STOP_MODE in {"mean", "min"}, STOP_MODE
GAME = "sc25-635fd71a"

JOB_DIR = Path("/kaggle/working") / f"arm_{TAG}"
JOB_DIR.mkdir(parents=True, exist_ok=True)
os.environ["RECORDINGS_DIR"] = str(JOB_DIR / "server_recording")

import arc_agi  # noqa: E402
import taaf  # noqa: E402
import inference  # noqa: E402
from taaf.game_api import ArcadeSpec, GameAPI  # noqa: E402


def _module_root(mod):
    f = getattr(mod, "__file__", None)
    if f:
        return str(Path(f).resolve())
    paths = list(getattr(mod, "__path__", []))
    return str(Path(paths[0]).resolve()) if paths else "?"


for _mod in (taaf, inference):
    _root = _module_root(_mod)
    assert str(BUNDLE_DIR.resolve()) in _root, (
        f"[{TAG}] {_mod.__name__} imported from OUTSIDE the v12 bundle: {_root}"
    )
    print(f"[{TAG}] {_mod.__name__} from: {_root}", flush=True)
print(f"[{TAG}] config: temp={os.environ.get('LOCAL_ANALYZER_TEMPERATURE')} "
      f"upscale={os.environ.get('MULTIMODAL_UPSCALE')} "
      f"model={os.environ.get('INFERENCE_ANALYZER_MODEL')}", flush=True)

# --- EFFORT_MEDIUM graft (validated baseline) — both arms, hard-gated ---
assert os.environ.get("EFFORT_MEDIUM") == "1", "EFFORT_MEDIUM must be 1 on both arms"
assert os.environ.get("EFFORT_DEAD_RETRY") == "0", "EFFORT_DEAD_RETRY must be 0 (baseline purity)"
_graft_path = Path("/kaggle/working/graft_effort.py")
_spec = importlib.util.spec_from_file_location("graft_effort", _graft_path)
_graft = importlib.util.module_from_spec(_spec)
sys.modules["graft_effort"] = _graft
_spec.loader.exec_module(_graft)
_graft_status = _graft.install()
print(f"[{TAG}] graft: {_graft_status}", flush=True)
assert _graft_status == "effort_medium: OK", (
    f"the A/B requires the validated baseline live, got: {_graft_status}")
from inference.utils import openai_compat as _oc  # noqa: E402
assert getattr(_oc.build_chat_payload, "_effort_medium_patched", False)

with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as f:
    target = pickle.load(f)
target.actual_run_as_submission = False
target.is_competition_rerun = False
with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as f:
    bm = pickle.load(f)


def _resolve_env_dir() -> str:
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
    raise RuntimeError("environment_files dir not found under /kaggle/input")


env_dir = _resolve_env_dir()
spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
available = [e.game_id for e in arcade.available_environments]
assert GAME in available, (GAME, sorted(available))

bm.games = [
    GameAPI(env_name=GAME, arcade_spec=spec, external_game_id=f"{GAME}-{TAG}{i}")
    for i in range(N_SESSIONS)
]
bm.n_passes = 1
bm.game_weights = None
bm.label = f"sc25-thresh-{TAG}"
bm.job_dir = JOB_DIR

solver = bm.solver
print(f"[{TAG}] solver: {type(solver).__name__} pickled_concurrency={getattr(solver, 'concurrency', None)} "
      f"start_local_server={getattr(solver, 'start_local_server', None)}", flush=True)
solver.concurrency = N_SESSIONS
solver.start_local_server = False  # the shared vLLM serve is already up


def rows_snapshot():
    rows = []
    for r in bm.game_runs:
        try:
            toks = sum(int(h.generated_tokens or 0) for h in r.history)
            toks += int(getattr(r, "final_generated_tokens", 0) or 0)
            rows.append({
                "game_id": r.game_id,
                "state": r.state,
                "levels_completed": int(r.levels_completed),
                "nonzero": bool(r.levels_completed > 0),
                "actions": len(r.history),
                "generated_tokens": toks,
                "final_score": r.final_score,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"game_id": getattr(r, "game_id", "?"), "error": repr(exc)})
    return rows


STOP_INFO = {"reason": "box_or_natural", "fired_at_min": None}


def write_out(status: str) -> None:
    OUT.write_text(json.dumps({
        "tag": TAG,
        "bundle": str(BUNDLE_DIR),
        "game": GAME,
        "n_sessions": N_SESSIONS,
        "concurrency": N_SESSIONS,
        "box_minutes": BOX_MINUTES,
        "stop_mode": STOP_MODE,
        "stop_tokens": STOP_TOKENS,
        "stop_info": STOP_INFO,
        "status": status,
        "written_at": datetime.now().isoformat(),
        "sessions": rows_snapshot(),
    }, indent=2) + "\n")


_hb_stop = threading.Event()


def _heartbeat() -> None:
    while not _hb_stop.wait(120):
        try:
            write_out("running")
        except Exception:  # noqa: BLE001
            traceback.print_exc()


threading.Thread(target=_heartbeat, daemon=True).start()

started = datetime.now()
soft_end = started + timedelta(minutes=BOX_MINUTES)
print(f"[{TAG}] {N_SESSIONS}x {GAME} conc={N_SESSIONS} box={BOX_MINUTES:.0f}min "
      f"stop={STOP_MODE}>={STOP_TOKENS} soft_end={soft_end}", flush=True)


def _target_met(rows) -> bool:
    if len(rows) < N_SESSIONS:
        return False
    toks = []
    for row in rows:
        if "error" in row:
            return False
        toks.append(int(row.get("generated_tokens") or 0))
    if STOP_MODE == "mean":
        return statistics.mean(toks) >= STOP_TOKENS
    return all(
        t >= STOP_TOKENS or row.get("state") != "playing"
        for t, row in zip(toks, rows)
    )


async def _watched_run() -> None:
    run_task = asyncio.create_task(
        bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)
    )
    fired = False
    while not run_task.done():
        await asyncio.sleep(20)
        if fired:
            continue
        try:
            rows = rows_snapshot()
        except Exception:  # noqa: BLE001
            continue
        if _target_met(rows):
            fired = True
            elapsed_min = (datetime.now() - started).total_seconds() / 60
            STOP_INFO["reason"] = f"token_target_{STOP_MODE}"
            STOP_INFO["fired_at_min"] = round(elapsed_min, 1)
            print(f"[{TAG}] TOKEN TARGET met at {elapsed_min:.1f} min — graceful stop", flush=True)
            stop_event = getattr(solver, "_stop_event", None)
            if stop_event is not None:
                stop_event.set()  # the soft-end seam: sessions wind down as "cancelled"
            else:
                print(f"[{TAG}] no _stop_event on solver — falling back to task cancel", flush=True)
                run_task.cancel()
    await run_task


status = "crashed"
try:
    asyncio.run(_watched_run())
    status = "complete"
except asyncio.CancelledError:
    status = "complete_cancelled"
except Exception as exc:  # noqa: BLE001
    traceback.print_exc()
    status = f"run_error:{type(exc).__name__}"
finally:
    _hb_stop.set()
    write_out(status)
rows = rows_snapshot()
l1 = sum(1 for row in rows if row.get("nonzero"))
toks = [int(row.get("generated_tokens") or 0) for row in rows if "error" not in row]
print(f"[{TAG}] DONE status={status} L1+={l1}/{N_SESSIONS} tokens={toks} "
      f"stop={STOP_INFO}", flush=True)
'''

CELL_ARMS = r'''# 6. Sequential token-boxed arms: A (starved, conc 6) then B waves (rich, conc 2).
RUNNER_PATH = WORKING_DIR / "sc25_arm_runner.py"
RUNNER_PATH.write_text(ARM_RUNNER_SOURCE)
GRAFT_PATH = WORKING_DIR / "graft_effort.py"
GRAFT_PATH.write_text(GRAFT_SOURCE)
COMBINED_PATH = WORKING_DIR / "sc25_thresh_results.json"

# Token geometry (calibrated on dc22-ab's same-rig measurement: ~1430
# tok/min/session at conc 8, 60-min box; see build_sc25_thresh.py docstring).
A_BOX_MIN = 60.0          # clock backstop only; the token stop should fire ~32-36 min
A_STOP_TOKENS = 56_000    # mean-mode => sessions land ~50-62k (starved band <63k)
B_BOX_MIN = 75.0          # per wave
B_STOP_TOKENS = 112_000   # min-mode => every session >=112k or terminal
TOTAL_CAP_MIN = 300.0     # hard planning cap (5h): waves are skipped, loudly


def _source_path_entries(bundle_dir: Path) -> list:
    # Same precedence as dc22-ab / the scored scaffolds; grafts repos excluded
    # (the effort graft is installed explicitly by the runner instead).
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        if not repo.is_dir() or "graft" in repo.name.lower():
            continue
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(str(candidate))
    return entries


def _arm_env(bundle_dir: Path) -> dict:
    env = os.environ.copy()
    entries = _source_path_entries(bundle_dir)
    assert entries, f"no source entries under {bundle_dir}/src"
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(existing + entries)
    return env


def _persist_combined(payload: dict) -> None:
    payload["written_at"] = datetime.now().isoformat()
    COMBINED_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def run_batch(tag: str, box_minutes: float, n_sessions: int, stop_mode: str, stop_tokens: int) -> dict:
    out_path = WORKING_DIR / f"sc25_arm_{tag}.json"
    cmd = [sys.executable, str(RUNNER_PATH), str(ANIM_BUNDLE_DIR), tag, str(out_path),
           str(box_minutes), str(n_sessions), stop_mode, str(stop_tokens)]
    print(f"=== BATCH {tag}: {n_sessions} sessions conc={n_sessions}, box {box_minutes:.0f} min, "
          f"stop {stop_mode}>={stop_tokens} ===", flush=True)
    hard_timeout = box_minutes * 60 + 25 * 60
    started = time.time()
    try:
        proc = subprocess.run(cmd, env=_arm_env(ANIM_BUNDLE_DIR), timeout=hard_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print(f"BATCH {tag}: HARD TIMEOUT after {hard_timeout / 60:.0f} min — using last heartbeat snapshot",
              flush=True)
    wall_min = (time.time() - started) / 60
    result = {"tag": tag, "returncode": rc, "wall_minutes": round(wall_min, 1), "status": "missing_output"}
    if out_path.is_file():
        try:
            result = json.loads(out_path.read_text())
            result["returncode"] = rc
            result["wall_minutes"] = round(wall_min, 1)
        except Exception as exc:  # noqa: BLE001
            result["status"] = f"unreadable_output:{type(exc).__name__}"
    print(f"=== BATCH {tag} finished: rc={rc} status={result.get('status')} wall={wall_min:.1f}min ===",
          flush=True)
    return result


combined = {
    "design": "sc25 token-threshold A/B (55k vs 110k)",
    "game": "sc25-635fd71a",
    "baseline": {"EFFORT_MEDIUM": "1", "EFFORT_DEAD_RETRY": "0", "bundle": "duck38-v12 (anim)"},
    "kill_line": ("B >=3/6 (or 3/5) L1+ while A <=1/6 -> threshold REAL "
                  "(budget-allocation lever, ~+0.57/rescued game); A ~ B -> DEAD"),
    "geometry": {"A": {"n": 6, "conc": 6, "box_min": A_BOX_MIN, "stop": f"mean>={A_STOP_TOKENS}"},
                 "B": {"n": 6, "conc": 2, "waves": 3, "box_min": B_BOX_MIN, "stop": f"min>={B_STOP_TOKENS}"}},
    "arm_A": None, "arm_B_waves": [],
}
_persist_combined(combined)

try:
    combined["arm_A"] = run_batch("A", A_BOX_MIN, 6, "mean", A_STOP_TOKENS)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    combined["arm_A"] = {"tag": "A", "status": f"launcher_error:{type(exc).__name__}"}
_persist_combined(combined)

for wave in (1, 2, 3):
    elapsed_min = (time.time() - NOTEBOOK_START_EPOCH) / 60
    if elapsed_min + B_BOX_MIN + 10 > TOTAL_CAP_MIN:
        print(f"TIME-CAP: {elapsed_min:.1f} min elapsed — SKIPPING wave B{wave} and any later waves "
              f"(cap {TOTAL_CAP_MIN:.0f} min). The verdict cell scales its bar to the sessions run.",
              flush=True)
        combined["arm_B_waves"].append({"tag": f"B{wave}", "status": "skipped_time_cap",
                                        "elapsed_min": round(elapsed_min, 1)})
        _persist_combined(combined)
        continue
    try:
        combined["arm_B_waves"].append(run_batch(f"B{wave}", B_BOX_MIN, 2, "min", B_STOP_TOKENS))
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        combined["arm_B_waves"].append({"tag": f"B{wave}", "status": f"launcher_error:{type(exc).__name__}"})
    _persist_combined(combined)

print("sc25-thresh: all batches done; results at", COMBINED_PATH, flush=True)
'''

CELL_VERDICT = r'''# 7. 6v6 table + pre-registered verdict (grep for SC25 THRESH).
import math


def fisher_one_sided(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """P(X >= k_b), X ~ Hypergeom(pop=n_a+n_b, successes=k_a+k_b, draws=n_b)."""
    total_k = k_a + k_b
    total_n = n_a + n_b
    denom = math.comb(total_n, n_b)
    upper = min(total_k, n_b)
    return sum(math.comb(total_k, x) * math.comb(total_n - total_k, n_b - x)
               for x in range(k_b, upper + 1)) / denom


assert abs(fisher_one_sided(0, 6, 5, 6) - 7 / 924) < 1e-12

STARVED_MAX = 63_000    # A validity: the observed sc25 failure band
A_CONTAM = 70_000       # an A session past the hypothesized threshold = contaminated
RICH_MIN = 90_000       # B validity floor (target 110k+)
RICH_FAIL = 70_000      # a B session under the threshold never tested the hypothesis

combined = json.loads(COMBINED_PATH.read_text())
arm_a = combined.get("arm_A") or {}
b_waves = combined.get("arm_B_waves") or []


def batch_rows(res: dict, arm: str) -> list:
    rows = []
    for r in (res.get("sessions") or []):
        if "error" in r:
            rows.append({"arm": arm, "game_id": r.get("game_id", "?"), "invalid": True, "error": r["error"]})
            continue
        toks = int(r.get("generated_tokens") or 0)
        wall = res.get("wall_minutes") or 0
        rows.append({
            "arm": arm, "game_id": r["game_id"], "state": r["state"],
            "levels": int(r["levels_completed"]), "actions": int(r["actions"]),
            "tokens": toks, "final_score": r.get("final_score"),
            "tok_per_min": round(toks / wall, 0) if wall else None,
        })
    return rows


rows_a = batch_rows(arm_a, "A")
rows_b = []
for wave_res in b_waves:
    rows_b.extend(batch_rows(wave_res, wave_res.get("tag", "B?")))

print("=" * 96)
print("SC25 THRESH — TOKEN-THRESHOLD A/B RESULT TABLE (starved conc-6 vs rich conc-2)")
print("=" * 96)
print(f"{'session':<24} {'arm':<4} {'state':<11} {'L':>2} {'actions':>7} {'tokens':>9} {'tok/min':>8}  flags")
for row in rows_a + rows_b:
    if row.get("invalid"):
        print(f"{row['game_id']:<24} {row['arm']:<4} ROW ERROR: {row['error']}")
        continue
    flags = []
    if row["arm"] == "A":
        if row["tokens"] > A_CONTAM:
            flags.append("A-CONTAMINATED(>70k)")
        elif row["tokens"] > STARVED_MAX:
            flags.append("A-marginal(63-70k)")
    else:
        if row["tokens"] < RICH_FAIL:
            flags.append("B-NOT-RICH(<70k)")
        elif row["tokens"] < RICH_MIN:
            flags.append("B-marginal(70-90k)")
    if row["levels"] >= 1:
        flags.append("L1+")
    print(f"{row['game_id']:<24} {row['arm']:<4} {str(row['state']):<11} {row['levels']:>2} "
          f"{row['actions']:>7} {row['tokens']:>9} {str(row['tok_per_min']):>8}  {' '.join(flags)}")

valid_a = [r for r in rows_a if not r.get("invalid")]
valid_b = [r for r in rows_b if not r.get("invalid")]
a_l1 = sum(1 for r in valid_a if r["levels"] >= 1)
b_l1 = sum(1 for r in valid_b if r["levels"] >= 1)
a_contam = sum(1 for r in valid_a if r["tokens"] > A_CONTAM)
b_not_rich = sum(1 for r in valid_b if r["tokens"] < RICH_FAIL and r["levels"] == 0)
n_a, n_b = len(valid_a), len(valid_b)
mean_a = sum(r["tokens"] for r in valid_a) / n_a if n_a else 0
mean_b = sum(r["tokens"] for r in valid_b) / n_b if n_b else 0

print("-" * 96)
print(f"ARM A (starved): L1+ {a_l1}/{n_a}, mean tokens {mean_a:,.0f}, contaminated(>70k) {a_contam}")
print(f"ARM B (rich):    L1+ {b_l1}/{n_b}, mean tokens {mean_b:,.0f}, not-rich(<70k, L0) {b_not_rich}")

verdict = []
p_value = None
if n_a == 0 or n_b == 0:
    verdict.append("NO READ — an arm produced no sessions.")
else:
    p_value = fisher_one_sided(a_l1, n_a, b_l1, n_b)
    print(f"Fisher one-sided (B>A): p = {p_value:.4f}")
    if a_contam >= 2:
        verdict.append(f"VALIDITY: A-arm token geometry failed ({a_contam} sessions >70k) — "
                       "the starved arm was not starved; read DEGRADED.")
    if b_not_rich >= 2:
        verdict.append(f"VALIDITY: B-arm token geometry failed ({b_not_rich} L0 sessions <70k) — "
                       "the rich arm was not fed; read DEGRADED.")
    b_bar = 3  # pre-registered: >=3/6, or >=3/5 with a lost session
    if n_b <= 3:
        verdict.append(f"NO PRE-REGISTERED READ: only {n_b} B sessions ran (bar needs >=5); "
                       "counts above are suggestive only.")
    elif b_l1 >= b_bar and a_l1 <= 1:
        verdict.append(f"THRESHOLD REAL: B {b_l1}/{n_b} L1+ vs A {a_l1}/{n_a} — production lever = "
                       "budget allocation (triage + concurrency shaping), ~+0.57 per rescued "
                       "sc25-class game.")
    elif b_l1 <= a_l1 + 1:
        verdict.append(f"THRESHOLD DEAD: A {a_l1}/{n_a} ~ B {b_l1}/{n_b} — doubling the token budget "
                       "did not move sc25; the token-threshold hypothesis dies (the <63k/>=70k split "
                       "was correlational, not causal).")
    else:
        verdict.append(f"INCONCLUSIVE: A {a_l1}/{n_a} vs B {b_l1}/{n_b} — neither pre-registered line met.")

print()
for line in verdict:
    print("**", line)

combined["fisher_one_sided_p"] = p_value
combined["verdict"] = verdict
combined["summary"] = {"A_L1": a_l1, "A_n": n_a, "B_L1": b_l1, "B_n": n_b,
                       "A_mean_tokens": mean_a, "B_mean_tokens": mean_b,
                       "A_contaminated": a_contam, "B_not_rich": b_not_rich}
COMBINED_PATH.write_text(json.dumps(combined, indent=2) + "\n")
print("\nfinal results JSON:", COMBINED_PATH)
'''

CELL_TEARDOWN = r'''# 8. Teardown: stop vLLM and drop the temp install (bundle's own teardown).
for command in json.loads((ANIM_BUNDLE_DIR / "teardown_commands.json").read_text()):
    print(f"sc25-thresh: teardown command: {command[:120]}...", flush=True)
    subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())
print("sc25-thresh: teardown complete")
'''


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main() -> None:
    graft_source = GRAFT_PY.read_text()
    assert "def install() -> str:" in graft_source

    sources_cell = (
        "# Embedded sources (written to /kaggle/working by the batch cell):\n"
        "# - the per-batch runner (dc22-ab chassis + effort graft + token stop)\n"
        "# - the validated EFFORT_MEDIUM graft, byte-identical to\n"
        "#   submission/_effort_medium/graft_effort.py (13/13 offline + live smoke)\n"
        "ARM_RUNNER_SOURCE = " + repr(ARM_RUNNER) + "\n"
        "GRAFT_SOURCE = " + repr(graft_source) + "\n"
    )
    assert repr(graft_source) in sources_cell

    cells = [
        md_cell(MD_INTRO),
        code_cell(CELL_ENV),
        code_cell(CELL_WHEELS),
        code_cell(CELL_MOUNTS),
        code_cell(CELL_SERVE),
        code_cell(CELL_ATTEST),
        code_cell(sources_cell),
        code_cell(CELL_ARMS),
        code_cell(CELL_VERDICT),
        code_cell(CELL_TEARDOWN),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    joined = "\n".join("".join(c["source"]) for c in cells)
    assert joined.count("attest: OK") >= 1
    assert "external_game_id" in joined and "sc25-635fd71a" in joined
    assert joined.index("attest: OK") < joined.index('run_batch("A"')
    assert "effort_medium: OK" in joined
    assert "_stop_event" in joined

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
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
        "dataset_sources": [WHEELHOUSE_REF, ANIM_REF],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [MODEL_SOURCE],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    main()
