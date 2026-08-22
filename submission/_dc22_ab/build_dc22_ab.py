#!/usr/bin/env python3
"""build_dc22_ab.py — DC22 x BUNDLE same-rig A/B falsifier (zero submission slots).

MOTIVATION (docs/RESEARCH-2026-08-22-slotmath-and-top3.md, Addendum 2): stock
June bundle scored 0/6 nonzero on dc22 across the 60-min screens + the 132-min
wave, while duck38-v12/xpl-graft smokes scored 5/6 at identical config
(temp 0.6 / upscale 4; Fisher one-sided p = 7/924 = 0.0076) — but the two
cohorts ran on different rigs. This kernel settles the confound: BOTH arms on
the SAME GPU session, same vLLM boot, same config.

DESIGN — one kernel (ahmedmobasher86/arc3-dc22-ab), sequential arms:
  ARM A (stock): 8 offline dc22-fdcac232 sessions with the STOCK June bundle —
      thtennant/taaf-kaggle-source-share-fork (source sha ae49c5db, exactly
      what pack-v22 mounts), taaf_grafts NEVER imported and its repo dir
      excluded from the arm's PYTHONPATH → byte-stock June source.
  ARM B (v12):   8 offline dc22-fdcac232 sessions with the duck38-v12 bundle —
      jakobbrggen/taaf-kaggle-source-anim-20260807-anim, exactly the bundle
      the v12 smokes ran (build_duck38_v12.py; no extra grafts there either —
      the behavioral delta IS the bundle: hard_noop_guard + animation
      awareness + prompt/tool_agent changes).
  Shared serve chain: ONE vLLM boot (the bundles' setup_commands.json are
  md5-identical — 99e4b35d / 3eca3c6f, asserted at run time — patched to the
  official Qwen3.8-FP8 Kaggle Model exactly as pack-v22 does, boot attestation
  cell verbatim). Config is arm-invariant by construction: the bundle's own
  setup env pins LOCAL_ANALYZER_TEMPERATURE=0.6 and MULTIMODAL_UPSCALE=4.
  Each arm: 60-min soft box, solver.concurrency = sessions, sessions spun as
  distinct games via GameAPI(external_game_id=...) (proven dup-game seam,
  game_api.py:155). Arms run as SUBPROCESSES with per-arm PYTHONPATH so the
  two source trees never share an interpreter. The bundle pickles
  (deploy_target.pkl / benchmark_initial.pkl) are the established, audited
  bundle format — same load path every scored kernel uses.

PRE-REGISTERED KILL LINE (from the task):
  stock >= 3/8 nonzero on this rig  → the flip was RIG; lever DEAD.
  stock 0-1/8 AND v12 >= 5/8        → v12 bundle genuinely unlocks dc22
                                       (+~0.19 full-25 mean candidate).
  anything else                      → inconclusive zone.
  VALIDITY: per-session generated tokens should be >= ~60k; a session under
  50k is starved and flagged; if most of an arm starves the read is VOID.
  TIME-PRESSED fallback: if boot overruns and the per-arm box would fall
  under 55 min, drop to 6+6 sessions and say so (kill line scales: stock>=3
  → rig; stock<=1 AND v12>=4 → bundle).

Usage:
  python3 submission/_dc22_ab/build_dc22_ab.py
  cd submission/_dc22_ab && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
KERNEL_SLUG = "arc3-dc22-ab"

JUNE_REF = "thtennant/taaf-kaggle-source-share-fork"
ANIM_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"
WHEELHOUSE_REF = "driessmit1/arc3-vllm-h100-wheelhouse-v3"
MODEL_SOURCE = "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
COMPETITION = "arc-prize-2026-arc-agi-3"

MD_INTRO = """\
# arc3-dc22-ab — DC22 x BUNDLE same-rig A/B falsifier

**Not a submission candidate.** Settles the rig confound behind the dc22 x
bundle signal (stock June 0/6 vs duck38-v12 smokes 5/6 nonzero, Fisher
p=0.0076): ARM A = 8 dc22 sessions on the STOCK June bundle, ARM B = 8 dc22
sessions on the duck38-v12 (anim) bundle, sequential on the SAME GPU session
and the SAME Qwen3.8-FP8 vLLM serve, identical config (temp 0.6 / upscale 4).

Pre-registered kill line: stock >= 3/8 nonzero -> flip was RIG, lever dead;
stock 0-1/8 AND v12 >= 5/8 -> the v12 bundle genuinely unlocks dc22
(+~0.19 full-25 mean candidate). Sessions under ~50k generated tokens are
starved -> flagged; a mostly-starved arm voids the read.
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
assert not TRUE_SUBMISSION, "arc3-dc22-ab is a falsifier kernel, never a submission candidate"
NOTEBOOK_START_EPOCH = time.time()

os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "0"
# Both arms run with minimal diagnostics (no frame sidecars / movie renders):
# identical across arms, and 8 concurrent sessions would otherwise eat RAM+time.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"
os.environ["ONLY_RESET_LEVELS"] = "true"

cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# P100 fail-fast: this study is only valid on the scored GPU class.
_gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
_gpu_names = [line.strip() for line in _gpu.stdout.splitlines() if line.strip()]
assert _gpu_names and all("rtx pro 6000" in name.lower() for name in _gpu_names), (
    f"FAIL-FAST: expected RTX Pro 6000, got {_gpu_names!r} (rc={_gpu.returncode}, err={_gpu.stderr.strip()!r})"
)
print(f"dc22-ab: GPU OK: {_gpu_names}")
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
print("dc22-ab: arc-agi installed from competition wheels")
'''

CELL_MOUNTS = r'''# 3. Resolve the Qwen3.8 model mount and BOTH source bundles.
JUNE_REF = "thtennant/taaf-kaggle-source-share-fork"       # ARM A: stock June source (sha ae49c5db)
ANIM_REF = "jakobbrggen/taaf-kaggle-source-anim-20260807-anim"  # ARM B: duck38-v12 bundle
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

# June bundle shell-surface fingerprints (2026-08-17 audit): the serve chain
# is arm-invariant only if BOTH bundles carry the identical setup/teardown.
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
print(f"dc22-ab: qwen3.8 model = {QWEN_MODEL_PATH} ({len(_qwen_shards)} shards)")


def _dataset_mount_candidates(ref: str) -> list:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _resolve_bundle(ref: str) -> Path:
    for cand in _dataset_mount_candidates(ref):
        if (cand / DATASET_BUNDLE_MARKER).is_file():
            return cand
    raise FileNotFoundError(f"bundle {ref} not mounted with marker; tried {_dataset_mount_candidates(ref)}")


JUNE_BUNDLE_DIR = _resolve_bundle(JUNE_REF)
ANIM_BUNDLE_DIR = _resolve_bundle(ANIM_REF)
print(f"dc22-ab: ARM A (stock June) bundle = {JUNE_BUNDLE_DIR}")
print(f"dc22-ab: ARM B (duck38-v12) bundle = {ANIM_BUNDLE_DIR}")
for _tag, _bdir in (("A/june", JUNE_BUNDLE_DIR), ("B/anim", ANIM_BUNDLE_DIR)):
    print(f"dc22-ab: bundle[{_tag}] marker = "
          + json.dumps(json.loads((_bdir / DATASET_BUNDLE_MARKER).read_text())))
    _setup_md5 = hashlib.md5((_bdir / "setup_commands.json").read_bytes()).hexdigest()
    _teardown_md5 = hashlib.md5((_bdir / "teardown_commands.json").read_bytes()).hexdigest()
    print(f"dc22-ab: bundle[{_tag}] setup md5={_setup_md5} teardown md5={_teardown_md5}")
    assert _setup_md5 == EXPECTED_SETUP_MD5, f"{_tag}: setup_commands.json drifted — serve chain no longer arm-invariant"
    assert _teardown_md5 == EXPECTED_TEARDOWN_MD5, f"{_tag}: teardown_commands.json drifted"
print("dc22-ab: setup/teardown md5-identical across bundles — ONE serve boot covers both arms")

_wheelhouse_dir = next((c for c in _dataset_mount_candidates(WHEELHOUSE_REF) if c.exists()), None)
assert _wheelhouse_dir is not None, "vLLM wheelhouse dataset not mounted"

kaggle_input_paths = {
    JUNE_REF: str(JUNE_BUNDLE_DIR),
    ANIM_REF: str(ANIM_BUNDLE_DIR),
    WHEELHOUSE_REF: str(_wheelhouse_dir),
    QWEN_MODEL_REF: str(QWEN_MODEL_PATH),
}
setup_env = {
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps([JUNE_REF, WHEELHOUSE_REF]),
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps([]),
    "TAAF_QWEN_MODEL_PATH": str(QWEN_MODEL_PATH),
    "TAAF_QWEN_SERVED_MODEL_NAME": QWEN_SERVED_MODEL_NAME,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\n")
print(f"dc22-ab: input paths = {setup_env['TAAF_KAGGLE_INPUT_PATHS']}")
'''

CELL_SERVE = r'''# 4. Patch the bundled setup's model identity to Qwen3.8 and boot vLLM ONCE.
# (Identical mechanism to pack-v22 / duck38-v12: rewrite exactly the three
# top-level string assignments in the here-doc serve blob; fail loudly if the
# bundle drifted.) Serve config the blob pins — and both arms inherit —
# includes LOCAL_ANALYZER_TEMPERATURE=0.6 and MULTIMODAL_UPSCALE=4.
import re as _re


def _command_env() -> dict:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(JUNE_BUNDLE_DIR)
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
    print(f"dc22-ab: qwen3.8 setup patch = {counts}")
    return patched


env = _command_env()
_setup_commands = _patch_qwen38_setup_commands(json.loads((JUNE_BUNDLE_DIR / "setup_commands.json").read_text()))
for command in _setup_commands:
    print(f"dc22-ab: setup command: {command[:160]}...", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    env = _command_env()
    os.environ.update(env)

_served = os.environ.get("INFERENCE_ANALYZER_MODEL", "")
if _served != QWEN_SERVED_MODEL_NAME:
    raise RuntimeError(f"setup completed but analyzer model id is {_served!r}, expected {QWEN_SERVED_MODEL_NAME!r}")
print(f"DC22AB_MODEL_PIN SERVED_MODEL_NAME={_served} MODEL_PATH={QWEN_MODEL_PATH}")
print(f"DC22AB_CONFIG temp={os.environ.get('LOCAL_ANALYZER_TEMPERATURE')} "
      f"upscale={os.environ.get('MULTIMODAL_UPSCALE')} "
      f"multimodal={os.environ.get('MULTIMODAL_CONTEXT')}")
assert os.environ.get("LOCAL_ANALYZER_TEMPERATURE") == "0.6", "config drift: temperature != 0.6"
assert os.environ.get("MULTIMODAL_UPSCALE") == "4", "config drift: upscale != 4"
'''

CELL_ATTEST = r'''# 5. Boot attestation (doctrine v2, verbatim from pack-v22): the mounted
# weights must be the OFFICIAL Qwen3.8-FP8. A wrong mount must DIE here.
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

# The per-arm runner: executed as a subprocess with PYTHONPATH pointing at ONE
# bundle's source tree. Embedded into the notebook via repr() — no .format().
ARM_RUNNER = r'''"""dc22_arm_runner.py — one A/B arm: N concurrent offline dc22 sessions.

argv: bundle_dir arm_name out_json box_minutes n_sessions
Imports taaf + inference from PYTHONPATH (set by the parent to THIS arm's
bundle), loads the bundle's own deploy_target/benchmark_initial payloads
(the audited bundle format every scored kernel uses), spins n_sessions copies
of dc22-fdcac232 as distinct games via external_game_id, runs them
concurrently under a soft deadline, and writes per-session rows
(levels_completed, generated tokens, state) to out_json — re-written every
2 minutes so a crash still leaves a partial read.
"""
import asyncio
import json
import os
import pickle
import sys
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path

BUNDLE_DIR = Path(sys.argv[1])
ARM = sys.argv[2]
OUT = Path(sys.argv[3])
BOX_MINUTES = float(sys.argv[4])
N_SESSIONS = int(sys.argv[5])
GAME = "dc22-fdcac232"

JOB_DIR = Path("/kaggle/working") / f"arm_{ARM}"
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
        f"[{ARM}] {_mod.__name__} imported from OUTSIDE this arm's bundle: {_root}"
    )
    print(f"[{ARM}] {_mod.__name__} from: {_root}", flush=True)
assert "taaf_grafts" not in sys.modules, f"[{ARM}] taaf_grafts leaked into a stock arm"
print(f"[{ARM}] config: temp={os.environ.get('LOCAL_ANALYZER_TEMPERATURE')} "
      f"upscale={os.environ.get('MULTIMODAL_UPSCALE')} "
      f"multimodal={os.environ.get('MULTIMODAL_CONTEXT')} "
      f"model={os.environ.get('INFERENCE_ANALYZER_MODEL')}", flush=True)

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
    GameAPI(env_name=GAME, arcade_spec=spec, external_game_id=f"{GAME}-{ARM}{i}")
    for i in range(N_SESSIONS)
]
bm.n_passes = 1
bm.game_weights = None
bm.label = f"dc22-ab-{ARM}"
bm.job_dir = JOB_DIR

solver = bm.solver
print(f"[{ARM}] solver: {type(solver).__name__} pickled_concurrency={getattr(solver, 'concurrency', None)} "
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


def write_out(status: str) -> None:
    OUT.write_text(json.dumps({
        "arm": ARM,
        "bundle": str(BUNDLE_DIR),
        "game": GAME,
        "n_sessions": N_SESSIONS,
        "box_minutes": BOX_MINUTES,
        "status": status,
        "written_at": datetime.now().isoformat(),
        "temperature": os.environ.get("LOCAL_ANALYZER_TEMPERATURE"),
        "upscale": os.environ.get("MULTIMODAL_UPSCALE"),
        "sessions": rows_snapshot(),
    }, indent=2) + "\n")


_stop = threading.Event()


def _heartbeat() -> None:
    while not _stop.wait(120):
        try:
            write_out("running")
        except Exception:  # noqa: BLE001
            traceback.print_exc()


threading.Thread(target=_heartbeat, daemon=True).start()

soft_end = datetime.now() + timedelta(minutes=BOX_MINUTES)
print(f"[{ARM}] {N_SESSIONS} sessions of {GAME}, concurrency={N_SESSIONS}, soft_end={soft_end}", flush=True)
status = "crashed"
try:
    asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True))
    status = "complete"
except Exception as exc:  # noqa: BLE001
    traceback.print_exc()
    status = f"run_error:{type(exc).__name__}"
finally:
    _stop.set()
    write_out(status)
nonzero = sum(1 for row in rows_snapshot() if row.get("nonzero"))
print(f"[{ARM}] DONE status={status} nonzero={nonzero}/{N_SESSIONS}", flush=True)
print(f"[{ARM}] rows: {json.dumps(rows_snapshot())}", flush=True)
'''

CELL_ARMS = r'''# 6. Run the arms sequentially: A = stock June, B = duck38-v12 (anim).
RUNNER_PATH = WORKING_DIR / "dc22_arm_runner.py"
RUNNER_PATH.write_text(ARM_RUNNER_SOURCE)
COMBINED_PATH = WORKING_DIR / "dc22_ab_results.json"


def _source_path_entries(bundle_dir: Path) -> list:
    # Mirror of the scaffolds' _source_path_entries, minus any grafts repo:
    # ARM A must be byte-stock June, and neither arm imports taaf_grafts.
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
    # The scored scaffolds put the setup-exported PYTHONPATH (vllm site-packages)
    # ahead of the bundle sources; replicate that precedence.
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(existing + entries)
    print(f"arm env: PYTHONPATH={env['PYTHONPATH']}")
    return env


def _persist_combined(payload: dict) -> None:
    payload["written_at"] = datetime.now().isoformat()
    COMBINED_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def run_arm(arm: str, bundle_dir: Path, box_minutes: float, n_sessions: int) -> dict:
    out_path = WORKING_DIR / f"dc22_arm_{arm}.json"
    cmd = [sys.executable, str(RUNNER_PATH), str(bundle_dir), arm, str(out_path),
           str(box_minutes), str(n_sessions)]
    print(f"=== ARM {arm}: {n_sessions} sessions, {box_minutes:.0f}-min box, bundle={bundle_dir} ===", flush=True)
    hard_timeout = box_minutes * 60 + 25 * 60  # box + drain(120s) + teardown + slack
    started = time.time()
    try:
        proc = subprocess.run(cmd, env=_arm_env(bundle_dir), timeout=hard_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print(f"ARM {arm}: HARD TIMEOUT after {hard_timeout / 60:.0f} min — using last heartbeat snapshot", flush=True)
    wall_min = (time.time() - started) / 60
    result = {"arm": arm, "returncode": rc, "wall_minutes": round(wall_min, 1), "status": "missing_output"}
    if out_path.is_file():
        try:
            result = json.loads(out_path.read_text())
            result["returncode"] = rc
            result["wall_minutes"] = round(wall_min, 1)
        except Exception as exc:  # noqa: BLE001
            result["status"] = f"unreadable_output:{type(exc).__name__}"
    print(f"=== ARM {arm} finished: rc={rc} status={result.get('status')} wall={wall_min:.1f}min ===", flush=True)
    return result


# Phase budget: target total <= 3h. The 60-min per-session box is the
# pre-registered geometry; if boot overran so badly that two 55+ min boxes no
# longer fit, drop to 6+6 sessions and shrink the box (stated loudly).
ARM_BOX_MIN = 60.0
N_SESSIONS = 8
_elapsed_min = (time.time() - NOTEBOOK_START_EPOCH) / 60
_per_arm_budget = (180.0 - _elapsed_min - 12.0) / 2  # 12 min for drains/teardowns/reads
if _per_arm_budget < 55.0:
    ARM_BOX_MIN = max(45.0, _per_arm_budget)
    N_SESSIONS = 6
    print(f"TIME-PRESSED: boot took {_elapsed_min:.1f} min -> dropping to 6+6 sessions, "
          f"{ARM_BOX_MIN:.0f}-min boxes (pre-registered fallback)", flush=True)
else:
    print(f"phase budget OK: boot {_elapsed_min:.1f} min, running 8v8 with 60-min boxes", flush=True)

combined = {"design": "dc22 x bundle same-rig A/B", "game": "dc22-fdcac232",
            "kill_line": "stock>=3/8 nonzero -> RIG/dead; stock<=1/8 AND v12>=5/8 -> BUNDLE (+~0.19)",
            "n_sessions_per_arm": N_SESSIONS, "box_minutes": ARM_BOX_MIN,
            "arm_A_stock_june": None, "arm_B_duck38_v12": None}
_persist_combined(combined)

try:
    combined["arm_A_stock_june"] = run_arm("A", JUNE_BUNDLE_DIR, ARM_BOX_MIN, N_SESSIONS)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    combined["arm_A_stock_june"] = {"arm": "A", "status": f"launcher_error:{type(exc).__name__}"}
_persist_combined(combined)

# Arm B gets whatever honestly remains, floored at 45 min.
_elapsed_min = (time.time() - NOTEBOOK_START_EPOCH) / 60
_box_b = min(ARM_BOX_MIN, max(45.0, 180.0 - _elapsed_min - 8.0))
if _box_b < ARM_BOX_MIN:
    print(f"NOTE: arm B box shrunk to {_box_b:.0f} min to hold the 3h cap "
          f"(arm A geometry was {ARM_BOX_MIN:.0f} min — read with caution)", flush=True)
try:
    combined["arm_B_duck38_v12"] = run_arm("B", ANIM_BUNDLE_DIR, _box_b, N_SESSIONS)
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    combined["arm_B_duck38_v12"] = {"arm": "B", "status": f"launcher_error:{type(exc).__name__}"}
_persist_combined(combined)
print("dc22-ab: both arms done; results at", COMBINED_PATH, flush=True)
'''

CELL_VERDICT = r'''# 7. Fisher table + pre-registered verdict.
import math


def fisher_one_sided(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """P(X >= k_b), X ~ Hypergeom(pop=n_a+n_b, successes=k_a+k_b, draws=n_b).
    Sanity: fisher_one_sided(0, 6, 5, 6) == 7/924 == 0.007576 (the motivating p)."""
    total_k = k_a + k_b
    total_n = n_a + n_b
    denom = math.comb(total_n, n_b)
    upper = min(total_k, n_b)
    return sum(math.comb(total_k, x) * math.comb(total_n - total_k, n_b - x)
               for x in range(k_b, upper + 1)) / denom


assert abs(fisher_one_sided(0, 6, 5, 6) - 7 / 924) < 1e-12

STARVE_TOKENS = 50_000
TARGET_TOKENS = 60_000

combined = json.loads(COMBINED_PATH.read_text())
arm_a = combined.get("arm_A_stock_june") or {}
arm_b = combined.get("arm_B_duck38_v12") or {}


def arm_stats(res: dict) -> dict:
    sessions = res.get("sessions") or []
    rows = [r for r in sessions if "error" not in r]
    nonzero = sum(1 for r in rows if r.get("nonzero"))
    toks = [int(r.get("generated_tokens") or 0) for r in rows]
    starved = sum(1 for t in toks if t < STARVE_TOKENS)
    return {"n": len(rows), "nonzero": nonzero, "tokens": toks, "starved": starved,
            "status": res.get("status"), "rows": rows}


sa, sb = arm_stats(arm_a), arm_stats(arm_b)
print("=" * 88)
print("DC22 x BUNDLE SAME-RIG A/B — RESULT TABLE")
print("=" * 88)
for label, s in (("ARM A (stock June)", sa), ("ARM B (duck38-v12)", sb)):
    print(f"\n{label}: status={s['status']} sessions={s['n']} nonzero={s['nonzero']}/{s['n']} "
          f"starved(<{STARVE_TOKENS // 1000}k tok)={s['starved']}")
    for r in s["rows"]:
        flag = " STARVED" if int(r.get("generated_tokens") or 0) < STARVE_TOKENS else ""
        print(f"  {r['game_id']:<24} state={r['state']:<10} levels={r['levels_completed']} "
              f"actions={r['actions']:>4} gen_tokens={r['generated_tokens']:>8}{flag}")

verdict_lines = []
p_value = None
if sa["n"] == 0 or sb["n"] == 0:
    verdict_lines.append("VERDICT: NO READ — at least one arm produced no sessions "
                         f"(A status={sa['status']}, B status={sb['status']}).")
else:
    p_value = fisher_one_sided(sa["nonzero"], sa["n"], sb["nonzero"], sb["n"])
    print(f"\nFisher (one-sided, B>A): stock {sa['nonzero']}/{sa['n']} vs v12 {sb['nonzero']}/{sb['n']} "
          f"-> p = {p_value:.4f}")
    # Validity gate first.
    void_a = sa["starved"] > sa["n"] / 2
    void_b = sb["starved"] > sb["n"] / 2
    if void_a or void_b:
        verdict_lines.append(
            f"READ VOID: session starvation (A {sa['starved']}/{sa['n']}, B {sb['starved']}/{sb['n']} "
            f"under {STARVE_TOKENS // 1000}k generated tokens; target was ~{TARGET_TOKENS // 1000}k+). "
            "Token throughput, not the bundle, may explain any gap — do not read the kill line.")
    # Pre-registered kill line (8v8; scaled thresholds if the 6+6 fallback fired).
    n = sa["n"]
    rig_thresh = 3
    bundle_b_thresh = 5 if n >= 8 else 4
    if sa["nonzero"] >= rig_thresh:
        verdict_lines.append(
            f"KILL LINE: stock {sa['nonzero']}/{n} >= {rig_thresh} nonzero on this rig -> "
            "the screens/wave flip was RIG, the bundle lever is DEAD.")
    elif sa["nonzero"] <= 1 and sb["nonzero"] >= bundle_b_thresh:
        verdict_lines.append(
            f"KILL LINE: stock {sa['nonzero']}/{n} AND v12 {sb['nonzero']}/{n} >= {bundle_b_thresh} -> "
            "the v12 bundle GENUINELY unlocks dc22 (+~0.19 full-25 mean candidate).")
    else:
        verdict_lines.append(
            f"INCONCLUSIVE: stock {sa['nonzero']}/{n}, v12 {sb['nonzero']}/{n} — neither "
            "pre-registered line met; treat as a partial update, not a verdict.")

print()
for line in verdict_lines:
    print("**", line)

combined["fisher_one_sided_p"] = p_value
combined["verdict"] = verdict_lines
COMBINED_PATH.write_text(json.dumps(combined, indent=2) + "\n")
print("\nfinal results JSON:", COMBINED_PATH)
'''

CELL_TEARDOWN = r'''# 8. Teardown: stop vLLM and drop the temp install (bundle's own teardown).
for command in json.loads((JUNE_BUNDLE_DIR / "teardown_commands.json").read_text()):
    print(f"dc22-ab: teardown command: {command[:120]}...", flush=True)
    subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())
print("dc22-ab: teardown complete")
'''


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def main() -> None:
    # The runner source is embedded into the notebook as a module-level string
    # assigned in its own cell, so cell 6 can write it to /kaggle/working.
    runner_cell = "# Per-arm runner source (written to /kaggle/working by the next cell).\n" \
                  "ARM_RUNNER_SOURCE = " + repr(ARM_RUNNER) + "\n"

    cells = [
        md_cell(MD_INTRO),
        code_cell(CELL_ENV),
        code_cell(CELL_WHEELS),
        code_cell(CELL_MOUNTS),
        code_cell(CELL_SERVE),
        code_cell(CELL_ATTEST),
        code_cell(runner_cell),
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
    assert "external_game_id" in joined and "dc22-fdcac232" in joined
    assert joined.index("attest: OK") < joined.index('run_arm("A"')

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
        "dataset_sources": [WHEELHOUSE_REF, JUNE_REF, ANIM_REF],
        "kernel_sources": [],
        "competition_sources": [COMPETITION],
        "model_sources": [MODEL_SOURCE],
    }, indent=2) + "\n")

    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    main()
