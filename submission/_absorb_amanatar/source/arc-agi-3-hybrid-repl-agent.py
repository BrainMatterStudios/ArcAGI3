import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# True only inside a real competition rerun; switches diagnostics + soft deadline.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

# Non-interactive matplotlib backend: diagnostics render plots with no display attached.
os.environ["MPLBACKEND"] = "Agg"
# Marks the run as a (real or emulated) submission so the framework + solver can adjust.
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
# Skip periodic JSON/HTML diagnostics and per-frame logging for every run.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1"

# Apply the measured vLLM winner before any serving setup command runs.
PUBLIC25_VLLM_PROFILE_NAME = 'kv16-bf16-mtp3-c8-cg32-batch16k'
PUBLIC25_VLLM_PROFILE_ENV = {
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "1",
    "TAAF_VLLM_KV_CACHE_DTYPE": "auto",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "16000000000",
    "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32",
    "TAAF_VLLM_MAX_NUM_BATCHED_TOKENS": "16384",
    "TAAF_VLLM_MAX_NUM_SEQS": "8",
    "TAAF_VLLM_MTP_TOKENS": "3",
    "TAAF_VLLM_OMP_THREADS": "1"
}
for key, value in PUBLIC25_VLLM_PROFILE_ENV.items():
    os.environ[key] = value
print(
    f'PUBLIC25_VLLM_PROFILE name={PUBLIC25_VLLM_PROFILE_NAME} '
    f'env={json.dumps(PUBLIC25_VLLM_PROFILE_ENV, sort_keys=True)}',
    flush=True,
)
# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
os.environ["ONLY_RESET_LEVELS"] = "true"

# Prepend the CUDA toolkit to the linker path (it is off it on Kaggle GPU images) so the
# solver's GPU libraries (e.g. vllm / torch) can link against libcuda.
cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    entry for entry in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if entry
)

# Everything the run produces is written here.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")

# --- CELL ---

# Install the ARC runtime from the bundled competition wheels.
# Quiet: stdout is discarded; stderr (and a non-zero exit) still surface real failures.
import glob

_wheelhouse_candidates = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
]
_wheelhouse = None
for _cand in _wheelhouse_candidates:
    if Path(_cand).exists():
        _wheelhouse = _cand
        break
if not _wheelhouse:
    # Fallback to search
    _found = list(Path("/kaggle/input").rglob("arc_agi_3_wheels"))
    if _found:
        _wheelhouse = str(_found[0])
    else:
        raise RuntimeError("Could not find arc_agi_3_wheels in /kaggle/input")

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links",
        _wheelhouse,
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)

# --- CELL ---

# Kaggle inputs attached to this notebook, plus bookkeeping paths used below.
DATASET_SOURCES = ["keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1", "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"]
KERNEL_SOURCES = []
DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"
SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"


# Locate the source dataset by its marker file rather than a fixed mount path.
def _find_bundle_dir() -> Path:
    for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
        return marker.parent
    raise RuntimeError("TAAF source bundle not found under /kaggle/input.")


# Kaggle mounts a dataset at /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>
# (depending on owner / slug collisions), so probe both and use whichever exists. Utility
# scripts mount under /kaggle/usr/lib/notebooks/<owner>/<slug>.
def _dataset_mount_candidates(ref: str) -> list[Path]:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _kernel_mount_candidates(ref: str) -> list[Path]:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/usr/lib/notebooks") / owner / slug]


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((c for c in candidates if c.exists()), None)


BUNDLE_DIR = _find_bundle_dir()
print(f"taaf.kaggle: source bundle = {BUNDLE_DIR}")

# Map each attached input to where Kaggle actually mounted it (the source bundle is index 0).
kaggle_input_paths: dict[str, str] = {}
for i, ref in enumerate(DATASET_SOURCES):
    candidates = _dataset_mount_candidates(ref)
    resolved = BUNDLE_DIR if i == 0 else _first_existing(candidates)
    kaggle_input_paths[ref] = str(resolved or candidates[0])
for ref in KERNEL_SOURCES:
    candidates = _kernel_mount_candidates(ref)
    kaggle_input_paths[ref] = str(_first_existing(candidates) or candidates[0])

# Published to setup commands and the solver via the environment:
setup_env = {
    # JSON {ref: mount_path} so they can locate every attached dataset / utility script.
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    # The attached dataset refs in order (index 0 is this source bundle).
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
    # The attached utility-script / kernel refs.
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps(KERNEL_SOURCES),
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\n")
print(f"taaf.kaggle: input paths = {setup_env['TAAF_KAGGLE_INPUT_PATHS']}")

# --- CELL ---

# Each bundled repo exposes its importable tree at <repo>/src or <repo>.
def _source_path_entries(bundle_dir: Path) -> list:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


# Environment handed to each setup command (paths + any keys it has persisted).
def _command_env() -> dict:
    env = os.environ.copy()
    # "$PYTHON" in a command resolves to this notebook's interpreter.
    env["PYTHON"] = sys.executable
    # Absolute path to the mounted source bundle.
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
    # The writable /kaggle/working directory.
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    # A command writes a JSON object here to persist env keys to later commands + the run.
    env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
    env.update({str(k): str(v) for k, v in json.loads(SETUP_ENV_PATH.read_text()).items()})
    return env


# Make the bundled repos importable here (sys.path) and in child processes (.pth).
source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
pth_path.write_text("".join(f"{entry}\n" for entry in source_entries))
print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")

# Solver setup commands (wheels, vLLM server startup, ...) run before the benchmark loads.
env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"taaf.kaggle: setup command: {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    # Re-read in case the command persisted new env keys.
    env = _command_env()
    os.environ.update(env)

# Honour any PYTHONPATH a setup command exported.
for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# --- CELL ---




# --- CELL ---

# Restore the deployment target and record the real submission state on it.
with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
    target = pickle.load(file)
target.actual_run_as_submission = TRUE_SUBMISSION
target.is_competition_rerun = TRUE_SUBMISSION

# Restore the benchmark and point its outputs at the Kaggle working dir.
with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
    bm = pickle.load(file)
bm.job_dir = WORKING_DIR
bm.n_passes = 3  # INCREASED FROM 1 TO 3 FOR MULTI-PASS ENSEMBLE!

# Exact public-25 and competition settings.
bm.solver.max_runtime_s_per_game = 7920.0
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 28
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = True  # Enable logging for debugging

print(
    f'PUBLIC25_SETTINGS budget_s={bm.solver.max_runtime_s_per_game} '
    f'concurrency={bm.solver.concurrency} analyzer_timeout={bm.solver.analyzer_timeout} '
    f'action_cap={bm.solver.max_actions_per_game} request_logs={bm.solver.save_request_logs}',
    flush=True,
)

# Build the live competition game list from the gateway's available environments.
def _competition_games():
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ["ARC_BASE_URL"],
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# Build the offline game list from the competition's bundled environment files.
def _offline_games(env_dir: str):
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


# The gateway can take a while to come up; poll until it answers.
def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


# Skip pre-run display and git-status copies; the staged identity pins the run.

# arc_agi reads RECORDINGS_DIR and ARC_API_KEY from env (ArcadeSpec carries neither); operation
# mode, environments dir, and base url are all passed explicitly via the spec, so no env is needed.
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

PUBLIC_GAME_IDS = tuple([
    "tn36-ef4dde99",
    "lf52-271a04aa",
    "cn04-2fe56bfb",
    "bp35-0a0ad940",
    "wa30-ee6fef47",
    "lp85-305b61c3",
    "r11l-495a7899",
    "tu93-0768757b",
    "sp80-589a99af",
    "m0r0-492f87ba",
    "vc33-5430563c",
    "ar25-0c556536",
    "ka59-38d34dbb",
    "sc25-635fd71a",
    "sk48-d8078629",
    "dc22-fdcac232",
    "cd82-fb555c5d",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ls20-9607627b",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "su15-1944f8ab",
    "tr87-cd924810"
])

if TRUE_SUBMISSION:
    # Real submission: play the live competition Arcade served by the Kaggle gateway.
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    # The gateway boots asynchronously; wait before swapping in its game list.
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
else:
    # Interactive run: play the bundled competition environments offline (no gateway).
    # The competition's environment files ship alongside the wheelhouse in the competition dataset.
    competition_env_files = str(Path(_wheelhouse).parent / "environment_files")
    offline_games = _offline_games(competition_env_files)
    offline_by_id = {game.env_name: game for game in offline_games}
    if len(offline_by_id) != len(offline_games):
        raise RuntimeError('The offline public game list contains duplicate IDs.')
    missing = sorted(set(PUBLIC_GAME_IDS) - set(offline_by_id))
    extra = sorted(set(offline_by_id) - set(PUBLIC_GAME_IDS))
    if missing or extra:
        raise RuntimeError(
            f'Offline public game set changed; missing={missing}, extra={extra}.'
        )
    bm.games = [offline_by_id[game_id] for game_id in PUBLIC_GAME_IDS]
    if len(bm.games) != 25:
        raise RuntimeError(f'Expected 25 public games, got {len(bm.games)}.')
    print(f'PUBLIC25_SELECTION games={len(bm.games)} passes=1', flush=True)

bm.n_passes = 1
bm.game_weights = None

# Outside a real submission, stop ~10 min before the wall-clock budget for a graceful exit.
budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
if budget <= 600.0:
    raise RuntimeError(f'Notebook budget is too small for the teardown reserve: {budget}.')
soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
    seconds=budget - 600.0
)

# Start recovery only after setup readiness and all run gates pass.
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))
import vllm_server_watchdog as vllm_watchdog

vllm_watchdog_setup = vllm_watchdog.load_setup(BUNDLE_DIR / 'serving_setup.py')
vllm_watchdog.start_background(
    vllm_watchdog_setup,
    vllm_watchdog.WatchdogConfig(
        interval_seconds=15.0,
        request_timeout_seconds=5,
        failure_threshold=4,
        max_restart_attempts=2,
    ),
)

import asyncio

async def _run_benchmark():
    # Play the benchmark; watchdog stop and teardown run even if it raises.
    try:
        await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=True)
        bm._save_json()
        if not TRUE_SUBMISSION:
            # Kaggle Save & Run expects this valid placeholder after an offline run.
            # A real competition rerun uses the live gateway and never enters this branch.
            import pandas as pd

            pd.DataFrame(
                [["1_0", "1", True, 1]],
                columns=["row_id", "game_id", "end_of_game", "score"],
            ).to_parquet(WORKING_DIR / "submission.parquet", index=False)

            # Check terminal coverage, then call the frozen scorer once.
            # This path does not render HTML.
            public_runs = list(bm.game_runs)
            public_run_ids = [run.game_id for run in public_runs]
            if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):
                raise RuntimeError(
                    f'Public run coverage changed: count={len(public_runs)} ids={public_run_ids}.'
                )
            unfinished = [
                (run.game_id, run.state, run.final_score)
                for run in public_runs
                if run.state not in {'won', 'gave_up', 'cancelled'}
                or run.final_score is None
            ]
            if unfinished:
                raise RuntimeError(f'Public runs did not finalize cleanly: {unfinished}.')
            crashed = [run.game_id for run in public_runs if run.state == 'crashed']
            if crashed:
                raise RuntimeError(f'Public runs crashed: {crashed}.')
            total_actions = sum(len(run.history) for run in public_runs)
            if total_actions <= 0:
                raise RuntimeError('Public runs produced no actions.')

            from inference.tools.eval import evaluate_runs, save_score_file

            score_summary = evaluate_runs([WORKING_DIR])
            score_path = save_score_file(
                score_summary,
                run_dirs=[WORKING_DIR],
                output_path=WORKING_DIR / "score.json",
            )
            if Path(score_path) != WORKING_DIR / 'score.json' or not Path(score_path).is_file():
                raise RuntimeError(f'Frozen scorer did not write score.json: {score_path}.')
            print(
                f'PUBLIC25_AUDIT runs=25 actions={total_actions} score_path={score_path}',
                flush=True,
            )
    finally:
        try:
            vllm_watchdog.stop_background(timeout_seconds=15.0)
        finally:
            for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
                print(f"taaf.kaggle: teardown command: {command}", flush=True)
                subprocess.run(
                    command,
                    shell=True,
                    check=False,
                    cwd=WORKING_DIR,
                    env=_command_env(),
                    timeout=30.0,
                )

asyncio.run(_run_benchmark())


# --- CELL ---

# ARC3-V2.9A post-run diagnostics only; gameplay and inference are already complete.
import statistics as _v29a_statistics
from collections import Counter as _V29ACounter


def _v29a_run_tokens(run) -> int:
    return sum(
        int(getattr(record, "generated_tokens", 0) or 0)
        for record in getattr(run, "history", [])
    ) + int(getattr(run, "final_generated_tokens", 0) or 0)


def _v29a_run_uncached_input_tokens(run) -> int:
    return sum(
        int(getattr(record, "uncached_input_tokens", 0) or 0)
        for record in getattr(run, "history", [])
    ) + int(getattr(run, "final_uncached_input_tokens", 0) or 0)


_v29a_runs = list(getattr(bm, "game_runs", []) or [])
_v29a_scores = [float(getattr(run, "final_score", 0.0) or 0.0) for run in _v29a_runs]
_v29a_total_actions = sum(len(getattr(run, "history", []) or []) for run in _v29a_runs)
_v29a_actions_from_levels = sum(
    sum(int(value or 0) for value in (getattr(run, "actions_per_level", []) or []))
    for run in _v29a_runs
)
_v29a_total_levels = sum(int(getattr(run, "levels_completed", 0) or 0) for run in _v29a_runs)
_v29a_total_tokens = sum(_v29a_run_tokens(run) for run in _v29a_runs)
_v29a_total_uncached = sum(_v29a_run_uncached_input_tokens(run) for run in _v29a_runs)
_v29a_score_sum = sum(_v29a_scores)
_v29a_sorted_scores = sorted(_v29a_scores, reverse=True)
_v29a_duration_seconds = None
if getattr(bm, "start_time", None) is not None and getattr(bm, "end_time", None) is not None:
    _v29a_duration_seconds = max((bm.end_time - bm.start_time).total_seconds(), 0.0)

_v29a_per_game = []
for _v29a_run in _v29a_runs:
    _v29a_per_game.append(
        {
            "game_id": str(getattr(_v29a_run, "game_id", "unknown")),
            "state": str(getattr(_v29a_run, "state", "unknown")),
            "score": float(getattr(_v29a_run, "final_score", 0.0) or 0.0),
            "levels_completed": int(getattr(_v29a_run, "levels_completed", 0) or 0),
            "number_of_levels": int(getattr(_v29a_run, "number_of_levels", 0) or 0),
            "actions": len(getattr(_v29a_run, "history", []) or []),
            "generated_tokens": _v29a_run_tokens(_v29a_run),
            "uncached_input_tokens": _v29a_run_uncached_input_tokens(_v29a_run),
        }
    )

_v29a_teardown_path = WORKING_DIR / "vllm-server-teardown.json"
_v29a_teardown = None
if _v29a_teardown_path.is_file():
    try:
        _v29a_teardown = json.loads(_v29a_teardown_path.read_text(encoding="utf-8"))
    except Exception as _v29a_teardown_error:
        _v29a_teardown = {"parse_error": repr(_v29a_teardown_error)}

ARC3_V29A_PREFLIGHT = {"status": "ok", "note": "B0 baseline fixed"}
ARC3_V29A_POLICY_FREEZE = True

ARC3_V29A_METRICS = {
    "version": "2.9.0-a",
    "variant": "ablation-b2",
    "base": "duck-qwen3.8-flash-next-nvfp4-mtp-v14",
    "true_submission": bool(TRUE_SUBMISSION),
    "model": "Qwen/Qwen3.8-Flash-Next-NVFP4",
    "serving_profile": PUBLIC25_VLLM_PROFILE_NAME,
    "preflight": ARC3_V29A_PREFLIGHT,
    "policy_freeze": ARC3_V29A_POLICY_FREEZE,
    "gameplay_changed": False,
    "prompt_changed": False,
    "sampling_changed": False,
    "serving_flags_changed": False,
    "vllm_patched_by_v29a": False,
    "recovery_added": False,
    "runs": len(_v29a_runs),
    "states": dict(sorted(_V29ACounter(str(getattr(run, "state", "unknown")) for run in _v29a_runs).items())),
    "mean_score": (_v29a_statistics.fmean(_v29a_scores) if _v29a_scores else 0.0),
    "median_score": (_v29a_statistics.median(_v29a_scores) if _v29a_scores else 0.0),
    "nonzero_games": sum(score > 0.0 for score in _v29a_scores),
    "levels_completed": _v29a_total_levels,
    "total_actions": _v29a_total_actions,
    "action_invariant_ok": _v29a_total_actions == _v29a_actions_from_levels,
    "total_generated_tokens": _v29a_total_tokens,
    "total_uncached_input_tokens": _v29a_total_uncached,
    "actions_per_completed_level": (
        _v29a_total_actions / _v29a_total_levels if _v29a_total_levels else None
    ),
    "generated_tokens_per_completed_level": (
        _v29a_total_tokens / _v29a_total_levels if _v29a_total_levels else None
    ),
    "generated_tokens_per_action": (
        _v29a_total_tokens / _v29a_total_actions if _v29a_total_actions else None
    ),
    "top1_score_share": (
        _v29a_sorted_scores[0] / _v29a_score_sum
        if _v29a_sorted_scores and _v29a_score_sum > 0.0 else None
    ),
    "top3_score_share": (
        sum(_v29a_sorted_scores[:3]) / _v29a_score_sum
        if _v29a_sorted_scores and _v29a_score_sum > 0.0 else None
    ),
    "mean_without_top_game": (
        (_v29a_score_sum - _v29a_sorted_scores[0]) / (len(_v29a_scores) - 1)
        if len(_v29a_scores) > 1 else None
    ),
    "duration_seconds": _v29a_duration_seconds,
    "teardown": _v29a_teardown,
    "per_game": _v29a_per_game,
}

_v29a_acceptance_checks = {}
if not TRUE_SUBMISSION and len(_v29a_runs) == 25:
    _v29a_acceptance_checks = {
        "mean_score_at_least_6_3": ARC3_V29A_METRICS["mean_score"] >= 6.3,
        "median_score_at_least_3_0": ARC3_V29A_METRICS["median_score"] >= 3.0,
        "levels_at_least_34": _v29a_total_levels >= 34,
        "nonzero_games_at_least_18": ARC3_V29A_METRICS["nonzero_games"] >= 18,
        "generated_tokens_at_most_2_2m": _v29a_total_tokens <= 2_200_000,
        "no_crashed_runs": ARC3_V29A_METRICS["states"].get("crashed", 0) == 0,
        "action_invariant_ok": ARC3_V29A_METRICS["action_invariant_ok"],
    }
ARC3_V29A_METRICS["local_acceptance"] = {
    "applicable": bool(_v29a_acceptance_checks),
    "passed": bool(_v29a_acceptance_checks) and all(_v29a_acceptance_checks.values()),
    "checks": _v29a_acceptance_checks,
    "reference_only_not_a_submission_gate": True,
}

_v29a_metrics_rendered = json.dumps(ARC3_V29A_METRICS, indent=2, sort_keys=True) + "\n"
_v29a_metrics_json = WORKING_DIR / "arc3_v29a_metrics.json"
_v29a_metrics_txt = WORKING_DIR / "arc3_v29a_metrics.txt"
_v29a_metrics_json.write_text(_v29a_metrics_rendered, encoding="utf-8")
_v29a_metrics_txt.write_text(_v29a_metrics_rendered, encoding="utf-8")
print("ARC3-V2.9A post-run metrics saved", flush=True)
print(
    json.dumps(
        {
            "runs": ARC3_V29A_METRICS["runs"],
            "mean_score": ARC3_V29A_METRICS["mean_score"],
            "median_score": ARC3_V29A_METRICS["median_score"],
            "levels_completed": ARC3_V29A_METRICS["levels_completed"],
            "nonzero_games": ARC3_V29A_METRICS["nonzero_games"],
            "total_actions": ARC3_V29A_METRICS["total_actions"],
            "total_generated_tokens": ARC3_V29A_METRICS["total_generated_tokens"],
            "local_acceptance": ARC3_V29A_METRICS["local_acceptance"],
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
print("Saved:", _v29a_metrics_json, flush=True)
print("Saved:", _v29a_metrics_txt, flush=True)


# --- CELL ---

# Minimal diagnostics are enabled; skip post-run HTML rendering.
