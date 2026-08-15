#!/usr/bin/env python3
"""Build submission/_duck_38/duck-38.ipynb.

SINGLE-VARIABLE arm: shipped duck-base v2 bytes + exactly one behavioural
delta — the served brain swapped vrfai/Qwen3.6-27B-FP8 → Qwen/Qwen3.8-27B-FP8
(official FP8, published 2026-08-14 14:44Z, apache-2.0).

WHY (2026-08-15, docs/RESEARCH-2026-08-15-field-sweep-and-qwen38.md):
Franzen 2.58 / Sorokin 2.10 / AbeLincoln 1.90 all posted within ~10h of the
model drop; one forum report of a consistent 2x on the local 25 in a
duck-class harness; config VERIFIED identical architecture class/shape to the
served 3.6 (Qwen3_5ForConditionalGeneration, 64L/5120h/interval-4 GDN);
vLLM 0.19.0 loaded it clean on the Modal exact-fidelity rig.

MECHANISM (three coordinated text swaps, all asserted, zero other deltas):
  1. kernel-metadata.json dataset_sources: vrfai snapshot dataset →
     ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot.
  2. Notebook cell DATASET_SOURCES list: same swap (drives
     TAAF_KAGGLE_INPUT_PATHS, which resolve_kaggle_dataset_path consults).
  3. Cell 8's setup-command loop: rewrite the command TEXT before execution —
     MODEL_OWNER/MODEL_SLUG/SERVED_MODEL_NAME constants inside the PYSETUP
     blob (LOCAL_ANALYZER_MODEL_ID / INFERENCE_ANALYZER_MODEL derive from
     SERVED_MODEL_NAME, so the one constant covers the client side too).
Plus the commit-only probe-both mount cells proven by duck-mem v4 and
duck-p3 v2 (the scored rerun never executes the environment_files branch).

Usage:  .venv/bin/python submission/_duck_38/build_duck_38.py
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
OUT = Path(__file__).parent / "duck-38.ipynb"

OLD_SNAPSHOT_REF = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
NEW_SNAPSHOT_REF = "ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot"

# The exact constants inside the bundle's setup_commands.json PYSETUP blob
# (scratchpad/taaf_scored_ref/setup_commands.json is the tracked reference).
CMD_SWAPS = (
    ("MODEL_OWNER = 'driessmit1'", "MODEL_OWNER = 'ahmedmobasher86'"),
    ("MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'",
     "MODEL_SLUG = 'qwen3-8-27b-fp8-hf-snapshot'"),
    ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
     "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
)

RUN_LOOP_OLD = """\
    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)"""

RUN_LOOP_NEW = """\
    # duck-38: swap the served brain by rewriting the setup-command text before
    # execution. Each replacement must fire — a silent no-op here would serve
    # the wrong model while the submission claims 3.8 (fail loudly instead).
    _38_SWAPS = (
        ("MODEL_OWNER = 'driessmit1'", "MODEL_OWNER = 'ahmedmobasher86'"),
        ("MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'",
         "MODEL_SLUG = 'qwen3-8-27b-fp8-hf-snapshot'"),
        ("SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'",
         "SERVED_MODEL_NAME = 'Qwen/Qwen3.8-27B-FP8'"),
    )
    _38_fired = {old: False for old, _ in _38_SWAPS}
    for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
        for _38_old, _38_new in _38_SWAPS:
            if _38_old in command:
                command = command.replace(_38_old, _38_new)
                _38_fired[_38_old] = True
        print(f"taaf.kaggle: setup command: {command}", flush=True)
        subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)"""

RUN_LOOP_ASSERT_OLD = """\
        env = _command_env()
        os.environ.update(env)
else:
    print("[duck] commit: skipping setup_commands (no GPU serve)", flush=True)"""

RUN_LOOP_ASSERT_NEW = """\
        env = _command_env()
        os.environ.update(env)
    _38_missed = [k for k, fired in _38_fired.items() if not fired]
    assert not _38_missed, f"[duck-38] model swap did NOT fire for: {_38_missed}"
    print("[duck-38] served brain swapped to Qwen/Qwen3.8-27B-FP8: OK", flush=True)
else:
    print("[duck] commit: skipping setup_commands (no GPU serve)", flush=True)
    print("[duck-38] commit mode: swap loop present but not exercised "
          "(TRUE_SUBMISSION only)", flush=True)"""

# --- commit-only probe-both mount cells (lineage: build_duck_mem.py, proven
# by duck-mem v4 GPU commit + scored run, and duck-p3 v2 commit) -------------
INSTALL_MARKER = "# Install the ARC runtime from the bundled competition wheels."
INSTALL_NEW = '''\
# duck-38: install the ARC runtime from the bundled competition wheels.
# Probe both mount forms; fail loudly naming both (mount form varies by machine).
_d38_wheel_dirs = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
]
_d38_wheels = next((p for p in _d38_wheel_dirs if Path(p).is_dir()), None)
if _d38_wheels is None:
    raise RuntimeError(
        f"arc_agi_3_wheels directory not found at either mount form: {_d38_wheel_dirs}")
print(f"[duck-38] installing arc-agi from {_d38_wheels}", flush=True)
_d38_pip = subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--no-index", "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links", _d38_wheels,
        "arc-agi",
    ],
    capture_output=True, text=True,
)
if _d38_pip.stdout.strip():
    print(_d38_pip.stdout, flush=True)
if _d38_pip.stderr.strip():
    print(f"[duck-38] pip stderr:\\n{_d38_pip.stderr}", flush=True)
if _d38_pip.returncode != 0:
    raise RuntimeError(f"[duck-38] pip install of arc-agi FAILED (rc={_d38_pip.returncode})")
print(f"[duck-38] arc-agi installed from {_d38_wheels}", flush=True)\
'''

ENVFILES_OLD = ('    competition_env_files = str(Path("/kaggle/input/competitions/'
                'arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")')
ENVFILES_NEW = '''\
    _d38_env_candidates = [
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
        "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files",
    ]
    competition_env_files = next(
        (p for p in _d38_env_candidates if Path(p).is_dir()), _d38_env_candidates[0])\
'''


def ledg_hash(nb: dict) -> str:
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()


def replace_in_cell(nb: dict, old: str, new: str, what: str) -> None:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if old in src:
            cell["source"] = src.replace(old, new).splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            return
    raise SystemExit(f"FATAL: {what} anchor not found in duck-base.ipynb")


def main() -> None:
    nb = json.loads(BASE.read_text())

    # 1+2. dataset ref swap: notebook DATASET_SOURCES list.
    replace_in_cell(nb, f'"{OLD_SNAPSHOT_REF}"', f'"{NEW_SNAPSHOT_REF}"',
                    "DATASET_SOURCES snapshot ref")
    # 3. setup-command rewrite loop + post-loop assert.
    replace_in_cell(nb, RUN_LOOP_OLD, RUN_LOOP_NEW, "setup-command run loop")
    replace_in_cell(nb, RUN_LOOP_ASSERT_OLD, RUN_LOOP_ASSERT_NEW,
                    "setup-command loop tail")
    # commit-only mount probes.
    install_done = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and INSTALL_MARKER in "".join(cell["source"]):
            cell["source"] = INSTALL_NEW.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            install_done = True
    if not install_done:
        raise SystemExit("FATAL: install cell (INSTALL_MARKER) not found")
    replace_in_cell(nb, ENVFILES_OLD, ENVFILES_NEW, "environment_files line")

    # Sanity: no stale refs to the old model anywhere in the code cells.
    joined = "\n".join("".join(c["source"]) for c in nb["cells"]
                       if c["cell_type"] == "code")
    assert OLD_SNAPSHOT_REF not in joined, "stale vrfai snapshot ref survived"
    assert NEW_SNAPSHOT_REF in joined, "new snapshot ref missing"
    assert "Qwen/Qwen3.8-27B-FP8" in joined, "new served name missing"

    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"built {OUT}")
    print(f"canonical hash: {ledg_hash(nb)}")


if __name__ == "__main__":
    main()
