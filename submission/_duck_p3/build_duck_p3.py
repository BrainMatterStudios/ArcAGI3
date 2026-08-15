#!/usr/bin/env python3
"""Build submission/_duck_p3/duck-p3.ipynb.

SINGLE-VARIABLE arm: shipped duck-base v2 bytes + exactly one behavioural
delta — the "minimize actions" own-goal line in the system prompt replaced by
an exploration-neutral line, rebound in BOTH namespaces (the prompts module
AND tool_agent's import-time by-value binding — the silent-no-op bug found in
the duck-mem review, tool_agent.py:17-19 vs :352).

WHY THIS LEVER, ALONE (2026-08-14): the duck-mem P1-P4 bundle measured
NEGATIVE pooled offline (-0.29), but attribution across its four levers is
unknown. P3 is the one member whose direction rests on measured facts (8/10
completed levels are efficiency-cap-BOUND — an action-minimization instruction
buys nothing on cleared levels and suppresses the exploration that unlocks
depth, worth 3x per level step) and whose mechanism cannot degrade throughput
(pure prompt text; no trimming, no batch stops, no estimator change).
Live n=1 reads under the standard band rule; this is a lottery ticket WITH a
falsifiable hypothesis, chosen over an idle slot per Ahmed's directive.

Usage:  .venv/bin/python submission/_duck_p3/build_duck_p3.py
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
OUT = Path(__file__).parent / "duck-p3.ipynb"

# duck-base hardcodes /kaggle/input/competitions/<slug>; commit machines may
# mount at /kaggle/input/<slug> instead (proven by duck-mem v1/v2 commit ERRORs
# AND the duck-p3 v1 commit ERROR 2026-08-15 — a GPU commit machine used the
# short form too, so this is not CPU-only). Probe-both cells ported verbatim
# from build_duck_mem.py (its v4 GPU commit + scored run proved them); scored
# reruns use the gateway and never touch the environment_files branch.
INSTALL_MARKER = "# Install the ARC runtime from the bundled competition wheels."
INSTALL_NEW = '''\
# duck-p3: install the ARC runtime from the bundled competition wheels.
# Probe both mount forms; fail loudly naming both (mount form varies by machine).
_p3_wheel_dirs = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
]
_p3_wheels = next((p for p in _p3_wheel_dirs if Path(p).is_dir()), None)
if _p3_wheels is None:
    raise RuntimeError(
        f"arc_agi_3_wheels directory not found at either mount form: {_p3_wheel_dirs}")
print(f"[duck-p3] installing arc-agi from {_p3_wheels}", flush=True)
_p3_pip = subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--no-index", "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links", _p3_wheels,
        "arc-agi",
    ],
    capture_output=True, text=True,
)
if _p3_pip.stdout.strip():
    print(_p3_pip.stdout, flush=True)
if _p3_pip.stderr.strip():
    print(f"[duck-p3] pip stderr:\\n{_p3_pip.stderr}", flush=True)
if _p3_pip.returncode != 0:
    raise RuntimeError(f"[duck-p3] pip install of arc-agi FAILED (rc={_p3_pip.returncode})")
print(f"[duck-p3] arc-agi installed from {_p3_wheels}", flush=True)\
'''

ENVFILES_OLD = ('    competition_env_files = str(Path("/kaggle/input/competitions/'
                'arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")')
ENVFILES_NEW = '''\
    _p3_env_candidates = [
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
        "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files",
    ]
    competition_env_files = next(
        (p for p in _p3_env_candidates if Path(p).is_dir()), _p3_env_candidates[0])\
'''

HOOK_OLD = (
    "# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.\n"
    "# Example:\n"
    "# bm.label = f\"{bm.label}-debug\""
)

HOOK_NEW = '''\
# duck-p3: SINGLE-VARIABLE arm — neutralize the "minimize actions" own-goal.
# Shipped duck-base v2 in every other byte. Rationale + evidence in
# submission/_duck_p3/build_duck_p3.py (this cell is generated).
_P3_OLD = "- Optimize for as few in-game actions as possible while still being reliable.\\n"
_P3_NEW = "- Completing levels is the goal; exploration that reveals mechanics is worth its actions.\\n"
from inference.agent import prompts as _p3_prompts
from inference.agent import tool_agent as _p3_ta
assert _P3_OLD in _p3_prompts.GAME_OVERVIEW_ADDENDUM, "[duck-p3] anchor line MISSING from prompts"
_p3_replacement = _p3_prompts.GAME_OVERVIEW_ADDENDUM.replace(_P3_OLD, _P3_NEW)
_p3_prompts.GAME_OVERVIEW_ADDENDUM = _p3_replacement
_p3_ta.GAME_OVERVIEW_ADDENDUM = _p3_replacement
assert _P3_OLD not in _p3_ta.GAME_OVERVIEW_ADDENDUM, "[duck-p3] tool_agent rebind FAILED"
assert _P3_NEW in _p3_ta.GAME_OVERVIEW_ADDENDUM, "[duck-p3] replacement MISSING after rebind"
print("[duck-p3] minimize-actions line neutralized in BOTH namespaces: OK", flush=True)\
'''


def ledg_hash(nb: dict) -> str:
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()


def main() -> None:
    nb = json.loads(BASE.read_text())
    hook_idx = None
    install_idx = None
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src == HOOK_OLD:
            hook_idx = idx
        elif INSTALL_MARKER in src:
            install_idx = idx
    if hook_idx is None:
        raise SystemExit("FATAL: empty customization-hook cell not found in duck-base.ipynb")
    if install_idx is None:
        raise SystemExit("FATAL: install cell (INSTALL_MARKER) not found in duck-base.ipynb")
    nb["cells"][hook_idx]["source"] = HOOK_NEW.splitlines(keepends=True)
    nb["cells"][hook_idx]["outputs"] = []
    nb["cells"][hook_idx]["execution_count"] = None
    nb["cells"][install_idx]["source"] = INSTALL_NEW.splitlines(keepends=True)
    nb["cells"][install_idx]["outputs"] = []
    nb["cells"][install_idx]["execution_count"] = None

    replaced_env = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if ENVFILES_OLD in src:
            cell["source"] = src.replace(ENVFILES_OLD, ENVFILES_NEW).splitlines(keepends=True)
            replaced_env = True
    if not replaced_env:
        raise SystemExit("FATAL: environment_files line not found in duck-base.ipynb")

    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"built {OUT}")
    print(f"canonical hash: {ledg_hash(nb)}")


if __name__ == "__main__":
    main()
