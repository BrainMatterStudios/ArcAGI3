#!/usr/bin/env python3
"""Build submission/_duck_mem/duck-mem.ipynb.

Takes the SHIPPED duck-base v2 notebook (canonical code-cell hash 886dbc8a..., the
live pinned submission config, NO patches) and replaces only its empty "Customization
hook" cell with the duck-mem patch stack (harness_mem.py, self-contained payload).

Everything else is byte-identical to duck-base v2: model, sampling, concurrency,
per-game budget, game list, submission path, CPU-safe-commit guard.

Usage:  .venv/bin/python submission/_duck_mem/build_duck_mem.py
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
PATCHES = Path(__file__).parent / "harness_mem.py"
OUT = Path(__file__).parent / "duck-mem.ipynb"

HOOK_OLD = (  # the exact empty hook cell shipped in base v2 (no trailing newline)
    "# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.\n"
    "# Example:\n"
    "# bm.label = f\"{bm.label}-debug\""
)

# duck-base hardcodes /kaggle/input/competitions/<slug>; the competition data
# mounts at EITHER that OR /kaggle/input/<slug> depending on the machine (CPU
# commit machines use the second form — proven by the v1 commit ERROR 2026-08-11
# and, earlier, by 3 rig-candidate ERRORs on 2026-08-08). This mount-probing
# install cell is the rig's proven fix (build_patch_closure.INSTALL_CELL
# lineage); on machines with the original form it behaves identically, so the
# scored rerun is unchanged.
INSTALL_MARKER = "# Install the ARC runtime from the bundled competition wheels."
INSTALL_NEW = '''\
# duck-mem: install the ARC runtime from the bundled competition wheels.
# Probe both mount forms; fail loudly naming both (mount form varies by machine).
_dm_wheel_dirs = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
    "/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
]
_dm_wheels = next((p for p in _dm_wheel_dirs if Path(p).is_dir()), None)
if _dm_wheels is None:
    raise RuntimeError(
        f"arc_agi_3_wheels directory not found at either mount form: {_dm_wheel_dirs}")
print(f"[duck-mem] installing arc-agi from {_dm_wheels}", flush=True)
_dm_pip = subprocess.run(
    [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--no-index", "--no-warn-conflicts",
        "--disable-pip-version-check",
        "--find-links", _dm_wheels,
        "arc-agi",
    ],
    capture_output=True, text=True,
)
if _dm_pip.stdout.strip():
    print(_dm_pip.stdout, flush=True)
if _dm_pip.stderr.strip():
    print(f"[duck-mem] pip stderr:\\n{_dm_pip.stderr}", flush=True)
if _dm_pip.returncode != 0:
    raise RuntimeError(f"[duck-mem] pip install of arc-agi FAILED (rc={_dm_pip.returncode})")
print(f"[duck-mem] arc-agi installed from {_dm_wheels}", flush=True)\
'''


def ledg_hash(nb: dict) -> str:
    """Ledger method: sha256 over '\n'.join(code-cell sources), no trailing newline."""
    sources = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    nb = json.loads(BASE.read_text())
    patches_src = PATCHES.read_text()
    hook_new = (
        "# duck-mem: memory + anti-waste patch stack (see harness_mem.py).\n"
        "# Runs after the bundle loads and BEFORE bm.run() reads setup_commands.json,\n"
        "# so P6 can edit the serving config in place. Fail-safe by construction.\n"
        "# json/subprocess/Path are already bound in the notebook namespace by cell 8.\n"
        + patches_src
        + "\n"
        + "\n"
        + "apply_all(BUNDLE_DIR)\n"
    )
    hook_idx = None
    install_idx = None
    for idx, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if src == HOOK_OLD:
            hook_idx = idx
        elif INSTALL_MARKER in src:
            install_idx = idx
    if hook_idx is None:
        raise SystemExit("FATAL: empty customization-hook cell not found in duck-base.ipynb")
    if install_idx is None:
        raise SystemExit("FATAL: install cell (INSTALL_MARKER) not found in duck-base.ipynb")
    nb["cells"][hook_idx]["source"] = hook_new.splitlines(keepends=True)
    nb["cells"][hook_idx]["outputs"] = []
    nb["cells"][hook_idx]["execution_count"] = None
    nb["cells"][install_idx]["source"] = INSTALL_NEW.splitlines(keepends=True)
    nb["cells"][install_idx]["outputs"] = []
    nb["cells"][install_idx]["execution_count"] = None

    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"built {OUT}")
    print(f"code-cell hash (ledger method, 16 chars): {ledg_hash(nb)}")
    print(f"hook cell {hook_idx} replaced; other code cells untouched: "
          f"{sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} total")


if __name__ == "__main__":
    main()