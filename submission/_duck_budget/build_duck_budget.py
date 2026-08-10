#!/usr/bin/env python3
"""Build submission/_duck_budget/duck-budget.ipynb.

SINGLE-VARIABLE arm: shipped duck-base v2 bytes with exactly one behavioural
delta — the per-game wall-clock budget raised 7920 s -> 13200 s (2.2 h -> 3.67 h).

WHY (evidence chain, 2026-08-10):
  * 9/10 rig games at eval geometry end with <6 min of the 7920 s budget left
    (docs/FINDING-2026-08-02-efficiency-headroom-is-exhausted.md side data) —
    wall-clock is the binding budget, and depth (level+1 multiplier) is what
    the min-cap scoring pays.
  * The sonpham-org/arc-3 fork validated a 4 h/game cap on a REAL 55-game
    Kaggle timing run: 8h00m total at concurrency 28 (their commits around
    "the 1.12 submission"). Same brain, same serving, same concurrency.
  * 13200 s is chosen over 14400 s because it survives even a 3-wave world
    (>56 hidden games): 3 x 13200 s + ~35 min serve setup = ~11.7 h < 12 h.
    In the evidenced 2-wave world: 2 x 13200 + setup = ~7.9 h, wide margin.

MECHANISM: HarnessSolver.max_runtime_s_per_game is read live at every per-game
budget check (scored bundle inference/framework/solver.py:214-236), and the
hook cell is duck-base's designed override point for `bm.solver`. No taaf code
is modified; one attribute assignment with a hard anchor assert.

Usage:  .venv/bin/python submission/_duck_budget/build_duck_budget.py
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
OUT = Path(__file__).parent / "duck-budget.ipynb"

HOOK_OLD = (
    "# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.\n"
    "# Example:\n"
    "# bm.label = f\"{bm.label}-debug\""
)

HOOK_NEW = '''\
# duck-budget: SINGLE-VARIABLE arm — raise the per-game wall-clock budget.
# Shipped duck-base v2 in every other byte. Evidence + timing math in
# submission/_duck_budget/build_duck_budget.py (this cell is generated).
_db_prev = bm.solver.max_runtime_s_per_game
assert _db_prev == 7920.0, f"[duck-budget] expected shipped 7920.0, got {_db_prev!r}"
bm.solver.max_runtime_s_per_game = 13200.0
print(f"[duck-budget] max_runtime_s_per_game {_db_prev} -> "
      f"{bm.solver.max_runtime_s_per_game}", flush=True)\
'''


def ledg_hash(nb: dict) -> str:
    """Ledger method: sha256 over '\\n'.join(code-cell sources), no trailing newline."""
    sources = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
    return hashlib.sha256("\n".join(sources).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    nb = json.loads(BASE.read_text())
    hook_idx = None
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "".join(cell["source"]) == HOOK_OLD:
            hook_idx = idx
            break
    if hook_idx is None:
        raise SystemExit("FATAL: empty customization-hook cell not found in duck-base.ipynb")
    nb["cells"][hook_idx]["source"] = HOOK_NEW.splitlines(keepends=True)
    nb["cells"][hook_idx]["outputs"] = []
    nb["cells"][hook_idx]["execution_count"] = None

    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"built {OUT}")
    print(f"code-cell hash (ledger method, 16 chars): {ledg_hash(nb)}")


if __name__ == "__main__":
    main()
