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
    print(f"canonical hash: {ledg_hash(nb)}")


if __name__ == "__main__":
    main()
