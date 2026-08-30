#!/usr/bin/env python3
"""validate_tp_smoke.py — structural + syntax checks on the built notebook."""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
NB = HERE / "arc3-tp-smoke.ipynb"


def main() -> int:
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NB
    nb = json.loads(nb_path.read_text())
    cells = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(cells)
    checks = {
        "graft install asserted": 'assert _tp_status == "throughput: OK"' in joined,
        "two phases": joined.count(", GAMES_25, ") == 2,
        "phase env set": joined.count("'TP_ENABLE': ") == 2 and joined.count("'TP2_ENABLE': ") == 2 and joined.count("'TP4_ENABLE': ") == 2,
        "all grafts install asserted": 'assert _tc_status == "control: OK"' in joined and 'assert _te_status == "explore: OK"' in joined and 'assert _tm_status == "emission: OK"' in joined,
        "scored branch intact": "bm.games = _competition_games()" in joined and "KAGGLE_IS_COMPETITION_RERUN" in joined,
        "metrics scrape": "vllm:prefix_cache_hits_total" in joined,
        "phase begin/end wired": "_tp_phase_begin(_phase_name)" in joined and "_tp_phase_end(_phase_name" in joined,
        "report": "TP SMOKE READ" in joined,
        "graft source embedded": "def _patch_trim" in joined and "def time_guard_per_game_s" in joined and "def run_probe" in joined and "class FrontierExplorer" in joined,
        "25 games": joined.count("GAMES_25 = [") == 1,
    }
    ok = True
    for name, passed in checks.items():
        print(("OK  " if passed else "FAIL"), name)
        ok = ok and passed
    # per-cell syntax check (top-level await is legal in notebooks; wrap it)
    for i, src in enumerate(cells):
        try:
            ast.parse(src, feature_version=(3, 12)) if "await " not in src else \
                ast.parse("async def _cell():\n" + "".join("    " + line for line in src.splitlines(True)))
        except SyntaxError as exc:
            print("FAIL syntax in code cell", i, exc)
            ok = False
    print("VALIDATE", "OK" if ok else "FAIL", "cells:", len(nb["cells"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
