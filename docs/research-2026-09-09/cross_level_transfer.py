#!/usr/bin/env python3
"""Does a backtest-GREEN executable model predict the NEXT level of the same game?

WHY: the whole economics of the executable-model lane (Stage-1, Polyphony, A2)
turns on whether one verified model amortises across a game's levels. A model
costs ~65k completion tokens = 79% of a whole game's decode budget. If it works
for one level only, it can never pay for itself. If it transfers, the cost is
divided by however many levels it covers and the lane reopens.

RESULT 2026-09-09: it does NOT transfer. On the level it was built on, every
model scores 20/20. On the neighbouring level of the SAME game: 10%, 10%, 0%,
45%. Reading the model source shows why -- they encode literal row/column
rectangles of that level's board, not the game's mechanic. Backtest-green is
therefore weaker evidence of understanding than it looks.

Run: .venv/bin/python docs/research-2026-09-09/cross_level_transfer.py
"""
import importlib.util
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "stage1-own-transitions"


def load(name):
    spec = importlib.util.spec_from_file_location("m_" + name, BASE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def backtest(mod, tr):
    """Teacher-forced replay: the model always gets the REAL previous board, so an
    error at step k cannot cascade into step k+1. This is the same contract the
    Stage-0/Stage-1 instrument used, and it is deliberately generous."""
    grid = [r[:] for r in tr["entry_grid"]]
    state = mod.init_state([r[:] for r in tr["entry_grid"]]) if hasattr(mod, "init_state") else None
    ok = 0
    for t in tr["transitions"]:
        try:
            if hasattr(mod, "predict"):
                pred, _flags, state = mod.predict(state, [r[:] for r in grid],
                                                  t["action"], t.get("x"), t.get("y"))
            else:
                pred, _flags = mod.step([r[:] for r in grid], t["action"], t.get("x"), t.get("y"))
        except Exception:
            pred = None            # a crash is simply a wrong prediction here
        if pred == t["grid"]:
            ok += 1
        grid = [r[:] for r in t["grid"]]
    return ok, len(tr["transitions"])


def main():
    levels = ("dc22L1", "dc22L2")
    trans = {g: json.loads((BASE / f"{g}_transitions.json").read_text()) for g in levels}
    models = {f"{g}_{d}": load(f"{g}_green_{d}") for g in levels for d in ("draw1", "draw2")}

    print("CROSS-LEVEL TRANSFER of backtest-green executable models")
    print(f"{'model':>18} | " + " | ".join(f"{t:>12}" for t in trans))
    for name, mod in models.items():
        cells = []
        for tr in trans.values():
            ok, n = backtest(mod, tr)
            cells.append(f"{ok:>3}/{n:<3} {100 * ok / n:3.0f}%")
        print(f"{name:>18} | " + " | ".join(f"{c:>12}" for c in cells))
    print("\ndiagonal = the level it was verified on; off-diagonal = transfer")


if __name__ == "__main__":
    main()
