"""Backtest a candidate world-model file against recorded level-0 transitions.

Contract (as used by the frontier harness files):
  stateful : init_state(entry_grid) -> state ; predict(state, grid, action, x=None, y=None) -> (grid, flags, state)
  stateless: step(grid, action, x=None, y=None) -> (grid, flags)
  flags = {"level_up": bool, "dead": bool, "win": bool}; optional is_goal(state, grid).
  A module-level global ENTRY_GRID (the level's entry grid) is injected before exec.

Scoring (mirrors the frontier harness's run_backtest, made stricter on step 0):
  * every non-RESET transition is checked (step 0 included; the frontier harness skipped it)
  * grid compared exactly on non-terminal steps; on terminal steps (level_up/win/dead
    recorded) only the flags are compared (the post-level-up grid is the NEXT level's board)
  * flags compared on every checked step
  * RESET (action 0) transitions are skipped; the model is re-synced to the recorded
    post-reset grid (reset_policy: "reinit" -> init_state again, "continue" -> keep state)
  * teacher forcing: after each step the recorded grid becomes the next input grid
  * an exception = mismatch at that index, traceback captured
"""
import copy, json, sys, traceback, types

FLAG_KEYS = ("level_up", "dead", "win")


def load_model(path, entry_grid):
    src = open(path).read()
    mod = types.ModuleType("candidate_world_model")
    mod.__dict__["ENTRY_GRID"] = copy.deepcopy(entry_grid)
    mod.__dict__["__name__"] = "candidate_world_model"
    exec(compile(src, path, "exec"), mod.__dict__)
    if hasattr(mod, "predict"):
        kind = "stateful"
    elif hasattr(mod, "step"):
        kind = "stateless"
    else:
        raise RuntimeError("model exposes neither predict() nor step()")
    return mod, kind


def _to_grid(g):
    try:
        import numpy as np
        if isinstance(g, np.ndarray):
            g = g.tolist()
    except ImportError:
        pass
    return [[int(v) for v in row] for row in g]


def _flags(info):
    if info is None:
        info = {}
    if not isinstance(info, dict):
        info = dict(info)
    return {k: bool(info.get(k, False)) for k in FLAG_KEYS}


def _diff(exp, got, limit=60):
    cells = []
    if got is None or len(got) != len(exp) or any(len(a) != len(b) for a, b in zip(exp, got)):
        return None, cells
    n = 0
    for r in range(len(exp)):
        er, gr = exp[r], got[r]
        for c in range(len(er)):
            if er[c] != gr[c]:
                n += 1
                if len(cells) < limit:
                    cells.append((r, c, er[c], gr[c]))
    return n, cells


def run(path, entry_grid, transitions, reset_policy="reinit", max_mismatches=None):
    result = {"path": path, "matched": 0, "total": 0, "skipped": 0, "mismatches": [],
              "first_mismatch": None, "kind": None, "load_error": None}
    try:
        mod, kind = load_model(path, entry_grid)
    except Exception:
        result["load_error"] = traceback.format_exc()
        result["total"] = sum(1 for t in transitions if t["action"] != 0)
        result["first_mismatch"] = {"index": -1, "action": None, "error": "model failed to load",
                                    "traceback": result["load_error"]}
        return result
    result["kind"] = kind
    state = None
    if kind == "stateful":
        try:
            state = mod.init_state(copy.deepcopy(entry_grid))
        except Exception:
            result["load_error"] = traceback.format_exc()
            result["total"] = sum(1 for t in transitions if t["action"] != 0)
            result["first_mismatch"] = {"index": -1, "action": None, "error": "init_state failed",
                                        "traceback": result["load_error"]}
            return result
    before = copy.deepcopy(entry_grid)
    for t in transitions:
        a, x, y = t["action"], t.get("x"), t.get("y")
        if a == 0:
            result["skipped"] += 1
            before = copy.deepcopy(t["grid"])
            if kind == "stateful" and reset_policy == "reinit":
                try:
                    state = mod.init_state(copy.deepcopy(before))
                except Exception:
                    pass
            continue
        result["total"] += 1
        exp_flags = {k: bool(t.get(k, False)) for k in FLAG_KEYS}
        terminal = any(exp_flags.values())
        mm = None
        got_grid = None
        try:
            if kind == "stateful":
                out = mod.predict(copy.deepcopy(state), copy.deepcopy(before), a, x, y)
                if not (isinstance(out, (tuple, list)) and len(out) == 3):
                    raise TypeError(f"predict must return (grid, flags, state); got {type(out).__name__} of len {len(out) if hasattr(out,'__len__') else '?'}")
                got_grid, info, new_state = out
            else:
                out = mod.step(copy.deepcopy(before), a, x, y)
                if not (isinstance(out, (tuple, list)) and len(out) == 2):
                    raise TypeError(f"step must return (grid, flags); got {type(out).__name__} of len {len(out) if hasattr(out,'__len__') else '?'}")
                got_grid, info = out
                new_state = None
            got_grid = _to_grid(got_grid)
            got_flags = _flags(info)
        except Exception:
            mm = {"index": t["index"], "action": a, "x": x, "y": y, "kind": "exception",
                  "traceback": traceback.format_exc()[-3000:], "expected_flags": exp_flags}
            new_state = state
        if mm is None:
            flag_ok = got_flags == exp_flags
            if terminal:
                grid_ok, ncells, cells = True, 0, []
            else:
                ncells, cells = _diff(t["grid"], got_grid)
                grid_ok = (ncells == 0)
            if not (flag_ok and grid_ok):
                mm = {"index": t["index"], "action": a, "x": x, "y": y,
                      "kind": ("flags" if grid_ok else "grid") + ("" if flag_ok else "+flags"),
                      "expected_flags": exp_flags, "got_flags": got_flags,
                      "cells_differ": ncells, "cells": cells,
                      "shape_error": (ncells is None)}
        if mm is None:
            result["matched"] += 1
        else:
            result["mismatches"].append(mm)
            if result["first_mismatch"] is None:
                result["first_mismatch"] = mm
            if max_mismatches and len(result["mismatches"]) >= max_mismatches:
                pass
        # teacher forcing
        before = copy.deepcopy(t["grid"])
        state = new_state
    result["green"] = (result["matched"] == result["total"] and result["total"] > 0)
    return result


def summarize(res):
    s = f"backtest: {res['matched']}/{res['total']} transitions fully correct ({res['skipped']} reset(s) skipped); kind={res['kind']}"
    if res.get("load_error"):
        s += "\nLOAD ERROR:\n" + res["load_error"][-2000:]
    return s


def run_subprocess(model_path, trans_path, out_json, timeout_s=180, python=sys.executable):
    """Run the backtest in a fresh interpreter (guards against hangs / crashes in candidate code)."""
    import subprocess, os
    cmd = [python, os.path.abspath(__file__), model_path, trans_path, "reinit", "--json", out_json]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        rec = json.load(open(trans_path))
        total = sum(1 for t in rec["transitions"] if t["action"] != 0)
        res = {"path": model_path, "matched": 0, "total": total, "skipped": 0, "green": False, "kind": None,
               "mismatches": [], "first_mismatch": {"index": -1, "action": None, "kind": "timeout",
               "traceback": f"backtest exceeded {timeout_s}s (candidate hangs or is far too slow)"}}
        json.dump(res, open(out_json, "w"))
        return res
    if not os.path.exists(out_json):
        rec = json.load(open(trans_path))
        total = sum(1 for t in rec["transitions"] if t["action"] != 0)
        res = {"path": model_path, "matched": 0, "total": total, "skipped": 0, "green": False, "kind": None,
               "mismatches": [], "first_mismatch": {"index": -1, "action": None, "kind": "crash",
               "traceback": (p.stderr or p.stdout)[-3000:]}}
        json.dump(res, open(out_json, "w"))
        return res
    return json.load(open(out_json))


if __name__ == "__main__":
    model_path, trans_path = sys.argv[1], sys.argv[2]
    policy = sys.argv[3] if len(sys.argv) > 3 else "reinit"
    rec = json.load(open(trans_path))
    res = run(model_path, rec["entry_grid"], rec["transitions"], reset_policy=policy)
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), default=str)
    print(summarize(res))
    for mm in res["mismatches"][:5]:
        print(json.dumps({k: v for k, v in mm.items() if k != "cells"}, default=str)[:600])
        if mm.get("cells"):
            print("  cells (r,c,exp,got):", mm["cells"][:20])
