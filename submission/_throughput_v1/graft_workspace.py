"""Persistent workspace + transition log + verifier (WS / Track A2, 2026-09-09).

WHY THIS EXISTS, AND WHAT ALREADY PROVED IT
-------------------------------------------
A1 (graft_carry) established that persistent *prose* knowledge is dead: the model
wrote excellent compacted notes and the 25-game wave read 41 levels (+0.71 sd, in
band) with 0 of 12 never-passed walls. Stage-0 and Stage-1 established the
opposite for *executable* knowledge on this brain:

  * Stage-0 (docs/research-2026-09-09/PREREG-stage0-rerun-flashnext.md): given a
    frontier agent's recorded level-0 transitions plus a verifier, Flash-Next
    emitted a green executable world model on 2 of 3 games - cn04 18/18
    (0->1->2->1->3->6->9->17->18), a game the 27B never produced one file for.
  * Stage-1 (PREREG-stage1-own-transitions.md): given OUR OWN agent's transitions
    from a level it was STUCK on (dc22 L2, 85 actions, never cleared), it produced
    a green 20/20 model in 2-3 calls, in BOTH draws, and likewise for the control.
    The models are mechanistic (3-5 KB, no embedded grids, no index tricks).

So the brain can build a verified executable model of a wall from the agent's own
observations. What it has never had, live, is (a) somewhere to keep that model,
(b) the transition log to build it from, and (c) a verifier to check it against.
This graft supplies exactly those three and nothing else; what to DO with a
verified model is left to the model, which already has a Python sandbox.

WHAT IT ADDS (all host-side; the sandbox bootstrap and every stock byte untouched)
----------------------------------------------------------------------------------
1. TRANSITION LOG, captured automatically. Every executed action is recorded as
   {index, action, x, y, grid, level_up, dead, win, level} - the SAME schema the
   Stage-0/1 instrument used, so the verifier below is the instrument we validated.
   Captured by wrapping ToolAgent._compact_action_result, which the stock calls
   once per action immediately AFTER step_env has appended the post-action frame
   to the live session's history_entries (the solver keeps that list untrimmed),
   so the after-grid is history_entries[-1].frame.grid and the before-grid [-2].

2. `backtest` TOOL. Arguments {code, level?}. Replays the candidate against this
   game's recorded transitions for that level, teacher-forced, in an ISOLATED
   subprocess (argv list, no shell, python -I -S, temp dir, hard timeout) under
   the Stage-0 contract: stateful `init_state(entry)` +
   `predict(state, grid, action, x, y) -> (grid, flags, state)`, or stateless
   `step(grid, action, x, y) -> (grid, flags)`; ENTRY_GRID injected as a global.
   Returns a compact {matched, total, green, first_mismatch{index, action, kind,
   cells}}. Running model-authored code is inherent to a verifier; it is confined
   exactly as the stock python tool confines the model's own snippets.

3. PERSISTENT WORKSPACE. A `workspace` tool (op=save|load|list|delete) keeps named
   text files for the life of the game, and every python call receives them as a
   WORKSPACE dict literal plus TRANSITIONS (the log) in a preamble prepended to the
   model's snippet, so a saved model can be re-run inside the sandbox. Saving is a
   tool call rather than a sandbox function because the stock sandbox returns only
   stdout/result/action-results and cannot hand arbitrary state back to the host.

WHY A PREAMBLE AND NOT A PATCHED SANDBOX: the sandbox bootstrap is a string
constant inside the stock file and its runtime_globals are rebuilt per call from a
fixed key set, so extra payload keys are never exposed. Prepending to the model's
own code is the only injection point that leaves stock bytes alone.

DESIGN CONSTRAINT CARRIED FROM STAGE-1: the iterate-against-a-verifier loop
self-starves inside the 32,768 window when candidate+feedback rounds accumulate in
the chat history (prompt 7.7k -> 11.9k, output 24.6k -> 20.3k, a call length-capped
with no code). So verifier results are returned COMPACT (a first mismatch, not a
transcript) and candidates live in the workspace, not in the history.

FLAGS (read at call time; WS_ENABLE=0 or not installed = stock behaviour exactly)
  WS_ENABLE [1], WS_MAX_FILES [12], WS_MAX_FILE_CHARS [20000],
  WS_BACKTEST_TIMEOUT [30], WS_LOG_MAX [400], WS_PREAMBLE [1],
  WS_PREAMBLE_MAX_CHARS [12000], WS_MISMATCH_CELLS [12].

TELEMETRY: "[HARNESS WS]" transcript sections -
  "[WS-BACKTEST] game=<stem> level=<L> matched=<m>/<t> green=<0|1> code_chars=<n> ms=<t>"
  "[WS-SAVE] game=<stem> name=<f> chars=<n> files=<k>"
status(): python_calls, transitions_logged, backtests, backtests_green, saves,
loads, lists, deletes, preambles, preamble_chars_total, backtest_ms_total,
green_share, errors, skips, per_game.

Conventions (graft_probe / graft_carry): module-level _STATE/_STOCK, install()
rebinds ToolAgent methods only, fail-open try/except around every graft branch,
no threads, no writes outside the transcript. install() -> "workspace: OK".
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_PER_GAME_KEYS = ("python_calls", "transitions_logged", "backtests", "backtests_green", "saves", "loads",
                  "lists", "deletes", "preambles", "preamble_chars_total", "backtest_ms_total",
                  "direct_calls", "direct_attempts", "direct_green")
_STATE: dict[str, Any] = {"installed": False, "errors": 0, "skips": {}, "per_game": {}}
for _k in _PER_GAME_KEYS:
    _STATE[_k] = 0
_STOCK: dict[str, Any] = {}
_LOCK = threading.Lock()
_OFF = {"0", "false", "no", "off"}

DEFAULT_MAX_FILES = 12
DEFAULT_MAX_FILE_CHARS = 20000
DEFAULT_BACKTEST_TIMEOUT = 30
DEFAULT_LOG_MAX = 400
DEFAULT_PREAMBLE_MAX_CHARS = 12000
DEFAULT_MISMATCH_CELLS = 12

# The stock prompts assert, in the SYSTEM prompt once and in the USER prompt EVERY turn (44x in a
# 60-call run), that python is the only tool. That contradicts the tool schema this graft extends and
# is why the first keith_ws kill test got 181 python calls and ZERO backtest/workspace calls: the model
# was told the capability does not exist. These two rewrites make the prompt match the schema.
SYSTEM_ONLY_TOOL = "- The only tool is `python`; call it with one ephemeral `code` string.\n"
SYSTEM_TOOLS_LINE = ("- Three tools: `python` (one ephemeral `code` string), `backtest` (check a candidate world "
                     "model against the transitions you have actually recorded in THIS game) and `workspace` "
                     "(save/load files that persist across turns, unlike `python`).\n")
USER_ONLY_TOOL = "Only tool: `python`."
# ends with "`python`." so the stock sentence that follows (" It receives ...") still reads correctly
USER_TOOLS_LINE = ("Tools: `python` (ephemeral), `backtest` (verify a candidate world model against this game's "
                   "recorded transitions), `workspace` (files that persist across turns).\n`python`.")

TRANSCRIPT_LABEL = "HARNESS WS"
BACKTEST_MARK = "[WS-BACKTEST]"
SAVE_MARK = "[WS-SAVE]"
_CLICK_RE = re.compile(r"row\s*=\s*(-?\d+).*?col\s*=\s*(-?\d+)", re.S)
_ACTION_NUM_RE = re.compile(r"ACTION(\d+)")

BACKTEST_TOOL_DESC = (
    "Verify an executable world model against THIS game's own recorded transitions for one level. "
    "Your code must define either a stateful pair `init_state(entry_grid)` and "
    "`predict(state, grid, action, x, y) -> (next_grid, flags, state)`, or a stateless "
    "`step(grid, action, x, y) -> (next_grid, flags)`. `ENTRY_GRID` is injected as a module global. "
    "`action` is an int (1-5 = ACTION1..5, 6 = click with x=col, y=row, 0 = RESET); grids are lists of "
    "lists of ints. `flags` is a dict that may set level_up/dead/win. The harness replays every recorded "
    "transition teacher-forced and reports how many your model predicts exactly, plus the first mismatch. "
    "Use this to test a mechanics hypothesis against evidence instead of guessing. The verifier runs your code "
    "with the full standard library available; note that the python tool's own sandbox is more restricted, so a "
    "model you also want to RUN in python should stick to the stdlib."
)
WORKSPACE_TOOL_DESC = (
    "Persistent per-game scratch storage that survives across turns (the python sandbox does not). "
    "op='save' with name+content stores a file; op='load' returns one; op='list' names them; "
    "op='delete' removes one. Saved files are also injected into every python call as the dict "
    "WORKSPACE, so a saved model can be run inside python."
)


def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("WS_ENABLE", "1").lower() not in _OFF


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(_env(name, str(default))))
    except ValueError:
        return default


def max_files() -> int:
    return _env_int("WS_MAX_FILES", DEFAULT_MAX_FILES, 1)


def max_file_chars() -> int:
    return _env_int("WS_MAX_FILE_CHARS", DEFAULT_MAX_FILE_CHARS, 100)


def backtest_timeout() -> int:
    return _env_int("WS_BACKTEST_TIMEOUT", DEFAULT_BACKTEST_TIMEOUT, 1)


def log_max() -> int:
    return _env_int("WS_LOG_MAX", DEFAULT_LOG_MAX, 1)


def preamble_on() -> bool:
    return _env("WS_PREAMBLE", "1").lower() not in _OFF


def preamble_max_chars() -> int:
    return _env_int("WS_PREAMBLE_MAX_CHARS", DEFAULT_PREAMBLE_MAX_CHARS, 200)


def mismatch_cells() -> int:
    return _env_int("WS_MISMATCH_CELLS", DEFAULT_MISMATCH_CELLS, 0)


def _share(a: float, b: float) -> float | None:
    return None if not b else a / b


def status() -> dict[str, Any]:
    with _LOCK:
        out: dict[str, Any] = {"installed": _STATE["installed"], "enabled": enabled(), "max_files": max_files(),
                               "max_file_chars": max_file_chars(), "backtest_timeout": backtest_timeout(),
                               "log_max": log_max(), "preamble": preamble_on()}
        for k in _PER_GAME_KEYS:
            out[k] = _STATE[k]
        out["green_share"] = _share(_STATE["backtests_green"], _STATE["backtests"])
        out["errors"] = _STATE["errors"]
        out["skips"] = dict(_STATE["skips"])
        out["per_game"] = {k: dict(v) for k, v in _STATE["per_game"].items()}
        return out


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] = _STATE["skips"].get(reason, 0) + 1


def _error() -> None:
    with _LOCK:
        _STATE["errors"] += 1


def _bump(game: str, key: str, n: float = 1) -> None:
    with _LOCK:
        _STATE[key] += n
        pg = _STATE["per_game"].setdefault(game, {k: 0 for k in _PER_GAME_KEYS})
        pg[key] = pg.get(key, 0) + n


class WsState:
    """One per agent (= per game run); reset when the runtime dir (game) changes."""

    def __init__(self) -> None:
        self.runtime_dir: Any = None
        self.game: str = "?"
        self.transcript_path: Path | None = None
        self.agent_mod: Any = None
        self.files: dict[str, str] = {}
        self.log: list[dict[str, Any]] = []
        self.entry_by_level: dict[int, Any] = {}
        self.best_by_level: dict[int, str] = {}
        self.level: int = 1
        self.direct_done: set = set()        # levels a directed build has already been attempted on
        self.direct_notice: str = ""         # one-shot line telling the play loop a verified model exists

    def transitions_for(self, level: int) -> list[dict[str, Any]]:
        return [t for t in self.log if t.get("level_before") == level]


def _wstate(agent: Any, state_path: Any = None) -> WsState:
    st = getattr(agent, "_ws", None)
    if st is None:
        st = WsState()
        try:
            agent._ws = st
        except Exception:  # noqa: BLE001
            pass
    if state_path is not None:
        try:
            rd = Path(state_path).parent
            if st.runtime_dir is not None and st.runtime_dir != rd:
                fresh = WsState()
                fresh.runtime_dir = rd
                try:
                    agent._ws = fresh
                except Exception:  # noqa: BLE001
                    pass
                return fresh
            st.runtime_dir = rd
        except Exception:  # noqa: BLE001
            pass
    return st


def _game_key(agent: Any, state_path: Any, step_env: Any = None, transcript_path: Any = None) -> str:
    """The RUN STEM ("<gid>_p<draw>") when we can get it: run_regime_wave keys per-run telemetry by stem,
    so returning a bare game id would silently orphan every counter. Falls back to the game id, then the
    runtime dir. `step_env` is passed explicitly from the analyze() wrapper because the stock sets
    self._step_env_callback only AFTER our wrapper has run."""
    tp = transcript_path if transcript_path is not None else getattr(getattr(agent, "_ws", None), "transcript_path", None)
    if tp is not None:
        try:
            stem = Path(tp).stem
            if stem:
                return stem
        except Exception:  # noqa: BLE001
            pass
    sess = getattr(step_env, "__self__", None) or getattr(getattr(agent, "_step_env_callback", None), "__self__", None)
    try:
        gid = str(sess.game.game_run.game_id or "")
        if gid:
            return gid
    except Exception:  # noqa: BLE001
        pass
    try:
        return Path(state_path).parent.name or "?"
    except Exception:  # noqa: BLE001
        return "?"


def _write_marker(agent_mod: Any, path: Any, line: str) -> None:
    if path is None or agent_mod is None:
        return
    try:
        agent_mod._append_transcript_section(Path(path), TRANSCRIPT_LABEL, line)
    except Exception:  # noqa: BLE001
        _error()


def action_to_int(payload: dict) -> tuple:
    """(action int, x=col, y=row) matching the Stage-0/1 schema: 1-5 = ACTION1..5, 6 = click, 0 = RESET."""
    name = str(payload.get("action_name") or payload.get("action_display") or "").strip()
    x = y = None
    m = _CLICK_RE.search(str(payload.get("action_display") or ""))
    if m:
        y, x = int(m.group(1)), int(m.group(2))
    up = name.upper()
    if up.startswith("RESET"):
        return 0, x, y
    if "MOUSE" in up or "CLICK" in up:
        return 6, x, y
    am = _ACTION_NUM_RE.search(up)
    if am:
        return int(am.group(1)), x, y
    return (0 if not up else -1), x, y


def _grid_of(frame: Any) -> Any:
    g = getattr(frame, "grid", None)
    if g is None:
        return None
    return [list(row) for row in g]


_RUNNER_SRC = '\n'.join([
    "import json, sys",
    "spec = json.load(open(sys.argv[1]))",
    "g = {'ENTRY_GRID': spec['entry']}",
    "compiled = compile(spec['code'], '<world_model>', 'exec')",
    "eval(compiled, g, g)",
    "init_state, predict, step = g.get('init_state'), g.get('predict'), g.get('step')",
    "stateful = callable(init_state) and callable(predict)",
    "if not stateful and not callable(step):",
    "    print(json.dumps({'error': 'no init_state/predict pair and no step(grid, action, x, y)'})); sys.exit(0)",
    "state = init_state(spec['entry']) if stateful else None",
    "grid = spec['entry']",
    "out = {'matched': 0, 'total': len(spec['transitions']), 'first_mismatch': None}",
    "for t in spec['transitions']:",
    "    exp, terminal = t['grid'], bool(t['level_up'] or t['win'])",
    "    try:",
    "        if stateful:",
    "            got, flags, state = predict(state, grid, t['action'], t['x'], t['y'])",
    "        else:",
    "            got, flags = step(grid, t['action'], t['x'], t['y'])",
    "    except Exception as e:",
    "        out['first_mismatch'] = {'index': t['index'], 'action': t['action'], 'kind': 'exception',",
    "                                 'detail': (type(e).__name__ + ': ' + str(e))[:300]}",
    "        break",
    "    flags = flags if isinstance(flags, dict) else {}",
    "    bad = []",
    "    for k in ('level_up', 'dead', 'win'):",
    "        if bool(flags.get(k)) != bool(t[k]):",
    "            bad.append(k + ': predicted ' + str(bool(flags.get(k))) + ', actual ' + str(bool(t[k])))",
    "    cells = []",
    "    if not terminal:",
    "        try:",
    "            for r in range(len(exp)):",
    "                for c in range(len(exp[r])):",
    "                    if got[r][c] != exp[r][c]:",
    "                        cells.append([r, c, exp[r][c], got[r][c]])",
    "                        if len(cells) > 400: raise StopIteration",
    "        except StopIteration:",
    "            pass",
    "        except Exception as e:",
    "            out['first_mismatch'] = {'index': t['index'], 'action': t['action'], 'kind': 'bad_grid',",
    "                                     'detail': (type(e).__name__ + ': ' + str(e))[:200]}",
    "            break",
    "    if bad or cells:",
    "        out['first_mismatch'] = {'index': t['index'], 'action': t['action'],",
    "                                 'kind': 'flags' if bad and not cells else 'grid',",
    "                                 'flags': bad, 'n_cells': len(cells), 'cells': cells[:spec['cells']]}",
    "        break",
    "    out['matched'] += 1",
    "    grid = exp",
    "print(json.dumps(out))",
])


def run_backtest(code: str, entry: Any, transitions: list, *, timeout: int, cells: int) -> dict:
    """Stage-0 contract, teacher-forced replay, in an isolated subprocess. Never raises.

    Isolation: argv list (never a shell string), python -I -S, a temp working dir and a hard
    timeout. Running model-authored code is inherent to a verifier and is confined exactly the
    way the stock python tool confines the model's own snippets."""
    total = len(transitions)
    if not transitions:
        return {"matched": 0, "total": 0, "green": False, "error": "no recorded transitions for that level yet"}
    spec = {"code": code, "entry": entry, "cells": max(0, cells),
            "transitions": [{"index": t["index"], "action": t["action"], "x": t["x"], "y": t["y"],
                             "grid": t["grid"], "level_up": t["level_up"], "dead": t["dead"], "win": t["win"]}
                            for t in transitions]}
    with tempfile.TemporaryDirectory(prefix="ws_bt_") as d:
        sp, rp = os.path.join(d, "spec.json"), os.path.join(d, "runner.py")
        try:
            with open(sp, "w") as fh:
                json.dump(spec, fh)
            with open(rp, "w") as fh:
                fh.write(_RUNNER_SRC)
            # -I isolates env vars, user site and cwd from sys.path[0]; -S is deliberately NOT used:
            # it would hide site-packages, so a model importing numpy (which the validated Stage-0/1
            # instrument allows, and one Stage-1 green model uses) would score 0 with a confusing error.
            p = subprocess.run([sys.executable, "-I", rp, sp], capture_output=True, text=True,
                               timeout=timeout, cwd=d, shell=False)
        except subprocess.TimeoutExpired:
            return {"matched": 0, "total": total, "green": False,
                    "first_mismatch": {"kind": "timeout", "detail": "model did not finish in %ds" % timeout}}
        except Exception as exc:  # noqa: BLE001
            return {"matched": 0, "total": total, "green": False,
                    "first_mismatch": {"kind": "harness_error", "detail": type(exc).__name__}}
        text = (p.stdout or "").strip()
        line = text.splitlines()[-1] if text else ""
        try:
            res = json.loads(line)
        except Exception:  # noqa: BLE001
            return {"matched": 0, "total": total, "green": False,
                    "first_mismatch": {"kind": "crash", "detail": ((p.stderr or p.stdout) or "")[-400:]}}
        if "error" in res:
            return {"matched": 0, "total": total, "green": False,
                    "first_mismatch": {"kind": "contract", "detail": res["error"]}}
        res["green"] = res.get("matched") == total and total > 0
        return res


def build_preamble(files: dict, log: list, cap: int) -> str:
    """WORKSPACE (saved files) + TRANSITIONS (the log, grids omitted) as literals."""
    compact = [{"index": t["index"], "action": t["action"], "x": t["x"], "y": t["y"], "level": t["level_before"],
                "level_up": t["level_up"], "dead": t["dead"], "changed_cells": t.get("changed_cells")}
               for t in log[-log_max():]]
    # repr(), NOT json.dumps(): this is injected as PYTHON SOURCE, and JSON renders
    # None/True/False as null/true/false, which raise NameError inside the sandbox.
    body = ("WORKSPACE = " + repr(files) + "\n"
            "TRANSITIONS = " + repr(compact) + "\n"
            "# WORKSPACE: your saved files. TRANSITIONS: every action executed this game (grids omitted;\n"
            "#   use history/transitions for boards). Verify a world model with the backtest tool.\n")
    if len(body) > cap:
        small = ("WORKSPACE = " + repr(files) + "\n"
                 "TRANSITIONS = " + repr(compact[-40:]) + "\n")
        body = small if len(small) <= cap else ("WORKSPACE_NAMES = " + repr(sorted(files)) + "\n"
                                                "# workspace too large to inline; use the workspace tool op='load'\n")
    return body


def _fn(name: str, desc: str, props: dict, required: list) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


def workspace_op(st: "WsState", args: dict) -> dict:
    op = str(args.get("op") or "").strip().lower()
    nm = str(args.get("name") or "").strip()
    if op == "list":
        _bump(st.game, "lists")
        return {"tool": "workspace", "files": {k: len(v) for k, v in st.files.items()}}
    if op == "load":
        _bump(st.game, "loads")
        if nm not in st.files:
            return {"tool": "workspace", "error": "no file %r" % nm, "files": sorted(st.files)}
        return {"tool": "workspace", "name": nm, "content": st.files[nm]}
    if op == "delete":
        _bump(st.game, "deletes")
        st.files.pop(nm, None)
        return {"tool": "workspace", "deleted": nm, "files": sorted(st.files)}
    if op == "save":
        content = str(args.get("content") or "")
        if not nm:
            return {"tool": "workspace", "error": "save needs a name"}
        if len(content) > max_file_chars():
            return {"tool": "workspace", "error": "content %d chars exceeds the %d cap" % (len(content), max_file_chars())}
        if nm not in st.files and len(st.files) >= max_files():
            return {"tool": "workspace", "error": "workspace holds %d files; delete one first" % max_files(),
                    "files": sorted(st.files)}
        st.files[nm] = content
        _bump(st.game, "saves")
        _write_marker(st.agent_mod, st.transcript_path,
                      "%s game=%s name=%s chars=%d files=%d" % (SAVE_MARK, st.game, nm, len(content), len(st.files)))
        return {"tool": "workspace", "saved": nm, "chars": len(content), "files": sorted(st.files)}
    return {"tool": "workspace", "error": "unknown op %r; use save|load|list|delete" % op}


def backtest_op(st: "WsState", args: dict) -> dict:
    code = str(args.get("code") or "")
    if not code.strip():
        return {"tool": "backtest", "error": "code is required"}
    try:
        level = int(args.get("level")) if args.get("level") is not None else st.level
    except (TypeError, ValueError):
        level = st.level
    trans = st.transitions_for(level)
    entry = st.entry_by_level.get(level)
    if entry is None and trans:
        entry = trans[0].get("grid")
    t0 = time.monotonic()
    res = run_backtest(code, entry, trans, timeout=backtest_timeout(), cells=mismatch_cells())
    ms = int((time.monotonic() - t0) * 1000)
    _bump(st.game, "backtests")
    _bump(st.game, "backtest_ms_total", ms)
    if res.get("green"):
        _bump(st.game, "backtests_green")
    cur = "%s/%s" % (res.get("matched", 0), res.get("total", 0))
    prev = st.best_by_level.get(level)
    if prev is None or int(res.get("matched", 0)) > int(prev.split("/")[0]):
        st.best_by_level[level] = cur
    _write_marker(st.agent_mod, st.transcript_path,
                  "%s game=%s level=%s matched=%s green=%d code_chars=%d ms=%d"
                  % (BACKTEST_MARK, st.game, level, cur, 1 if res.get("green") else 0, len(code), ms))
    out = {"tool": "backtest", "level": level, "matched": res.get("matched"), "total": res.get("total"),
           "green": bool(res.get("green")), "best_so_far": st.best_by_level.get(level)}
    if res.get("error"):
        out["error"] = res["error"]
    if res.get("first_mismatch"):
        out["first_mismatch"] = res["first_mismatch"]
    if out["green"]:
        out["note"] = ("Model reproduces every recorded transition on this level. Save it with the workspace tool, "
                       "then use it in python to search for a move sequence that clears the level, and execute that "
                       "sequence with action().")
    return out


def install() -> str:
    if _STATE["installed"]:
        return "workspace: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return "workspace: SKIP (import failed: %r)" % (exc,)
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "workspace: SKIP (missing ToolAgent)"
    for n in ("analyze", "_tools", "_dispatch_tool", "_run_python_tool", "_compact_action_result",
              "_render_tool_payload", "_ensure_session"):
        if getattr(cls, n, None) is None:
            return "workspace: SKIP (ToolAgent.%s missing)" % n
    if getattr(agent_mod, "_ToolDispatchResult", None) is None or getattr(agent_mod, "_append_transcript_section", None) is None:
        return "workspace: SKIP (module helpers missing)"

    _STOCK["build_system_prompt"] = getattr(agent_mod, "_build_system_prompt", None)
    _STOCK["build_user_prompt"] = cls._build_user_prompt
    _STOCK["analyze"] = cls.analyze
    _STOCK["tools"] = cls._tools
    _STOCK["dispatch"] = cls._dispatch_tool
    _STOCK["run_python_tool"] = cls._run_python_tool
    _STOCK["compact"] = cls._compact_action_result
    dispatch_cls = agent_mod._ToolDispatchResult

    def _build_system_prompt(*args, **kwargs):
        text = _STOCK["build_system_prompt"](*args, **kwargs)
        if not enabled():
            return text
        try:
            if SYSTEM_ONLY_TOOL in text:
                return text.replace(SYSTEM_ONLY_TOOL, SYSTEM_TOOLS_LINE, 1)
            _skip("system_only_tool_line_absent")
        except Exception:  # noqa: BLE001
            _error()
        return text

    def _build_user_prompt(self, action_num, *args, **kwargs):
        text = _STOCK["build_user_prompt"](self, action_num, *args, **kwargs)
        if not enabled():
            return text
        try:
            notice = ""
            st = getattr(self, "_ws", None)
            if st is not None and st.direct_notice:
                notice = st.direct_notice + "\n"
                st.direct_notice = ""
            if USER_ONLY_TOOL in text:
                return notice + text.replace(USER_ONLY_TOOL, USER_TOOLS_LINE, 1)
            _skip("user_only_tool_line_absent")
            return notice + text
        except Exception:  # noqa: BLE001
            _error()
        return text

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        # the solver passes the real transcript path here; guessing it from state_path is wrong
        # (the runtime stem is "<run>_tool_runtime_state", not "<run>_state").
        if enabled():
            try:
                st = _wstate(self, state_path)
                st.agent_mod = agent_mod
                tp = kwargs.get("transcript_path")
                if tp is not None:
                    st.transcript_path = Path(tp)
                st.game = _game_key(self, state_path, step_env=step_env, transcript_path=st.transcript_path)
            except Exception:  # noqa: BLE001
                _error()
        if enabled() and direct_enabled():
            try:
                st = _wstate(self, state_path)
                lvl = st.level
                if (lvl not in st.direct_done
                        and len(st.direct_done) < direct_max_per_game()
                        and len(st.transitions_for(lvl)) >= direct_after_actions()):
                    run_directed_build(self, agent_mod, st, lvl)
            except Exception:  # noqa: BLE001
                _error()
        return _STOCK["analyze"](self, state_path, action_num, valid_actions=valid_actions,
                                 step_env=step_env, **kwargs)

    def _tools(self, state_path):
        tools = _STOCK["tools"](self, state_path)
        if not enabled():
            return tools
        try:
            return [tools[0] if tools else None,
                    _fn("backtest", BACKTEST_TOOL_DESC,
                        {"code": {"type": "string", "description": "The complete world-model source."},
                         "level": {"type": "integer", "description": "Level to verify against; default = current."}},
                        ["code"]),
                    _fn("workspace", WORKSPACE_TOOL_DESC,
                        {"op": {"type": "string", "enum": ["save", "load", "list", "delete"]},
                         "name": {"type": "string"}, "content": {"type": "string"}},
                        ["op"])] if tools else tools
        except Exception:  # noqa: BLE001
            _error()
            return tools

    def _compact_action_result(self, payload):
        out = _STOCK["compact"](self, payload)
        if not enabled():
            return out
        try:
            st = getattr(self, "_ws", None)
            if st is None or not isinstance(out, dict):
                return out
            sess = getattr(getattr(self, "_step_env_callback", None), "__self__", None)
            entries = getattr(sess, "history_entries", None)
            if not entries:
                return out
            after = _grid_of(entries[-1].frame)
            before = _grid_of(entries[-2].frame) if len(entries) >= 2 else None
            if after is None:
                return out
            lvl = st.level
            act, x, y = action_to_int(out)
            changed = None
            if before is not None and len(before) == len(after):
                changed = sum(1 for r in range(len(after)) for c in range(len(after[r]))
                              if before[r][c] != after[r][c])
            if lvl not in st.entry_by_level and before is not None:
                st.entry_by_level[lvl] = before
            st.log.append({"index": len(st.transitions_for(lvl)), "action": act, "x": x, "y": y, "grid": after,
                           "level_up": bool(out.get("level_completed")), "dead": bool(out.get("game_over")),
                           "win": bool(out.get("run_complete")), "level_before": lvl, "changed_cells": changed})
            _bump(st.game, "transitions_logged")
            if len(st.log) > log_max() * 3:
                del st.log[: len(st.log) - log_max() * 3]
            if out.get("level_completed"):
                st.level = lvl + 1
                st.entry_by_level.setdefault(st.level, after)
        except Exception:  # noqa: BLE001
            _error()
        return out

    def _run_python_tool(self, state_path, arguments):
        if not enabled():
            return _STOCK["run_python_tool"](self, state_path, arguments)
        try:
            st = _wstate(self, state_path)
            st.agent_mod = agent_mod
            st.game = _game_key(self, state_path)
            if st.transcript_path is None:
                st.transcript_path = Path(state_path).parent / ("%s_analyzer.txt" % Path(state_path).stem)
            _bump(st.game, "python_calls")
            if preamble_on() and (st.files or st.log):
                pre = build_preamble(st.files, st.log, preamble_max_chars())
                arguments = dict(arguments or {})
                arguments["code"] = pre + str(arguments.get("code", "") or "")
                _bump(st.game, "preambles")
                _bump(st.game, "preamble_chars_total", len(pre))
        except Exception:  # noqa: BLE001
            _error()
        return _STOCK["run_python_tool"](self, state_path, arguments)

    def _dispatch_tool(self, state_path, name, arguments):
        if not enabled() or name not in ("backtest", "workspace"):
            return _STOCK["dispatch"](self, state_path, name, arguments)
        try:
            self._ensure_session(state_path)
            st = _wstate(self, state_path)
            st.agent_mod = agent_mod
            st.game = _game_key(self, state_path)
            if st.transcript_path is None:
                st.transcript_path = Path(state_path).parent / ("%s_analyzer.txt" % Path(state_path).stem)
            args = arguments if isinstance(arguments, dict) else {}
            payload = workspace_op(st, args) if name == "workspace" else backtest_op(st, args)
            return dispatch_cls(self._render_tool_payload(payload, truncate_fields=("content", "error", "detail")),
                                step_executed=False)
        except Exception as exc:  # noqa: BLE001
            _error()
            return dispatch_cls(json.dumps({"tool": name, "error": type(exc).__name__}, indent=2), step_executed=False)

    if _STOCK["build_system_prompt"] is not None:
        _build_system_prompt._ws_stock = _STOCK["build_system_prompt"]
        agent_mod._build_system_prompt = _build_system_prompt
    _build_user_prompt._ws_stock = _STOCK["build_user_prompt"]
    cls._build_user_prompt = _build_user_prompt
    analyze._ws_stock = _STOCK["analyze"]
    _tools._ws_stock = _STOCK["tools"]
    _dispatch_tool._ws_stock = _STOCK["dispatch"]
    _run_python_tool._ws_stock = _STOCK["run_python_tool"]
    _compact_action_result._ws_stock = _STOCK["compact"]
    cls.analyze = analyze
    cls._tools = _tools
    cls._dispatch_tool = _dispatch_tool
    cls._run_python_tool = _run_python_tool
    cls._compact_action_result = _compact_action_result
    _STATE["installed"] = True
    return "workspace: OK"


# ---------------------------------------------------------------------------
# DIRECTED MODEL BUILDING (WS_DIRECT_*, 2026-09-09)
#
# Two kill tests established that the model never ELECTS to build a world model
# mid-game: 0 verifier calls in 358 python calls, both with the tools merely
# offered and with the prompt corrected to name them. Stage-1 established that the
# same brain builds a green 20/20 model of this very game's wall in 2-3 calls when
# it is handed the transitions and asked. So the missing piece is election, not
# capability or discoverability, and the harness supplies it: when a run has spent
# WS_DIRECT_AFTER_ACTIONS actions on one level without clearing it, the harness
# runs the Stage-1 procedure itself (build -> verify -> iterate once), saves a green
# model to the workspace, and tells the play loop it exists.
#
# This is harness-INITIATED but model-EXECUTED. It differs from the five
# engaged-and-flat behaviour-shaping levers (patch 21, yield900, probe, carry x2),
# which forced the model to do things it was already doing; this asks for something
# it demonstrably does well and never chooses. That is the hypothesis under test.
# ---------------------------------------------------------------------------

DEFAULT_DIRECT_AFTER_ACTIONS = 40
DEFAULT_DIRECT_MAX_CALLS = 3
DEFAULT_DIRECT_MAX_PER_GAME = 2
DEFAULT_DIRECT_MAX_TRANSITIONS = 12
DEFAULT_DIRECT_MAX_CELLS = 60
DIRECT_MARK = "[WS-DIRECT]"
DIRECT_MODEL_FILE = "world_model.py"

DIRECT_SYSTEM = (
    "You are building an executable WORLD MODEL for one level of an ARC-AGI-3 game, from transitions that YOU "
    "recorded while playing this level. Reply with exactly one ```python fence containing the COMPLETE file and "
    "nothing after it. Define either a stateful pair `init_state(entry_grid)` and "
    "`predict(state, grid, action, x, y) -> (next_grid, flags, state)`, or a stateless "
    "`step(grid, action, x, y) -> (next_grid, flags)`. `ENTRY_GRID` is available as a module global. Grids are "
    "lists of lists of ints. `action` is an int: 1-5 = ACTION1..5, 6 = a click at x=col, y=row, 0 = RESET. "
    "`flags` is a dict; set level_up/dead/win when the transition ends the level. The harness replays every "
    "transition and tells you the first one you get wrong. Keep your thinking short: the file must fit in the "
    "output budget. Model the MECHANICS (what moves, what blocks, what a click does), not the specific sequence."
)


def direct_enabled() -> bool:
    return _env("WS_DIRECT_ENABLE", "0").lower() not in _OFF


def direct_after_actions() -> int:
    return _env_int("WS_DIRECT_AFTER_ACTIONS", DEFAULT_DIRECT_AFTER_ACTIONS, 1)


def direct_max_calls() -> int:
    return _env_int("WS_DIRECT_MAX_CALLS", DEFAULT_DIRECT_MAX_CALLS, 1)


def direct_max_per_game() -> int:
    return _env_int("WS_DIRECT_MAX_PER_GAME", DEFAULT_DIRECT_MAX_PER_GAME, 0)


def direct_max_transitions() -> int:
    return _env_int("WS_DIRECT_MAX_TRANSITIONS", DEFAULT_DIRECT_MAX_TRANSITIONS, 4)


def direct_max_cells() -> int:
    """Changed cells listed per transition. The first live directed run left 18-22k of output and the model
    overran it every time; Stage-1's greens had ~25k. Live changed-cell lists are far denser than the curated
    offline slice, so they are what must be capped to get back inside that window."""
    return _env_int("WS_DIRECT_MAX_CELLS", DEFAULT_DIRECT_MAX_CELLS, 4)


_HEX = "0123456789abcdef"


def _rows(grid: Any) -> str:
    """One HEX CHAR per cell, as in the Stage-0/1 encoding every verified green model was built from.
    Space-separated ints (the first live attempt) tokenise ~5-10x worse: it put cd82's prompt at 27,999
    tokens and left 4,769 for output, so the model could not emit a file at all."""
    out = []
    for row in (grid or []):
        out.append("".join(_HEX[v] if isinstance(v, int) and 0 <= v < 16 else "?" for v in row))
    return "\n".join(out)


def _changed(before: Any, after: Any, cap: int | None = None) -> str:
    if before is None or after is None or len(before) != len(after):
        return "(unknown)"
    cap = direct_max_cells() if cap is None else cap
    out = []
    for r in range(len(after)):
        for c in range(len(after[r])):
            if before[r][c] != after[r][c]:
                out.append("r%dc%d:%s->%s" % (r, c, before[r][c], after[r][c]))
                if len(out) >= cap:
                    return " ".join(out) + " ...(+more changed cells; this transition changed a large region)"
    return " ".join(out) if out else "(no cell changed)"


def render_transitions(entry: Any, trans: list) -> str:
    """Entry grid in full, then each transition as its action and its CHANGED CELLS only.
    The Stage-0 lesson: full grids per step blow the window; changed-cell lists are what the
    model can actually reason over (its green cn04/dc22 models were built from this shape)."""
    parts = ["ENTRY GRID (%d rows x %d cols), one HEX DIGIT per cell (0-f = colour 0-15), rows top to bottom:"
             % (len(entry or []), len((entry or [[]])[0])),
             _rows(entry), "",
             "TRANSITIONS (teacher-forced; the grid before transition k is the grid after k-1). Changed cells are "
             "listed as rROWcCOL:old->new with DECIMAL colour values:"]
    prev = entry
    for t in trans:
        act = t.get("action")
        where = "" if t.get("x") is None else " at x=%s y=%s" % (t.get("x"), t.get("y"))
        flags = [k for k in ("level_up", "dead", "win") if t.get(k)]
        parts.append("[%d] action=%s%s%s changed: %s"
                     % (t.get("index"), act, where, (" flags=" + ",".join(flags)) if flags else "",
                        _changed(prev, t.get("grid"))))
        prev = t.get("grid")
    return "\n".join(parts)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m[-1].strip() if m else ""


def run_directed_build(agent: Any, agent_mod: Any, st: "WsState", level: int) -> dict:
    """The Stage-1 procedure, live: render this level's transitions, ask for a model, verify, iterate.
    Returns {calls, green, best, saved}. Never raises; every model call is counted against the clock."""
    trans = st.transitions_for(level)[: direct_max_transitions()]
    entry = st.entry_by_level.get(level) or (trans[0].get("grid") if trans else None)
    out = {"calls": 0, "green": False, "best": "0/%d" % len(trans), "saved": False, "level": level}
    if not trans or entry is None:
        _skip("direct_no_transitions")
        return out
    data = render_transitions(entry, trans)
    feedback = ""
    for attempt in range(direct_max_calls()):
        user = ("Level %d. You have spent %d actions here without clearing it. Build the world model.\n\n%s%s"
                % (level, len(st.transitions_for(level)), data,
                   ("\n\nYour previous model was wrong. " + feedback) if feedback else ""))
        try:
            res = agent._chat_completion([{"role": "system", "content": DIRECT_SYSTEM},
                                          {"role": "user", "content": user}], tools=None)
        except Exception as exc:  # noqa: BLE001
            _write_marker(st.agent_mod, st.transcript_path,
                          "%s game=%s level=%d attempt=%d call_failed=%s" % (DIRECT_MARK, st.game, level, attempt + 1,
                                                                            type(exc).__name__))
            _bump(st.game, "direct_calls")
            out["calls"] += 1
            break
        out["calls"] += 1
        _bump(st.game, "direct_calls")
        code = _extract_code(_text_of_message(agent_mod, res))
        if not code:
            feedback = "You produced no ```python fence. Reply with the complete file in one fence, thinking briefly."
            _write_marker(st.agent_mod, st.transcript_path,
                          "%s game=%s level=%d attempt=%d no_code" % (DIRECT_MARK, st.game, level, attempt + 1))
            continue
        bt = run_backtest(code, entry, trans, timeout=backtest_timeout(), cells=mismatch_cells())
        _bump(st.game, "backtests")
        if bt.get("green"):
            _bump(st.game, "backtests_green")
        cur = "%s/%s" % (bt.get("matched", 0), bt.get("total", 0))
        out["best"] = cur if int(bt.get("matched", 0)) >= int(out["best"].split("/")[0]) else out["best"]
        _write_marker(st.agent_mod, st.transcript_path,
                      "%s game=%s level=%d attempt=%d matched=%s green=%d code_chars=%d"
                      % (DIRECT_MARK, st.game, level, attempt + 1, cur, 1 if bt.get("green") else 0, len(code)))
        if bt.get("green"):
            st.files[DIRECT_MODEL_FILE] = code[: max_file_chars()]
            _bump(st.game, "saves")
            _bump(st.game, "direct_green")
            out["green"] = True
            out["saved"] = True
            st.direct_notice = (
                "The harness built a world model of this level from your own recorded transitions and VERIFIED it: "
                "it reproduces all %d of them exactly. It is saved as WORKSPACE['%s']. Load it in python with "
                "exec(WORKSPACE['%s'], globals()), then search over action sequences with it to find one that "
                "clears the level, and execute that sequence with action(). Trust it over guessing."
                % (len(trans), DIRECT_MODEL_FILE, DIRECT_MODEL_FILE))
            break
        mm = bt.get("first_mismatch") or {}
        feedback = ("The first transition it got wrong is [%s] (action=%s, kind=%s%s). Fix the mechanics that "
                    "explains it." % (mm.get("index"), mm.get("action"), mm.get("kind"),
                                      (", " + str(mm.get("flags"))) if mm.get("flags") else ""))
    st.direct_done.add(level)
    _bump(st.game, "direct_attempts")
    return out


def _text_of_message(agent_mod: Any, res: Any) -> str:
    try:
        msg = getattr(res, "message", None) or {}
        return str(agent_mod._normalize_message_content(msg.get("content", "")) or "")
    except Exception:  # noqa: BLE001
        return ""
