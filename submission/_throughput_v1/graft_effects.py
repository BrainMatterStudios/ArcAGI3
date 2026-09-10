"""graft_effects.py — the harness computes the dynamics the agent never computes.

THE MEASUREMENT THIS COMES FROM (docs/research-2026-09-10/R-what-our-agent-actually-does.md):
across 2,047 tool calls made on levels our agent went on to CLEAR, it forward-simulated a
next state **zero times**, ran any search in 2.2 % of calls, and touched `transitions` --
the harness's own before/after affordance -- in 3.2 %. It is a reactive perceive-and-act
loop. Six experiments say we cannot talk it into being anything else.

So this graft does not ask. It computes an object-level effect table in pure python from
frames the harness already holds, and hands the agent the result. Zero extra model calls.
No behaviour change is required for the information to arrive.

MECHANISM
  * Objects come from the stock `segment_layer`, whose `hash` is translation-invariant, so
    the same object is matchable across frames by hash alone.
  * For each consecutive (before, action, after) in history: objects present in both with
    the same hash but a different top-left are MOVED (with the offset); hashes only in
    after are APPEARED; only in before are VANISHED.
  * Occurrences accumulate per action name. The block reports the MODAL effect and how
    consistent it is, e.g. `LEFT x7: moves 1 obj by (0,-2) [6/7]`.

DELIBERATE LIMITS
  * Read-only. It never changes what the agent may do, only what it is told.
  * Segmentation is cached by frame ascii, so a game with N actions segments each frame
    once, not once per turn (the naive version is O(N^2) and would cost seconds by endgame).
  * Capped output: EFFECTS_MAX_ACTIONS lines, EFFECTS_MAX_CHARS total, so it cannot crowd
    the context the way an uncapped world-model block can.
  * If anything fails it disables itself for the run and the stock prompt is unchanged --
    a graft must never be able to cost a game.
"""
from __future__ import annotations

import os
import threading
from collections import Counter, defaultdict
from typing import Any

BLOCK_MARK = "[EFFECTS]"
SKIP_MARK = "[EFFECTS-SKIP]"

_STATE: dict[str, Any] = {"installed": False, "disabled": False, "skips": Counter()}
_LOCK = threading.Lock()
# per-run accumulators, keyed by the transcript path the analyzer is writing
_RUNS: dict[str, dict[str, Any]] = {}


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _cfg() -> dict[str, int]:
    return {
        "max_actions": _env_int("EFFECTS_MAX_ACTIONS", 8, 1),
        "max_chars": _env_int("EFFECTS_MAX_CHARS", 900, 100),
        "min_occurrences": _env_int("EFFECTS_MIN_OCCURRENCES", 1, 1),
    }


def status() -> dict[str, Any]:
    return {"installed": _STATE["installed"], "disabled": _STATE["disabled"],
            "skips": dict(_STATE["skips"]), "runs": len(_RUNS), "cfg": _cfg()}


def _skip(reason: str) -> None:
    with _LOCK:
        _STATE["skips"][reason] += 1


# ---------------------------------------------------------------- segmentation

def _nodes(grid, cache: dict) -> list[dict] | None:
    """Segment one frame, memoised on its grid so each frame is segmented once.

    NOTE the stock contract: `segment_layer(grid, color_chars)` takes the INTEGER grid
    and the ARC colour-symbol string, NOT ascii rows. Frame.ascii is a derived property;
    the grid is the real input, and it is a tuple of tuples so it is directly hashable.
    """
    if not grid:
        return None
    hit = cache.get(grid)
    if hit is not None:
        return hit
    try:
        from inference.utils.segmentation import segment_layer
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        out = segment_layer(grid, ARC_COLOR_CHARS).get("nodes") or []
    except Exception:  # noqa: BLE001 — a segmentation failure must not cost the game
        _skip("segment_failed")
        return None
    # keep only what the diff needs, so the cache stays small on long games
    slim = [{"hash": n.get("hash"), "color": n.get("color"), "pixels": n.get("pixels"),
             "at": tuple(n.get("boundary", [[None, None]])[0] or (None, None))} for n in out]
    cache[grid] = slim
    return slim


def _diff(before: list[dict], after: list[dict]) -> dict[str, Any] | None:
    """Object-level delta between two segmented frames, matched by translation-invariant hash."""
    if before is None or after is None:
        return None
    b_by, a_by = defaultdict(list), defaultdict(list)
    for n in before:
        b_by[n["hash"]].append(n)
    for n in after:
        a_by[n["hash"]].append(n)
    moved = []
    for h, bs in b_by.items():
        a_list = a_by.get(h)
        if not a_list or len(bs) != 1 or len(a_list) != 1:
            continue
        (br, bc), (ar, ac) = bs[0]["at"], a_list[0]["at"]
        if None in (br, bc, ar, ac):
            continue
        if (ar, ac) != (br, bc):
            moved.append((h, ar - br, ac - bc))
    appeared = [h for h in a_by if h not in b_by]
    vanished = [h for h in b_by if h not in a_by]
    return {"moved": moved, "appeared": len(appeared), "vanished": len(vanished),
            "changed": bool(moved or appeared or vanished)}


# ------------------------------------------------------------------ accumulate

def _run(key: str) -> dict[str, Any]:
    with _LOCK:
        return _RUNS.setdefault(key, {"seg_cache": {}, "effects": defaultdict(list), "seen": 0})


def _observe(run: dict[str, Any], history: list) -> None:
    """Fold every not-yet-seen (before, action, after) triple into the effect table.

    `history` is a list of runtime_state.HistoryEntry dataclasses: `.action` (str) and
    `.frame` (Frame, whose `.ascii` is a property derived from `.grid`)."""
    cache = run["seg_cache"]
    start = max(1, run["seen"])
    for i in range(start, len(history)):
        prev, cur = history[i - 1], history[i]
        act = str(getattr(cur, "action", "") or "").strip()
        if not act:
            continue
        pf, cf = getattr(prev, "frame", None), getattr(cur, "frame", None)
        if pf is None or cf is None:
            continue
        d = _diff(_nodes(getattr(pf, "grid", None), cache),
                  _nodes(getattr(cf, "grid", None), cache))
        if d is None:
            continue
        run["effects"][act].append(d)
    run["seen"] = len(history)


def _describe(occurrences: list[dict]) -> str:
    n = len(occurrences)
    unchanged = sum(1 for o in occurrences if not o["changed"])
    if unchanged == n:
        return f"no board change [{n}/{n}]"
    # the modal movement signature across occurrences
    sigs = Counter()
    for o in occurrences:
        if o["moved"]:
            sigs[tuple(sorted((dr, dc) for _h, dr, dc in o["moved"]))] += 1
    parts = []
    if sigs:
        sig, cnt = sigs.most_common(1)[0]
        offs = ", ".join(f"({dr:+d},{dc:+d})" for dr, dc in sig[:3])
        parts.append(f"moves {len(sig)} obj by {offs} [{cnt}/{n}]")
    # 18 of 25 games tick a HUD bar every action (the standing "HUD breaks frame
    # identity" law), so appear/vanish fires on nearly every action and carries no
    # information when it is near-universal. Report it only when it DISCRIMINATES
    # between occurrences of this action; the modal movement above is the real signal.
    app = sum(1 for o in occurrences if o["appeared"])
    van = sum(1 for o in occurrences if o["vanished"])
    universal = _env_int("EFFECTS_UNIVERSAL_PCT", 90, 1) / 100.0
    if app and app / n < universal:
        parts.append(f"objects appear [{app}/{n}]")
    if van and van / n < universal:
        parts.append(f"objects vanish [{van}/{n}]")
    if unchanged:
        parts.append(f"no change [{unchanged}/{n}]")
    return "; ".join(parts) if parts else f"changes board [{n - unchanged}/{n}]"


def _block(run: dict[str, Any]) -> str:
    cfg = _cfg()
    rows = sorted(run["effects"].items(), key=lambda kv: -len(kv[1]))
    lines = []
    for act, occ in rows[: cfg["max_actions"]]:
        if len(occ) < cfg["min_occurrences"]:
            continue
        lines.append(f"- {act} x{len(occ)}: {_describe(occ)}")
    if not lines:
        return ""
    body = "\n".join(lines)
    if len(body) > cfg["max_chars"]:
        body = body[: cfg["max_chars"]].rsplit("\n", 1)[0]
    return ("Observed action effects so far this game (computed from the real frames, "
            "object-level, translation-invariant):\n" + body + "\n")


# --------------------------------------------------------------------- install

def install() -> str:
    if _STATE["installed"]:
        return "effects: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"effects: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "effects: SKIP (missing ToolAgent)"
    for name in ("analyze", "_build_user_prompt"):
        if getattr(cls, name, None) is None:
            return f"effects: SKIP (ToolAgent.{name} missing)"
    if getattr(agent_mod, "load_runtime_state", None) is None:
        return "effects: SKIP (load_runtime_state missing)"

    stock_analyze = cls.analyze
    stock_prompt = cls._build_user_prompt
    load_runtime_state = agent_mod.load_runtime_state

    def analyze(self, *args, **kwargs):
        # The solver calls analyze(state_path, action_count, ...) with state_path
        # POSITIONAL and transcript_path as a keyword. Read both defensively: a graft
        # that guesses a signature wrong writes its telemetry to the wrong place and
        # reports zero while working (the 09-08 transcript-path bug).
        state_path = args[0] if args else kwargs.get("state_path")
        key = str(kwargs.get("transcript_path") or (args[4] if len(args) > 4 else "") or "")
        self._effects_key = key
        if not _STATE["disabled"] and key and state_path is not None:
            try:
                _, history = load_runtime_state(state_path)
                if history:
                    _observe(_run(key), history)
            except Exception:  # noqa: BLE001 — never let the graft cost a turn
                _skip("observe_failed")
        return stock_analyze(self, *args, **kwargs)

    def _build_user_prompt(self, *args, **kwargs):
        text = stock_prompt(self, *args, **kwargs)
        if _STATE["disabled"]:
            return text
        key = getattr(self, "_effects_key", "")
        if not key or key not in _RUNS:
            return text
        try:
            block = _block(_RUNS[key])
        except Exception:  # noqa: BLE001
            _skip("block_failed")
            return text
        if not block:
            return text
        return text + "\n" + block

    cls.analyze = analyze
    cls._build_user_prompt = _build_user_prompt
    _STATE["installed"] = True
    # the runner enforces a "<name>: OK" suffix on install()
    return "effects: analyze+_build_user_prompt: OK"
