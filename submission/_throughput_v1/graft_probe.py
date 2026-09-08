"""Harness-enforced probe discipline (PROBE, 2026-09-08).

Evidence (docs/research-2026-09-08/R-loss-ledger-3.md): under the 900 s turn
budget turns average 2.05 calls but the analysis-only call share is unchanged
(49%); 84 turns / 75 runs burn 5-7 analysis calls and yield with nothing
executed; ANALYSIS-PARALYSIS is 33% of stuck tails; wall-level exploration
fell to 0.72x the human baseline. Appended prompt blocks are ignored (uptake
< 8%, graft_hypo 0.8%), so the discipline is ENFORCED by the harness: the
model can spend at most PROBE_MAX_ANALYSIS [2] analysis-only python calls per
turn; the next snippet that would again execute no action is not run and gets
a tool result demanding a <= PROBE_MAX_PROBE [5]-action test instead.

WHERE IT HOOKS (verified on the june_stock bundle, tool_agent.py):
  * ToolAgent.analyze(state_path, action_num, ..., transcript_path=,
    analysis_step=) is one TURN. Its loop calls the model, dispatches every
    tool call through _dispatch_tool -> _run_python_tool, and ends the turn
    on the first dispatch whose `step_executed` is True (or on a 60/900 s
    yield / stop). The wrapper resets the per-turn counters when a turn
    starts and records a NOACT turn when it ends with step_executed False.
  * ToolAgent._run_python_tool(state_path, arguments) -> _ToolDispatchResult
    (content=JSON string the model reads as the tool message, step_executed).
    The stock computes step_executed = any(item["executed"]) over the
    sandbox's `action_results` (the per-action() payloads the solver's
    step_env returned) — that is the PAYLOAD read of "did this snippet
    execute a game action": an action() that failed (invalid action, error
    payload, terminal state) has executed=False and counts as analysis-only,
    exactly like a snippet that never called action(). The wrapper counts
    analysis-only calls from this field; it never re-parses the code for the
    count.
  * The REFUSAL must decide BEFORE the snippet runs (the refused snippet is
    not executed), so there is no payload yet: the pre-run predicate is a
    static read of the code — ast.parse, refuse iff the tree contains no
    Call whose func is the Name `action`. A snippet that mentions action()
    (even inside a branch that will not run) is always executed and then
    counted from its payload. A snippet that does not parse is left to the
    stock syntax-error path (more useful to the model than a refusal) and
    counts as analysis-only afterwards.
  * The refusal result goes out the same way a python error does:
    _ToolDispatchResult(self._render_tool_payload({"tool": "python",
    "error": TEXT}, truncate_fields=("stdout", "error", "result")),
    step_executed=False) — analyze() appends it as the {"role": "tool"}
    message, renders it under [TOOL RESULT: python] via
    _render_tool_result_display (an error-only payload displays as the bare
    text), and the persisted history carries it forward.
  * ToolAgent._build_user_prompt: after a NOACT turn the next turn's prompt
    for the same game gets ONE informational line prepended (<= 160 chars).

FLAGS (read at call time; PROBE_ENABLE=0 or the graft not installed = the
stock bytes' behaviour, byte-identical transcript):
  PROBE_ENABLE [1 once installed], PROBE_MAX_ANALYSIS [2], PROBE_MAX_PROBE [5],
  PROBE_MAX_REFUSALS [2] (after this many refusals in one turn the graft
  stops refusing so a turn can never deadlock), PROBE_NOTE_LINES [3] (lines
  of the carried note quoted in the refusal).

TELEMETRY: transcript markers written with the harness's own
_append_transcript_section under a "[HARNESS PROBE]" section —
  "[PROBE-REFUSE] game=<id> turn=<analysis_step> analysis_calls=<n> refusal=<k>/<max>"
  "[PROBE-NOACT] game=<id> turn=<analysis_step> analysis_calls=<n> refusals=<k> reason=<yield|no_capture|…>"
status(): refusals, turns_with_refusal, noact_turns, analysis_calls_total,
acting_calls_total, calls_after_refusal, acting_calls_after_refusal (the key
mechanism read: how often the very next python call after a refusal executed
an action), turns_total, turns_ge3_analysis (turns whose EXECUTED
analysis-only calls reached 3 = enforcement leaked), per_game, errors, skips.

Conventions: module-level _STATE/_STOCK, install() rebinds ToolAgent methods
only, fail-open try/except around every graft branch (an exception returns
the stock result), no threads, no file writes outside the transcript.
install() -> "probe: OK" / "probe: SKIP (...)".
"""
from __future__ import annotations

import ast
import os
import re
import threading
from pathlib import Path
from typing import Any

_STATE: dict[str, Any] = {
    "installed": False,
    "refusals": 0,
    "turns_with_refusal": 0,
    "noact_turns": 0,
    "analysis_calls_total": 0,
    "acting_calls_total": 0,
    "calls_after_refusal": 0,
    "acting_calls_after_refusal": 0,
    "turns_total": 0,
    "turns_ge3_analysis": 0,
    "errors": 0,
    "skips": {},
    "per_game": {},
}
_STOCK: dict[str, Any] = {}
_LOCK = threading.Lock()
_OFF = {"0", "false", "no", "off"}

DEFAULT_MAX_ANALYSIS = 2
DEFAULT_MAX_PROBE = 5
DEFAULT_MAX_REFUSALS = 2
DEFAULT_NOTE_LINES = 3
LEAK_ANALYSIS_CALLS = 3          # a turn whose executed analysis-only calls reach this = enforcement leaked
REFUSAL_CHARS = 900              # hard cap on the refusal text
NOTE_LINE_CHARS = 140
NOACT_LINE_CHARS = 160
TRANSCRIPT_LABEL = "HARNESS PROBE"
REFUSE_MARK = "[PROBE-REFUSE]"
NOACT_MARK = "[PROBE-NOACT]"

REFUSAL_TEXT = (
    "Analysis budget for this turn is spent ({n} analysis-only calls). "
    "Only a snippet that executes a game action is accepted now: run a <={k}-action test of your "
    "leading hypothesis with action([...]) and read the result. "
    "Untested hypotheses in your notes: {notes}"
)
NOACT_TEXT = "Previous turn executed no action after {n} analysis calls."
_NO_NOTES = "(none recorded - state one now and test it)"
_PER_GAME_KEYS = ("refusals", "turns_with_refusal", "noact_turns", "analysis_calls_total", "acting_calls_total",
                  "calls_after_refusal", "acting_calls_after_refusal", "turns_total", "turns_ge3_analysis")


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("PROBE_ENABLE", "1").lower() not in _OFF


def _env_int(name: str, default: int, lo: int = 0) -> int:
    try:
        return max(lo, int(_env(name, str(default))))
    except ValueError:
        return default


def max_analysis() -> int:
    return _env_int("PROBE_MAX_ANALYSIS", DEFAULT_MAX_ANALYSIS, 0)


def max_probe() -> int:
    return _env_int("PROBE_MAX_PROBE", DEFAULT_MAX_PROBE, 1)


def max_refusals() -> int:
    return _env_int("PROBE_MAX_REFUSALS", DEFAULT_MAX_REFUSALS, 0)


def note_lines() -> int:
    return _env_int("PROBE_NOTE_LINES", DEFAULT_NOTE_LINES, 0)


def status() -> dict[str, Any]:
    with _LOCK:
        out = {
            "installed": _STATE["installed"],
            "enabled": enabled(),
            "max_analysis": max_analysis(),
            "max_probe": max_probe(),
            "max_refusals": max_refusals(),
            "note_lines": note_lines(),
        }
        for key in _PER_GAME_KEYS:
            out[key] = _STATE[key]
        out["acting_after_refusal_share"] = (
            _STATE["acting_calls_after_refusal"] / _STATE["calls_after_refusal"]
            if _STATE["calls_after_refusal"] else None)
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


def _bump(game: str, key: str, n: int = 1) -> None:
    with _LOCK:
        _STATE[key] += n
        pg = _STATE["per_game"].setdefault(game, {k: 0 for k in _PER_GAME_KEYS})
        pg[key] = pg.get(key, 0) + n


# ------------------------------------------------------------ per-game state ---
class ProbeState:
    """One per agent (= per game run). Turn fields reset on every analyze()."""

    def __init__(self) -> None:
        self.runtime_dir: Any = None
        self.game: str = "?"
        self.transcript_path: Path | None = None
        self.agent_mod: Any = None
        self.turn: int = 0                 # analysis_step of the turn in flight (or a counter)
        self.in_turn: bool = False
        # per-turn counters
        self.analysis_calls: int = 0       # executed python calls that executed no action
        self.acting_calls: int = 0
        self.refusals: int = 0
        self.after_refusal: bool = False   # the next python call is "the call after a refusal"
        self.leak_counted: bool = False
        # cross-turn
        self.noact_pending: dict[str, Any] | None = None
        self.turns_seen: int = 0

    def reset_turn(self) -> None:
        self.analysis_calls = 0
        self.acting_calls = 0
        self.refusals = 0
        self.after_refusal = False
        self.leak_counted = False


def _pstate(agent: Any, state_path: Any = None) -> ProbeState:
    """Get (or create) the agent's probe state; fresh when the game (runtime dir) changes.
    Mirrors ToolAgent._ensure_session, which keys a session by state_path.parent."""
    st = getattr(agent, "_probe", None)
    if st is None:
        st = ProbeState()
        try:
            agent._probe = st
        except Exception:  # noqa: BLE001
            pass
    if state_path is not None:
        try:
            runtime_dir = Path(state_path).parent
            if st.runtime_dir is not None and st.runtime_dir != runtime_dir:
                fresh = ProbeState()
                fresh.runtime_dir = runtime_dir
                try:
                    agent._probe = fresh
                except Exception:  # noqa: BLE001
                    pass
                return fresh
            st.runtime_dir = runtime_dir
        except Exception:  # noqa: BLE001
            pass
    return st


def _game_key(step_env: Any, transcript_path: Path | None, state_path: Any) -> str:
    """The run stem when the harness gave us a transcript path (<stem>.txt = <gid>_p<draw>),
    else the engine game id, else the runtime dir name."""
    if transcript_path is not None:
        try:
            stem = Path(transcript_path).stem
            if stem:
                return stem
        except Exception:  # noqa: BLE001
            pass
    sess = getattr(step_env, "__self__", None)
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


def _transcript_path(state_path: Any, kwargs: dict[str, Any]) -> Path | None:
    path = kwargs.get("transcript_path")
    if path is not None:
        return Path(path)
    try:
        sp = Path(state_path)
        return sp.parent / f"{sp.stem}_analyzer.txt"  # stock's analyzer_log fallback
    except Exception:  # noqa: BLE001
        return None


def _write_marker(agent_mod: Any, transcript_path: Path | None, line: str) -> None:
    if transcript_path is None or agent_mod is None:
        return
    try:
        agent_mod._append_transcript_section(transcript_path, TRANSCRIPT_LABEL, line)
    except Exception:  # noqa: BLE001
        _error()


# ------------------------------------------------------------- code predicate ---
def code_calls_action(code: str) -> bool | None:
    """True when the snippet's AST contains a call to the sandbox's `action`; False when it
    provably does not; None when the code does not parse (left to the stock syntax-error path)."""
    try:
        tree = ast.parse(str(code or ""), "<python_tool>", "exec")
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "action":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "action":
                return True
    return False


# ------------------------------------------------------------- note extraction ---
_SPLIT_RE = re.compile(r"\s*;\s*|(?<=[.?!])\s+(?=[A-Z0-9(\[\"'])|\s+(?=\(?\d+[.)]\s)|\s+[-*•]\s+")


def note_hypotheses(knowledge: Any, limit: int | None = None) -> list[str]:
    """Up to `limit` fragments from the carried note: 'Open questions' first (the stock maps
    the model's `Hypothesis:` label onto world_model, and `Next test:` onto current_plan, so
    those two fields come next). Whitespace is already collapsed by the stock extractor, so
    fragments are split on ';', sentence ends, list markers and numbering."""
    if limit is None:
        limit = note_lines()
    if limit <= 0 or not isinstance(knowledge, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for key in ("open_questions", "current_plan", "world_model"):
        text = " ".join(str(knowledge.get(key) or "").split())
        if not text:
            continue
        for frag in _SPLIT_RE.split(text):
            frag = (frag or "").strip(" -*•\t")
            frag = re.sub(r"^\(?\d+[.)]\s*", "", frag).strip()
            if len(frag) < 8:
                continue
            if len(frag) > NOTE_LINE_CHARS:
                frag = frag[: NOTE_LINE_CHARS - 1].rstrip() + "…"
            low = frag.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(frag)
            if len(out) >= limit:
                return out
    return out


def render_refusal(n_analysis: int, knowledge: Any) -> str:
    notes = note_hypotheses(knowledge)
    rendered = ("\n" + "\n".join(f"- {line}" for line in notes)) if notes else _NO_NOTES
    text = REFUSAL_TEXT.format(n=n_analysis, k=max_probe(), notes=rendered)
    if len(text) > REFUSAL_CHARS:
        text = text[: REFUSAL_CHARS - 1].rstrip() + "…"
    return text


def render_noact_line(info: dict[str, Any]) -> str:
    line = NOACT_TEXT.format(n=int(info.get("analysis_calls") or 0))
    refused = int(info.get("refusals") or 0)
    if refused:
        line = line[:-1] + f" ({refused} refused)."
    if len(line) > NOACT_LINE_CHARS:
        line = line[: NOACT_LINE_CHARS - 1].rstrip() + "…"
    return line


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "probe: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"probe: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "probe: SKIP (missing ToolAgent)"
    for name in ("analyze", "_run_python_tool", "_build_user_prompt", "_render_tool_payload"):
        if getattr(cls, name, None) is None:
            return f"probe: SKIP (ToolAgent.{name} missing)"
    for name in ("_append_transcript_section", "_ToolDispatchResult"):
        if getattr(agent_mod, name, None) is None:
            return f"probe: SKIP ({name} missing)"

    _STOCK["analyze"] = cls.analyze
    _STOCK["run_python_tool"] = cls._run_python_tool
    _STOCK["build_user_prompt"] = cls._build_user_prompt
    dispatch_cls = agent_mod._ToolDispatchResult

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        st = None
        if enabled():
            try:
                st = _pstate(self, state_path)
                st.agent_mod = agent_mod
                st.transcript_path = _transcript_path(state_path, kwargs)
                st.game = _game_key(step_env, kwargs.get("transcript_path"), state_path)
                step = kwargs.get("analysis_step")
                st.turns_seen += 1
                st.turn = int(step) if step is not None else st.turns_seen
                st.reset_turn()
                st.in_turn = True
            except Exception:  # noqa: BLE001
                _error()
                st = None
        result = _STOCK["analyze"](self, state_path, action_num, valid_actions=valid_actions,
                                   step_env=step_env, **kwargs)
        if st is not None:
            try:
                st.in_turn = False
                _bump(st.game, "turns_total")
                stopped = False
                should_stop = kwargs.get("should_stop")
                if should_stop is not None and getattr(result, "yielded_control", False):
                    try:
                        stopped = bool(should_stop())      # the run is ending (stop_requested): not a NOACT turn
                    except Exception:  # noqa: BLE001
                        stopped = False
                if stopped:
                    _skip("stop_requested")
                elif result is not None and not getattr(result, "retryable_failure", False) \
                        and not getattr(result, "step_executed", False):
                    reason = "yield" if getattr(result, "yielded_control", False) else "no_capture"
                    info = {"analysis_calls": st.analysis_calls, "refusals": st.refusals, "turn": st.turn,
                            "game": st.game, "reason": reason}
                    st.noact_pending = info
                    _bump(st.game, "noact_turns")
                    _write_marker(agent_mod, st.transcript_path,
                                  f"{NOACT_MARK} game={st.game} turn={st.turn} analysis_calls={st.analysis_calls} "
                                  f"refusals={st.refusals} reason={reason}")
            except Exception:  # noqa: BLE001
                _error()
        return result

    def _run_python_tool(self, state_path, arguments):
        st = getattr(self, "_probe", None) if enabled() else None
        if st is not None and not st.in_turn:
            st = None                      # a call outside analyze(): pure pass-through
        # 1. refusal decision BEFORE the snippet runs
        if st is not None:
            try:
                if st.analysis_calls >= max_analysis() and st.refusals < max_refusals():
                    code = str((arguments or {}).get("code", "") or "")
                    verdict = code_calls_action(code)
                    if verdict is False:
                        if st.after_refusal:
                            # a refusal right after a refusal: the model's "next call" did not act
                            _bump(st.game, "calls_after_refusal")
                        st.refusals += 1
                        st.after_refusal = True
                        _bump(st.game, "refusals")
                        if st.refusals == 1:
                            _bump(st.game, "turns_with_refusal")
                        _write_marker(agent_mod, st.transcript_path,
                                      f"{REFUSE_MARK} game={st.game} turn={st.turn} analysis_calls={st.analysis_calls} "
                                      f"refusal={st.refusals}/{max_refusals()}")
                        text = render_refusal(st.analysis_calls, getattr(self, "_summarized_knowledge", None))
                        payload = {"tool": "python", "error": text}
                        content = self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result"))
                        return dispatch_cls(content, step_executed=False)
                    if verdict is None:
                        _skip("unparsable_code")
                elif st.analysis_calls >= max_analysis() and st.refusals >= max_refusals():
                    _skip("refusal_cap")
            except Exception:  # noqa: BLE001
                _error()
        # 2. the stock call (the snippet runs)
        result = _STOCK["run_python_tool"](self, state_path, arguments)
        # 3. count from the payload the harness computed (step_executed = any action_results[].executed)
        if st is not None:
            try:
                acted = bool(getattr(result, "step_executed", False))
                if st.after_refusal:
                    st.after_refusal = False
                    _bump(st.game, "calls_after_refusal")
                    if acted:
                        _bump(st.game, "acting_calls_after_refusal")
                if acted:
                    st.acting_calls += 1
                    _bump(st.game, "acting_calls_total")
                else:
                    st.analysis_calls += 1
                    _bump(st.game, "analysis_calls_total")
                    if st.analysis_calls >= LEAK_ANALYSIS_CALLS and not st.leak_counted:
                        st.leak_counted = True
                        _bump(st.game, "turns_ge3_analysis")
            except Exception:  # noqa: BLE001
                _error()
        return result

    def _build_user_prompt(self, action_num, *args, **kwargs):
        text = _STOCK["build_user_prompt"](self, action_num, *args, **kwargs)
        if not enabled():
            return text
        try:
            st = getattr(self, "_probe", None)
            if st is None or not st.noact_pending:
                return text
            info = st.noact_pending
            st.noact_pending = None
            return render_noact_line(info) + "\n" + text
        except Exception:  # noqa: BLE001
            _error()
            return text

    analyze._probe_stock = _STOCK["analyze"]
    _run_python_tool._probe_stock = _STOCK["run_python_tool"]
    _build_user_prompt._probe_stock = _STOCK["build_user_prompt"]
    cls.analyze = analyze
    cls._run_python_tool = _run_python_tool
    cls._build_user_prompt = _build_user_prompt
    _STATE["installed"] = True
    return "probe: OK"
