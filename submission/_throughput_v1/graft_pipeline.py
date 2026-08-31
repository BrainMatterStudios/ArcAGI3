"""Turn-pipeline repair graft (TP9, 2026-08-31) — no discarded thinking,
livelock breaker, one client-timeout retry.

Evidence (docs/research-2026-08-31/R9-path-to-7-gap-analysis.md §2.1): 48% of
all LLM turns are idle (`step_executed: False`), overwhelmingly "Yielded
control to solver: turn_time_budget"; 13 vLLM read-timeout turns stranded
games at run end; r11l sat in a 6,000 s deterministic livelock (20+
consecutive idle turns with byte-identical transcript lengths) after clearing
L1 in 5 actions.

Seams, verified in the anim bundle (the code that plays at eval —
submission/_inspect_replay/assets_build/ARC3-Inference):

- THE YIELD. tool_agent.py:154 reads LOCAL_ANALYZER_YIELD_SECONDS (60 in the
  measured run) into :1085 `self._yield_seconds`; `control_yield_reason`
  (:2154-2163) returns "turn_time_budget" once the turn is past it. The check
  fires between requests (:2168 loop top, :2288 after a no-tool-call reply,
  :2353 between tool calls) and breaks out of `analyze`.

- WHERE THE THINKING DIES. On a `requests.RequestException` (read-timeout
  included) `analyze` sets preserve_history=False (:2365), reverts
  `_history_messages` to the pre-turn snapshot (:2403), and returns
  `AnalyzerTurnResult(step_executed=False, retryable_failure=True,
  reasoning=captured_reasoning)` (:2380) — the ONLY surviving copy of the
  turn's reasoning is `result.reasoning`. The solver loop (solver.py:352-363)
  handles retryable_failure / yielded_control / idle by `continue` and NEVER
  reads `result.reasoning`: that field is dropped on the floor every idle
  turn. (On a clean turn_time_budget yield after a no-tool-call reply the
  reasoning-only assistant message does land in `_history_messages`, but the
  next `analyze` rebuilds context from a trimmed history and — at fixed
  seed/temperature — regenerates the same doomed turn; r11l's identical
  transcript lengths are exactly that. So even the "kept" case needs the
  explicit resume-and-act nudge, and the kept copy can be trimmed away.)

- THE HTTP CALL. `_chat_completion` (:1518-1546) does one `requests.post`
  with timeout=`request_timeout_seconds` (solver.py:267-284: min of analyzer
  timeout, wall remaining, soft remaining). A ReadTimeout surfaces as
  `requests.RequestException` -> the discard path above; the solver retries
  the analysis step after a 1 s backoff (solver.py:64,352-357) but the
  generation is already lost, and at run end `should_stop()` breaks first
  (solver.py:354-355) — the 13 stranded turns.

Three behaviours, each individually gated (TP9_ENABLE=0 turns all off):

(a) TURN PERSIST/RESUME (TP9_RESUME, default 1). Wrap `ToolAgent.analyze`:
    when the stock turn comes back idle (yielded_control or
    retryable_failure) with non-empty `result.reasoning`, stash the tail
    (TP9_RESUME_CHARS, default 1500) on the agent. Wrap
    `ToolAgent._build_user_prompt`: the next turn's user prompt gets a
    RESUME block carrying that tail plus an instruction to continue from it
    and act — consumed once, cleared on any executed step, reset on game
    change (mirrors _ensure_session's runtime-dir keying, :1140-1152).
    J9/F6: a consumed tail is parked, not dropped — if the very request it
    was injected into dies with NO new reasoning (timeout before any reply),
    the parked tail is restored and re-injected next turn; any new reasoning
    or an executed step discards the parked copy.

(b) LIVELOCK DETECTOR (TP9_LIVELOCK, default 1; TP9_LIVELOCK_K, default 3;
    TP9_LIVELOCK_COOLDOWN, default 5). Per game, hash each idle turn's
    `result.reasoning` (empty output hashes equal — that IS the r11l
    signature). K consecutive identical hashes with zero executed actions
    arms a LOOP BREAKER block for the next user prompt ("state ONE new
    hypothesis and emit one action([...]) batch now") and bumps a counter.
    Streak resets on an executed step or a differing hash.
    J9/F3a: `retryable_failure` turns are EXCLUDED from the streak — they
    carry no model-produced output (a 3 s server outage at the solver's 1 s
    retry cadence must never arm a false "identical output" accusation);
    they neither build nor reset the streak.
    J9/F3b: the breaker fires AT MOST ONCE per armed streak — injection
    resets the streak and starts a cooldown of TP9_LIVELOCK_COOLDOWN counted
    turns during which no streak accumulates, bounding the injection rate to
    1 per (K + cooldown) turns (~13 per 100-turn persistent livelock at
    defaults, vs every prompt before the fix).
    Scope (J9/F5): hash identity only catches the empty/deterministic-replay
    subspecies; a livelock that still emits sampled non-identical text at
    temp 0.6 is out of scope for this detector.

(c) CLIENT-TIMEOUT RETRY (TP9_RETRY, default 1; TP9_RETRY_BACKOFF, default
    1.0 s; TP9_RETRY_MIN_BUDGET, default 5.0 s). Wrap `_chat_completion`:
    retry `requests.Timeout` / `requests.ConnectionError` up to TP9_RETRY
    times before letting the exception reach the discard path. Other
    RequestExceptions (HTTP errors, context-length rejections handled at
    :2221) pass through untouched — a context-overflow retry of the
    identical payload deterministically fails and the analyze loop has its
    own recovery for it.
    J9/F4 wall-clock guard: the turn's `request_timeout_seconds` is the
    solver's min(analyzer timeout, wall remaining, soft remaining)
    (solver.py:267-284), so it IS the wall bound. The retry never exceeds
    it: the retry attempt gets only the leftover budget
    (budget - elapsed - backoff), and when that leftover is under
    TP9_RETRY_MIN_BUDGET the retry is skipped and the exception propagates
    (stock behaviour, letting the solver's own should_stop-guarded retry
    path take over). Net effect: a fast ConnectionError still gets its
    near-free retry; a ReadTimeout that consumed the whole budget is never
    doubled. With no budget argument and no analyzer timeout the retry is
    unbounded, as before.

Conventions: module-level _STOCK dict (never only a closure), fail-open
try/except around all graft logic, rebinding class attributes only, new file
only. install() -> "pipeline: OK" / "pipeline: SKIP (...)".
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any

_STATE: dict[str, Any] = {
    "installed": False,
    "resumes_injected": 0,
    "perturbations_injected": 0,
    "retries_used": 0,
}
_STOCK: dict[str, Any] = {}
_OFF = {"0", "false", "no", "off"}

DEFAULT_LIVELOCK_K = 3
DEFAULT_LIVELOCK_COOLDOWN = 5
DEFAULT_RETRY = 1
DEFAULT_RESUME_CHARS = 1500
DEFAULT_RETRY_BACKOFF = 1.0
DEFAULT_RETRY_MIN_BUDGET = 5.0

RESUME_BLOCK = (
    "RESUME: You were interrupted mid-thought last turn ({reason}) and nothing was applied. "
    "Your partial reasoning from that turn (tail):\n<<<\n{tail}\n>>>\n"
    "Do not re-derive this from scratch. Continue from those conclusions and emit a `python` "
    "tool call that reaches `action(actions)` promptly this turn."
)
PERTURB_BLOCK = (
    "LOOP BREAKER: You have produced identical output {n} times in a row with no actions "
    "executed. Break the loop: state ONE new hypothesis in a single sentence, then emit one "
    "`python` tool call that calls `action([...])` with a small batch NOW — act first, "
    "analyze the result after."
)


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("TP9_ENABLE", "1").lower() not in _OFF


def resume_enabled() -> bool:
    return enabled() and _env("TP9_RESUME", "1").lower() not in _OFF


def livelock_enabled() -> bool:
    return enabled() and _env("TP9_LIVELOCK", "1").lower() not in _OFF


def livelock_k() -> int:
    try:
        return max(2, int(_env("TP9_LIVELOCK_K", str(DEFAULT_LIVELOCK_K))))
    except ValueError:
        return DEFAULT_LIVELOCK_K


def livelock_cooldown() -> int:
    try:
        return max(0, int(_env("TP9_LIVELOCK_COOLDOWN", str(DEFAULT_LIVELOCK_COOLDOWN))))
    except ValueError:
        return DEFAULT_LIVELOCK_COOLDOWN


def retry_attempts() -> int:
    if not enabled():
        return 0
    try:
        return max(0, int(_env("TP9_RETRY", str(DEFAULT_RETRY))))
    except ValueError:
        return DEFAULT_RETRY


def retry_backoff() -> float:
    try:
        return max(0.0, float(_env("TP9_RETRY_BACKOFF", str(DEFAULT_RETRY_BACKOFF))))
    except ValueError:
        return DEFAULT_RETRY_BACKOFF


def retry_min_budget() -> float:
    try:
        return max(0.0, float(_env("TP9_RETRY_MIN_BUDGET", str(DEFAULT_RETRY_MIN_BUDGET))))
    except ValueError:
        return DEFAULT_RETRY_MIN_BUDGET


def resume_chars() -> int:
    try:
        return max(200, int(_env("TP9_RESUME_CHARS", str(DEFAULT_RESUME_CHARS))))
    except ValueError:
        return DEFAULT_RESUME_CHARS


def status() -> dict[str, Any]:
    return {
        "installed": _STATE["installed"],
        "enabled": enabled(),
        "resume": resume_enabled(),
        "livelock": livelock_enabled(),
        "livelock_k": livelock_k(),
        "livelock_cooldown": livelock_cooldown(),
        "retry": retry_attempts(),
        "retry_min_budget": retry_min_budget(),
        "resume_chars": resume_chars(),
        "resumes_injected": _STATE["resumes_injected"],
        "perturbations_injected": _STATE["perturbations_injected"],
        "retries_used": _STATE["retries_used"],
    }


# ----------------------------------------------------------- per-game state ---
class PipelineState:
    def __init__(self) -> None:
        self.runtime_dir: Any = None
        self.resume_tail: str | None = None
        self.resume_reason: str = ""
        # F6: the last-injected tail, parked so it can be restored if the
        # request it rode on dies with no new reasoning.
        self.injected_tail: str | None = None
        self.injected_reason: str = ""
        self.last_hash: str | None = None
        self.idle_streak = 0
        self.perturb_pending = False
        # F3b: counted turns left before the livelock streak may build again.
        self.cooldown = 0


def _pstate(agent: Any, state_path: Any = None) -> PipelineState:
    """Get (or create) the agent's TP9 state; reset it when the game changes.

    Mirrors ToolAgent._ensure_session, which keys a session by
    state_path.parent (the per-game runtime dir).
    """
    st = getattr(agent, "_tp9", None)
    if st is None:
        st = PipelineState()
        try:
            agent._tp9 = st
        except Exception:  # noqa: BLE001
            pass
    if state_path is not None:
        try:
            runtime_dir = state_path.parent
            if st.runtime_dir is not None and st.runtime_dir != runtime_dir:
                fresh = PipelineState()
                fresh.runtime_dir = runtime_dir
                try:
                    agent._tp9 = fresh
                except Exception:  # noqa: BLE001
                    pass
                return fresh
            st.runtime_dir = runtime_dir
        except Exception:  # noqa: BLE001
            pass
    return st


def _idle_reason(result: Any) -> str:
    if getattr(result, "retryable_failure", False):
        return "the model-server request failed"
    if getattr(result, "yielded_control", False):
        return "the turn time budget expired"
    return "no action call was captured"


def _note_turn_result(st: PipelineState, result: Any) -> None:
    """Update resume/livelock state from one stock analyze() result."""
    if getattr(result, "step_executed", False):
        st.resume_tail = None
        st.resume_reason = ""
        st.injected_tail = None
        st.injected_reason = ""
        st.last_hash = None
        st.idle_streak = 0
        st.perturb_pending = False
        st.cooldown = 0
        return
    reasoning = str(getattr(result, "reasoning", "") or "")
    if resume_enabled():
        if reasoning.strip():
            st.resume_tail = reasoning[-resume_chars():]
            st.resume_reason = _idle_reason(result)
            st.injected_tail = None
            st.injected_reason = ""
        elif st.injected_tail and getattr(result, "retryable_failure", False):
            # F6 (re-judged): restore the parked tail ONLY when the request it
            # rode on actually died (retryable_failure). A clean empty yield
            # means the model saw the nudge and ignored it — re-serving there
            # stacks duplicate RESUME blocks in a livelock (J9 regression iii).
            st.resume_tail = st.injected_tail
            st.resume_reason = st.injected_reason
    if livelock_enabled():
        if getattr(result, "retryable_failure", False):
            # F3a: a failed request produced no model output — a server
            # outage must neither build nor reset the identical-output streak.
            return
        if st.cooldown > 0:
            # F3b: cooling down after an injection; no streak accumulation.
            st.cooldown -= 1
            st.last_hash = None
            st.idle_streak = 0
            return
        digest = hashlib.sha1(reasoning.encode("utf-8", errors="replace")).hexdigest()
        if digest == st.last_hash:
            st.idle_streak += 1
        else:
            st.idle_streak = 1
            st.last_hash = digest
        if st.idle_streak >= livelock_k():
            st.perturb_pending = True


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "pipeline: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"pipeline: SKIP (import failed: {exc!r})"
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        return f"pipeline: SKIP (requests missing: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "pipeline: SKIP (missing ToolAgent)"
    for name in ("analyze", "_build_user_prompt", "_chat_completion"):
        if getattr(cls, name, None) is None:
            return f"pipeline: SKIP (ToolAgent.{name} missing)"

    _STOCK["analyze"] = cls.analyze
    _STOCK["build_user_prompt"] = cls._build_user_prompt
    _STOCK["chat_completion"] = cls._chat_completion

    # -- (a)+(b) capture: wrap analyze -------------------------------------
    def analyze(self, state_path, action_num, *args, **kwargs):
        if enabled():
            try:
                # Reset per-game state BEFORE the turn so a stale resume tail
                # from the previous game never leaks into this game's prompt.
                _pstate(self, state_path)
            except Exception:  # noqa: BLE001
                pass
        result = _STOCK["analyze"](self, state_path, action_num, *args, **kwargs)
        if not enabled() or result is None:
            return result
        try:
            _note_turn_result(_pstate(self, state_path), result)
        except Exception:  # noqa: BLE001
            pass
        return result

    # -- (a)+(b) injection: wrap _build_user_prompt ------------------------
    def build_user_prompt(self, action_num, **kwargs):
        text = _STOCK["build_user_prompt"](self, action_num, **kwargs)
        if not enabled():
            return text
        try:
            st = getattr(self, "_tp9", None)
            if st is None:
                return text
            blocks: list[str] = []
            if resume_enabled() and st.resume_tail:
                blocks.append(RESUME_BLOCK.format(reason=st.resume_reason, tail=st.resume_tail))
                # consumed once — but parked (F6) so a dead injected request
                # can restore it; discarded on new reasoning or executed step.
                st.injected_tail = st.resume_tail
                st.injected_reason = st.resume_reason
                st.resume_tail = None
                st.resume_reason = ""
                _STATE["resumes_injected"] += 1
            if livelock_enabled() and st.perturb_pending:
                blocks.append(PERTURB_BLOCK.format(n=st.idle_streak))
                st.perturb_pending = False
                # F3b: at most one injection per armed streak — reset the
                # streak and start the re-arm cooldown.
                st.idle_streak = 0
                st.last_hash = None
                st.cooldown = livelock_cooldown()
                _STATE["perturbations_injected"] += 1
            if blocks:
                return text + "\n" + "\n".join(blocks)
        except Exception:  # noqa: BLE001
            pass
        return text

    # -- (c) retry: wrap _chat_completion ----------------------------------
    def chat_completion(self, messages, **kwargs):
        attempts = 0
        budget = None
        started = None
        try:
            attempts = retry_attempts()
            # F4: the turn's request_timeout_seconds is the solver's wall
            # bound (min of analyzer timeout / wall remaining / soft
            # remaining) — the retry must stay inside it.
            budget = kwargs.get("request_timeout_seconds")
            if budget is None:
                budget = getattr(self, "_timeout", None)
            budget = None if budget is None else float(budget)
            started = time.monotonic()
        except Exception:  # noqa: BLE001
            attempts = 0
        if attempts <= 0:
            return _STOCK["chat_completion"](self, messages, **kwargs)
        for attempt in range(attempts + 1):
            try:
                return _STOCK["chat_completion"](self, messages, **kwargs)
            except (requests.Timeout, requests.ConnectionError):
                if attempt >= attempts:
                    raise
                try:
                    backoff = retry_backoff()
                    if budget is not None:
                        leftover = budget - (time.monotonic() - started) - backoff
                        if leftover < retry_min_budget():
                            # F4: no meaningful wall left — propagate (stock
                            # behaviour; the solver's should_stop-guarded
                            # retry path takes over).
                            raise
                        kwargs = dict(kwargs)
                        kwargs["request_timeout_seconds"] = leftover
                    _STATE["retries_used"] += 1
                    if backoff > 0:
                        time.sleep(backoff)
                except (requests.Timeout, requests.ConnectionError):
                    raise
                except Exception:  # noqa: BLE001
                    pass
        raise RuntimeError("unreachable: TP9 retry loop exhausted without raising")

    analyze._tp9_stock = _STOCK["analyze"]
    build_user_prompt._tp9_stock = _STOCK["build_user_prompt"]
    chat_completion._tp9_stock = _STOCK["chat_completion"]
    cls.analyze = analyze
    cls._build_user_prompt = build_user_prompt
    cls._chat_completion = chat_completion
    _STATE["installed"] = True
    return "pipeline: OK"
