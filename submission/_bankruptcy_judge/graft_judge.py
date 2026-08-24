"""Bankruptcy-judge graft — the two approved multi-role mechanisms from
docs/RESEARCH-2026-08-23-searchcore-and-multirole.md §B:

1. CONTRADICTION LEDGER (flag JUDGE_LEDGER, default on): zero-LLM-call
   bookkeeping. Consumes expect-queue stop events (``expect_mismatch`` /
   ``no_op`` from submission/_expect_queue/graft_expect.py — loose
   integration: with the expect graft absent the ledger still tracks strict
   no-ops from a per-action evidence log this graft records itself) plus
   blocks-since-progress, and renders one ``- Contradiction ledger: ...``
   line of STATED FACTS into the carried world-model block of every user
   prompt. Resets on level transition.

2. BANKRUPTCY JUDGE (flag JUDGE_CALL, default on): a fresh-context grounded
   rebuild call fired by the Stage-1-TUNED trigger
   (falsifier/stage1_results.json selected params, 988/1200 configs passed,
   recall 9/9 labeled terminal spirals, 5.6% fires on completed levels):

       A_click=45  A_avatar=40  A_mixed=45   (actions since level start)
       T_stall=22                            (blocks since level start — the
                                              OR-branch for action-light,
                                              turn-heavy spirals)
       cooldown 20 actions AND 8 blocks; caps 2/level, 3/session; a carried
       world model and >=1 evidence tuple must exist. Archetype from the
       frame-0 menu (the proven _archetype_triage rule), frozen per session.

   The judge call is framed EXACTLY as the falsifier's judge_instruction.txt
   (embedded verbatim below): the stock system prompt is kept unchanged (KV
   prefix stability — the serve reuses the system-prompt prefix cache), and
   the judge framing + evidence digest ride the single user message. Fresh
   context by construction: messages = [system, user], never the session
   history. temperature 1.0, reasoning_effort medium (setdefault into
   chat_template_kwargs, same transport as the effort graft), tools=None,
   max ~1200 output tokens. §B law: judge calls must be grounded in observed
   facts + fresh context + concrete comparison framing.

   Ruling application: REJECT with cited contradictions (a ``#N`` transition
   citation) and parsed hypotheses -> overwrite ``world_model`` with a
   ``REBUILT ...`` summary of the three structurally-different hypotheses,
   keep ``REJECTED: <old model>``, and set ``open_questions`` to the
   discriminating tests. An uncited REJECT is ignored (fail-open, keep
   stock); KEEP changes nothing. A fired-but-failed call (network/parse)
   changes nothing but still consumes the cooldown so a broken endpoint is
   never hammered.

Evidence digest (EXHIBIT B): the last 20 (action, coords[x,y], changed_px,
level, level_delta) tuples from a live per-action log this graft records at
the ``_execute_action`` seam, plus the strict no-op events of the current
level segment — the same shape corpus_lib.digest_as_text produced for the
Stage-2/3 falsifier prompts, so the Stage-3 serve validation transfers.

Seams (verified against the June stock tree,
scratchpad/bundles/june_stock/src/ARC3-Inference — same pin the
_expect_queue / _yield_carryover grafts were built against):

- tool_agent.py:1282-1334 — ``ToolAgent._chat_completion``: the request
  shape the judge call replicates (build_chat_payload + requests.post to
  ``{base_url}/chat/completions`` with ``self._headers()``); the graft
  builds its own payload (temp 1.0 / no tools / judge max_tokens) instead
  of wrapping it, so stock analyzer requests are untouched.
- tool_agent.py:957 / :984 — ``_summarized_knowledge`` init/reset;
  :1105-1111 assistant-note updates; :1113-1126 the LEVEL-RESET seam (the
  carried model is wiped on level_transition/run_complete/game_over), which
  is why a judge rebuild written into ``_summarized_knowledge`` inherits
  exactly the stock lifecycle.
- tool_agent.py:1128-1145 — ``_summarized_knowledge_lines``: single call
  site :1236 inside ``_build_user_prompt``; the ledger wrapper inserts its
  line before the trailing "Revise any item..." line.
- tool_agent.py:1161-1256 — ``_build_user_prompt``: single call site :1727
  in ``analyze`` => exactly one call per transcript block (the unit the
  Stage-1 trigger was tuned on, corpus_lib block == one analyze turn), and
  it runs BEFORE the turn's LLM call — the trigger fires here, ahead of the
  call, and the rebuilt model is what the model then reads.
- tool_agent.py:1024-1068 / :1495-1545 / :1583 — the trigger's signal
  sources: ``_summarize_step_sequence`` stop_reason plumbing, the sandbox
  ``action()`` handler, and the per-turn step-summary update. The graft
  reads these SIGNALS via ``_compact_action_result`` (:1416-1451, called at
  :1532 for every executed step_env payload) — expect-queue stop events are
  observed there without depending on the expect graft being installed.
- tool_agent.py:1721 — ``analyze`` binds ``self._step_env_callback`` to the
  session's bound ``step_env`` BEFORE ``_build_user_prompt`` (:1727), so
  ``callback.__self__`` reaches the live ``_HarnessGameSession`` (evidence
  log + frame-0 menu) at both graft entry points.
- solver.py:667-734 — ``_HarnessGameSession._execute_action``: the evidence
  recorder wraps it (class attribute; called only via ``self._execute_action``
  at :616/:665), computing changed_px from ``_grid_from_state`` (:93-98)
  pre/post and level_delta from ``levels_completed``; payload ``action_num``
  (= len(history) after the step, :208-210) numbers the tuples exactly like
  the falsifier corpus (#i zero-based).
- Frame-0 menu: ``session.game.current_state.available_actions`` — the
  archetype rule replicated verbatim from submission/_archetype_triage/
  graft_triage.py (RESET stripped; CLICK safe default; frozen once readable).

Fail-open invariants:
- Stock calls inside wrappers run unguarded — stock behavior (including
  exceptions) propagates exactly as before.
- Every graft-added step (evidence capture, trigger math, judge call,
  ruling application, ledger render) sits inside try/except; any error
  means "stock prompt / no fire / no ledger line".
- JUDGE_LEDGER=0 + JUDGE_CALL=0 at install time => no patches at all; at
  call time => installed wrappers are pure pass-throughs (byte-identical
  prompts, zero extra requests, no attributes consulted).
- The judge NEVER raises into the turn: a failed call logs a warning and
  the stock prompt proceeds unchanged.
"""

from __future__ import annotations

import os
import re
from typing import Any

# --- tuned trigger (falsifier Stage 1, pre-registered bars all PASS) ---------

A_CLICK = 45
A_AVATAR = 40
A_MIXED = 45
T_STALL = 22
COOLDOWN_ACTIONS = 20
COOLDOWN_BLOCKS = 8
CAP_LEVEL = 2
CAP_SESSION = 3

DIGEST_TUPLES = 20
DIGEST_NOOPS_SHOWN = 10
EVIDENCE_CAP = 512
JUDGE_TEMPERATURE = 1.0
JUDGE_EFFORT = "medium"
DEFAULT_JUDGE_MAX_TOKENS = 1200
REJECTED_OLD_MODEL_CAP = 300

LEDGER_MARKER = "Contradiction ledger:"
REBUILT_MARKER = "REBUILT"

ARCHETYPE_CLICK = "CLICK"
ARCHETYPE_AVATAR = "AVATAR"
ARCHETYPE_MIXED = "MIXED"
_ARCHETYPE_WINDOW = {
    ARCHETYPE_CLICK: A_CLICK,
    ARCHETYPE_AVATAR: A_AVATAR,
    ARCHETYPE_MIXED: A_MIXED,
}
_MOVEMENT_IDS = {1, 2, 3, 4}
_AVATAR_FAMILY = {1, 2, 3, 4, 5}
_CLICK_ID = 6

_EXPECT_STOP_REASONS = ("expect_mismatch", "no_op")

# Verbatim falsifier/judge_instruction.txt (frozen §B design law). The Stage-3
# serve falsifier validated the judge on exactly this framing — do not edit one
# without the other. test_bankruptcy_judge asserts byte-parity with the file.
JUDGE_INSTRUCTION = """\
You are a fresh-context BANKRUPTCY JUDGE for an agent that is solving a multi-level grid-puzzle game.

You receive exactly two exhibits and nothing else:
  EXHIBIT A — the agent's carried working world model (its own claims; possibly wrong).
  EXHIBIT B — an evidence digest reconstructed from the environment's ground-truth log:
              the last observed transitions as tuples (action, coords[x,y], changed_px,
              level, level_delta) plus the recorded no-op events on the current level
              (actions that changed zero pixels).

Law of this court: only EXHIBIT B is ground truth. EXHIBIT A is a hypothesis under audit.
Do not extend trust to any claim in EXHIBIT A that EXHIBIT B does not support. You have a
fresh context on purpose: you must not try to rescue the carried model out of sympathy or
momentum. Judge it against the observed facts by concrete comparison.

Your ruling, in exactly this format:

CONTRADICTIONS:
- <each concrete conflict between a claim in EXHIBIT A and the observations in EXHIBIT B,
  citing transition numbers like #37; also cite predicted-but-absent effects — e.g. the
  model predicts progress or board change from an action class whose observed transitions
  are no-ops. If there are none, write "- none">
VERDICT: KEEP or REJECT
  REJECT if a core mechanic or goal claim of EXHIBIT A is contradicted by cited evidence,
  or if the model's plan has had ample evidence-visible opportunity (many transitions on
  this level, level_delta always 0) and its predicted progress never appears.
  KEEP if EXHIBIT A is consistent with the citations and the evidence shows normal
  mid-execution progress toward its stated goal.
HYPOTHESES: (only when the verdict is REJECT; otherwise omit this section)
H1: <replacement mechanic hypothesis> | test: <one discriminating next action>
H2: <replacement mechanic hypothesis> | test: <one discriminating next action>
H3: <replacement mechanic hypothesis> | test: <one discriminating next action>

The three hypotheses must be STRUCTURALLY DIFFERENT — three distinct mechanism families
(e.g. selection-then-placement, toggle-neighborhood, order/sequence dependence, gating by
hidden state, movement/physics), not parameter variants of one another and not a reworded
version of the rejected model. Each hypothesis must be grounded in at least one cited
observation from EXHIBIT B, and each test must be a single concrete action (with
coordinates when it is a click) whose outcome would separate that hypothesis from the
other two."""


def _ledger_enabled() -> bool:
    return os.environ.get("JUDGE_LEDGER", "1").strip() not in {"0", "false", "False"}


def _call_enabled() -> bool:
    return os.environ.get("JUDGE_CALL", "1").strip() not in {"0", "false", "False"}


def _any_enabled() -> bool:
    return _ledger_enabled() or _call_enabled()


def _judge_max_tokens() -> int:
    try:
        value = int(os.environ.get("JUDGE_MAX_TOKENS", "").strip() or DEFAULT_JUDGE_MAX_TOKENS)
    except ValueError:
        return DEFAULT_JUDGE_MAX_TOKENS
    return max(256, value)


# --- frame-0 archetype (replicated verbatim from graft_triage.py) ------------


def classify_menu(available_actions: Any) -> str | None:
    """Frame-0 archetype from an available_actions menu, or None if unreadable."""
    try:
        ids = {int(item) for item in available_actions}
    except (TypeError, ValueError):
        return ARCHETYPE_CLICK
    ids.discard(0)
    if not ids:
        return None  # not yet readable — retry next block
    has_movement = bool(ids & _MOVEMENT_IDS)
    has_click = _CLICK_ID in ids
    if has_movement and has_click:
        return ARCHETYPE_MIXED
    if has_movement and ids <= _AVATAR_FAMILY:
        return ARCHETYPE_AVATAR
    return ARCHETYPE_CLICK


# --- ruling parser (shared with the Stage-3 serve probe) ---------------------

VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*\**\s*(KEEP|REJECT)", re.I | re.M)
HYP_RE = re.compile(r"^\s*\**H([123])\**\s*:\s*(.+)$", re.M)
CONTRA_RE = re.compile(r"CONTRADICTIONS\s*:\s*(.*?)(?=^\s*\**\s*VERDICT\s*:)", re.S | re.M | re.I)
CITE_RE = re.compile(r"#\d+")


def parse_judge_ruling(text: str) -> dict[str, Any]:
    """Parse one judge completion into verdict / citations / hypotheses.

    ``cited`` is True only when the CONTRADICTIONS section (or, if that header
    is unparseable, the pre-VERDICT text) carries at least one ``#N``
    transition citation — the §B grounding requirement.
    """
    text = text or ""
    verdict_match = VERDICT_RE.search(text)
    verdict = verdict_match.group(1).upper() if verdict_match else None
    contra_match = CONTRA_RE.search(text)
    if contra_match:
        contradictions = contra_match.group(1).strip()
    elif verdict_match:
        contradictions = text[: verdict_match.start()].strip()
    else:
        contradictions = ""
    hypotheses: list[dict[str, str]] = []
    for number, raw in HYP_RE.findall(text):
        mechanic, _, test = raw.partition("| test:")
        if not test:
            mechanic, _, test = raw.partition("|test:")
        hypotheses.append(
            {"n": number, "mechanic": mechanic.strip(" |"), "test": test.strip(), "raw": raw.strip()}
        )
    return {
        "verdict": verdict,
        "contradictions": contradictions,
        "cited": bool(CITE_RE.search(contradictions)),
        "hypotheses": hypotheses,
    }


def build_judge_user_message(world_model_text: str, digest_text: str) -> str:
    """Exactly the falsifier Stage-2 framing (stage2_build.build_prompt)."""
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"EXHIBIT A — carried working world model:\n"
        f"---\n{world_model_text}\n---\n\n"
        f"EXHIBIT B — evidence digest:\n"
        f"---\n{digest_text}\n---\n\n"
        f"Deliver your ruling now, in the exact format specified."
    )


# --- live evidence digest (corpus_lib.digest_as_text shape) ------------------


def _current_level_segment(evidence: list[dict[str, Any]], level: Any) -> list[dict[str, Any]]:
    """Contiguous suffix of the evidence log on the given level."""
    segment: list[dict[str, Any]] = []
    for entry in reversed(evidence):
        if entry.get("level") != level:
            break
        segment.append(entry)
    segment.reverse()
    return segment


def _coords_text(coords: Any) -> str:
    if isinstance(coords, (list, tuple)) and len(coords) == 2:
        return f"({coords[0]},{coords[1]})"
    return "-"


def live_digest_text(evidence: list[dict[str, Any]], level: Any) -> str:
    segment = _current_level_segment(evidence, level)
    noops = [entry for entry in segment if not entry.get("changed_px")]
    lines = [
        f"Evidence digest (ground truth from the environment log; current level {level}, "
        f"{len(segment)} actions taken on this level so far).",
        "Last observed transitions, oldest first — (action, coords[x,y], changed_px, level, level_delta):",
    ]
    for entry in evidence[-DIGEST_TUPLES:]:
        lines.append(
            f"  #{entry['i']}: {entry['action']} {_coords_text(entry.get('coords'))} "
            f"changed_px={entry['changed_px']} level={entry['level']} level_delta={entry['level_delta']}"
        )
    lines.append(
        f"No-op events on this level (actions whose board did not change at all): {len(noops)}"
    )
    for entry in noops[-DIGEST_NOOPS_SHOWN:]:
        lines.append(f"  #{entry['i']}: {entry['action']} {_coords_text(entry.get('coords'))}")
    return "\n".join(lines)


# --- per-agent judge state ---------------------------------------------------


def _fresh_state(runtime_dir: Any) -> dict[str, Any]:
    return {
        "runtime_dir": runtime_dir,
        "archetype": None,
        "block_idx": -1,
        "cum_actions": 0,
        "level": None,
        "level_start_action": 0,   # trigger window zero (reset on level AND fire)
        "level_start_block": 0,
        "level_first_action": 0,   # ledger zero (reset on level change only)
        "level_first_block": 0,
        "last_fire_action": None,
        "last_fire_block": None,
        "fires_this_level": 0,
        "fires_session": 0,
        "expect_events": {},       # level -> {"expect_mismatch": n, "no_op": n}
        "rulings": [],             # observability: applied/ignored fire outcomes
    }


def _judge_state(agent: Any) -> dict[str, Any]:
    runtime_dir = getattr(agent, "_session_runtime_dir", None)
    state = getattr(agent, "_judge_state_v1", None)
    if not isinstance(state, dict) or state.get("runtime_dir") != runtime_dir:
        state = _fresh_state(runtime_dir)
        agent._judge_state_v1 = state
    return state


def _session_of(agent: Any) -> Any:
    return getattr(getattr(agent, "_step_env_callback", None), "__self__", None)


def _session_evidence(agent: Any) -> list[dict[str, Any]]:
    session = _session_of(agent)
    evidence = getattr(session, "_judge_evidence", None)
    return evidence if isinstance(evidence, list) else []


def _carried_model_exists(agent: Any) -> bool:
    knowledge = getattr(agent, "_summarized_knowledge", None)
    return isinstance(knowledge, dict) and any(bool(value) for value in knowledge.values())


# --- graft body --------------------------------------------------------------


def install() -> str:
    if not _any_enabled():
        return "bankruptcy_judge: SKIP (JUDGE_LEDGER=0 and JUDGE_CALL=0)"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"bankruptcy_judge: SKIP (solver module missing: {exc!r})"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"bankruptcy_judge: SKIP (tool_agent module missing: {exc!r})"

    # Presence gates — every seam symbol, fail toward stock on any mismatch.
    session_cls = getattr(solver_mod, "_HarnessGameSession", None)
    if session_cls is None:
        return "bankruptcy_judge: SKIP (missing _HarnessGameSession)"
    original_execute_action = getattr(session_cls, "_execute_action", None)
    if original_execute_action is None:
        return "bankruptcy_judge: SKIP (missing _HarnessGameSession._execute_action)"
    if getattr(solver_mod, "_grid_from_state", None) is None:
        return "bankruptcy_judge: SKIP (missing solver._grid_from_state)"
    agent_cls = getattr(agent_mod, "ToolAgent", None)
    if agent_cls is None:
        return "bankruptcy_judge: SKIP (missing ToolAgent)"
    original_build_user_prompt = getattr(agent_cls, "_build_user_prompt", None)
    original_knowledge_lines = getattr(agent_cls, "_summarized_knowledge_lines", None)
    original_compact = getattr(agent_cls, "_compact_action_result", None)
    for name, value in (
        ("_build_user_prompt", original_build_user_prompt),
        ("_summarized_knowledge_lines", original_knowledge_lines),
        ("_compact_action_result", original_compact),
        ("_headers", getattr(agent_cls, "_headers", None)),
        ("_accumulate_usage_tokens", getattr(agent_cls, "_accumulate_usage_tokens", None)),
    ):
        if value is None:
            return f"bankruptcy_judge: SKIP (missing ToolAgent.{name})"
    for name in (
        "build_chat_payload",
        "requests",
        "_LOCAL_ANALYZER_TOP_P",
        "_LOCAL_ANALYZER_TOP_K",
        "_LOCAL_ANALYZER_ENABLE_THINKING",
        "_LOCAL_ANALYZER_SEED",
        "_normalize_message_content",
        "_extract_reasoning_text",
        "log",
    ):
        if getattr(agent_mod, name, None) is None:
            return f"bankruptcy_judge: SKIP (missing tool_agent.{name})"
    if getattr(original_build_user_prompt, "_bankruptcy_judge_patched", False):
        return "bankruptcy_judge: SKIP (already applied)"

    log = agent_mod.log

    # --- 1. per-action evidence recorder (solver seam) ----------------------

    def execute_action_with_evidence(
        self: Any,
        action: Any,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        pre_grid = None
        pre_completed = 0
        if _any_enabled():
            try:
                pre_grid = solver_mod._grid_from_state(self.game.current_state)
                pre_completed = int(self.game.current_state.levels_completed)
            except Exception:  # noqa: BLE001 — capture failure => no tuple
                pre_grid = None
        payload = original_execute_action(
            self,
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )
        if pre_grid is not None:
            try:
                post_grid = solver_mod._grid_from_state(self.game.current_state)
                changed_px = sum(
                    1
                    for pre_row, post_row in zip(pre_grid, post_grid)
                    for pre_cell, post_cell in zip(pre_row, post_row)
                    if pre_cell != post_cell
                )
                post_completed = int(self.game.current_state.levels_completed)
                data = dict(getattr(action, "data", None) or {})
                coords = (
                    [data["x"], data["y"]] if "x" in data and "y" in data else None
                )
                evidence = getattr(self, "_judge_evidence", None)
                if not isinstance(evidence, list):
                    evidence = []
                    self._judge_evidence = evidence
                evidence.append(
                    {
                        "i": int(payload.get("action_num") or 0) - 1,
                        "action": str(
                            getattr(getattr(action, "id", None), "name", "") or "?"
                        ),
                        "coords": coords,
                        "changed_px": changed_px,
                        "level": payload.get("level"),
                        "level_delta": post_completed - pre_completed,
                    }
                )
                if len(evidence) > EVIDENCE_CAP:
                    del evidence[: len(evidence) - EVIDENCE_CAP]
            except Exception:  # noqa: BLE001 — evidence must never break a step
                pass
        return payload

    # --- 2. expect-queue stop-event observer (loose integration) ------------

    def compact_with_events(self: Any, payload: dict[str, Any]) -> dict[str, Any]:
        compact = original_compact(self, payload)
        if _any_enabled():
            try:
                reason = compact.get("stop_reason")
                if reason in _EXPECT_STOP_REASONS:
                    state = _judge_state(self)
                    bucket = state["expect_events"].setdefault(
                        compact.get("level"), {"expect_mismatch": 0, "no_op": 0}
                    )
                    bucket[reason] += 1
            except Exception:  # noqa: BLE001
                pass
        return compact

    # --- trigger + judge ----------------------------------------------------

    def _observe_block(agent: Any, action_num: int, current_frame: Any, summary: Any) -> dict[str, Any]:
        state = _judge_state(agent)
        state["block_idx"] += 1
        state["cum_actions"] = max(0, int(action_num or 0))
        current_level = getattr(current_frame, "level", None)
        current_level = int(current_level) if current_level is not None else 1
        if isinstance(summary, dict):
            try:
                summary_level = int(summary.get("level"))
            except (TypeError, ValueError):
                summary_level = None
            if summary_level is not None:
                current_level = max(current_level, summary_level)
        if state["level"] != current_level:
            is_transition = state["level"] is not None
            state["level"] = current_level
            state["level_start_action"] = state["cum_actions"]
            state["level_start_block"] = state["block_idx"]
            state["level_first_action"] = state["cum_actions"]
            state["level_first_block"] = state["block_idx"]
            state["fires_this_level"] = 0
            if is_transition:
                # Ledger resets on a REAL level transition; events recorded
                # before the session's first block belong to the current level.
                state["expect_events"] = {}
        if state["archetype"] is None:
            session = _session_of(agent)
            if session is not None:
                state["archetype"] = classify_menu(
                    session.game.current_state.available_actions
                )
        return state

    def _judge_request(agent: Any, world_model_text: str, digest_text: str) -> str:
        messages = [
            {"role": "system", "content": agent._system_prompt},
            {
                "role": "user",
                "content": build_judge_user_message(world_model_text, digest_text),
            },
        ]
        payload = agent_mod.build_chat_payload(
            provider=agent._model.provider,
            model=agent._model.model_id,
            messages=messages,
            max_tokens=_judge_max_tokens(),
            temperature=JUDGE_TEMPERATURE,
            top_p=agent_mod._LOCAL_ANALYZER_TOP_P,
            top_k=agent_mod._LOCAL_ANALYZER_TOP_K,
            thinking=bool(agent_mod._LOCAL_ANALYZER_ENABLE_THINKING),
            tools=None,
            tool_choice=None,
            seed=agent_mod._LOCAL_ANALYZER_SEED,
        )
        template_kwargs = payload.get("chat_template_kwargs")
        if isinstance(template_kwargs, dict):
            # Same transport as the effort graft; setdefault yields to any
            # future harness-set effort.
            template_kwargs.setdefault("reasoning_effort", JUDGE_EFFORT)
        response = agent_mod.requests.post(
            f"{agent._model.base_url.rstrip('/')}/chat/completions",
            headers=agent._headers(),
            json=payload,
            timeout=agent._timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("judge call returned no choices")
        message = choices[0].get("message", {})
        try:
            agent._accumulate_usage_tokens(body.get("usage"))
        except Exception:  # noqa: BLE001 — accounting only
            pass
        content = agent_mod._normalize_message_content(message.get("content", ""))
        if not (content or "").strip():
            # Robustness: some serves leave the final text in reasoning.
            content = agent_mod._extract_reasoning_text(message)
        return content or ""

    def _apply_rebuild(agent: Any, ruling: dict[str, Any]) -> None:
        knowledge = agent._summarized_knowledge
        old_model = str(
            knowledge.get("world_model")
            or knowledge.get("goal_model")
            or "(no explicit model recorded)"
        )
        if len(old_model) > REJECTED_OLD_MODEL_CAP:
            old_model = old_model[:REJECTED_OLD_MODEL_CAP].rstrip() + "..."
        hypotheses = ruling["hypotheses"][:3]
        mechanics = "; ".join(f"H{h['n']}: {h['mechanic']}" for h in hypotheses)
        tests = "; ".join(f"H{h['n']}: {h['test']}" for h in hypotheses if h["test"])
        knowledge["world_model"] = (
            f"{REBUILT_MARKER} after a fresh-context evidence audit rejected the carried model "
            f"on cited contradictions. Candidate mechanics (structurally different — discriminate "
            f"before committing): {mechanics}. REJECTED: {old_model}"
        )
        if tests:
            knowledge["open_questions"] = (
                f"Which rebuilt hypothesis holds? Run the discriminating tests first — {tests}"
            )

    def _maybe_fire_judge(agent: Any, state: dict[str, Any]) -> None:
        asl = state["cum_actions"] - state["level_start_action"]
        bsl = state["block_idx"] - state["level_start_block"]
        window = _ARCHETYPE_WINDOW.get(state["archetype"] or ARCHETYPE_CLICK, A_CLICK)
        act_hit = asl >= window
        stall_hit = bsl >= T_STALL
        if not (act_hit or stall_hit):
            return
        if (
            state["last_fire_action"] is not None
            and state["cum_actions"] - state["last_fire_action"] < COOLDOWN_ACTIONS
        ):
            return
        if (
            state["last_fire_block"] is not None
            and state["block_idx"] - state["last_fire_block"] < COOLDOWN_BLOCKS
        ):
            return
        if state["fires_this_level"] >= CAP_LEVEL or state["fires_session"] >= CAP_SESSION:
            return
        if not _carried_model_exists(agent):
            return  # nothing carried to judge
        evidence = _session_evidence(agent)
        if not evidence:
            return  # no ground truth to ground the ruling in
        knowledge_lines = original_knowledge_lines(agent)
        if len(knowledge_lines) < 2:
            return

        # FIRE — the fire consumes cooldown/caps regardless of the outcome so a
        # broken endpoint is never hammered, and the fresh window starts now.
        state["last_fire_action"] = state["cum_actions"]
        state["last_fire_block"] = state["block_idx"]
        state["fires_this_level"] += 1
        state["fires_session"] += 1
        state["level_start_action"] = state["cum_actions"]
        state["level_start_block"] = state["block_idx"]

        world_model_text = "\n".join(knowledge_lines[1:])
        digest_text = live_digest_text(evidence, state["level"])
        outcome = {
            "block": state["block_idx"],
            "level": state["level"],
            "cum_actions": state["cum_actions"],
            "reason": "actions" if act_hit else "stall",
            "applied": False,
            "verdict": None,
        }
        try:
            ruling_text = _judge_request(agent, world_model_text, digest_text)
            ruling = parse_judge_ruling(ruling_text)
            outcome["verdict"] = ruling["verdict"]
            if ruling["verdict"] == "REJECT" and ruling["cited"] and ruling["hypotheses"]:
                _apply_rebuild(agent, ruling)
                outcome["applied"] = True
            log.warning(
                "bankruptcy_judge: fired (%s, level %s, %s actions) -> verdict=%s applied=%s",
                outcome["reason"],
                state["level"],
                state["cum_actions"],
                ruling["verdict"],
                outcome["applied"],
            )
        except Exception as exc:  # noqa: BLE001 — a failed judge changes nothing
            outcome["error"] = str(exc)
            log.warning("bankruptcy_judge: fired but call failed (fail-open): %s", exc)
        state["rulings"].append(outcome)

    def build_user_prompt_with_judge(
        self: Any,
        action_num: int,
        *,
        valid_actions: list[str] | None,
        current_frame: Any = None,
        history_entries: Any = None,
        previous_step_summary: Any = None,
    ) -> str:
        if _any_enabled():
            try:
                state = _observe_block(self, action_num, current_frame, previous_step_summary)
                if _call_enabled():
                    _maybe_fire_judge(self, state)
            except Exception:  # noqa: BLE001 — trigger errors => stock prompt
                pass
        return original_build_user_prompt(
            self,
            action_num,
            valid_actions=valid_actions,
            current_frame=current_frame,
            history_entries=history_entries,
            previous_step_summary=previous_step_summary,
        )

    # --- 3. contradiction ledger (zero LLM calls) ---------------------------

    def _ledger_line(agent: Any) -> str | None:
        state = getattr(agent, "_judge_state_v1", None)
        if not isinstance(state, dict):
            return None
        level = state.get("level")
        segment = _current_level_segment(_session_evidence(agent), level)
        noops = [entry for entry in segment if not entry.get("changed_px")]
        events = state.get("expect_events", {}).get(level, {})
        mismatch_halts = int(events.get("expect_mismatch", 0))
        noop_halts = int(events.get("no_op", 0))
        if not (noops or mismatch_halts or noop_halts):
            return None
        facts: list[str] = []
        if noops:
            last = noops[-1]
            coords = _coords_text(last.get("coords"))
            where = f" {coords}" if coords != "-" else ""
            facts.append(
                f"{len(noops)} of the last {len(segment)} actions on this level changed "
                f"zero pixels (latest no-op: {last['action']}{where} at #{last['i']})"
            )
        if mismatch_halts:
            facts.append(f"{mismatch_halts} action batch(es) halted early on an expect mismatch")
        if noop_halts:
            facts.append(f"{noop_halts} batch(es) halted early on a strict no-op step")
        blocks_no_progress = state["block_idx"] - state.get("level_first_block", 0)
        actions_no_progress = state["cum_actions"] - state.get("level_first_action", 0)
        facts.append(
            f"{blocks_no_progress} model turns and {actions_no_progress} actions on this "
            f"level without a level-up"
        )
        return f"- {LEDGER_MARKER} " + "; ".join(facts) + "."

    def knowledge_lines_with_ledger(self: Any) -> list[str]:
        lines = original_knowledge_lines(self)
        if not _ledger_enabled():
            return lines
        try:
            # Render only into an existing carried block (design: the ledger is
            # stated facts INSIDE the carried world model lines).
            if len(lines) >= 2:
                fact_line = _ledger_line(self)
                if fact_line:
                    lines = list(lines)
                    lines.insert(len(lines) - 1, fact_line)
        except Exception:  # noqa: BLE001 — ledger errors => stock lines
            pass
        return lines

    build_user_prompt_with_judge._bankruptcy_judge_patched = True  # type: ignore[attr-defined]
    # Single-namespace by design: _execute_action is only called via
    # ``self._execute_action`` (solver.py:616/:665), _build_user_prompt via
    # ``self._build_user_prompt`` (tool_agent.py:1727), _summarized_knowledge_lines
    # via ``self.`` (:1236), _compact_action_result via ``self.`` (:1532).
    session_cls._execute_action = execute_action_with_evidence
    agent_cls._compact_action_result = compact_with_events
    agent_cls._build_user_prompt = build_user_prompt_with_judge
    agent_cls._summarized_knowledge_lines = knowledge_lines_with_ledger
    install.originals = {  # type: ignore[attr-defined]
        "_execute_action": original_execute_action,
        "_compact_action_result": original_compact,
        "_build_user_prompt": original_build_user_prompt,
        "_summarized_knowledge_lines": original_knowledge_lines,
    }
    return "bankruptcy_judge: OK"
