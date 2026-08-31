"""Memory-spine graft (TP10, 2026-08-31) — persistent notes, note-to-self echo,
honest last-turn accounting. Rank-2 in the path-to-7 plan.

Design source: docs/research-2026-08-31/R11-schema-traces-mining.md §3 (the
95-99% Schema harness re-injects the agent's own notes.md every turn, echoes
its prior intent/note-to-self verbatim, and reports honest last-turn
accounting). Why it should pay here: R9 §2 forensics — cross-attempt amnesia
(sp80: 15 attempts, fresh exploration each time) and the 08-29 review's 43%
zero-level plays that are memory/control-bound, not action-starved.

STOCK BEHAVIOUR (verified in the bundled tool_agent.py) and the TRUE DELTA:

(a) NOTES. Stock already harvests labelled world-model text from the
    assistant channel (_extract_scientist_note :412, applied :1335) and
    re-injects the LIVE values into every user prompt
    (_summarized_knowledge_lines :1358, used :1472) — reinjection per turn is
    NOT the gap. The gaps are: each harvest OVERWRITES the field (no history);
    _update_summarized_knowledge_from_step_summary :1343 WIPES every field
    except cross_level_notes on level_transition/run_complete/game_over
    (graft_throughput's TP_KEEP_NOTES_ON_GAME_OVER suppresses only the
    game_over wipe); and _ensure_session :1140 clears everything on a session
    change, so nothing survives into a new pass of the same game.
    DELTA: a module-level per-GAME journal, synced by diffing
    _summarized_knowledge (so it also catches graft_emission's direct
    reasoning-channel writes), snapshotted BEFORE the wipe and BEFORE a
    session reset, and re-injected as a capped tail (TP10_NOTES_CAP, default
    4000 chars) under "YOUR NOTES (persistent):". Entries identical to the
    live block are skipped, so on healthy turns the block only carries what
    stock has lost (earlier levels, pre-wipe state, earlier passes).

(b) SUGGESTION ECHO. Stock maps "Plan:"/"Next test:" into current_plan —
    also overwritten and wiped; there is no verbatim echo. DELTA: harvest the
    last one-line `Next:` / `Suggestion:` from assistant text and echo it
    verbatim at the TOP of the next user prompt ("YOUR PRIOR INTENT: ...");
    consumed after one echo so a stale note is never re-served as fresh.
    J10-F1: the harvested line is STRIPPED from the text handed to the stock
    harvest — _extract_labeled_blocks (:375-:408) glues any unlabeled line
    into the preceding labeled block, so an unstripped `Next:` line would be
    absorbed into current_plan/world_model and re-served every turn as a
    standing plan (and journaled as a stale imperative).

(c) HONEST ACCOUNTING. Stock's prompt header (:1416) reports executed count
    and names but never committed-vs-executed: requested_count /
    stopped_early / state ARE recorded per action() payload (:329-:338,
    :1674-:1682) and then dropped by _summarize_step_sequence (:1250) —
    stop_reason and level ARE kept by the stock summary (:1288, :1293).
    DELTA: augment the step summary with the committed total and end state
    from the SAME recorded payloads, and prepend one line:
    "LAST TURN: committed N action(s), M executed, ended level=L, state=S."
    plus how many were dropped and the recorded stop_reason when a batch was
    cut short. No mispredict detection is invented — only recorded fields.
    J10-F2: requested_count is computed AFTER graft_throughput's batch-cap
    truncation (:1760, :1890 sit downstream of _normalize_python_actions,
    which TP wraps to truncate), so requested_count alone under-reports what
    the model committed. TP10 therefore also wraps _normalize_python_actions
    — whose stock either normalizes EVERY item or raises, never partially
    drops — to record the RAW batch size per action() call (works in both
    install orders because TP's cap wrapper hands the raw value to the inner
    chain before truncating), and reports cap-truncated actions explicitly
    ("N truncated by the harness batch cap").

Seams (all rebinding; stock callables in the module-level _STOCK dict so
stacked wrappers survive later installs):
  ToolAgent._build_user_prompt                       — inject (a)(b)(c)
  ToolAgent._update_summarized_knowledge_from_assistant — harvest (a)(b),
                                                       intent line stripped first
  ToolAgent._update_summarized_knowledge_from_step_summary — pre-wipe snapshot
  ToolAgent._ensure_session                          — game key + pre-reset snapshot
  ToolAgent._summarize_step_sequence                 — committed/state fields
  ToolAgent._normalize_python_actions                — raw pre-clamp batch size
  ToolAgent._run_python_tool                         — per-tool-call raw-count reset

Flags (read at call time): TP10_ENABLE=1 master ("0" = pass-through);
TP10_NOTES=1, TP10_ECHO=1, TP10_ACCOUNTING=1 per behaviour;
TP10_NOTES_CAP=2000 chars of injected notes tail (J10-F4: R11 §3 prescribes
<=2KB for the A/B; 4000 measured -4 retained history turns under 32k);
TP10_HOWTO_EVERY=8 — the note-to-self how-to line rides prompt 1, every Nth
prompt after, and the first prompt after a knowledge wipe (0 = wipe-only).
Fail-open: every graft path is try/except'd back to stock output.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_STATE = {"installed": False}
_STOCK: dict[str, Any] = {}
_OFF = {"0", "false", "no", "off"}

# game_key -> chronological journal of {"key","label","level","text"}
_NOTES: dict[str, list[dict[str, Any]]] = {}
_JOURNAL_MAX = 500
_JOURNAL_TRIM_TO = 400
_INTENT_MAX = 300

_LABELS = (
    ("world_model", "World model"),
    ("goal_model", "Goal model"),
    ("action_model", "Action model"),
    ("recent_findings", "Recent findings"),
    ("open_questions", "Open questions"),
    ("current_plan", "Plan"),
    ("cross_level_notes", "Cross-level notes"),
)

NOTES_HEADER = (
    "YOUR NOTES (persistent): saved from your earlier world-model updates in THIS game; "
    "they survive GAME_OVER and level changes. Reuse them instead of re-discovering."
)
ECHO_LINE = (
    "YOUR PRIOR INTENT (your note-to-self from last turn — reconsider, don't just obey): {intent}"
)
ECHO_HOWTO = (
    "To leave a note-to-self for your next turn, put one line starting `Next:` "
    "(or `Suggestion:`) in your assistant text; it will be echoed back to you verbatim."
)


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def enabled() -> bool:
    return _env("TP10_ENABLE", "1").lower() not in _OFF


def notes_enabled() -> bool:
    return enabled() and _env("TP10_NOTES", "1").lower() not in _OFF


def echo_enabled() -> bool:
    return enabled() and _env("TP10_ECHO", "1").lower() not in _OFF


def accounting_enabled() -> bool:
    return enabled() and _env("TP10_ACCOUNTING", "1").lower() not in _OFF


def notes_cap() -> int:
    try:
        return max(200, int(_env("TP10_NOTES_CAP", "2000")))
    except ValueError:
        return 2000


def howto_every() -> int:
    try:
        return max(0, int(_env("TP10_HOWTO_EVERY", "8")))
    except ValueError:
        return 8


def status() -> dict[str, Any]:
    return {
        "installed": _STATE["installed"],
        "enabled": enabled(),
        "notes": notes_enabled(),
        "echo": echo_enabled(),
        "accounting": accounting_enabled(),
        "notes_cap": notes_cap(),
        "howto_every": howto_every(),
        "games_tracked": len(_NOTES),
    }


# ------------------------------------------------------------- game key ---
def _game_key(state_path: Any) -> str:
    p = Path(state_path)
    stem = p.stem  # e.g. "<artifact_stem>_p0_tool_runtime_state"
    base = re.sub(r"_p\d+_.*$", "", stem)
    if base == stem:
        base = re.sub(r"_p\d+$", "", stem)
    return f"{p.parent}::{base or stem}"


def game_key_of(agent: Any) -> str:
    key = getattr(agent, "_tp10_game_key", None)
    return key if key else f"agent-{id(agent)}"


def _summary_level(agent: Any) -> int | None:
    try:
        summary = getattr(agent, "_last_step_summary", None) or {}
        level = summary.get("level")
        return int(level) if level is not None else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------- notes journal ---
def _latest_for_key(journal: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for entry in reversed(journal):
        if entry.get("key") == key:
            return entry
    return None


def sync_journal(agent: Any, level: int | None) -> None:
    """Diff the live _summarized_knowledge into the per-game journal.

    Diff-based (not harvest-hook-based) so notes written by ANY channel —
    stock assistant harvest, graft_emission's reasoning-channel fallback,
    direct writes — are captured before a wipe can destroy them.
    """
    if not notes_enabled():
        return
    try:
        knowledge = getattr(agent, "_summarized_knowledge", None)
        if not isinstance(knowledge, dict):
            return
        journal = _NOTES.setdefault(game_key_of(agent), [])
        for key, label in _LABELS:
            text = str(knowledge.get(key) or "").strip()
            if not text:
                continue
            latest = _latest_for_key(journal, key)
            if latest is not None and latest.get("text") == text:
                continue
            journal.append({"key": key, "label": label, "level": level, "text": text})
        if len(journal) > _JOURNAL_MAX:
            del journal[: len(journal) - _JOURNAL_TRIM_TO]
    except Exception:  # noqa: BLE001
        pass


def render_notes(agent: Any) -> str | None:
    journal = _NOTES.get(game_key_of(agent))
    if not journal:
        return None
    live = getattr(agent, "_summarized_knowledge", None) or {}
    seen: set[tuple[Any, Any]] = set()
    keys_seen: set[Any] = set()
    selected: list[dict[str, Any]] = []
    for entry in reversed(journal):  # newest -> oldest
        key = entry.get("key")
        level = entry.get("level")
        # J10-F3: a pre-first-summary (level=None) note is superseded by ANY
        # newer entry for the same key — don't re-serve refuted early guesses
        # beside their own corrections.
        if level is None and key in keys_seen:
            continue
        ident = (key, level)
        if ident in seen:
            continue
        seen.add(ident)
        keys_seen.add(key)
        # already shown verbatim in the live world-model block -> skip
        if str(live.get(entry.get("key")) or "").strip() == entry.get("text"):
            continue
        selected.append(entry)
    if not selected:
        return None
    selected.reverse()  # chronological, newest last
    lines = []
    for entry in selected:
        level = entry.get("level")
        tag = f"[L{level}] " if level is not None else ""
        lines.append(f"- {tag}{entry.get('label')}: {entry.get('text')}")
    cap = notes_cap()
    kept: list[str] = []
    total = 0
    trimmed = False
    for line in reversed(lines):  # keep the newest tail within the cap
        if kept and total + len(line) + 1 > cap:
            trimmed = True
            break
        if not kept and len(line) + 1 > cap:
            line = line[: max(0, cap - 16)].rstrip() + "... [truncated]"
        kept.append(line)
        total += len(line) + 1
    kept.reverse()
    header = NOTES_HEADER + (" (older notes trimmed)" if trimmed else "")
    return header + "\n" + "\n".join(kept)


# ------------------------------------------------------- suggestion echo ---
def split_intent(content: str) -> tuple[str, str | None]:
    """(text with `Next:`/`Suggestion:` lines removed, last one-line note).

    J10-F1: the intent line MUST be removed from the text the stock harvest
    sees — _extract_labeled_blocks glues unlabeled lines into the preceding
    labeled block, so an unstripped note-to-self would be absorbed into
    current_plan/world_model, re-served every turn, and journaled.
    """
    if not content or not content.strip():
        return content, None
    found: str | None = None
    kept: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        while line[:1] in {"-", "*"}:
            line = line[1:].lstrip()
        lowered = line.lower()
        matched = False
        for prefix in ("next:", "suggestion:"):
            if lowered.startswith(prefix):
                matched = True
                value = line[len(prefix):].strip()
                if value:
                    found = value
                break
        if not matched:
            kept.append(raw_line)
    if found and len(found) > _INTENT_MAX:
        found = found[:_INTENT_MAX].rstrip() + "..."
    return ("\n".join(kept), found) if found is not None else (content, None)


def harvest_intent(content: str) -> str | None:
    """Last one-line `Next:` / `Suggestion:` note in the assistant text."""
    return split_intent(content)[1]


# ----------------------------------------------------- honest accounting ---
def accounting_line(summary: Any) -> str | None:
    """One line from the recorded per-turn payload fields only."""
    if not isinstance(summary, dict) or not summary:
        return None
    try:
        executed = int(summary.get("executed_count") or 0)
    except (TypeError, ValueError):
        return None
    try:
        committed = int(summary.get("tp10_committed"))
    except (TypeError, ValueError):
        committed = executed
    committed = max(committed, executed)
    level = summary.get("level")
    state = summary.get("tp10_state")
    if not state:
        if summary.get("game_over"):
            state = "GAME_OVER"
        elif summary.get("run_complete"):
            state = "WIN"
        else:
            state = "NOT_FINISHED"
    line = (
        f"LAST TURN: committed {committed} action(s), {executed} executed, "
        f"ended level={level}, state={state}."
    )
    dropped = committed - executed
    if dropped > 0:
        try:
            cap_dropped = min(dropped, max(0, int(summary.get("tp10_cap_dropped") or 0)))
        except (TypeError, ValueError):
            cap_dropped = 0
        details = []
        if cap_dropped > 0:
            details.append(f"{cap_dropped} truncated by the harness batch cap")
        reason = str(summary.get("stop_reason") or "").strip()
        if reason and dropped - cap_dropped > 0:
            details.append(f"stop_reason={reason}")
        line += f" {dropped} committed action(s) were dropped before execution"
        line += f" ({'; '.join(details)})." if details else "."
    return line


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "memoryspine: SKIP (already applied)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"memoryspine: SKIP (import failed: {exc!r})"
    cls = getattr(agent_mod, "ToolAgent", None)
    if cls is None:
        return "memoryspine: SKIP (ToolAgent missing)"
    for name in (
        "_build_user_prompt",
        "_update_summarized_knowledge_from_assistant",
        "_update_summarized_knowledge_from_step_summary",
        "_ensure_session",
        "_summarize_step_sequence",
        "_normalize_python_actions",
        "_run_python_tool",
    ):
        if getattr(cls, name, None) is None:
            return f"memoryspine: SKIP ({name} seam missing)"

    _STOCK["build_user_prompt"] = cls._build_user_prompt
    _STOCK["update_from_assistant"] = cls._update_summarized_knowledge_from_assistant
    _STOCK["update_from_step_summary"] = cls._update_summarized_knowledge_from_step_summary
    _STOCK["ensure_session"] = cls._ensure_session
    _STOCK["summarize_step_sequence"] = cls._summarize_step_sequence
    _STOCK["normalize_python_actions"] = cls._normalize_python_actions
    _STOCK["run_python_tool"] = cls._run_python_tool

    # -- seam: session identity + pre-reset snapshot ------------------------
    def ensure_session(self, state_path):
        try:
            if enabled():
                old_dir = getattr(self, "_session_runtime_dir", None)
                new_dir = getattr(Path(state_path), "parent", None)
                if old_dir is not None and new_dir is not None and old_dir != new_dir:
                    # stock is about to clear _summarized_knowledge: snapshot
                    # it under the OLD game key first.
                    sync_journal(self, _summary_level(self))
        except Exception:  # noqa: BLE001
            pass
        result = _STOCK["ensure_session"](self, state_path)
        try:
            self._tp10_game_key = _game_key(state_path)
        except Exception:  # noqa: BLE001
            pass
        return result

    ensure_session._tp10_stock = _STOCK["ensure_session"]
    cls._ensure_session = ensure_session

    # -- seam: harvest (notes sync + intent) --------------------------------
    def update_from_assistant(self, content):
        text = content
        intent = None
        if enabled() and echo_enabled():
            try:
                # J10-F1: remove the note-to-self line BEFORE the stock
                # harvest sees it, or _extract_labeled_blocks glues it into
                # the preceding labeled block for the rest of the game.
                stripped, intent = split_intent(str(content or ""))
                if intent is not None:
                    text = stripped
            except Exception:  # noqa: BLE001
                text, intent = content, None
        result = _STOCK["update_from_assistant"](self, text)
        if not enabled():
            return result
        try:
            if intent:
                self._tp10_intent = intent
            sync_journal(self, _summary_level(self))
        except Exception:  # noqa: BLE001
            pass
        return result

    update_from_assistant._tp10_stock = _STOCK["update_from_assistant"]
    cls._update_summarized_knowledge_from_assistant = update_from_assistant

    # -- seam: pre-wipe snapshot --------------------------------------------
    def update_from_step_summary(self):
        if enabled():
            try:
                summary = getattr(self, "_last_step_summary", None) or {}
                level = _summary_level(self)
                if summary.get("level_transition") and isinstance(level, int) and level > 1:
                    # the knowledge being wiped described the level just left
                    level = level - 1
                sync_journal(self, level)
                if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
                    # J10-F4: re-teach the note-to-self affordance on the
                    # first prompt after a knowledge wipe
                    self._tp10_wipe_pending = True
            except Exception:  # noqa: BLE001
                pass
        return _STOCK["update_from_step_summary"](self)

    update_from_step_summary._tp10_stock = _STOCK["update_from_step_summary"]
    cls._update_summarized_knowledge_from_step_summary = update_from_step_summary

    # -- seam: raw pre-clamp batch size (J10-F2) ----------------------------
    # graft_throughput's batch cap truncates inside its _normalize_python_actions
    # wrapper, and requested_count (:1760, :1890) is computed AFTER that — so
    # requested_count alone under-reports what the model committed. Stock
    # normalize either accepts every item or raises (:1614-:1650, no partial
    # drop), and TP's cap wrapper hands the raw value to the inner chain
    # before truncating, so this wrapper sees the true batch size in BOTH
    # install orders. Recorded only when the chain returns (a cap REFUSAL
    # raises out of the outer wrapper and the model gets the explicit error).
    def normalize_python_actions(self, value):
        result = _STOCK["normalize_python_actions"](self, value)
        if enabled():
            try:
                if isinstance(value, (list, tuple)):
                    raw = len(value)
                elif isinstance(value, (str, dict)):
                    raw = 1
                else:
                    raw = len(result)
                calls = getattr(self, "_tp10_raw_committed", None)
                if not isinstance(calls, list):
                    # lazily create so the capture also works when a caller
                    # bypasses _run_python_tool (the run wrapper still resets
                    # the list at the top of every real tool call)
                    calls = []
                    self._tp10_raw_committed = calls
                calls.append(max(int(raw), len(result)))
            except Exception:  # noqa: BLE001
                pass
        return result

    normalize_python_actions._tp10_stock = _STOCK["normalize_python_actions"]
    cls._normalize_python_actions = normalize_python_actions

    def run_python_tool(self, state_path, arguments):
        try:
            self._tp10_raw_committed = []
        except Exception:  # noqa: BLE001
            pass
        return _STOCK["run_python_tool"](self, state_path, arguments)

    run_python_tool._tp10_stock = _STOCK["run_python_tool"]
    cls._run_python_tool = run_python_tool

    # -- seam: committed/state ride the recorded step summary ---------------
    def summarize_step_sequence(self, action_results):
        summary = _STOCK["summarize_step_sequence"](self, action_results)
        if not enabled() or summary is None:
            return summary
        try:
            committed = 0
            state = None
            for item in action_results or []:
                if not isinstance(item, dict):
                    continue
                requested = item.get("requested_count")
                if requested is None:
                    requested = item.get("executed_count")
                if requested is None:
                    requested = 1
                try:
                    committed += max(0, int(requested))
                except (TypeError, ValueError):
                    committed += 1
                if item.get("executed") and item.get("state") is not None:
                    state = item.get("state")
            # J10-F2: prefer the true pre-clamp count captured at the
            # normalize seam; the difference is what the batch cap discarded.
            raw_calls = getattr(self, "_tp10_raw_committed", None)
            if isinstance(raw_calls, list) and raw_calls:
                raw_total = sum(int(n) for n in raw_calls)
                if raw_total > committed:
                    summary["tp10_cap_dropped"] = raw_total - committed
                    committed = raw_total
            summary["tp10_committed"] = committed
            if state is not None:
                summary["tp10_state"] = str(state)
        except Exception:  # noqa: BLE001
            pass
        return summary

    summarize_step_sequence._tp10_stock = _STOCK["summarize_step_sequence"]
    cls._summarize_step_sequence = summarize_step_sequence

    # -- seam: prompt injection ---------------------------------------------
    def build_user_prompt(self, action_num, *args, **kwargs):
        text = _STOCK["build_user_prompt"](self, action_num, *args, **kwargs)
        if not enabled():
            return text
        try:
            prefix: list[str] = []
            if accounting_enabled():
                line = accounting_line(getattr(self, "_last_step_summary", None))
                if line:
                    prefix.append(line)
            if echo_enabled():
                intent = getattr(self, "_tp10_intent", None)
                if intent:
                    prefix.append(ECHO_LINE.format(intent=intent))
                    self._tp10_intent = None  # one echo per note; never re-serve stale intent
            suffix: list[str] = []
            if notes_enabled():
                # final sync catches direct writes (e.g. TP5 reasoning harvest)
                sync_journal(self, _summary_level(self))
                block = render_notes(self)
                if block:
                    suffix.append(block)
            if echo_enabled():
                # J10-F4: the how-to line is ~150 chars on EVERY turn if
                # unconditional; ride prompt 1, every Nth prompt, and the
                # first prompt after a knowledge wipe.
                count = int(getattr(self, "_tp10_prompt_count", 0) or 0) + 1
                self._tp10_prompt_count = count
                every = howto_every()
                periodic = every > 0 and (count - 1) % every == 0
                if periodic or bool(getattr(self, "_tp10_wipe_pending", False)):
                    suffix.append(ECHO_HOWTO)
                    self._tp10_wipe_pending = False
            if not prefix and not suffix:
                return text
            return "\n".join([*prefix, text, *suffix])
        except Exception:  # noqa: BLE001
            return text

    build_user_prompt._tp10_stock = _STOCK["build_user_prompt"]
    cls._build_user_prompt = build_user_prompt

    _STATE["installed"] = True
    return "memoryspine: OK"
