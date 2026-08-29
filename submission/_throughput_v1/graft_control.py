"""Memory & control graft (Pack 2, 2026-08-29) — the loop keeps its state,
probes before it reasons, and notices when it is stuck.

Installed AFTER graft_throughput (it relies on its ON_CUT hook and wraps the
seams it already wrapped). Every behaviour has a TP2_* flag read at call time;
TP2_ENABLE=0 makes every seam a pass-through.

Seams:
  (A) SUMMARY  graft_throughput.ON_CUT + ToolAgent._update_summarized_knowledge_from_step_summary
               harness-owned summary: when the trimmer cuts history, a short
               non-thinking completion compresses the dropped turns into the
               seven carried note fields (the existing prompt channel); on a
               level-up the whole history is compressed into cross-level notes
               + action model before the level-specific keys are wiped.
  (B) PROBE    _HarnessGameSession.play — at game start each available
               keyboard action is executed once and up to TP2_PROBE_CLICKS
               salient components are clicked; the effect table rides the
               user prompt while the agent is on that level.
  (C) STALL    _HarnessGameSession._execute_action + ToolAgent._build_user_prompt
               + ToolAgent.analyze — HUD-aware frame hashing; after
               TP2_STALL_T1 actions without a new board state a STAGNATION
               directive is appended to the prompt; after TP2_STALL_T2 the
               harness issues a level RESET (max TP2_STALL_RESETS_PER_LEVEL).
  (D) STREAK   _HarnessGameSession.step_env — inside one python tool call,
               after TP2_STREAK_N consecutive no-effect actions further
               actions are refused with a readable error.
  (E) DIFF     _HarnessGameSession._execute_action + ToolAgent._compact_action_result
               every action result carries a compact diff summary (changed
               cells, HUD-excluded count, bbox, colours added/removed).

Evidence: docs/STRATEGY-2026-08-29-independent-review-path-to-6.md §4,
docs/research-2026-08-29/R2-literature-mechanisms.md, R4 §1.2/§4.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter, deque
from typing import Any

_STATE = {"installed": False}
_OFF = {"0", "false", "no", "off"}
_lock = threading.Lock()

SUMMARY_SYSTEM_PROMPT = (
    "You compress the working notes of an agent playing an unknown 64x64 grid game. "
    "From the transcript, write EXACTLY these seven lines and nothing else:\n"
    "World model: <what the level contains and how it behaves, verified facts first>\n"
    "Goal model: <what seems to complete the level; mark guesses as guesses>\n"
    "Action model: <what each action does, with exact effects>\n"
    "Recent findings: <the newest confirmed observations>\n"
    "Open questions: <what is still unknown>\n"
    "Plan: <the best next steps>\n"
    "Cross-level notes: <rules likely to hold on later levels>\n"
    "Keep coordinates, colours and counts exact. Keep ruled-out hypotheses as ruled out. "
    "At most 220 words total."
)
LEVEL_SUMMARY_SYSTEM_PROMPT = (
    "The agent just completed a level of an unknown 64x64 grid game and moves to the next "
    "level of the same game. From the transcript and notes, write EXACTLY these two lines:\n"
    "Action model: <what each action does, with exact effects; controls usually persist>\n"
    "Cross-level notes: <mechanics, goal pattern and lessons that should transfer; "
    "drop layout details specific to the finished level>\n"
    "At most 150 words total."
)
STAGNATION_T1_TEXT = (
    "STAGNATION WARNING: the last {n} actions produced NO board state you had not already "
    "seen on this level (HUD counters ignored). Do not repeat that pattern. Enumerate every "
    "(action, target) you have NOT tried on this level — untested keyboard actions, unclicked "
    "objects, different orderings — and take the most informative untried one now."
)
STAGNATION_RESET_TEXT = (
    "HARNESS NOTE: the level was RESET by the harness at action {at} after {n} actions without "
    "a new board state; the board is back at the level's opening state. Your earlier notes still "
    "apply. Choose a different approach than the one that stalled."
)
STREAK_MESSAGE = (
    "no_effect_streak: the last {n} actions changed nothing on the board. The rest of this batch "
    "was not executed. Observe current_frame and pick a different action or object."
)


# ----------------------------------------------------------------- flags ---
def _env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _flag(name: str, default: str = "1") -> bool:
    return _env(name, default).lower() not in _OFF


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def enabled() -> bool:
    return _flag("TP2_ENABLE")


def summary_enabled() -> bool:
    return enabled() and _flag("TP2_SUMMARY")


def probe_enabled() -> bool:
    return enabled() and _flag("TP2_PROBE")


def stall_enabled() -> bool:
    return enabled() and _flag("TP2_STALL")


def streak_enabled() -> bool:
    return enabled() and _flag("TP2_STREAK")


def diff_enabled() -> bool:
    return enabled() and _flag("TP2_DIFF")


def probe_clicks() -> int:
    return max(0, _int("TP2_PROBE_CLICKS", 3))


def stall_t1() -> int:
    return max(1, _int("TP2_STALL_T1", 10))


def stall_t2() -> int:
    return max(stall_t1() + 1, _int("TP2_STALL_T2", 40))


def stall_resets_per_level() -> int:
    return max(0, _int("TP2_STALL_RESETS_PER_LEVEL", 2))


def streak_n() -> int:
    return max(1, _int("TP2_STREAK_N", 3))


def summary_max_tokens() -> int:
    return max(100, _int("TP2_SUMMARY_MAX_TOKENS", 600))


def status() -> dict[str, Any]:
    return {
        "installed": _STATE["installed"], "enabled": enabled(),
        "summary": summary_enabled(), "probe": probe_enabled(), "probe_clicks": probe_clicks(),
        "stall": stall_enabled(), "stall_t1": stall_t1(), "stall_t2": stall_t2(),
        "streak": streak_enabled(), "streak_n": streak_n(), "diff": diff_enabled(),
    }


# ------------------------------------------------------------ primitives ---
Grid = tuple[tuple[int, ...], ...]


def grid_hash(grid: Grid, mask: set[tuple[int, int]] | None = None) -> str:
    h = hashlib.sha1()
    for r, row in enumerate(grid):
        if mask:
            row = tuple(0 if (r, c) in mask else v for c, v in enumerate(row))
        h.update(bytes(int(v) & 0xFF for v in row))
        h.update(b"\n")
    return h.hexdigest()[:16]


class HudMask:
    """Online HUD/timer detector: cells that change in most transitions."""

    def __init__(self, min_transitions: int = 8, threshold: float = 0.6) -> None:
        self.counts: Counter = Counter()
        self.n = 0
        self.min_transitions = min_transitions
        self.threshold = threshold

    def observe(self, before: Grid, after: Grid) -> None:
        if not before or not after or len(before) != len(after):
            return
        changed = 0
        for r, (rb, ra) in enumerate(zip(before, after)):
            if rb == ra:
                continue
            for c, (vb, va) in enumerate(zip(rb, ra)):
                if vb != va:
                    self.counts[(r, c)] += 1
                    changed += 1
        if changed:
            self.n += 1

    def mask(self) -> set[tuple[int, int]] | None:
        if self.n < self.min_transitions:
            return None
        cut = self.threshold * self.n
        out = {cell for cell, cnt in self.counts.items() if cnt > cut}
        return out or None


def diff_summary(before: Grid, after: Grid, mask: set[tuple[int, int]] | None = None) -> dict[str, Any]:
    cells: list[tuple[int, int]] = []
    added: Counter = Counter()
    removed: Counter = Counter()
    if before and after and len(before) == len(after):
        for r, (rb, ra) in enumerate(zip(before, after)):
            if rb == ra:
                continue
            for c, (vb, va) in enumerate(zip(rb, ra)):
                if vb != va:
                    cells.append((r, c))
                    added[int(va)] += 1
                    removed[int(vb)] += 1
    ex = [cell for cell in cells if not mask or cell not in mask]
    box = ex or cells
    bbox = [min(r for r, _ in box), min(c for _, c in box), max(r for r, _ in box), max(c for _, c in box)] if box else None
    return {
        "changed": len(cells),
        "changed_ex_hud": len(ex),
        "bbox": bbox,
        "colors_added": sorted(added),
        "colors_removed": sorted(removed),
    }


def components(grid: Grid, background: int | None = None) -> list[dict[str, Any]]:
    if not grid:
        return []
    h, w = len(grid), len(grid[0])
    if background is None:
        background = Counter(v for row in grid for v in row).most_common(1)[0][0]
    seen = [[False] * w for _ in range(h)]
    out: list[dict[str, Any]] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] == background:
                continue
            color = grid[r0][c0]
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            out.append({
                "color": int(color), "area": len(cells), "cells": cells,
                "bbox": [min(rs), min(cs), max(rs), max(cs)],
                "center": (round(sum(rs) / len(rs)), round(sum(cs) / len(cs))),
            })
    out.sort(key=lambda d: d["area"])
    return out


def salient_clicks(grid: Grid, k: int) -> list[tuple[int, int, dict[str, Any]]]:
    """Centres of up to k small, rare-colour, non-background components."""
    comps = components(grid)
    if not comps or k <= 0:
        return []
    color_area: Counter = Counter()
    for comp in comps:
        color_area[comp["color"]] += comp["area"]

    def score(comp: dict[str, Any]) -> tuple:
        small = 0 if 2 <= comp["area"] <= 60 else 1
        return (small, color_area[comp["color"]], comp["area"])

    picked: list[tuple[int, int, dict[str, Any]]] = []
    used_colors: Counter = Counter()
    for comp in sorted(comps, key=score):
        if used_colors[comp["color"]] >= 2:
            continue
        r, c = comp["center"]
        if (r, c) not in set(comp["cells"]):
            r, c = min(comp["cells"], key=lambda rc: abs(rc[0] - r) + abs(rc[1] - c))
        picked.append((r, c, {"color": comp["color"], "area": comp["area"], "bbox": comp["bbox"]}))
        used_colors[comp["color"]] += 1
        if len(picked) >= k:
            break
    return picked


# ------------------------------------------------------- session state ---
class SessionState:
    def __init__(self) -> None:
        self.hud = HudMask()
        self.seen: set[str] = set()
        self.since_new = 0
        self.level: int | None = None
        self.resets_this_level = 0
        self.last_reset_note: str | None = None
        self.streak = 0
        self.actions = 0
        self.recent_hashes: deque = deque(maxlen=64)


def _state(session: Any) -> SessionState:
    st = getattr(session, "_tp2", None)
    if st is None:
        st = SessionState()
        try:
            session._tp2 = st
        except Exception:  # noqa: BLE001
            pass
    return st


def _session_of(agent: Any) -> Any:
    cb = getattr(agent, "_step_env_callback", None)
    return getattr(cb, "__self__", None)


def _grid(session: Any, solver_mod: Any) -> Grid:
    try:
        return solver_mod._grid_from_state(session.game.current_state)
    except Exception:  # noqa: BLE001
        return ()


def _after_action(st: SessionState, before: Grid, after: Grid, payload: dict[str, Any]) -> None:
    """Update diff / HUD / stall / streak state after one executed action."""
    mask = st.hud.mask()
    diff = diff_summary(before, after, mask) if (before and after) else None
    if diff is not None and diff_enabled():
        payload["diff"] = diff
    st.hud.observe(before, after)
    st.actions += 1
    level = payload.get("level")
    if level is not None and level != st.level:
        st.level = level
        st.seen = set()
        st.since_new = 0
        st.resets_this_level = 0
    if after:
        h = grid_hash(after, mask)
        st.recent_hashes.append(h)
        if h in st.seen:
            st.since_new += 1
        else:
            st.seen.add(h)
            st.since_new = 0
    if payload.get("executed"):
        no_effect = (diff["changed_ex_hud"] == 0) if diff is not None else (not payload.get("board_changed"))
        animated = int(payload.get("frame_count") or 1) > 1
        st.streak = st.streak + 1 if (no_effect and not animated) else 0


# ------------------------------------------------------------ summaries ---
def _transcript(messages: list[dict[str, Any]], cap: int = 14000) -> str:
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        if role == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                parts.append("ASSISTANT: " + content.strip()[:1500])
            for call in m.get("tool_calls") or []:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                parts.append("TOOL CALL: " + str(args)[:600])
        elif role == "tool":
            parts.append("TOOL RESULT: " + str(m.get("content", ""))[:700])
        elif role == "user":
            content = m.get("content")
            if isinstance(content, list):
                content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
            text = str(content or "")
            head = text.split("\n", 2)[:2]
            parts.append("USER: " + " ".join(head)[:300])
    text = "\n".join(parts)
    return text[-cap:] if len(text) > cap else text


def _post_summary(agent: Any, system_prompt: str, user_text: str, agent_mod: Any) -> str:
    import requests  # noqa: PLC0415 — the bundle already depends on it

    from inference.utils.openai_compat import build_chat_payload  # noqa: PLC0415

    model = agent._model
    payload = build_chat_payload(
        provider=model.provider, model=model.model_id,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}],
        max_tokens=summary_max_tokens(), temperature=0.2, top_p=0.95, top_k=20,
        thinking=False, tools=None, tool_choice=None, seed=None,
    )
    response = requests.post(f"{model.base_url.rstrip('/')}/chat/completions",
                             headers=agent._headers(), json=payload, timeout=180)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return str(content)


def _notes_text(agent: Any) -> str:
    notes = getattr(agent, "_summarized_knowledge", {}) or {}
    labels = [("World model", "world_model"), ("Goal model", "goal_model"), ("Action model", "action_model"),
              ("Recent findings", "recent_findings"), ("Open questions", "open_questions"),
              ("Plan", "current_plan"), ("Cross-level notes", "cross_level_notes")]
    lines = [f"{label}: {notes.get(key)}" for label, key in labels if notes.get(key)]
    return "\n".join(lines)


def _summarize_into_notes(agent: Any, dropped: list[dict[str, Any]], agent_mod: Any) -> bool:
    text = _transcript(dropped)
    if not text.strip():
        return False
    prev = _notes_text(agent)
    user_text = ("Previous notes:\n" + prev + "\n\n" if prev else "") + "Transcript of the turns being compressed:\n" + text
    content = _post_summary(agent, SUMMARY_SYSTEM_PROMPT, user_text, agent_mod)
    note = agent_mod._extract_scientist_note(content)
    if not note or not any(note.values()):
        return False
    for key, value in note.items():
        if value:
            agent._summarized_knowledge[key] = value
    return True


def _summarize_level_boundary(agent: Any, agent_mod: Any) -> dict[str, str]:
    history = list(getattr(agent, "_history_messages", []) or [])
    text = _transcript(history)
    prev = _notes_text(agent)
    user_text = ("Notes so far:\n" + prev + "\n\n" if prev else "") + "Transcript:\n" + text
    content = _post_summary(agent, LEVEL_SUMMARY_SYSTEM_PROMPT, user_text, agent_mod)
    note = agent_mod._extract_scientist_note(content)
    return {k: v for k, v in (note or {}).items() if v and k in ("action_model", "cross_level_notes")}


# --------------------------------------------------------------- probe ---
_KEYBOARD = [("ACTION1", "UP"), ("ACTION2", "DOWN"), ("ACTION3", "LEFT"), ("ACTION4", "RIGHT"), ("ACTION5", "SPACE")]


def _fmt_effect(diff: dict[str, Any] | None, payload: dict[str, Any]) -> str:
    if payload.get("level_completed"):
        return "COMPLETED THE LEVEL"
    if payload.get("game_over"):
        return "GAME OVER (level was reset)"
    if not diff:
        return "board changed" if payload.get("board_changed") else "no effect"
    if diff["changed_ex_hud"] == 0 and diff["changed"] == 0:
        return "no effect"
    if diff["changed_ex_hud"] == 0:
        return f"only HUD-like cells changed ({diff['changed']})"
    b = diff["bbox"]
    return (f"changed {diff['changed_ex_hud']} cells in rows {b[0]}-{b[2]} cols {b[1]}-{b[3]}; "
            f"colours appeared {diff['colors_added']} vanished {diff['colors_removed']}")


def run_probe(session: Any, solver_mod: Any, arcengine: Any) -> dict[str, Any] | None:
    """Execute the level-start probe on a fresh game. Returns the table record."""
    game = session.game
    state = game.current_state
    try:
        available = set(int(a) for a in state.available_actions)
    except Exception:  # noqa: BLE001
        return None
    st = _state(session)
    try:
        level = int(solver_mod._level_number(game))
    except Exception:  # noqa: BLE001
        level = int(getattr(state, "levels_completed", 0) or 0) + 1
    lines: list[str] = []
    executed = 0

    def do(action_name: str, data: dict[str, Any], label: str) -> dict[str, Any]:
        nonlocal executed
        action = arcengine.ActionInput(id=arcengine.GameAction.from_name(action_name), data=data)
        before = _grid(session, solver_mod)
        payload = session._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
        executed += 1
        diff = payload.get("diff")
        if diff is None:
            after = _grid(session, solver_mod)
            diff = diff_summary(before, after, st.hud.mask()) if before and after else None
        lines.append(f"- {label}: {_fmt_effect(diff, payload)}")
        return payload

    stop = False
    for engine_name, label in _KEYBOARD:
        value = int(arcengine.GameAction.from_name(engine_name).value)
        if value not in available:
            continue
        payload = do(engine_name, {}, label)
        if payload.get("level_completed") or payload.get("run_complete") or payload.get("game_over"):
            stop = True
            break
        try:
            available = set(int(a) for a in game.current_state.available_actions)
        except Exception:  # noqa: BLE001
            pass
    click_value = int(arcengine.GameAction.from_name("ACTION6").value)
    if not stop and click_value in available and probe_clicks() > 0:
        grid = _grid(session, solver_mod)
        for r, c, info in salient_clicks(grid, probe_clicks()):
            payload = do("ACTION6", {"x": int(c), "y": int(r)},
                         f"MOUSE(row {r}, col {c}) on a colour-{info['color']} object of {info['area']} cells")
            if payload.get("level_completed") or payload.get("run_complete") or payload.get("game_over"):
                break
    if not lines:
        return None
    text = ("Harness probe at the start of this level (each available action tried once; "
            f"{executed} actions spent, all recorded in `history`):\n" + "\n".join(lines))
    return {"level": level, "text": text, "actions": executed}


# --------------------------------------------------------------- install ---
def install() -> str:
    if _STATE["installed"]:
        return "control: SKIP (already applied)"
    try:
        import graft_throughput as tp  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"control: SKIP (graft_throughput missing: {exc!r})"
    if not tp._STATE.get("installed"):
        return "control: SKIP (graft_throughput not installed)"
    try:
        import arcengine  # noqa: PLC0415
        from inference.agent import tool_agent as agent_mod  # noqa: PLC0415
        from inference.framework import solver as solver_mod  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"control: SKIP (import failed: {exc!r})"
    agent_cls = agent_mod.ToolAgent
    session_cls = solver_mod._HarnessGameSession
    for name in ("_build_user_prompt", "analyze", "_compact_action_result", "_run_python_tool",
                 "_update_summarized_knowledge_from_step_summary"):
        if getattr(agent_cls, name, None) is None:
            return f"control: SKIP (ToolAgent.{name} missing)"
    for name in ("play", "step_env", "_execute_action", "_error_payload"):
        if getattr(session_cls, name, None) is None:
            return f"control: SKIP (_HarnessGameSession.{name} missing)"

    # (E)+(C)+(D) state: _execute_action ---------------------------------
    stock_execute_action = session_cls._execute_action

    def execute_action_wrapped(self, action, *args, **kwargs):
        before = _grid(self, solver_mod) if enabled() else ()
        payload = stock_execute_action(self, action, *args, **kwargs)
        if not enabled():
            return payload
        try:
            _after_action(_state(self), before, _grid(self, solver_mod), payload)
        except Exception:  # noqa: BLE001
            pass
        return payload

    execute_action_wrapped._tp2_stock = stock_execute_action
    session_cls._execute_action = execute_action_wrapped

    stock_compact = agent_cls._compact_action_result

    def compact(self, payload):
        out = stock_compact(self, payload)
        try:
            if diff_enabled() and isinstance(payload, dict) and isinstance(payload.get("diff"), dict):
                out["diff"] = dict(payload["diff"])
        except Exception:  # noqa: BLE001
            pass
        return out

    compact._tp2_stock = stock_compact
    agent_cls._compact_action_result = compact

    # (D) streak halt: step_env + reset at each python tool call ----------
    stock_step = session_cls.step_env

    def step_env(self, arguments):
        try:
            if streak_enabled() and not (isinstance(arguments, dict) and arguments.get("query")):
                st = _state(self)
                if st.streak >= streak_n():
                    return self._error_payload(STREAK_MESSAGE.format(n=st.streak))
        except Exception:  # noqa: BLE001
            pass
        return stock_step(self, arguments)

    step_env._tp2_stock = stock_step
    session_cls.step_env = step_env

    stock_run = agent_cls._run_python_tool

    def run_tool(self, state_path, arguments):
        try:
            sess = _session_of(self)
            if sess is not None:
                _state(sess).streak = 0
        except Exception:  # noqa: BLE001
            pass
        return stock_run(self, state_path, arguments)

    run_tool._tp2_stock = stock_run
    agent_cls._run_python_tool = run_tool

    # (B)+(C) prompt: probe table + stagnation directive -----------------
    stock_prompt = agent_cls._build_user_prompt

    def build_prompt(self, action_num, **kwargs):
        text = stock_prompt(self, action_num, **kwargs)
        if not enabled():
            return text
        try:
            extra: list[str] = []
            current_frame = kwargs.get("current_frame")
            level = getattr(current_frame, "level", None)
            probe = getattr(self, "_tp2_probe", None)
            if probe and probe_enabled() and (level is None or int(level) == int(probe["level"])):
                extra.append(probe["text"])
            sess = _session_of(self)
            if sess is not None and stall_enabled():
                st = _state(sess)
                if st.last_reset_note:
                    extra.append(st.last_reset_note)
                    st.last_reset_note = None
                if st.since_new >= stall_t1():
                    extra.append(STAGNATION_T1_TEXT.format(n=st.since_new))
            if extra:
                text = text + "\n" + "\n".join(extra)
        except Exception:  # noqa: BLE001
            pass
        return text

    build_prompt._tp2_stock = stock_prompt
    agent_cls._build_user_prompt = build_prompt

    stock_analyze = agent_cls.analyze

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, **kwargs):
        try:
            sess = getattr(step_env, "__self__", None)
            if sess is not None and stall_enabled():
                st = _state(sess)
                if (st.since_new >= stall_t2() and st.resets_this_level < stall_resets_per_level()
                        and "RESET" in [str(a).upper() for a in (valid_actions or [])]):
                    action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
                    sess._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)
                    st.resets_this_level += 1
                    st.last_reset_note = STAGNATION_RESET_TEXT.format(at=sess.action_count, n=st.since_new)
                    st.since_new = 0
                    try:
                        sess.write_runtime_state()
                    except Exception:  # noqa: BLE001
                        pass
                    action_num = sess.action_count
                    valid_actions = solver_mod._engine_action_names(sess.game)
        except Exception:  # noqa: BLE001
            pass
        stock = agent_cls.analyze._tp2_stock
        return stock(self, state_path, action_num, valid_actions=valid_actions, step_env=step_env, **kwargs)

    analyze._tp2_stock = stock_analyze
    agent_cls.analyze = analyze

    # (B) probe at game start ---------------------------------------------
    stock_play = session_cls.play

    def play(self):
        if probe_enabled():
            try:
                if int(getattr(self, "action_count", 0) or 0) == 0:
                    self.seed_initial_history()
                    record = run_probe(self, solver_mod, arcengine)
                    if record:
                        self.analyzer._tp2_probe = record
                        self.write_runtime_state()
            except Exception:  # noqa: BLE001
                pass
        return stock_play(self)

    play._tp2_stock = stock_play
    session_cls.play = play

    # (A) summaries -------------------------------------------------------
    def on_cut(agent, dropped):
        if summary_enabled():
            _summarize_into_notes(agent, dropped, agent_mod)

    tp.ON_CUT = on_cut

    stock_notes = agent_cls._update_summarized_knowledge_from_step_summary

    def update_notes(self):
        try:
            summary = self._last_step_summary
            if summary_enabled() and summary and summary.get("level_transition"):
                carried = _summarize_level_boundary(self, agent_mod)
                result = stock_notes(self)
                for key, value in carried.items():
                    self._summarized_knowledge[key] = value
                return result
        except Exception:  # noqa: BLE001
            pass
        return stock_notes(self)

    update_notes._tp2_stock = stock_notes
    agent_cls._update_summarized_knowledge_from_step_summary = update_notes

    _STATE["installed"] = True
    return "control: OK"
