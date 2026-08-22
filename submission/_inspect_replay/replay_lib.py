"""replay_lib — shared machinery for the INSPECTION-PROMPT REPLAY falsifier.

Used in two places:
  1. locally by extract_samples.py (context reconstruction + sandbox
     fidelity gate against the recorded 27B outputs);
  2. on Kaggle by the arc3-inspect-replay kernel (sandbox execution of the
     small model's emitted snippets + deterministic grading).

It vendors NOTHING itself: it drives the exact taaf_anim ARC3-Inference
bundle (feature/animation-awareness 9158303 — the code that produced the
corpus transcripts). Point INSPECT_BUNDLE_SRC at the bundle's src dir
(the dir containing ARC3-Inference/).

Design doc: docs/RESEARCH-2026-08-22-slotmath-and-top3.md Addendum 2.
Pre-registered kill criteria:
  - small-model probe-code execution-error rate > 1.5x the 27B's recorded
    error rate on the same samples, OR
  - < 70% key-fact recovery of the 27B's printed probe outputs
  => inspection-routing cascade DEAD.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# --- bundle import ----------------------------------------------------------

_DEFAULT_BUNDLE = (
    "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
    "9a35ebe3-1f42-4aac-97ed-d1de24bf2346/scratchpad/taaf_anim/src"
)
BUNDLE_SRC = Path(os.environ.get("INSPECT_BUNDLE_SRC", _DEFAULT_BUNDLE))

# Production env for the corpus runs (xpl7 taaf_setup_env.json, verified).
PRODUCTION_ENV = {
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "1",
    "LOCAL_ANALYZER_MODEL_ID": "Qwen/Qwen3.8-27B-FP8",
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "LOCAL_ANALYZER_BASE_URL": "http://127.0.0.1:9/v1",
}


def bundle_setup() -> Any:
    """Set production env, put the bundle on sys.path, import tool_agent."""
    for key, value in PRODUCTION_ENV.items():
        os.environ.setdefault(key, value)
    arc3 = BUNDLE_SRC / "ARC3-Inference"
    if str(arc3) not in sys.path:
        sys.path.insert(0, str(arc3))
    import inference.agent.tool_agent as TA  # noqa: PLC0415

    return TA


# --- transcript parsing -----------------------------------------------------

HDR_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| (\S+) ---\s*$")
SECTION_LABELS = (
    "SYSTEM PROMPT",
    "USER PROMPT",
    "MODEL RESPONSE META",
    "THINKING",
    "ASSISTANT",
    "ANALYZER STATUS",
)
SECTION_RE = re.compile(
    r"^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|ASSISTANT|ANALYZER STATUS|TOOL CALL: [^\]]+|TOOL RESULT: [^\]]+)\]\s*$"
)

ERROR_SIGNATURES = (
    "Traceback (most recent call last):",
    "Tool timed out after",
    "Sandbox process",
    "Python syntax error:",
    "is not allowed in the sandbox",
)


def parse_transcript(path: str | Path) -> list[dict[str, Any]]:
    """Parse one game transcript into blocks of ordered (label, content) sections."""
    lines = Path(path).read_text(errors="replace").split("\n")
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section_label: str | None = None
    section_lines: list[str] = []

    def flush_section() -> None:
        nonlocal section_label, section_lines
        if current is not None and section_label is not None:
            current["sections"].append((section_label, "\n".join(section_lines).strip()))
        section_label, section_lines = None, []

    for line in lines:
        hm = HDR_RE.match(line)
        if hm:
            flush_section()
            if current is not None:
                blocks.append(current)
            current = {
                "astep": int(hm.group(1)),
                "action": int(hm.group(2)),
                "time": hm.group(3),
                "agent": hm.group(4),
                "sections": [],
            }
            continue
        sm = SECTION_RE.match(line)
        if sm and current is not None:
            flush_section()
            section_label = sm.group(1)
            continue
        if section_label is not None:
            section_lines.append(line)
    flush_section()
    if current is not None:
        blocks.append(current)
    return blocks


META_FIELD_RE = {
    "finish_reason": re.compile(r"^finish_reason: (.*)$", re.M),
    "tool_call_count": re.compile(r"^tool_call_count: (\d+)", re.M),
    "content_chars": re.compile(r"^content_chars: (\d+)", re.M),
    "reasoning_chars": re.compile(r"^reasoning_chars: (\d+)", re.M),
    "markup_in_text": re.compile(r"^tool_call_markup_in_text: (\S+)", re.M),
    "recovered": re.compile(r"^tool_calls_recovered_from_markup: (\S+)", re.M),
}


def parse_meta(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, rx in META_FIELD_RE.items():
        m = rx.search(text)
        out[key] = m.group(1) if m else None
    for key in ("tool_call_count", "content_chars", "reasoning_chars"):
        out[key] = int(out[key]) if out[key] is not None else 0
    # raw_tool_calls JSON (may be _trim_log_text-truncated at 4000 chars)
    out["raw_tool_calls"] = None
    idx = text.find("raw_tool_calls:")
    if idx >= 0:
        raw = text[idx + len("raw_tool_calls:"):].strip()
        try:
            out["raw_tool_calls"] = json.loads(raw)
        except json.JSONDecodeError:
            out["raw_tool_calls"] = None  # truncated
    return out


TOOLCALL_CODE_RE = re.compile(
    r"<parameter=code>\n(.*)\n</parameter>", re.S
)


def code_from_markup(markup: str) -> str | None:
    m = TOOLCALL_CODE_RE.search(markup)
    if m:
        return m.group(1)
    # empty parameter body
    if "<parameter=code>\n</parameter>" in markup or "<parameter=code></parameter>" in markup:
        return ""
    return None


def split_requests(block: dict[str, Any]) -> dict[str, Any]:
    """Split a block's sections into: system, first user prompt, and a list of
    requests. Each request = one LLM call: its meta + the response artifacts
    (thinking/assistant/tool calls+results/followup user prompt) that follow it.
    """
    sections = block["sections"]
    system_prompt = None
    user_prompt = None
    pre_statuses: list[str] = []
    requests: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for label, content in sections:
        if label == "SYSTEM PROMPT" and system_prompt is None:
            system_prompt = content
            continue
        if label == "USER PROMPT" and user_prompt is None and cur is None:
            user_prompt = content
            continue
        if label == "MODEL RESPONSE META":
            if cur is not None:
                requests.append(cur)
            cur = {
                "meta": parse_meta(content),
                "thinking": None,
                "assistant": None,
                "tool_events": [],  # list of {"name","markup","code","result_display"}
                "followup_user": None,
                "statuses": [],
            }
            continue
        if cur is None:
            if label == "ANALYZER STATUS":
                pre_statuses.append(content)
            continue
        if label == "THINKING":
            cur["thinking"] = content
        elif label == "ASSISTANT":
            cur["assistant"] = content
        elif label.startswith("TOOL CALL: "):
            cur["tool_events"].append(
                {
                    "name": label[len("TOOL CALL: "):],
                    "markup": content,
                    "code": code_from_markup(content),
                    "result_display": None,
                }
            )
        elif label.startswith("TOOL RESULT: "):
            if cur["tool_events"] and cur["tool_events"][-1]["result_display"] is None:
                cur["tool_events"][-1]["result_display"] = content
        elif label == "USER PROMPT":
            cur["followup_user"] = content
        elif label == "ANALYZER STATUS":
            cur["statuses"].append(content)
    if cur is not None:
        requests.append(cur)
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "pre_statuses": pre_statuses,
        "requests": requests,
    }


ACTION_CALL_RE = re.compile(r"(?<![#\w.])action\s*\(")


def code_calls_action(code: str) -> bool:
    stripped = "\n".join(l.split("#")[0] for l in (code or "").split("\n"))
    return bool(ACTION_CALL_RE.search(stripped))


def display_is_error(display: str) -> bool:
    d = display or ""
    return any(sig in d for sig in ERROR_SIGNATURES)


# --- tool-result display inversion ------------------------------------------
# The transcript records _render_tool_result_display(dispatch.content); the
# model actually saw dispatch.content = json.dumps(payload, indent=2). Invert
# the three unambiguous shapes; return None when ambiguous.

def invert_display_to_payload(display: str) -> dict[str, Any] | None:
    d = (display or "").rstrip("\n")
    if not d:
        return None
    has_result_hdr = "\n\nresult:\n" in d or d.startswith("result:\n")
    has_error_hdr = "\n\nerror:\n" in d
    pure_error = display_is_error(d) and not has_error_hdr and not has_result_hdr
    payload: dict[str, Any] = {"tool": "python"}
    if pure_error and (d.startswith("Traceback") or d.startswith("Tool timed out")
                       or d.startswith("Sandbox process") or "Python syntax error:" in d.split("\n")[0]):
        payload["error"] = d
        return payload
    if not has_result_hdr and not has_error_hdr:
        # stdout-only (the dominant case)
        payload["returncode"] = 0
        payload["stdout"] = d + "\n" if not d.endswith("\n") else d
        # NOTE: production stdout usually ends with a newline from print();
        # keep as captured — display rstrips, so re-append one "\n".
        if "... [truncated" in d:
            payload["truncated"] = True
            payload["truncation_note"] = (
                "Tool output was cut off to stay within the ~1024-token response budget."
            )
        return payload
    if has_error_hdr and not has_result_hdr:
        stdout_part, _, err = d.rpartition("\n\nerror:\n")
        payload["error"] = err
        if stdout_part:
            payload["stdout"] = stdout_part + "\n"
        return payload
    # result-bearing displays are not safely invertible (human-readable
    # rendering of a dict). Give the caller the display so it can decide.
    return None


# --- state rehydration -------------------------------------------------------

class StateRebuilder:
    """Rebuild the exact per-call sandbox state from taaf intermediate states.

    Consumes a pre-serialized per-game state pack (built by extract_samples
    from intermediate_states.pkl so the Kaggle side never needs arcengine):
      pack = {
        "game_id", "number_of_levels",
        "frames": [grid, ...]           # final visible frame per state index
        "actions": [display, ...]       # action display leading INTO state i (i>=1)
        "levels": [levels_completed, ...]
        "all_frames": {i: [grid,...]}   # full raw frame list per state (for animation)
        "available": [[engine ids],...]
        "won": [bool,...]
      }
    """

    def __init__(self, pack: dict[str, Any], TA: Any):
        self.pack = pack
        self.TA = TA
        self._record_cache: dict[int, dict[str, Any] | None] = {}

    def level_number(self, idx: int) -> int:
        completed = int(self.pack["levels"][idx])
        n = int(self.pack["number_of_levels"])
        if self.pack["won"][idx]:
            return max(1, n)
        return max(1, min(n, completed + 1))

    def frame_payload(self, idx: int) -> dict[str, Any]:
        from inference.agent.runtime_state import Frame  # noqa: PLC0415

        grid = tuple(tuple(int(c) for c in row) for row in self.pack["frames"][idx])
        frame = Frame(grid=grid, step=idx, level=self.level_number(idx))
        return self.TA._ascii_frame_view_payload(frame)

    def history_payload(self, upto_idx: int) -> list[dict[str, Any]]:
        out = []
        for i in range(0, upto_idx + 1):
            action = "" if i == 0 else str(self.pack["actions"][i])
            out.append({"action": action, "frame": self.frame_payload(i)})
        return out

    def valid_action_names(self, idx: int) -> list[str]:
        from inference.agent.action_names import to_model_actions  # noqa: PLC0415

        id_to_name = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3",
                      4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "ACTION7"}
        names = []
        for aid in self.pack["available"][idx]:
            name = id_to_name.get(int(aid))
            if name is None or name == "RESET":
                continue
            if name not in names:
                names.append(name)
        return to_model_actions(names)

    def sandbox_state(self, idx: int, last_action_result: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "current_frame": self.frame_payload(idx),
            "history": self.history_payload(idx),
            "valid_actions": self.valid_action_names(idx),
            "last_action_result": dict(last_action_result) if isinstance(last_action_result, dict) else {},
        }

    def _record_for(self, idx: int) -> dict[str, Any] | None:
        """The solver-side animation_history record for action #idx, exactly
        as production builds it (solver.py:860-868), or None if that action
        did not animate."""
        if idx in self._record_cache:
            return self._record_cache[idx]
        self._record_cache[idx] = self._build_record(idx)
        return self._record_cache[idx]

    def _build_record(self, idx: int) -> dict[str, Any] | None:
        from inference.utils.animation import summarize_animation  # noqa: PLC0415

        frames = self.pack["all_frames"].get(str(idx)) or self.pack["all_frames"].get(idx)
        if not frames or idx < 1:
            return None
        norm = [tuple(tuple(int(c) for c in row) for row in f) for f in frames]
        before = tuple(tuple(int(c) for c in row) for row in self.pack["frames"][idx - 1])
        board_changed = self.pack["frames"][idx] != self.pack["frames"][idx - 1]
        summary = summarize_animation(norm, board_changed=board_changed)
        if summary is None:
            return None
        return {
            "action_num": idx,
            "action_display": str(self.pack["actions"][idx]),
            "before": before,
            "frames": norm,
            "summary": summary,
        }

    ANIMATION_HISTORY_DEPTH = 4  # solver.py:62

    def animation_record(self, current_idx: int, wanted: Any = None) -> dict[str, Any] | None:
        """Mirror solver.animation_record: a deque of the last 4 ANIMATED
        actions up to current_idx; None-arg returns the newest."""
        animated = [i for i in range(1, current_idx + 1) if self._record_for(i) is not None]
        window = animated[-self.ANIMATION_HISTORY_DEPTH:]
        if not window:
            return None
        if wanted is None:
            return self._record_for(window[-1])
        try:
            wanted_idx = int(wanted)
        except (TypeError, ValueError):
            return None
        for i in reversed(window):
            if i == wanted_idx:
                return self._record_for(i)
        return None


def make_replay_step_env(rebuilder: StateRebuilder, current_idx: int):
    """A step_env callback for replay: serves animation queries from the
    recorded frames; refuses real actions with a distinctive marker the
    harness host can detect (action-escalation in the cascade)."""

    class ActionAttempt(Exception):
        pass

    attempts: list[Any] = []

    def step_env(arguments: dict[str, Any]) -> dict[str, Any]:
        if str(arguments.get("query") or "").strip() == "animation":
            return {
                "executed": False,
                "query": "animation",
                "record": rebuilder.animation_record(current_idx, arguments.get("action_num")),
            }
        attempts.append(arguments.get("actions"))
        raise ActionAttempt("replay: real actions are not available")

    step_env.attempts = attempts  # type: ignore[attr-defined]
    return step_env


def run_snippet(
    TA: Any,
    rebuilder: StateRebuilder,
    state_idx: int,
    code: str,
    last_action_result: dict[str, Any] | None,
    valid_actions: list[str] | None,
    workdir: str | Path,
) -> dict[str, Any]:
    """Execute one python snippet through the REAL ToolAgent dispatch path
    against rehydrated state. Returns
    {display, payload_content, error, action_attempted}.
    """
    from inference.agent.runtime_state import write_runtime_state, Frame, HistoryEntry  # noqa: PLC0415

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = workdir / "tool_runtime_state.json"

    hist_payload = rebuilder.history_payload(state_idx)
    entries = []
    for item in hist_payload:
        fp = item["frame"]
        grid = tuple(tuple(int(c) for c in row) for row in fp["grid"])
        entries.append(
            HistoryEntry(action=item["action"], frame=Frame(grid=grid, step=fp["step"], level=fp["level"]))
        )
    current = entries[-1].frame
    write_runtime_state(state_path, current_frame=current, history=entries)

    agent = TA.ToolAgent(provider="vllm", base_url="http://127.0.0.1:9/v1")
    agent._ensure_session(state_path)
    step_env = make_replay_step_env(rebuilder, state_idx)
    agent._step_env_callback = step_env
    agent._current_valid_actions = list(
        valid_actions if valid_actions is not None else rebuilder.valid_action_names(state_idx)
    )
    agent._last_action_result = dict(last_action_result) if isinstance(last_action_result, dict) else None

    dispatch = agent._run_python_tool(state_path, {"code": code})
    content = dispatch.content
    display = TA._render_tool_result_display(content)
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return {
        "display": display,
        "content": content,
        "error": str(payload.get("error", "") or ""),
        "stdout": str(payload.get("stdout", "") or ""),
        "action_attempted": bool(getattr(step_env, "attempts")),
    }


# --- deterministic fact grading ----------------------------------------------

NUM_RE = re.compile(r"-?\d+\.?\d*")
COORD_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)|row\s*[=:]\s*(-?\d+)\s*,\s*col\s*[=:]\s*(-?\d+)")


def extract_facts(text: str) -> dict[str, Any]:
    """Numeric-token multiset + coordinate-pair multiset from a printed output."""
    nums: dict[str, int] = {}
    for m in NUM_RE.finditer(text or ""):
        tok = m.group(0)
        # normalize "3.0" vs "3"
        try:
            v = float(tok)
            tok = str(int(v)) if v == int(v) else repr(v)
        except ValueError:
            pass
        nums[tok] = nums.get(tok, 0) + 1
    coords: dict[str, int] = {}
    for m in COORD_RE.finditer(text or ""):
        r = m.group(1) or m.group(3)
        c = m.group(2) or m.group(4)
        key = f"{int(r)},{int(c)}"
        coords[key] = coords.get(key, 0) + 1
    return {"nums": nums, "coords": coords}


def _multiset_recall(ref: dict[str, int], got: dict[str, int]) -> tuple[int, int]:
    total = sum(ref.values())
    hit = sum(min(count, got.get(tok, 0)) for tok, count in ref.items())
    return hit, total


def grade_fact_recovery(ref_text: str, got_text: str, prompt_text: str = "") -> dict[str, Any]:
    """Partial-credit fact recovery of `got_text` against reference `ref_text`.

    Headline (pre-registered): numeric+coordinate multiset recall.
    Diagnostic: novel-fact recall (facts absent from the current user prompt).
    """
    ref = extract_facts(ref_text)
    got = extract_facts(got_text)
    nh, nt = _multiset_recall(ref["nums"], got["nums"])
    ch, ct = _multiset_recall(ref["coords"], got["coords"])
    hit, total = nh + ch, nt + ct
    recovery = (hit / total) if total else None

    prompt_facts = extract_facts(prompt_text)
    novel_nums = {t: c for t, c in ref["nums"].items() if t not in prompt_facts["nums"]}
    novel_coords = {t: c for t, c in ref["coords"].items() if t not in prompt_facts["coords"]}
    nnh, nnt = _multiset_recall(novel_nums, got["nums"])
    nch, nct = _multiset_recall(novel_coords, got["coords"])
    novel_hit, novel_total = nnh + nch, nnt + nct
    novel = (novel_hit / novel_total) if novel_total else None
    return {
        "recovery": recovery,
        "facts_total": total,
        "facts_hit": hit,
        "novel_recovery": novel,
        "novel_total": novel_total,
        "novel_hit": novel_hit,
    }
