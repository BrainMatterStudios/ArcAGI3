#!/usr/bin/env python3
"""Shared corpus access for the bankruptcy-judge falsifier (Stages 1-3).

Corpus: scratchpad/multirole_corpus/ — 10 run dirs with transcripts (36 files),
benchmark.json and intermediate_states.pkl per dir (v12smoke has no transcripts).
The .pkl files are first-party run artifacts produced by our own taaf harness
(explicitly named by the falsifier design); loading them requires pickle.

Design source: docs/RESEARCH-2026-08-23-searchcore-and-multirole.md §B.

Alignment facts (verified against xd/sb26 on 2026-08-24):
  - benchmark.json game_runs[i].history[j]   = j-th env action (dict: action.id, action.data{x,y})
  - intermediate_states.pkl[i][j]            = taaf GameState AFTER history[j-1]; [0] = initial state
    (len(states) == len(history)+1); state.raw.frame -> list of 64x64 grids (last = settled board),
    state.raw.levels_completed -> completed-level count; transcript 'level L' == levels_completed+1.
  - transcript blocks: one per LLM call, split on '--- analysis_step=N | action=M | HH:MM:SS | role ---';
    each block's USER PROMPT carries 'The code executed N actions in the previous sequence',
    'Current state: step S, level L', and the carried world model between
    'Working world model carried from earlier turns:' and 'end of world model'.
  - header field action=M is exact: M - 1 == env actions performed BEFORE that block's LLM call
    (verified on all 36 transcripts: final block's M == len(history) + 1, monotone throughout).
    The 'executed N actions' prompt line double-counts across resumed blocks — do NOT sum it.

Requires the project venv (taaf + arcengine) to unpickle intermediate states:
    /Users/ahmed/Documents/ArcAGI3/.venv/bin/python
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPUS = os.path.join(REPO, "scratchpad", "multirole_corpus")

RUN_DIRS = ["depthdiag", "digest1", "packv22", "xd", "xpl2", "xpl4", "xpl5", "xpl7", "y180", "y180b"]

_BLOCK_RE = re.compile(r"\n--- analysis_step=(\d+) \| action=(\d+) \| ([\d:]+) \| ([^ ]+) ---")
_LEVEL_RE = re.compile(r"Current state: step (\d+), level (\d+)")
_EXEC_RE = re.compile(r"The code executed (\d+) actions? in the previous sequence")
_WM_RE = re.compile(
    r"Working world model carried from earlier turns:\n(.*?)\nend of world model",
    re.DOTALL,
)

# Frame-0 archetype rule replicated from submission/_archetype_triage/graft_triage.py
# (the proven 0-FN dispatch): click-family only -> CLICK; movement family only -> AVATAR;
# both -> MIXED; unreadable -> CLICK (safe default).
_AVATAR_FAMILY = {1, 2, 3, 4}  # ACTION5 alone does not make a game AVATAR
_CLICK_ID = 6


@dataclass
class Block:
    idx: int              # block index within transcript (0-based)
    step: int             # analysis_step
    act_hdr: int          # 'action=' field from the header
    time: str
    level: int | None     # from 'Current state' (1-based; None if absent)
    executed_n: int       # 'executed N' prompt line (informational; double-counts on resumes)
    cum_actions: int      # env actions performed BEFORE this block's LLM call (= act_hdr - 1)
    wm_text: str | None   # carried world model (verbatim block), None if absent
    wm_hash: str          # md5[:8] of normalized wm_text ('-' if absent)


@dataclass
class Transcript:
    run_dir: str          # e.g. 'xd'
    game_key: str         # e.g. 'sb26-7fbdac44' (matches benchmark game_id)
    path: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def sid(self) -> str:
        return f"{self.run_dir}/{self.game_key}"

    def final_level(self) -> int:
        lv = [b.level for b in self.blocks if b.level is not None]
        return max(lv) if lv else 0

    def level_eventually_completed(self, level: int) -> bool:
        """True if the transcript ever shows a level greater than `level`."""
        return any(b.level is not None and b.level > level for b in self.blocks)


def parse_transcript(path: str, run_dir: str, game_key: str) -> Transcript:
    text = open(path, encoding="utf-8", errors="replace").read()
    matches = list(_BLOCK_RE.finditer(text))
    t = Transcript(run_dir=run_dir, game_key=game_key, path=path)
    last_level = None
    for i, m in enumerate(matches):
        body = text[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        lm = _LEVEL_RE.search(body)
        level = int(lm.group(2)) if lm else last_level
        em = _EXEC_RE.search(body)
        executed = int(em.group(1)) if em else 0
        wm = _WM_RE.search(body)
        wm_text = wm.group(1).strip() if wm else None
        norm = re.sub(r"\s+", " ", wm_text).strip() if wm_text else ""
        wm_hash = hashlib.md5(norm.encode()).hexdigest()[:8] if norm else "-"
        act_hdr = int(m.group(2))
        t.blocks.append(Block(
            idx=i, step=int(m.group(1)), act_hdr=act_hdr, time=m.group(3),
            level=level, executed_n=executed, cum_actions=act_hdr - 1,
            wm_text=wm_text, wm_hash=wm_hash,
        ))
        last_level = level
    return t


def list_corpus() -> list[Transcript]:
    """All 36 transcripts, parsed."""
    out = []
    for d in RUN_DIRS:
        tdir = os.path.join(CORPUS, d, "transcripts")
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith("_p0.txt"):
                continue
            game_key = fn[: -len("_p0.txt")]
            out.append(parse_transcript(os.path.join(tdir, fn), d, game_key))
    return out


# ---------------------------------------------------------------- benchmark + states

def load_benchmark(run_dir: str) -> dict:
    return json.load(open(os.path.join(CORPUS, run_dir, "benchmark.json")))


def game_run(run_dir: str, game_key: str) -> tuple[int, dict]:
    b = load_benchmark(run_dir)
    for i, gr in enumerate(b["game_runs"]):
        if gr["game_id"] == game_key:
            return i, gr
    raise KeyError(f"{game_key} not in {run_dir}/benchmark.json")


_state_cache: dict[str, list] = {}


def load_states(run_dir: str):
    if run_dir not in _state_cache:
        with open(os.path.join(CORPUS, run_dir, "intermediate_states.pkl"), "rb") as f:
            _state_cache[run_dir] = pickle.load(f)
    return _state_cache[run_dir]


def archetype(run_dir: str, game_key: str) -> str:
    """Frame-0 archetype per the proven triage rule (available_actions of the initial state)."""
    idx, _ = game_run(run_dir, game_key)
    states = load_states(run_dir)[idx]
    try:
        avail = states[0].raw.available_actions or []
        ids = set()
        for a in avail:
            s = str(getattr(a, "value", a))
            m = re.search(r"(\d+)", s)
            if m:
                ids.add(int(m.group(1)))
        has_move = bool(ids & _AVATAR_FAMILY)
        has_click = _CLICK_ID in ids
        if has_click and not has_move:
            return "CLICK"
        if has_move and not has_click:
            return "AVATAR"
        if has_move and has_click:
            return "MIXED"
    except Exception:
        pass
    return "CLICK"  # unreadable => safe default, per graft_triage.py


def _grid(state):
    import numpy as np
    return np.asarray(state.raw.frame[-1])


def evidence_digest(run_dir: str, game_key: str, upto_action: int, n: int = 20) -> dict:
    """Last-n evidence tuples ending at env action index `upto_action` (exclusive),
    plus recorded no-op events for the current level segment.

    Tuple: (action_id, coords_or_null, changed_px, level, level_delta)."""
    import numpy as np
    idx, gr = game_run(run_dir, game_key)
    states = load_states(run_dir)[idx]
    hist = gr["history"]
    upto = min(upto_action, len(hist), len(states) - 1)
    tuples = []
    for j in range(max(0, upto - n), upto):
        a = hist[j]["action"]
        coords = None
        if a.get("data") and "x" in a["data"]:
            coords = [a["data"]["x"], a["data"]["y"]]
        changed = int(np.sum(_grid(states[j]) != _grid(states[j + 1])))
        lc_before = states[j].raw.levels_completed
        lc_after = states[j + 1].raw.levels_completed
        tuples.append({
            "i": j,
            "action": a["id"],
            "coords": coords,
            "changed_px": changed,
            "level": lc_after + 1,
            "level_delta": lc_after - lc_before,
        })
    # no-op events over the whole current-level segment (not just the window)
    cur_level = states[upto].raw.levels_completed + 1 if upto < len(states) else None
    seg_start = 0
    for j in range(upto - 1, -1, -1):
        if states[j].raw.levels_completed + 1 != cur_level:
            seg_start = j + 1
            break
    noops = []
    for j in range(seg_start, upto):
        changed = int(np.sum(_grid(states[j]) != _grid(states[j + 1])))
        if changed == 0:
            a = hist[j]["action"]
            coords = None
            if a.get("data") and "x" in a["data"]:
                coords = [a["data"]["x"], a["data"]["y"]]
            noops.append({"i": j, "action": a["id"], "coords": coords})
    return {
        "upto_action": upto,
        "level": cur_level,
        "actions_this_level": upto - seg_start,
        "tuples": tuples,
        "noop_events_this_level": noops,
        "noop_count_this_level": len(noops),
    }


def digest_as_text(d: dict) -> str:
    lines = [
        f"Evidence digest (ground truth from the environment log; current level {d['level']}, "
        f"{d['actions_this_level']} actions taken on this level so far).",
        "Last observed transitions, oldest first — (action, coords[x,y], changed_px, level, level_delta):",
    ]
    for t in d["tuples"]:
        c = f"({t['coords'][0]},{t['coords'][1]})" if t["coords"] else "-"
        lines.append(
            f"  #{t['i']}: {t['action']} {c} changed_px={t['changed_px']} "
            f"level={t['level']} level_delta={t['level_delta']}"
        )
    lines.append(
        f"No-op events on this level (actions whose board did not change at all): "
        f"{d['noop_count_this_level']}"
    )
    for e in d["noop_events_this_level"][-10:]:
        c = f"({e['coords'][0]},{e['coords'][1]})" if e["coords"] else "-"
        lines.append(f"  #{e['i']}: {e['action']} {c}")
    return "\n".join(lines)
