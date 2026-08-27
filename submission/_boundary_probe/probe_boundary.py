"""probe_boundary — does the winframe substitution actually confuse OUR agent?

The step-1 falsifier from docs/RESEARCH-2026-08-27-the-two-level-wall.md.

WHAT IS BEING TESTED
--------------------
On a level-completing action the engine appends one extra render: the freshly
loaded NEXT level's opening board. `GameState.frame` returns `raw.frame[-1]`,
so at the instant the agent is scored it is shown the next level and told it is
current. The board it won on is demoted into `animation_frames` and reaches the
agent only as `animation_frame_count` / `animation_changed_cell_count`.

That is verified at the engine (docs §4). This probe asks the separate,
decisive question: **does it bind for us?** It reads the agent's own words in
the turn immediately after each level-up and classifies what it does.

PRE-REGISTERED READING (docs §7.1)
----------------------------------
  KILL   < 30% of post-win turns show boundary confusion
         => the perception fix is not our binding constraint; only the
            actions-per-game lane survives.
  PASS   >= 30% => winframe/carryover is a real lever for us and earns a place
            in the bundle.

Classification is keyword-based and deliberately conservative: every category
requires the agent to say something checkable, and the raw evidence line is
kept in the output so every count can be audited by hand.

CATEGORIES (a turn may carry more than one)
  attributes_to_action  treats the new board as the RESULT OF ITS OWN ACTION
                        -- the false-law generator. The strongest signal.
  lost_the_win          says it cannot see / does not have the winning board,
                        or reconstructs it from memory.
  restarts              explicitly re-explores / re-grounds from scratch.
  carries_a_rule        states a mechanic it intends to reuse (the healthy case).
  notes_the_boundary    correctly identifies that a new level has started.

Usage:  python probe_boundary.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSCRIPT_GLOBS = [
    "scratchpad/ft09_ablation/*/taaf_harness_artifacts/transcripts/*.txt",
    "scratchpad/multirole_corpus/*/transcripts/*.txt",
    "taaf_harness_artifacts/transcripts/*.txt",
]

TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.M)
LEVELUP_RE = re.compile(r"^level_completed: true\s*$", re.M)
ANIM_COUNT_RE = re.compile(r"^animation_frame_count: (\d+)\s*$", re.M)
ANIM_CELLS_RE = re.compile(r"^animation_changed_cell_count: (\d+)\s*$", re.M)
LEVEL_RE = re.compile(r"^level: (\d+)\s*$", re.M)

# --- classification patterns ------------------------------------------------
# Each is (category, compiled pattern). Patterns are matched against the
# agent's THINKING + TOOL CALL text of the post-win turn only.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("attributes_to_action", re.compile(
        r"(my (last |previous )?(click|action|move)s? (has |have )?(just )?"
        r"(caused|created|produced|changed|moved|reset|rearranged|scrambled)"
        r"|that (click|action|move) (caused|created|changed|reset|rearranged)"
        r"|the board (was |has been )?(completely )?(changed|reset|rearranged|scrambled) by"
        r"|after my (click|action|move) the (whole )?board)", re.I)),
    ("lost_the_win", re.compile(
        r"(don'?t have (it|the (winning|previous|final) board)( any ?more)?"
        r"|no longer (have|see) the (winning|previous|solved|final) board"
        r"|can'?t see (the )?(winning|previous|solved|final) (board|frame|state)"
        r"|from memory"
        r"|i (did|do) not (get to )?(see|observe) (the )?(winning|final|solved) (board|frame|configuration)"
        r"|the (winning|solved) (board|configuration|state) is (gone|not available|lost))", re.I)),
    ("restarts", re.compile(
        r"(start(ing)? (over|from scratch|afresh)"
        r"|re-?explore|re-?ground|re-?discover|from scratch"
        r"|need to (re-?)?(learn|figure out|establish) (the )?(mechanic|rules?|goal) again"
        r"|probe (the board |each action )?again to (see|learn|find))", re.I)),
    ("carries_a_rule", re.compile(
        r"(same (mechanic|rule|pattern|logic) as (the )?(previous|last|level)"
        r"|as (in|on) level \d"
        r"|carr(y|ies|ied) over"
        r"|reuse (the )?(same )?(mechanic|rule|strategy|approach)"
        r"|the mechanic (from|learned (on|in)) level \d"
        r"|worked (on|in|for) level \d)", re.I)),
    ("notes_the_boundary", re.compile(
        r"(level \d+ (has )?(started|begun|is now|begins)"
        r"|now (on|in|at) level \d"
        r"|new level|next level|advanced to level|moved to level"
        r"|level_completed[^\n]{0,40}true)", re.I)),
]

SECTION_RE = re.compile(r"^\[(THINKING|TOOL CALL|MODEL RESPONSE META|TOOL RESULT|"
                        r"USER PROMPT|SYSTEM PROMPT|ANALYZER STATUS)\]?", re.M)


def split_turns(text: str) -> list[tuple[int, int, str]]:
    """-> [(analysis_step, action_num, turn_text)] in file order."""
    marks = [(m.start(), int(m.group(1)), int(m.group(2))) for m in TURN_RE.finditer(text)]
    out = []
    for i, (pos, step, act) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((step, act, text[pos:end]))
    return out


def agent_voice(turn: str) -> str:
    """Only what the MODEL wrote this turn: THINKING + TOOL CALL.

    The system prompt is re-emitted every turn and would otherwise dominate
    every keyword count, and the TOOL RESULT is the harness speaking.
    """
    chunks, keep, buf = [], False, []
    for line in turn.splitlines():
        m = SECTION_RE.match(line)
        if m:
            if keep:
                chunks.append("\n".join(buf))
                buf = []
            keep = m.group(1) in ("THINKING", "TOOL CALL")
            continue
        if keep:
            buf.append(line)
    if keep and buf:
        chunks.append("\n".join(buf))
    return "\n".join(chunks)


def evidence(pat: re.Pattern, text: str) -> str:
    m = pat.search(text)
    if not m:
        return ""
    a, b = max(0, m.start() - 70), min(len(text), m.end() + 70)
    return " ".join(text[a:b].split())


def scan(paths: list[str]) -> dict:
    events, files_with_events = [], 0
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        turns = split_turns(text)
        if not turns:
            continue
        hits = 0
        for i, (step, act, turn) in enumerate(turns):
            if not LEVELUP_RE.search(turn):
                continue
            hits += 1
            anim = ANIM_COUNT_RE.search(turn)
            cells = ANIM_CELLS_RE.search(turn)
            lvl = LEVEL_RE.search(turn)
            # the turn AFTER the level-up is where the agent reacts
            nxt = turns[i + 1][2] if i + 1 < len(turns) else ""
            voice = agent_voice(nxt)
            cats, ev = [], {}
            for name, pat in PATTERNS:
                e = evidence(pat, voice)
                if e:
                    cats.append(name)
                    ev[name] = e
            events.append({
                "file": os.path.relpath(path, ROOT),
                "at_step": step, "at_action": act,
                "level_reached": int(lvl.group(1)) if lvl else None,
                "animation_frame_count": int(anim.group(1)) if anim else None,
                "animation_changed_cells": int(cells.group(1)) if cells else None,
                "post_win_turn_exists": bool(nxt),
                "post_win_voice_chars": len(voice),
                "categories": cats, "evidence": ev,
            })
        if hits:
            files_with_events += 1
    return {"events": events, "files_with_events": files_with_events}


def report(res: dict) -> int:
    ev = res["events"]
    n = len(ev)
    print(f"level-up events found : {n}  (across {res['files_with_events']} transcripts)")
    if not n:
        print("NO EVENTS — nothing to read. Check the transcript globs.")
        return 1

    # --- the mechanism, as the harness itself reported it live ---
    with_anim = [e for e in ev if (e["animation_frame_count"] or 0) >= 1]
    cells = sorted(e["animation_changed_cells"] for e in ev if e["animation_changed_cells"])
    print("\n--- the substitution, as reported to the agent at the level-up ---")
    print(f"  level-ups carrying >=1 hidden 'animation' frame : {len(with_anim)}/{n}")
    if cells:
        med = cells[len(cells) // 2]
        print(f"  cells that frame differed by (median)          : {med}")
        print(f"  range                                          : {cells[0]} .. {cells[-1]}")
    print("  the agent receives these as COUNTS; the board itself is never shown.")

    readable = [e for e in ev if e["post_win_turn_exists"] and e["post_win_voice_chars"] > 0]
    print(f"\npost-win turns with readable agent text: {len(readable)}/{n}")
    if not readable:
        print("VERDICT: UNREADABLE — no post-win agent text; instrument cannot decide.")
        return 2

    counts = Counter(c for e in readable for c in e["categories"])
    print("\n--- what the agent does in the turn after it wins ---")
    for name, _ in PATTERNS:
        k = counts.get(name, 0)
        print(f"  {name:22s} {k:>3}/{len(readable)}  {100*k/len(readable):5.1f}%")

    confused = [e for e in readable
                if {"attributes_to_action", "lost_the_win", "restarts"} & set(e["categories"])]
    rate = len(confused) / len(readable)
    print(f"\nBOUNDARY CONFUSION (attributes_to_action | lost_the_win | restarts)")
    print(f"  {len(confused)}/{len(readable)} = {100*rate:.1f}%   bar = 30%")
    verdict = "PASS" if rate >= 0.30 else "KILL"
    print(f"\nPRE-REGISTERED VERDICT: {verdict}")
    if verdict == "KILL":
        print("  => the perception fix is NOT our binding constraint.")
        print("     Only the actions-per-game lane survives step 1.")
    else:
        print("  => winframe/carryover binds for us; it earns its place in the bundle.")

    if confused:
        print("\n--- sample evidence (audit these by hand) ---")
        for e in confused[:6]:
            cat = next(c for c in e["categories"]
                       if c in ("attributes_to_action", "lost_the_win", "restarts"))
            print(f"\n  {e['file']} step {e['at_step']} -> level {e['level_reached']}  [{cat}]")
            print(f"    \"{e['evidence'][cat][:240]}\"")
    return 0


def main() -> int:
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full event list here")
    args = ap.parse_args()
    paths = sorted({p for g in TRANSCRIPT_GLOBS for p in glob.glob(os.path.join(ROOT, g))})
    print(f"transcripts scanned: {len(paths)}\n")
    res = scan(paths)
    rc = report(res)
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)
        print(f"\nfull event list -> {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
