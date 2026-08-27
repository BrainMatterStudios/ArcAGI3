"""probe_budget — where the action budget actually goes, per LLM turn.

The step-2 instrument from docs/RESEARCH-2026-08-27-the-two-level-wall.md.

THE TARGET, restated: a 3+ score needs ~115 actions/game and a 5+ score needs
~178, against today's 45. Efficiency is already saturated (median 0.86x human
baseline), so the whole programme is buying actions.

There are exactly three ways a turn can fail to produce actions, and they have
different fixes. This instrument separates them, from recorded transcripts, with
no GPU:

  ZERO-ACTION TURNS   the model spent a full deliberation and executed nothing
                      (inspection, planning, a dead decode). Fix: fewer of them.
  SINGLE-ACTION TURNS one deliberation -> one action. Fix: batching.
  BATCHED TURNS       one deliberation -> k actions. Already what we want.

  actions/game = turns/game x actions/turn

so the reachable ceiling from batching alone is
  actions_per_game x (target_actions_per_turn / observed_actions_per_turn).

Anything the instrument cannot reach that way has to come from more turns
(cheaper deliberation) or from a non-LLM executor.

Usage:  python probe_budget.py [--json OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRANSCRIPT_GLOBS = [
    "scratchpad/ft09_ablation/*/taaf_harness_artifacts/transcripts/*.txt",
    "scratchpad/multirole_corpus/*/transcripts/*.txt",
    "taaf_harness_artifacts/transcripts/*.txt",
]

TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.M)
RESULT_RE = re.compile(r"^\[TOOL RESULT: python\]", re.M)
EXECUTED_RE = re.compile(r"^executed: (true|false)\s*$", re.M)
EXEC_COUNT_RE = re.compile(r"^executed_count: (\d+)\s*$", re.M)
REQ_COUNT_RE = re.compile(r"^requested_count: (\d+)\s*$", re.M)
ACTION_NUM_RE = re.compile(r"^action_num: (\d+)\s*$", re.M)
STOPPED_RE = re.compile(r"^stopped_early: (true|false)\s*$", re.M)
STOPREASON_RE = re.compile(r"^stop_reason: (\S+)\s*$", re.M)
LEVELUP_RE = re.compile(r"^level_completed: true\s*$", re.M)
THINK_RE = re.compile(r"^\[THINKING\]?$", re.M)
SECTION_RE = re.compile(r"^\[(THINKING|TOOL CALL|MODEL RESPONSE META|TOOL RESULT|"
                        r"USER PROMPT|SYSTEM PROMPT|ANALYZER STATUS)\]?", re.M)


def split_turns(text: str):
    marks = [(m.start(), int(m.group(1)), int(m.group(2))) for m in TURN_RE.finditer(text)]
    return [(s, a, text[p:(marks[i + 1][0] if i + 1 < len(marks) else len(text))])
            for i, (p, s, a) in enumerate(marks)]


def model_chars(turn: str) -> int:
    """Characters the MODEL produced this turn (THINKING + TOOL CALL only)."""
    keep, n = False, 0
    for line in turn.splitlines():
        m = SECTION_RE.match(line)
        if m:
            keep = m.group(1) in ("THINKING", "TOOL CALL")
            continue
        if keep:
            n += len(line) + 1
    return n


def scan(paths):
    games = []
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        turns = split_turns(text)
        if not turns:
            continue
        rec = {"file": os.path.relpath(path, ROOT), "turns": [], "levelups": 0}
        for step, act, turn in turns:
            body = turn[RESULT_RE.search(turn).start():] if RESULT_RE.search(turn) else ""
            ex = EXEC_COUNT_RE.search(body)
            executed = EXECUTED_RE.search(body)
            n_act = int(ex.group(1)) if ex else (
                1 if (executed and executed.group(1) == "true") else 0)
            req = REQ_COUNT_RE.search(body)
            stopped = STOPPED_RE.search(body)
            if LEVELUP_RE.search(turn):
                rec["levelups"] += 1
            rec["turns"].append({
                "step": step,
                "actions": n_act,
                "requested": int(req.group(1)) if req else None,
                "stopped_early": bool(stopped and stopped.group(1) == "true"),
                "stop_reason": (STOPREASON_RE.search(body).group(1)
                                if STOPREASON_RE.search(body) else None),
                "model_chars": model_chars(turn),
                "after_levelup": False,
            })
        for i, t in enumerate(rec["turns"][:-1]):
            if LEVELUP_RE.search(turns[i][2]):
                rec["turns"][i + 1]["after_levelup"] = True
        an = [int(m.group(1)) for m in ACTION_NUM_RE.finditer(text)]
        rec["actions_total"] = max(an) if an else sum(t["actions"] for t in rec["turns"])
        games.append(rec)
    return games


def report(games):
    turns = [t for g in games for t in g["turns"]]
    n = len(turns)
    if not n:
        print("no turns parsed")
        return 1
    acts = [t["actions"] for t in turns]
    total_actions = sum(acts)

    print(f"transcripts : {len(games)}")
    print(f"LLM turns   : {n}")
    print(f"actions     : {total_actions}")
    print(f"\nactions per game  : median {statistics.median(g['actions_total'] for g in games):.0f}")
    print(f"turns per game    : median {statistics.median(len(g['turns']) for g in games):.0f}")
    print(f"ACTIONS PER TURN  : mean {total_actions/n:.2f}   median {statistics.median(acts):.0f}")

    zero = sum(1 for a in acts if a == 0)
    one = sum(1 for a in acts if a == 1)
    many = sum(1 for a in acts if a >= 2)
    print("\n--- where the deliberations go ---")
    print(f"  ZERO-action turns   {zero:>4}/{n}  {100*zero/n:5.1f}%   (deliberation, no action)")
    print(f"  SINGLE-action turns {one:>4}/{n}  {100*one/n:5.1f}%   (the batching target)")
    print(f"  BATCHED turns (>=2) {many:>4}/{n}  {100*many/n:5.1f}%   (already what we want)")
    if many:
        b = [a for a in acts if a >= 2]
        print(f"     batch size when it happens: median {statistics.median(b):.0f}, max {max(b)}")

    zc = [t["model_chars"] for t in turns if t["actions"] == 0]
    ac = [t["model_chars"] for t in turns if t["actions"] > 0]
    if zc and ac:
        print(f"\n  model output on a ZERO-action turn : median {statistics.median(zc):.0f} chars")
        print(f"  model output on an ACTING turn     : median {statistics.median(ac):.0f} chars")
        share = sum(zc) / (sum(zc) + sum(ac))
        print(f"  => zero-action turns consume {100*share:.1f}% of all model output")

    post = [t for t in turns if t["after_levelup"]]
    if post:
        pa = [t["actions"] for t in post]
        print("\n--- the level boundary ---")
        print(f"  post-win turns              : {len(post)}")
        print(f"  actions on a post-win turn  : mean {statistics.mean(pa):.2f} vs {total_actions/n:.2f} overall")
        print(f"  model output on a post-win turn: median {statistics.median(t['model_chars'] for t in post):.0f}"
              f" vs {statistics.median(t['model_chars'] for t in turns):.0f} overall")
        print("  (the boundary costs RE-DERIVATION, not confusion — see probe_boundary.py)")

    stops = Counter(t["stop_reason"] for t in turns if t["stop_reason"])
    if stops:
        print(f"\n  batch stop reasons: {dict(stops)}")
    trunc = [t for t in turns if t["requested"] and t["actions"] < t["requested"]]
    if trunc:
        print(f"  batches truncated : {len(trunc)}/{sum(1 for t in turns if t['requested'])}")

    # --- what batching alone can reach ---
    apt = total_actions / n
    apg = statistics.median(g["actions_total"] for g in games)
    print("\n--- ceiling from batching alone (turns/game held fixed) ---")
    print(f"{'actions/turn':>13} {'actions/game':>13}   verdict")
    for target in (apt, 2.0, 3.0, 4.0, 6.0):
        proj = apg * target / apt
        tag = "  <- today" if abs(target - apt) < 1e-9 else (
            "  3+ bar cleared" if proj >= 115 and proj < 178 else
            "  5+ bar cleared" if proj >= 178 else "")
        print(f"{target:>13.2f} {proj:>13.0f}{tag}")
    print(f"\n  bars: ~115 actions/game = 3+ ; ~178 = 5+")
    need3 = apt * 115 / apg
    need5 = apt * 178 / apg
    print(f"  actions/turn required: {need3:.2f} for 3+ , {need5:.2f} for 5+")
    print(f"  (struct plan-channel already measured 2.01 actions/turn live)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    paths = sorted({p for g in TRANSCRIPT_GLOBS for p in glob.glob(os.path.join(ROOT, g))})
    games = scan(paths)
    rc = report(games)
    if args.json:
        json.dump(games, open(args.json, "w"), indent=1)
        print(f"\nper-turn detail -> {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
