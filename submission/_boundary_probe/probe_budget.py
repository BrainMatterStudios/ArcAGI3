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
ACTION_NUM_RE = re.compile(r"(?:^action_num: |'action_num': )(\d+)", re.M)

# Two result schemas exist in the corpus and they must BOTH parse or any
# cross-arm comparison is an artifact:
#   BLOCK  (stock harness) multi-line YAML-ish:      executed_count: 5
#   INLINE (patch-21 plan channel) one-line dict:   Result: {... 'executed_count': 5 ...}
# A single turn can contain SEVERAL result blocks (an inspection call and then
# an action call); every one of them counts.
_PairsBlock = {
    "executed": re.compile(r"^executed: (true|false)\s*$", re.M),
    "executed_count": re.compile(r"^executed_count: (\d+)\s*$", re.M),
    "requested_count": re.compile(r"^requested_count: (\d+)\s*$", re.M),
    "stopped_early": re.compile(r"^stopped_early: (true|false)\s*$", re.M),
    "stop_reason": re.compile(r"^stop_reason: (\S+)\s*$", re.M),
    "level_completed": re.compile(r"^level_completed: true\s*$", re.M),
}
_PairsInline = {
    "executed": re.compile(r"'executed': (True|False)"),
    "executed_count": re.compile(r"'executed_count': (\d+)"),
    "requested_count": re.compile(r"'requested_count': (\d+)"),
    "stopped_early": re.compile(r"'stopped_early': (True|False)"),
    "stop_reason": re.compile(r"'stop_reason': '([^']+)'"),
    "level_completed": re.compile(r"'level_completed': True"),
}
# executed_actions is the most trustworthy count when present (both schemas).
EXEC_ACTIONS_BLOCK = re.compile(r"^executed_actions:\n((?:  - .*\n)+)", re.M)
EXEC_ACTIONS_INLINE = re.compile(r"'executed_actions': \[([^\]]*)\]")
LEVELUP_RE = re.compile(r"^level_completed: true\s*$|'level_completed': True", re.M)


def result_blocks(turn: str) -> list[str]:
    """Every [TOOL RESULT: python] payload in this turn, in order."""
    starts = [m.start() for m in RESULT_RE.finditer(turn)]
    if not starts:
        return []
    bounds = starts + [len(turn)]
    return [turn[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def parse_result(block: str) -> dict:
    """Schema-agnostic read of one result payload."""
    inline = "Result: {" in block
    pats = _PairsInline if inline else _PairsBlock
    out: dict = {}
    for key, pat in pats.items():
        m = pat.search(block)
        if not m:
            continue
        if key == "level_completed":
            out[key] = True
        elif key == "stop_reason":
            out[key] = m.group(1)
        elif key in ("executed", "stopped_early"):
            out[key] = m.group(1).lower() == "true"
        else:
            out[key] = int(m.group(1))
    m = (EXEC_ACTIONS_INLINE if inline else EXEC_ACTIONS_BLOCK).search(block)
    if m:
        body = m.group(1)
        if inline:
            # entries look like 'MOUSE(row=36, col=36)' — the commas INSIDE the
            # parens make a naive split double-count, so count quoted items.
            n = len(re.findall(r"'[^']*'", body))
        else:
            n = len([ln for ln in body.splitlines() if ln.strip().startswith("- ")])
        out["executed_actions_n"] = n
    return out


def actions_in(block: str) -> int:
    """Actions this result payload actually executed. Most trustworthy first."""
    r = parse_result(block)
    if "executed_actions_n" in r:
        return r["executed_actions_n"]
    if "executed_count" in r:
        return r["executed_count"]
    if r.get("executed"):
        return 1
    return 0
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
        rec = {"file": os.path.relpath(path, ROOT), "turns": [], "levelups": 0,
               "schema": "inline" if "Result: {" in text else "block"}
        # ACTION ACCOUNTING — validated 16/16 against the wave result JSONs.
        # The turn header carries the harness's own cumulative action counter
        # ("--- analysis_step=N | action=M |"), so actions taken DURING a turn
        # are the delta to the next header. Schema-independent, and it survives
        # transcript truncation; counting `executed_count` inside result blocks
        # does NOT — on the struct arm that agreed with ground truth in only
        # 1/8 transcripts, because results go missing when history trims.
        heads = [a for _, a, _ in turns]
        for i, (step, act, turn) in enumerate(turns):
            blocks = result_blocks(turn)
            parsed = [parse_result(b) for b in blocks]
            n_act = (heads[i + 1] - heads[i]) if i + 1 < len(turns) else None
            if LEVELUP_RE.search(turn):
                rec["levelups"] += 1
            rec["turns"].append({
                "step": step,
                "actions": n_act,          # None on the final turn (no delta)
                "requested": sum(p.get("requested_count", 0) for p in parsed) or None,
                "stopped_early": any(p.get("stopped_early") for p in parsed),
                "stop_reason": next((p["stop_reason"] for p in parsed
                                     if p.get("stop_reason")), None),
                "model_chars": model_chars(turn),
                "after_levelup": False,
            })
        for i, t in enumerate(rec["turns"][:-1]):
            if LEVELUP_RE.search(turns[i][2]):
                rec["turns"][i + 1]["after_levelup"] = True
        rec["actions_total"] = (max(heads) - 1) if heads else 0
        games.append(rec)
    return games


def report(games):
    turns = [t for g in games for t in g["turns"]]
    n = len(turns)
    if not n:
        print("no turns parsed")
        return 1
    turns = [t for t in turns if t["actions"] is not None]
    n = len(turns)
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

    post = [t for t in turns if t["after_levelup"] and t["actions"] is not None]
    if post:
        pa = [t["actions"] for t in post if t["actions"] is not None]
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
    # Relative headroom only. These transcripts are smokes and ablations with
    # their own box geometry, so their absolute actions/game does NOT compare
    # to the live 45; only the RATIOS transfer.
    zshare = sum(zc) / (sum(zc) + sum(ac)) if (zc and ac) else 0.0
    print("\n--- relative headroom (ratios only — do not read absolutes here) ---")
    print(f"  zero-action turns hold {100*zshare:.1f}% of the model's output.")
    print(f"  recovering ALL of it is a x{1/(1-zshare):.2f} ceiling on actions.")
    print(f"  live bars need x{115/45:.2f} (3+) and x{178/45:.2f} (5+) on 45 actions/game,")
    print("  so removing zero-action turns is NECESSARY BUT NOT SUFFICIENT — the")
    print("  rest has to come from actions per acting turn, or a non-LLM executor.")
    print("\n  !! ACTIONS ARE NOT LEVELS. The one controlled test of that conversion")
    print("     (patch 21, ft09 28v28) bought 1.40x actions/game and cleared 25%")
    print("     FEWER levels. Never read this instrument without level counts.")
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
