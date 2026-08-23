#!/usr/bin/env python3
"""Stage-1 labeling: terminal wrong-model spirals, from an objective rule.

TERMINAL SPIRAL (pre-registered, transcript-observable only):
  a maximal run of >= MIN_TURNS consecutive LLM-call blocks in which
    (1) the carried world model is verbatim-identical (whitespace-normalized), and
    (2) the level does not change, and
    (3) that level is NEVER completed later in the transcript
        (the session dies inside the stuck level).
This is the signature cited in the design doc (sb26: one wrong model held
verbatim 15 turns) made mechanical. Non-terminal stale runs (level later
completed) are recorded too — they are the 'recoverable' population that the
<=35% false-fire bar protects.

Usage:
  .venv/bin/python label_spirals.py [--min-turns 10] [--all]  # --all lists every run >= 6
Writes labels_terminal_spirals.json next to this script.
"""
from __future__ import annotations

import argparse
import json
import os

import corpus_lib as cl


def stale_runs(t: cl.Transcript, min_turns: int):
    """Maximal same-wm same-level runs of >= min_turns blocks (wm present)."""
    runs = []
    i = 0
    blocks = t.blocks
    while i < len(blocks):
        j = i
        while (
            j + 1 < len(blocks)
            and blocks[j + 1].wm_hash != "-"
            and blocks[j + 1].wm_hash == blocks[i].wm_hash
            and blocks[j + 1].level == blocks[i].level
        ):
            j += 1
        length = j - i + 1
        if blocks[i].wm_hash != "-" and length >= min_turns:
            runs.append((i, j))
        i = j + 1
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-turns", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="list every stale run >= 6 turns with terminality")
    args = ap.parse_args()

    labels = []
    listing_min = 6 if args.all else args.min_turns
    for t in cl.list_corpus():
        for (i, j) in stale_runs(t, listing_min):
            b0, b1 = t.blocks[i], t.blocks[j]
            level = b0.level
            terminal = not t.level_eventually_completed(level)
            row = {
                "sid": t.sid,
                "game": t.game_key.split("-")[0],
                "archetype": cl.archetype(t.run_dir, t.game_key),
                "level": level,
                "onset_block": i,
                "end_block": j,
                "turns": j - i + 1,
                "onset_action": b0.cum_actions,
                "end_action": b1.cum_actions,
                "terminal": terminal,
                "wm_hash": b0.wm_hash,
            }
            if args.all:
                print(f"{'TERM' if terminal else 'recov'} {t.sid:30s} L{level} "
                      f"turns={row['turns']:3d} actions {row['onset_action']}..{row['end_action']}")
            if terminal and row["turns"] >= args.min_turns:
                labels.append(row)

    labels.sort(key=lambda r: (-r["turns"]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_terminal_spirals.json")
    with open(out, "w") as f:
        json.dump({"rule": f"verbatim-stale wm run >= {args.min_turns} blocks, same level, "
                           "level never completed later in transcript",
                   "labels": labels}, f, indent=1)
    print(f"\nTERMINAL SPIRALS (min_turns={args.min_turns}): {len(labels)}")
    for r in labels:
        print(f"  {r['sid']:30s} L{r['level']} {r['archetype']:6s} turns={r['turns']:3d} "
              f"actions {r['onset_action']}..{r['end_action']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
