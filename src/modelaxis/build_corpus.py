#!/usr/bin/env python3
"""build_corpus.py — M0 of the model axis: a STaR / rejection-sampling SFT corpus
built from OUR OWN wave transcripts.

Gate and rationale: docs/research-2026-09-09/PREREG-model-axis.md

SELECTION RULE (the rejection-sampling step, fixed in the pre-registration):
keep a model call ONLY if it was made while the run was on a level it went on to
CLEAR. The wall level -- the one the run died on -- is the failure and is dropped.
By the measured action split that keeps roughly 37.5 % of calls.

RECORD SHAPE: (system + user prompt) -> (reasoning + tool call), i.e. exactly what
the model emits live. The emitted python lives in the `raw_tool_calls` block inside
[MODEL RESPONSE META]; there is no separate [TOOL CALL] section in these transcripts.

HELD-OUT SPLIT: every read in M3/M4 is on games the model never saw. The split is by
GAME, never by call, and is written into the manifest so it cannot drift.

Usage:
  .venv/bin/python src/modelaxis/build_corpus.py --out scratchpad/modelaxis/corpus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "offkaggle/results"

# A turn header looks like:  --- analysis_step=3 | action=6 | 20:13:56 | tool-agent ---
TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.M)
SECTION_RE = re.compile(r"^\[([A-Z0-9 _-]+)\]$", re.M)


def split_calls(text: str) -> list[dict]:
    """Split a transcript into per-model-call blocks of {section: body}.

    One [MODEL RESPONSE META] marks the end of one model call. Sections before it
    (SYSTEM PROMPT / USER PROMPT / THINKING / ASSISTANT) belong to that call.
    """
    marks = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
    calls, cur = [], {}
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[pos:end].split("\n", 1)[1] if "\n" in text[pos:end] else ""
        # a turn header can sit inside the body; strip it
        body = TURN_RE.sub("", body).strip("\n")
        if name == "MODEL RESPONSE META":
            cur["META"] = body
            calls.append(cur)
            # the system prompt is re-emitted per turn, so carry it forward
            cur = {k: v for k, v in cur.items() if k == "SYSTEM PROMPT"}
        else:
            cur[name] = body
    return calls


def tool_calls_from_meta(meta: str) -> list[dict]:
    """Pull the raw_tool_calls JSON array out of a META block."""
    i = meta.find("raw_tool_calls:")
    if i < 0:
        return []
    frag = meta[i + len("raw_tool_calls:"):].strip()
    if not frag.startswith("["):
        return []
    depth, end = 0, None
    for j, ch in enumerate(frag):
        depth += (ch == "[") - (ch == "]")
        if depth == 0:
            end = j + 1
            break
    if end is None:
        return []
    try:
        return json.loads(frag[:end])
    except json.JSONDecodeError:
        return []


def level_of_turn(events: list[dict], analysis_step: int) -> int | None:
    """Level the run was on at a given analysis step, from the events log."""
    lvl = None
    for e in events:
        if e.get("analysis_step") is not None and e["analysis_step"] <= analysis_step:
            if e.get("level") is not None:
                lvl = e["level"]
    return lvl


def build(out_dir: Path, holdout_n: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_game_records: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()

    for tdir in sorted(RESULTS.glob("*/transcripts")) + sorted(RESULTS.glob("*/*/transcripts")):
        run_dir = tdir.parent
        res_path = run_dir / "results.json"
        if not res_path.exists():
            continue
        try:
            res = json.loads(res_path.read_text())
        except json.JSONDecodeError:
            continue
        if res.get("dry_run"):
            continue
        by_stem = {g["run_stem"]: g for g in res.get("games", [])
                   if isinstance(g, dict) and "run_stem" in g}

        for tf in sorted(tdir.glob("*.txt")):
            stem = tf.stem
            game = by_stem.get(stem)
            if not game:
                continue
            cleared = game.get("levels_completed") or 0
            stats["runs_seen"] += 1
            if cleared < 1:
                stats["runs_dropped_zero_levels"] += 1
                continue

            ev_path = run_dir / "artifacts" / f"{stem}_events.jsonl"
            events = []
            if ev_path.exists():
                for line in ev_path.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            text = tf.read_text(errors="replace")
            steps = [int(m.group(1)) for m in TURN_RE.finditer(text)]
            calls = split_calls(text)
            gid = stem.split("-")[0]

            for idx, call in enumerate(calls):
                stats["calls_seen"] += 1
                sysmsg = call.get("SYSTEM PROMPT")
                user = call.get("USER PROMPT")
                meta = call.get("META", "")
                tcs = tool_calls_from_meta(meta)
                # A [USER PROMPT] is emitted once per TURN, so calls 2..n of a
                # multi-call turn have no user section of their own. Their true
                # prompt includes the preceding tool RESULT, which these
                # transcripts do not log, so they cannot be reconstructed
                # faithfully and are excluded. The corpus is therefore
                # first-call-of-turn records only -- stated, not hidden.
                if not (sysmsg and user and tcs):
                    stats["calls_dropped_mid_turn_continuation"] += 1
                    continue
                step = steps[idx] if idx < len(steps) else None
                lvl = level_of_turn(events, step) if step is not None else None
                # SELECTION RULE: keep only calls made on a level the run CLEARED.
                # `level` in events is 1-based and is the level the run is ON.
                if lvl is None or lvl > cleared:
                    stats["calls_dropped_wall_level"] += 1
                    continue
                code = ""
                try:
                    args = tcs[0]["function"]["arguments"]
                    code = json.loads(args).get("code", "")
                except (KeyError, IndexError, json.JSONDecodeError, TypeError):
                    pass
                if not code.strip():
                    stats["calls_dropped_no_code"] += 1
                    continue
                per_game_records[gid].append({
                    "game": gid, "run_stem": stem, "analysis_step": step, "level": lvl,
                    "levels_cleared_by_run": cleared,
                    "messages": [{"role": "system", "content": sysmsg},
                                 {"role": "user", "content": user}],
                    "reasoning": call.get("THINKING", ""),
                    "content": call.get("ASSISTANT", ""),
                    "tool_call": {"name": "python", "arguments": {"code": code}},
                })
                stats["calls_kept"] += 1

    # dedup on (user prompt, code) -- identical situations recur across waves
    seen, deduped = set(), defaultdict(list)
    for gid, recs in per_game_records.items():
        for r in recs:
            key = hashlib.sha256(
                (r["messages"][1]["content"] + "\x00" + r["tool_call"]["arguments"]["code"]).encode()
            ).hexdigest()
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(key)
            deduped[gid].append(r)

    games = sorted(deduped)
    rng = random.Random(seed)
    holdout = sorted(rng.sample(games, min(holdout_n, len(games))))
    train = [g for g in games if g not in holdout]

    for name, gids in (("train", train), ("holdout", holdout)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w") as fh:
            for gid in gids:
                for r in deduped[gid]:
                    fh.write(json.dumps(r) + "\n")

    manifest = {
        "built_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_rule": "keep a call only if made on a level the run went on to CLEAR",
        "split_seed": seed,
        "holdout_games": holdout,
        "train_games": train,
        "records": {"train": sum(len(deduped[g]) for g in train),
                    "holdout": sum(len(deduped[g]) for g in holdout)},
        "per_game_records": {g: len(deduped[g]) for g in games},
        "stats": dict(stats),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratchpad/modelaxis/corpus")
    ap.add_argument("--holdout", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260909)
    a = ap.parse_args()
    m = build(REPO / a.out if not Path(a.out).is_absolute() else Path(a.out), a.holdout, a.seed)
    print(json.dumps({k: v for k, v in m.items() if k != "per_game_records"}, indent=2))
    print("\nper-game records:")
    for g, n in sorted(m["per_game_records"].items(), key=lambda x: -x[1]):
        tag = "HELDOUT" if g in m["holdout_games"] else ""
        print(f"  {g:>6} {n:>5}  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
