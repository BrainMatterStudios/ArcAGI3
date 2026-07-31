"""prep_dataset_v3.py — SFT corpus v3: episode-isolated split, single teacher, MEASURED stats.

Corrects three things found auditing v2 on 2026-07-31:

1. **The split is by EPISODE, not (game, level) family.** v2 grouped by family and
   claimed "0% of validation episodes overlap with training". That claim was false:
   an episode spans several levels, so v2 sent an episode's L1 turns to train and its
   L2 turns to val. Measured: 7 of 9 val episodes also appeared in training, covering
   52.2% of val rows. Splitting by episode makes episode overlap 0 by construction.

2. **Leakage is measured, never asserted.** v2 hardcoded
   `"val_leakage_percentage": 0.0  # Enforced 0%`, a literal that could not detect its
   own violation. v3 computes episode / family / game overlap from the emitted splits
   and records all three, including the ones that are non-zero.

3. **One teacher.** v2 silently mixed in 22 `moonshotai/kimi-k2.7-code` samples (the
   entire 435 -> 457 delta) alongside kimi-k3. K3 is the validated teacher; k2.7-code
   is not validated in this role. v3 keeps K3 only by default (--teachers to override).

NOTE ON `game` OVERLAP: episode-isolation does NOT make val games unseen — the same
game appears in train under a different episode. That is disclosed in prep_stats, not
hidden. The val NLL remains FIT evidence per A1-PROTOCOL §3 and must never be a
selection signal; the behavioural instrument is the pinned 13-game arc-interactive
holdout, which is disjoint from every game here.

Flail-turn pruning is carried over from v2 unchanged, and is near-inert by
measurement: it removed 1 turn across 457 samples. Kept for continuity, not credited.

Usage: python3 submission/_sft_k3/prep_dataset_v3.py [--max-len 32768]
"""
from __future__ import annotations

import os
import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sft_common import normalize_sample, encode_with_mask  # noqa: E402

SRC = REPO / "scratchpad/rl_gate/sft_data/all_wins.jsonl"
EPISODES = REPO / "scratchpad/rl_gate/episodes"


def load_tools() -> list[dict]:
    """The python-tool schema from win episodes."""
    tools_seen = []
    for ep in sorted({json.loads(l)["meta"]["episode"] for l in SRC.open()}):
        trace_file = EPISODES / ep / "trace.jsonl"
        if trace_file.exists():
            with trace_file.open() as f:
                tools_seen.append(json.loads(f.readline())["request"]["tools"])
    if not tools_seen:
        # Fallback tool definition if local traces missing
        return [{
            "type": "function",
            "function": {
                "name": "python",
                "description": "Execute Python code against the game environment",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]
                }
            }
        }]
    assert all(t == tools_seen[0] for t in tools_seen), "tool schema differs across episodes"
    return tools_seen[0]


def is_flail_turn(msg: dict) -> bool:
    """Detect repetitive/unproductive turns (e.g. repeated invalid moves)."""
    content = str(msg.get("content", ""))
    reasoning = str(msg.get("reasoning", "")) + str(msg.get("reasoning_content", ""))
    # Detect common flail markers or empty actions
    if "action([])" in content or "no-op" in content.lower():
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "corpus_v3")
    ap.add_argument("--teachers", default="moonshotai/kimi-k3",
                    help="comma-separated teachers to keep; '*' keeps all")
    args = ap.parse_args()
    keep_teachers = None if args.teachers.strip() == "*" else set(
        t.strip() for t in args.teachers.split(",") if t.strip())

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(HERE / "tokenizer_bundle")
    tools = load_tools()

    rows, dropped, excluded_teacher = [], [], defaultdict(int)
    for i, line in enumerate(SRC.open()):
        raw = json.loads(line)

        # Single-teacher purity: drop off-teacher samples before any encoding work.
        teacher = raw.get("meta", {}).get("teacher")
        if keep_teachers is not None and teacher not in keep_teachers:
            excluded_teacher[teacher] += 1
            continue

        msgs, tgt = normalize_sample(raw)
        
        # Filter flail turns from history
        filtered_msgs = [m for m in msgs if not is_flail_turn(m)]
        
        feats, info = encode_with_mask(proc, filtered_msgs, tgt, tools, args.max_len)
        if feats is None:
            dropped.append({"idx": i, "meta": raw["meta"], **info})
            continue
            
        rows.append({
            "messages": filtered_msgs,
            "target": tgt,
            "meta": {
                **raw["meta"],
                "qwen_total": info["n_total"],
                "qwen_target": info["n_target"],
                "n_images": info["n_images"],
                "dropped_turns": info["dropped_turns"]
            }
        })

    # Group by EPISODE. An episode spans multiple levels, so a (game, level) split
    # leaks the same trajectory across train and val — the defect this file fixes.
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["meta"]["episode"]].append(r)

    all_eps = sorted(episodes.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_eps)

    n_val_eps = max(1, int(len(all_eps) * args.val_ratio))
    val_eps = set(all_eps[:n_val_eps])

    train, val = [], []
    for ep, ep_rows in episodes.items():
        (val if ep in val_eps else train).extend(ep_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with (args.out / f"{name}.jsonl").open("w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    (args.out / "tools.json").write_text(json.dumps(tools, indent=2))
    shutil.copy(HERE / "sft_common.py", args.out / "sft_common.py")
    if not (args.out / "tokenizer_bundle").exists():
        shutil.copytree(HERE / "tokenizer_bundle", args.out / "tokenizer_bundle")

    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "arc3-sft-k3-corpus-v3",
        "id": "ahmedmobasher86/arc3-sft-k3-corpus-v3",
        "licenses": [{"name": "CC0-1.0"}]
    }, indent=2))

    # Leakage is MEASURED from the emitted splits, never asserted. v2 hardcoded a 0.0
    # literal that could not detect its own violation — and did not.
    def overlap(field_fn):
        T = {field_fn(r) for r in train}
        V = {field_fn(r) for r in val}
        shared = T & V
        rows_hit = sum(1 for r in val if field_fn(r) in shared)
        return {"train": len(T), "val": len(V), "overlap": len(shared),
                "val_rows_affected_pct": round(100 * rows_hit / len(val), 1) if val else 0.0}

    leakage = {
        "by_episode": overlap(lambda r: r["meta"]["episode"]),
        "by_family_game_level": overlap(lambda r: (r["meta"]["game"], r["meta"]["level"])),
        "by_game": overlap(lambda r: r["meta"]["game"]),
    }
    assert leakage["by_episode"]["overlap"] == 0, (
        f"episode split failed: {leakage['by_episode']}")

    tot = [r["meta"]["qwen_total"] for r in rows]
    tgt_toks = [r["meta"]["qwen_target"] for r in rows]
    stats = {
        "total_samples": len(rows),
        "train_samples": len(train),
        "val_samples": len(val),
        "split_unit": "episode",
        "teachers_kept": sorted(keep_teachers) if keep_teachers else "all",
        "samples_excluded_by_teacher": dict(excluded_teacher),
        # by_episode.overlap is 0 by construction; by_game is NOT zero and is disclosed
        # so nobody mistakes this val set for a game-generalization instrument.
        "leakage_measured": leakage,
        "qwen_total_tokens": sum(tot),
        "qwen_target_tokens": sum(tgt_toks),
        "avg_total_tokens": round(sum(tot) / len(tot)) if tot else 0,
        "avg_target_tokens": round(sum(tgt_toks) / len(tgt_toks)) if tgt_toks else 0,
    }
    (args.out / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"corpus v3 (episode-isolated, single-teacher) -> {args.out}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
