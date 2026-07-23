"""prep_dataset.py — build the Kaggle SFT corpus from teacher win-trajectories.

Reads scratchpad/rl_gate/sft_data/all_wins.jsonl, normalizes each sample to the
Qwen3.5 chat-template shape (sft_common.normalize_sample), verifies it encodes
end-to-end with the REAL Qwen3.6 tokenizer+processor (tokenizer_bundle/,
downloaded from vrfai/Qwen3.6-27B-FP8 — same files the Kaggle snapshot carries),
splits train/val (val = 1 sample per (game, level)), and emits corpus/ ready for
`kaggle datasets create`. Templating is REPEATED in-kernel from the snapshot's
own files; this local pass proves it works and pins exact token statistics.

Usage: .venv/bin/python submission/_sft_k3/prep_dataset.py [--max-len 32768]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sft_common import normalize_sample, encode_with_mask  # noqa: E402

SRC = REPO / "scratchpad/rl_gate/sft_data/all_wins.jsonl"
EPISODES = REPO / "scratchpad/rl_gate/episodes"


def load_tools() -> list[dict]:
    """The python-tool schema, from the first trace record of each win episode.
    Must be identical everywhere (it renders into the system block)."""
    tools_seen = []
    for ep in sorted({json.loads(l)["meta"]["episode"] for l in SRC.open()}):
        with (EPISODES / ep / "trace.jsonl").open() as f:
            tools_seen.append(json.loads(f.readline())["request"]["tools"])
    assert all(t == tools_seen[0] for t in tools_seen), "tool schema differs across episodes"
    return tools_seen[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--out", type=Path, default=HERE / "corpus")
    args = ap.parse_args()

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(HERE / "tokenizer_bundle")
    tools = load_tools()

    rows, dropped = [], []
    for i, line in enumerate(SRC.open()):
        raw = json.loads(line)
        msgs, tgt = normalize_sample(raw)
        feats, info = encode_with_mask(proc, msgs, tgt, tools, args.max_len)
        if feats is None:
            dropped.append({"idx": i, "meta": raw["meta"], **info})
            continue
        rows.append({"messages": msgs, "target": tgt,
                     "meta": {**raw["meta"], "qwen_total": info["n_total"],
                              "qwen_target": info["n_target"],
                              "n_images": info["n_images"],
                              "dropped_turns": info["dropped_turns"]}})
        if (i + 1) % 20 == 0:
            print(f"  encoded {i + 1} samples...")

    # split: val = first sample of each (game, level) family
    val, train, seen = [], [], set()
    for r in rows:
        fam = (r["meta"]["game"], r["meta"]["level"])
        (val if fam not in seen else train).append(r)
        seen.add(fam)

    args.out.mkdir(exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        with (args.out / f"{name}.jsonl").open("w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
    (args.out / "tools.json").write_text(json.dumps(tools, indent=2))
    shutil.copy(HERE / "sft_common.py", args.out / "sft_common.py")
    if not (args.out / "tokenizer_bundle").exists():
        shutil.copytree(HERE / "tokenizer_bundle", args.out / "tokenizer_bundle")
    (args.out / "dataset-metadata.json").write_text(json.dumps({
        "title": "arc3-sft-k3-corpus", "id": "ahmedmobasher86/arc3-sft-k3-corpus",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2))

    tot = [r["meta"]["qwen_total"] for r in rows]
    tgt_toks = [r["meta"]["qwen_target"] for r in rows]
    per_fam = defaultdict(int)
    for r in rows:
        per_fam[f"{r['meta']['game']}-L{r['meta']['level']}"] += 1
    stats = {
        "samples": len(rows), "train": len(train), "val": len(val),
        "dropped_overlong": dropped, "max_len": args.max_len,
        "truncated_samples": sum(1 for r in rows if r["meta"]["dropped_turns"]),
        "qwen_total_tokens": sum(tot), "qwen_target_tokens": sum(tgt_toks),
        "len_min": min(tot), "len_mean": round(sum(tot) / len(tot)), "len_max": max(tot),
        "target_min": min(tgt_toks), "target_mean": round(sum(tgt_toks) / len(tgt_toks)),
        "target_max": max(tgt_toks),
        "images": sum(r["meta"]["n_images"] for r in rows),
        "families": dict(sorted(per_fam.items())),
    }
    (args.out / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "families"}, indent=2))
    print(f"corpus -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
