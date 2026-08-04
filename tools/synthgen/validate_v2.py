"""v2 acceptance gates: game/episode validation + corpus-level quality gates.

Game gates (per game, at generation time — gen_v2.py):
  1. v1 acceptance suite on the CLEAN solution (loadability through the real
     local_wrapper, solvability with exact level boundaries, determinism) —
     unchanged from validate.py.
  2. EPISODE replay: the scripted teacher episode (missteps included) replays
     through the real engine to a final WIN, with no GAME_OVER anywhere and
     level completions exactly on the turns the script asserts.

Corpus gates (on the built corpus — run by gen/corpus CLI, saved to gates.json):
  A. LENGTH DISTRIBUTION: target text tokens (meta.qwen_target_text, computed
     through the kernel's own sft_common.render path) must match corpus_v3's
     measured distribution within ±25% on mean, P10 and P90.
     Reference (measured 2026-08-04 from corpus_v3 meta.qwen_target, lengths
     only): mean 1487, P10 354, P90 2937.
  B. ANTI-TEMPLATE OVERLAP: mean pairwise 8-gram Jaccard over sampled
     CROSS-GAME target pairs <= 0.02 and P90 <= 0.05.
     Justification: corpus_v3's own cross-episode baseline measures 0.0000
     mean / 0.0000 P90 (400 sampled pairs) — genuinely deliberated text
     shares essentially no 8-grams. A small allowance (2%) is granted for
     programmatic phrasing banks; anything above it indicates template
     convergence of the kind that produced the v1 under-deliberation defect.
     (The v1 corpus measures ~0.5+ on the same metric — reported for contrast.)
  C. SPLIT INTEGRITY: held-out family absent from train/val; holdout file is
     pure; train/val split is by game (episode-disjoint).
  D. KERNEL ENCODE / CAP BEHAVIOR: every sampled record renders through
     sft_common.render with the target as a clean suffix; estimated full
     lengths (text tokens + ~85/image) are reported against the kernel caps
     (24576 default, 32768 restorable) with the number of front-truncation
     drops needed; no record may be untruncatable at either cap.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

REPO_ROOT = _HERE.parents[1]
SFT_K3_DIR = REPO_ROOT / "submission" / "_sft_k3"
CORPUS_V3 = SFT_K3_DIR / "corpus_v3"

REF_TARGET = {"mean": 1487, "p10": 354, "p90": 2937}
LENGTH_TOLERANCE = 0.25
OVERLAP_MEAN_MAX = 0.02
OVERLAP_P90_MAX = 0.05
IMAGE_TOKENS_EST = 85
KERNEL_CAPS = (24576, 32768)


def _pct(xs: list, p: float):
    return xs[min(len(xs) - 1, int(p * len(xs)))]


# ---------------------------------------------------------------------------
# episode replay gate (game-level)
# ---------------------------------------------------------------------------


def validate_episode(env_root: Path, spec: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    from validate import replay

    trace = [a for lvl in episode["levels"] for t in lvl for a in t["actions"]]
    recs = replay(env_root, spec["game_id"], trace)
    final = recs[-1]
    report = {
        "episode_len": len(trace),
        "episode_final_state": final["state"],
        "episode_win": final["state"] == "WIN" and final["score"] == len(spec["levels"]),
        "episode_no_game_over": all(r["state"] != "GAME_OVER" for r in recs),
    }
    i, score_seen, bounds_ok = 0, 0, True
    for lvl in episode["levels"]:
        for t in lvl:
            i += len(t["actions"])
            got = recs[i]["score"] > score_seen
            if got != t["assert"]["level_completed"]:
                bounds_ok = False
            if got:
                score_seen = recs[i]["score"]
    report["episode_bounds_ok"] = bounds_ok
    report["episode_ok"] = bool(
        report["episode_win"] and report["episode_no_game_over"] and bounds_ok)
    return report


# ---------------------------------------------------------------------------
# corpus gates
# ---------------------------------------------------------------------------


def _load_rows(out_dir: Path) -> dict[str, list[dict]]:
    return {
        name: [json.loads(l) for l in (out_dir / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "val", "holdout_transfer")
    }


def gate_length_distribution(rows: list[dict]) -> dict[str, Any]:
    targets = sorted(r["meta"]["qwen_target_text"] for r in rows
                     if r["meta"].get("qwen_target_text"))
    mean = sum(targets) / len(targets)
    p10, p90 = _pct(targets, 0.10), _pct(targets, 0.90)
    checks = {
        "mean": (mean, REF_TARGET["mean"]),
        "p10": (p10, REF_TARGET["p10"]),
        "p90": (p90, REF_TARGET["p90"]),
    }
    detail = {}
    ok = True
    for name, (got, ref) in checks.items():
        lo, hi = ref * (1 - LENGTH_TOLERANCE), ref * (1 + LENGTH_TOLERANCE)
        passed = lo <= got <= hi
        ok = ok and passed
        detail[name] = {"got": round(got, 1), "ref": ref,
                        "band": [round(lo, 1), round(hi, 1)], "ok": passed}
    return {"gate": "length_distribution", "ok": ok, "n": len(targets),
            "p50": _pct(targets, 0.5), "max": targets[-1], "checks": detail}


def _target_text(r: dict) -> str:
    t = r["target"]
    return (t.get("reasoning_content") or t.get("reasoning") or "") + "\n" + (t.get("content") or "")


def _ngrams(text: str, n: int = 8) -> set:
    toks = text.split()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def gate_overlap(rows: list[dict], n_pairs: int = 400, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(r["meta"]["episode"], []).append(r)
    games = list(by_game)
    cross, within = [], []
    for _ in range(n_pairs):
        g1, g2 = rng.sample(games, 2)
        a, b = rng.choice(by_game[g1]), rng.choice(by_game[g2])
        s1, s2 = _ngrams(_target_text(a)), _ngrams(_target_text(b))
        if s1 and s2:
            cross.append(len(s1 & s2) / len(s1 | s2))
    for g in rng.sample(games, min(40, len(games))):
        rs = by_game[g]
        for a, b in zip(rs, rs[1:]):
            s1, s2 = _ngrams(_target_text(a)), _ngrams(_target_text(b))
            if s1 and s2:
                within.append(len(s1 & s2) / len(s1 | s2))
    cross.sort()
    mean = statistics.mean(cross)
    p90 = _pct(cross, 0.90)
    ok = mean <= OVERLAP_MEAN_MAX and p90 <= OVERLAP_P90_MAX
    return {"gate": "anti_template_overlap", "ok": ok,
            "cross_game": {"n": len(cross), "mean": round(mean, 5),
                           "p90": round(p90, 5), "max": round(cross[-1], 5)},
            "within_game_adjacent": {
                "n": len(within),
                "mean": round(statistics.mean(within), 5) if within else None,
                "max": round(max(within), 5) if within else None},
            "thresholds": {"cross_mean_max": OVERLAP_MEAN_MAX,
                           "cross_p90_max": OVERLAP_P90_MAX,
                           "reference_corpus_v3_cross_mean": 0.0}}


def gate_split_integrity(out_dir: Path, holdout_family: str) -> dict[str, Any]:
    rows = _load_rows(out_dir)
    train_val = rows["train"] + rows["val"]
    leak = [r["meta"]["episode"] for r in train_val
            if r["meta"]["family"] == holdout_family]
    impure = [r["meta"]["episode"] for r in rows["holdout_transfer"]
              if r["meta"]["family"] != holdout_family]
    tg = {r["meta"]["episode"] for r in rows["train"]}
    vg = {r["meta"]["episode"] for r in rows["val"]}
    overlap = sorted(tg & vg)
    ok = not leak and not impure and not overlap and bool(rows["holdout_transfer"])
    return {"gate": "split_integrity", "ok": ok,
            "holdout_family": holdout_family,
            "holdout_leaked_into_train_val": sorted(set(leak)),
            "non_holdout_in_holdout_file": sorted(set(impure)),
            "train_val_game_overlap": overlap,
            "n_train_games": len(tg), "n_val_games": len(vg),
            "n_holdout_rows": len(rows["holdout_transfer"])}


def gate_kernel_encode(out_dir: Path, sample_n: int = 40, seed: int = 0) -> dict[str, Any]:
    """Render a sample through the EXACT kernel path (sft_common.render) and
    check the cap/truncation behavior for every record via token estimates."""
    if str(SFT_K3_DIR) not in sys.path:
        sys.path.insert(0, str(SFT_K3_DIR))
    from sft_common import normalize_sample, render, truncate_front
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CORPUS_V3 / "tokenizer_bundle"))
    tools = json.loads((CORPUS_V3 / "tools.json").read_text())
    rows = [r for split in _load_rows(out_dir).values() for r in split]
    rng = random.Random(seed)

    # exact render on a sample: suffix property + true token counts
    sample = rng.sample(rows, min(sample_n, len(rows)))
    suffix_ok = 0
    for r in sample:
        msgs, tgt = normalize_sample(r)
        full, target = render(tok, msgs, tgt, tools)
        if full.endswith(target) and len(target) > 0:
            suffix_ok += 1

    # cap behavior for EVERY record (text tokens exact from meta + image est)
    caps: dict[str, Any] = {}
    for cap in KERNEL_CAPS:
        fits = 0
        needs_trunc = 0
        untruncatable = 0
        drops: list[int] = []
        for r in rows:
            total_est = r["meta"]["qwen_total_text"] + IMAGE_TOKENS_EST * r["meta"]["n_images"]
            if total_est <= cap:
                fits += 1
                continue
            needs_trunc += 1
            # simulate the kernel's drop-oldest loop on message metadata
            msgs = list(r["messages"])
            est = total_est
            d = 0
            while est > cap:
                nxt = truncate_front(msgs)
                if nxt is None:
                    untruncatable += 1
                    d = -1
                    break
                dropped = msgs[2 : 2 + (len(msgs) - len(nxt))]
                for m in dropped:
                    est -= _rough_msg_tokens(m)
                msgs = nxt
                d += 1
            if d >= 0:
                drops.append(d)
        caps[str(cap)] = {
            "fits": fits, "needs_front_truncation": needs_trunc,
            "untruncatable": untruncatable,
            "max_drops": max(drops) if drops else 0,
        }
    ok = (suffix_ok == len(sample)
          and all(c["untruncatable"] == 0 for c in caps.values()))
    return {"gate": "kernel_encode", "ok": ok,
            "suffix_render_ok": f"{suffix_ok}/{len(sample)}",
            "caps": caps, "n_rows": len(rows)}


def _rough_msg_tokens(m: dict) -> float:
    t = 8.0
    c = m.get("content")
    if isinstance(c, str):
        t += len(c) / 3.0
    elif isinstance(c, list):
        for p in c:
            if p.get("type") == "text":
                t += len(p["text"]) / 3.0
            elif p.get("type") == "image_url":
                t += IMAGE_TOKENS_EST
    if m.get("reasoning_content"):
        t += len(m["reasoning_content"]) / 3.0
    for tc in m.get("tool_calls") or []:
        args = tc.get("function", {}).get("arguments")
        t += len(args["code"] if isinstance(args, dict) else str(args)) / 3.0
    return t


def run_corpus_gates(out_dir: Path, holdout_family: str, seed: int = 0) -> dict[str, Any]:
    rows_by = _load_rows(out_dir)
    all_rows = [r for split in rows_by.values() for r in split]
    gates = {
        "length_distribution": gate_length_distribution(all_rows),
        "anti_template_overlap": gate_overlap(all_rows, seed=seed),
        "split_integrity": gate_split_integrity(out_dir, holdout_family),
        "kernel_encode": gate_kernel_encode(out_dir, seed=seed),
    }
    gates["all_ok"] = all(g["ok"] for g in gates.values() if isinstance(g, dict))
    (out_dir / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    return gates


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=_HERE / "out" / "corpus_synth_v2")
    ap.add_argument("--holdout-family", default="mirror")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gates = run_corpus_gates(args.corpus, args.holdout_family, args.seed)
    print(json.dumps(gates, indent=2))
    return 0 if gates["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
