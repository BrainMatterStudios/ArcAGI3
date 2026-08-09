#!/usr/bin/env python3
"""Re-score banked screen waves on the TRUE scored objective.

WHY THIS EXISTS
    Every A/B verdict this campaign recorded was adjudicated on an unweighted,
    ft09-excluded level count (struct_screen_config.py:158,239,309;
    patch_closure_config.py:96-101,216-230; package_screen_config.py:108-113).
    The scored objective is different in three ways that all matter:
      * per level: min(115, 100*(baseline/actions)**2)   <- efficiency, quadratic
      * weighted by (level_index + 1)                    <- depth dominates
      * capped by completion share                       <- unfinished levels cap it
    pc_driver.pc_env_score already computes exactly this per row and stores it,
    under a comment reading "informative only; the classifier reads levels".

VALIDATION
    This module's `env_score` is an independent reimplementation. `--validate`
    checks it against the driver's stored per-row `score` on every row of every
    wave and reports mismatches. It must be 0 before any number here is trusted.

USAGE
    python3 scripts/rescore_waves.py                # re-adjudication table
    python3 scripts/rescore_waves.py --validate     # implementation check only
    python3 scripts/rescore_waves.py --per-game ft09
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WAVE_DIR = REPO / "scratchpad/banked_waves_20260809"
ENV_DIR = REPO / "environment_files"

# label -> (file stem, arm)
WAVES = [
    ("closure_base_w1", "pc_base", "base"),
    ("closure_base_w2", "w2_base", "base"),
    ("closure_cand_w1", "pc_cand", "candidate"),
    ("closure_cand_w2", "w2_cand", "candidate"),
    ("package_w1", "pkg", "package"),
    ("struct_w1", "struct", "struct"),
    ("struct_w2", "struct2", "struct"),
    ("struct_w3", "struct3", "struct"),
]


def env_score(levels_completed, actions_per_level, n_levels, baselines):
    """The scored objective. Mirrors arc_agi/scorecard.py:168-206.

    Independent of pc_driver.pc_env_score by construction — see --validate.
    """
    if not n_levels or not baselines or len(baselines) != n_levels:
        return None
    total_w = 0
    total_score = 0.0
    max_w = 0
    for i in range(n_levels):
        w = i + 1
        total_w += w
        actions = actions_per_level[i] if i < len(actions_per_level) else 0
        completed = i < (levels_completed or 0)
        s = min(115.0, (baselines[i] / actions) ** 2 * 100.0) if (completed and actions > 0) else 0.0
        if s > 0:
            max_w += w
        total_score += s * w
    return round(min(total_score / total_w, max_w / total_w * 100.0), 4)


_BASELINES: dict[str, list[int] | None] = {}


def baselines_for(game: str):
    if game not in _BASELINES:
        found = None
        for meta in sorted(glob.glob(str(ENV_DIR / game / "*/metadata.json"))) + sorted(
            glob.glob(str(ENV_DIR / game / "*/*/metadata.json"))
        ):
            try:
                d = json.loads(Path(meta).read_text())
            except Exception:  # noqa: BLE001
                continue
            if d.get("game_id") and d.get("baseline_actions"):
                found = list(d["baseline_actions"])
        _BASELINES[game] = found
    return _BASELINES[game]


def load_wave(stem: str):
    return json.loads((WAVE_DIR / f"{stem}.json").read_text())


def score_wave(stem: str):
    """Per-game best-over-clones true score, plus level bookkeeping."""
    rows = load_wave(stem)["rows"]
    best: dict[str, float] = {}
    levels: dict[str, int] = {}
    depth: dict[int, int] = {}
    checked = mism = 0
    for r in rows:
        game = r["source_game"]
        base = baselines_for(game)
        if base is None:
            continue
        s = env_score(
            r.get("levels_completed"), r.get("actions_per_level") or [],
            r.get("levels_total"), base,
        )
        if s is None:
            continue
        checked += 1
        stored = r.get("score")
        if stored is not None and abs(s - stored) > 5e-4:
            mism += 1
        if s > best.get(game, -1.0):
            best[game] = s
        lv = int(r.get("levels_completed") or 0)
        if lv > levels.get(game, -1):
            levels[game] = lv
        for i in range(lv):
            depth[i + 1] = depth.get(i + 1, 0) + 1
    return best, levels, depth, checked, mism


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="only check the implementation")
    ap.add_argument("--per-game", metavar="GAME", help="show one game across all waves")
    args = ap.parse_args()

    data = {}
    tot_checked = tot_mism = 0
    for label, stem, arm in WAVES:
        best, levels, depth, checked, mism = score_wave(stem)
        data[label] = dict(arm=arm, best=best, levels=levels, depth=depth)
        tot_checked += checked
        tot_mism += mism

    print(f"IMPLEMENTATION CHECK: {tot_checked} rows vs the driver's stored score, "
          f"{tot_mism} mismatches > 5e-4")
    if tot_mism:
        print("!! implementation disagrees with the driver — every number below is suspect")
        return 1
    if args.validate:
        return 0

    if args.per_game:
        g = args.per_game
        print(f"\n=== {g} across waves ===")
        for label, _, arm in WAVES:
            d = data[label]
            print(f"  {label:16s} {arm:10s} score={d['best'].get(g, 0.0):8.3f} "
                  f"levels={d['levels'].get(g, 0)}")
        return 0

    print(f"\n{'wave':17s} {'arm':10s} {'raw lv':>7s} {'TRUE(25)':>9s} {'TRUE-exFT09':>12s} {'ft09':>8s}")
    print("-" * 70)
    for label, _, arm in WAVES:
        d = data[label]
        allv = list(d["best"].values())
        exft = [v for k, v in d["best"].items() if k != "ft09"]
        rawlv = sum(v for k, v in d["levels"].items() if k != "ft09")
        print(f"{label:17s} {arm:10s} {rawlv:7d} {st.mean(allv):9.4f} "
              f"{st.mean(exft):12.4f} {d['best'].get('ft09', 0.0):8.3f}")

    print("\n=== arm means (the re-adjudication) ===")
    groups: dict[str, list[str]] = {}
    for label, _, arm in WAVES:
        groups.setdefault(arm, []).append(label)
    means = {}
    for arm, labels in groups.items():
        allm = [st.mean(list(data[l]["best"].values())) for l in labels]
        exm = [st.mean([v for k, v in data[l]["best"].items() if k != "ft09"]) for l in labels]
        means[arm] = (allm, exm)
        print(f"  {arm:10s} n={len(labels)}  TRUE(25) {st.mean(allm):7.4f} "
              f"{[round(x, 3) for x in allm]}   exFT09 {st.mean(exm):7.4f}")

    print("\n=== contrasts vs base ===")
    if "base" in means:
        b_all, b_ex = means["base"]
        for arm in means:
            if arm == "base":
                continue
            a_all, a_ex = means[arm]
            d_all = st.mean(a_all) - st.mean(b_all)
            d_ex = st.mean(a_ex) - st.mean(b_ex)
            print(f"  {arm:10s} TRUE(25) {d_all:+7.4f} ({d_all/st.mean(b_all)*100:+6.1f}%)   "
                  f"exFT09 {d_ex:+7.4f} ({d_ex/st.mean(b_ex)*100:+6.1f}%)   "
                  f"n={len(a_all)} vs {len(b_all)}")
        print("\n  NOTE: these n are 1-3 per arm. Contrasts are DIRECTIONAL, not certified.")

    print("\n=== completed-level depth histogram (all clones, excl ft09 not applied) ===")
    for label, _, arm in WAVES:
        print(f"  {label:17s} {dict(sorted(data[label]['depth'].items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
