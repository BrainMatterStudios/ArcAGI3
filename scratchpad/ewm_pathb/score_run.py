#!/usr/bin/env python3
"""Depth-weighted RHAE scorer for a PATH-B run's session dir.

Mirrors arc_agi/scorecard.py exactly:
  per-level score = min(115, (baseline/actions)^2 * 100), 0 if not completed
  game score = sum(score_i * weight_i)/sum(weight_i), weight_i = 1-indexed level number,
               capped at (sum weights of completed levels)/(sum all weights) * 100
Uses MAX-over-attempts efficiency: for each level, the FEWEST real actions any single
attempt used to complete it (what a clean execution play would score under max-over-plays).

Usage: score_run.py <workspace_session_dir> <game_alias e.g. tu93>
"""
import json
import sys
from pathlib import Path

REPO = Path("/Users/ahmed/Documents/ArcAGI3")


def baseline_actions(game: str) -> list[int]:
    md = next((REPO / "environment_files" / game).glob("*/metadata.json"))
    return json.loads(md.read_text())["baseline_actions"]


def level_min_actions(session_dir: Path) -> dict[int, int]:
    """For each level, the fewest within-attempt actions that COMPLETED it.

    An attempt on level L completes it iff some step's metadata has levels_completed >= L.
    The action count = that step's within-attempt index (steps taken in that attempt).
    """
    best: dict[int, int] = {}
    for attempt_dir in sorted(session_dir.glob("level_*_attempt_*")):
        # level index from dir name level_XX_attempt_YY
        lvl = int(attempt_dir.name.split("_")[1])
        steps = sorted(attempt_dir.glob("step_*_metadata.json"))
        for i, sp in enumerate(steps, start=1):
            md = json.loads(sp.read_text())
            if int(md.get("levels_completed", 0)) >= lvl:
                # this attempt completed level `lvl` at within-attempt action i
                if lvl not in best or i < best[lvl]:
                    best[lvl] = i
                break
    return best


def score_game(session_dir: Path, game: str) -> dict:
    base = baseline_actions(game)
    n_levels = len(base)
    completed = level_min_actions(session_dir)
    total_score = 0.0
    total_w = 0
    max_w = 0
    per_level = []
    for idx in range(1, n_levels + 1):
        w = idx
        total_w += w
        if idx in completed:
            acts = completed[idx]
            s = min(115.0, (base[idx - 1] / acts) ** 2 * 100.0)
            max_w += w
        else:
            acts = None
            s = 0.0
        total_score += s * w
        per_level.append({"level": idx, "baseline": base[idx - 1], "actions": acts, "score": round(s, 2)})
    game_score = 0.0
    if total_w:
        game_score = min(total_score / total_w, max_w / total_w * 100.0)
    return {
        "game": game,
        "n_levels": n_levels,
        "levels_solved": len(completed),
        "game_score": round(game_score, 3),
        "per_level": per_level,
    }


if __name__ == "__main__":
    sd = Path(sys.argv[1])
    game = sys.argv[2]
    res = score_game(sd, game)
    print(json.dumps(res, indent=2))
