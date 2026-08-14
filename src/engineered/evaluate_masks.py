"""Stage 1a milestone (a): learned-mask ≡ hand-mask agreement on the 25 games.

Protocol (pre-registered in the Stage-1a task):
  1. Per game, run the ≤16-action ProbeBattery -> learned Perception.
  2. On a FRESH env, replay the Stage-0 p2 protocol — 1200 random actions,
     seed 0 — and measure the no-op rate three ways on the SAME trajectory:
     raw, hand-masked (scratchpad/ideas/hud_mask.py, loaded by path), and
     learned-masked.
  3. Agreement on a game = |learned_rate - reference| <= 0.05, where the
     reference is the in-tree MEASURED_NOOP masked rate when the game has
     one, else the same-trajectory hand-masked rate (the in-tree table only
     covers 16 of 25 games).
  4. Milestone (a) passes at >= 20/25 agreements.

Standalone: .venv/bin/python src/engineered/evaluate_masks.py
Writes scratchpad/engineered_stage1/milestone_a.json next to the printed table.
"""
from __future__ import annotations

import importlib.util
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engineered.battery import BatteryConfig, ProbeBattery
from engineered.envs import open_arcade, repo_root, resolve_game_ids
from engineered.perception import settled_frame

STEPS = 1200
SEED = 0
TOLERANCE = 0.05
MILESTONE_MIN_AGREE = 20


def load_hand_reference() -> Any:
    """The frozen hand-mask asset, by explicit path (not an import — the
    scratchpad is not a package and src/engineered must not depend on it)."""
    path = repo_root() / "scratchpad" / "ideas" / "hud_mask.py"
    spec = importlib.util.spec_from_file_location("hand_hud_mask", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class GameResult:
    stem: str
    avail: list[int]
    probe_actions: int
    learned_lines: list[list[str | int]]
    hand_lines: str
    steps: int
    raw_rate: float
    hand_rate: float
    learned_rate: float
    reference: float
    ref_source: str  # "MEASURED_NOOP" | "hand@same-trajectory"
    delta: float
    agree: bool


def measure_game(arcade: Any, gid: str, stem: str, hand: Any) -> GameResult:
    from arcengine import GameAction, GameState

    # 1. learn the mask with the battery
    battery = ProbeBattery(BatteryConfig())
    env = arcade.make(game_id=gid, scorecard_id=f"eng1a-bat-{stem}")
    profile, state = battery.run(env)
    perception = profile.perception()

    # 2. p2 protocol on a fresh env, three rates on one trajectory
    rng = random.Random(SEED)
    env = arcade.make(game_id=gid, scorecard_id=f"eng1a-noop-{stem}")
    o = env.reset()
    prev = settled_frame(o).copy()
    avail_at_start = sorted(a for a in (o.available_actions or []) if a)
    raw_eq = hand_eq = learned_eq = n = 0
    while n < STEPS:
        if o.state == GameState.WIN:
            break
        if o.state == GameState.GAME_OVER:
            o = env.step(GameAction.RESET)
            prev = settled_frame(o).copy()
            continue
        avail = [a for a in (o.available_actions or []) if a]
        if not avail:
            break
        aid = rng.choice(avail)
        if aid == 6:
            o = env.step(GameAction.ACTION6,
                         data={"x": rng.randrange(64), "y": rng.randrange(64)})
        else:
            o = env.step(GameAction.from_id(aid))
        cur = settled_frame(o)
        raw_eq += int(np.array_equal(cur, prev))
        hand_eq += int(hand.frames_equal(cur, prev, gid))
        learned_eq += int(perception.frames_equal(cur, prev))
        prev = cur.copy()
        n += 1

    raw_rate = raw_eq / max(n, 1)
    hand_rate = hand_eq / max(n, 1)
    learned_rate = learned_eq / max(n, 1)
    measured = hand.MEASURED_NOOP.get(stem)
    if measured is not None:
        reference, ref_source = float(measured[1]), "MEASURED_NOOP"
    else:
        reference, ref_source = hand_rate, "hand@same-trajectory"
    delta = learned_rate - reference

    hand_spec = hand.HUD_REGIONS.get(stem)
    return GameResult(
        stem=stem,
        avail=avail_at_start,
        probe_actions=profile.actions_spent,
        learned_lines=[list(l) for l in profile.hud_lines],
        hand_lines=str(hand_spec) if hand_spec else "-",
        steps=n,
        raw_rate=round(raw_rate, 4),
        hand_rate=round(hand_rate, 4),
        learned_rate=round(learned_rate, 4),
        reference=round(reference, 4),
        ref_source=ref_source,
        delta=round(delta, 4),
        agree=abs(delta) <= TOLERANCE,
    )


def main() -> int:
    hand = load_hand_reference()
    arcade = open_arcade()
    gid_of = resolve_game_ids(arcade)
    results: list[GameResult] = []
    t0 = time.time()
    hdr = (f"{'game':5} {'probe':>5} {'learned_lines':28} {'hand':16} "
           f"{'raw':>6} {'hand':>6} {'learn':>6} {'ref':>6} {'Δ':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for stem in sorted(gid_of):
        r = measure_game(arcade, gid_of[stem], stem, hand)
        results.append(r)
        lines = ",".join(f"{k}{i}" for k, i in r.learned_lines) or "-"
        print(f"{r.stem:5} {r.probe_actions:>5} {lines:28} {r.hand_lines:16} "
              f"{r.raw_rate:>6.3f} {r.hand_rate:>6.3f} {r.learned_rate:>6.3f} "
              f"{r.reference:>6.3f} {r.delta:>+7.3f}  "
              f"{'AGREE' if r.agree else 'DISAGREE'} ({r.ref_source})")

    n_agree = sum(r.agree for r in results)
    passed = n_agree >= MILESTONE_MIN_AGREE
    print("-" * len(hdr))
    print(f"MILESTONE (a): {n_agree}/{len(results)} games agree "
          f"(tolerance ±{TOLERANCE}) -> {'PASS' if passed else 'FAIL'} "
          f"(need ≥{MILESTONE_MIN_AGREE})  elapsed {time.time() - t0:.0f}s")

    out_dir = repo_root() / "scratchpad" / "engineered_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {"steps": STEPS, "seed": SEED, "tolerance": TOLERANCE,
                     "milestone_min_agree": MILESTONE_MIN_AGREE},
        "n_agree": n_agree,
        "passed": passed,
        "games": [asdict(r) for r in results],
    }
    (out_dir / "milestone_a.json").write_text(json.dumps(payload, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
