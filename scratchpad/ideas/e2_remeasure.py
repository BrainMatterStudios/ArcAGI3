"""E2 — re-measure what was built on broken frame identity.

Three quantities were measured earlier today with an unmasked frame, which we now know
differs on every action in 6 games and on 13 of 25 overall:

  (a) STATE COLLAPSE. Distinct board states per action. If the HUD makes every frame
      unique, a graph search never merges nodes and degenerates to a tree. This is the
      leading candidate explanation for the graph explorer's failure, and it is a
      precondition for every graph/novelty/empowerment/cycle-elimination idea.

  (b) DETERMINISM VIOLATIONS. Same (board, action) -> two different successors would
      prove hidden state. The earlier figure (up to 0.596) is untrustworthy: it was
      unmasked AND contaminated by animation, since the engine returns up to 61 frame
      layers per action on g50t. Here the settled frame is masked, and violations are
      reported separately for single-layer responses so animation cannot inflate them.

  (c) PROVABLE NULL CYCLES. If two masked boards are identical, the actions between
      them are a null cycle and are deletable with no replay. Counting them bounds what
      E5 (geodesic replay) can recover.

No GPU, no model.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("ONLY_RESET_LEVELS", "true")
logging.disable(logging.CRITICAL)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hud_mask import frame_key, hud_mask, settled_frame  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

STEPS = int(os.environ.get("STEPS", 3000))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]


def run(game_id, arcade, seed, steps=STEPS):
    rng = random.Random(seed)
    env = arcade.make(game_id=game_id, scorecard_id=f"e2-{seed}-{game_id[:8]}")
    o = env.reset()
    prev_raw = settled_frame(o).copy()
    prev_key = frame_key(prev_raw, game_id)

    seen_raw, seen_key = set(), set()
    succ_all, succ_single = {}, {}
    viol_all = pairs_all = viol_single = pairs_single = 0
    cycles = 0
    first_seen = {prev_key: 0}
    n = 0

    for t in range(steps):
        if o.state == GameState.WIN:
            break
        if o.state == GameState.GAME_OVER:
            o = env.step(GameAction.RESET)
            prev_raw = settled_frame(o).copy()
            prev_key = frame_key(prev_raw, game_id)
            continue
        avail = [a for a in (o.available_actions or []) if a != 0]
        if not avail:
            break
        aid = rng.choice(avail)
        if aid == 6:
            y, x = rng.randrange(64), rng.randrange(64)
            o = env.step(GameAction.ACTION6, data={"x": x, "y": y})
            akey = ("C", y, x)
        else:
            o = env.step(GameAction.from_id(aid))
            akey = ("S", aid)

        arr = np.asarray(o.frame)
        n_layers = len(arr) if arr.ndim == 3 else 1
        cur_raw = settled_frame(o)
        cur_key = frame_key(cur_raw, game_id)
        n += 1

        seen_raw.add(cur_raw.tobytes())
        seen_key.add(cur_key)

        k = (prev_key, akey)
        if k in succ_all:
            pairs_all += 1
            if succ_all[k] != cur_key:
                viol_all += 1
        else:
            succ_all[k] = cur_key
        # Animation-free subset: only steps where the engine settled in one layer.
        if n_layers == 1:
            if k in succ_single:
                pairs_single += 1
                if succ_single[k] != cur_key:
                    viol_single += 1
            else:
                succ_single[k] = cur_key

        if cur_key in first_seen:
            cycles += 1
        else:
            first_seen[cur_key] = t

        prev_raw, prev_key = cur_raw.copy(), cur_key

    if n == 0:
        return None
    return {
        "n": n,
        "distinct_raw": len(seen_raw),
        "distinct_masked": len(seen_key),
        "collapse": len(seen_raw) / max(len(seen_key), 1),
        "states_per_action_raw": len(seen_raw) / n,
        "states_per_action_masked": len(seen_key) / n,
        "viol_all": (viol_all / pairs_all) if pairs_all else None,
        "pairs_all": pairs_all,
        "viol_single": (viol_single / pairs_single) if pairs_single else None,
        "pairs_single": pairs_single,
        "null_cycle_frac": cycles / n,
        "has_hud": bool(hud_mask(game_id).any()),
    }


def main():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    envs = sorted(arcade.get_environments(), key=lambda e: e.game_id)
    out = {}
    t0 = time.time()
    print(f"steps={STEPS} seeds={SEEDS}\n")
    print(f"{'game':6} {'HUD':>4} {'st/act raw':>11} {'st/act msk':>11} {'collapse':>9} "
          f"{'viol_all':>9} {'viol_1lyr':>10} {'nullCyc':>8}")
    for e in envs:
        stem = e.game_id.split("-")[0][:4]
        rs = [r for r in (run(e.game_id, arcade, s) for s in SEEDS) if r]
        if not rs:
            continue
        avg = lambda k: np.mean([r[k] for r in rs if r[k] is not None]) if any(
            r[k] is not None for r in rs) else None
        out[stem] = rs
        va, vs = avg("viol_all"), avg("viol_single")
        print(f"{stem:6} {('Y' if rs[0]['has_hud'] else '-'):>4} "
              f"{avg('states_per_action_raw'):>11.3f} {avg('states_per_action_masked'):>11.3f} "
              f"{avg('collapse'):>9.2f}x {('-' if va is None else f'{va:>9.3f}')} "
              f"{('-' if vs is None else f'{vs:>10.3f}')} {avg('null_cycle_frac'):>8.3f}", flush=True)
        json.dump(out, open("scratchpad/ideas/e2_remeasure.json", "w"), indent=1, default=str)

    rows = [r for rs in out.values() for r in rs]
    hud = [r for r in rows if r["has_hud"]]
    print(f"\ncorpus states/action: raw {np.mean([r['states_per_action_raw'] for r in rows]):.3f}"
          f"  masked {np.mean([r['states_per_action_masked'] for r in rows]):.3f}")
    if hud:
        print(f"HUD games only:       raw {np.mean([r['states_per_action_raw'] for r in hud]):.3f}"
              f"  masked {np.mean([r['states_per_action_masked'] for r in hud]):.3f}"
              f"  collapse {np.mean([r['collapse'] for r in hud]):.2f}x")
    print(f"mean provable null-cycle fraction: {np.mean([r['null_cycle_frac'] for r in rows]):.3f}")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
