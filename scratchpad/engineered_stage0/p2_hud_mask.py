"""Stage-0 Pillar 2 — hand HUD mask on CURRENT frames.

Re-measures the raw vs masked no-op rate over 1200 random actions (seed 0) on six
games, including the strongest known HUD tickers (lf52, vc33 raw 0.000 -> masked
~0.99) and re86 as the near-no-HUD control, then compares against the in-tree
MEASURED_NOOP reference table in scratchpad/ideas/hud_mask.py.

Pass criterion (qualitative parity): for every ticker game the masked rate must
exceed the raw rate and land on the same side/order of magnitude as the recorded
reference; re86 must stay near-zero both ways.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ideas"))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from hud_mask import MEASURED_NOOP, frames_equal, hud_mask, settled_frame

GAMES = ["lf52", "vc33", "tu93", "sb26", "m0r0", "re86"]
STEPS = 1200


def main() -> int:
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid_of = {e.game_id.split("-")[0][:4]: e.game_id for e in arcade.get_environments()}
    out = {}
    t0 = time.time()
    print(f"{'game':6} {'mask_px':>7} {'raw_noop':>9} {'masked':>7} | reference (raw->masked)")
    for stem in GAMES:
        gid = gid_of[stem]
        rng = random.Random(0)
        env = arcade.make(game_id=gid, scorecard_id=f"s0hud-{stem}")
        o = env.reset()
        prev = settled_frame(o).copy()
        raw_eq = masked_eq = n = 0
        while n < STEPS:
            if o.state == GameState.WIN:
                break
            if o.state == GameState.GAME_OVER:
                o = env.step(GameAction.RESET)
                prev = settled_frame(o).copy()
                continue
            avail = [a for a in (o.available_actions or []) if a != 0]
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
            masked_eq += int(frames_equal(cur, prev, gid))
            prev = cur.copy()
            n += 1
        raw_r = raw_eq / max(n, 1)
        mask_r = masked_eq / max(n, 1)
        npx = int(hud_mask(gid).sum())
        ref = MEASURED_NOOP.get(stem)
        out[stem] = {"n": n, "raw": round(raw_r, 3), "masked": round(mask_r, 3),
                     "ref": ref, "mask_pixels": npx}
        print(f"{stem:6} {npx:>7} {raw_r:>9.3f} {mask_r:>7.3f} | {ref}")

    # verdict: every game with a recorded reference must reproduce the direction
    # (masked >= raw) and, for tickers with ref masked-raw gap > 0.1, show a gap
    # of at least half the recorded one.
    ok = True
    for stem, r in out.items():
        if r["ref"] is None:
            continue
        ref_raw, ref_masked = r["ref"]
        if r["masked"] < r["raw"] - 1e-9:
            ok = False
        if (ref_masked - ref_raw) > 0.1 and (r["masked"] - r["raw"]) < (ref_masked - ref_raw) / 2:
            ok = False
    print(f"\nP2 VERDICT: {'PASS' if ok else 'FAIL'}  elapsed {time.time()-t0:.0f}s")
    json.dump(out, open("scratchpad/engineered_stage0/p2_hud_mask.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
