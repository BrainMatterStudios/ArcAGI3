"""Stage-0 Pillar 1 — offline engine + 25 public games load & step.

For every environment under environment_files/: make env, reset, take 30 random
actions, confirm frames are 64x64 int grids and the env transitions without
exceptions. Prints a per-game PASS/FAIL table plus toolchain pins.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from importlib import metadata

import numpy as np

logging.disable(logging.CRITICAL)

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState


def settled(obs):
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a


def main() -> int:
    pins = {}
    for pkg in ("arc-agi", "arcengine", "numpy", "scipy"):
        try:
            pins[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pins[pkg] = "NOT INSTALLED"
    print("toolchain pins:", pins)

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    envs = sorted(arcade.get_environments(), key=lambda e: e.game_id)
    print(f"environments discovered: {len(envs)}")

    results = {}
    t0 = time.time()
    for e in envs:
        stem = e.game_id.split("-")[0][:4]
        rng = random.Random(0)
        try:
            env = arcade.make(game_id=e.game_id, scorecard_id=f"stage0-{stem}")
            o = env.reset()
            f = settled(o)
            assert f.shape == (64, 64), f"bad frame shape {f.shape}"
            steps = 0
            for _ in range(30):
                if o.state == GameState.WIN:
                    break
                if o.state == GameState.GAME_OVER:
                    o = env.step(GameAction.RESET)
                    steps += 1
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
                f = settled(o)
                assert f.shape == (64, 64)
                steps += 1
            results[stem] = {"ok": True, "steps": steps,
                             "win_levels": int(o.win_levels or 0),
                             "avail": sorted(set(o.available_actions or []))}
        except Exception as ex:  # noqa: BLE001
            results[stem] = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}

    n_ok = sum(1 for r in results.values() if r["ok"])
    for stem in sorted(results):
        r = results[stem]
        if r["ok"]:
            print(f"  {stem}  PASS  steps={r['steps']:>2}  win_levels={r['win_levels']}  avail={r['avail']}")
        else:
            print(f"  {stem}  FAIL  {r['error']}")
    print(f"\nP1 VERDICT: {n_ok}/{len(envs)} games load+step  elapsed {time.time()-t0:.1f}s")
    json.dump({"pins": pins, "results": results},
              open("scratchpad/engineered_stage0/p1_engine_load.json", "w"), indent=1)
    return 0 if n_ok == len(envs) == 25 else 1


if __name__ == "__main__":
    sys.exit(main())
