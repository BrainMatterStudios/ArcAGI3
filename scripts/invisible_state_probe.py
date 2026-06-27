"""Phase P active probe: prove ls20's invisible rotation is real and outcome-gating.

Offline deterministic ls20 engine. We drive the known solution to the goal-adjacent cell, then for
each rotation index r in {0,1,2,3} INJECT the hidden rotation (`env._game.cklxociuu = r`, exactly the
variable the win predicate `bejndxqqzf` reads) and take the goal-entry action. The observable frame is
identical across r (rotation is invisible in the render — avatar pixels are byte-equal across
rotations), yet the outcome is WIN only when r matches GoalRotation (L1 idx 0) and BLOCKED otherwise.
That is direct proof the win depends on a state variable absent from the frame — and that it is
resolvable by a rotation counter. See docs/superpowers/specs/2026-06-21-invisible-state-prevalence-map-design.md
"""
from __future__ import annotations

import logging

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
ED = "environment_files"
SOLUTION = [3, 3, 3, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1]   # last action enters the goal (rotation idx 0 == GoalRotation)


def _won(obs):
    return int(obs.levels_completed or 0) >= 1 or obs.state == GameState.WIN


def _masked_key(grid):
    """The banked-agent masked key for a single frame (fresh VolatilityTracker => border/bg masking)."""
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    pol.vt.update(grid)
    pol.bg = P.detect_background(grid)
    return pol._key(grid)


def main():
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ED,
                    logger=logging.getLogger("probe"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))

    # 1) sanity: does the known solution actually win (validates solution + rotation model)?
    env = client.make(game_id=gid, scorecard_id="probe")
    obs = env.reset()
    for a in SOLUTION:
        obs = env.step(GameAction.from_id(a))
    print(f"full solution wins: {_won(obs)} (levels={obs.levels_completed}, state={obs.state})", flush=True)

    # 2) injection: reach goal-adjacent, inject each rotation, attempt entry.
    grids, keys, results = [], [], []
    for r in range(4):
        env = client.make(game_id=gid, scorecard_id="probe")
        obs = env.reset()
        for a in SOLUTION[:-1]:
            obs = env.step(GameAction.from_id(a))
        grid_adj = P.to_grid(obs.frame)
        grids.append(grid_adj)
        keys.append(_masked_key(grid_adj))
        env._game.cklxociuu = r                      # inject the invisible rotation
        obs2 = env.step(GameAction.from_id(SOLUTION[-1]))
        results.append((r, _won(obs2)))

    frames_equal = all(np.array_equal(grids[0], g) for g in grids)
    keys_equal = len(set(keys)) == 1
    win_rs = [r for r, w in results if w]
    print(f"goal-adjacent frame identical across injected rotations: {frames_equal}", flush=True)
    print(f"masked state-key identical across injected rotations: {keys_equal}", flush=True)
    print(f"outcomes by injected rotation idx: {results}", flush=True)
    ok = frames_equal and keys_equal and win_rs == [0]
    print("RESULT: " + ("INVISIBLE-STATE CONFIRMED — identical observable frame+key, outcome gated "
                        "by the hidden rotation (win only at idx 0); resolvable by a rotation counter."
                        if ok else
                        f"NOT confirmed (win_rs={win_rs}, frames_equal={frames_equal}, keys_equal={keys_equal})"),
          flush=True)


if __name__ == "__main__":
    main()
