"""Phase Q ls20 validation: does HistoryAugmentedExplorer crack ls20, and does its rot-tile counter
track the engine's hidden rotation (cklxociuu)? Offline deterministic engine. See the Phase Q spec.
"""
from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def run(pol, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("ls20"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    env = client.make(game_id=gid, scorecard_id="ls20crack")
    obs = env.reset()
    n, best = 0, 0
    rot_checks, rot_mismatch = 0, 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        # ORACLE: agent's total visit-count mod 4 should equal (engine cklxociuu - start) mod 4.
        if isinstance(pol, HistoryAugmentedExplorer) and pol._counts:
            agent_rot = sum(pol._counts.values()) % 4
            start_idx = env._game.dhksvilbb.index(
                env._game.current_level.get_data("StartRotation"))
            engine_rot_delta = (env._game.cklxociuu - start_idx) % 4
            rot_checks += 1
            if agent_rot != engine_rot_delta:
                rot_mismatch += 1
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        best = max(best, int(obs.levels_completed or 0))
        n += 1
    counts = getattr(pol, "_counts", {})
    return best, n, rot_checks, rot_mismatch, dict(counts)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    base = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    b_lv, b_n, _, _, _ = run(base, budget)
    print(f"banked (augment off): ls20 best_level={b_lv} actions={b_n}", flush=True)
    aug = HistoryAugmentedExplorer(seed=0, trust_threshold=3, border_mask=2, augment=True, counter_mod=4)
    a_lv, a_n, rc, mism, counts = run(aug, budget)
    print(f"history-augmented:     ls20 best_level={a_lv} actions={a_n}  "
          f"rot_oracle: {mism}/{rc} mismatches", flush=True)
    print(f"  visit counts (sig->count): {counts}", flush=True)
    cracked = a_lv > b_lv or (a_lv >= 1 and a_n < 2000)
    print("RESULT: " + ("ls20 CRACKED by augmentation" if cracked
                        else "no crack — investigate visit detection / exploration"), flush=True)


if __name__ == "__main__":
    main()
