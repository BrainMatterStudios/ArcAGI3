"""Phase Q: the REAL crack metric -- actions-to-first-L1-completion for banked vs history-augmented,
across seeds. Augmentation should let the explorer treat (position, rotation) as distinct states and
solve the rotation-gated level FASTER than banked's brute-force stumbling. Reports per-seed and mean.
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


def time_to_level(make_pol, seed, budget, target=1):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("speed"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    env = client.make(game_id=gid, scorecard_id=f"speed{seed}")
    pol = make_pol(seed)
    obs = env.reset()
    n = 0
    while n < budget:
        if obs.state == GameState.WIN or int(obs.levels_completed or 0) >= target:
            return n
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, obs.state == GameState.GAME_OVER, obs.state == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
    return None  # did not reach target within budget


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
    target = 1

    def banked(seed):
        return SalienceExplorer(seed=seed, trust_threshold=3, border_mask=2)

    def histaug(seed):
        # v5: semantic gate-identification + EAGER retry. Only blocked moves into a goal-like object
        # are recorded, so tier-0 retry fires at the gate without re-bumping walls.
        return HistoryAugmentedExplorer(seed=seed, trust_threshold=3, border_mask=2,
                                        augment=True, counter_mod=4, gate_only=True, retry_tier=0)

    print(f"actions-to-L{target} (budget {budget}), per seed:")
    print(f"{'seed':>4} {'banked':>10} {'histaug':>10}")
    b_all, h_all = [], []
    for s in seeds:
        b = time_to_level(banked, s, budget, target)
        h = time_to_level(histaug, s, budget, target)
        b_all.append(b); h_all.append(h)
        print(f"{s:>4} {str(b):>10} {str(h):>10}")
    bs = [x for x in b_all if x is not None]
    hs = [x for x in h_all if x is not None]
    print(f"\nbanked : solved {len(bs)}/{len(seeds)}  mean_actions={round(sum(bs)/len(bs)) if bs else 'NA'}")
    print(f"histaug: solved {len(hs)}/{len(seeds)}  mean_actions={round(sum(hs)/len(hs)) if hs else 'NA'}")


if __name__ == "__main__":
    main()
