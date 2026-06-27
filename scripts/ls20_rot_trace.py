"""Phase Q: trace the histaug agent's OWN exploration. At every step where the engine rotation
(cklxociuu) flips, dump what the detector saw -- whether the rot-glyph cluster was a confirmed-static
present cluster, whether a count fired, and the avatar sprite position -- so we can see WHY the agent's
rot counter does/doesn't track the engine during real exploration (vs the scripted solution).
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)

# rot-glyph cluster signature (avatar absent), from ls20_rot_diag.py
GLYPH_SIG = ((31, 21, 0), (32, 20, 1), (32, 21, 0), (32, 22, 0), (33, 21, 1))


def main():
    budget = 1500
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("trace"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    env = client.make(game_id=gid, scorecard_id="trace")
    pol = HistoryAugmentedExplorer(seed=0, trust_threshold=3, border_mask=2, augment=True, counter_mod=4)
    obs = env.reset()
    g = env._game
    n = 0
    sum_checks = sum_mis = 0
    glyph_checks = glyph_mis = 0
    global_occ: dict = {}     # sig -> total occlusion events across all lives (survives _reset_history)
    prev_counts: dict = {}
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        levels = int(obs.levels_completed or 0)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED, levels,
                         list(obs.available_actions or []))
        # tally global occlusion events per sig (detect count increments before any reset clears them)
        for sig, c in pol._counts.items():
            inc = c - prev_counts.get(sig, 0)
            if inc > 0:
                global_occ[sig] = global_occ.get(sig, 0) + inc
        prev_counts = dict(pol._counts)
        # engine ground truth rotation delta this step
        start_idx = g.dhksvilbb.index(g.current_level.get_data("StartRotation"))
        engine_delta = (g.cklxociuu - start_idx) % 4
        # oracle A: sum of all counters (current ls20_crack oracle)
        if pol._counts:
            sum_checks += 1
            if sum(pol._counts.values()) % 4 != engine_delta:
                sum_mis += 1
        # oracle B: rot-glyph counter ALONE
        glyph_checks += 1
        if pol._counts.get(GLYPH_SIG, 0) % 4 != engine_delta:
            glyph_mis += 1
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
    print(f"oracle A (sum of all counters):  {sum_mis}/{sum_checks} mismatches")
    print(f"oracle B (rot-glyph counter):    {glyph_mis}/{glyph_checks} mismatches")
    print(f"\nglobal occlusion events per sig (total across all lives), most-occluded first:")
    for sig, tot in sorted(global_occ.items(), key=lambda kv: -kv[1]):
        colors = sorted({c for *_, c in sig})
        is_glyph = " <-- ROT GLYPH" if sig == GLYPH_SIG else ""
        r0 = min(r for r, c, col in sig); c0 = min(c for r, c, col in sig)
        print(f"  occluded x{tot:>4}  size={len(sig):>2} colors={colors} topleft=({r0},{c0}){is_glyph}")


if __name__ == "__main__":
    main()
