"""META-SOLVER: the archetype library as a regression-safe router. For any game, try each archetype
RECOGNIZER; run the matching solver; if none matches (or none wins), ABSTAIN. Running across all dev games
validates (a) it cracks the archetype games (wa30 grab-drag, sb26 pattern-match) and (b) it does NOT falsely
"win" or misbehave on the other 23 (safety = no regression when wired as an additive portfolio play).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 python scripts/research_2026_07_01/meta_solver.py
"""
import logging
import numpy as np
from arc_agi import Arcade, OperationMode
from arcagi3 import perception as P
from ghp import GHP
import general_grabdrag as GD
import pattern_match as PM
logging.basicConfig(level=logging.ERROR)

ALL = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
       "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]


def recognize_patternmatch(grid):
    ak, tiles, slots = PM.perceive(grid)
    return len(ak) >= 2 and len(tiles) >= len(ak) and len(slots) >= len(ak)


def recognize_grabdrag(env, obs):
    """needs a probe to find the avatar + movable blocks + a color-coded goal candidate."""
    ghp = GHP(); ghp.learn(env, obs)
    grid = P.to_grid(obs.frame)
    av, bl, goals = GD.detect_roles(grid, ghp)
    if av is None or bl is None or not goals:
        return False
    avatar, blocks, _ = GD.perceive_roles(grid, av, bl, goals[0])
    return avatar is not None and len(blocks) >= 1


def route(game):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"meta-{game}")
    obs = env.reset()
    grid = P.to_grid(obs.frame)
    # try pattern-match first (cheap, static perception)
    if recognize_patternmatch(grid):
        lv = PM.solve(game, verbose=False)
        if lv > 0:
            return ("pattern-match", lv)
    # then grab-drag (needs a probe)
    try:
        if recognize_grabdrag(env, obs):
            lv = GD.solve(game, verbose=False)
            if lv > 0:
                return ("grab-drag", lv)
    except Exception:
        pass
    return ("abstain", 0)


def main():
    print("META-SOLVER across all 25 dev games. Expect: crack wa30+sb26, ABSTAIN (safe) on the rest.\n")
    cracked = []
    false_pos = []
    for gm in ALL:
        try:
            arch, lv = route(gm)
        except Exception as e:
            print(f"  {gm}: ERROR {type(e).__name__}"); continue
        tag = ""
        if lv > 0:
            cracked.append((gm, arch, lv))
            tag = f"  <-- {arch} solved {lv}"
        print(f"{gm:>6}: {arch}{tag}")
    print(f"\nCRACKED: {cracked}")
    print(f"Safety: {len(cracked)} games solved, {25-len(cracked)} abstained. "
          f"Expected cracks = wa30, sb26. Any OTHER crack = a bonus (or a false positive to check).")


if __name__ == "__main__":
    main()
