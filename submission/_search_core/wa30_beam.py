"""wa30_beam — joint (order + assignment) beam planner for grab-drag levels.

WHY (docs/RESEARCH-2026-08-27 §11c/§11f). wa30 level 3 enforces a 100-move
budget and the shipped `solve_wa30` planner emits 169, so the level is
unreachable. The step-budget finding proved a within-budget solution must
exist (the published human baseline of 183 spans multiple attempts, each
capped at 100), so this is a planning-quality gap, not a dead level.

WHAT THE SHIPPED PLANNER DOES

    greedy_assign()   sorts every (block, pad) pair by Manhattan distance and
                      takes them nearest-first -- ONE fixed assignment
    then              permutes the ORDER of the blocks (n! x 4 facings) and
                      A*s each leg with the remaining blocks as obstacles

On level 3 that is 5 blocks and **8** pads, so the pad CHOICE matters as much
as the order, and greedy fixes it before any planning happens. Measured, it
hands the farthest block (row 8) the deepest pad row (56) and pays 62 moves
for that one leg:

    block (20,20) -> pad (52,28)  33
    block (32,12) -> pad (52,24)  17
    block (32,32) -> pad (52,32)  14
    block (12,44) -> pad (52,36)  43
    block ( 8,16) -> pad (56,24)  62   <- greedy's leftover pad
                                 ---
                                 169   against a budget of 100

WHAT THIS DOES INSTEAD

Beam search over the sequence of (block, pad) commitments, scored by the REAL
A* leg cost rather than a Manhattan proxy. A partial plan is
(avatar, facing, blocks left, pads left, moves so far); each expansion picks
any remaining block and any remaining pad, plans that leg against the current
obstacle set, and keeps the cheapest `beam` partials.

Complexity is beam x |blocks| x |pads| legs per layer instead of n! x n!
enumeration, and it optimises assignment and order together, which is what
the 62-move leg above shows greedy cannot do.

Usage:  ONLY_RESET_LEVELS=true python wa30_beam.py [level] [beam]
"""
from __future__ import annotations

import heapq
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import specialists as sp  # noqa: E402


def plan_beam(avatar, blocks, pads, walls, latent, *, beam: int = 16,
              facings=(0, 90, 180, 270), deadline: float | None = None,
              leg_cache: dict | None = None):
    """-> (moves, legs) for the cheapest plan found, or (None, None).

    `legs` is [(block, pad, n_moves), ...] in execution order.
    """
    if leg_cache is None:
        leg_cache = {}
    nb = len(blocks)

    def leg(ax, ay, facing, bi, pad, blocked_key, blocked):
        key = (ax, ay, facing, bi, pad, blocked_key)
        if key in leg_cache:
            return leg_cache[key]
        model = sp._DragModel(blocked)          # noqa: SLF001
        b = blocks[bi]
        path = sp._astar_leg(model, (ax, ay, b[0], b[1], facing, False), pad)  # noqa: SLF001
        if path is None:
            leg_cache[key] = None
            return None
        s = (ax, ay, b[0], b[1], facing, False)
        for a in path:
            s = model.step(s, a)
        out = (path, s[0], s[1], s[4])
        leg_cache[key] = out
        return out

    # partial: (cost, tiebreak, ax, ay, facing, remaining_blocks, used_pads, legs, moves)
    start = [(0, i, avatar[0], avatar[1], f, frozenset(range(nb)), (), (), ())
             for i, f in enumerate(facings)]
    layer = start
    for _ in range(nb):
        nxt = []
        for cost, _tb, ax, ay, facing, remaining, used, legs, moves in layer:
            if deadline and time.time() > deadline:
                break
            placed = [p for (_b, p, _n) in legs]
            for bi in sorted(remaining):
                others = set(blocks[j] for j in remaining if j != bi)
                lat = latent - {blocks[bi]}
                blocked = walls | others | lat | set(placed)
                bkey = (frozenset(remaining) - {bi}, tuple(sorted(placed)))
                for pad in pads:
                    if pad in used:
                        continue
                    got = leg(ax, ay, facing, bi, pad, bkey, blocked)
                    if got is None:
                        continue
                    path, nax, nay, nfacing = got
                    nxt.append((cost + len(path), len(nxt), nax, nay, nfacing,
                                remaining - {bi}, used + (pad,),
                                legs + ((blocks[bi], pad, len(path)),),
                                moves + tuple(path)))
        if not nxt:
            return None, None
        nxt.sort(key=lambda t: t[0])
        layer = nxt[:beam]
    best = min(layer, key=lambda t: t[0])
    return list(best[8]), list(best[7])


def main() -> int:
    import logging

    logging.disable(logging.INFO)
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    if os.environ.get("ONLY_RESET_LEVELS") != "true":
        print("ERROR: run with ONLY_RESET_LEVELS=true (reset must stay in-level)")
        return 2

    import search_core as sc
    from arc_agi import Arcade, OperationMode
    from step_budgets import budgets_for

    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                               "environment_files"))
    env = arc.make("wa30")
    env.reset()
    buds = budgets_for(env._game)                     # noqa: SLF001
    env = arc.make("wa30")
    env.reset()
    core = sc.SearchCore(env, backend="snapshot", max_states=20000)
    core.warmup_and_freeze()
    for lvl in range(1, level):
        r = sp.solve_level(core, "wa30_grabdrag", lvl, 120.0)
        if not r.get("solved"):
            print(f"could not reach level {level}: stalled at L{lvl}")
            return 1
        core.backend.adopt(r["handle"])

    root = core.backend.root()
    g0 = sp.settled(root.obs)
    av = getattr(core, "_wa30_avcolor", None) or sp._learn_avatar_color(core.backend, root, g0)
    per = sp._wa30_perceive(g0, av)                   # noqa: SLF001
    if per is None:
        print("perception failed")
        return 1
    avatar, blocks, pads, walls, latent = per
    budget = buds[level - 1]
    print(f"wa30 level {level}: {len(blocks)} blocks, {len(pads)} pads, "
          f"move budget {budget}")

    t0 = time.time()
    moves, legs = plan_beam(avatar, blocks, pads, walls, latent, beam=beam,
                            deadline=t0 + 600)
    if moves is None:
        print("beam found no plan")
        return 1
    print(f"\nbeam={beam} planned in {time.time() - t0:.1f}s")
    print(f"PLAN: {len(moves)} moves   budget {budget}   "
          f"{'FITS' if len(moves) <= budget else 'OVER by ' + str(len(moves) - budget)}")
    for b, p, n in legs:
        print(f"   block {b} -> pad {p}: {n} moves")

    # engine verification — the only thing that counts
    toks = [sp.S(a) for a in moves]
    final, won = sp.chain(core.backend, root, toks, stop_on_level=level)
    print(f"\nENGINE VERIFY: level {level} {'SOLVED' if won is not None else 'NOT solved'}"
          f" by the {len(moves)}-move plan")
    return 0 if won is not None else 1


if __name__ == "__main__":
    sys.exit(main())
