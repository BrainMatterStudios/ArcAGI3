"""wa30_coop — plan wa30 levels against the TRUE dynamics, carriers included.

THE MECHANIC THE SHIPPED SOLVER DOES NOT MODEL (found 2026-08-27)

`wa30.py:1198 dhrikuybfo()` runs after EVERY player action and calls
`ynmgxjqkgh()`, which drives every sprite tagged `kdweefinfi`:

    carrying a block -> BFS toward a drop cell, step ONE cell, release on arrival
    adjacent to a free block -> grab it
    otherwise -> BFS toward a cell adjacent to an unplaced block, step one cell

These are **autonomous carriers that move blocks for you, one cell per player
action**. Per level:

    level      1  2  3  4  5  6  7  8  9
    carriers   0  1  1  3  1  0  0  2  2
    blocks     3  5  5  7  6  2  2 13  9
    budget   200 70 100 100 125 75 125 150 70

Level 1 has none, which is exactly why the drag-only `solve_wa30` cracks it in
26 moves. On level 3 the carrier delivers **2 of the 5 blocks for free** within
30 player actions and then idles (the remaining three are unreachable to it),
while `solve_wa30` plans 169 moves to drag all five itself against a 100-move
budget.

So the avatar does not have to deliver every block — it has to deliver the
blocks the carrier cannot reach, and it can hand the rest over by dragging them
into the carrier's reachable region.

WHAT THIS DOES

Beam search over avatar actions using the ENGINE ITSELF as the forward model,
via the snapshot backend (deepcopy 0.8 ms, ~57k steps/s offline, and provably
unbilled — see docs §11a). No hand-written carrier model can be wrong, because
there is no hand-written model: every successor is a real engine step with the
real carrier behaviour baked in.

Scoring, in order: blocks placed (win condition is
`wa30.py:1194` — every block on a target and not held), then total remaining
block->nearest-free-target distance, then avatar->nearest-unplaced-block
distance as a tiebreak so the beam does not stall.

Usage:  ONLY_RESET_LEVELS=true python wa30_coop.py [level] [beam] [budget]
"""
from __future__ import annotations

import copy
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

ACTIONS = (1, 2, 3, 4, 5)     # 4 moves + grab/release


def _action(a: int):
    """GameAction by NAME.

    `GameAction(1)` raises even though `GameAction.ACTION1.value == 1` — the
    enum exposes `value` as a property over a different `_value_`, so value
    lookup does not resolve. Name lookup is the reliable form.
    """
    from arcengine import GameAction
    return GameAction[f"ACTION{a}"]


def game_of(env):
    return env._game            # noqa: SLF001


def placed_count(g) -> tuple[int, int]:
    bl = g.current_level.get_sprites_by_tag("geezpjgiyd")
    return (sum(1 for b in bl if g.shbxbhnhjc((b.x, b.y)) and b not in g.zmqreragji),
            len(bl))


def score(g) -> tuple:
    """Lower is better. (unplaced, matched block->target cost, avatar->block).

    Three things the first version got wrong and that mattered:
      * every unplaced block was scored against its own nearest target, so two
        blocks could both "claim" the same pad and the total understated the
        real work — this uses a greedy DISJOINT matching instead;
      * targets already covered by a placed block were still counted as free;
      * nothing rewarded handing a block to a carrier, which is the whole
        point of the mechanic, so a block half-dragged toward the carrier
        looked no better than one left where it was.
    """
    bl = g.current_level.get_sprites_by_tag("geezpjgiyd")
    unplaced, occupied = [], set()
    for b in bl:
        if g.shbxbhnhjc((b.x, b.y)) and b not in g.zmqreragji:
            occupied.add((b.x, b.y))
        else:
            unplaced.append(b)
    free = [t for t in _targets(g) if t not in occupied]

    # disjoint greedy matching: cheapest (block, target) pair first
    pairs = sorted((abs(b.x - t[0]) + abs(b.y - t[1]), bi, ti)
                   for bi, b in enumerate(unplaced) for ti, t in enumerate(free))
    used_b, used_t, d = set(), set(), 0
    for cost, bi, ti in pairs:
        if bi in used_b or ti in used_t:
            continue
        used_b.add(bi)
        used_t.add(ti)
        d += cost

    # handoff credit: a carrier that is holding a block, or standing next to
    # one, is about to do the avatar's work for free.
    carriers = g.current_level.get_sprites_by_tag("kdweefinfi")
    hand = 0
    for c in carriers:
        if c in g.nsevyuople:
            hand -= 12                        # already carrying: worth a lot
        elif unplaced:
            hand += min(abs(c.x - b.x) + abs(c.y - b.y) for b in unplaced) // 8

    av = g.current_level.get_sprites_by_tag("wbmdvjhthc")
    ad = 0
    if av and unplaced:
        a = av[0]
        ad = min(abs(a.x - b.x) + abs(a.y - b.y) for b in unplaced)
    return (len(unplaced), d + hand, ad)


def _targets(g) -> list[tuple[int, int]]:
    """Target cells, read off the engine's own predicate over the board grid."""
    cached = getattr(g, "_coop_targets", None)
    if cached is not None:
        return cached
    out = []
    for x in range(0, 64, 4):
        for y in range(0, 64, 4):
            try:
                if g.shbxbhnhjc((x, y)):
                    out.append((x, y))
            except Exception:  # noqa: BLE001
                pass
    g._coop_targets = out       # noqa: SLF001
    return out


def solve(env, budget: int, beam: int = 60, deadline: float | None = None,
          verbose: bool = True):
    """-> (plan, actions_used) or (None, None). Plan is a list of action ids."""
    g0 = game_of(env)
    lc0 = None
    layer = [(score(g0), [], env)]
    best_seen = layer[0][0]
    for step in range(budget):
        if deadline and time.time() > deadline:
            break
        nxt = []
        for sc_, plan, e in layer:
            for a in ACTIONS:
                child = copy.deepcopy(e)
                obs = child.step(_action(a))
                if obs is None:
                    continue
                gg = game_of(child)
                if lc0 is None:
                    lc0 = obs.levels_completed
                if obs.levels_completed > lc0:
                    return plan + [a], len(plan) + 1
                if str(obs.state).endswith("GAME_OVER"):
                    continue
                nxt.append((score(gg), plan + [a], child))
        if not nxt:
            return None, None
        nxt.sort(key=lambda t: t[0])
        # de-duplicate on the scoring signature so the beam keeps real variety
        seen, keep = set(), []
        for item in nxt:
            k = item[0]
            if k in seen and len(keep) > beam // 3:
                continue
            seen.add(k)
            keep.append(item)
            if len(keep) >= beam:
                break
        layer = keep
        if layer[0][0] < best_seen:
            best_seen = layer[0][0]
            if verbose:
                print(f"   move {step+1:>3}: best {best_seen}  (unplaced, blockdist, avdist)")
    return None, None


def main() -> int:
    logging.disable(logging.INFO)
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if os.environ.get("ONLY_RESET_LEVELS") != "true":
        print("ERROR: run with ONLY_RESET_LEVELS=true")
        return 2

    import search_core as sc
    import specialists as sp
    from arc_agi import Arcade, OperationMode
    from step_budgets import budgets_for

    root_dir = os.path.dirname(os.path.dirname(_HERE))
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(root_dir, "environment_files"))
    probe = arc.make("wa30")
    probe.reset()
    buds = budgets_for(game_of(probe))

    env = arc.make("wa30")
    env.reset()
    core = sc.SearchCore(env, backend="snapshot", max_states=20000)
    core.warmup_and_freeze()
    for lvl in range(1, level):
        r = sp.solve_level(core, "wa30_grabdrag", lvl, 120.0)
        if not r.get("solved"):
            print(f"stalled reaching level {level} at L{lvl}")
            return 1
        core.backend.adopt(r["handle"])
    live = core.backend.env
    budget = buds[level - 1]
    p, t = placed_count(game_of(live))
    print(f"wa30 level {level}: {t} blocks, {p} placed, budget {budget}, beam {beam}")
    carr = game_of(live).current_level.get_sprites_by_tag("kdweefinfi")
    print(f"autonomous carriers: {len(carr)}\n")

    t0 = time.time()
    plan, used = solve(live, budget, beam=beam, deadline=t0 + 900)
    if plan is None:
        print(f"\nno plan within the {budget}-move budget ({time.time()-t0:.0f}s)")
        return 1
    print(f"\nPLAN FOUND: {used} moves (budget {budget}) in {time.time()-t0:.0f}s")

    # independent engine verification from a clean state
    env2 = arc.make("wa30")
    env2.reset()
    core2 = sc.SearchCore(env2, backend="snapshot", max_states=20000)
    core2.warmup_and_freeze()
    for lvl in range(1, level):
        r = sp.solve_level(core2, "wa30_grabdrag", lvl, 120.0)
        core2.backend.adopt(r["handle"])
    v = core2.backend.env
    lc = None
    ok = False
    for a in plan:
        o = v.step(_action(a))
        if lc is None:
            lc = o.levels_completed
        if o.levels_completed > lc:
            ok = True
            break
    print(f"ENGINE VERIFY (clean replay): level {level} "
          f"{'SOLVED' if ok else 'NOT solved'} in {len(plan)} moves")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
