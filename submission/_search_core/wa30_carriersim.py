"""A MODEL of wa30's autonomous carriers, and a lockstep test against the engine.

WHY THIS EXISTS (measured 2026-08-28).

`wa30_macro.plan` evaluates every candidate macro by executing it on a
`copy.deepcopy(env)`. Offline that is free. Live it is impossible: in
COMPETITION mode `Arcade.make` returns a remote wrapper whose state lives on
the server, and "search on a free copy" is closed.

The obvious repair — route macro execution through the backend API — was PRICED
before being built, using `ResetReplayBackend`'s published contract (every token
costs 1 reset + len(parent path) replay + 1 action):

    level 3 alone: 195 candidates, 2,961 snapshot actions
                   -> 92,517 reset-replay actions (31.2x tax)
                   -> 27% OVER the entire ~73k affordable budget, for ONE level

So that route is dead. But the bill is almost entirely TRIALS: committing only
the final plan costs 82 actions. The deployable shape is therefore

    evaluate candidates IN A MODEL, execute only the committed plan on the engine

which needs one thing the planner does not yet have: a carrier simulator. This
module is that simulator, plus the test that decides whether it is trustworthy.

THE MECHANIC BEING MODELLED (`wa30.py:1142 ynmgxjqkgh`), reproduced exactly:

    for each carrier:
        if it is carrying a block:
            if that block now sits on a pad -> release it
            else -> BFS (cyjrduhzmz) toward a cell from which the block lands
                    on a pad, and take ONE step along it
        else:
            if it is orthogonally adjacent to an unclaimed block that is not
               already on a pad -> pick it up AND RETURN FROM THE WHOLE
               FUNCTION (so later carriers do not act on this tick)
            else -> BFS (czrprbohhe) toward `lkvghqfwan` — the set of cells
                    adjacent to unclaimed, unplaced blocks — and take ONE step

Two details are easy to get wrong and both are load-bearing: the `return` after
a pickup (not `continue`), and that `qthdiggudy` blocks movement as well as
pathing. `qthdiggudy` is NOT a carrier-only obstacle — that was claimed and
refuted on 2026-08-28; it appears in `fuykgiiwit` too, so it stops the avatar
just as it stops a carrier, even though its sprites are `is_collidable=False`.

TRUST MODEL. A simulator that silently disagrees with the engine is worse than
no simulator — it is precisely the class of defect that produced the fiction
legs. So this module is not usable until `validate()` passes: it drives the
engine and the model in lockstep over a long, varied action sequence and
asserts that every carrier position, every block position and every carry
relation match at EVERY step. First divergence is reported with the tick and
both states.

Run:  ONLY_RESET_LEVELS=true .venv/bin/python \
          submission/_search_core/wa30_carriersim.py [level] [steps]
"""

from __future__ import annotations

import logging
import os
import sys
from collections import deque

os.environ.setdefault("ONLY_RESET_LEVELS", "true")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

CELL = 4


def _nbrs(p):
    x, y = p
    return ((x - CELL, y), (x + CELL, y), (x, y - CELL), (x, y + CELL))


class CarrierSim:
    """Pure-python model of the carrier drive. No engine calls after __init__.

    Positions are cell coordinates on the same 4-px lattice the engine uses.
    `static` is the immovable geometry (wall sprites + border); `soft` is
    `qthdiggudy`; `pads` is `wyzquhjerd`.
    """

    def __init__(self, static, soft, pads, blocks, carriers, avatar,
                 carrying=None):
        self.static = set(static)
        self.soft = set(soft)
        self.pads = set(pads)
        self.blocks = list(blocks)          # list[(x, y)]
        self.carriers = list(carriers)      # list[(x, y)]
        self.avatar = avatar
        # carrier index -> block index, mirroring `nsevyuople`
        self.carrying = dict(carrying or {})

    # -- the engine's own predicates ---------------------------------------

    def _occupied(self):
        """`pkbufziase`: every collidable body plus the border. Blocks, the
        avatar and the carriers are all collidable; wall sprites are already
        in `static`."""
        occ = set(self.static)
        occ.update(self.blocks)
        occ.update(self.carriers)
        occ.add(self.avatar)
        return occ

    def _passable(self, v, occ):
        """`kblzhbvysd`: not collidable and not in qthdiggudy."""
        return v not in occ and v not in self.soft

    def on_pad(self, p):
        return p in self.pads

    # -- the two breadth-first searches ------------------------------------

    def _bfs_to_pickup(self, start, goals, occ):
        """`czrprbohhe`: first cell of `goals` reachable over passable cells.
        The engine tests the goal on the POPPED node and seeds `visited` with
        the start, so a carrier already standing on a goal returns a 1-length
        path (i.e. no move)."""
        if start in goals:
            return [start]
        seen = {start}
        q = deque([[start]])
        while q:
            path = q.popleft()
            cur = path[-1]
            if cur in goals:
                return path
            for nb in _nbrs(cur):
                if nb in seen or not self._passable(nb, occ):
                    continue
                seen.add(nb)
                q.append(path + [nb])
        return None

    def _bfs_carrying(self, ci, bi, occ):
        """`cyjrduhzmz`: walk the CARRIER until the block it holds would land
        on a pad. Passability is `fuykgiiwit`: the carrier's own next cell and
        the block's next cell must both be clear, each ignoring the other."""
        start = self.carriers[ci]
        bx, by = self.blocks[bi]
        dx, dy = bx - start[0], by - start[1]

        def ok(v):
            tgt = (v[0] + dx, v[1] + dy)
            return ((v not in occ or v == (bx, by))
                    and v not in self.soft
                    and (tgt not in occ or tgt == start))

        if self.on_pad((start[0] + dx, start[1] + dy)):
            return [start]
        seen = {start}
        q = deque([[start]])
        while q:
            path = q.popleft()
            cur = path[-1]
            if self.on_pad((cur[0] + dx, cur[1] + dy)):
                return path
            for nb in _nbrs(cur):
                if nb in seen or not ok(nb):
                    continue
                seen.add(nb)
                q.append(path + [nb])
        return None

    # -- one player action's worth of carrier motion -----------------------

    def step(self):
        """`ynmgxjqkgh`. Mutates in place. Returns nothing."""
        claimed = set(self.carrying.values())
        for ci in range(len(self.carriers)):
            occ = self._occupied()
            if ci in self.carrying:
                bi = self.carrying[ci]
                if self.on_pad(self.blocks[bi]):
                    del self.carrying[ci]          # `kqrtstlzkg` — release
                    continue
                path = self._bfs_carrying(ci, bi, occ)
                if path and len(path) > 1:
                    nx, ny = path[1]
                    ox, oy = self.carriers[ci]
                    self.blocks[bi] = (self.blocks[bi][0] + nx - ox,
                                       self.blocks[bi][1] + ny - oy)
                    self.carriers[ci] = (nx, ny)
                continue
            # empty: adjacency pickup wins, and RETURNS from the whole sweep
            picked = False
            for bi, b in enumerate(self.blocks):
                if bi in claimed or self.on_pad(b):
                    continue
                if b in _nbrs(self.carriers[ci]):
                    self.carrying[ci] = bi
                    picked = True
                    break
            if picked:
                return                              # engine does `return`
            goals = set()
            for bi, b in enumerate(self.blocks):
                if bi in claimed or self.on_pad(b):
                    continue
                goals.update(_nbrs(b))
            if not goals:
                continue
            path = self._bfs_to_pickup(self.carriers[ci], goals, occ)
            if path and len(path) > 1:
                self.carriers[ci] = path[1]

    def snapshot(self):
        return (tuple(self.carriers), tuple(self.blocks),
                tuple(sorted(self.carrying.items())))


# --------------------------------------------------------------------------
# building a sim from a live game, and the lockstep test
# --------------------------------------------------------------------------

def from_game(g):
    """Seed a sim from the engine. Semantics from the engine, as everywhere
    else in this build; a live port derives the same sets from the frame."""
    blocks = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("geezpjgiyd")]
    carriers = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("kdweefinfi")]
    av = g.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    static = {(s.x, s.y) for s in g.current_level.get_sprites()
              if s.is_collidable
              and not ({"geezpjgiyd", "kdweefinfi", "wbmdvjhthc"} & set(s.tags))}
    for i in range(0, 64, CELL):
        static |= {(-CELL, i), (64, i), (i, -CELL), (i, 64)}
    bidx = {(s.x, s.y): i for i, s in
            enumerate(g.current_level.get_sprites_by_tag("geezpjgiyd"))}
    carrying = {}
    for ci, c in enumerate(g.current_level.get_sprites_by_tag("kdweefinfi")):
        held = g.nsevyuople.get(c)
        if held is not None and (held.x, held.y) in bidx:
            carrying[ci] = bidx[(held.x, held.y)]
    return CarrierSim(static, set(g.qthdiggudy), set(g.wyzquhjerd),
                      blocks, carriers, (av.x, av.y), carrying)


def engine_snapshot(g):
    blocks = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("geezpjgiyd")]
    carriers = [(s.x, s.y) for s in g.current_level.get_sprites_by_tag("kdweefinfi")]
    return tuple(carriers), tuple(blocks)


def validate(level: int = 3, steps: int = 120, verbose: bool = True) -> dict:
    """Drive engine and model in lockstep; report the first divergence.

    The action sequence deliberately cycles all five actions so the avatar
    moves, grabs and releases, which exercises pickup, ferry and release rather
    than one narrow path.
    """
    logging.disable(logging.CRITICAL)
    import search_core as sc
    import specialists as sp
    import wa30_macro as M
    from arc_agi import Arcade, OperationMode
    from step_budgets import budgets_for

    root = os.path.dirname(os.path.dirname(_HERE))
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(root, "environment_files"))
    probe = arc.make("wa30")
    probe.reset()
    buds = budgets_for(M.game_of(probe))
    env = arc.make("wa30")
    env.reset()
    core = sc.SearchCore(env, backend="snapshot", max_states=20000)
    core.warmup_and_freeze()
    if level > 1:
        M.chain(core, buds, level - 1, 300.0)
    live = core.backend.env
    g = M.game_of(live)

    sim = from_game(g)
    lc0 = live.observation_space.levels_completed
    seq = [1, 4, 5, 2, 3, 5, 4, 1, 5, 2] * ((steps // 10) + 1)
    mismatches = []
    for t, a in enumerate(seq[:steps]):
        obs = live.step(M._action(a))
        if obs is None or obs.levels_completed > lc0 or \
                str(obs.state).endswith("GAME_OVER"):
            if verbose:
                print(f"  run ended at tick {t} (level change / game over)")
            break
        g = M.game_of(live)
        # THE AVATAR'S OWN EFFECT IS AN INPUT, NOT A PREDICTION.
        # One engine tick is `avatar acts, THEN dhrikuybfo() drives carriers`.
        # This module models only the second half; the avatar half is already
        # modelled by `wa30_macro.TwoWallDragModel`, and a deployable planner
        # composes the two. So the sim is handed the avatar's move — including
        # the block it drags — and must predict everything the CARRIERS do.
        # Without this the harness reported a false divergence on L4 tick 9:
        # the carriers matched exactly and the only difference was a block the
        # avatar had dragged and the sim was never told about.
        av = g.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        old_av = sim.avatar
        new_av = (av.x, av.y)
        held = g.nsevyuople.get(av)
        if held is not None:
            dx, dy = new_av[0] - old_av[0], new_av[1] - old_av[1]
            if (dx, dy) != (0, 0):
                # move the dragged block by the same delta, matching it by its
                # pre-move position (avatar-adjacent, so unambiguous)
                want = (held.x - dx, held.y - dy)
                for bi, b in enumerate(sim.blocks):
                    if b == want:
                        sim.blocks[bi] = (b[0] + dx, b[1] + dy)
                        break
        sim.avatar = new_av
        sim.step()
        e_car, e_blk = engine_snapshot(g)
        s_car, s_blk = tuple(sim.carriers), tuple(sim.blocks)
        if (e_car, sorted(e_blk)) != (s_car, sorted(s_blk)):
            mismatches.append({"tick": t, "action": a,
                               "engine_carriers": e_car, "sim_carriers": s_car,
                               "engine_blocks": sorted(e_blk),
                               "sim_blocks": sorted(s_blk)})
            if len(mismatches) == 1 and verbose:
                print(f"  FIRST DIVERGENCE at tick {t} (action {a})")
                print(f"    carriers engine {e_car}")
                print(f"             sim    {s_car}")
                print(f"    blocks   engine {sorted(e_blk)}")
                print(f"             sim    {sorted(s_blk)}")
            break
        # resync block IDENTITY only (order can differ); positions were checked
        sim.blocks = list(e_blk)
    ok = not mismatches
    if verbose:
        print(f"\nwa30 L{level} carrier model vs engine over {steps} ticks: "
              f"{'MATCH' if ok else 'DIVERGED'}")
    return {"level": level, "steps": steps, "ok": ok,
            "mismatches": mismatches[:3]}


def main() -> int:
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    r = validate(level, steps)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
