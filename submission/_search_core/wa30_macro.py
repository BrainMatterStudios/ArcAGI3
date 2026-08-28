"""wa30_macro — hierarchical planner for grab-drag levels with autonomous agents.

WHY THIS SHAPE (docs/RESEARCH-2026-08-27 §12)

Two planners already failed on wa30 level 3 and each failed informatively:

  solve_wa30 (shipped)  A* legs, avatar delivers EVERY block   -> 169 moves / 100
  wa30_beam             joint order+assignment over those legs -> 185 moves
  wa30_coop             flat beam over primitive actions with
                        the engine as the model                -> stalls at 2/5

The first two solve the wrong problem: the level's carriers deliver blocks for
free (2 of 5 on level 3, 4 of 5 on level 2), so the avatar's job is the
REMAINDER. The third models the carriers exactly but searches primitives, and a
flat beam cannot find a coherent 30-move drag across a 100-move horizon.

So: keep the exact model, raise the action granularity.

MACROS
  DELIVER(block, pad)   one A* drag leg, replanned against the CURRENT board
                        (the shipped `_astar_leg` already does this well —
                        level 1 falls in 26 moves)
  WAIT(k)               k mark-time actions; the carriers keep working, which
                        is the only way to spend the free labour

Each macro is executed on a deepcopy of the real engine, so carrier behaviour
during the macro is exact — including a carrier stealing the block we were
dragging, which the search simply sees as the resulting state.

Search is best-first on (blocks unplaced, moves spent), bounded by the level's
own StepCounter budget.

Usage:  ONLY_RESET_LEVELS=true python wa30_macro.py [level] [beam] [time_s]
"""
from __future__ import annotations

import copy
import heapq
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import specialists as sp  # noqa: E402

WAITS = (4, 10, 20)
PRUNE = os.environ.get("WA30_MACRO_PRUNE", "1") != "0"


def _action(a: int):
    from arcengine import GameAction
    return GameAction[f"ACTION{a}"]


def game_of(env):
    return env._game                      # noqa: SLF001


def targets_of(g) -> list[tuple[int, int]]:
    cached = getattr(g, "_macro_targets", None)
    if cached is None:
        cached = [(x, y) for x in range(0, 64, 4) for y in range(0, 64, 4)
                  if g.shbxbhnhjc((x, y))]
        g._macro_targets = cached         # noqa: SLF001
    return cached


def status(g):
    """-> (unplaced_positions, occupied_target_positions)."""
    bl = g.current_level.get_sprites_by_tag("geezpjgiyd")
    unplaced, occupied = [], set()
    for b in bl:
        if g.shbxbhnhjc((b.x, b.y)) and b not in g.zmqreragji:
            occupied.add((b.x, b.y))
        else:
            unplaced.append((b.x, b.y))
    return unplaced, occupied


def run_actions(env, acts, lc0):
    """Execute action ids on `env`. -> ('win'|'dead'|'ok', moves_done)."""
    done = 0
    for a in acts:
        o = env.step(_action(a))
        done += 1
        if o is None:
            return "dead", done
        if o.levels_completed > lc0:
            return "win", done
        if str(o.state).endswith("GAME_OVER"):
            return "dead", done
    return "ok", done


def engine_percept(g, frame=None):
    """Board model read from the ENGINE, not the frame.

    `sp._wa30_perceive` is frame-based and returns None on mid-level states
    (measured: 0 legs after the first macro) — the avatar merges with a held
    block, and pads stop reading as pads once covered. That is a perception
    problem, and it is not the one this planner is testing, so the planner
    takes the board from the engine and leaves perception to be solved
    separately once a within-budget plan is known to exist.

    Walls come from the game's OWN passability predicate `kblzhbvysd`, the same
    one the carriers' BFS uses, so the obstacle set cannot disagree with the
    engine.
    """
    av = g.current_level.get_sprites_by_tag("wbmdvjhthc")
    if not av:
        return None
    avatar = (av[0].x, av[0].y)
    unplaced, occupied = status(g)
    blocks = [tuple(b) for b in unplaced]
    free_pads = [p for p in targets_of(g) if p not in occupied]
    # WALLS COME FROM THE FRAME, not from `kblzhbvysd`.
    # kblzhbvysd reported a solid column at x=32 spanning the whole board, with
    # every block on one side and every pad on the other — a phantom the
    # carrier demonstrably walks through. It is not avatar passability.
    # The frame rule (a cell holding anything that is not background, and not
    # the avatar / a block / a pad) is the one the shipped perceiver uses and it
    # is right at level start. Geometry from the frame, semantics from the
    # engine.
    from graft_explorer import _background_color                    # noqa: PLC0415
    if frame is None:
        return None
    grid = sp.settled(frame)
    rows = grid.tolist()
    bg = _background_color(rows)
    known = {avatar} | set(blocks) | set(free_pads) | set(occupied)
    walls = set()
    for yy in range(0, 64, 4):
        for xx in range(0, 64, 4):
            if (xx, yy) in known:
                continue
            cell = grid[yy:yy + 4, xx:xx + 4]
            if cell.size and (cell != bg).any():
                walls.add((xx, yy))
    for i in range(0, 64, 4):
        walls |= {(-4, i), (64, i), (i, -4), (i, 64)}
    return avatar, blocks, free_pads, walls, occupied


def idle_action(env, lc0):
    """An action that spends a move without disturbing the board.

    Waiting is the whole point of the mechanic — the carriers only advance when
    the player acts — but ACTION5 is grab/release, so 'waiting' beside a block
    picks it up and drags it around. This probes the five actions on a copy and
    returns one that leaves the avatar and every block where they were,
    preferring a blocked move (a wall bump is a true no-op).
    """
    g = game_of(env)
    av = g.current_level.get_sprites_by_tag("wbmdvjhthc")
    if not av:
        return None
    before = ((av[0].x, av[0].y),
              tuple(sorted((b.x, b.y) for b in g.current_level.get_sprites_by_tag("geezpjgiyd"))))
    for a in (1, 2, 3, 4, 5):
        probe = copy.deepcopy(env)
        o = probe.step(_action(a))
        if o is None or o.levels_completed > lc0 or str(o.state).endswith("GAME_OVER"):
            continue
        gg = game_of(probe)
        av2 = gg.current_level.get_sprites_by_tag("wbmdvjhthc")
        if not av2:
            continue
        after = ((av2[0].x, av2[0].y),
                 tuple(sorted((b.x, b.y) for b in gg.current_level.get_sprites_by_tag("geezpjgiyd"))))
        if after == before and not gg.zmqreragji:
            return a
    return None


def handoff_cells(walls, occupied, blocks):
    """Cells in the dividing wall line that a block can be parked in.

    MEASURED on level 3: the avatar's free-walk region is entirely x <= 28
    (BFS over real engine moves), while every pad is at x = 52/56. **The avatar
    can never deliver a block to a pad.** The two cells at (32,12) and (32,32)
    are gaps in the dividing wall, plugged at level start by two blocks; they
    are reachable by the carrier from the right and pushable-into by the avatar
    from the left (stand at x=24, drag a block 28 -> 32).

    So the avatar's only useful move is a HANDOFF: park a block in a gap for
    the carrier to collect. `deliver_legs` targeting pads was planning moves the
    avatar is physically incapable of making, which is why its "16 legs" at
    move 0 were mostly fiction.
    """
    xs = sorted({x for (x, _y) in walls})
    if not xs:
        return []
    # the dividing line is the column that walls occupy most densely
    from collections import Counter
    col = Counter(x for (x, _y) in walls if 0 <= x < 64).most_common(1)[0][0]
    blockset = {tuple(b) for b in blocks}
    # EVERY cell of the divider is a handoff target, INCLUDING the ones that
    # read as wall (2026-08-28). The old version returned only the non-wall
    # gaps, which on level 3 is exactly the two cells the starting blocks
    # already plug -> after filtering, ZERO handoff legs, which is why the
    # handoff macro never fired and the search fell back on fiction pad legs.
    # The engine settles it: a block was pushed to (32,24), a cell the frame
    # reads as solid wall. Blocks enter the divider; the avatar does not.
    out = []
    for y in range(0, 64, 4):
        c = (col, y)
        if c in occupied or c in blockset:
            continue
        out.append(c)
    return out


class TwoWallDragModel:
    """`sp._DragModel` with SEPARATE passability for the avatar and the block.

    MEASURED 2026-08-28 (sprite-by-sprite trace of one executed leg). The
    shipped model carries a single wall set and applies it to both bodies. On
    level 3 that is simply false, and it is the reason every leg the planner
    liked was fiction:

        avatar : (16,36) -> (16,32) -> (28,32) -> (28,24)   never passes x=28
        block  : pushed (32,32) -> (32,24)                  INTO the divider

    The divider column reads as wall in the frame. The avatar can never enter
    it; a block can be pushed into it. With one shared wall set the planner
    must get one of those two wrong, and it chose to let the avatar walk to
    x=52 — so `_astar_leg` happily returned 16 block->pad legs that the engine
    then refused to carry out, at ~50 ms each, every single expansion.

    Step semantics are otherwise identical to `sp._DragModel`.
    """

    def __init__(self, avatar_walls: set, block_walls: set):
        self.avatar_walls = avatar_walls
        self.block_walls = block_walls
        self.walls = avatar_walls          # for anything introspecting .walls

    def step(self, state, action):
        ax, ay, bx, by, facing, held = state
        if action == 5:
            if held:
                return (ax, ay, bx, by, facing, False)
            if sp._faced_cell(ax, ay, facing) == (bx, by):   # noqa: SLF001
                return (ax, ay, bx, by, facing, True)
            return state
        dx, dy = sp.DELTAS[action]
        if not held:
            nf = sp._facing_of(dx, dy)                        # noqa: SLF001
            tgt = (ax + dx, ay + dy)
            if tgt not in self.avatar_walls and tgt != (bx, by):
                return (tgt[0], tgt[1], bx, by, nf, held)
            return (ax, ay, bx, by, nf, held)
        nav = (ax + dx, ay + dy)
        nbl = (bx + dx, by + dy)
        ok = ((nav not in self.avatar_walls or nav == (bx, by)) and
              (nbl not in self.block_walls or nbl == (ax, ay)))
        if ok:
            return (nav[0], nav[1], nbl[0], nbl[1], facing, held)
        return state


def avatar_region(avatar, walls) -> set[tuple[int, int]]:
    """Every cell the avatar could stand on, ignoring blocks.

    Blocks are treated as PASSABLE on purpose: they can be pushed, so counting
    them as obstacles would under-estimate the region and could wrongly discard
    a real leg. Ignoring them over-estimates it, which makes every filter built
    on this region a SOUND necessary condition — it can keep a leg that turns
    out to be impossible (the A* then fails, as before) but can never throw away
    a leg that was possible.
    """
    seen = {avatar}
    stack = [avatar]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 4, y), (x + 4, y), (x, y - 4), (x, y + 4)):
            if (nx, ny) in seen or (nx, ny) in walls:
                continue
            if not (0 <= nx < 64 and 0 <= ny < 64):
                continue
            seen.add((nx, ny))
            stack.append((nx, ny))
    return seen


def pushable_targets(cells, region) -> list[tuple[int, int]]:
    """Targets a block could actually be pushed INTO by this avatar.

    To push a block into cell `c` the avatar ends up two cells behind it: the
    block passes through `c-d` and the avatar stands on `c-2d`. So `c` is only
    reachable as a push target if some `c-2d` is in the avatar's region. This
    is the test that makes the level-3 pad legs disappear — the avatar's region
    is x <= 28 and every pad is at x = 52/56, so no pad has a legal push stance.
    """
    out = []
    for (x, y) in cells:
        for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4)):
            if (x + 2 * dx, y + 2 * dy) in region:
                out.append((x, y))
                break
    return out


def deliver_legs(env, av_color, deadline):
    """-> [(label, action_ids)] A* drag legs to a free pad OR a handoff gap.

    REACHABILITY GATE (2026-08-28, measured). Profiling one expansion of the
    level-3 search: `deliver_legs` was **8,052 ms of an 8,086 ms expansion —
    99.6% of the planner's entire cost** — and 160 of its 200 A* calls were
    block->pad legs on a level where the correction (§12c-quater) had already
    established the avatar can never reach a pad. The correction was written
    down but never applied to the code, so every expansion re-derived the same
    16 fictions at ~50 ms each. Filtering targets by whether a legal push
    stance exists removes them at flood-fill cost, and self-disables on levels
    where the avatar CAN reach pads (L1, L2), so it is not a level-3 special
    case.
    """
    g = game_of(env)
    per = engine_percept(g, env.observation_space)
    if per is None:
        return []
    avatar, blocks, free_pads, walls, occupied = per
    blockset = {tuple(b) for b in blocks}
    # The avatar's TRUE region: frame walls block it, and so do other blocks.
    # This is the set the engine trace agrees with (max x = 28 on level 3).
    region = avatar_region(avatar, walls | occupied)
    gaps = handoff_cells(walls, occupied, blocks)
    free_pads = pushable_targets(free_pads, region)
    out = []
    # HANDOFF LEGS. The avatar cannot reach any pad (walk region x <= 28,
    # pads at x = 52/56, verified by engine BFS both before and after the slots
    # are vacated), so a leg to a pad is fiction. What it CAN do is stand at
    # x = 24 and push a block from 28 into a slot at x = 32, where the carrier
    # collects it. The slot is a wall cell for walking but a legal resting place
    # for a block, so it is opened in the obstacle set only as that leg's goal.
    # Any leg the model gets wrong simply fails when executed on the engine and
    # the child is discarded — the engine is the arbiter, not the model.
    for b in blocks:
        others = blockset - {tuple(b)}
        for gcell in gaps:
            if time.time() > deadline:
                return out
            # avatar keeps ALL walls (it cannot enter the divider); the block
            # gets the goal cell opened (it can be pushed into it).
            model = TwoWallDragModel(
                avatar_walls=walls | others | occupied,
                block_walls=(walls - {gcell}) | others | occupied)
            best = None
            for facing in (0, 90, 180, 270):
                path = sp._astar_leg(                                     # noqa: SLF001
                    model, (avatar[0], avatar[1], b[0], b[1], facing, False), gcell)
                if path and (best is None or len(path) < len(best)):
                    best = path
            if best:
                out.append((f"handoff{tuple(b)}->{gcell}", best))
    for b in blocks:
        others = set(x for x in blocks if x != b)
        blocked = walls | others | occupied
        # FLOOD-FILL GATE before any A*. A block can only be dragged to a pad
        # it can physically travel to; reachability over the same obstacle set
        # is a necessary condition and costs one BFS instead of
        # len(free_pads) * 4 A* searches. On level 3 the divider is in `walls`
        # for both bodies here, so no pad is reachable and this loop's 160 A*
        # calls per expansion — every one of which returned a leg the engine
        # refused to execute — collapse to a single flood fill.
        reach, stack = {tuple(b)}, [tuple(b)]
        while stack:
            x, y = stack.pop()
            for nx, ny in ((x - 4, y), (x + 4, y), (x, y - 4), (x, y + 4)):
                if (nx, ny) in reach or (nx, ny) in blocked:
                    continue
                if not (0 <= nx < 64 and 0 <= ny < 64):
                    continue
                reach.add((nx, ny))
                stack.append((nx, ny))
        model = sp._DragModel(blocked)                     # noqa: SLF001
        for pad in [p for p in free_pads if p in reach]:
            if time.time() > deadline:
                return out
            # FACING MATTERS. The drag model can only grab in the direction the
            # avatar is facing, and facing changes with every macro. Hard-coding
            # 0 made A* fail on every (block, pad) pair from move 33 onward --
            # 0 legs, which looked like a dead end and was really a bad start
            # state. The true facing is not in the sprite data we read, so try
            # all four and keep the cheapest leg that exists.
            best = None
            for facing in (0, 90, 180, 270):
                path = sp._astar_leg(                       # noqa: SLF001
                    model, (avatar[0], avatar[1], b[0], b[1], facing, False), pad)
                if path and (best is None or len(path) < len(best)):
                    best = path
            if best:
                out.append((f"deliver{b}->{pad}", best))
    return out


CARRIER_TAGS = ("kdweefinfi", "ysysltqlke")
ETA_STUCK = 99          # a carrier with no path at all


def carrier_etas(g) -> list[int]:
    """Remaining travel for every autonomous carrier, one entry each.

    THE ENGINE IS THE MODEL. `wa30.py:1142 ynmgxjqkgh` drives a carrier with
    the game's OWN breadth-first searches — `czrprbohhe` when it is empty
    (goal: any staging cell in `lkvghqfwan`) and `cyjrduhzmz` when it is
    carrying (goal: a cell from which the held block lands on a pad). Those are
    methods on the live game object, and the planner already runs every macro
    on a `deepcopy`, so we can simply CALL them instead of reimplementing the
    pick rule. A reimplementation could disagree with the engine; this cannot.

    Returns path length - 1 = moves still needed, since the returned path
    includes the carrier's current cell.
    """
    out = []
    for tag in CARRIER_TAGS:
        for c in g.current_level.get_sprites_by_tag(tag):
            try:
                if c in g.nsevyuople:
                    p = (g.cyjrduhzmz(c) if tag == "kdweefinfi"
                         else g.egqayvffim(c))
                else:
                    p = (g.czrprbohhe(c) if tag == "kdweefinfi"
                         else g.zauouvdhta(c))
            except Exception:  # noqa: BLE001 — a missing helper must not kill the search
                p = None
            out.append(len(p) - 1 if p and len(p) > 1 else
                       (0 if p else ETA_STUCK))
    return out


def progress_key(env):
    """(blocks needing avatar work, blocks not on a pad, total carrier travel).

    A block parked in the dividing wall column is HANDED OFF: the avatar's job
    on it is done and the carrier will collect it. Scoring only on 'unplaced'
    made every handoff look like zero progress, so best-first had no gradient
    and wandered — the search was already discovering handoffs at (32,24),
    (32,28), (32,36) and then throwing them away.

    THIRD TERM ADDED 2026-08-28. The first two terms tie almost everywhere:
    every channel row scores identically even though a block parked next to the
    carrier is a far shorter ferry than one parked across the board, and every
    WAIT of a different length scores identically until a delivery actually
    lands. Total remaining carrier travel breaks both ties in the right
    direction and is exact (it is the engine's own BFS). It sits LAST so it can
    only order states that are otherwise equal in real progress — a shorter
    ferry never outranks an actually-placed block.
    """
    g = game_of(env)
    per = engine_percept(g, env.observation_space)
    unplaced, occupied = status(g)
    eta = sum(carrier_etas(g))
    if per is None:
        return (len(unplaced), len(unplaced), eta)
    _av, _bl, _fp, walls, _oc = per
    from collections import Counter
    cols = Counter(x for (x, _y) in walls if 0 <= x < 64)
    col = cols.most_common(1)[0][0] if cols else None
    todo = sum(1 for (x, _y) in unplaced if x != col)
    return (todo, len(unplaced), eta)


def all_blocks_viable(env, av_color, deadline) -> bool:
    """True when every unplaced block still has at least one route to a pad.

    A block with no route is unplaceable, so the level can no longer be won —
    the classic Sokoban deadlock test, which this level needs because the
    carrier can seal the board while apparently helping.
    """
    g = game_of(env)
    per = engine_percept(g, env.observation_space)
    if per is None:
        return False
    avatar, blocks, free_pads, walls, occupied = per
    if len(free_pads) < len(blocks):
        return False
    # STATIC obstacles only. Other unplaced blocks are movable, so treating
    # them as walls would call the START state dead: the left-hand blocks reach
    # a pad only through a gap another block is sitting in, and that block can
    # be moved. A block is dead only when the immovable geometry — walls plus
    # already-placed blocks — cuts it off from every free pad, which is exactly
    # the sealed board the carrier creates.
    #
    # This is a FLOOD FILL, not an A* set. The A* version ran
    # blocks x pads x facings searches per child and the whole planner managed
    # 4 expansions in 600 s. Reachability over static geometry is a necessary
    # condition for placeability and costs one BFS per child.
    # GOALS INCLUDE THE HANDOFF CHANNEL (2026-08-28).
    # Reaching a pad is not the only way a block gets placed — reaching the
    # divider is enough, because the carrier ferries it the rest of the way.
    # That is the level's whole mechanic, and `progress_key` already scores it
    # that way. Asking only for pads is the SAME one-obstacle-set error that
    # produced the fiction legs: it applies avatar-side geometry to a body that
    # does not obey it.
    #
    # MEASURED: level 4 start reported `all_blocks_viable = False` — the entire
    # level dead before a single expansion, so `plan` returned in 0.5 s of a
    # 900 s budget and the chain stopped. Level 3 slipped through only by
    # accident: its two divider cells were occupied by BLOCKS, which
    # `engine_percept` subtracts from the wall set, leaving a hole the flood
    # fill could pass through. Level 4 has no such hole.
    blocked = walls | occupied
    pads = set(free_pads) | set(handoff_cells(walls, occupied, blocks))
    for b in blocks:
        start = tuple(b)
        stack, seen = [start], {start}
        found = False
        while stack:
            x, y = stack.pop()
            if (x, y) in pads:
                found = True
                break
            for nx, ny in ((x - 4, y), (x + 4, y), (x, y - 4), (x, y + 4)):
                if (nx, ny) in seen or (nx, ny) in blocked:
                    continue
                if not (0 <= nx < 64 and 0 <= ny < 64):
                    continue
                seen.add((nx, ny))
                stack.append((nx, ny))
        if not found:
            return False
    return True


def plan(env0, budget: int, beam: int = 8, time_s: float = 600.0, verbose=True,
         av_color: int | None = None):
    """Best-first over macro sequences. -> (plan_actions, moves) or (None, None)."""
    t0 = time.time()
    deadline = t0 + time_s
    g0 = game_of(env0)
    frame = env0.observation_space
    lc0 = frame.levels_completed
    # The avatar colour must be the REAL one. Taking "the first colour that
    # perceives successfully" silently produces a garbage board model, so it
    # is passed in from the specialist's own probe (core._wa30_avcolor).
    if av_color is None:
        return None, None
    grid = sp.settled(frame)

    start = (progress_key(env0), 0, 0, [], env0)
    heap = [start]
    seen_best = start[0]
    expansions = 0
    while heap and time.time() < deadline:
        heap.sort(key=lambda s: (s[0], s[1]))
        node = heap.pop(0)
        nun, moves, _tb, acts, env = node
        expansions += 1
        if nun < seen_best:
            seen_best = nun
            if verbose:
                print(f"   {moves:>3} moves: todo={nun[0]} unplaced={nun[1]} "
                      f"carrier_eta={nun[2]}  "
                      f"({expansions} macro expansions, {time.time()-t0:.0f}s)")
        legs = deliver_legs(env, av_color, deadline)
        cands = list(legs)
        idle = idle_action(env, lc0)
        if idle is not None:
            # ADAPTIVE WAIT (2026-08-28). The fixed ladder (4, 10, 20) is a
            # guess about a quantity the engine will tell us exactly: each
            # carrier's own BFS says how many player actions it still needs to
            # reach its goal. Waiting that many moves is the ONLY wait length
            # that is guaranteed to change the board — anything shorter buys
            # nothing, anything longer overspends a per-level move budget.
            # Waiting eta+1 covers the pick-up/drop action that follows arrival.
            waits = set(WAITS)
            for e in carrier_etas(game_of(env)):
                if 0 < e < ETA_STUCK:
                    waits.add(e + 1)
            for k in sorted(waits):
                cands.append((f"wait{k}", [idle] * k))
        if verbose and expansions <= 4:
            per = engine_percept(game_of(env), env.observation_space)
            if per is None:
                print(f"      [expand {expansions}] PERCEPT=None moves={moves}")
            else:
                av_, bl_, fp_, wl_, oc_ = per
                print(f"      [expand {expansions}] {len(legs)} legs "
                      f"(lens {sorted(len(p) for _l, p in legs)[:6]}) moves={moves} "
                      f"| avatar {av_} blocks {bl_} freepads {len(fp_)} walls {len(wl_)} occupied {len(oc_)}")
                if not legs:
                    for yy in range(0, 64, 4):
                        row = ""
                        for xx in range(0, 64, 4):
                            c = (xx, yy)
                            carr_cells = {(cc.x, cc.y) for cc in
                                          game_of(env).current_level.get_sprites_by_tag("kdweefinfi")}
                            row += ("A" if c == av_ else "B" if c in bl_ else
                                    "C" if c in carr_cells else
                                    "o" if c in oc_ else "." if c in fp_ else
                                    "#" if c in wl_ else " ")
                        print("        |" + row + "|")
        for label, path in cands:
            if moves + len(path) > budget:
                continue
            child = copy.deepcopy(env)
            verdict, done = run_actions(child, path, lc0)
            if verdict == "win":
                return acts + path[:done], moves + done
            if verdict == "dead":
                continue
            un = progress_key(child)
            # DEADLOCK PRUNE. Measured on level 3: the wall column at x=32 has
            # exactly two gaps, and both are held open by blocks. The carrier
            # autonomously delivers those two — which SEALS the board, stranding
            # the other three blocks on the far side from every pad. The avatar
            # then cannot cross x=32 at any row (probed: blocked at x=28 for
            # y=12, 24 and 32).
            # Best-first on (unplaced, moves) walks straight into that state,
            # because the carrier's two free deliveries look like the fastest
            # progress available. So a child is discarded when ANY unplaced
            # block has no leg to any free pad: that block can never be placed
            # again and the branch is already lost, however good its count is.
            # PRUNE IS A SWITCH, NOT A LAW (measured 2026-08-28 on level 4).
            # It is load-bearing on level 3, where the carrier can seal the
            # board. On level 4 it is net-NEGATIVE: with it the search stalls
            # at todo=3 by 51 moves, without it it reaches todo=2 by 79. The
            # difference is not beam width — beam 8 and beam 32 produce a
            # byte-identical trace, so children are being DISCARDED, not
            # trimmed. Default ON (level 3 needs it); WA30_MACRO_PRUNE=0 turns
            # it off for levels where free carrier labour is absent.
            if (PRUNE and un[0]
                    and not all_blocks_viable(child, av_color, deadline)):
                continue
            heap.append((un, moves + done, expansions, acts + path[:done], child))
        heap = sorted(heap, key=lambda s: (s[0], s[1]))[:beam]
    return None, None


def chain(core, buds, max_level: int, time_s: float, av_color=None) -> dict:
    """Solve wa30 level by level, shipped A* first and the macro planner as the
    fallback. -> per-level record.

    WHY BOTH. The shipped `solve_wa30` is a pure avatar->pad A*; it clears the
    levels where the avatar really can deliver (L1, L2) in far less wall time
    than a macro search. It cannot clear a level whose pads sit behind a
    divider, because there the avatar delivers NOTHING and the whole job is
    handoff + carrier. Running the cheap planner first and the macro planner
    only on its failures keeps the fast path fast and is exactly how a shipped
    specialist would have to be structured.
    """
    rows = []
    for lvl in range(1, max_level + 1):
        a0 = core.backend.actions_spent
        t0 = time.time()
        r = sp.solve_level(core, "wa30_grabdrag", lvl, 120.0)
        if r.get("solved"):
            core.backend.adopt(r["handle"])
            rows.append({"level": lvl, "by": "shipped", "moves": None,
                         "actions": core.backend.actions_spent - a0,
                         "wall": round(time.time() - t0, 1)})
            print(f"  L{lvl}: shipped A*  "
                  f"({core.backend.actions_spent - a0} actions, "
                  f"{time.time() - t0:.1f}s)")
            continue
        live = core.backend.env
        budget = buds[lvl - 1] if lvl - 1 < len(buds) else 100
        # The avatar colour is a by-product of the SPECIALIST'S probe, so it
        # only exists on `core` after `solve_level` has run at least once —
        # reading it before the chain starts yields None, and `plan` bails
        # instantly on a None colour. That is not a stall; it is a missing
        # input, and it cost a full chain run to spot. Read it here, lazily.
        av = av_color or getattr(core, "_wa30_avcolor", None)
        if av is None:
            print(f"  L{lvl}: avatar colour unknown — cannot plan")
            break
        p, moves = plan(live, budget, beam=8, time_s=time_s, verbose=False,
                        av_color=av)
        if p is None:
            rows.append({"level": lvl, "by": None, "moves": None,
                         "actions": core.backend.actions_spent - a0,
                         "wall": round(time.time() - t0, 1)})
            print(f"  L{lvl}: NO PLAN inside {budget} moves "
                  f"({time.time() - t0:.1f}s)  <- chain stops here")
            break
        lc = live.observation_space.levels_completed
        verdict, done = run_actions(live, p, lc)
        rows.append({"level": lvl, "by": "macro", "moves": moves,
                     "budget": budget, "verdict": verdict,
                     "actions": core.backend.actions_spent - a0,
                     "wall": round(time.time() - t0, 1)})
        print(f"  L{lvl}: macro {moves}/{budget} moves -> {verdict} "
              f"({core.backend.actions_spent - a0} actions, "
              f"{time.time() - t0:.1f}s)")
        if verdict != "win":
            break
    return {"rows": rows,
            "levels_won": sum(1 for r in rows
                              if r["by"] and (r.get("verdict", "win") == "win")),
            "actions": core.backend.actions_spent}


def main() -> int:
    logging.disable(logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "chain":
        max_level = int(sys.argv[2]) if len(sys.argv) > 2 else 9
        time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
        if os.environ.get("ONLY_RESET_LEVELS") != "true":
            print("ERROR: run with ONLY_RESET_LEVELS=true")
            return 2
        import json

        import search_core as sc
        from arc_agi import Arcade, OperationMode
        from step_budgets import budgets_for

        root_dir = os.path.dirname(os.path.dirname(_HERE))
        arc = Arcade(operation_mode=OperationMode.OFFLINE,
                     environments_dir=os.path.join(root_dir,
                                                   "environment_files"))
        probe = arc.make("wa30")
        probe.reset()
        buds = budgets_for(game_of(probe))
        env = arc.make("wa30")
        env.reset()
        core = sc.SearchCore(env, backend="snapshot", max_states=20000)
        core.warmup_and_freeze()
        av = getattr(core, "_wa30_avcolor", None)
        print(f"wa30 CHAIN to L{max_level}, budgets {buds}, avatar colour {av}\n")
        res = chain(core, buds, max_level, time_s, av)
        print(f"\nlevels won: {res['levels_won']} / {max_level}"
              f"   total engine actions (snapshot): {res['actions']}")
        out = os.path.join(_HERE, "results", "wa30_chain.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {out}")
        return 0

    level = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    beam = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    time_s = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    if os.environ.get("ONLY_RESET_LEVELS") != "true":
        print("ERROR: run with ONLY_RESET_LEVELS=true")
        return 2

    import search_core as sc
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
    un, _ = status(game_of(live))
    carr = len(game_of(live).current_level.get_sprites_by_tag("kdweefinfi"))
    print(f"wa30 L{level}: {len(un)} unplaced blocks, {carr} carriers, "
          f"budget {budget}, beam {beam}\n")

    av = getattr(core, "_wa30_avcolor", None)
    print(f"avatar colour from the specialist probe: {av}")
    p, moves = plan(live, budget, beam=beam, time_s=time_s, av_color=av)
    if p is None:
        print(f"\nno plan inside {budget} moves")
        return 1
    print(f"\nPLAN: {moves} moves (budget {budget})")

    # clean-room verification
    env2 = arc.make("wa30")
    env2.reset()
    c2 = sc.SearchCore(env2, backend="snapshot", max_states=20000)
    c2.warmup_and_freeze()
    for lvl in range(1, level):
        c2.backend.adopt(sp.solve_level(c2, "wa30_grabdrag", lvl, 120.0)["handle"])
    v = c2.backend.env
    lc = v.observation_space.levels_completed
    verdict, done = run_actions(v, p, lc)
    print(f"ENGINE VERIFY (clean replay): {verdict} after {done} moves -> "
          f"level {level} {'SOLVED' if verdict == 'win' else 'NOT solved'}")
    return 0 if verdict == "win" else 1


if __name__ == "__main__":
    sys.exit(main())
