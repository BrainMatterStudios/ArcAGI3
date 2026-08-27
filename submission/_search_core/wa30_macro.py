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
    gaps = []
    for y in range(0, 64, 4):
        if (col, y) in walls or (col, y) in occupied:
            continue
        gaps.append((col, y))
    return gaps


def deliver_legs(env, av_color, deadline):
    """-> [(label, action_ids)] A* drag legs to a free pad OR a handoff gap."""
    g = game_of(env)
    per = engine_percept(g, env.observation_space)
    if per is None:
        return []
    avatar, blocks, free_pads, walls, occupied = per
    blockset = {tuple(b) for b in blocks}
    gaps = [c for c in handoff_cells(walls, occupied, blocks) if c not in blockset]
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
            model = sp._DragModel((walls - {gcell}) | others | occupied)  # noqa: SLF001
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
        model = sp._DragModel(walls | others | occupied)   # noqa: SLF001
        for pad in free_pads:
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


def progress_key(env):
    """(blocks needing avatar work, blocks not yet on a pad).

    A block parked in the dividing wall column is HANDED OFF: the avatar's job
    on it is done and the carrier will collect it. Scoring only on 'unplaced'
    made every handoff look like zero progress, so best-first had no gradient
    and wandered — the search was already discovering handoffs at (32,24),
    (32,28), (32,36) and then throwing them away.
    """
    g = game_of(env)
    per = engine_percept(g, env.observation_space)
    unplaced, occupied = status(g)
    if per is None:
        return (len(unplaced), len(unplaced))
    _av, _bl, _fp, walls, _oc = per
    from collections import Counter
    cols = Counter(x for (x, _y) in walls if 0 <= x < 64)
    col = cols.most_common(1)[0][0] if cols else None
    todo = sum(1 for (x, _y) in unplaced if x != col)
    return (todo, len(unplaced))


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
    blocked = walls | occupied
    seen = {tuple(blocks[0])} if blocks else set()
    pads = set(free_pads)
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
                print(f"   {moves:>3} moves: {nun} unplaced  "
                      f"({expansions} macro expansions, {time.time()-t0:.0f}s)")
        legs = deliver_legs(env, av_color, deadline)
        cands = list(legs)
        idle = idle_action(env, lc0)
        if idle is not None:
            for k in WAITS:
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
            if un[0] and not all_blocks_viable(child, av_color, deadline):
                continue
            heap.append((un, moves + done, expansions, acts + path[:done], child))
        heap = sorted(heap, key=lambda s: (s[0], s[1]))[:beam]
    return None, None


def main() -> int:
    logging.disable(logging.INFO)
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
