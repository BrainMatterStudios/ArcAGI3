import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset()
    toks = []
    for _ in range(steps):
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame),
                         gstate_terminal=(obs.state == GameState.GAME_OVER),
                         gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        toks.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
    return toks


def test_augment_off_is_byte_identical():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base = _drive(SalienceExplorer(**cfg), "push", 400)
    off = _drive(HistoryAugmentedExplorer(augment=False, **cfg), "push", 400)
    assert off == base and len(base) > 50


def test_key_unaugmented_when_no_counts():
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    base = SalienceExplorer(seed=0)
    base.bg = 0
    base.vt.update(grid)
    assert pol._key(grid) == base._key(grid)


def test_key_is_never_augmented():
    """v4: phase-gated RETRY changes the explorer's behaviour via action retries, NOT the node key.
    Keys must stay byte-identical to banked so navigation/pathing is never broken (the v3 failure)."""
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    pol._counts = {((20, 20, 7),): 3}                 # any phase
    base = SalienceExplorer._key(pol, grid)
    assert pol._key(grid) == base


def test_phase_change_frees_blocked_at_untried_phase():
    """On a phase change, a blocked self-loop NOT yet confirmed blocked at the new phase is re-opened
    ACROSS ALL nodes -- so a gate the agent left becomes a frontier again. Navigation edges untouched."""
    from arcagi3.salience_explorer import _Node
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    for k in (b"gate", b"wall"):
        n = _Node(k, [(("S", 1), 0), (("S", 2), 0)])
        n.edges[("S", 1)] = (b"elsewhere", 0.0)       # real navigation edge
        n.edges[("S", 2)] = (k, 0.0)                  # blocked self-loop
        pol.nodes[k] = n
        pol._blocked_phases[(k, ("S", 2))] = {0}      # blocked only at phase 0 so far
    pol._counts = {((20, 20, 7),): 1}                 # phase now 1 (untried for these moves)

    pol._free_stale_blocks_global()
    assert ("S", 2) not in pol.nodes[b"gate"].edges   # re-opened (phase 1 not yet confirmed blocked)
    assert ("S", 2) not in pol.nodes[b"wall"].edges
    assert ("S", 1) in pol.nodes[b"gate"].edges       # navigation edge untouched


def test_reopened_move_is_deprioritised():
    """A re-opened blocked move must be lower priority than real moves, so the explorer only retries it
    after its tier-0 frontiers are exhausted -- this is what keeps productive games undisturbed."""
    from arcagi3.salience_explorer import _Node
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4, retry_tier=1)
    n = _Node(b"k", [(("S", 2), 0)])                  # ("S",2) starts at tier 0
    n.edges[("S", 2)] = (b"k", 0.0)
    pol.nodes[b"k"] = n
    pol._blocked_phases = {(b"k", ("S", 2)): {0}}
    pol._counts = {((20, 20, 7),): 1}                 # phase 1 untried
    pol._free_stale_blocks_global()
    assert ("S", 2) not in n.edges                    # re-opened (untried again)
    assert n.tier[("S", 2)] == 1                      # ... deprioritised to the configured retry tier


def test_confirmed_wall_not_reopened():
    """A move blocked at ALL counter_mod phases is a confirmed static wall and is never re-opened
    again -- this caps re-tries and stops the dev-suite thrash."""
    from arcagi3.salience_explorer import _Node
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    n = _Node(b"wall", [(("S", 2), 0)])
    n.edges[("S", 2)] = (b"wall", 0.0)
    pol.nodes[b"wall"] = n
    pol._blocked_phases[(b"wall", ("S", 2))] = {0, 1, 2, 3}   # blocked at every phase
    pol._counts = {((20, 20, 7),): 1}                          # phase 1 (already in the set)
    pol._free_stale_blocks_global()
    assert ("S", 2) in pol.nodes[b"wall"].edges               # confirmed wall -> stays blocked


def test_no_reopen_at_already_blocked_phase():
    from arcagi3.salience_explorer import _Node
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    n = _Node(b"k", [(("S", 2), 0)])
    n.edges[("S", 2)] = (b"k", 0.0)
    pol.nodes[b"k"] = n
    pol._blocked_phases = {(b"k", ("S", 2)): {1}}     # already confirmed blocked at phase 1
    pol._counts = {((20, 20, 7),): 1}                 # phase 1
    pol._free_stale_blocks_global()
    assert ("S", 2) in pol.nodes[b"k"].edges          # phase already known blocked -> not re-opened


def test_record_tracks_blocked_phase_set_only_for_self_loops():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol._counts = {((20, 20, 7),): 2}                 # phase 2
    pol._record(b"k", ("S", 3), b"k", 0.0, [(("S", 3), 0)], False)     # blocked self-loop
    assert pol._blocked_phases.get((b"k", ("S", 3))) == {2}
    pol._counts = {((20, 20, 7),): 3}                 # phase 3
    pol._record(b"k", ("S", 3), b"k", 0.0, [(("S", 3), 0)], False)     # blocked again, new phase
    assert pol._blocked_phases.get((b"k", ("S", 3))) == {2, 3}
    pol._record(b"k2", ("S", 1), b"k3", 0.0, [(("S", 1), 0)], False)   # real move (next != key)
    assert (b"k2", ("S", 1)) not in pol._blocked_phases


# ---- v5 SEMANTIC gate-identification --------------------------------------------------------------
# Only blocked moves INTO a goal-like object (a medium, non-floor, non-avatar, interior component the
# avatar is adjacent to in the move direction) are recorded -> eligible for eager retry. Walls are
# never recorded, so the eager (tier-0) retry can crack ls20 without re-bumping walls everywhere.

def _avatar_goal_grid(avatar=(30, 30), goal=(24, 30), goal_color=5, goal_w=6, goal_h=5,
                      floor_color=3, bg=0):
    """A grid with a big floor (largest component), a 3x3 mobile avatar (color 7), and a goal block."""
    g = np.zeros((64, 64), dtype=np.int8)
    g[5:21, 5:41] = floor_color           # the floor mega-component (largest -> floor color)
    ar, ac = avatar
    g[ar:ar + 3, ac:ac + 3] = 7           # avatar
    gr, gc = goal
    g[gr:gr + goal_h, gc:gc + goal_w] = goal_color
    return g


def _comps_flags(grid, bg=0, avatar_color=7):
    from arcagi3.history_augmented_explorer import OBJ_MAX_SIZE
    all_comps = P.connected_components(grid, background=bg)
    small = [o for o in all_comps if o.size <= OBJ_MAX_SIZE]
    flags = [int(o.color) == avatar_color for o in small]   # mark the avatar component mobile
    return all_comps, small, flags


def test_gate_fires_toward_adjacent_medium_object():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True)
    pol.bg = 0
    grid = _avatar_goal_grid(avatar=(30, 30), goal=(25, 30))   # goal above avatar, gap=2
    assert pol._compute_gate_actions(grid, *_comps_flags(grid)) == {("S", 1)}


def test_gate_excludes_tiny_object():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True)
    pol.bg = 0
    grid = _avatar_goal_grid(avatar=(30, 30), goal=(27, 30), goal_w=2, goal_h=1)  # size-2 speck
    assert pol._compute_gate_actions(grid, *_comps_flags(grid)) == set()


def test_gate_excludes_edge_object():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True)
    pol.bg = 0
    grid = _avatar_goal_grid(avatar=(6, 30), goal=(0, 30))   # goal touches the top edge (r0=0)
    assert pol._compute_gate_actions(grid, *_comps_flags(grid)) == set()


def test_gate_excludes_floor_colored_object():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True)
    pol.bg = 0
    grid = _avatar_goal_grid(avatar=(30, 30), goal=(25, 30), goal_color=3)  # same color as floor
    assert pol._compute_gate_actions(grid, *_comps_flags(grid)) == set()


def test_stall_gated_retry_suppressed_until_stall():
    """stall_trigger>0: the eager retry fires ONLY after the explorer saturates state-discovery (walls),
    so it cracks ls20-class walls without diverting a still-progressing game."""
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True, retry_tier=0, stall_trigger=100)
    pol._stalled = False
    assert pol._retry_active() is False
    pol._stalled = True
    assert pol._retry_active() is True


def test_no_stall_trigger_means_retry_always_active():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True, retry_tier=0, stall_trigger=0)
    assert pol._retry_active() is True


def test_stall_counter_trips_after_threshold():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True, stall_trigger=3)
    pol._update_stall(0); pol._update_stall(0)         # stuck at L0
    assert pol._stalled is False
    pol._update_stall(0)                               # 3rd action with no level-up -> stalled
    assert pol._stalled is True


def test_levelup_resets_stall_counter():
    """A still-progressing game never activates the eager retry: each level-up resets the stall."""
    pol = HistoryAugmentedExplorer(augment=True, seed=0, gate_only=True, stall_trigger=3)
    for _ in range(5):
        pol._update_stall(0)
    assert pol._stalled is True
    pol._update_stall(1)                               # level-up -> counter resets
    assert pol._stalled is False


def test_gate_only_records_only_gate_moves():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4, gate_only=True)
    pol._gate_actions = {("S", 1)}                    # only "up" points at the goal this frame
    pol._counts = {((20, 20, 7),): 1}                 # phase 1
    pol._record(b"k", ("S", 1), b"k", 0.0, [(("S", 1), 0)], False)   # blocked gate move -> recorded
    assert pol._blocked_phases.get((b"k", ("S", 1))) == {1}
    pol._record(b"k", ("S", 2), b"k", 0.0, [(("S", 2), 0)], False)   # blocked WALL move -> ignored
    assert (b"k", ("S", 2)) not in pol._blocked_phases


def test_augment_false_never_retries():
    """Firewall: with augment=False the retry machinery is inert (no blocked-phase tracking)."""
    pol = HistoryAugmentedExplorer(augment=False, seed=0, counter_mod=4)
    pol._counts = {((20, 20, 7),): 2}
    pol._record(b"k", ("S", 3), b"k", 0.0, [(("S", 3), 0)], False)
    assert pol._blocked_phases == {}


def _gd(tile=True, occ=None):
    """A 2-cell static glyph at (20,20)=7,(20,21)=8; `occ` = cells an occluder (color 9) covers."""
    g = np.zeros((64, 64), dtype=np.int8)
    if tile:
        g[20, 20] = 7
        g[20, 21] = 8
    for cell in (occ or []):
        g[cell] = 9
    return g


GLYPH_SIG = ((20, 20, 7), (20, 21, 8))


def test_disappearance_visit_counter():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0
    # A mobile occluder (2-cell color-9 block) slides up its column ONTO the static glyph and off again.
    pol._update_history(_gd(True, occ=[(25, 20), (25, 21)]))        # f0: glyph seen=1; occluder below
    pol._update_history(_gd(True, occ=[(23, 20), (23, 21)]))        # f1: glyph seen=2; occluder moved (mobile)
    pol._update_history(_gd(False, occ=[(20, 20), (20, 21)]))       # f2: mobile occluder covers glyph -> 1
    assert pol._counts.get(GLYPH_SIG) == 1
    pol._update_history(_gd(True, occ=[(18, 20), (18, 21)]))        # f3: glyph reappears; occluder above
    pol._update_history(_gd(True, occ=[(18, 20), (18, 21)]))        # f4: glyph seen=2 again (occluder paused, still mobile)
    pol._update_history(_gd(False, occ=[(20, 20), (20, 21)]))       # f5: mobile occluder covers glyph again -> 2
    assert pol._counts.get(GLYPH_SIG) == 2
    assert list(pol._counts.keys()) == [GLYPH_SIG]                  # the mobile occluder is never counted


def test_moving_object_not_counted():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0

    def gmov(pos):
        g = np.zeros((64, 64), dtype=np.int8)
        g[pos] = 9
        return g

    pol._update_history(gmov((5, 5)))
    pol._update_history(gmov((5, 6)))   # moved -> previous never confirmed static
    pol._update_history(gmov((5, 7)))   # moved again
    assert pol._counts == {}            # a thing that never stays put is never counted


def test_vanish_into_background_not_counted():
    """A confirmed-static glyph that simply disappears leaving BACKGROUND (the avatar leaving its spawn
    cell; a HUD digit blanking) is NOT a visit. A visit requires a MOVING object to cover the cells --
    that is what 'the avatar stepped onto it' means. This kills the ls20 spawn-block + HUD pollution.
    """
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    pol.bg = 0
    pol._update_history(_gd(True))    # f0 glyph seen=1
    pol._update_history(_gd(True))    # f1 glyph seen=2 (confirmed static)
    pol._update_history(_gd(False))   # f2 glyph gone, nothing covers the cells (-> background)
    assert pol._counts == {}


def test_static_glyph_replaced_by_static_not_counted():
    """A confirmed-static glyph replaced by a DIFFERENT static thing (a HUD digit changing value, no
    motion) is NOT a visit -- the replacement never moved onto it, so no mobile occluder is present.
    """
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    pol.bg = 0

    def g(color):
        a = np.zeros((64, 64), dtype=np.int8)
        a[20, 20] = color
        a[20, 21] = color
        return a

    pol._update_history(g(7))   # f0 seen=1
    pol._update_history(g(7))   # f1 seen=2 confirmed
    pol._update_history(g(5))   # f2 same cells, different color, but it simply APPEARED (never moved)
    assert pol._counts == {}


def test_paused_then_moving_object_not_counted():
    """The real ls20 failure mode: the avatar MOVES, then PAUSES >=2 frames (blocked by a wall),
    then moves away. Stability alone confirms it 'static' and counts its disappearance as a visit —
    but it moved before pausing, so mobility memory must keep it excluded the whole time.
    """
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    pol.bg = 0

    def gmov(pos):
        g = np.zeros((64, 64), dtype=np.int8)
        g[pos] = 9
        return g

    pol._update_history(gmov((5, 5)))   # appears
    pol._update_history(gmov((5, 10)))  # MOVED 5 cells -> ever_moved
    pol._update_history(gmov((5, 10)))  # paused (static, frame 2)
    pol._update_history(gmov((5, 10)))  # paused (static, frame 3 -> stability would 'confirm')
    pol._update_history(gmov((20, 20)))  # moves away -> its (5,10) cluster disappears
    assert pol._counts == {}            # a mover, even when paused, is never a 'visit'


def test_decide_runs_when_augmenting():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    pol = HistoryAugmentedExplorer(augment=True, **cfg)
    _drive(pol, "navg", 300)
    assert isinstance(pol._counts, dict)   # ran end-to-end without error
