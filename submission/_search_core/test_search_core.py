"""Unit tests for SearchCore v1 (stages 1-2).

Run: .venv/bin/python -m pytest submission/_search_core/test_search_core.py -v

Covers the falsifier-mandated properties:
  - snapshot determinism (real engine)
  - backend equivalence: snapshot and reset-replay find the SAME solution
    on a small real game (tu93 L1)
  - nbfs completeness: the deferred (non-novel) queue eventually expands,
    and a game solvable only THROUGH a non-novel state is solved
  - click-tier escalation on exhaustion
  - dead-click memory correctness
  - T4 never-changed-any-frame pruning
  - reset-replay live cost model (d+1 actions per test at depth d; one
    reset per rollout)
  - crc transposition key byte-identical to FrontierGraph.node_key
"""

from __future__ import annotations

import copy
import os
import random
import sys

os.environ["ONLY_RESET_LEVELS"] = "true"

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from search_core import (  # noqa: E402
    ChangeMemory,
    ClickGenerator,
    DISPATCH_ORDER,
    DeadClickMemory,
    FrontierGraph,
    GoExplorer,
    Handle,
    ResetReplayBackend,
    SearchCore,
    SnapshotBackend,
    _estimate_pitch,
    archetype_frame0,
    coarse_cell_key,
    crc_key,
    discover_games,
    mask_to_bool,
    settled,
    sh_promote,
)

ROOT = os.path.dirname(os.path.dirname(_HERE))
ENVS = os.path.join(ROOT, "environment_files")


# --------------------------------------------------------------------------
# mock environment (deterministic finite state machine, deepcopy-able)
# --------------------------------------------------------------------------

class MockObs:
    def __init__(self, frame, state, levels_completed, available_actions):
        self.frame = frame
        self.state = state
        self.levels_completed = levels_completed
        self.available_actions = available_actions
        self.win_levels = 1


class MockEnv:
    """states: name -> (frame rows, levels_completed); transitions:
    (state, token) -> state; token = ("S", id) | ("C", x, y).
    Unlisted transitions are no-ops (stay). reset() returns to `start`."""

    def __init__(self, states, transitions, start, avail, win_state=None):
        self.states = states
        self.transitions = transitions
        self.start = start
        self.avail = avail
        self.win_state = win_state
        self.cur = start

    def _obs(self):
        from arcengine import GameState

        frame, levels = self.states[self.cur]
        state = GameState.WIN if self.cur == self.win_state else GameState.NOT_FINISHED
        return MockObs([row[:] for row in frame], state, levels, list(self.avail))

    def reset(self):
        self.cur = self.start
        return self._obs()

    def step(self, action, data=None, reasoning=None):
        aid = int(action.value)
        tok = ("C", int(data["x"]), int(data["y"])) if aid == 6 else ("S", aid)
        self.cur = self.transitions.get((self.cur, tok), self.cur)
        return self._obs()


def make_core(env, backend="snapshot", **kw):
    core = SearchCore(env, backend=backend, **kw)
    core.mask.freeze()   # tests: no warmup, empty frozen mask
    return core


# --------------------------------------------------------------------------
# nbfs completeness — solution only reachable THROUGH a non-novel state
# --------------------------------------------------------------------------

def test_nbfs_completeness_deferred_queue_expands():
    A1, A2 = ("S", 1), ("S", 2)
    states = {
        "s0": ([[1, 2]], 0),
        "sA": ([[2, 1]], 0),      # novel atoms (0,0,2), (0,1,1)
        "sB": ([[2, 2]], 0),      # atom-subset of {s0, sA}: NON-novel
        "win": ([[3, 3]], 1),
    }
    transitions = {
        ("s0", A1): "sA",
        ("sA", A1): "s0",         # transposition back (seen)
        ("sA", A2): "sB",         # deferred
        ("sB", A1): "win",        # only path to the win
    }
    env = MockEnv(states, transitions, "s0", avail=[1, 2])
    core = make_core(env)
    res = core.solve_level(target=1, budget_s=10, max_tier=1)
    assert res["solved"], res
    assert core.stats["deferred_expanded"] >= 1, \
        "win required expanding the deferred (non-novel) queue"
    assert res["handle"].path == [A1, A2, A1]


def test_nbfs_strict_iw1_would_fail_setup_is_really_nonnovel():
    """Guard: sB truly adds no new atom (the completeness test is honest)."""
    atoms = {(0, 0, 1), (0, 1, 2), (0, 0, 2), (0, 1, 1)}  # s0 + sA
    sB_atoms = {(0, 0, 2), (0, 1, 2)}
    assert sB_atoms <= atoms


# --------------------------------------------------------------------------
# click tiers
# --------------------------------------------------------------------------

def _big_comp_frame(n=20, comp=11, color=5):
    """121-cell component (outside the T1 band) on a LARGER background."""
    frame = [[0] * n for _ in range(n)]
    for y in range(1, 1 + comp):
        for x in range(1, 1 + comp):
            frame[y][x] = color
    return frame


def test_click_tier_escalation_on_exhaustion():
    frame = _big_comp_frame()          # one 121-cell component: outside T1 band
    win_click = ("C", 6, 10)           # on T2's stride-4 interior lattice
    states = {"s0": (frame, 0), "win": ([[9] * 20 for _ in range(20)], 1)}
    env = MockEnv(states, {("s0", win_click): "win"}, "s0", avail=[6])
    core = make_core(env)
    res = core.solve_level(target=1, budget_s=10, max_tier=4)
    assert res["solved"], res
    assert res["tier"] == 2, f"expected escalation to tier 2, got {res['tier']}"
    assert res["handle"].path == [win_click]


def test_no_escalation_without_clicks():
    """A game with no ACTION6 must not re-run identical searches per tier."""
    states = {"s0": ([[1, 2]], 0), "sA": ([[2, 1]], 0)}
    env = MockEnv(states, {("s0", ("S", 1)): "sA"}, "s0", avail=[1, 2])
    core = make_core(env)
    res = core.solve_level(target=1, budget_s=10, max_tier=4)
    assert not res["solved"]
    assert res["tier"] == 1 and res["reason"] == "exhausted"


def test_tier1_uncapped_and_snap_to_own_color():
    n = 64
    rows = [[0] * n for _ in range(n)]
    # 30 compact 2x2 components — a 16-cap would truncate; we must not
    for i in range(30):
        y, x = 2 + (i // 6) * 6, 2 + (i % 6) * 10
        for dy in range(2):
            for dx in range(2):
                rows[y + dy][x + dx] = 3
    gen = ClickGenerator()
    t1 = gen.tier1(rows)
    assert len(t1) == 30
    # snap: ring component whose centroid is off-color must land on own color
    ring = [[0] * 9 for _ in range(9)]
    for i in range(2, 7):
        ring[2][i] = ring[6][i] = ring[i][2] = ring[i][6] = 4
    t1 = gen.tier1(ring)
    assert len(t1) == 1
    x, y = t1[0]
    assert ring[y][x] == 4


def test_tier3_lattice_estimator():
    g = np.full((64, 64), 5, dtype=np.int8)
    g[::4, :] = 1     # gridlines every 4 px
    g[:, ::4] = 1
    py = _estimate_pitch(g, axis=0)
    px = _estimate_pitch(g, axis=1)
    assert py is not None and py[0] == 4
    assert px is not None and px[0] == 4
    out: list = []
    ClickGenerator().tier3(g, out)
    assert len(out) >= 100                      # a real lattice of cell centers
    xs = sorted({p[0] for p in out})
    assert all(b - a == 4 for a, b in zip(xs, xs[1:]))   # 4-px pitch
    assert _estimate_pitch(np.random.default_rng(0).integers(
        0, 9, (64, 64)).astype(np.int8), axis=0) is None


# --------------------------------------------------------------------------
# learned pruning memories
# --------------------------------------------------------------------------

def test_dead_click_memory():
    dead = DeadClickMemory(k=3)
    for skey in ("k1", "k2"):
        dead.record((5, 5), skey, changed=False)
    assert not dead.is_dead((5, 5))            # only 2 distinct states
    dead.record((5, 5), "k2", changed=False)   # duplicate state: still 2
    assert not dead.is_dead((5, 5))
    dead.record((5, 5), "k3", changed=False)
    assert dead.is_dead((5, 5))                # 3 distinct no-op states
    dead.record((7, 7), "k1", changed=True)    # one real effect...
    for skey in ("k2", "k3", "k4", "k5"):
        dead.record((7, 7), skey, changed=False)
    assert not dead.is_dead((7, 7))            # ...protects forever
    assert dead.prune([(5, 5), (7, 7)]) == [(7, 7)]
    # static-cell veto: a position whose cell EVER changed is never dead,
    # however many no-ops accumulated (the vc33-L4 false-kill fix)
    ever = np.zeros((10, 10), dtype=bool)
    assert dead.is_dead((5, 5), ever)          # cell never changed: dead
    ever[5, 5] = True
    assert not dead.is_dead((5, 5), ever)      # dynamic cell: protected
    assert dead.prune([(5, 5)], ever) == [(5, 5)]


def test_change_memory_and_t4_pruning():
    mem = ChangeMemory()
    f0 = np.zeros((16, 16), dtype=np.int8)
    f1 = f0.copy()
    f1[4, 8] = 7
    f1[6, 2] = 3
    mem.observe(f0)
    mem.observe(f1)
    mem.observe(f0)                            # changing back still counts
    assert mem.ever_changed[4, 8] and mem.ever_changed[6, 2]
    assert mem.ever_changed.sum() == 2
    out: list = []
    ClickGenerator().tier4(f1, mem, out)
    assert (8, 4) in out                       # changed cell, on T4 stride
    assert all(mem.ever_changed[y, x] for x, y in out), \
        "never-changed-any-frame cells must be pruned from T4"


# --------------------------------------------------------------------------
# reset-replay cost model
# --------------------------------------------------------------------------

def _linear_mock(depth=4):
    A1 = ("S", 1)
    states = {f"s{i}": ([[i, 0]], 0) for i in range(depth + 1)}
    transitions = {(f"s{i}", A1): f"s{i+1}" for i in range(depth)}
    return MockEnv(states, transitions, "s0", avail=[1]), A1


def test_reset_replay_cost_model():
    env, A1 = _linear_mock()
    be = ResetReplayBackend(env)
    root = be.root()
    assert (be.resets, be.actions_spent) == (1, 1)
    # test 1 action at depth 0: 1 reset + 0 replay + 1 action
    (tok, child), = list(be.children(root, [A1]))
    assert child is not None
    assert (be.resets, be.actions_spent) == (2, 3)


def test_reset_replay_cost_model_depth():
    env, A1 = _linear_mock()
    be = ResetReplayBackend(env)
    root = be.root()
    (_, c1), = list(be.children(root, [A1]))
    (_, c2), = list(be.children(c1, [A1]))
    spent_before = be.actions_spent
    resets_before = be.resets
    (_, c3), = list(be.children(c2, [A1]))          # depth d=2
    assert be.resets - resets_before == 1
    assert be.actions_spent - spent_before == 1 + 2 + 1   # reset + d + 1


def test_reset_replay_rollout_one_reset():
    env, A1 = _linear_mock(depth=10)
    be = ResetReplayBackend(env)
    root = be.root()
    (_, c1), = list(be.children(root, [A1]))
    resets_before = be.resets
    traj, final = be.rollout(c1, choose=lambda obs, prev: A1, k=5,
                             momentum=0.9, rng=random.Random(0))
    assert be.resets - resets_before == 1, "rollout must amortize to ONE reset"
    assert len(traj) == 5 and final.depth == 6
    assert all(len(t) == 3 for t in traj)   # (tok, obs, snap-handle|None)


def test_snapshot_rollout_momentum_repeat():
    env, A1 = _linear_mock(depth=30)
    be = SnapshotBackend(env)
    root = be.root()
    calls = []

    def choose(obs, prev):
        calls.append(1)
        return A1

    traj, final = be.rollout(root, choose, k=20, momentum=0.9,
                             rng=random.Random(1))
    assert len(traj) == 20
    assert len(calls) < 8, "momentum repeat must reuse the previous action"


# --------------------------------------------------------------------------
# stage 3a: run-length macros MUST emit intermediate states (ls20 regression)
# --------------------------------------------------------------------------

def _chain_env(n, branch_from=None, branch_tok=None):
    """s0 -A1-> s1 -A1-> ... -A1-> s{n} (then A1 stops changing); optional
    win branch from an INTERMEDIATE state."""
    A1 = ("S", 1)
    states = {f"s{i}": ([[i + 1, 0]], 0) for i in range(n + 1)}
    states["win"] = ([[15, 15]], 1)
    transitions = {(f"s{i}", A1): f"s{i+1}" for i in range(n)}
    if branch_from is not None:
        transitions[(f"s{branch_from}", branch_tok)] = "win"
    return MockEnv(states, transitions, "s0", avail=[1, 2])


def test_macro_emits_intermediate_states_ls20_regression():
    """The probe's mnbfs ran macros WITHOUT emitting intermediates and
    falsely exhausted ls20-L2. Here the win branches off a state that the
    macro run passes through: it must exist as an expandable node."""
    A1, A2 = ("S", 1), ("S", 2)
    env = _chain_env(3, branch_from=2, branch_tok=A2)   # win = A1 A1 A2
    core = make_core(env)
    core.use_macros = True
    res = core.solve_level(target=1, budget_s=10, max_tier=1)
    assert res["solved"], res
    assert res["handle"].path == [A1, A1, A2]
    # s2/s3 were first discovered BY the macro extension of root's A1 child
    assert core.stats["macro_nodes"] >= 2
    # env-integrity: replay the found path on a FRESH env and verify the win
    env2 = _chain_env(3, branch_from=2, branch_tok=A2)
    from arcengine import GameAction

    obs = env2.reset()
    for tok in res["handle"].path:
        obs = env2.step(GameAction.from_id(tok[1]))
    assert obs.levels_completed == 1, \
        "macro-built path does not replay — env corruption in extension"


def test_macro_stops_on_unchanged_and_stays_complete():
    A1 = ("S", 1)
    env = _chain_env(20)                     # 21 states, no win
    core = make_core(env)
    core.use_macros = True
    res = core.nbfs_level(target=1, budget_s=10, tier=1)
    assert not res["solved"]
    assert res["reason"] == "exhausted"      # terminates (no infinite macro)
    assert res["states"] == 21               # complete: every state emitted
    # first extension capped at MACRO_CAP-1 = 11 new nodes, second run adds
    # the remaining 7 from the next frontier state
    assert core.stats["macro_nodes"] == 18


def test_macros_do_not_regress_plain_solutions():
    A1, A2 = ("S", 1), ("S", 2)
    env = _chain_env(3, branch_from=2, branch_tok=A2)
    core = make_core(env)                    # macros OFF
    res = core.solve_level(target=1, budget_s=10, max_tier=1)
    assert res["solved"] and res["handle"].path == [A1, A1, A2]
    assert core.stats["macro_nodes"] == 0


# --------------------------------------------------------------------------
# stage 3b: composite ignition probes fire ONLY at inert roots
# --------------------------------------------------------------------------

class LatchEnv:
    """Inert root: every single action is a frame no-op. The frame changes
    only for the composite click(c1) THEN click(c2) — the sc25 class."""

    def __init__(self, frame, c1, c2):
        self.frame = frame
        self.c1, self.c2 = c1, c2
        self.latch = False
        self.won = False

    def _obs(self):
        from arcengine import GameState

        if self.won:
            return MockObs([[15] * len(self.frame[0])
                            for _ in self.frame], GameState.WIN, 1, [6])
        return MockObs([row[:] for row in self.frame],
                       GameState.NOT_FINISHED, 0, [6])

    def reset(self):
        self.latch = False
        self.won = False
        return self._obs()

    def step(self, action, data=None, reasoning=None):
        if int(action.value) == 6:
            xy = (int(data["x"]), int(data["y"]))
            if xy == self.c1:
                self.latch = True          # frame unchanged: single = no-op
            elif xy == self.c2 and self.latch:
                self.won = True
        return self._obs()


def _two_comp_frame():
    frame = [[0] * 16 for _ in range(16)]
    for y in range(2, 4):
        for x in range(2, 4):
            frame[y][x] = 3               # comp A
    for y in range(10, 12):
        for x in range(10, 12):
            frame[y][x] = 5               # comp B
    return frame


def test_ignition_probe_cracks_inert_root():
    frame = _two_comp_frame()
    t1, t2 = ClickGenerator().tier1(frame)[:2]
    env = LatchEnv(frame, t1, t2)
    core = make_core(env)
    res = core.solve_level(target=1, budget_s=10, max_tier=4)
    assert res["solved"], res
    assert res.get("ignition") is True
    assert core.stats["ignition_pairs"] >= 1
    assert res["handle"].path == [("C", *t1), ("C", *t2)]


def test_ignition_does_not_fire_on_live_root():
    """Root NOT inert (a click changes the frame): ignition must stay off."""
    frame = _two_comp_frame()
    t1, t2 = ClickGenerator().tier1(frame)[:2]
    alt = [row[:] for row in frame]
    alt[2][2] = 9                          # toggled pixel
    states = {"s0": (frame, 0), "s1": (alt, 0)}
    transitions = {("s0", ("C", *t1)): "s1", ("s1", ("C", *t1)): "s0"}
    env = MockEnv(states, transitions, "s0", avail=[6])
    core = make_core(env)
    res = core.solve_level(target=1, budget_s=10, max_tier=4)
    assert not res["solved"]
    assert core.stats["ignition_pairs"] == 0, \
        "ignition must fire ONLY when no single action changes the root"


# --------------------------------------------------------------------------
# stage 4a: go-explore archive — replacement rule + coarse switch
# --------------------------------------------------------------------------

def test_archive_shorter_trajectory_replacement():
    gm = np.zeros((4, 4), dtype=np.int8)
    long_h = Handle(None, 9, [("S", 1)] * 9)
    short_h = Handle(None, 4, [("S", 1)] * 4)
    longer_h = Handle(None, 12, [("S", 1)] * 12)
    archive: dict = {}
    assert GoExplorer.archive_insert(archive, "k", long_h, gm) is True
    archive["k"][2] = 3                            # 3 visits accumulated
    assert GoExplorer.archive_insert(archive, "k", short_h, gm) is False
    assert archive["k"][0] is short_h and archive["k"][3] == 4
    assert archive["k"][2] == 3, "replacement must KEEP the visit count"
    GoExplorer.archive_insert(archive, "k", longer_h, gm)
    assert archive["k"][0] is short_h, "longer trajectory must not replace"


def test_coarse_cell_key_groups_and_separates():
    def grid_with_comp(y, x, color):
        g = np.zeros((16, 16), dtype=np.int8)
        g[y:y + 2, x:x + 2] = color
        return g

    a = coarse_cell_key(0, grid_with_comp(8, 8, 3))
    b = coarse_cell_key(0, grid_with_comp(9, 9, 3))   # same bbox//4 bucket
    c = coarse_cell_key(0, grid_with_comp(8, 8, 5))   # recolored
    d = coarse_cell_key(1, grid_with_comp(8, 8, 3))   # other level
    assert a == b
    assert a != c and a != d


def test_goexplore_coarse_switch_past_threshold():
    """Exact archive outgrows coarse_switch -> auto-rebuild under coarse
    component-multiset keys."""
    A1 = ("S", 1)
    states = {}
    for i in range(13):
        f = [[0] * 8 for _ in range(8)]
        f[1 + (i % 6)][1] = 3              # comp slides: 13 exact states
        f[7][7] = (i % 9) + 1              # distinct exact hash for each
        states[f"s{i}"] = (f, 0)
    transitions = {(f"s{i}", A1): f"s{i+1}" for i in range(12)}
    env = MockEnv(states, transitions, "s0", avail=[1])
    core = make_core(env)
    go = GoExplorer(core, coarse_switch=5, max_cells=50, k_rollout=6,
                    rng_seed=0)
    res = go.explore_level(target=1, budget_s=2.0)
    assert not res["solved"]
    assert res["coarse"] is True, "archive never switched to coarse cells"
    assert res["states"] <= 13


def test_goexplore_solves_chain_and_path_replays():
    env = _chain_env(6, branch_from=6, branch_tok=("S", 1))  # win at depth 7
    core = make_core(env)
    go = GoExplorer(core, k_rollout=10, rng_seed=0)
    res = go.explore_level(target=1, budget_s=10.0)
    assert res["solved"], res
    from arcengine import GameAction

    env2 = _chain_env(6, branch_from=6, branch_tok=("S", 1))
    obs = env2.reset()
    for tok in res["handle"].path:
        obs = env2.step(GameAction.from_id(tok[1]))
    assert obs.levels_completed == 1


# --------------------------------------------------------------------------
# stage 4c: racer promotion logic + frame-0 archetype dispatch
# --------------------------------------------------------------------------

def test_racer_promotion_logic():
    order = ["nbfs_macros", "nbfs", "goexplore"]
    # plain best-2: drops the worst
    assert sh_promote({"nbfs_macros": 1.0, "nbfs": 5.0, "goexplore": 3.0},
                      order, 2) == ["nbfs", "goexplore"]
    # ties broken by dispatch order
    assert sh_promote({"nbfs_macros": 2.0, "nbfs": 2.0, "goexplore": 2.0},
                      order, 2) == ["nbfs_macros", "nbfs"]
    assert sh_promote({"nbfs_macros": 0.0, "nbfs": 9.0, "goexplore": 9.0},
                      order, 1) == ["nbfs"]
    # output preserves dispatch order regardless of score order
    assert sh_promote({"nbfs_macros": 5.0, "nbfs": 1.0, "goexplore": 9.0},
                      order, 2) == ["nbfs_macros", "goexplore"]


def test_racer_banks_probe_solve():
    from search_core import portfolio_race
    import time as _time

    env = _chain_env(2, branch_from=2, branch_tok=("S", 2))
    core = make_core(env)
    go = GoExplorer(core, rng_seed=0)
    winner, log, solved = portfolio_race(
        core, go, 1, ["nbfs", "goexplore"], t1=5.0,
        deadline=_time.time() + 30, max_tier=1)
    assert winner == "nbfs"
    assert solved is not None and solved["solved"], \
        "a probe that solves the level must end the race with the result"


def test_archetype_frame0_dispatch():
    assert archetype_frame0([6]) == "CLICK"
    assert archetype_frame0([1, 2, 3, 4]) == "AVATAR"
    assert archetype_frame0([1, 2, 3, 4, 5]) == "AVATAR"
    assert archetype_frame0([3, 4, 6, 7]) == "MIXED"
    assert archetype_frame0([1, 2, 3, 4, 5, 6]) == "MIXED"
    assert archetype_frame0([]) == "MIXED"
    assert "nbfs_macros" not in DISPATCH_ORDER["CLICK"]


# --------------------------------------------------------------------------
# transposition key parity with the shipped FrontierGraph machinery
# --------------------------------------------------------------------------

def test_crc_key_matches_frontier_graph_node_key():
    rng = np.random.default_rng(7)
    graph = FrontierGraph()
    for _ in range(20):
        g = rng.integers(0, 16, (16, 16)).astype(np.int8)
        cells = [(int(rng.integers(0, 16)), int(rng.integers(0, 16)))
                 for _ in range(5)]
        level = int(rng.integers(0, 9))
        assert crc_key(level, g, mask_to_bool(cells, g.shape)) == \
            graph.node_key(level, g.tolist(), cells)
    assert crc_key(3, np.zeros((8, 8), dtype=np.int8), None) == \
        graph.node_key(3, [[0] * 8 for _ in range(8)], [])


# --------------------------------------------------------------------------
# real engine: determinism + backend equivalence (small game: tu93 L1)
# --------------------------------------------------------------------------

def _make_env(stem="tu93"):
    import logging

    logging.disable(logging.CRITICAL)
    from arc_agi import Arcade, OperationMode

    games = discover_games(ENVS)
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENVS)
    env = client.make(games[stem])
    env.reset()
    return env


def test_snapshot_determinism_real_engine():
    from arcengine import GameAction

    env = _make_env("tu93")
    env.reset()
    seq = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION4,
           GameAction.ACTION3, GameAction.ACTION2]
    for a in seq[:2]:
        env.step(a)
    twin = copy.deepcopy(env)
    for a in seq[2:]:
        o1 = env.step(a)
        o2 = twin.step(a)
        assert np.array_equal(settled(o1), settled(o2)), \
            "deepcopy twin diverged — snapshot backend assumption broken"
        assert o1.state == o2.state and o1.levels_completed == o2.levels_completed


@pytest.mark.slow
def test_backend_equivalence_tu93_l1():
    """Both backends must find the SAME winning sequence for tu93 level 1."""
    paths = {}
    for kind in ("snapshot", "reset_replay"):
        core = SearchCore(_make_env("tu93"), backend=kind)
        core.warmup_and_freeze()
        res = core.solve_level(target=1, budget_s=120, max_tier=4)
        assert res["solved"], f"{kind} failed to solve tu93 L1: {res}"
        paths[kind] = res["handle"].path
    assert paths["snapshot"] == paths["reset_replay"], \
        f"backends diverged: {paths}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
