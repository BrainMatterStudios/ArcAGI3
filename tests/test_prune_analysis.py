# tests/test_prune_analysis.py
import numpy as np
from arcagi3 import prune_analysis as A


def _steps():
    # level 0: detour a->d->a, then path a->b->c; c is the level-up (reward) state.
    # fields: idx, level, from_key, action, reward, tier
    return [
        A.Step(0, 0, b"a", ("S", 4), 0.0, 0),
        A.Step(1, 0, b"d", ("S", 1), 0.0, 9),
        A.Step(2, 0, b"a", ("S", 1), 0.0, 0),
        A.Step(3, 0, b"b", ("S", 2), 0.0, 0),
        A.Step(4, 0, b"c", ("S", 3), 1.0, 0),   # reward -> level up
        A.Step(5, 1, b"z", ("S", 1), 0.0, 0),   # first state of level 1
    ]


def test_build_edges_excludes_reset_and_levelup():
    edges, first_seen = A.build_edges(_steps())
    assert set(edges[b"a"]) == {(("S", 4), b"d"), (("S", 1), b"b")}
    assert edges[b"b"] == [(("S", 2), b"c")]
    assert b"c" not in edges  # step 4 is a reward edge -> excluded
    assert first_seen[b"a"] == 0 and first_seen[b"d"] == 1
    assert first_seen[b"b"] == 3 and first_seen[b"c"] == 4

def test_segment_and_ceilings():
    steps = _steps()
    edges, first_seen = A.build_edges(steps)
    segs = A.segment_levels(steps, first_seen)
    assert set(segs) == {0}                       # only level 0 ends in a reward here
    seg = segs[0]
    assert seg.start_key == b"a" and seg.target_key == b"c"
    assert seg.member_keys == {b"a", b"b", b"c", b"d"}
    assert seg.actual_actions == 4                # end_idx 4 - start_idx 0

    path = A.shortest_path(edges, seg.start_key, seg.target_key, seg.member_keys)
    assert path == [b"a", b"b", b"c"]

    c = A.ceilings(seg, path)
    assert c["reachable"] is True
    assert c["discovered_states"] == 4 and c["path_states"] == 3
    assert c["ceiling_states"] == 0.25
    assert c["path_actions"] == 2 and c["ceiling_actions"] == 0.5


def test_shortest_path_unreachable_returns_none():
    edges = {b"a": [(("S", 1), b"b")]}
    assert A.shortest_path(edges, b"a", b"x", {b"a", b"b"}) is None
    c = A.ceilings(A.LevelSeg(0, b"a", b"x", {b"a", b"b"}, 0, 3), None)
    assert c["reachable"] is False and c["ceiling_actions"] is None

def test_state_features():
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[0:2, 0:2] = 3          # one 4-cell object, color 3
    grid[10, 10] = 5            # one 1-cell object, color 5
    f = A.state_features(grid, background=0, discovery_tier=7, parent_colors={3})
    assert f["n_objects"] == 2.0
    assert f["n_small"] == 2.0           # both objects size <= 4
    assert f["max_obj_size"] == 4.0
    assert f["n_distinct_colors"] == 2.0  # {3,5}
    assert f["discovery_tier"] == 7.0
    assert f["n_new_colors"] == 1.0       # 5 is new vs parent {3}
    assert list(A.FEATURE_ORDER)  # non-empty, stable order
    assert len(A.feature_vector(f)) == len(A.FEATURE_ORDER)

def test_roc_auc_basic():
    assert A.roc_auc([0.1, 0.9], [0, 1]) == 1.0
    assert A.roc_auc([0.9, 0.1], [0, 1]) == 0.0
    assert A.roc_auc([0.5, 0.5], [0, 1]) == 0.5      # ties -> 0.5


def test_logo_auc_separable_vs_shuffle():
    rng = np.random.default_rng(0)
    per_game = {}
    for g in ("g1", "g2", "g3"):
        # feature 0 separates: on-path high, off-path low; other features noise.
        Xpos = np.column_stack([rng.normal(3, 0.3, 40), rng.normal(0, 1, 40)])
        Xneg = np.column_stack([rng.normal(0, 0.3, 60), rng.normal(0, 1, 60)])
        X = np.vstack([Xpos, Xneg])
        y = np.array([1] * 40 + [0] * 60)
        per_game[g] = (X, y)
    auc = A.logo_auc(per_game)
    sh = A.shuffle_auc(per_game, seed=0)
    assert auc >= 0.9                 # held-out separability is real
    assert abs(sh - 0.5) < 0.15       # shuffle control collapses to chance


def _reset_steps():
    # Intro frame I is disconnected: I->A, then RESET to root R, then R->B->T (level-up).
    # The shortest productive path must start from the reset-root R, not the first frame I.
    return [
        A.Step(0, 0, b"I", ("S", 1), 0.0, 0),
        A.Step(1, 0, b"A", ("reset",), 0.0, 0),
        A.Step(2, 0, b"R", ("S", 1), 0.0, 0),
        A.Step(3, 0, b"B", ("S", 2), 0.0, 0),
        A.Step(4, 0, b"T", ("S", 3), 1.0, 0),   # reward -> level up
        A.Step(5, 1, b"z", ("S", 1), 0.0, 0),
    ]


def test_level_entries_includes_reset_root():
    steps = _reset_steps()
    edges, first_seen = A.build_edges(steps)
    seg = A.segment_levels(steps, first_seen)[0]
    assert seg.start_key == b"I" and seg.target_key == b"T"
    entries = A.level_entries(steps, seg)
    assert entries == [b"I", b"R"]   # first frame + the post-reset root


def test_best_path_uses_reset_root_when_first_frame_disconnected():
    steps = _reset_steps()
    edges, first_seen = A.build_edges(steps)
    seg = A.segment_levels(steps, first_seen)[0]
    # The old single-start path is unreachable from the disconnected first frame:
    assert A.shortest_path(edges, seg.start_key, seg.target_key, seg.member_keys) is None
    # best_path recovers it via the reset-root entry:
    path, entry = A.best_path(edges, A.level_entries(steps, seg), seg.target_key, seg.member_keys)
    assert entry == b"R" and path == [b"R", b"B", b"T"]
    c = A.ceilings(seg, path)
    assert c["reachable"] is True and c["ceiling_actions"] == 0.5


def test_near_optimal_states_union_of_shortest_paths():
    steps = _reset_steps()
    edges, first_seen = A.build_edges(steps)
    seg = A.segment_levels(steps, first_seen)[0]
    on = A.near_optimal_states(edges, b"R", seg.target_key, seg.member_keys, slack=0)
    assert on == {b"R", b"B", b"T"}   # all 3 productive states, none of I/A


def test_mlp_separates_and_logo_per_game():
    rng = np.random.default_rng(0)
    per_game = {}
    for g in ("g1", "g2", "g3"):
        Xpos = np.column_stack([rng.normal(3, 0.3, 40), rng.normal(0, 1, 40)])
        Xneg = np.column_stack([rng.normal(0, 0.3, 60), rng.normal(0, 1, 60)])
        per_game[g] = (np.vstack([Xpos, Xneg]), np.array([1] * 40 + [0] * 60))
    assert A.logo_auc_mlp(per_game) >= 0.85
    per = A.logo_auc_per_game(per_game)
    assert set(per) == {"g1", "g2", "g3"} and all(v >= 0.85 for v in per.values())
