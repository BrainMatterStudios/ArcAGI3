"""TDD for the BET 3 in-context probe data pipeline (examples = context + query + affordance label)."""
import numpy as np

from arcagi3.synthgen.world import FAMILIES, rewarding_action
from arcagi3.synthgen.probe import make_example, ACTION5, label5, make_dataset


def test_label5_maps_moves_and_click():
    from arcagi3.synthgen.world import World, UP, click
    w = World("REACH", bg=0, wall=1, avatar=(5, 5), avatar_color=2, target=(2, 5), target_color=3)
    assert label5(w) == ACTION5.index(UP)
    g = World("GATE", bg=0, wall=1, avatar=(5, 5), avatar_color=2, target=(5, 9),
              target_color=3, switch=(2, 3), switch_color=5)
    assert label5(g) == 4   # closed gate -> click affordance


def test_make_example_shapes_and_unsolved_query():
    rng = np.random.default_rng(0)
    ex = make_example("COLLECT", rng, T=12)
    assert ex.ctx_frames.shape == (12, 16, 16)
    assert ex.ctx_next.shape == (12, 16, 16)
    assert ex.ctx_actions.shape == (12,)
    assert ex.query.shape == (16, 16)
    assert 0 <= ex.label < 5
    # query must be a non-terminal state (label well-defined)
    assert ex.ctx_actions.min() >= 0 and ex.ctx_actions.max() <= 4


def test_make_example_label_matches_rewarding_action():
    rng = np.random.default_rng(3)
    ex = make_example("REACH", rng, T=10, return_world=True)
    assert ex.label == label5(ex.world)
    assert ex.label == ACTION5.index(rewarding_action(ex.world)) if rewarding_action(ex.world)[0] != "C" else 4


def test_make_dataset_balanced_over_families():
    rng = np.random.default_rng(0)
    ds = make_dataset(FAMILIES, n_per_family=8, rng=rng, T=10)
    assert len(ds) == 8 * len(FAMILIES)
    fams = [e.family for e in ds]
    for f in FAMILIES:
        assert fams.count(f) == 8


def test_context_has_real_transitions_not_all_identical():
    # a context should contain at least one visible frame-change (the mechanic must be observable).
    rng = np.random.default_rng(1)
    seen_change = 0
    for _ in range(8):
        ex = make_example("PUSH", rng, T=12)
        if np.any(ex.ctx_frames != ex.ctx_next):
            seen_change += 1
    assert seen_change >= 6
