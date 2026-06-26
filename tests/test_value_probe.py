import numpy as np

from arcagi3.world_model import WorldModel
from scripts.value_probe import build_labeled_rows, encode_graph_state, fit_and_report


def test_encode_graph_state_uses_graph_observables():
    wm = WorldModel()
    node = wm.observe(b"root", (("S", 1), ("S", 2)))
    wm.record(b"root", ("S", 1), b"a", 0.0)

    features = encode_graph_state(node, depth=3, reward_distance=2, reward_seen=1.0)

    assert np.allclose(features, np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 0.5]))


def test_build_labeled_rows_marks_reward_reachable_states_positive():
    wm = WorldModel()
    wm.observe(b"root", (("S", 1), ("S", 2)))
    wm.observe(b"a", (("S", 3),))
    wm.observe(b"b", ())
    wm.observe(b"dead", ())
    wm.record(b"root", ("S", 1), b"a", 0.0)
    wm.record(b"a", ("S", 3), b"b", 1.0)
    wm.record(b"root", ("S", 2), b"dead", 0.0)

    rows = build_labeled_rows(wm, root=b"root")
    by_key = {key: label for key, _features, label in rows}

    assert by_key[b"root"] == 1.0
    assert by_key[b"a"] == 1.0
    assert by_key[b"b"] == 1.0
    assert by_key[b"dead"] == 0.0


def test_build_labeled_rows_can_limit_positive_distance():
    wm = WorldModel()
    wm.observe(b"root", (("S", 1),))
    wm.observe(b"a", (("S", 2),))
    wm.observe(b"b", (("S", 3),))
    wm.observe(b"c", ())
    wm.record(b"root", ("S", 1), b"a", 0.0)
    wm.record(b"a", ("S", 2), b"b", 0.0)
    wm.record(b"b", ("S", 3), b"c", 1.0)

    rows = build_labeled_rows(wm, root=b"root", max_positive_distance=1)
    by_key = {key: label for key, _features, label in rows}

    assert by_key[b"c"] == 1.0
    assert by_key[b"b"] == 1.0
    assert by_key[b"a"] == 0.0
    assert by_key[b"root"] == 0.0


def test_fit_and_report_handles_large_scale_nuisance_features():
    rows = [
        (b"p1", np.array([1.0, 0.0, 0.0, 1000.0]), 1.0),
        (b"p2", np.array([1.0, 0.0, 1.0, 1200.0]), 1.0),
        (b"n1", np.array([0.0, 1.0, 0.0, 2000.0]), 0.0),
        (b"n2", np.array([0.0, 1.0, 1.0, 2200.0]), 0.0),
    ]

    report = fit_and_report(rows)

    assert report["separable"] is True
    assert report["mean_positive"] > report["mean_negative"]
