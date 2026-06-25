from arcagi3.world_model import WorldModel

from scripts.reward_edge_probe import summarize_reward_paths


def test_summarize_reward_paths_counts_positive_edges():
    wm = WorldModel()
    wm.observe(b"root", (("S", 1), ("S", 2)))
    wm.observe(b"a", (("S", 3),))
    wm.observe(b"b", ())
    wm.observe(b"c", ())
    wm.record(b"root", ("S", 1), b"a", 0.0)
    wm.record(b"root", ("S", 2), b"b", 1.0)
    wm.record(b"a", ("S", 3), b"c", 0.0)

    summary = summarize_reward_paths(wm, root=b"root")

    assert summary["positive_edges"] == 1
    assert summary["reachable_positive_nodes"] == 1
    assert summary["shortest_reward_distance"] == 1
