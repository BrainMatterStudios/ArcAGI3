from scripts.curiosity_reward_edge_probe import gate1_summary


def test_gate1_passes_when_curiosity_unlocks_a_wall_game():
    rows = [("ls20", 0, 1), ("re86", 0, 0)]  # (game, base_pos_edges, curiosity_pos_edges)
    summary = gate1_summary(rows)
    assert summary["unlocked"] == 1
    assert summary["pass"] is True


def test_gate1_fails_when_both_stay_zero():
    rows = [("ls20", 0, 0), ("re86", 0, 0)]
    summary = gate1_summary(rows)
    assert summary["unlocked"] == 0
    assert summary["pass"] is False
