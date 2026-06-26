from scripts.interaction_coverage_probe import compare_rows


def test_compare_rows_flags_more_coverage():
    rows = [("ls20", 5, 9), ("re86", 4, 4)]  # (game, base_distinct, curiosity_distinct)
    summary = compare_rows(rows)
    assert summary["more_coverage"] == 1   # ls20 only
    assert summary["worse_coverage"] == 0
