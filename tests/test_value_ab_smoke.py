from scripts.value_ab import summarize_rows


def test_summarize_rows_counts_improvements():
    rows = [
        ("g1", 1, 2),
        ("g2", 3, 3),
        ("g3", 2, 1),
    ]

    summary = summarize_rows(rows)

    assert summary["improved"] == 1
    assert summary["regressed"] == 1
