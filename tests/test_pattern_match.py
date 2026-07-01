"""The general source-free PATTERN-MATCH solver must crack sb26 L0 using perceived roles
(answer key + slots + palette), no hardcoded colors — so it transfers to a hidden pattern-match game."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import pattern_match as PM


def test_source_free_pattern_match_cracks_sb26():
    levels = PM.solve("sb26", verbose=False)
    assert levels >= 1, f"pattern-match solver should crack sb26 L0, got {levels}"
