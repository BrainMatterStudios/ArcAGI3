"""The sc25 glyph-cast solver must crack L0: draw the diamond glyph (shrink the avatar), then navigate to
the goal. Validates the glyph-draw+cast mechanic (prime click + 4 diamond slots) and scale-aware navigation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import sc25_solve as S


def test_sc25_glyphcast_cracks_l0():
    assert S.solve(verbose=False) >= 1, "sc25 solver should crack L0 (glyph-cast + navigate)"
