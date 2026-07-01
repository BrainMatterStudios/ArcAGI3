"""sc25 SOURCE-FREE glyph-cast solver must crack L0 from frame perception only: read the glyph from the
spell-icon (minority color), toggle the slots to cast (avatar shrinks), navigate the small avatar to goal."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import sc25_sf as S


def test_sc25_sourcefree_cracks_l0():
    assert S.solve(verbose=False) >= 1, "sc25 source-free solver should crack L0"
