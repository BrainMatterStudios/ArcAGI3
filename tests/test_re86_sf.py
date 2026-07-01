"""re86 SOURCE-FREE solver must crack L0 from frame perception only (pieces vs targets split by frame-
adjacency; active piece = perceived piece nearest the cursor; drag the cursor-head over the target)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import re86_sf as R


def test_re86_sourcefree_cracks_l0():
    assert R.solve(verbose=False) >= 1, "re86 source-free solver should crack L0"
