"""re86 paint-to-stencil solver must crack L0: drag each single-color piece over its matching-color target
cells so the composited trail matches the stencil."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import re86_solve as R


def test_re86_paint_cracks_l0():
    assert R.solve(verbose=False) >= 1, "re86 solver should crack L0 (paint-to-stencil)"
