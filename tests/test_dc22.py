"""dc22 SOURCE-FREE reach + bridge-panel solver must crack L0 via joint interleaved click+move search
(clicking panel buttons toggles linked bridges to walkable; the avatar navigates the opened path)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import dc22_solve as D


def test_dc22_sourcefree_cracks_l0():
    assert D.solve(verbose=False) >= 1, "dc22 source-free solver should crack L0"
