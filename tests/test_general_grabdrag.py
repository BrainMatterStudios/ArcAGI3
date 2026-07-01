"""The GENERAL (source-free) grab-drag solver must solve wa30 using DETECTED roles — no hardcoded colors —
which is what makes it transfer to an unseen grab-drag game."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import general_grabdrag as G


def test_source_free_solver_cracks_wa30():
    levels = G.solve("wa30", verbose=False)
    assert levels >= 1, f"source-free grab-drag solver should crack wa30 L0, got {levels} levels"
