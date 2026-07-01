"""The meta-solver (archetype router) must crack the archetype games AND abstain safely on others — the
regression-safety property that makes it wirable as an additive portfolio play without risking the 0.33 floor.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "research_2026_07_01"))
import meta_solver as MS


def test_cracks_grabdrag_game():
    arch, lv = MS.route("wa30")
    assert arch == "grab-drag" and lv >= 1, f"expected grab-drag crack, got {(arch, lv)}"


def test_cracks_patternmatch_game():
    arch, lv = MS.route("sb26")
    assert arch == "pattern-match" and lv >= 1, f"expected pattern-match crack, got {(arch, lv)}"


def test_abstains_safely_on_non_archetype_games():
    for gm in ("tu93", "dc22"):
        arch, lv = MS.route(gm)
        assert arch == "abstain" and lv == 0, f"{gm} should abstain (no false crack), got {(arch, lv)}"
