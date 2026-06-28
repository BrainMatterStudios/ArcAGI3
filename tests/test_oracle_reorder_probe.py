"""Unit tests for the pure scoring math of the BET 2 oracle-reorder ceiling probe."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "oracle_reorder_probe", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "oracle_reorder_probe.py")
orp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(orp)


def test_no_noops_zero_gain():
    # no no-op clicks -> oracle == actual -> zero gain on every axis.
    res = orp.oracle_gain([(200, 0), (400, 0)])
    assert res["weighted_gain"] == 0.0
    assert res["unweighted_gain"] == 0.0


def test_empty_is_inert():
    res = orp.oracle_gain([])
    assert res["n_levels"] == 0 and res["weighted_gain"] == 0.0


def test_single_level_removes_noops():
    # one level, 200 actions of which 100 are no-op clicks: actual S=min(1.15,(200/200)^2)=1.0,
    # oracle S=min(1.15,(200/100)^2)=1.15 (capped). weighted == unweighted with a single level.
    res = orp.oracle_gain([(200, 100)])
    assert abs(res["unweighted_actual"] - 1.0) < 1e-9
    assert abs(res["unweighted_oracle"] - 1.15) < 1e-9
    assert abs(res["unweighted_gain"] - 0.15) < 1e-9
    assert abs(res["weighted_gain"] - 0.15) < 1e-9


def test_level_weighting_favours_deep_levels():
    # gain concentrated on the deep (heavily-weighted) level should weight UP vs unweighted mean.
    # L1 cheap+clean (no gain), L2 expensive with big no-op savings.
    levels = [(100, 0), (1000, 800)]  # L2: actual S=(200/1000)^2=0.04, oracle (200/200)^2=1.0
    res = orp.oracle_gain(levels)
    # unweighted gain = (1.0 - 0.04)/... summed = 0.96; weighted divides by (1+2)=3, weights L2 by 2.
    assert res["unweighted_gain"] > 0
    # weighted = (1*0 + 2*0.96)/3 = 0.64; deep-level gain dominates.
    assert abs(res["weighted_gain"] - (2 * 0.96) / 3) < 1e-3


def test_oracle_never_below_one_action():
    # removing more no-ops than actions can't drive actions below 1 (guard).
    res = orp.oracle_gain([(5, 99)])
    assert res["unweighted_oracle"] == orp.s_level(1)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"\n{len(fns)} passed")
    sys.exit(0)
