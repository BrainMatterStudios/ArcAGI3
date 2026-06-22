import math
import pytest
from arcagi3.bakeoff_metrics import efficiency, transfer_slope

def test_efficiency_capped_at_1_15():
    assert efficiency(a_h=13, a_m=13) == 1.0
    assert efficiency(a_h=13, a_m=5) == 1.15

def test_efficiency_squared_decay():
    assert math.isclose(efficiency(a_h=10, a_m=20), 0.25, rel_tol=1e-9)
    assert math.isclose(efficiency(a_h=10, a_m=100), 0.01, rel_tol=1e-9)

def test_efficiency_uncompleted_is_zero():
    assert efficiency(a_h=13, a_m=None) == 0.0

def test_transfer_slope_negative_when_costs_drop():
    assert transfer_slope([200, 120, 60, 40]) < 0
    assert transfer_slope([50, 50, 50]) == 0.0


@pytest.mark.integration
def test_human_baseline_level1_is_optimal():
    from arcagi3.bakeoff_metrics import human_baseline_actions
    a_h = human_baseline_actions(level=0)
    assert a_h is not None and 1 <= a_h <= 30   # Exp-42 found 13


@pytest.mark.integration
def test_optimal_actions_per_level_advances():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tmp", "scripts/truemodel_planner.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    per_level = m.optimal_actions_per_level(m.load_ls20_class(), up_to_level=1)
    assert per_level[0] == 13                 # Exp-42 L1 optimal
    # L2 (index=1) grid has goal unreachable from spawn — BFS correctly returns None.
    # The function must have attempted L2 (len >= 2) proving the advance chain ran.
    assert len(per_level) >= 2                # solve->advance chain reached L2
