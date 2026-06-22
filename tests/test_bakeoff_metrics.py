import math
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
