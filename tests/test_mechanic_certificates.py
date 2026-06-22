"""Tests for MechanicCertificateStore — certificates are proof-from-witnessed-transitions only."""
import numpy as np

from arcagi3.mechanic_certificates import MechanicCertificateStore


def _grid(fill=0):
    return np.full((8, 8), fill, dtype=np.int64)


def test_proven_noop_requires_min_samples():
    s = MechanicCertificateStore(noop_min_samples=4)
    g = _grid()
    # ACTION1 never changes the frame
    for _ in range(4):
        s.observe(("S", 1), g, g, level_up=False)
    assert s.is_proven_noop(("S", 1))
    # only 3 samples -> not yet certified
    s2 = MechanicCertificateStore(noop_min_samples=4)
    for _ in range(3):
        s2.observe(("S", 1), g, g, level_up=False)
    assert not s2.is_proven_noop(("S", 1))


def test_a_single_change_blocks_noop_certificate():
    s = MechanicCertificateStore(noop_min_samples=4)
    g0, g1 = _grid(0), _grid(0)
    g1[0, 0] = 5
    for _ in range(5):
        s.observe(("S", 2), g0, g0, level_up=False)
    s.observe(("S", 2), g0, g1, level_up=False)  # one real change
    assert not s.is_proven_noop(("S", 2))


def test_level_trigger_certificate():
    s = MechanicCertificateStore()
    g0, g1 = _grid(0), _grid(0)
    g1[1, 1] = 7
    assert not s.is_level_trigger(("S", 5))
    s.observe(("S", 5), g0, g1, level_up=True)
    assert s.is_level_trigger(("S", 5))
    assert ("S", 5) in s.level_triggers()


def test_click_action_class_uses_color_signature():
    s = MechanicCertificateStore()
    g = _grid(0)
    g[3, 4] = 9  # row 3, col 4
    # token is ("C", col, row) = ("C", 4, 3) -> color at (row=3,col=4)=9
    key = s.action_class(("C", 4, 3), g)
    assert key == ("C", 9)


def test_structural_certificate_on_large_delta():
    s = MechanicCertificateStore(structural_min_cells=10.0)
    g0 = _grid(0)
    g1 = _grid(0); g1[:4, :] = 3  # 32 cells change
    s.observe(("C", 4, 3), g0, g1, level_up=False)
    s.observe(("C", 4, 3), g0, g1, level_up=False)
    assert s.is_structural(("C", 0))  # color at (3,4) in g0 is 0


def test_no_certificate_for_out_of_bounds_click():
    s = MechanicCertificateStore()
    g = _grid(0)
    assert s.action_class(("C", 99, 99), g) is None
    assert s.observe(("C", 99, 99), g, g, level_up=False) is None
