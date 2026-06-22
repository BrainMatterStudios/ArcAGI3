"""CausalProbeExplorer firewall + behavior tests.

The load-bearing guarantee: enable_causal=False is byte-identical to TransferExplorer (== v13),
and structural promotion is strictly additive (promote only, never demote/remove a candidate).
"""
import numpy as np

from arcagi3.causal_probe_explorer import CausalProbeExplorer
from arcagi3.transfer_explorer import TransferExplorer


def _run(pol, seed=0, steps=60, levelup_at=30):
    rng = np.random.default_rng(seed)
    out = []
    lvl = 0
    for t in range(steps):
        g = rng.integers(0, 6, size=(16, 16)).astype(np.int8)
        out.append(pol.decide(g, False, False, lvl, [1, 2, 3, 4, 5, 6]))
        if t == levelup_at:
            lvl = 1
    return out


def test_firewall_off_is_byte_identical_to_transfer():
    a = _run(CausalProbeExplorer(seed=0, trust_threshold=3, border_mask=2, enable_causal=False))
    b = _run(TransferExplorer(seed=0, trust_threshold=3, border_mask=2))
    assert a == b


def test_enabled_builds_certificates_without_crashing():
    c = CausalProbeExplorer(seed=0, trust_threshold=3, border_mask=2,
                            coarse_grid_step=4, max_click_targets=256)
    _run(c)
    # on random grids every simple action changes the multiset -> all become structural triggers
    assert c.certs.summary()["n_classes"] >= 1


def test_promotion_is_additive_never_drops_candidates():
    """Structural promotion must preserve the full candidate set (coverage), only change tiers."""
    c = CausalProbeExplorer(seed=1, trust_threshold=3, border_mask=2)
    c.bg = 0
    grid = np.zeros((16, 16), dtype=np.int8)
    grid[2, 2] = 4
    avail = [1, 2, 3, 4, 5, 6]
    base = {a for a, _t in TransferExplorer(seed=1, trust_threshold=3, border_mask=2)
            ._candidates(grid, avail)}
    c.struct_triggers = {("S", 3)}
    promoted = c._candidates(grid, avail)
    assert {a for a, _t in promoted} == base               # same candidate set (no drop)
    assert dict(promoted)[("S", 3)] == 0                   # the certified class is promoted
