import numpy as np

from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer


def _trace(policy, n=400):
    rng = np.random.default_rng(123)
    out = []
    levels = 0
    for i in range(n):
        grid = rng.integers(0, 4, size=(16, 16)).astype(np.int8)
        tok = policy.decide(
            grid=grid, gstate_terminal=False, gstate_notplayed=False,
            levels=levels, available=[1, 2, 3, 4, 5, 6],
        )
        out.append(tok)
    return out


def test_disabled_curiosity_is_byte_identical_to_transfer():
    base = _trace(TransferExplorer(seed=0))
    off = _trace(ObjectCuriosityExplorer(seed=0, enable_object_curiosity=False))
    assert base == off


def test_score_prefers_unseen_then_rarest():
    eng = ObjectCuriosityExplorer(seed=0)
    eng.reset_all()
    sig_rare, sig_common = ("S", 1, "appeared", frozenset({3})), ("S", 2, "moved", frozenset({4}))
    eng.descriptor_apps[("S", 1)] = 1
    eng.descriptor_sigs[("S", 1)] = {sig_rare}
    eng.descriptor_apps[("S", 2)] = 1
    eng.descriptor_sigs[("S", 2)] = {sig_common}
    eng.sig_counts[sig_rare] = 1
    eng.sig_counts[sig_common] = 9
    assert eng._curiosity_score(("S", 1)) > eng._curiosity_score(("S", 2))
    assert eng._curiosity_score(("S", 9)) > eng._curiosity_score(("S", 1))  # unseen wins
