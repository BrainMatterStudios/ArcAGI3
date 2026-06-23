import numpy as np
from arcagi3.hybrid_transfer_explorer import HybridTransferExplorer
from arcagi3.transfer_explorer import TransferExplorer

def _seq():
    frames = []
    for i in range(6):
        g = np.zeros((10, 10), dtype=np.int8); g[2, 2 + (i % 3)] = 5; g[2, 8] = 9
        frames.append(g)
    return frames

def _run(explorer, frames):
    explorer.reset_all()
    toks = []
    for g in frames:
        toks.append(explorer.decide(g, gstate_terminal=False, gstate_notplayed=False,
                                    levels=0, available=[1, 2, 3, 4]))
    return toks

def test_firewall_guidance_off_matches_transfer_explorer():
    frames = _seq()
    hyb = HybridTransferExplorer(seed=0, enable_model_guidance=False)
    base = TransferExplorer(seed=0)
    assert _run(hyb, frames) == _run(base, frames)   # byte-identical trace with guidance off

def test_promotes_guide_suggestion(monkeypatch):
    hyb = HybridTransferExplorer(seed=0, enable_model_guidance=True)
    hyb.reset_all()
    g = np.zeros((10, 10), dtype=np.int8); g[2, 2] = 5
    # Test the promotion MECHANISM directly: set suggestion and check _candidates output
    hyb._guide_suggestion = 3
    cands = hyb._candidates(g, [1, 2, 3, 4])
    assert (("S", 3), 0) in cands   # the suggested simple action is promoted to tier 0
