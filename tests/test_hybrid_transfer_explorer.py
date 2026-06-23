import os
import pytest
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


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live lp85 bake-off; set RUN_BAKEOFF=1")
def test_hybrid_not_worse_than_transfer_on_lp85():
    """Safety guard: HybridTransferExplorer must never clear fewer levels or spend more total
    actions than the signature-transfer baseline on lp85. NOTE (Exp-46): on lp85 the model guide
    never fires (guide_fires=0) because lp85 is a pure CLICK game (available_actions==[6]) and the
    discovery model is movement-centric — so the hybrid degrades byte-identically to TransferExplorer.
    This guard therefore verifies SAFE DEGRADATION (no regression), not model value-add (which lp85
    cannot test). Budget 8000 is ample (transfer clears all 5 levels in ~4281 actions)."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("hb", "scripts/hybrid_bakeoff.py")
    hb = importlib.util.module_from_spec(spec); sys.modules["hb"] = hb; spec.loader.exec_module(hb)
    from arcagi3.transfer_explorer import TransferExplorer
    from arcagi3.hybrid_transfer_explorer import HybridTransferExplorer
    _, tlev, tper, _ = hb.run_arm("transfer", TransferExplorer(seed=0), 8000)
    _, hlev, hper, _ = hb.run_arm("hybrid", HybridTransferExplorer(seed=0), 8000)
    assert hlev >= tlev and sum(a for _, a in hper) <= sum(a for _, a in tper)
