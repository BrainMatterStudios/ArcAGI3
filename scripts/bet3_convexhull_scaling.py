"""CONVEX-HULL SCALING PROBE (2026-06-28) — the decisive gate for the offline rich-prior program.

BET 3 (4 families, 1 click family=GATE) found: held-out GATE's CLICK affordance = HARD 0.000, because no
other click-reward family was in training. The convex-hull hypothesis: with MORE affordance-diverse
families in training, a held-out family's affordance becomes INTERPOLABLE (in the convex hull of seen
affordances) → held-out accuracy rises. If it stays flat at chance as families scale, the offline-prior
program is capped near 0.33 (ARC's adversarially-novel mechanics are out-of-support no matter the scale).

Design: for family-count N in {4,6,8}, leave-one-mechanic-family-out over the first N families; track
mean in-distribution acc (validity), mean held-out acc, and — the decisive signal — the CLICK-labeled
held-out accuracy on the click families (GATE/TOGGLE) as a function of how many OTHER click families are in
training. Now there are TWO click families (GATE, TOGGLE), so holding out one leaves the other in training.

Pre-registered: if held-out CLICK-labeled acc stays ~chance (≈0) even with another click family in training
→ convex-hull CLOSED (KILL the offline-prior program). If it rises clearly → OPEN (green-light a large
curriculum build).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/bet3_convexhull_scaling.py [n_per] [epochs] [T]
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

from arcagi3.synthgen.world import FAMILIES
from arcagi3.synthgen.probe import make_dataset

# load the model + training helpers from the BET 3 probe script
_p = pathlib.Path(__file__).resolve().parent / "bet3_incontext_probe.py"
_spec = importlib.util.spec_from_file_location("bet3_incontext_probe", _p)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
fit, score_detail = _m.fit, _m.score_detail

CLICK_FAMILIES = {"GATE", "TOGGLE"}   # families whose rewarding affordance includes CLICK (label 4)


def run_fold(train_fams, held, n_per, epochs, T, rng, seed):
    train_ds = make_dataset(train_fams, n_per, rng, T=T)
    val_in = make_dataset(train_fams, max(n_per // 4, 30), rng, T=T)
    test_ds = make_dataset([held], max(n_per // 2, 60), rng, T=T)
    net = fit(train_ds, epochs, use_context=True, seed=seed, width=64, head=128)
    acc_in, _ = _m.score(net, val_in)
    d = score_detail(net, test_ds)
    return acc_in, d


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    counts = [4, 6, 8]
    print(f"CONVEX-HULL SCALING PROBE | n_per={n_per} epochs={epochs} T={T} chance=0.20")
    print(f"families order: {FAMILIES}\n")
    rows = []
    for N in counts:
        fams = FAMILIES[:N]
        rng = np.random.default_rng(0)
        in_accs, held_accs, click_rows = [], [], []
        for held in fams:
            acc_in, d = run_fold(fams, held, n_per, epochs, T, rng, seed=0)
            in_accs.append(acc_in)
            held_accs.append(d["acc"])
            if held in CLICK_FAMILIES:
                others = sum(1 for f in fams if f in CLICK_FAMILIES and f != held)
                click_rows.append((held, others, d["click_acc"], d["click_share"]))
        mean_in = float(np.mean(in_accs))
        mean_held = float(np.mean(held_accs))
        rows.append((N, mean_in, mean_held, click_rows))
        print(f"  N={N} families={fams}")
        print(f"    mean in-dist acc={mean_in:.3f}  mean held-out acc={mean_held:.3f}")
        for (held, others, cacc, cshare) in click_rows:
            ca = f"{cacc:.3f}" if cacc == cacc else "n/a"
            print(f"    held-out CLICK family {held:>7}: CLICK-labeled acc={ca}  "
                  f"(other click families in training={others}, click-share={cshare:.2f})")
        print()

    # ---- verdict: does held-out CLICK accuracy rise when another click family is in training? ----
    print("  === CONVEX-HULL READ ===")
    base = []   # CLICK acc with 0 other click families in training (N=4: GATE held, TOGGLE absent)
    withother = []  # CLICK acc with >=1 other click family in training (N>=6)
    for (N, _i, _h, click_rows) in rows:
        for (held, others, cacc, _s) in click_rows:
            if cacc != cacc:
                continue
            (withother if others >= 1 else base).append(cacc)
    b = float(np.mean(base)) if base else float("nan")
    w = float(np.mean(withother)) if withother else float("nan")
    in_at_8 = next((r[1] for r in rows if r[0] == 8), float("nan"))
    print(f"  held-out CLICK-labeled acc with 0 other click families = {b:.3f}  (BET 3 regime)")
    print(f"  held-out CLICK-labeled acc with >=1 other click family  = {w:.3f}")
    if in_at_8 == in_at_8 and in_at_8 < 0.42:
        print(f"  !! VALIDITY WARNING: in-dist acc at N=8 = {in_at_8:.3f} (~chance). The held-out read is")
        print("     confounded by IN-DIST COLLAPSE -> run scripts/bet3_aliasing_diag.py to attribute it")
        print("     (family-confusion/aliasing vs capacity) before trusting the verdict below.")
    if w == w and b == b and w > b + 0.15 and w > 0.40:
        print("  VERDICT: CONVEX-HULL OPENS — a held-out affordance becomes inferable once a sibling")
        print("           affordance is in training. Green-light a large mechanic-family curriculum build.")
    elif w == w and w < 0.30:
        print("  VERDICT: CONVEX-HULL CLOSED — held-out CLICK affordance stays ~chance even WITH a sibling")
        print("           click family in training. Scaling families won't cross the novelty boundary;")
        print("           the offline rich-prior program is capped near 0.33. Bank it.")
    else:
        print("  VERDICT: PARTIAL — re-examine (some lift but below the green-light bar).")


if __name__ == "__main__":
    main()
