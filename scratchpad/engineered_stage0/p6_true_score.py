"""Stage-0 Pillar 6 — true_score.py wiring.

1. Imports submission/_ab_patch_closure/true_score.py by path.
2. Scores a synthetic rows structure with hand-computable expectations.
3. Scores every REAL banked wave under scratchpad/banked_waves_20260809/ and
   reports rows-with-score coverage (the module was validated 224/224 rows).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_module():
    p = REPO / "submission/_ab_patch_closure/true_score.py"
    spec = importlib.util.spec_from_file_location("true_score", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ts = load_module()

    # --- synthetic check: max over clones, mean over games -------------------
    sample = {"rows": [
        {"source_game": "ft09", "score": 1.2},
        {"source_game": "ft09", "score": 0.4},   # max -> 1.2
        {"source_game": "vc33", "score": 0.6},
        {"source_game": "tu93", "score": None},  # ignored
        {"source_game": "tu93", "score": "0.9"}, # coerced -> 0.9
    ]}
    m = ts.true_score_metrics(sample)
    expect = round((1.2 + 0.6 + 0.9) / 3, 4)
    ok_synth = (m["true_score_all_games"] == expect and m["true_score_n_games"] == 3
                and m["true_score_per_game"]["ft09"] == 1.2)
    print(f"synthetic: mean={m['true_score_all_games']} (expect {expect}) "
          f"n_games={m['true_score_n_games']}  -> {'OK' if ok_synth else 'MISMATCH'}")

    # --- real banked waves ---------------------------------------------------
    ok_real, n_files = True, 0
    for f in sorted((REPO / "scratchpad/banked_waves_20260809").glob("*.json")):
        try:
            result = json.load(open(f))
        except Exception as ex:  # noqa: BLE001
            print(f"  {f.name}: unreadable ({ex})")
            continue
        rows = result.get("rows") or []
        if not rows:
            print(f"  {f.name}: no rows (not a wave result) — skipped")
            continue
        scored = sum(1 for r in rows if r.get("score") is not None)
        mm = ts.true_score_metrics(result)
        n_files += 1
        ok_real &= mm["true_score_n_games"] > 0 and scored == len(rows)
        print(f"  {f.name}: rows={len(rows)} scored={scored} "
              f"true_score_all_games={mm['true_score_all_games']} n_games={mm['true_score_n_games']}")
    # documented cross-check (true_score.py docstring): base 1.475, closure cand 0.313
    ok = ok_synth and ok_real and n_files > 0
    print(f"\nP6 VERDICT: {'PASS' if ok else 'FAIL'} ({n_files} banked wave files)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
