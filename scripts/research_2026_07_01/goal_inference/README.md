# Goal-inference research assets (2026-07-01)

Preserved from the session scratchpad. See docs/HANDOFF-2026-07-01-research-continuation.md for context and the
forward menu. NOTE: several files hardcode the old scratchpad path in `sys.path.insert(...)` — replace that line
with the directory containing `goal_harness.py` before running, e.g.:
  sys.path.insert(0, os.path.dirname(__file__))
Run with: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python <file> [games...]

- goal_harness.py    — FALSE-POSITIVE-PROOF measurement (capture L0 + win transition; evaluate in-window k->k+1)
- stage1_v4.py       — structural goal inducer (translation-search template, cover, causal + submit-aware filter)
- goal_directed.py   — goal-directed search (predicate as progress heuristic)
- verify_gate.py     — VERIFY gate (re-derive L0; NOTE: certifies L0-validity NOT L1-correctness — see handoff)
- effect_probe.py / greedy_cover.py / beam_cover.py — mechanic/effect learning (probe action effects on predicate)
- kill2.py / fix_hypo.py — the factored-model amortization experiments (kill2 had the measurement bug; see memory)
