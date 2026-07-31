# B-TRACK PROTOCOL — research-derived arms (PRE-REGISTERED 2026-07-27)

**Written before any GPU spend or data collection on these arms; all code built
and Stage-1-gated 2026-07-27, before Gate 0 of the A-track. Thresholds below
are frozen. Post-hoc changes go in §6 as append-only amendments and demote
results to exploratory.** Derived from the 2026-07-27 verified research sweep
(memory `arcagi3-research-2026-07-27-external-landscape`,
`docs/EXTERNAL-PATTERNS-2026-07-27.md`). Companion to
`docs/A1-PROTOCOL-2026-08.md`, whose §0 standing conditions (scored-ref tree,
0.6/0.95/20 sampling, applied-but-toggled patches, ab_driver instrument,
pinned 13-game holdout, paired levels-completed metric) apply verbatim.

## 0. The arms

| arm | toggle | what it does | provenance |
|---|---|---|---|
| B1 effect memory | `EFFECT_MEMORY=1` (`submission/_duck_effects/`, `APPLY_EFFECTS_PATCH=1`) | harness-measured (action → tried/changed) table + 64×64 click-productivity grid per level, surfaced in every user prompt; system line binds CONFIRMED marks to observed transitions | StochasticGoose frame-change signal + Sensi grounding fix |
| B3 phase gate | `DOCTRINE_PHASEGATE=1` (doctrine D4) | explore-then-commit prompt protocol: no long plans while >2 load-bearing uncertainties open | AERA (arXiv 2605.25931), implemented as advertised (threshold), not as shipped (keyword) |

B2 (ledger anchoring) is folded into B1's system line — programmatic
verification of free-text claims is infeasible; the measured block plus the
CONFIRMED-requires-observed-transition rule is the implementable anchor.
The A-track ledger arm itself is untouched.

Evidence basis for dropping Reki-style repairs (recorded in
EXTERNAL-PATTERNS §1): illegal actions are already engine-rejected at zero
scored cost, MOUSE is clamped, and malformed-JSON / non-acting-turn failures
are absent from all real current-config episodes.

## 1. Scheduling (no new hours; B rides A's panel)

B-track runs ONLY after A-track Gates 0-3 have verdicts, in the week-2+
budget, on the SAME frozen panel from A1 §2 (Gate-1 banding). Server config =
the A-track SELECTED config (adapter if GO, else base). No B measurement
before the A-track selection exists — B deltas must be measured on the config
that would actually ship.

## 2. Gate B (paired A/B, ~6-8h GPU)

- Arms: control (A-selected stack) vs control+B1, and control vs control+B3,
  2 paired rollouts per game per arm via ab_driver (one session per arm).
- **GO per arm iff** (mirrors A1 Gate 3): (a) total paired levels-completed
  delta ≥ **+2** across the panel; (b) no single game regresses ≥ **2**
  levels summed across rollouts; (c) scorecard non-inferior (first
  board-changing action median ≤ 2; HUD-anchoring game_overs not increased).
- If BOTH GO independently: one confirmation pair (1 rollout × panel) with
  both toggles on; stack ships only if the combined delta ≥ the better
  single-arm delta − 1 (no destructive interaction).
- Hours short → B1 has priority (grounded-signal arm; B3 is prompt-only and
  keeps for the following week).

## 3. Debut interaction

GO'd B arms join the A-track debut stack (A1 §5) only if their gate verdict
lands BEFORE that debut is submitted; otherwise they queue for the next
gated debut. Never toggle an unmeasured arm into a scored submission. The
7-draw stop rule and base-mean floor of A1 §5 govern any stacked debut.

## 4. Registered but NOT built (no code exists; pre-registered as candidates)

- **B4 verification-first world model**: baseline1-style replay verification
  (predicted next frame/status vs observed) of a persistent model, ported to
  the sandbox; verification-before-executable-model per the 2607.15439
  ablation. Build only if an A/B slot exists after B1/B3 verdicts.
- **B5 budget restraint on unmodelable games**: OPINE-style action-budget
  down-scaling where the model's predictions keep failing. Interacts with
  the adaptive-budget lever; needs its own design note first.

## 5. Stage-1 record (2026-07-27)

- Unit tests: 58/58 across `_duck_effects`, `_duck_doctrine`, `_duck_fixes`,
  `_serve_verify_k3`.
- ReplayMockLLM (smoke_sk01_152145, scored-ref tree): effects-only applied-off
  BYTE-IDENTICAL; all four families applied, all toggles off BYTE-IDENTICAL.
- Note: the same replay run demonstrated the `_adopt` tree drift live
  (helpers block + temp 0.3/top_p 0.9 diff) — Law 11 re-confirmed.

## 6. Amendments

### 6.1 — B1 shipped ungated in the 2026-08-01 slot (2026-07-31, Ahmed)

**Deviation from §1 and §3.** B1 was submitted to the public leaderboard before
the A-track had any gate verdict and before Gate B measured it, which §1
("no B measurement before the A-track selection exists") and §3 ("never toggle
an unmeasured arm into a scored submission") both forbid. Authorized
explicitly by Ahmed after the conflict was raised.

**Consequence, per the preamble:** the resulting score is **exploratory, not
confirmatory**. It cannot be read as evidence for or against B1. A single draw
sits inside the base distribution (n=8 on identical bytes: 0.92, 1.14, 0.82,
0.75, 0.96, 0.88, 1.27, 0.69 → mean 0.929, sd 0.195); separating a real effect
from that variance needs the paired ab_driver measurement of §2, which remains
the only instrument that can issue a GO. Gate B is NOT satisfied or waived by
this submission.

**Context — why the build was rebuilt first.** The kernel armed for this slot
(`_duck_effects` v1) set `EFFECT_MEMORY=1` and nothing else. No shipped code
reads that variable: `effect_memory.py` was never imported into the notebook
and is absent from the `taaf-src-hybrid` bundle; its only other consumer is
`scratchpad/rl_gate/run_rollout.py`, a local-harness path gated on
`APPLY_EFFECTS_PATCH`. v1 would have run as plain base duck under an
effect-memory label and corrupted the ledger with a phantom arm. v2 inlines
the pack into the customization-hook cell (the `duck-patched` mechanism) and
hard-fails the cell if the wrappers do not land. The Kaggle commit run for v2
logged all four verify lines plus `[effects] ARM ACTIVE`, so the arm is
confirmed live. **Standing rule added: an env toggle is not a shipped arm —
every scored arm must print positive in-kernel proof that its code applied.**
