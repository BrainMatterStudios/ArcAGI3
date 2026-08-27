# Stage-2b — fixes, tests, full-25 verdict (TRACK 2, engineered agent)

Branch `winning/duck-patched`, nothing committed (per task). Code in
`src/engineered/`, tests in `tests/engineered/`, artifacts here.

## 1. ft09 L1 regression — diagnosis and fix

**Measured cause (from `milestone_s2a.json`, not conjecture):** the s2a run
finished ft09 L0 in 268 actions (effects genuinely helped: t0c needed ~1984),
then burned 3732 actions on L1 without the win. At L1 the click_null rules for
colors 0/2/11/12 re-gated (each 1.0 prequential on its own 10 transitions),
and the planner's goal-2a *excluded* every rule-predicted pair while c=9
clicks — 3590 transitions, 97% board-changing — kept spawning fresh states.
The unpredicted frontier therefore never exhausted, goal 2c was unreachable,
and every demoted click was starved for the entire budget. The menu demotion
was demotion-in-name-only: the goal ladder turned it into a PRUNE.

**Fix (reorder, never prune — with a starvation bound):** an **audit share**.
Every `audit_every`-th (default 8) frontier plan is forced to probe the
cheapest rule-PREDICTED untried pair (planner `audit=True` path, below the
commit goals, above the normal frontier). Predicted pairs now get a
guaranteed 1/N probe share instead of "after an infinite frontier".
Audit probes also feed the prequential windows, so a wrong null rule
un-gates from its own audits.

**Verified:** ft09 with effects ON now completes **2 levels** (L0 616, L1
3302, budget end 82 actions into L2; 351 audit plans). The audit tax on L0
(616 vs 268 when the demotion happened to be right) is the price of
completeness. Pinned by `TestFt09Regression::test_ft09_recovers_l2_with_effects_on`
(also pins L0 <= 1000 actions so the efficiency win is not silently lost).

## 2. Blocked-move sub-family (`MoveBlockedRule`, T1, priority 1)

Per-direction blocked-POSITION predicate for the tu93/re86/tr87/wa30 class
(invisible fences — legality is memory, not layout): learns the mover blob
(colors + pixel-size range, majority vote over `_fit_blob_relocate` fits),
then keys null outcomes by the blob's bbox anchor. `blocked(p, A)` (>=
min_fit nulls, never moved-from) predicts null FROM ANY board state with the
blob at p — the generalization T0 cannot make. A successful move from p
unblocks immediately; the prequential window de-gates state-dependent
legality. `priority = 1`: fence memory outranks a gated TranslationRule that
would happily predict a move through the fence.

Smoke (tu93, 1500 actions): move_blocked A2 **gated at 1.0 held-out**, all
four directions scoring 1.0; 3 levels vs s2a's ~1-2 at that budget.

## 3. Win-condition predicate (`WinRule`, game-level, scope per action id)

Cross-level channel the per-level graphs lack: learned from observed win
transitions (win sources feed `EffectEngine.observe_win`; they stay out of
`observe` — rules model board dynamics, not level boundaries). Gate =
>= 2 wins sharing the signature (action id; for ACTION6 also the clicked
masked color). Precondition = color-presence envelope of observed win
sources (declines novel-colored level states — conservative). Serves
candidate (state, action) pairs; the planner targets them as goal 1.5
("win_pred", below observed-win replay, above all exploration); the agent
verifies on execution; a hard failure cap (5) kills a wrong predicate.

## 4. Tests

20 new unit tests + 1 e2e pin in `tests/engineered/test_stage2b.py`:
MoveBlocked learning/generalization-over-board-state/prequential gate/
priority-over-translation/unblock-on-move/ambiguity-decline; WinRule
gate/signature/precondition/failure-cap/non-click; planner audit
reorder-not-prune (same pair reached, only audit flags it; predicted
preferred under audit), win_pred goal precedence and tried-pair skip.
Full suite: **86 passed, 1 xfailed** (sb26 xfail-with-teeth unchanged).
Scorer cross-validation: reimplemented `true_score` reproduces **56/56**
stored banked duck row scores exactly.

## 5. Full-25 evaluation (official run: `stage2b_full25.md` / `.json`)

Protocol: all 25 public games, 4000 actions/game, wall cap 900 s, effects ON.
Runtime ~7 min wall (5 workers); slowest single game 227 s — the engineered
agent is not wall-clock-bound at eval scale. Two independent full-25 runs
produced identical means (deterministic agent). One earlier full-25 run
(before the WinRule precondition fix, `stage2b_run.log`) differed only in
vc33's win-pred attempt count — no score change.

### Means (true objective, duck = banked max-over-clones pc_base+w2_base)

| engineered | duck BASE_ENV | hybrid (per-game max) | hybrid (frame-Markov dispatch rule) |
|-----------:|--------------:|----------------------:|------------------------------------:|
| 0.2463 | 1.6333 | 1.7836 | 0.7314 |

### The six outright true-score wins (engineered > duck)

| game | ours | duck | note |
|------|-----:|-----:|------|
| tn36 | 3.5714 | 0.0 | L1 in **26 actions = 0.62x human median** — first sub-human-budget completion of the campaign; hits the completion-share cap |
| sp80 | 0.5672 | 0.4947 | genuine head-to-head win vs a NONZERO duck (113 actions, 2.46x human) |
| vc33 | 0.1082 | 0.0022 | 2 levels vs 1 |
| ls20 | 0.0050 | 0.0 | the rotation-gated game the retired-era agent never cracked |
| m0r0 | 0.0013 | 0.0 | hidden-state class |
| s5i5 | 0.0005 | 0.0 | |

Also: tu93 4 levels vs duck 2 (score still loses 0.08 vs 3.74 — efficiency,
not depth, is the binding constraint everywhere).

### VERDICT (pre-registered Stage-2 criteria, stated plainly)

* **(b) FAIL** — full-25 true mean 0.2463 vs duck 1.6333, a miss of **84.9%**.
* **(c) PASS** — 6 outright true-score wins (>= 5 required); 6 level wins.
* **KILL rule** ((b) miss > 20% AND (c) < 3): **NOT triggered** (6 >= 3).
* **Formal verdict: MARGINAL.** The paradigm as a duck REPLACEMENT is dead —
  0.15x of the duck's mean; (b/a)^2 crushes every completion that costs
  >~3x the human budget, and 15 of 19 completed levels sit in games above
  that line.
  What survives is a narrow, real portfolio edge: the engineered agent
  completes levels on games where the duck scores exactly zero.

### Dispatch analysis (the honest part of the hybrid line)

* **Per-game-max (oracle)**: 1.7836 = duck **+0.150**.
* **Pre-registered frame-Markov routing rule**: 0.7314 — value-DESTROYING
  (-0.90 vs duck). Routing by frame-Markov membership hands the duck's
  biggest scorers (ft09 14.29, r11l 4.76, tu93 3.74, lp85 2.78, dc22 2.47)
  to an agent that scores them ~0. Archetype/Markov-violation features do
  not predict WHERE THE DUCK IS WEAK, which is the only routing question
  that matters.
* **Realizable capture** is sequential, not parallel: duck plays first; if
  it completes zero levels, the engineered agent continues the same game in
  the leftover per-game window (actions accumulate, but a duck-zero game
  has score floor 0 — the fallback can only add). Public-set projection:
  **+0.143** (or +0.147 counting near-zero vc33). Both far below the
  pre-registered +0.3 rig ship gate.
* **Concentration caveat (W1)**: 96% of the realizable gain is ONE game
  (tn36). This is a single-game story, not a distributional edge; its
  transfer to the hidden 55-set is exactly the kind of claim this campaign
  has repeatedly measured to be false.

## 6. What Stage 3 (rig A/B) would need — and why it should not run yet

The design's Stage-3 gate is a rig delta >= +0.3. The measured ceiling of
the only realizable hybrid is +0.15 (oracle) / +0.14 (sequential fallback),
concentrated in one game. Running the 28-clone rig now would spend days to
measure a delta already known to be below the ship gate. Stage 3 is
warranted only if one of these first moves the offline ceiling:

1. **Efficiency, not coverage, is the whole game**: a completed level pays
   ~nothing beyond ~3x human budget. The decode/commit split must get a
   REPLAY discount — after a level is decoded once, death-replay of known
   levels is near-optimal (commit mode already does this within a level;
   the loss is in the first decode). Concretely: the frontier sweep cost on
   L0/L1 (dc22 1901, ft09 616+3302, s5i5 1343) must come down ~10x, or
   those completions stay worthless under (b/a)^2.
2. **The win predicate needs a state precondition, not just an action
   signature** (vc33 measured: signature-color clicks at the wrong board
   state, 0/5). Candidate: condition on the diff-pattern that preceded
   observed wins (last-action effect class), which the effect engine
   already computes.
3. **Duck-zero detection on hidden games**: the sequential fallback needs no
   detector (duck's own result is the trigger), which makes it the one
   Stage-3 arm that is cheap to wire: patch the duck harness to hand the
   remaining wall-clock to the engineered agent when levels_completed == 0
   at duck exhaustion. If built, THAT is the arm to rig-test — expected
   delta +0.1-0.15, below gate, so it rides only as a final-2 selection
   argument, not a slot.

Files: src/engineered/{effects,planner,agent,evaluate_stage2b}.py,
tests/engineered/test_stage2b.py, this dir's stage2b_full25.{json,md},
stage2b_run.log / stage2b_run2.log. Nothing committed.
