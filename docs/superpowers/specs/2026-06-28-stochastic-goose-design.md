# StochasticGooseExplorer — faithful standalone CNN-RL policy (2026-06-28)

## Motivation
The campaign concluded "the offline non-LLM paradigm caps near 0.33." The live Kaggle leaderboard
(verified 2026-06-28, no-internet eval confirmed) refutes the *paradigm* claim: Tufa 1.21, a dense
cluster 0.5–0.8, us 0.33, 1478 teams. Offline methods reach 2–4× our score. So 0.33 is THIS codebase's
ceiling, not the paradigm's. See `arcagi3-033-ceiling-is-codebase-not-paradigm` memory.

The documented winning offline family = informed action-space search (= our SalienceExplorer ✓) +
**StochasticGoose** (Tufa, 1st preview, 12.58%): a CNN action-effect predictor + sparse RL on level-
completion + replay buffer + state-hash dedup, **retrained between levels**, run **STANDALONE as the
whole agent**. Every learned attempt in this repo (GraphRanker, primary_model_explorer, EffectLearner,
DynamicsLearner, online_explorer) bolted a CNN onto the graph explorer as a reranker/proposer/pruner —
which "breaks the graph." NONE faithfully built the standalone architecture. That is the untested gap.

## Goal & success bar
Build `StochasticGooseExplorer` as a standalone policy (no graph backbone). **Success = total levels
across the 18 offline dev games strictly exceeds banked TransferExplorer at matched budget, with no
catastrophic per-game collapse (tu93 stays).** Otherwise it is a characterized negative.

## Architecture (`src/arcagi3/stochastic_goose_explorer.py`)
Reactive `decide(grid, gstate_terminal, gstate_notplayed, levels, available)` → `("reset",)` /
`("S", aid)` / `("C", x, y)`. Exposes `.gs`/`.wm` duck-typing for the runner's `states_seen`.

1. **State dedup hash table.** Masked key reusing the proven keying from `SalienceExplorer._key`
   (`VolatilityTracker` mask | `border_mask` band, then `P.object_state_key`). Per key: set of tried
   actions. NOT a navigation graph — no shortest-path replay (that is the graph explorer; StochasticGoose
   has no nav).

2. **Effect-CNN policy.** Reuse `learned_dynamics._build_net` (16-ch one-hot → per-cell click-effect map
   + 5 simple-action logits). Among UNTRIED available actions at the current state, select by predicted
   P(state-change), SAMPLED ∝ effect-probability at temperature τ (the "stochastic" in StochasticGoose;
   keeps exploring, avoids argmax lock-in). Candidate actions come from the same source as the explorer:
   simple actions in `available`, and `P.salient_click_targets(grid, max_targets, coarse_grid_step)` for clicks.

3. **Online TTT.** Replay buffer of `(one_hot, action, changed?)` where `changed? = (next_key != key)`.
   Train every `train_every` steps once `n_seen >= min_train` (BCE on the predicted action's effect logit),
   mirroring `DynamicsLearner._train_step`. **Model + buffer CARRIED across levels** (faithful: "iteratively
   retrains between levels"; do NOT reset per level — the key difference from DynamicsLearner).

4. **Cold-start** (`n_seen < min_train`): act by salience-tier order (simple actions first, then salient
   clicks by priority) with RNG tie-break — early behavior ≈ the working explorer before the model warms up.

5. **Stuck handling.** When all untried actions at the current state are exhausted: take a random available
   action (accept a revisit) to break the cycle. AVOID `("reset",)` as a frontier tool — `env.reset()`
   returns to L0 (verified engine-scoring), catastrophic for deep levels. `("reset",)` ONLY on GAME_OVER/
   NOT_PLAYED/terminal.

## Firewall
Brand-new file; new `--agent goose` in `runner.py`. Banked TransferExplorer (0.33) and the submission are
untouched. Torch imported lazily; CPU-allowed for local dev via the existing `ARCAGI3_ALLOW_CPU` convention.
Nothing ships without explicit user go-ahead.

## Empirical forks (resolved by A/B, not pre-decided)
- sampling temperature τ (greedy→stochastic);
- model-carry-across-levels (default, faithful) vs per-level reset;
- random-on-stuck (default) vs bounded reset-on-stuck.

## Validation
TDD: unit tests for masked keying, per-state tried-action dedup, effect-sampling among untried, cold-start
order, model-carry-across-levels, terminal→reset, firewall (deterministic with a stub/zeroed model). Then
A/B `goose` vs `transfer` on the 18 dev games at matched budget (CPU), watching tu93 (no-collapse), lp85
(deep levels), cd82/vc33/ar25 (regression-prone). Standard gate scripts in scratchpad per the handoff.

## Non-goals (YAGNI)
No T4 path yet (local CPU dev first). No reward-shaped value head in v1 (add only if effect+dedup+TTT
under-performs and the A/B points to it). No graph navigation. No submission changes.
