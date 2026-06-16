# Step-change strategy: closing 0.33 → 0.7–1.2 on ARC-AGI-3

> Source: deep-research workflow (101 agents, cited, 2026-06-16). Verified vs inferred flagged.
> Our banked: v6 = **0.33** (training-free salience-tiered graph exploration). #1 Tufa Labs = **1.21**.

## The core finding (why we're capped at 0.33)

Our agent is the **same architectural class as the preview's 2nd place (Blind Squirrel,
directed state graph): it scored 6.71% — roughly HALF the winner.** The winner
(**StochasticGoose / Tufa Labs / Dries Smit, advised by Jack Cole**) is **not
training-free** — it *learns a model of game dynamics* and uses it to explore far more
efficiently. The benchmark is deliberately engineered (squared action-efficiency metric,
1.15× human cap, level weighting) to reward four capabilities a blind graph explorer
structurally lacks: **Exploration, Modeling (a world model that predicts future states),
Goal-Setting, Planning.** No amount of graph-exploration tuning escapes that cap. This is
why SAGE and the prior six levers were marginal — they optimize within a capped paradigm.

LLM-driven leaders (Symbolica Arcgentica; Rodionov executable-Python world model) are
**NOT eligible** under the offline/no-hosted-LLM Kaggle constraint — ignore them as entries
(useful only as design inspiration).

## The verified eligible top recipe (StochasticGoose, preview 12.58% / 18 levels)
- **Input:** 16-channel one-hot 64×64 frame.
- **Model:** 4-layer CNN backbone → an action head predicting *which of ACTION1-5 cause a
  frame change* (legality/effect probabilities) + a **spatially-aware convolutional
  coordinate head for ACTION6 clicks** (2D inductive bias, NOT flattened features).
- **Training:** online, per-game. Every transition stored in a replay buffer, **hash-dedup**
  of (state,action) pairs, train a frame-change classifier (BCE) every few steps,
  **reset buffer + model between levels.** Single-GPU, offline-feasible.
- This *learned* exploration beat our directed-graph class ~2:1 in the preview.

## Recommended path (phased, no-regression-gated, never ship below 0.33)

**Phase A — bolt a learned action-effect model onto the existing explorer (cheapest, mirrors #1).**
Keep SalienceExplorer's graph; add an online-trained CNN that predicts, per state, which
simple actions and which click coordinates actually change the frame / make progress, and
use it to *rank* exploration (replacing random tie-break and hand-crafted salience tiers —
especially for the 4096-click space, where their learned conv head is the big lever). This is
the smallest categorical change with the highest evidence base (it *is* the #1 recipe).
Train per-game via hashed replay; reset between levels. Single-GPU, offline.

**Phase B — Dreamer-style model-based RL (higher ceiling, higher risk).**
Learn a latent world model and improve the policy by imagining rollouts (actor-critic via
backprop through the model). DreamerV3 is single-GPU trainable, ~20× more sample-efficient
than model-free, and solves sparse-reward long-horizon instruction-free tasks (Minecraft
diamonds from scratch). Directly supplies the Modeling+Planning the metric rewards.

**Phase C — cheap complement: filtered behavior cloning** on the top-return subset of
collected exploration trajectories (matches/beats Decision Transformer at less compute).

**Decision (open question, unresolved by evidence):** Phase A (bolt-on) vs jumping to
Phase B (full MBRL). Recommendation: **A first** — cheapest, mirrors the proven #1, lowest
regression risk, reuses our perception/graph; measure the lift; escalate to B if A plateaus.

## Hard caveats (from the research)
- The **1.21 evolution from the 12.58% preview CNN is INFERRED, not documented** — treat the
  preview architecture as the credible foundation, not a guaranteed 1.21 recipe.
- **Cross-GAME generalization of a learned world model to unseen games under <12h is the
  central unverified risk** — Dreamer's "one config, 150+ tasks" is within a domain, not
  across unseen games. Design for it explicitly (per-game online adaptation, like StochasticGoose).
- **Overfitting the public set is a real failure mode** (StochasticGoose's own launch collapse;
  the squared-efficiency metric punishes brute force). Keep the generalization guards.

## What this means for in-flight work
- **SAGE** (graph navigate-back) is a *within-paradigm* increment — low EV vs the step-change.
  Keep it only as a cheap gated experiment; it is not the path to winning.
- The real program is **introducing online learning** (Phase A) — a categorical shift from
  training-free. Needs a GPU training harness inside the agent (per-game, offline, single-GPU).
  This is a multi-week build; the competition runs to **Nov 2**, so there is runway.

**Bottom line:** to win we must stop being training-free and learn a model of game dynamics,
starting with the proven StochasticGoose-style action-effect model (Phase A). Everything we
built (perception, graph, the measurement factory) becomes the scaffolding around it.
