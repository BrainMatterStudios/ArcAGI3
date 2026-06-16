# Phase A′ — Model-as-Primary Explorer (StochasticGoose-faithful)

> Pivot after GraphRanker (re-ranker) was killed at the A.4 gate (tu93 9→2). Authored by the
> orchestrator from the judged C2 candidate + research (memory arcagi3-research-findings) + the
> GraphRanker post-mortem. HALTS for approval before the multi-week build. Floor: v6 = 0.33;
> never ship below it.

## 1. Why model-as-primary (and why GraphRanker failed)
GraphRanker bolted the model onto SalienceExplorer's *tie-break*, so the model fought a
well-tuned tier-selector and corrupted exploration on games the graph already won (tu93
9→2, stalled cycling). The verified #1 (StochasticGoose) instead makes the **learned model the
primary action selector** — it *is* the explorer, with the graph as memory. That removes the
interference: there is no tier-selector to fight; novelty comes from the graph, not from a
hand-crafted salience the model overrides.

## 2. Architecture
A NEW policy `PrimaryModelExplorer` (does not modify salience_explorer.py; reuses its `_Node`,
perception, masks, encode_onehot, and the `online_model.OnlineActionEffectModel` harness).

Per state (masked object_state_key node, same as v6):
1. **Exploit:** if a known action here produced reward → take it (graph `reward_action`).
2. **Model-driven frontier pick:** among the state's UNTRIED candidates (simple actions +
   salient click targets), pick the one with the **highest model-predicted frame-change /
   novelty score** — biased AWAY from candidates whose recorded edge leads to an already-heavily-
   visited state (graph novelty prevents the tu93 cycling that killed GraphRanker).
3. **Backtrack:** if no untried candidate here → BFS shortest path to nearest node with untried
   candidates (reuse the graph), then resume.
4. **Reset/bounce** as v6.
Online training identical to the harness: int8 replay, hash-dedup, BCE on masked-key-change,
reset model+buffer per level. The model's job is **action-efficiency** (try effect-causing
actions first), which is exactly what the squared metric rewards.

## 3. Firewall (unchanged discipline)
- **GPU-only + fail-safe:** no usable CUDA → `PrimaryModelExplorer` falls back to pure
  SalienceExplorer (v6). New opt-in flag `ARCAGI3_ONLINE=primary`; committed default OFF.
- **Anti-cycle by construction:** graph-novelty in the scorer + the novelty-stall breaker →
  if model-driven selection stops discovering new states, revert that level to v6 tier-exploration.
- **Never ship < 0.33:** gate = 177 tests + 8 local games (no regression, esp. push 3/3) +
  25-game @30k real sweep **≥34 strict-superset** + first leaderboard submission > 0.33.

## 4. Phased build (each gated, default-off until proven)
- **A′.1** PrimaryModelExplorer skeleton: model-driven frontier selection + graph reuse +
  graph-novelty scorer + fail-safe. Unit tests (fail-safe, model-off == ... n/a since it's a new
  selector; instead: model-off → behaves like salience tier order).
- **A′.2** Local validation: MUST keep push 3/3, navg, btnc; ideally improve action-efficiency.
  Kill if push regresses.
- **A′.3** Real-game subset A/B (tu93 must NOT regress — the GraphRanker killer; watch sc25 for
  the unlock). Tune novelty/conf once.
- **A′.4** Full 25-game @30k sweep: ship only if ≥34 strict-superset, no per-game regression.
- **A′.5** Notebook embed + submit at a daily window; promote only if leaderboard > 0.33.

## 5. Honest EV
This is the *proven #1 architecture* (StochasticGoose preview 12.58% beat our directed-graph
class ~2:1). Realistic if the eval GPU works (docs: T4×2): **0.5–0.9**; floor 0.33 (firewalled).
Key risks: (a) deeper integration than GraphRanker → more care needed; (b) eval-GPU (P100 caution,
T4 documented — resolves on first online submission, no downside); (c) cross-game generalization
of the online procedure. Effort ~2–4 weeks; competition runs to Nov 2.

**Plan only — awaiting go-ahead before A′.1.**
