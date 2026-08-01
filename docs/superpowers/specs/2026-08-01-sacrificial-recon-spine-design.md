# Sacrificial-reconnaissance spine — design

**Status:** proposed, awaiting approval. No implementation started.
**Date:** 2026-08-01
**Supersedes nothing.** Complements `docs/A1-PROTOCOL-2026-08.md` (which governs adapter arms);
this document governs a harness arm and does not touch the adapter track.

---

## 1. Why this exists

Every experimental arm this campaign has scored lands inside the base distribution
(n=8, mean 0.9288, sd 0.1947). The best experimental result ever recorded, 1.26, is
beaten by a plain base draw of 1.27, and that base draw is what holds our rank. The
public leaderboard cannot resolve the effects we have been chasing: at CV 21%,
detecting a true +0.10 needs ~60 draws per arm.

Meanwhile ARC Prize ran a controlled experiment on a fixed model (technical report
§4.3.1) that moved one environment from **0.0% to 97.1%** by changing the harness
alone, and named context management as the central challenge. The lever is the
harness, and specifically what the agent *knows* and *retains* — not the weights.

This design proposes the first arm aimed at that, built on a measurement nobody in
this campaign had made until today.

---

## 2. The measurement this design rests on

Settled 2026-08-01 by static trace and controlled experiment, agreeing to four
decimal places across both the direct wrapper and the competition-mode REST path.
Scripts under `scratchpad/` (see the audit record).

**Scored actions accumulate across level attempts. They do not zero on reset.**

Measured on tu93 (human baseline h(1)=19, an 18-action winning sequence):

| arm | what happened | a(1) | S(1) |
|---|---|---|---|
| A | clean win in 18 actions | 18 | 111.42 |
| B | 31 wasted → RESET → the same 18-action win | 50 | **14.44** |
| E | 4×(40 wasted + RESET) → the same win | 182 | 1.09 |

For arm B the two hypotheses were `H_zero = 111.42` and `H_accumulate = 100·(19/50)² = 14.4400`.
Observed: **14.4400**. RESET costs exactly 1 scored action. In competition mode a
second play is unreachable (HTTP 400), so max-over-plays is closed and there is no
clean scored replay to fall back on.

**Three consequences that define the architecture:**

1. **Brute-force search cannot buy back its cost.** A searcher that spends 92k actions
   on a level with baseline 30 scores approximately zero, *even if it wins*. This is
   why the published graph-exploration agents are not directly transplantable.
2. **`a(l)` is per-level and independent.** Burning level *j* destroys only `w_j/W` of
   that game's score, leaves `S(1..j-1)` frozen, and leaves `a(j+1..)` untouched.
3. **A level you never complete scores 0 regardless of what you spent.** A failed
   search is therefore free.

**The break-even that follows.** Writing off level *j* entirely is worth it if you
then complete level *j+1* at `S ≥ 100·j/(j+1)` — 50% for j=1, 67% for j=2, 75% for
j=3. Since the weight ratio always exceeds 1, depth bought with a burnt level is
almost always net-positive. On a 9-level game (W=45):

```
play L1 perfectly, then stall     ->  100·1/45         =  2.2
burn L1 (S≈1), clear L2–L5 at ~100 -> (1 + 1400)/45    = 31.1   (cap 33.3)
```

Against a field leader at 1.86.

---

## 3. The bet, stated so it can be falsified

> **Spending level 1 as tuition — exhaustive, deliberately uneconomic exploration —
> buys enough transferable knowledge of a game's mechanics that levels 2+ can be
> cleared at a small multiple of the human baseline.**

This is the inversion of what every published implementation does. Both reference
agents (Rudakov et al.'s graph explorer; Occam) **discard the graph, the effect table
and the status-bar mask at every level boundary**, and both authors name cross-level
transfer as their single most important open problem. Their per-level results show
exactly the predicted failure: shallow levels solved exhaustively, then a wall
(ls20 levels 3–8 never solved; ft09 levels 4–10 never solved).

**The bet fails if** mechanics do not in fact transfer across levels within a game —
i.e. if level 2 introduces enough new structure that level-1 knowledge does not
reduce the search. In that case we have burnt level 1 for nothing and score *worse*
than the conservative policy. This is a real risk: ARC-AGI-3 environments are
required by design to contain *multiple* mechanics, and levels escalate structure.

**Kill criterion (pre-registered).** On the 25-game offline harness, scored with the
official formula, the sacrificial policy must beat the conservative policy
(every level budgeted at ~1.5×h, never deliberately burnt) on **total score across
games**, not on levels reached. If it does not, we ship the conservative policy and
record the bet as refuted. This is measured offline at zero GPU cost before any
submission.

---

## 4. Architecture

Three phases per game. The LLM is never in the per-action loop.

```
per game:
  Phase A — RECON (level 1 only, deliberately uneconomic)
     exhaustive graph exploration; build:
       · state graph            (frame hash -> action -> frame hash)
       · effect table           (action -> P(state change), per context)
       · dead-action set        (actions that never do anything)
       · counter/HUD mask       (pixels that change on every action)
       · goal predicate         (LLM-proposed, validated against observed completions)
     exit when: level cleared, or recon budget exhausted
        |
  Phase B — EXPLOIT (levels 2+, budgeted ~2-3x baseline per level)
     plan against the learned model; verify every prediction against the engine;
     halt on mismatch and fall back to bounded local search
        |
  Phase C — FLOOR (on plateau or budget exhaustion)
     hand control to the duck, injecting the learned artifacts as context
```

### 4.1 Why Phase C exists

Our rank is held by a base duck draw. Two arms in campaign history have scored 0.00.
A new spine that fails silently would cost a slot and tell us nothing. Phase C
bounds the downside to approximately base duck, which is the floor-safe posture
chosen for this arm.

### 4.2 Integration seam

`HarnessSolver.analyzer_factory` (`.../inference/framework/solver.py:775`, consumed at
`_make_analyzer`, `solver.py:1182-1189`). It is a documented extension point,
`AnalyzerFactory = Callable[[Game, int], Any]`, defaulting to `None` — and **nothing
in this repo currently uses it**. Setting `bm.solver.analyzer_factory` in the
notebook's customization-hook cell is strictly cleaner than the monkey-patching every
prior arm used, and it survives the solver copy performed by the benchmark runner.

The spine object must satisfy the analyzer contract: `.analyze(...) -> AnalyzerTurnResult`
(`tool_agent.py:1706-1716`, result dataclass at `tool_agent.py:370-375`), executing
actions through the harness-supplied `step_env` callback.

**Constraint:** `_HarnessGameSession.play()` still owns the outer loop and the
`should_stop` cadence. Driving a whole level inside a single `analyze()` call is legal
and is what we want, but the spine must poll the injected `should_stop` itself or it
will overrun the 132-minute per-game cap and be cancelled mid-search.

### 4.3 Components

| component | build or reuse | source |
|---|---|---|
| frame hashing | **reuse** | `geodesic_postpass.py:64,69` — structural tuple hash, already handles the 3D frame case |
| shortest-path over action graph | **reuse** | `geodesic_postpass.py:103` `_bfs` |
| connected-component segmentation | **reuse** | `utils/segmentation.py:76` — 4-connectivity, returns nodes with `id/color/hash/pixels/boundary/children` plus adjacency |
| level-completion signal | **reuse** | `solver.py:705-707` |
| official scorer | **reuse** | `taaf/game.py:381-413`; local reimplementation at `scripts/research_2026_07_01/ab_efficiency.py:52-68` |
| priority-tiered frontier | **build** | adapted, see 4.4 |
| effect table + dead-action set | **build** | |
| cross-level transfer carrier | **build** | this is the novel part |
| goal-predicate validation | **build** | |

The genuinely new work is the frontier policy, the transfer carrier, and the phase
controller. Most primitives already exist.

### 4.4 What to adopt from the references, and what not to

**Adopt.** The five-tier priority gate with *global* escalation — tier 0 exhausted
across the whole graph before tier 1 is touched anywhere. The published ablation is
unambiguous: segmentation alone helps, but "favour new actions" *without* the full
graph actively regressed (as66 dropped to 4 levels). Arrows always in tier 0. Random
interior pixel per component rather than centroid (correct for concave shapes).
Reverse-BFS seeded from all frontier nodes for distances. Suspicious-transition
quarantine with threshold 3 — the bug it fixes (an un-marked reset-triggering edge
becoming the perpetual nearest frontier) cost the reference implementation its
official run, and with RESET costing us a scored action it would be worse here.

**Adapt.** Counter/HUD detection by empirical "changes on every action" rather than
edge-geometry rules — the rule-based detector is the authors' acknowledged
brittleness, and the scored set is 55 unseen games. Generalize dead-action pruning
*across* nodes; both references prune per node, which is the cheapest available
improvement over the published method.

**Do not adopt.** Reset-replay as the primary navigation primitive — the reference
agent with the best evidence never rewinds; it walks the graph, and under our cost
model (RESET = 1 action, replay fully billed) walking strictly dominates. Environment
cloning — `deepcopy` works offline and is **meaningless in competition**, where the
env is an HTTP wrapper around a server-side guid. A search built and unit-tested
locally on cloning would pass every local test and silently alias at submission.
Per-step LLM gating — measured at ~24× interaction cost and scoring *below* a random
agent in the reference paper's own baseline.

### 4.5 Where the LLM sits

Three calls per game, none in the action loop, every output falsified before it steers:

1. **Goal predicate.** Given a compressed diff of what changed when `levels_completed`
   incremented, emit a Python predicate over the frame. Validate against every recorded
   completion; discard on failure. Surviving predicates rank the frontier.
2. **Feature naming.** "The avatar is the 2×2 blue block"; "yellow squares are
   collectibles." This supplies the state abstraction the effect table is built over.
   It is where LLM priors genuinely help and where the algorithm cannot bootstrap.
3. **Plateau diagnosis.** On stall, propose a macro-action or a region to concentrate on.

This is the literature's largest replicated effect applied correctly: an external
sound verifier is worth far more than model self-critique (Blocksworld 40%→88% with a
real verifier versus 55% with LLM critique). A deterministic, resettable environment
with observable level completion *is* that verifier, and we have never organized
around it. **No LLM self-critique anywhere in the control loop** — intrinsic
self-correction degrades models monotonically at this size class.

---

## 5. Budget policy

Derived from the accounting measurement.

- **Recon (level 1):** unbounded in score terms, bounded in wall clock. Level 1
  carries weight 1 of `W = N(N+1)/2`, so on a 9-level game the entire recon phase
  costs at most 2.2 of 100.
- **Exploit (levels 2+):** target `a(l) ≈ 2–3·h(l)`, retaining 11–25% of that level's
  score. `h(l)` is **not available at eval** (`game_api.py:233-243` — `base_actions_per_level`
  is `None` in submission mode), so it must be estimated from level-1 experience and
  frame complexity. This is an open sub-problem, flagged rather than hidden.
- **Marginal cost of an action is `2/a(l)` of the level's remaining score** — heavily
  front-loaded. The expensive zone is `a(l) ∈ [h, 5h]`; past ~10h the level is already
  written off and further search is nearly free. So: either stop at ~1.5h or commit.
- **Never RESET with zero actions taken since level start.** Competition mode takes an
  `_action_count == 0` shortcut that does not step the game but still charges 1 action.
- **Keep `ONLY_RESET_LEVELS=true`.** It costs nothing and prevents a snap-back to level 1.

---

## 6. Development loop

Entirely offline, no GPU, no LLM for the search components.

```bash
cd /Users/ahmed/Documents/ArcAGI3 && PYTHONPATH=src .venv/bin/python - <<'EOF'
import os; os.environ["ONLY_RESET_LEVELS"] = "true"   # MUST set — production semantics
from arc_agi import Arcade, OperationMode
c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
...
EOF
```

25 official games in `environment_files/`, 252 community games in
`scratchpad/arc_interactive_upstream/environment_files/`, a 13-game holdout in
`scratchpad/holdout_arcint/`. Measured throughput 37,205 steps/sec tight-loop,
2,000–5,000 steps/sec sustained.

**Gate on score, never on depth.** The prior kill experiment on level-reset
Go-Explore (`scratchpad/killexp_levelreset.md`, DOWNGRADE) failed on exactly this
axis: it measured repositioning efficiency and depth, not scored outcome. Wire
`ab_efficiency.official_run_score` into the loop from day one.

**Before any scored slot**, run against `ArcadeSpec(competition_sim=True)`
(`competition_arcade.py`), which exercises the submission-shaped arcade, the shared
scorecard, and the once-per-game constraint. Three known dev/prod divergences make
local success insufficient evidence: cloning (works offline, meaningless in
competition), step cost (0.027 ms in-process versus an HTTP round-trip), and the
GIL (28 CPU-bound search threads in one process, where today they are I/O-blocked
on vLLM).

---

## 7. Risks

| # | risk | mitigation |
|---|---|---|
| 1 | **The transfer bet is wrong** — level 2 mechanics don't follow from level 1 | Pre-registered offline kill criterion (§3). Conservative policy is the fallback and is a strictly smaller change. |
| 2 | **Scoring economics, not engineering, binds** | Score offline with the official formula from day one; never report depth as progress. This is how the prior kill experiment went wrong. |
| 3 | **GIL contention** — 28 concurrent CPU-bound searches in one process | Keep hot loops in numpy (releases the GIL); measure turn throughput under simulated concurrency before submitting. |
| 4 | **Dev/prod divergence** (cloning, step cost, reset semantics) | Ban `deepcopy` in the design; assert `ONLY_RESET_LEVELS` inside the spine; validate under `competition_sim`. |
| 5 | **State aliasing** — a frame is not a sufficient statistic | Confirmed today: 13 of 25 games change internal state with a byte-identical frame; wa30 ends the game that way. Detect non-determinism (same `(hash, action)` yielding two successors) and split the node by extending its key; never assume frame identity implies state identity. |
| 6 | **Silent failure at eval** | Same stance as the levers arm: hard-fail, never warn. Scored-run logs are unretrievable, so ERROR-vs-score is the only channel that reaches us, and ERROR costs no slot. |

---

## 8. Scope boundaries

**In scope:** the spine, the transfer carrier, the phase controller, the offline
scoring loop, and the floor-safe handoff.

**Explicitly out of scope for this spec:** the adapter/SFT track (governed by A1);
any change to model, sampling, serving or concurrency; the Claude-teacher corpus.

**Not decided here, deliberately:** how `h(l)` is estimated at eval (§5); whether the
graph is carried across *games* as well as levels. Both need their own measurement.

---

## 9. Build sequence

Each step gated on the previous, each producing evidence before code depends on it.

1. **Offline scoring harness** with the official formula — the instrument. Nothing
   downstream is trustworthy without it.
2. **Recon engine**: segmentation-reduced action space, tiered frontier, state graph,
   aliasing detection. Measured on the 25 games by *scored* outcome.
3. **Transfer carrier**: what survives a level boundary, and measured proof it reduces
   level-2 search versus a cold start. **This is the bet — it is measured before
   anything is built on top of it.**
4. **Phase controller** with budget policy.
5. **Floor-safe handoff** to the duck.
6. **`competition_sim` validation**, then a scored slot.

Steps 1–3 answer the question this design exists to test. If step 3 fails the kill
criterion, we stop, ship the conservative policy, and record the refutation.
