# What our agent actually does — 2,047 real tool calls, 2026-09-10

Measured on the M0 corpus: every tool call the agent made **on a level it went on to CLEAR**,
pooled across 247 runs and 11 waves. These are its *successful* calls, not its failures.

## 1. The loop, as the code runs it

Per turn: `play()` checks stop conditions → auto-RESETs if the engine says GAME_OVER (one game
action, no model call, the model is never consulted and RESET is filtered out of `valid_actions`) →
writes runtime state → calls `analyze()`. `analyze()` assembles `[system] + history + [user prompt +
a 256x256 PNG of the grid]`, posts it, gets back reasoning plus one tool call, runs the emitted
python in a fresh isolated subprocess, and **ends the turn on the first tool call that fires a game
action**. One acting call per turn, maximum.

Inside the sandbox the agent is handed `current_frame` (ascii + segmentation objects), `previous_frame`,
`history`, and — purpose-built for dynamics learning — **`transitions`**, a list of real
`(before_frame, action, after_frame)` triples. `action(list)` fires any number of game actions, with
no cap.

## 2. What it actually writes

| what the code does | calls | share |
|---|---|---|
| perception: ascii slicing / printing | 2,047 | 100 % |
| perception: segmentation objects | 1,086 | 53.1 % |
| observation: diff previous vs current | 563 | 27.5 % |
| **search: bfs / dfs / queue / shortest path** | **46** | **2.2 %** |
| search: builds a graph or map | 21 | 1.0 % |
| **forward-simulates a next state** | **0** | **0.0 %** |
| verifies a prediction against reality | 5 | 0.2 % |
| fires `action()` | 1,161 | 56.7 % |

Median code length 9 lines. Median action-list length where it acts: **1**. Batches of ≥5 actions:
0.3 %. `transitions` — the harness's own dynamics affordance — is touched in **3.2 %** of calls.
43.3 % of calls fire no action at all.

## 3. The finding

**Our agent is a reactive perceive-and-act loop. In 2,047 successful tool calls it forward-simulated
a next state exactly zero times.** It looks at the board, sometimes diffs it against the previous
board, fires one action, and prints something. It almost never searches (2.2 %) and never predicts.

This is not a budget problem, a knowledge problem, or a tooling problem:
* **Not tooling** — `transitions`, `previous_frame` and an unbounded `action()` are all already there
  and mostly unused.
* **Not capability in isolation** — Stage-1 showed this exact brain writing backtest-green transition
  models offline in 2–9 calls when handed clean transitions.
* **Not persuadable** — A2 gave it a workspace and a backtest tool: 0 verifier calls in 358. The
  harness-directed variant produced 0 models in 24 calls even inside Stage-1's own working window.

So the brain *can* model offline, *never* models live, and *cannot be pushed into it* by tools,
prompts or direction. That is the sharpest statement of our ceiling we have, and it explains a lot of
past nulls at once: macro-batching failed because there was no model to plan a batch with; the
workspace failed because verification is not in the live repertoire at all; probe discipline failed
because the missing thing is an operation, not a habit.

## 4. Where an improvement could live, ranked

**(a) Let the HARNESS do the modelling the agent won't.** The harness already holds every
`(before, action, after)` triple. Fitting a cheap object-level effect table — which segmentation
nodes moved, appeared or vanished per action, and by what offset — is pure code, costs zero model
calls, and hands the agent the dynamics summary it demonstrably never computes. This is the only
proposal here that needs **no behaviour change from the model**, which matters because six
independent experiments say behaviour change does not stick. Cost: a few hundred prompt tokens per
turn. Honest prior: moderate at best — it is still information-in-the-prompt, the class that has
failed repeatedly — but it is mechanically different from restating knowledge, because it supplies a
*derived* quantity the agent never derives.

**(b) Relax the verification contract.** Stage-1 demanded exact reproduction of 20 full 64x64
transitions. That is why its models cost ~65k tokens AND why they memorised coordinates and
transferred at 0–45 % across levels. A *partial* model — predict one feature of the next state, e.g.
the avatar's position or whether a move is legal — is a few lines instead of hundreds, and is enough
to plan a path. This reframes A2's failure: we asked for something two orders of magnitude beyond
the agent's 9-line median, not for something it refused.

**(c) Free observation.** 43 % of calls fire no action and are pure inspection; 27.5 % hand-roll a
frame diff. Computing the diff in the harness and putting it in the tool result would convert some
inspection calls into acting calls. Bounded by the measured elasticity (~0.4), so worth tens of
percent at most.

**(d) The level-transition wipe.** Verified: `world_model`, `goal_model`, `action_model`,
`recent_findings`, `open_questions` and `current_plan` are all blanked at every level transition.
Untested to fix — but the cross-level transfer test says level-specific structure genuinely does not
carry, so the wipe may be correct.

## 5. What this does NOT support

Nothing here reaches a step change on its own. Every mechanism above is bounded by the same
elasticity that capped the throughput family. The measurement's real value is diagnostic: it says
our gap to the leaders is a **loop-shape** gap — they run a model-based loop, we run a reactive one —
and every route we have tried for changing our loop's shape from the inside has failed. That is the
strongest argument yet for Track D absorption being the structurally correct answer rather than a
fallback.
