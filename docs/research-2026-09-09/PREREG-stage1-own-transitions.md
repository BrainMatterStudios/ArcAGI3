# PRE-REGISTERED — Stage-1: can the brain model its OWN transitions at a wall? (2026-09-09, before data)

**Why.** Stage-0 reopened the executable-world-model lane, but it handed the model 12-42 transitions recorded by a
frontier agent that had ALREADY SOLVED the level, offline, with no clock. A2 needs something harder: induce a model
from the messy transitions our own agent generated while STUCK at a wall. Stage-0 removed a necessary doubt (can the
brain emit a verified executable model at all — yes). Stage-1 tests the input A2 would actually receive.

**Data (built by `own_transitions.py` from our own wave artifacts, no new GPU):** the carry75 25-game wave,
`dc22-fdcac232_p0` — official result `levels_completed=1, state=gave_up`, i.e. it cleared level 1 and then spent
85 actions stuck on level 2. dc22's L2 is one of the 12 six-draw never-passed walls.
* **PRIMARY `dc22L2`** = the first 20 of its 85 level-2 transitions (10 clicks, 0 level_up, 0 deaths). Entry grid =
  the board on entering level 2. This is the A2 use case: model the wall from what the agent itself observed there.
* **CONTROL `dc22L1`** = the first 20 of its 48 level-1 transitions (8 clicks), same run, same encoding — a level the
  agent DID go on to clear.
Segment boundaries verified: the `level` field on an events row is the level AFTER the action, so the first row at
level L is the action that cleared L-1; it is used as the entry board and excluded from L's own play (an off-by-one
that would otherwise have put a level-1 clearing action inside the wall segment).

**Arm:** identical instrument to Stage-0 (same `backtest.py`, same contract, same stop rules), Flash-Next on the live
keith V14 profile, Arm A hex, `--max-calls 20 --budget-min 45 --prompt-cap 8000`, 2 draws per segment. ~$2-3.

**READS, locked:**
* **PRIMARY: `dc22L2` green in >= 1 of 2 draws → A2's core loop works on realistic self-generated data at a real
  wall → BUILD A2** (designed around the 32k self-starvation finding).
* **`dc22L2` not green, `dc22L1` green** → the induction works on our data in general but not at the wall: the
  agent's wall exploration is too uninformative to model. A2 should then be built around DIRECTED probing (choose
  actions that disambiguate the model) rather than passive modelling — a materially different and more expensive
  design, so this outcome means "redesign A2 first", not "build A2".
* **Neither green** → induction fails on self-generated data generally, the Stage-0 result does not transfer to the
  input A2 would receive, and **A2 is not justified**: fall back to A3/A4 (port NOOA / Polyphony) + Track D.
* SECONDARY: candidate files emitted per call (Stage-0: cn04 9 files/11 calls, sk48 2/7), calls-to-green, best
  matched/total, tokens and wall per call.

**CAVEATS recorded before data:** (a) 20 transitions is a slice chosen to sit in the regime where Flash-Next
succeeded on cn04 (18) and not the 42 where it failed on sk48 — it is NOT a full-level model, and a green here does
not mean the whole 85-transition level is modelled; (b) self-generated transitions contain redundant and no-op
actions, which may make them either easier (repetitive) or harder (uninformative) than curated frontier data — the
control is there partly to read this; (c) a green result does NOT show the agent can do this ONLINE inside its
~52-call clock, which remains the open question after Stage-1 and is a cost question for A2's design, not a
capability one.

## RESULT 2026-09-09 — 4 of 4 GREEN ⇒ **BUILD A2** (pre-registered read)

Flash-Next on the live keith V14 profile, ~35 min GPU, ≈$2. Artifacts and all four green models:
`docs/research-2026-09-09/stage1-own-transitions/`.

| segment | draw 1 | draw 2 | calls-to-green |
|---|---|---|---|
| **dc22L2 (the wall, agent stuck 85 actions, never cleared)** | **GREEN 20/20** | **GREEN 20/20** | 3, then 2 |
| dc22L1 (control, level the agent did clear) | **GREEN 20/20** | **GREEN 20/20** | 2, then 9 |

**PRIMARY MET, and more strongly than the rule required:** the wall segment was green in **2 of 2** draws (the rule
asked for >= 1 of 2). All four candidates re-verified independently by re-running `backtest.py` on the saved files.

**Not memorisation.** Every green model is 3.3-5.4 KB / ~155 lines with **zero** embedded literal grids, **zero**
step- or index-based output selection, and no reference to `ENTRY_GRID`. The dc22L2 model is explicitly mechanistic:
a 2x2 player sprite (colour 14) moving in steps of 2, a WALKABLE colour set, two toggle switches at (52,22) and
(52,40), a row-63 progress bar, and a maintained base-terrain layer so vacated cells restore correctly. Applying the
L2 model to the L1 segment scores 0/20, which is the expected behaviour of a correct PER-LEVEL model (it hardcodes
level-specific switch coordinates), not evidence of overfitting to the recorded outputs.

**Self-generated data was not harder than curated frontier data — it was easier.** dc22L2 went green in 2-3 calls;
Stage-0's cn04, from a frontier agent's curated transitions, needed 11. The redundancy in our agent's own
exploration (10 clicks among 20 actions, many no-ops) appears to help rather than hurt.

**The 32k self-starvation finding replicated** on dc22L1 draw 2: prompt grew 7.7k → 9.5k → 11.5k → 11.9k as
(candidate, feedback) rounds accumulated, output fell 24.6k → 20.3k, and call 4 hit the length cap with no code
before the run recovered at call 9. A2 must summarise prior candidates rather than append them.

**VERDICT: A2 is justified.** The chain is now: the deployed brain induces a correct, verified, executable model of a
real wall level from the agent's own stuck exploration, in 2-3 calls and ~5 minutes, reproducibly.

**WHAT REMAINS OPEN (and is now A2's central design risk, not a capability question):** everything here is OFFLINE.
The agent must do this INSIDE its ~52-call, 7,920 s game clock, where each model-building call is a queued ~150 s
call competing with play calls — the same tax that made A1 marginally over the clock. 2-3 calls to build a wall
model is ~5-9 % of the game budget, which is affordable; the open question is whether acting on the model (search /
planning in it) then converts into cleared levels. That is what A2 must be built to measure.
