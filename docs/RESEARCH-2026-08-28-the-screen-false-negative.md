# RESEARCH 2026-08-28 — the v8 null, and the false negative underneath it

Working record for 08-28. Everything below was measured today against the live
Kaggle API, the `arcengine` source, the 25 public games, or the 13-game
arc-interactive holdout corpus.

---

## 1. The slot fired and read null

Submission **55829275**, `COMPLETE`, public **1.65**. The runner behaved
exactly as armed: re-attested `scriptVersionId 345333330` at 00:01:02Z, claimed
the marker, gated on kernel COMPLETE, settled 10 min, submitted 00:11:07Z with
`rc=0`.

Against the pre-registered table (0 cracks ~1.4 / 1 crack ~3.2 / 2 cracks
~5.0), **1.65 is a null — zero cracks banked.**

Series context, n=12 pooled across arms: mean **1.409**, sd **0.245**,
**CV 0.174** — an independent re-confirmation of the CV≈0.20 law. 1.65 is
z=+0.98, third-best draw, still under the 1.74 LB best. No re-bank, no harm.

## 2. There is NO channel out of a scored rerun — the logging idea is dead

The obvious follow-up ("log the screen's decisions so the next null is
readable") **cannot be built.** Kaggle exposes no output files for a
competition rerun: `kaggle competitions` has `logs` only for simulation
episodes, and no submission-output endpoint of any kind. Nothing the notebook
writes during the scored run is ever readable afterward.

Corollary worth carrying: `kaggle kernels output <kernel>` returns the
**commit-time run**, not the scored rerun. Today's fetch returned
`benchmark.json` with `label: duck38-v12-smoke`, 3 public games (vc33/sb26/
tn36), `OperationMode.OFFLINE`, 08-27 09:33→10:26. Reading those three
`pre-screen DECLINED` lines as evidence about the hidden set would have been a
straight error — they are a 1-hour smoke declining games it was always going to
decline. **Check `benchmark.json:label` before drawing any conclusion from a
kernel-output log.**

So the generalisation question had to move offline. It did, and it paid.

## 3. THE FINDING: the screen throws away a crackable game

`submission/_search_core/holdout_screen.py` (new, zero-slot) runs the frame-0
screen and then the real `specialists.detect_matrix` over
`scratchpad/holdout_arcint` — **13 arc-interactive games that took no part in
any of our development.**

**Result: `cx01` is a genuine ft09_gf2 game that the screen DECLINED.**

| | ft09 (dev) | cx01 (holdout) | all 36 negatives |
|---|---|---|---|
| `detect_ft09` | True | **True** | False |
| screen verdict (old bars) | admit | **DECLINE** | decline |
| ring-isolated tiles `n_strict` | 32 | **3** | **0** |
| dominant strict phase | 16 | 2 | 0 |
| `block_frac` | 0.50 | 0.308 | ≤ 0.208 |

`detect_matrix` on cx01 returns `{ft09_gf2: True, tn36: False, sc25: False,
wa30: False}` — same class, unambiguously, and no other detector fires.

**And it cracks.** `run_game('cx01', 900s, portfolio)`:

| backend | levels won | cracked | actions | wall |
|---|---|---|---|---|
| snapshot | 7 / 7 | **yes** | 2,408 | 19.1 s |
| **reset_replay (live cost model)** | **7 / 7** | **yes** | **7,045** | 2.1 s |

7,045 actions against ~73k affordable — under 10% of budget. **The deployable
crack inventory is 2, not 1.** The second one was invisible because the screen
declined it at zero actions.

### Why the screen failed, and why the fix is not a hack

The published headline — 1 TP / 0 FN / 0 FP — was measured on the 25 public
games, **every one of which the screen was developed against**. That is an
in-sample fit, not a generalisation estimate, and the first out-of-sample test
broke it in the failure direction the module's own contract calls fatal
("being wrong in the decline direction destroys the whole value case").

The **signal** was never wrong. Across all 38 games in both corpora `n_strict`
is categorical:

    positives:  ft09 32,  cx01 3
    negatives:  all 36 games exactly 0

What was wrong were the **branch-A thresholds**, both set to 4 by reading
ft09's own dense board. cx01's board is sparse (3 ring-isolated tiles, 2 on the
dominant phase) and fell straight through. Fix: split the coupled constant and
lower branch A to the smallest values that admit the known positives —

    MIN_STRICT = 3            # was MIN_TILES = 4
    MIN_STRICT_LATTICE = 2    # was MIN_LATTICE = 4

The retained margin is **3-vs-0 — categorical, not a numeric hair**. `MIN_TILES`
(the raw block-count gate) and all of branch B are untouched, so branch B keeps
its conservative `FRAC_MIN = 0.35`.

Post-fix confusion matrix over both corpora: **2 TP / 0 FN / 0 FP** (dev 1/25 =
ft09; holdout 1/13 = cx01).

### Honest caveat on this fix

Tuning branch A on cx01 **spends the holdout's independence for the ft09_gf2
screen.** It is no longer a clean out-of-sample corpus for that screen, and the
2 TP / 0 FN / 0 FP above is partly in-sample again. Two things make this the
right trade anyway: the alternative was knowingly shipping a screen that
discards a crackable game, and the threshold is not being fitted to noise — the
negatives are categorically zero, so any positive bar separates them.

The next genuinely independent test needs a corpus we have not touched.

## 4. What this does to the reading of the 1.65

The pre-registered reading offered two worlds. There is a third, and it was
measured today:

- **world A** — no ft09-class game in the public half; the ticket never came
  up, the lane is untested.
- **world B** — the screen fired, the solver failed; the lane is dead.
- **world C — the screen met an ft09-class game and silently declined it for
  free.** Measured false-negative rate on out-of-sample same-class games
  **1 of 1**.

World C is now the *best-evidenced* of the three, because it is the only one
with a measurement behind it. The 1.65 is therefore **much weaker evidence
against the crack lane than the pre-registration assumed** — the arm may never
have been given the chance the EV case was priced on.

This does not license re-flying the arm on optimism. It licenses re-flying it
**with the corrected screen**, and it means the ft09_gf2 class is roughly twice
as common as the 1-in-25 estimate the whole EV table was built on (2 of 38
games across both corpora ≈ 5.3%, vs the 4% assumed).

## 4b. wa30 LEVEL 3 IS SOLVED — 82 moves, engine-verified

The standing falsifier ("a ≤100-move plan for wa30 level 3, verified on the
engine") is **met**: `wa30_macro.py 3 8 900` returns an **82-move plan inside
the level's own 100-move budget**, and the script's clean-room replay from a
fresh environment confirms `win`.

    27 moves: todo=2 unplaced=3 carrier_eta=11
    50 moves: todo=1 unplaced=2 carrier_eta=6
    66 moves: todo=0 unplaced=1 carrier_eta=7
    82 moves: WIN     (4 macro expansions, 9 s)

**The handoff's prescribed next steps were not the constraint.** Both were
implemented — carrier travel in the search key (via the engine's own
`czrprbohhe`/`cyjrduhzmz`, not a reimplementation) and an adaptive WAIT of
`eta+1` — and profiling then showed they could not have been the blocker:

| component | ms / expansion |
|---|---|
| **`deliver_legs`** | **8052 — 99.6%** |
| `idle_action` | 30.9 |
| `deepcopy(env)` | 3.3 |
| `progress_key` incl. carrier ETA | 0.4 |
| `all_blocks_viable` | 0.5 |

Four expansions fitted in 600 s, and at level-3 start the planner produced
**0 handoff legs and 16 pad legs** — the exact inverse of what §12c-quater says
is physically possible.

**Root cause: `sp._DragModel` carries ONE wall set and applies it to both the
avatar and the block.** A sprite-by-sprite trace of one executed leg:

    avatar : (16,36) -> (16,32) -> (28,32) -> (28,24)      never passes x=28
    carrier: (48,12) -> (44,12) -> (40,12) -> (56,12) -> (56,24)
    block  : (32,12) -> (36,12) -> (52,12) -> (52,24)      moved by the CARRIER
    block  : (32,32) -> (32,24)                            pushed INTO the divider

So §12c-quater is right — the avatar never reaches a pad, and the block that
landed on one was the carrier's. But the shared wall set let A* believe the
avatar could escort a block to x=52, so every expansion re-derived 16 legs the
engine refused, at ~50 ms each.

Two fixes:

1. **`TwoWallDragModel`** — separate `avatar_walls` / `block_walls`. The avatar
   cannot enter the divider; a block can be pushed into it.
2. **`handoff_cells` returns EVERY divider cell, including ones that read as
   wall.** The old version returned only non-wall gaps, which on level 3 are
   exactly the two cells the starting blocks plug — so after filtering there
   were zero handoff targets and the handoff macro could never fire. The engine
   settles it: a block was pushed to (32,24), a cell the frame reads as solid.

Result: 16 fiction legs -> **81 real handoff legs**, each landing the block on
exactly the requested cell when executed. Pad legs vanish on their own, which
is correct — only the carrier delivers. A flood-fill reachability gate replaces
the pad loop's 160 A* calls.

**Law:** when a planner's legs do not survive execution, suspect the model's
*body semantics* before its search. One obstacle set for two bodies with
different passability is a phantom-wall bug in either direction.

## 4c. wa30 chained to 3/9 — and exactly why level 4 does not fall

`wa30_macro.py chain N T` runs shipped A* first and the macro planner only on
its failures. **L1 28 actions/0.1 s, L2 201/2.7 s (shipped), L3 82/100 moves
(macro) — 3 of 9, one better than the census's 2.**

Two more instances of the SAME one-body error surfaced getting there, making
four in total across the planner:

| component | collapsed two bodies into one obstacle set | symptom |
|---|---|---|
| `_DragModel` | avatar allowed to escort a block to x=52 | 16 fiction legs, 99.6% of runtime |
| `handoff_cells` | only non-wall gaps are targets | 0 handoff legs; macro never fired |
| `all_blocks_viable` | blocks cannot enter the divider | **L4 dead at start; 0.5 s of a 900 s budget** |

L3 had only passed `all_blocks_viable` by accident: its two divider cells held
BLOCKS, which `engine_percept` subtracts from the wall set, leaving a hole the
flood fill could cross. L4 has no such hole. Goals now include the handoff
channel — reaching the divider IS placement, since the carrier ferries the rest
— which is how `progress_key` already scored it.

(A fourth was mine: `chain()` read `core._wa30_avcolor` before any
`solve_level` had run, so it was always `None` and `plan` bailed instantly. A
missing input presenting as a stall.)

**Why L4 still does not fall — three things ruled out and one cause found.**

- **Not beam width.** Beam 8 and beam 32 produce a *byte-identical* trace —
  same 7 expansions, same stall. Children are being discarded, not trimmed.
- **Not the budget, at first.** It reaches `todo=3` at 41 moves with 59 of its
  100 left and shortest legs of 6-8.
- **The deadlock prune is net-negative here.** Prune ON stalls at `todo=3` by
  51 moves; prune OFF reaches `todo=2` by 79. Now a switch —
  `WA30_MACRO_PRUNE=0` — defaulting ON, because level 3 genuinely needs it.
- **THE CAUSE: L4 has almost no free labour.** Per-carrier ETA at level start
  is `[99, 5, 99]` — **two of its three carriers have NO path to any of the 25
  staging cells before the avatar has moved at all.** L3 solved because one
  carrier ferried continuously while the avatar worked; L4's avatar must
  personally handle ~6 of 7 blocks inside 100 moves at ~12 moves per leg. That
  matches the independently measured free-delivery count (L4 = 1/7).

So L4 is an EFFICIENCY problem, not a search-quality one: the remaining gain
has to come from shorter handoff legs (better divider-cell choice), not from
waiting on carriers that cannot move. **wa30 is not a third crack**: a crack
needs all 9 levels, and L5-L9 are unexplored.

**Deployability is a separate, unstarted problem.** `plan()` uses raw
`copy.deepcopy(env)` — `SnapshotBackend` semantics, offline only
(`search_core.py:30`). Live, `Arcade.make` returns a remote wrapper with
server-side state and "search on a free copy" is closed. A port means routing
macro execution through the existing backend-agnostic
`sp.chain(backend, handle, toks)` and pricing it under `ResetReplayBackend`.

## 5. Standing laws added today

1. **An in-sample confusion matrix is not a generalisation estimate.** Any
   detector/screen quoted as "N TP / 0 FP" on the corpus it was built from is
   unvalidated until it is run on games it has never seen. The first
   out-of-sample test of our best-validated screen found a fatal-direction
   error.
2. **`kaggle kernels output` returns the COMMIT run, not the scored rerun.**
   Check `benchmark.json:label` and the game count before reading anything from
   it. There is no way to retrieve a scored rerun's artifacts.
3. **Where the engine exposes its own model, call it — do not reimplement it.**
   `wa30`'s carriers are driven by `czrprbohhe`/`cyjrduhzmz`, methods on the
   live game object; the planner runs macros on a `deepcopy` and can invoke
   them directly. A reimplementation can disagree with the engine; a call
   cannot.

## 6. Instruments added

| tool | question it answers |
|---|---|
| `submission/_search_core/holdout_screen.py` | does the screen generalise off the dev corpus? (`--detect` adds the false-negative cross-tab) |

Results: `submission/_search_core/results/holdout_screen.json`.
