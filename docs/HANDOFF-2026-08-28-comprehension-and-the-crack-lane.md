# HANDOFF 2026-08-28 — START HERE

Supersedes `HANDOFF-2026-08-27-search-lane-and-measurement-laws.md` as the entry
point. Full working record: `docs/RESEARCH-2026-08-27-the-two-level-wall.md`.
Memory index: `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/MEMORY.md`.

Everything below was measured on 08-27 against the live Kaggle API, the
`arcengine` source, the 25 public games, or our own recorded waves.

---

## 0. FIRST: the clock

`date -u` said **2026-08-27 21:09 UTC** while the harness date notice said
08-28. **The harness date runs ahead — verify with `date -u` before treating a
slot as missed.** This is the second time; it cost an armed runner on 08-25.

## 1. TONIGHT (2026-08-28 00:01Z): v8 crack-or-nothing — STILL ARMED, DO NOT TOUCH

- Runner `scripts/submit_v8_20260828.py`, **pid 29105**, alive, detached.
  Marker `logs/v8_20260828.marker` unclaimed. Kernel
  `ahmedmobasher86/arc3-duck38-v8` v1, **svid 345333330**; the runner
  re-attests and aborts if the bound scriptVersionId does not match.
- 08-27's census independently **validated** this arm's narrow configuration.
  No better candidate exists — every alternative is a measured negative (§3).
- **Reading (pre-registered).** Each banked crack is a flat **+1.82**:

  | cracks | LB |
  |---|---|
  | 0 (most likely) | ~1.4 — base band, floor measured 0.00 |
  | 1 | ~3.2 |
  | 2 | ~5.0 |

  Zero cracks makes 3 unreachable (11-draw ceiling 1.74). One crack clears 3
  from any draw above 1.18 — 9 of our 11 draws. A null is EXPECTED and is
  evidence about hidden-set composition, not about the mechanism.
- Honest caveat to carry into the read: `ft09_gf2`'s detector fires on 1/25
  public games with **zero false positives**, but every one of those 25 is a
  game it was developed against. **Its true-positive rate on an unseen
  same-class game is unmeasured.** If it generalises poorly, effective p ≈ 0
  regardless of class frequency.

## 2. The two reframings that changed the campaign

**(a) COMPREHENSION IS THE CONSTRAINT, NOT BUDGET.** The depth-by-budget thesis
is refuted by its own controlled test. Patch 21 (`TAAF_STRUCT`, the mandatory
plan channel) bought **×1.40 actions/game and cleared 25% FEWER levels**
(ft09 28v28; 36 → 27), and **88% of every extra action went into a level that
was never cleared**. On the 25-game wave the games that clear nothing spend a
median **0.93× the level-1 human baseline** — they are not starved. On a game
it understands the agent plays at **0.86× human pace**; on one it does not, no
budget helps. ⇒ plan channels, batching and inspection-budget work are all
**closed**. `actions/game` is a diagnostic, never an objective.

**(b) A CRACK MUST BE COMPUTED, NOT SEARCHED.** Census of 25 games on the live
cost model (reset-replay, 900 s/game): **48 levels, 2 cracks, deployable
inventory = 1.** ft09 cracks at **1,231 actions** via its specialist; tu93
cracks at 84,685 by generic search but is undeployable (16% over the engagement
cap *and* no detector to pre-screen on). Every other game costs 0.4M–3.5M
actions and cracks nothing, against ~73k affordable.

Design law from three solvers:

| specialist | shape | cost |
|---|---|---|
| `ft09_gf2` | perceive → linear system → execute | 1,231 — cracks |
| `wa30_grabdrag` | perceive → A* on a **model** → execute | 380 for a depth-26 level |
| `sc25_glyph` | perceive → cast → **BFS on the real env** | 15,703 for level 1 alone |

**A specialist that plans in a model is affordable; one that searches the
environment is not.** Env-search re-imports an 11–17× replay tax (one reset per
node, ~94% of actions replay — three attempts to shave it returned 0 nodes,
9 actions and 4 actions respectively).

Growing the inventory = shipping **(cheap frame-0 detector + model-based
solver) pairs**. The detector side is already solved: all four screens fire on
exactly their own class with zero false positives.

## 3. Threads retired on evidence — do not reopen without new facts

- **Recipe parity.** The public lane's own authors cap at 1.9–2.4
  (foysal 2.23/97 subs, keithtyser 2.13/69, thtennant v22+v26 1.93/36).
- **winframe / carryover bundle.** The bug is verified three ways, but
  **0 of 52** post-win turns show it binding for us — the result dict already
  reports `level_completed: true`, so the agent is never confused about
  *whether* a boundary happened.
- **Depth by budget**, in every form (see §2a).
- **Adding wa30/sc25/tn36 to tonight's engaged set.** Each cracks only 2 levels
  then burns (97k / 278k / 909k actions) — a failed engagement is a
  level-weighted step function, −1.19 on level 1 to −12.14 on level 3.
- **"Search on a free copy" live.** Offline `deepcopy(env)` is free and unbilled,
  but `Arcade.make` returns a *remote* wrapper in COMPETITION mode; the state is
  server-side. Closed.

## 4. THE ACTIVE BUILD: wa30, and exactly where it stands

Target: a **≤100-move plan for wa30 level 3, verified on the engine** — zero
slots. Cracking wa30 makes the lottery 2-class instead of 1-class, which on the
§1 table is the difference between ~3.2 and ~5.0.

**What level 3 is** (this took several wrong turns; the final account is
§12c-quater of the research doc):

- A wall column at **x=32** divides the board. The avatar's free-walk region is
  **x ≤ 28, measured before AND after** the carrier acts.
- **The avatar can never place a block on a pad** (pads are at x=52/56). Only
  the carrier can. The shipped `solve_wa30` plans avatar→pad legs, so its
  169-move plans are not merely long — they are unexecutable.
- Blocks, unlike the avatar, **can be pushed into the column**. It is a
  **handoff channel**: the avatar pushes in from the left (stand x=24, drag
  28→32), the carrier collects from the right.
- The blocks starting at (32,12)/(32,32) are already in the channel — which is
  why the carrier delivers exactly two for free, then idles.

**A general mechanic nobody had modelled:** `wa30.py:1198 dhrikuybfo()` runs
after every player action and drives autonomous carriers (`kdweefinfi`, plus
`ysysltqlke`) that move blocks one cell per player action. Free deliveries per
level, measured: L1 **0/3 (no agents — which is why drag-only cracks L1 and
nothing else)**, L2 4/5, L3 2/5, L4 1/7, L5 4/6, L6 1/2, L8 5/13, L9 4/9.

**State of `wa30_macro.py`:** correct HANDOFF macro, frame-derived board model,
flood-fill deadlock prune, inert wait action, and a progress key that counts a
handed-off block as progress (without it every handoff scored zero and
best-first had no gradient). Reaches `(3 needing the avatar, 4 not on a pad)` at
13 moves. **No ≤100 plan yet.**

**Next, in order — search quality, not modelling:**

1. Simulate the carrier's next pick (deterministic: nearest reachable unclaimed
   block by its own BFS) so `WAIT` is scored by what it will achieve.
2. Put carrier travel distance into the search key — channel rows are not
   equivalent and currently tie.
3. Re-run. Avatar legs are 7–13 moves with the correct model and the carrier
   ferries in parallel; nothing yet says 100 is out of reach.

Also open on this game: `sc25` L3 fails because the **cast is wrong**
(perception is fine, the maze BFS exhausts a fully-explored 1,795-state space).

## 5. Standing laws added 2026-08-27

1. **Actions are not levels.** Any lever that moves a process metric is unread
   until its level count sits beside it. `probe_budget.py` prints this itself.
2. **Action accounting:** count actions from the turn-header delta
   (`--- analysis_step=N | action=M |`), never from `executed_count` inside
   `[TOOL RESULT]` — those blocks vanish when history trims (they matched
   ground truth in 1/8 struct transcripts; the header matched 16/16).
3. **10 of 25 games enforce a per-level move budget** (`StepCounter`, enforced
   at `wa30.py:1252` / `tu93.py:1270`, checked *after* the win test). On **30
   levels the published human baseline EXCEEDS that level's own budget** ⇒
   **the baselines count multiple attempts**, and a clean single-attempt solve
   always saturates the efficiency cap.
4. **`ONLY_RESET_LEVELS=true` must be in the environment before `arcengine`
   imports** (`run_falsifier` sets it at import). Without it every
   `backend.root()` silently resets to level 0 — it invalidated two experiments
   before I caught it.
5. **A null optimisation that corrupts a measuring instrument is a loss.** The
   reset-replay path cache saved 9 actions of 58,522 and broke
   `test_reset_replay_cost_model`; reverted.

## 6. Instruments (all zero-slot, all offline)

| tool | question it answers |
|---|---|
| `submission/_boundary_probe/probe_boundary.py` | does the agent get confused at a level boundary? |
| `submission/_boundary_probe/probe_budget.py` | where does the action budget go? (never read without level counts) |
| `submission/_boundary_probe/probe_plans.py` | why did extra actions not become levels? |
| `submission/_search_core/step_budgets.py` | per-level move limits and what they imply |
| `submission/_search_core/run_falsifier.py` | the crack census on the live cost model |
| `submission/_search_core/wa30_{beam,coop,macro}.py` | the wa30 build (first two are measured negatives) |

## 7. Repo state

Branch `winning/duck-patched`, commits `6606bc3 … ff32df1`, **not pushed**
(push gate per CLAUDE.md). No shipped file was modified by the 08-27 work
except `search_core.py`'s firewalled `max_depth` (default `None` =
byte-identical; measured null). 33/33 search-core tests pass.
