# DESIGN 2026-08-14 — Engineered effect-model + search agent (brief for review)

**Status: DESIGN ONLY — nothing built. Deliverable for Ahmed's go/no-go.**
Scope: an LLM-light agent that learns a transition model of each game *during play* and
plans over it, with the 27B duck as a per-game fallback. Inspired by (a) the leader
hypothesis — team **cstl, 2.70 public, rank 1, 25 submissions** (leaderboard snapshot
`.playwright-mcp/page-2026-08-14T11-13-58-518Z.yml`; members Tehnar + TG, competitive-
programming profile per campaign intel) — and (b) preview winner StochasticGoose.
Calendar: final deadline **Nov 2** (memory `arcagi3-competition-facts`) → ~11 weeks.

---

## 0. Why this is worth a design at all

Every LLM-side lever is closed by measurement (`docs/HANDOFF-2026-08-10-three-levers-closed.md`):
RESET replay server-blocked, distillation compresses deliberation, instrument repaired and
no killed arm rescued. Duck base distribution: **mean 0.965, sd 0.208, best 1.30 (n=10)**.
The leader sits at **2.70** — 2.8× our mean — and Kojima's 1.86 is a best-of-46 draw. Under
the verified scoring law — per level `min(115, 100·(b/a)²)`, weighted **(level+1)**, capped
by completion share (`scorecard.py:170-206`, memory `arcagi3-eval-noops-and-scoring-cap`)
— 2.70 requires *deep completion at near-baseline efficiency* on many games. An LLM paying
one deliberation per keypress structurally cannot produce near-baseline action paths; a
planner that has identified the transition function can. That is the simulator-fitting
hypothesis for cstl, and it is the one paradigm this repo has serious prior art in.

## 1. Prior art audit — what the repo already proved (and what killed it)

| Fact | Evidence |
|---|---|
| Search over the real env solved **all 25 dev games** eventually; 13/25 at L0 zero-code, palette-invariant | `HANDOFF-2026-07-01:16-18`, memory `arcagi3-all-dev-games-cracked` |
| Snapshot oracle: tu93 FULL L1→L9 in 57 s offline | `HANDOFF-2026-07-02:43-46` |
| **The paradigm's scoring trick is dead**: dirty-run/clean-replay needed max-over-plays; at eval runs=1, second play = HTTP 400 | memory `arcagi3-action-accounting-measured` |
| Actions **accumulate** across attempts of a level; RESET costs 1; a level never completed scores 0 → failed search is free; sacrificing early low-weight levels is net-positive (break-even S ≥ 100·j/(j+1)) | same memory |
| HUD bars tick into frames on **≥24/25 games**; masking gives **16.14× state collapse**; with masking + animation control, **~12 games are frame-Markov — Dijkstra over masked boards is a complete optimal planner**; 6 have genuine hidden state (g50t m0r0 sc25 bp35 sk48 ar25); null-cycle fraction 0.765 | memory `arcagi3-hud-breaks-frame-identity` |
| Role-typed perception flipped goal-naming 0/9 → 6/6; role filter cut false goals ~9→1 | memory `arcagi3-role-typed-perception` |
| CLICK dominates: 19/25 dev games need ACTION6 | memory `arcagi3-click-dominates-distribution` |
| Click effects are deterministic (10/10) and sparse but **non-local**; the click-target→affected-target model was **never built** | `HANDOFF-2026-07-02:15-18` (addendum #3) |
| Wiggle/contingency battery: avatar ID **14/14**, archetype dispatch 92–100% at **mean 5.0 actions/game**, offline-verified GO | memory `arcagi3-research-2026-08-07-human-play-sweep` |
| Hidden games expose public `tags` (`api.py:56-77`) → dispatch has a real channel on the private set | memory `arcagi3-eval-noops-and-scoring-cap` §4 |
| 340 full per-step human replays, all 25 games, with behavioral budgets already extracted (probe-before-first-level: click ~20 [p90 38], avatar ~31 [p90 94]; actions/completed-level median click 36, avatar 70) | memory `arcagi3-human-replay-dataset` |

**StochasticGoose, studied.** Local clone: `scratchpad/external/ARC3-solution/`
(`custom_agents/action.py` 489 lines, `view_utils.py` 317). Verified architecture: 4-layer
CNN, BCE frame-*changed* classifier (no value head), hierarchical action→coordinate spatial
decoder, hash-dedup buffer, **model+buffer reset at every level boundary** (memory
`arcagi3-stochasticgoose-facts`). Two hard caveats:
1. **No LICENSE file** — repo page re-checked 2026-08-14: none declared. Default
   all-rights-reserved → **ideas only, no code vendoring**. We already hold a clean-room
   reimplementation (`src/arcagi3/stochastic_goose_explorer.py` + `tests/test_stochastic_goose.py`).
2. **It loses to our own retired explorer.** Faithful offline run (2026-06-29): goose L2
   total vs TransferExplorer L16 on 8 dev games, same budget — and the reward-RL variant
   was killed at T4 scale (regressed all 6 games). The published 3rd-place reference also
   lost to us (L4 vs L19). So the design below does **not** copy goose; it adopts exactly
   two of its ideas — the frame-change objective as an *exploration prior*, and the
   *per-level model reset* as the W1 escape — on top of the graph backbone that our own
   A/Bs proved is the load-bearing part (memory `arcagi3-033-ceiling-is-codebase-not-paradigm`).

## 2. Architecture

**Principle: graph-first, model-second, LLM-last.** The learned model's job is not to
infer novel mechanics (W1 says small nets can't); it is to *compress observed transitions*
so planning generalizes a few steps beyond the visited set, with an accuracy gate that
collapses back to exact memory when it's wrong.

**Layer 1 — Perception (all reused).** Corrected HUD mask (`scratchpad/ideas/hud_mask.py`
+ the 6-game 08-07 correction) for dev games; on **hidden games the mask must be learned**,
via the wiggle battery's per-pixel SELF/REACTIVE/DEAD + autonomous-dynamics classification
(hand-built row lists don't transfer). Animation settle (engine returns up to 61 frame
layers/action — consume all layers, unlike the duck's verified frame[-1]-only defect,
`RESEARCH-2026-08-09:41`). Connected components + role-typing
(`src/arcagi3/perception.py`, `scripts/research_2026_07_01/` role-typing assets).

**Layer 2 — Probe & dispatch (~5–15 actions/level, reused).** The offline-GO wiggle
battery + archetype classifier: AVATAR / CLICK / REACTIVE, per-level re-probe mandatory
(lf52 lesson). Public `tags` as a prior. Output: action-space abstraction (which of
ACTION1–6 matter; candidate click targets = component centroids + probe-discovered cells).

**Layer 3 — Effect model (the new build).** Three tiers, strictly gated:
- **T0 exact:** transition graph over masked board states (hash-dedup). Always on. On the
  ~12 frame-Markov games this alone supports complete optimal within-level planning.
- **T1 factored movement:** per-action object-delta model (measured accuracy 0.5–0.97,
  game-specific). Gate: ≥90% on held-out own transitions, else disabled per action.
- **T2 click mapping:** the never-built click-target→affected-cells table (deterministic +
  sparse per 07-02 probe), learned as condition→effect rules per component/color, not a CNN.
  Same accuracy gate.
Model + buffer **reset per level** (goose's one structural win); the *graph* of the current
level persists across GAME_OVER (no wipe defect). Model class is deliberately rule/table,
not CNN: every CNN variant this repo tried (5+ bolt-ons, standalone goose, reward-RL at T4
scale) lost to the graph; nothing in our evidence says a CNN earns its risk here.

**Layer 4 — Search & budget policy.** Within-level Dijkstra/BFS over masked states,
frontier = unvisited states reachable via T0 edges plus gated T1/T2 predicted successors
(model-proposed edges are verified by executing them — the env is the ground truth; a
wrong prediction just costs its action). Exploration is priced by the accounting law:
per-level budget keyed to human behavioral budgets (decode budget ≈ human p90: click 38,
avatar 94; commit stop-loss at ~1.5× human median unless already past ~10× where marginal
cost ≈ 0). **Sacrifice logic:** on L1 (weight 1/Σ) probe aggressively — decode is nearly
free there and pays at (level+1)-weighted depth. A-not-B brake (strip a (state,action)
pair after 3 null diffs — offline GO).

**Layer 5 — Hybrid dispatch (portfolio at game level).** Engineered agent runs first with
a per-game decision point: if by ~N actions (N ≈ 2× human decode p90) the probe classifies
the game as hidden-state (Markov-violation detector from the masked-frame test) **or**
search has made zero level progress by the stop-loss, hand the game to the duck for the
remaining per-game wall-clock. The engineered probe's action tax is bounded (≤~50 actions,
destroying at most the early-level share it spent them in — quantified by the accounting
law). Dispatch threshold is a rig-tuned parameter, not a guess. Expected split on dev
evidence: engineered on the ~12 frame-Markov games, duck on the ~6 hidden-state + unmodeled
rest.

## 3. Eval-fit

- **Wall-clock:** 9 h hard (organizer-confirmed), ~110 games; duck config = concurrency 28,
  7920 s/game cap, ≈12 min margin (memory `arcagi3-kaggle-submission-contract`). The
  engineered agent is orders cheaper per action than the 27B (~510 act/s measured through
  the REST gateway in the 07-02 audit, single stream — **re-verify on current stack**), so
  wall-clock is not its binding constraint; the *scored action count* is. A hybrid keeps
  the duck's concurrency/caps untouched and runs the engineered pass inside the same
  per-game window.
- **(b/a)² discipline:** never execute unplanned actions during COMMIT; all exploration is
  front-loaded into DECODE within the priced budget; deep levels are attempted only along
  planned paths. Levels the planner cannot close are abandoned early (score 0 regardless
  of spend — spending more is pure waste unless we commit past ~10×h).
- **Unmodelable games:** duck fallback (Layer 5). Cheap-probe-only floor (salience
  explorer) is strictly worse than the duck on the ledger — not proposed.
- **Live statistics honesty:** one slot detects **0.61** at 80% power (HANDOFF-08-10 §1).
  An engineered hybrid is only slot-worthy if the offline rig delta is large (≥ +0.3 on the
  true objective); anything smaller is unreadable live and must ride on offline evidence
  into the final-2 selection.

## 4. Evidence plan (stages, kill criteria, effort)

Offline scorer = the rig's `true_score.py` (already validated 224/224 rows). Baselines =
duck `BASE_ENV` banked waves (`scratchpad/banked_waves_20260809/`) — with the known caveat
that BASE_ENV ≠ shipped config (HANDOFF-08-10 §3); note verdicts against both if the §5.1
no-patch arm has been run by then.

**Stage 0 — re-verify the tooling (1 day).** Offline engine + 25 public games run; HUD
mask + wiggle battery execute on current toolchain; TransferExplorer reproduces its old
per-game levels. Kill: any pillar unrecoverable in a day → re-scope before building.

**Stage 1 — perception + probe + T0 graph planner (3–4 days).** Target: the ~12
frame-Markov dev games. **Milestones:** (a) learned-mask ≡ hand-mask on ≥20/25 games (the
hidden-set transfer test for perception); (b) on ≥8/12 frame-Markov games, levels ≥ duck's
banked per-game levels; (c) actions/completed-level ≤3× human median. **Kill:** <6/12
games at parity, or efficiency >5× human median on most — stop, write the negative.

**Stage 2 — effect model + depth + efficiency (1 week).** T1/T2 with accuracy gates,
decode/commit budgeting, sacrifice logic, GAME_OVER-persistent graph. **Milestones:** (a)
T1 held-out accuracy ≥90% on ≥8 games within ≤300 own transitions; (b) full-25 true-
objective mean ≥ duck BASE_ENV mean; (c) outright wins (engineered > duck) on ≥5 games.
**Kill:** (b) misses by >20% with (c) <3 — the paradigm loses to the duck even on its home
turf; keep only the perception layer as a duck patch candidate.

**Stage 3 — hybrid dispatch + rig A/B (3–4 days).** Dispatch threshold tuned on dev;
28-clone focus-subset rig run, engineered-hybrid vs BASE_ENV, full 25 games, true
objective, pre-registered. **Ship gate:** rig delta ≥ +0.3. Below that: no slot; archive.
Only then: one live slot as a *transfer test* (pre-registered reading rule, per standing law).

Reusable now: `hud_mask.py`, `perception.py`, role-typing + coverage/affordance assets
(`scripts/research_2026_07_01/`, old scratchpads — Stage 0 confirms survival),
`transfer_explorer.py` + offline harness, ab rig + `true_score.py`, human-replay
budget extractor (`extract_budgets.py`). StochasticGoose: ideas only (no license).
Total effort ≈ **2.5–3 weeks** of the 11 remaining — leaves selection margin.

## 5. Honest risk register

1. **W1 / hidden-set generalization — the dominant risk.** Dev milestones cannot measure
   it. The 0.33-era coverage did not transfer (dev ≠ hidden), the token-diet 2×'d dev and
   cratered hidden, and convex-hull says novel mechanics stay uninferable. Mitigations are
   real but partial: zero per-game code, learned (not hand-built) masks, behavioral
   probing, tags channel, duck fallback as the floor. The hybrid's floor is *approximately
   the duck minus the probe tax* — that bounded downside is the main reason to build this
   as a hybrid, not a replacement.
2. **Why 0.33 plateaued, and what's actually different now.** The audit verdict is
   codebase-not-paradigm, but the mechanism is now identifiable: the old explorer ran
   **unmasked** — HUD ticking meant frame hashing never merged states, so its "graph" was
   secretly a tree (16.14× collapse was discovered a month after that era ended); it
   optimized for max-over-plays replay, which eval then deleted; and it had no depth-first
   scoring model, no human budgets, no dispatch, no rig. Each of those is a measured,
   since-fixed defect. This is a genuine reason the old ceiling doesn't bind — but it is an
   argument, not a measurement, until Stage 1 (b).
3. **Hidden-state games (6/25 dev; up to ~13 under weaker criteria).** The planner is
   unsound there by construction. Dispatch must catch them; a missed catch wastes the
   game's clock on a doomed search. The Markov-violation detector exists but its
   false-negative rate on hidden games is unmeasured.
4. **Dispatch tax + our own shipping history.** Every shipped modification family scored
   *below* base on the live ledger (patched 0.788 vs base 0.965). A mis-tuned dispatch
   recreates that. Hence the hard +0.3 rig gate before any slot.
5. **cstl hypothesis is unverified intel.** 2.70 and the team profile are observed; the
   "simulator-fitting search" mechanism is inference. If their edge is something else
   (e.g. massive human-prior engineering), matching their method class may cap well below
   2.70. The design stands on our own prior art regardless.
6. **Engineering/serving risk.** The reactive `choose_action` port, gateway concurrency,
   and the silent-failure history (env toggles ≠ shipped arms) all apply; Stage 3 must
   exercise the real submission path, not an engine-level harness (08-10 method rule #1).

## 6. Recommended Stage-1 scope (the ask)

Approve **Stage 0 + Stage 1 only** (~1 week): re-verify tooling, then build the
perception + probe + T0 graph planner and measure it on the 12 frame-Markov dev games
against the duck's banked per-game levels, with the learned-mask transfer test as a
co-primary endpoint. No effect model, no hybrid, no slot. It reuses ~70% existing code,
has two crisp kill criteria, and its negative is cheap and informative (it would close the
engineered paradigm on evidence, the same way the other three levers were closed). Decision
on Stages 2–3 only after Stage 1 numbers exist.

---

## AMENDMENT 1 (2026-08-14, approved by Ahmed): Stage-1(c) kill scope

Stage 1 returned a split verdict: milestone (b) PASS at 10/13 parity with three
outright duck wins (tu93 4v2, vc33 2v1, s5i5 1v0 — the campaign's first
architecture to beat the duck on levels anywhere), milestone (c) KILL (6/7
completed games >5x human-median actions; only r11l at 1.07x). The letter of
the Stage-1 kill criterion fires on (c).

**Amendment: the Stage-1(c) kill is scoped to SHIPPING T0, not to Stage-2
entry.** Rationale, stated at amendment time and not retrofitted: (1) Stage 2's
T1/T2 effect models are the designed remedy for exactly this failure (blind
frontier sweep -> predicted edges); (2) r11l proves near-optimal planning when
the graph is small; (3) the struct precedent (capability-without-score, killed)
differs in that struct had no designed remedy. The Stage-2 kill criteria are
UNCHANGED and have the final word: T1 held-out accuracy >=90% on >=8 games
within <=300 own transitions; full-25 true-objective mean >= duck BASE_ENV
mean (miss by >20% with <3 outright wins = close the paradigm); the hybrid
dispatch floor stands (duck keeps ka59/re86/sb26-class games).
