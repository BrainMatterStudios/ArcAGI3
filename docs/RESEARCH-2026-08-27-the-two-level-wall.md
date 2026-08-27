# RESEARCH 2026-08-27 — the two-level wall, and what a 3+ score actually costs

Independent research pass over the 08-27 handoff. Everything below is measured
today against the live Kaggle API, the `arcengine` source in `.venv`, the 25
public games, or our own recorded waves. Nothing is quoted from another team's
claim without an independent reproduction.

Probe scripts (session scratchpad, reproducible):
`probe_winframe.py`, `verify_r11l.py`, `probe_levelpairs.py`.

---

## 1. The board has moved. The 08-22 field model is stale.

Leaderboard pulled 2026-08-27 11:12Z (2,566 teams):

| rank | team | score | subs |
|---|---|---|---|
| 1 | cstl | **5.99** | 37 |
| 2 | Lord Han Solo | 4.99 | 44 |
| 3 | Tufa Labs | 4.67 | 119 |
| 4 | Tong Hui Kang | 3.88 | 55 |
| 5 | rfbr | 3.37 | 13 |
| 6 | Tony G | 3.17 | 13 |
| … | | | |
| **275** | **Ahmed Mobasher** | **1.74** | **61** |

Teams ≥2.0 went 82 (08-22) → **144**. cstl went 3.57 → 5.99 in four days.
No rule, scoring or dataset change is visible on the competition pages, and no
new high-vote public notebook landed in that window — the surge is method.

**The public lane cannot reach 3.** Measured by its own authors' ranks:

| public asset | author's LB best | subs |
|---|---|---|
| `foysalemonshanto/lb-9-arc3-duck-v12-qwen3.8` (238 votes) | 2.23 | 97 |
| `keithtyser/duck-qwen3-8-27b-fp8` (91 votes) | 2.13 | 69 |
| `thtennant/arc3-duck-v22` / `v26` | **1.93** | 36 |
| ours (duck-38 v12 lane) | 1.74 | 61 |

The recipe everyone forks tops out ~1.9–2.4 as a best-of-many. Our 1.74 over
n=11 Qwen3.8 draws (mean 1.38) is *in family*, not behind it. **Recipe parity
was never the gap, and chasing it cannot produce a 3.** This retires the
"pack-recipe parity" thread as a scoring strategy.

---

## 2. The wall is at exactly two levels — replicated over 252 game-plays

Level counts from every recorded full 25-game wave in the tree (LLM lane):

| wave | 0 levels | 1 | 2 | 3+ |
|---|---|---|---|---|
| qwen38 wave1 shipped | 12 | 11 | 5 | **0** |
| banked_waves struct | 12 | 14 | 2 | **0** |
| banked_waves struct2 | 15 | 12 | 1 | **0** |
| banked_waves struct3 | 16 | 11 | 1 | **0** |
| banked_waves w2_base | 14 | 13 | 1 | **0** |
| banked_waves w2_cand | 15 | 10 | 3 | **0** |
| banked_waves pc_base | 17 | 9 | 2 | **0** |
| banked_waves pc_cand | 13 | 12 | 3 | **0** |
| banked_waves pkg | 19 | 8 | 1 | **0** |

**Nine independent waves, 252 plays, zero three-level games.** (The only
exception in the tree is the `ft09-*` single-game waves — 28 clones of one
game, where 3 is reached. So 3 is not an absolute ceiling; it is a wall on
*generic* games.)

Because score weight is the 1-indexed level number, the wall is the whole
story: level 2 is worth 2× level 1, level 3 is worth 3×.

---

## 3. It is a TOKEN wall, not an action wall

From `scratchpad/qwen38/wave1_shipped_result.json` (28 clones, 7,920 s/game,
the shipped Qwen3.8 config — its per-play mean 2.53 is the "wave-1 2.53" of the
campaign log, so this is the current-era arm):

- **28 of 28 plays ended `gave_up` at the wall clock.** Not one hit the action cap.
- median **45 actions per game**, median **60,895 generated tokens**, →
  **1,341 tokens per action**.
- After its last cleared level the agent gets a median of **14 actions** before
  the clock kills it.
- The next level costs a median of **52 baseline actions** — **104 at the 2×
  pace our agent actually plays at.**

> We are short of the budget to clear one more level by roughly **7×**.

This is consistent with the campaign's own throughput law (~60k tokens/session
whether the box is 60 or 132 minutes) — so these numbers transfer to live.

Corollary: it also explains why `effort_medium` reverted (−0.43). Cutting
thinking *per action* lowers decision quality without raising actions-per-game
enough to buy a level. The reduction has to come from **actions taken per
deliberation**, not from thinking less.

---

## 4. A verified harness bug: the agent is shown the wrong board at the moment it wins

The public `arc3-duck-v26` notebook asserts this. I did not take it on trust —
verified two independent ways.

**(a) Engine control flow** (`.venv/.../arcengine/base_game.py`):
`next_level()` sets `_next_level = True`; `is_action_complete()` returns
`not self._next_level and self._action_complete` (line 293), so the render loop
runs **one more iteration**, calls `_really_set_next_level()` → `set_level(i+1)`
(line 420-422), renders, and appends. The last layer of a level-completing
action is therefore always the **freshly loaded next level's opening board**.

`GameState.frame` returns `self.raw.frame[-1]`
(`taaf/game.py`, "the final visible frame"). `.ascii`, `.segmentation` and
every graft read it.

**(b) Empirical** (`verify_r11l.py`, r11l level-1 completion at action 17,
23 frame layers):

```
last layer == fresh level-1 opening : True    (0 cells differ)
prev layer == fresh level-1 opening : False
cells differing prev vs last        : 2,763
```

So at the exact instant the agent is scored, it is handed the next level's
board and told it is the current one. The board it actually won on — the one
frame in the game that states the objective — is discarded.

**Why it poisons induction rather than merely losing information:** across all
25 public games and all **158** consecutive level pairs (`probe_levelpairs.py`,
independently reproducing v26's pair count exactly):

- level openings that repeat: **0 / 158**
- mean overlap of the previous opening's non-zero cells: **0.68**

The substituted board is ~2/3 identical to what came before — similar enough to
read as "my action changed a few things", different enough to make every law
induced from it false. That is the mechanism that turns a win into a bad world
model right at the boundary.

**We do not have this fix, and our own code actively suppresses the signal.**
`submission/_anim_digest/digest.py:20,41,196` drops any step touching more than
`SCENE_CHANGE_CELLS = 300` cells as a scene change — with the comment
"*e.g. the next level's board appearing after a WIN check*". We identified the
frame and threw it away.

The public implementation is CC0-1.0 and downloadable:
`thtennant/taaf-kaggle-source-share-fork` → `src/taaf-grafts/taaf_grafts/`
(`winframe.py` 449 lines, `carryover.py`, `lawbook.py`, `clockwatch.py`). It
patches `_HarnessGameSession._execute_action` and
`ToolAgent._compact_action_result` — the same seams our carryover graft uses —
and is fail-open throughout.

Caveat kept honest: v26's own author sits at 1.93, so these grafts are
**not** demonstrated to move the LB. What is demonstrated is the bug.

---

## 5. What >3 costs, in exact arithmetic

Scoring applied per `scorecard.py`: per level `min(115, 100·(baseline/actions)²)`,
weight = 1-indexed level, capped by completion share. Baselines read from the
offline `EnvironmentInfo`. Counterfactuals recomputed over the same 28 plays;
"→ live" rescales by our own dev-overstates-live factor (dev per-play 2.53 vs
live series mean 1.38 = **1.8×**).

| scenario | dev LB | ratio | → live est. |
|---|---|---|---|
| observed (12/11/5 of 28 clear 0/1/2) | 2.72 | 1.00 | 1.38 |
| +1 level everywhere @ 2× baseline actions | 4.56 | 1.67 | 2.31 |
| +1 level everywhere @ 1.4× baseline | 6.22 | 2.29 | 3.15 |
| +1 level everywhere @ baseline pace | 8.94 | 3.29 | 4.53 |
| +2 levels everywhere @ 2× | 6.99 | 2.57 | 3.55 |
| +2 levels everywhere @ 1.4× | 11.28 | 4.14 | 5.72 |

### 5b. Efficiency is NOT a lever — it is already saturated

Realized pace on all 21 levels the wave actually cleared (actions ÷ human
baseline):

- **median 0.86×, mean 1.13×** — we clear levels *faster than the human baseline*
- 12/21 at or better than baseline; 16/21 within 1.4×; only 1 worse than 2× (tu93 L1, 3.63×)
- 9 of 21 cleared levels already hit the 115-point per-level cap

So the "@1.4×" row above is pessimistic. Re-running the ladder by sampling from
our **own measured efficiency distribution** gives the honest counterfactual —
what happens if the agent simply keeps playing at the pace it already plays:

| scenario | dev LB | → live est. | actions/game needed |
|---|---|---|---|
| observed today | 2.72 | 1.38 | 45 |
| **+1 level everywhere, our own pace** | **7.78** | **3.95** | **~115** |
| **+2 levels everywhere, our own pace** | **15.88** | **8.05** | **~178** |
| +3 levels everywhere, our own pace | 27.06 | 13.72 | ~282 |

**The entire gap is depth, and depth reduces to one number: actions per game.**
At a fixed ~60,900-token session:

| actions/game | tokens/action | vs today |
|---|---|---|
| 45 (today) | 1,353 | — |
| 115 (+1 level) | 530 | 2.6× cheaper |
| 178 (+2 levels) | 342 | 4.0× cheaper |

Model quality, sampling, serving, recipe parity and efficiency are all off the
critical path. The one thing that is on it is **how many actions the agent gets
to take before the clock ends the game.**

Caveat: the cleared levels are selection-biased — the agent clears the ones it
can crack fast. Cutting the other way, deeper levels carry larger baselines
(tu93 `[19,16,34,42,123,80,…]`), so they give proportionally *more* room, and
the 1.8× dev→live discount is already applied to every figure above.

Extra actions are only worth this much if they are *aimed*: the same extra
level is worth 2.31 played sloppily and 4.53 played clean, because the term is
squared. The pack recipe's failure mode on our rig — "sk48 390 actions never
clearing L1, blind identical 26–28-action batches" — is what unaimed actions
buy.

---

## 6. Where this agrees and disagrees with the 08-27 handoff

**Agrees:** the CV≈0.20 law (only ≥0.55 levers are slot-testable — the +1-level
lever qualifies, individual grafts do not); banking is the entire value of the
search lane; safe redraws are dead.

**Disagrees on priority.** The handoff's #2 is "grow the crack inventory". That
lane's value is `p × 85.7` per added crackable class with a zero floor — real,
but it is a lottery over hidden-set composition. The two-level wall is
**systematic**: it applies to every game we already partially solve, it is
replicated over 252 plays, and closing it once is worth ×1.7–2.3 on the entire
board. Both are worth running; the priority order should invert.

**Adds a retirement:** the recipe-parity thread (§1). The public lane's own
authors cap at 1.9–2.4.

---

## 7. Recommended programme (offline-falsifiable first, zero slots)

1. **Instrument the boundary.** Take the 16 recorded plays that cleared ≥1
   level and answer, from the transcripts: after the win, does the agent (a)
   restate a goal derived from the substituted board, (b) re-explore from
   scratch, or (c) run out of clock mid-thought? This decides whether
   winframe/carryover is a real lever for *us* or only for thtennant's archive.
   Falsifier: if <30% of post-win turns show boundary confusion, the perception
   fix is not our binding constraint and only §7.2 survives.
2. **Buy actions per game — this is the whole programme.** 45 → ~115 for a 3+
   score, → ~178 for a 5+ score, i.e. 530 and 342 tokens/action against today's
   1,353. The only lever class that gets there without cutting thinking
   quality is **more actions per deliberation** — our own struct plan-channel
   already measured 2.01 actions/LLM-turn (34% multi-step plans, first-ever
   target unlock on g50t) and was shelved uncertified when Qwen3.8 landed.
   Re-price it against the *level-depth* objective, not "levels/box".
   Hard constraint from the pack-recipe evidence: batches must be gated on a
   validated rule, or they burn the squared efficiency term.
3. **Graft the free fixes as one bundle** (winframe + carryover + lawbook +
   clockwatch, CC0, audited line-by-line as we did for v22). Per the CV law
   these are not individually slot-testable — they fly together or not at all,
   and only after §7.1 says the mechanism binds for us.
4. **Leave tonight's v8 arm alone.** It is a pre-registered zero-floor lottery
   with an independent thesis; it does not compete with this work.

## 8. STEP-1 VERDICT — winframe does NOT bind for us: **KILL**

Instrument: `submission/_boundary_probe/probe_boundary.py`, run over 120
recorded transcripts. **52 level-up events across 27 transcripts.**

First, the mechanism confirmed a third time, this time from *live-served
transcripts* rather than the engine: at 39 of 52 level-ups the result dict
carries `animation_frame_count >= 1` with `animation_changed_cell_count` of
**240–3,606 (median 1,316)**. That hidden "animation" frame is the board the
level was won on. The agent receives the count; it never sees the board.

But the pre-registered question was whether it *binds*. Reading the agent's own
words in the turn immediately after each win:

| behaviour | rate |
|---|---|
| recalls or asks for the winning board | **0 / 52** |
| says it no longer has / cannot see it | **0 / 52** |
| re-explores from scratch | **0 / 52** |
| attributes the new board to its own action | **0 / 52** |
| correctly notes a new level started | 41 / 52 (79%) |
| mentions the previous level | 35 / 52 (67%) |
| carries a hypothesis forward | 12 / 52 (23%) |

**Boundary confusion: 0/52 = 0.0%, against a 30% bar. KILL.**

Hand-audited, not just counted. The representative post-win turn opens:
*"Level 1 completed! Now I'm on level 2. The new level looks different… The
hypothesis from level 1 was that the markers encode the target pattern."* The
19% of turns matching "confused" phrasing are all grid-**counting** confusion
inside the new level ("Wait, that's 5 rows, not 4"), not boundary confusion.

**Why the v26 thesis does not transfer to us:** the result dict already tells
the agent `level_completed: true` and `level: N`. It therefore cannot be
confused about *whether* a boundary happened — only about the *content* of the
board it won on, and in 52 opportunities it never once reached for that
content. The false-law failure v26 documents on m0r0 is real for their archive;
it is not our binding constraint.

⇒ **The winframe/carryover bundle is deprioritised.** §7.3 is withdrawn as a
near-term build. The bug is real and worth fixing eventually; it is not worth a
slot now.

Sample caveat: the 52 events come from ft09 (36) plus one 28-clone Qwen3.8 wave
(16), so the sample is clone-heavy rather than game-diverse. The finding that
carries regardless is architectural, not game-specific: the boundary is
*announced* in the result dict on every game.

## 9. STEP-2 RESULT — 86% of the model's output produces no action at all

Instrument: `submission/_boundary_probe/probe_budget.py`, same 120 transcripts,
**12,694 LLM turns, 10,745 actions**.

> **CORRECTION.** The first run of this instrument reported 73% zero-action
> turns holding 86% of output. That was an artifact: it counted `executed_count`
> inside `[TOOL RESULT]` blocks, and those blocks go missing when the harness
> trims history. Against the wave result JSONs that estimator agreed with ground
> truth in only **1/8** struct transcripts. The turn header carries the
> harness's own cumulative action counter (`--- analysis_step=N | action=M |`),
> so actions-per-turn is the delta to the next header — schema-independent, and
> it matches ground truth **16/16**. All figures below use the header delta.

| | share of turns | share of model output |
|---|---|---|
| **zero-action turns** | **38.0%** | **42.8%** |
| single-action turns | 31.1% | — |
| batched turns (≥2 actions) | 27.9% | — |

- Acting turns average **1.37 actions each**; overall 0.85 actions/turn.
- Zero-action turns are not empty: median **3,516 chars** against 3,152 on an acting turn.

**The headroom is real but bounded.** Recovering *all* of the zero-action
budget is a **×1.75** ceiling on actions. The live bars need **×2.56** (3+) and
**×3.96** (5+) on 45 actions/game. So removing zero-action turns is **necessary
but not sufficient** — the rest must come from actions per acting turn, or from
a non-LLM executor.

### 9b. THE DECISIVE TEST — actions are NOT levels

Patch 21 (`TAAF_STRUCT=1`, the mandatory plan-list channel) already exists in
`submission/_duck_patched/duck_patches.py`, and both arms of a 28-vs-28 A/B are
on disk. Re-run on the corrected instrument:

| | base (patch 21 OFF) | struct (patch 21 ON) | |
|---|---|---|---|
| zero-action share of turns | 34.3% | **18.4%** | −15.9 pts |
| zero-action share of tokens | 36.3% | **18.4%** | −17.8 pts |
| actions per acting turn | 3.06 | **4.03** | ×1.32 |
| actions per turn | 2.01 | **3.29** | ×1.64 |
| median actions/game | 86 | **120** | ×1.40 |
| **total levels cleared (28 plays)** | **36** | **27** | **−25%** |

**The mechanism works perfectly and the score goes down.** Every process metric
moves the way the thesis wants; levels move the opposite way.

Cross-checked on the 25-game fixture (`scratchpad/banked_waves_20260809`, no
transcripts but level counts are in the JSONs): STRUCT waves cleared 18 / 14 /
13 levels (mean 15.0) against base 13 / 15 (mean 14.0) — a +1 difference on
~15, with both replications at or below base. That reproduces the campaign's own
08-09 verdict: *"adoption replicates, level spike does not."*

⇒ **§5's counterfactual assumed extra actions convert into extra levels at our
observed efficiency. The one controlled test of that assumption refutes it.**
The ladder in §5 remains correct arithmetic about what clearing more levels
would be worth; it is not a prediction that buying actions will clear them.

The binding question is not *how many* actions but whether they are **aimed**.
Two prior results say the same thing and were not connected until now: sk48
burned 390 actions in blind 26-action batches without clearing level 1, and
`effort_medium` cut deliberation and lost 0.43. Deliberation that ends in an
action is only valuable if the deliberation was what made the action right.

⇒ **No action-buying lever flies until it demonstrates levels held or raised.**
`probe_budget.py` must never be read without level counts beside it; the tool
now prints that warning itself.

## 9d. FORENSICS — the constraint is comprehension, not budget

Three follow-up measurements on why patch 21's extra actions did not land.
Instrument: `submission/_boundary_probe/probe_plans.py`.

**(1) Where the extra actions went.** ft09, 28 clones per arm:

| | base | struct | |
|---|---|---|---|
| actions total | 2,463 | 3,565 | +1,102 |
| burned on the level it never cleared | 1,537 | 2,504 | +967 |
| actions per level actually cleared | 68.4 | 132.0 | ×1.93 |

**88% of every extra action the channel bought was spent on a level that was
never cleared.** It did not buy progress; it bought deeper failure.

**(2) The committed plans were fine.** The obvious mechanism — a plan of k
actions chosen from one observation goes wrong after step 1 — is **refuted**.
Controlling for plan length (the raw position curve is confounded, since a plan
only reaches step 5 if it is ≥5 long):

```
 len 2 (n= 38):   87%  89%
 len 3 (n= 27):   93%  93%  93%
 len 4 (n= 29):   93%  93%  93%  97%
 len 5 (n= 12):  100% 100% 100% 100% 100%
 len 6 (n=  8):  100% 100% 100% 100% 100% 100%
```

Effectiveness is flat within a plan, and *longer* plans are effective
throughout — the model commits to long plans when it is confident and it is
right to. Immediate repeats are 0.6% of steps; direct reversals 0.0%. Only 3.1%
of steps follow a plan's first no-op.
Caveat: transcripts record 488 of the arm's 3,565 actions and the recorded
sample skews short (plans of 8+ are 12% of real acting turns, 2% of the
sample), so the long-plan tail is under-tested.

**(3) The failing games are not starved.** On the live-representative 25-game
Qwen3.8 wave, the 12 games that cleared nothing spent a median **0.93× the
level-1 human baseline** in actions — 6 of 12 spent *more* than a human needs
for level 1, none spent less than 0.53×, and their token spend (60.3k) matches
the scoring games' (63.1k). On ft09 the zero-level clones burned 82 (base) and
144 (struct) actions against a 43-action level-1 baseline — 1.9× and 3.3× the
human budget — and every one ended `gave_up` having cleared nothing.

Put together with §5b (levels we *do* clear come in at 0.86× human pace):

> **On a game the agent understands, it plays better than a human and is not
> budget-limited. On a game it does not understand, no amount of budget helps —
> giving it 3.3× a human's actions produced exactly zero levels.**

The binding constraint is **comprehension**, not budget. That closes the
depth-via-actions lane: actions/game is a diagnostic, never an objective.

### Correction to §6 — I inverted the priority on a thesis that is now refuted

This morning's pass argued the handoff's priority should invert: that the
systematic two-level wall should outrank "grow the crack inventory". That
argument rested on extra levels being *purchasable* with budget. §9b and §9d
refute that premise directly.

**The handoff's original ordering was right.** The search/banking lane is the
one route to a high score that does not depend on the agent comprehending
anything: a cracked and banked game pays a flat **+1.82** whether or not the
LLM ever understood it, and today's measurements do not touch it. This also
fits the 08-22 read that cstl's edge is *additive and model-agnostic* — the
leaders are most likely not comprehending more games than us, they are banking
more of them.

Depth is not dead as an *objective* — §5's arithmetic still says one more level
everywhere is worth ×2–3 on the board. What is dead is buying it with actions.

## 9c. TONIGHT'S SLOT (2026-08-28) — no new lever qualifies

Asked whether any new lever should ride tonight's submission. Answer: **no**,
and the reason is measured rather than procedural.

- **Patch 21 / mandatory action channel** — the only action-buying lever that is
  already built, tested and shippable. It is the one thing we could physically
  arm in time, and §9b is a direct 28-vs-28 experiment showing it clears 25%
  fewer levels. Flying it would be flying a measured negative.
- **winframe / carryover bundle** — killed in §8 (0/52 boundary confusion).
- **Anything new** — cannot be built, audited, GPU-smoked, attested and re-armed
  in the remaining window, and the `effort_medium` lesson stands: a perfect
  50-minute smoke certified an arm that then cost −0.43 live.

The v8 crack-or-nothing arm flies as armed. Its outcome distribution is lumpy
because a banked crack is worth a flat +1.82 per hidden game: 0 cracks ≈ 1.4
(base band, floor measured at 0.00), 1 ≈ 3.2, 2 ≈ 5.0, 3 ≈ 6.8. It is the only
thing on the pad with a 5-tail, and it is already smoke-verified 5/5.

## 11. THE CRACK LANE — what a live-affordable crack actually costs

Following §9d's conclusion that the banking lane is the surviving route, this
section prices it. All of it runs offline on the engine; no GPU, no slots.

### 11a. The live cost model is forced (checked, closed)

Offline, `Arcade(OFFLINE)` returns a `LocalEnvironmentWrapper` — an in-process
object. `copy.deepcopy(env)` costs 0.8 ms, the copy runs at ~57,000 steps/s,
and stepping it does **not** touch the real scorecard (verified with a working
positive control: 3 real steps move the card, 500 copy steps do not; the copy's
`scorecard_manager` is a distinct object). So offline search is effectively free.

That does **not** transfer. `Arcade.make` branches on operation mode
(`arc_agi/base.py`): OFFLINE → `_find_local_game`, COMPETITION →
`_create_remote_wrapper`, and a COMPETITION Arcade holds a
`requests.Session()` against `arc_base_url`. **Live, the game state lives
server-side**, so there is nothing to copy and every step is billed. The
campaign's reset-replay cost model is correct and the "search on a copy" idea
is closed.

### 11b. Replay overhead is 11–17×, and it is inherent

Measured on tu93, reset-replay, per level:

| level | nodes | actions | resets | actions/node |
|---|---|---|---|---|
| 1 | 246 | 2,807 | 247 | 11.4 |
| 5 | 395 | 5,759 | 396 | 14.6 |
| 6 (unsolved) | 2,610 | 44,834 | 2,612 | 17.2 |

**One reset per node.** ~94% of every action is replay, not new information.
Two attempts to shave it both came back null:

1. `ResetReplayBackend.children()` ignored its `consume` parameter and
   re-walked the prefix for every token. Added a path cache so a walk already
   standing on the requested state costs nothing. Result: **identical search
   results, 9 actions saved out of 58,522 (0.015%)** — the frontier order means
   the env is essentially never where the next request wants it.
2. A per-level depth bound (see 11c). **0 nodes pruned.**

The overhead is structural: star-expansion on a single env with no undo costs
`k × (1 + d)` per parent, and there is no local trick around it. Getting past it
needs a different search shape (deep chains rather than stars), or not searching
at all.

### 11c. NEW: 10 of 25 games enforce a per-level move budget

`submission/_search_core/step_budgets.py`. The games carry a `StepCounter` in
their level data and the engine renders it as the edge bar the system prompt
warns the model not to click on. It is enforced — `wa30.py:1252`,
`tu93.py:1270`: `if not current_steps: self.lose()`, checked *after* the win
test, so a solve landing on the last move still counts.

Games with budgets: ar25, dc22, ka59, lp85, ls20, re86, s5i5, tu93, vc33, wa30
(plus cn04 `MaxSteps`, sp80/su15 `steps`).

**The important consequence is about baselines, not search.** On **30 levels**
the published human baseline EXCEEDS that level's own move budget (wa30 L9:
baseline 415, budget 70; tu93 L5: 123 vs 50). A single attempt cannot exceed the
budget, so **the published baselines count multiple attempts.** Two things follow:

- A clean single-attempt solve on a budget level always saturates the
  efficiency cap, since `baseline/actions ≥ baseline/B > 1`. Independent
  confirmation of §5b.
- The wa30 specialist's open note is resolved: it reported "L3's optimal plan
  (169 acts) exceeds the level's step budget (~100)". A within-budget solution
  must exist, so 169 is **our planner being ~1.7× worse than the level allows**,
  not the level being unsolvable. That is a concrete solver target.

The bound was also wired into the search as `SearchCore(max_depth=...)`
(firewalled: `None` = byte-identical, verified deterministic). It is
**measured null** — 0 nodes pruned on tu93 and ar25 — because the frontier is
breadth-first and times out at depth 10–29 against budgets of 20–64, never
reaching the bound. Kept, off by default, for the knowledge rather than the
speed.

### 11d. Census: the generic search cannot produce a live-affordable crack

25 games, reset-replay (the live cost model), 900 s/game, portfolio algorithm.
**48 levels, 2 cracks, 1 of them affordable:**

| game | levels | crack | actions | lane |
|---|---|---|---|---|
| tu93 | 9/9 | **yes** | 84,685 | nbfs_macros |
| ft09 | 6/6 | **yes** | **1,231** | specialist |
| vc33 | 4 | no | 1,441,051 | nbfs |
| dc22 | 3 | no | 1,682,345 | nbfs_macros |
| …21 more | 0–2 | no | 0.4M–3.5M | — |

- every non-crack game: **0.4M–3.5M actions** for 0–3 levels.
- **only ft09's whole search fits the live budget.**

A full 1,500 s live engagement allows roughly **195,000 live actions**
(~73,000 offline-equivalent at the measured ×2.66 guard tax). The generic
search is **2–18× over that** without cracking anything, and v8's flight config
caps a specialist at 4,000 actions.

> **A live-affordable crack has to be COMPUTED, not searched.** ft09_gf2 costs
> 1,231 actions because it reads the board, solves a linear system, and plays
> the answer. tu93 is cracked only by generic search at 84,685 actions and is
> therefore not flightworthy at any cap we can afford.

So "grow the crack inventory" means **write more solvers**, not tune the search.
The existing inventory and its blockers:

| specialist | reaches | live cost so far | blocker |
|---|---|---|---|
| `ft09_gf2` | **6/6 — cracks** | 1,231 | none |
| `wa30_grabdrag` | 2 of 9 | L1 **380**, L2 6,771, L3 burns 97,859 | plan length: 169 moves for a 100-move level |
| `sc25_glyph` | 2 of 6 | L1 15,703, L2 46, L3 burns 278,741 | L3 cast is wrong — the maze BFS exhausts a fully-explored 1,795-state space |
| `tn36_program` | 2 of 7 | 909,403 over the run | enumeration does not reach L3 configs |

### 11e. The design law the numbers point at

Three solvers, three cost profiles, one pattern:

- **ft09_gf2** — perceive → solve a linear system → execute. **1,231 actions, cracks.**
- **wa30_grabdrag** — perceive → A* on a *model* → execute. **380 actions for a depth-26 level.** Affordable where the plan is good.
- **sc25_glyph** — perceive → cast → **BFS on the real env**. 15,703 actions for level 1 alone, and 278,741 burned on level 3.

> **A specialist that plans in a model is affordable. A specialist that searches
> the environment is not.** Env-search inside a solver reintroduces exactly the
> 11–17× replay tax of §11b. Every new solver must perceive → model → plan →
> execute, and verify at most the final plan.

### 11f. Deployability needs a DETECTOR, not just a crack

tu93 is cracked (84,685 actions) and still not flightworthy, for two
independent reasons:

1. **16% over budget.** ~73,300 offline-equivalent actions fit a 1,500 s
   engagement; tu93 needs 1,735 s. Three attempts to close that gap all came
   back null: the depth bound (0 pruned), the path cache (9 actions of 58,522),
   and skipping the portfolio racer's audition (**4 actions** — `nbfs_macros`
   direct costs 84,681 vs portfolio's 84,685).
2. **No detector.** tu93 is cracked by the *generic* lane, so there is nothing
   to pre-screen on. Without a zero-action frame-0 screen the arm has to engage
   blind, which is the v7 design that measured net-negative (law 6: partial
   grinding shares its play and charges its actions to whatever the LLM later
   completes).

⇒ **The deployable crack inventory is 1 (ft09), not 2.** Growing it means
shipping `(cheap frame-0 detector + model-based solver)` pairs. Each one adds
~`p × 85.7` with a floor of zero, where `p` is that class's frequency in the
hidden half.

The nearest target is **wa30**: its detector already exists and fires, its
solver is already model-based and affordable where it works (380 actions for
level 1), and §11c proved a within-budget solution to level 3 must exist. The
gap is a planning-quality problem — 169 moves against a 100-move budget —
not a perception or cost problem.

## 10. Open questions this pass did not close

- **The 1.8× dev→live discount** is now measured twice but still unexplained.
  If it is set composition, every offline counterfactual above is optimistic by
  that factor; if it is a serving or geometry artifact it is itself a lever
  bigger than anything in the queue.
- **What cstl actually does.** 5.99 ≈ two-levels-cleared on most games. That is
  the profile §5 prices at "+1 to +2 levels everywhere", which is consistent
  with the 08-22 read that their edge is additive and model-agnostic.
- My `probe_winframe.py` sweep under-sampled: `GameAction(int)` raises for
  several games' action ids, so 19 of 25 games took zero random actions. The
  one completion it found was enough to verify the claim, but the sweep is not
  a coverage measurement.
