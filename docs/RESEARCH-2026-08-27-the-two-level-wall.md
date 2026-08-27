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
**12,814 LLM turns, 7,877 actions**.

| | share of turns | share of model output |
|---|---|---|
| **zero-action turns** | **73.1%** | **85.9%** |
| single-action turns | 18.7% | — |
| batched turns (≥2 actions) | 8.2% | — |

- When the agent *does* act it already averages **2.28 actions per turn** (median batch 4, max 36) — batching is partly working already.
- Zero-action turns are not empty: median **3,516 chars** of model output each, against 1,805 on an acting turn.
- 34% of them are "long" (>5,000 chars) and eat **66% of the entire zero-action budget** — 3,149 deliberations that thought hard and touched nothing.
- Only 594 turns (6.3%) are genuine dead decodes, and they cost ~0% of output. The dead-decode class is a rounding error; the *long silent deliberation* is the whale.
- Overall cost: **7,237 chars of model output per environment action.**

**This replaces "batching" as the framing of the lever.** The target is not
"more actions per deliberation" — it is *deliberations that end in an action*.
Holding acting-turn productivity fixed, actions scale as `1 − z` where `z` is
the zero-action share of the token budget:

| zero-action share of budget | actions/game | |
|---|---|---|
| **0.86 (today)** | **45** | |
| 0.70 | 96 | |
| 0.65 | 112 | |
| **0.60** | **128** | **3+ cleared** |
| 0.50 | 160 | |
| **0.44** | **179** | **5+ cleared** |

**86% → 60% buys a 3+ score. 86% → 44% buys a 5+ score.**

The campaign already wrote down the correct intervention a month ago and never
built it (`RESULTS-2026-08-08`): *"advertised = performative; ENFORCED =
adopted. Next iteration: make batching STRUCTURAL (mandatory plan-list action
channel), not optional."* The structural form is a **mandatory action channel**
— every turn either executes at least one action or spends from a small,
explicit, capped inspection budget.

Two honest constraints on this lever:

1. **Do not simply make thinking cheaper.** `effort_medium` did exactly that and
   cost −0.43; the campaign's own reading was that xhigh "dead decode" was
   partly load-bearing deliberation. The intervention must redirect
   deliberation into actions, not shorten it.
2. **The freed budget must stay aimed.** The projection assumes recovered
   turns act at today's productivity. Unaimed actions are worthless — sk48,
   390 actions in blind 26-action batches, level 1 never cleared.

Modelling note: the ratios (86%, 2.28 actions/acting turn) come from 12,814
turns of smoke and ablation transcripts; the 45 actions/game baseline comes
from the shipped-config wave. Combining them assumes the ratio transfers to the
live geometry — which the throughput law (~60k tokens/session regardless of box
length) supports but does not prove.

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
