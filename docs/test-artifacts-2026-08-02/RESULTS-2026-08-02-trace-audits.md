# Retrospective trace audits — 2026-08-02

Per docs/RESEARCH-2026-08-02-unbiased-deep-research.md (Days 1-2 items E3/E4, open
question 3). No new runs; everything parsed from stored artifacts.

## Data located

- **Stored episodes (the "62")**: `/Users/ahmed/Documents/ArcAGI3/scratchpad/rl_gate/episodes/` —
  62 episode dirs with `artifacts/viewer_data_events.jsonl` (per-action 64x64 board,
  action_display, level, score, state, reward, game_over). 8,079 events; 5,743 scored
  actions; 2,664 clicks; 28 distinct games (25 ARC-3 public + sk01/ul01/sq01 from
  arc-interactive). Mixture of Claude-teacher (K3) sweeps, 27B control, smoke/replay runs.
- **HUD masks**: `/Users/ahmed/Documents/ArcAGI3/scratchpad/ideas/hud_mask.py` (18-game
  regional HUD map, validated 2026-08-01).
- **Rig dumps (shipped base config)**: prior-session scratchpad
  `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/066c8134-.../scratchpad/`:
  `g0/rig_result.json` (base, 7 usable repeats x 28 games, 3600s/game box),
  `bsm/rig_result.json` (base, 1 repeat, 1800s, with actions_per_level),
  `bfm`, `dpm` (boardfix/depth arms), `A.json`/`B2.json` (greedy temp-0 sweeps — excluded:
  not the shipped sampling config).
- **Submission ledger**: fetched live from Kaggle submissions API (44 submissions,
  descriptions + public scores). The 8 identical-bytes base draws verified:
  0.92, 1.14, 0.82, 0.75, 0.96, 0.88, 1.27, 0.69 (mean 0.9287, sd 0.1947).
- **Per-level baselines**: `environment_files/*/*/metadata.json` (same source as
  `scratchpad/ideas/score_rig.py`).

---

## AUDIT 1 — Dead-signature (K=2) blocking simulation

**Method.** For every stored click, the clicked cell's 4-connected component was
fingerprinted Reki-style as (color, size, normalized shape, twin-count). "No effect" =
HUD-masked board equality AND no score/level/state/reward change. Policy: after 2
observed no-effect uses of a signature class within a level, block the class for that
level. A "false block" = a click issued while its class was blocked that in fact
produced a real (masked) effect.

**Result (5,743 scored actions, 2,664 clicks):**

| metric | value | bar |
|---|---|---|
| blocked clicks | 734 = **12.78% of scored actions** (27.6% of clicks) | >= 8% PASS |
| false blocks | 135 = **18.39% of blocks** | < 1% **FAIL** |

Per-game (blocked / false-block rate): ft09 388/0.8%, dc22 112/0.9%, vc33 103/**76.7%**,
lf52 84/**34.5%**, s5i5 23/**78.3%**, sb26 21/23.8%, sp80 2/0%, ka59 1/0%.

**Verdict: KILL (as pre-registered).** Volume is there, but the false-block rate is 18x
over the bar. The failure is exactly the anticipated one — state-gated buttons: on
vc33/s5i5/lf52 the same signature class legitimately no-ops until a precondition is met,
then works (e.g. vc33 L3: repeated clicks on row 56 no-op twice, later the same class
fires 79 times with real effect). A global hard veto is unsafe. If revisited at all, it
would have to be per-game (ft09/dc22-shaped games are clean at <1%) — but that is a new,
un-pre-registered idea, not this one.

## AUDIT 2 — Death-signature repeats (hazard ledger gate)

**Method.** Every action event entering GAME_OVER hashed as (game, action-type,
HUD-masked pre-death frame) [full-frame] and a 9x9 local-crop variant around clicks.
Repeats = deaths minus distinct signatures, per game.

**Result:** 36 deaths total across 13 of 28 games represented.
Full-frame signature: 2 repeats total → **0.07/game** (0.15/game over death-games only).
Local-context signature: 4 repeats total → **0.14/game** (0.31/game). The only clusters:
ft09 (5 deaths, 3 local repeats), bp35 (9 deaths, 1 repeat), tn36 (1 local repeat).

**Verdict: KILL (as pre-registered — bar was >= 1 repeat/game).** Identical-death repeats
are rare in stored traces; a hazard ledger has almost nothing to prevent. Caveat: the
depth pack's death-safe memory (already shipped) may itself be why repeats stay low in
newer traces, and 36 deaths is a thin base — but even the generous denominator
(death-games only, local hash) is 3x under the bar.

## AUDIT 3 — Per-game score decomposition (open question 3)

**Method.** g0 dump = 7 repeats x 28 clones of the pinned base config (3600s/game box —
about half the eval wall-clock; treat depth as a mild lower bound). Per game-run, the
completed-share cap `sum(1..k)/sum(1..n)*100` gives the score at perfect efficiency;
bsm's actions_per_level + local baselines give actual efficiency via the exact scorer
formula (mirrors `scratchpad/ideas/score_rig.py`).

**Result (196 game-runs):**

- levels/game mean **0.347** (per-repeat 0.29-0.54) — matches the 0.36 figure in the
  Aug-2 submission description and lands the cap-score at **1.242 ≈ our 1.27 LB draw**.
  The model reproduces the leaderboard number with no free parameters.
- **70.4% of game-runs complete zero levels.** Distribution: 0 levels 138/196, 1 level
  49, 2 levels 8, 3 levels 1.
- 9 of 25 public games never produced a level in ANY of 7 repeats:
  cn04, dc22, g50t, lf52, ls20, m0r0, sk48, tr87, wa30.
- Efficiency haircut on completed levels: **zero.** All 5 completed levels in the bsm
  dump score at the cap (actions 0.5-0.97x human baseline; min() binds at completed-share).
  Same as FINDING-2026-08-02: 8/10 capped, and the 2 efficiency-bound levels were in the
  boardfix arm, not base.

**Decomposition of the gap to 1.8 (per-game % scale = LB scale):**

| component | value | share of gap |
|---|---|---|
| current expected score | 1.24 | — |
| efficiency-limited gap (perfect efficiency, same depth) | **0.00** | **0%** |
| unlock-limited gap (games/levels never reached) | **0.56** | **100%** |

**Verdict: the gap to 1.8 is entirely unlock-limited.** Efficiency polish (ideas 5, 6,
token diet, geodesic) has literally zero headroom at current depth. Getting one more
level on the 16 games that already unlock L1 (L1→L2 is a ~3x cap multiplier), or a first
level on any of the 9 never-unlocked games, is the only scoring currency. This confirms
and quantifies FINDING-2026-08-02-efficiency-headroom-is-exhausted with 7 repeats
instead of 2.

Caveats: public games, not the hidden 55/110; 3600s box vs 7920s eval; depth-pack arm not
included (its rig run showed no depth gain inside noise).

## AUDIT 4 — E[max] bootstrap (E4)

**Draws verified against the Kaggle ledger** (identical-bytes pinned duck-base v2):
0.92, 1.14, 0.82, 0.75, 0.96, 0.88, 1.27, 0.69 → mean 0.929, sd 0.195. (The 44-sub
ledger contains no other identical-config series of n>=3.)

E[max of N], 200k bootstrap:

| N | empirical bootstrap | Gaussian(0.929, 0.195) | P[max>=1.8] (Gauss) |
|---|---|---|---|
| 10 | 1.225 | 1.228 | ~0.0000 |
| 30 | 1.268 | 1.327 | 0.0001 |
| 50 | 1.270 | 1.367 | 0.0002 |
| 70 | 1.270 | 1.392 | 0.0003 |
| 90 | 1.270 | 1.409 | 0.0002 |
| **92 (slots left to Nov 2)** | **1.270** | **1.412 ± 0.084** | **0.0004** |

The empirical bootstrap is hard-capped at the observed max (1.27) — it says only that
more draws re-hit our current best. The Gaussian model is the honest extrapolation:
**E[max of all 92 remaining slots] ≈ 1.41, shortfall 0.39 vs 1.8; P[any draw >= 1.8] ≈
0.04%; P[>= 1.5] ≈ 14%.**

**Verdict: pure farming of the current config cannot reach 1.8** (nor, with ~86%
probability, 1.5). Farming is worth ~+0.14 over the current 1.27 if every remaining slot
were spent on it. Per open question 7 of the research doc: the plan MUST land a
mean-raising lever; slot allocation between farming and experiments is now arithmetic.

---

## Cross-audit synthesis

Ideas 5 (no-op immunization) and 9 (hazard ledger) are both dead on their pre-registered
kills, and AUDIT 3 explains why they were never going to matter: both are efficiency/
waste-reduction levers, and efficiency carries 0% of the remaining gap. AUDIT 4 shows
variance farming alone tops out ~1.41. Everything therefore concentrates on the
unlock levers of the research doc: #1 post-WIN replay (multiplies won games), #2 HUD
mask + graph substrate insofar as it unlocks new levels, #4 depth-preserving memory, and
#7 SFT — judged by levels completed, never actions saved.

## Data-quality caveats

1. Episodes are heterogeneous: mostly Claude-teacher (K3) traces, not 27B-duck traces;
   click/no-op behavior of the shipped 27B differs (K3 is more deliberate, so blocked%
   for the duck is likely HIGHER, and false-block% on state-gated games no better).
2. sk01/ul01/sq01 (17 episodes) have no HUD map; they contributed no blocks and 0 deaths,
   so they dilute denominators slightly but do not drive any verdict.
3. 36 deaths is a thin base for AUDIT 2; verdict is robust only because it misses the
   bar 7-14x, not marginally.
4. AUDIT 3 rests on public-game clones at half the eval budget; the hidden set could
   skew shallower or deeper. The 1.24-vs-1.27 LB agreement is reassuring but partly
   coincidental (one draw).
5. AUDIT 4's Gaussian tail is a model; with only n=8 draws the sd itself has ~±35%
   relative error (chi-square 95% CI ~0.13-0.40), so E[max of 92] is realistically
   1.33-1.65 — still short of 1.8 across the entire CI.

## Files

- `audit12_episodes.py`, `audit12_raw.json` — audits 1-2 (this directory)
- `audit34_score_emax.py` — audits 3-4 (this directory)
