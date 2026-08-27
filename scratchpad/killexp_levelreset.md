# KILL EXPERIMENT — level-checkpointed Go-Explore

Date: 2026-07-22 · offline (`environment_files`) · `.venv/bin/python` · `PYTHONPATH=src`
Scripts: `scratchpad/killexp_probe.py` (task 1), `scratchpad/killexp_cost.py` (tasks 2–3, an
instrumented fork of `scratchpad/prefix_replay_cost.py`), `scratchpad/killexp_blocksweep.py` (task 4).
Raw logs: `scratchpad/killexp_run.log`, `scratchpad/killexp_sweep.log`.

---

## VERDICT: **DOWNGRADE**

**The deciding number: 8.7 %.**
That is the median share of the env-step budget spent *re-reaching the current level's start state*
when the level savestate is used (`levelreset` mode, 7 non-degenerate dev games). The decision rule
said <5 % ⇒ collapse, >30 % ⇒ rebuild. We landed at 8.7 % — below the rebuild threshold and only
marginally above the collapse threshold.

Three facts make it a downgrade rather than a build:

1. **The savestate is real and uniform** — mid-level `reset()` preserves `levels_completed` on all
   8 games tested, with and without `ONLY_RESET_LEVELS`. No game silently wipes. The premise holds.
2. **We are already exploiting it in production.** Every submission notebook sets
   `ONLY_RESET_LEVELS=true`, and `submission/_adopt/.../taaf/game_api.py:214-221` documents the
   exact reason ("a mid-play RESET on level N>0 … snaps the player back to level 1"). The
   checkpoint is switched on in the deployed duck. There is no unbanked win here to collect.
3. **Repositioning is not the wall — intra-level re-treading is.** With the savestate on, the
   budget splits: repos **8.7 %** (median) / re-walking known candidate prefixes **82.7 %** (median)
   / genuinely-new frontier actions **8.7 %** (median). Level checkpointing cannot touch the 82.7 %.
   The engine offers *no* intra-level savestate, so that share is only addressable by a better cell
   function / search order — i.e. exactly the "tune the cell function" downgrade.

Depth payoff, the only thing that scores: at CAP = 200 000 env steps, `levelreset` beat `prefix` on
depth in **1 of 8 games** (tu93, level 4 vs 3). Seven of eight tied. Portfolio total 8 levels vs 7.

**Recommendation:** do not rebuild the explorer around level checkpointing. Keep
`ONLY_RESET_LEVELS=true` (it is load-bearing) and spend effort on the 83 % rewalk bucket instead.

---

## Task 1 — reset semantics, per game

Method: black-box `env.reset()` / `env.step()`; white-box only to *position* the game at level 3
(`set_level(3)` + `_score=3`) so "does a mid-level reset preserve progress?" can be asked without
first solving the game. Source read: `arcengine/base_game.py:305-330` (`handle_reset`,
`full_reset`, `level_reset`) and `arc_agi/api.py:42`. No dev game overrides any of the three reset
methods (grepped all 25 `environment_files/*/*/*.py`) — only `on_set_level`.

### `ONLY_RESET_LEVELS` **unset** (default)

| game | levels | mid-level reset preserves `levels_completed`? | double reset wipes to L0? | r1 (lv, idx, full_reset) | r2 (lv, idx, full_reset) |
|---|---|---|---|---|---|
| cd82 | 6 | YES | YES | 3, 3, False | 0, 0, True |
| ft09 | 6 | YES | YES | 3, 3, False | 0, 0, True |
| ls20 | 7 | YES | YES | 3, 3, False | 0, 0, True |
| ka59 | 7 | YES | YES | 3, 3, False | 0, 0, True |
| sb26 | 8 | YES | YES | 3, 3, False | 0, 0, True |
| sk48 | 8 | YES | YES | 3, 3, False | 0, 0, True |
| tu93 | 9 | YES | YES | 3, 3, False | 0, 0, True |
| lf52 | 10 | YES | YES | 3, 3, False | 0, 0, True |

### `ONLY_RESET_LEVELS=true` (the production setting)

| game | mid-level reset preserves? | double reset wipes to L0? | r2 (lv, idx, full_reset) |
|---|---|---|---|
| cd82 … lf52 (all 8) | YES | **NO** | 3, 3, False |

**Uniform across all 8 games in both configurations.** Zero variance, zero silent wipes.

Mechanism (verified in source, not inferred):
- `level_reset()` clones only `self._levels[current_index]` and calls `set_level(current_index)`.
  `_score` is untouched, and `levels_completed` on the frame *is* `_score`
  (`base_game.py:247,258`). Hence progress is preserved by construction.
- `set_level()` sets `_action_count = 0`, which is why the *second* consecutive reset is a full
  reset in the default config.
- With `ONLY_RESET_LEVELS=true`, `handle_reset` takes the `level_reset()` branch unconditionally
  (unless the game state is `WIN`), so `_action_count` never matters. **Reset becomes an
  unconditional level savestate and you can never return to level 0 within a play.**

---

## Task 2 — the prefix-replay tax

CAP = **200 000 env steps** per (game, mode); wall cap 240 s (never binding; the slowest run was
98 s). "Env steps" counts every `step()` **and** every `reset()`. Budget is split into three
buckets:

- **repos** — resets + replay of the solved prefix, i.e. re-reaching the current level's start.
- **rewalk** — re-executing the already-known part of a BFS candidate inside the level.
- **novel** — the single frontier macro at the end of each candidate (the only genuinely new step).

### Whole-run budget split

| game | mode | depth | total steps | repos % | rewalk % | novel % |
|---|---|---|---|---|---|---|
| cd82 | levelreset | 2 | 200 000 | **16.8 %** | 66.5 % | 16.8 % |
| cd82 | prefix | 2 | 200 006 | 68.3 % | 24.6 % | 7.1 % |
| ft09 | levelreset | 0 | 35 | 54.3 % | 0.0 % | 45.7 % |
| ft09 | prefix | 0 | 50 | 68.0 % | 0.0 % | 32.0 % |
| ls20 | levelreset | 1 | 200 002 | **4.6 %** | 90.7 % | 4.6 % |
| ls20 | prefix | 1 | 200 028 | 40.2 % | 56.3 % | 3.5 % |
| ka59 | levelreset | 0 | 200 003 | **12.0 %** | 76.0 % | 12.0 % |
| ka59 | prefix | 0 | 200 002 | 21.8 % | 67.3 % | 10.9 % |
| sb26 | levelreset | 0 | 200 002 | **14.1 %** | 71.9 % | 14.1 % |
| sb26 | prefix | 0 | 200 007 | 25.0 % | 62.5 % | 12.5 % |
| sk48 | levelreset | 1 | 200 010 | **8.7 %** | 82.7 % | 8.7 % |
| sk48 | prefix | 1 | 200 022 | 25.0 % | 68.0 % | 7.0 % |
| tu93 | levelreset | 4 | 200 002 | **6.3 %** | 87.3 % | 6.3 % |
| tu93 | prefix | 3 | 200 039 | 62.5 % | 34.6 % | 2.9 % |
| lf52 | levelreset | 0 | 113 323 | **3.1 %** | 93.7 % | 3.1 % |
| lf52 | prefix | 0 | 116 882 | 6.1 % | 90.9 % | 3.0 % |

ft09 is degenerate (BFS exhausts in 35 steps — see surprises) and is excluded from the medians.

- **levelreset repos: median 8.7 %** (range 3.1–16.8 %) → *below the 30 % rebuild bar.*
- **prefix repos: median 25.0 %** (range 6.1–68.3 %), and it climbs steeply with depth.
- **rewalk under levelreset: median 82.7 %** — the actual dominant cost.

### Per-level repositioning share (only levels where real search happened)

| game | level | levelreset repos / level steps | prefix repos / level steps |
|---|---|---|---|
| cd82 | L1 | 6 199 / 37 519 = 16.5 % | 43 393 / 74 713 = 58.1 % |
| cd82 | L2 | 26 209 / 157 324 = 16.7 % | 90 961 / 119 000 = **76.4 %** |
| ls20 | L1 | 7 489 / 181 025 = 4.1 % | 76 905 / 179 245 = 42.9 % |
| sk48 | L1 | 4 883 / 47 547 = 10.3 % | 25 200 / 35 104 = **71.8 %** |
| tu93 | L2 | 2 759 / 41 883 = 6.6 % | 82 770 / 121 894 = 67.9 % |
| tu93 | L3 | 2 058 / 27 865 = 7.4 % | 30 184 / 35 211 = **85.7 %** |
| ka59 | L0 | 24 001 / 199 999 = 12.0 % | 43 638 / 199 998 = 21.8 % |
| sb26 | L0 | 28 112 / 199 998 = 14.1 % | 49 978 / 200 003 = 25.0 % |
| lf52 | L0 | 3 559 / 113 319 = 3.1 % | 7 118 / 116 878 = 6.1 % |

The prefix tax is real and it *does* exceed 30 % — but only for the naive prefix-replay mode, which
is the thing already superseded in production. At L0 the two modes are near-identical by
construction (empty prefix; prefix mode just pays a second reset), which is the correct control.

### Raw speedup: env steps to reach depth k

| game | depth | levelreset | prefix | speedup |
|---|---|---|---|---|
| cd82 | 1 | 5 157 | 6 253 | 1.21× |
| cd82 | 2 | 42 676 | 80 980 | **1.90×** |
| ls20 | 1 | 18 977 | 20 753 | 1.09× |
| sk48 | 1 | 152 463 | 164 886 | 1.08× |
| tu93 | 1 | 30 213 | 32 326 | 1.07× |
| tu93 | 2 | 33 403 | 42 776 | 1.28× |
| tu93 | 3 | 75 286 | 164 730 | **2.19×** |
| tu93 | 4 | 103 151 | not reached (>200 000) | ≥1.94× |

Speedup is ~1.1× at depth 1 and grows to ~2× by depth 3 — real, monotone in depth, and consistent
with the theory. It is just not the order-of-magnitude the "wall-clock killer" framing implied.

---

## Task 3 — depth ceiling per mode (CAP = 200 000 env steps)

| game | levels in game | levelreset depth | prefix depth | Δ |
|---|---|---|---|---|
| cd82 | 6 | 2 | 2 | 0 |
| ft09 | 6 | 0 | 0 | 0 |
| ls20 | 7 | 1 | 1 | 0 |
| ka59 | 7 | 0 | 0 | 0 |
| sb26 | 8 | 0 | 0 | 0 |
| sk48 | 8 | 1 | 1 | 0 |
| tu93 | 9 | **4** | 3 | **+1** |
| lf52 | 10 | 0 | 0 | 0 |
| **total** | | **8** | **7** | **+1** |

Deepest consecutive level per budget bracket:

| game | mode | ≤10 k | ≤50 k | ≤150 k | ≤200 k |
|---|---|---|---|---|---|
| cd82 | levelreset | 1 | 2 | 2 | 2 |
| cd82 | prefix | 1 | 1 | 2 | 2 |
| tu93 | levelreset | 0 | 2 | 3 | 4 |
| tu93 | prefix | 0 | 2 | 2 | 3 |
| sk48 | levelreset | 0 | 0 | 0 | 1 |
| sk48 | prefix | 0 | 0 | 0 | 1 |
| ls20 | both | 0 | 1 | 1 | 1 |
| ka59 / sb26 / lf52 / ft09 | both | 0 | 0 | 0 | 0 |

The checkpoint buys depth **only where the search can already crack levels at all**. On the five
games where the search stalls at L0/L1 for representational reasons, the savestate is irrelevant —
which is the strongest single argument for the downgrade.

---

## Task 4 — cell-granularity sweep (`GoExploreExplorer.block`)

Budget 10 000 env steps, 5 games × block ∈ {2,4,8,16} × 3 seeds = 60 runs, offline.

| game | b=2 mean lv (max) / archive | b=4 | b=8 | b=16 |
|---|---|---|---|---|
| cd82 | 0.33 (1) / 866 | 1.33 (2) / 243 | 1.33 (2) / 66 | 0.67 (2) / 8 |
| ls20 | 0.33 (1) / 573 | 0.00 (0) / 295 | 0.33 (1) / 5 | 0.00 (0) / 1 |
| tu93 | 0.00 (0) / 412 | 0.67 (1) / 10 | 0.67 (2) / 2 | 0.00 (0) / 1 |
| sk48 | 0.00 (0) / 757 | 0.33 (1) / 165 | 0.00 (0) / 3 | 0.00 (0) / 1 |
| lf52 | 0.00 (0) / 376 | 1.00 (1) / 57 | 0.00 (0) / 1 | 0.00 (0) / 1 |
| **mean levels (all games × seeds)** | **0.133** | **0.667** | **0.467** | **0.133** |
| per-game best-seed level sum | 2 | 5 | 5 | 2 |

**The curve is not flat — it is an inverted U peaking at block = 4–8** (5× the mean levels of the
extremes). Archive size collapses ~4 orders of magnitude across the sweep (866 → 8 cells on cd82;
block 8/16 degenerates to 1–3 cells on tu93/sk48/lf52, i.e. the cell function stops discriminating
anything at all). The docstring's suspicion that intermediate granularity was never properly tested
is vindicated in direction, but the magnitude is small in absolute terms (best mean = 0.67 levels
per game) and seed variance is large (cd82 b=16 gives 2, 0, 0 across seeds). The current default is
`block=8`, i.e. already inside the good band; moving to `block=4` is a plausible but low-confidence
+0.2-levels tweak, not a lever.

Caveat: 3 seeds is thin, block=2 is also ~9× slower in wall-clock per env step (archive sampling is
O(|archive|) per decision), so at equal *wall-clock* budget fine granularity is worse still.

---

## Surprises / things that contradicted the premise

1. **`ONLY_RESET_LEVELS=true` turns off the double-reset escape hatch entirely.** Under the
   production env var, *two* consecutive resets still leave you on the same level (verified on all
   8 games). Any code that assumes "double reset ⇒ back to level 0" — including
   `goto_level0()` in `scratchpad/prefix_replay_cost.py` and, per the campaign memory, the
   geodesic / portfolio "new play" machinery — is a **silent no-op under the deployed
   configuration**. This is the most actionable finding in the whole experiment and is independent
   of the Go-Explore verdict. Worth auditing every call site that relies on it.
2. **The idea is already shipped.** `taaf/game_api.py:214-221` sets `ONLY_RESET_LEVELS=true` with a
   comment describing precisely the checkpoint semantics this experiment was meant to discover. The
   "free savestate" was found and banked months ago; the research note ranking it #1 was stale.
3. **The prefix tax was real but the wrong villain.** The campaign belief ("prefix replay is the
   wall-clock killer for going deep") is *directionally* correct — 76–86 % of the deep-level budget
   in prefix mode — but eliminating it entirely converts to only +1 level across 8 games, because
   the freed budget flows straight into the 83 % intra-level rewalk bucket instead.
4. **`novel` never exceeds ~17 % of the budget in any mode.** Even with a perfect level savestate,
   ≥83 % of every env step is re-execution of something already seen. That is the number a rebuild
   would have to attack, and the engine gives no primitive for it (no intra-level snapshot,
   `copy.deepcopy` unavailable at the REST eval).
5. **ft09 exhausts its BFS in 35 env steps** — 16 candidate macros, every resulting object-state key
   already in `seen`, queue empty. Identical in both modes. That is a perception/macro-vocabulary
   failure, not a budget failure, and no reset policy can fix it. lf52 similarly "exhausts" at
   113 k steps with an empty queue.
6. **Depth-1 speedup is only ~1.07–1.21×.** The savestate's advantage is invisible until level 3+.
   Since 5 of 8 dev games never get past level 1 with this search, the checkpoint is inert on the
   majority of the portfolio.

---

## What was capped / not measured

- CAP = 200 000 env steps per (game, mode); 240 s wall cap per run (never binding).
- 8 games × 2 modes; sweep on 5 games × 4 blocks × 3 seeds at 10 000 steps.
- Wall-clock cost of a reset (which *is* nonzero even though a reset contributes 0 to
  `actions_taken` in `scorecard.py`) was not separately profiled; all figures above are env-step
  counts, not seconds.
- The search used is the existing BFS-with-object-dedup from `prefix_replay_cost.py`, not the
  deployed duck's policy. Absolute depths are therefore not comparable to leaderboard results; only
  the mode-vs-mode contrast is.
