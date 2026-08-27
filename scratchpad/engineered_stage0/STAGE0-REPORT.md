# STAGE0-REPORT — engineered-agent tooling re-verification (2026-08-14)

Per `docs/DESIGN-2026-08-14-engineered-agent.md` §4 Stage 0. All runs on the
current toolchain: **arc-agi 0.9.9, arcengine 0.9.3, numpy 2.4.6, scipy 1.18.0**
(`.venv`, Python 3.12). Scripts + JSON evidence live in this directory; nothing
committed; **zero modifications to `src/arcagi3/`, `scripts/research_2026_07_01/`,
`submission/`, or any tracked file.**

## Verdict: ALL SIX PILLARS GREEN → Stage 1 may proceed. Kill criterion not triggered.

| # | Pillar | Verdict | Evidence |
|---|--------|---------|----------|
| 1 | Offline engine + 25 games | **GREEN** | `p1_engine_load.py`: 25/25 games load, reset, and step 30 random actions in **0.6 s total**; frames 64x64; per-game available_actions recorded. `p1_engine_load.json` |
| 2 | HUD hand-mask | **GREEN** | `p2_hud_mask.py`: 1200 random actions x 6 games, raw vs masked no-op rate vs the in-tree `MEASURED_NOOP` reference — near-exact parity: lf52 0.000→0.996 (ref 0.000→0.996), vc33 0.000→0.994 (0.995), tu93 0.000→0.594 (0.601), sb26 0.624→0.962 (0.961), re86 0.003→0.007 (0.008). m0r0 (no ref; 08-03 2-row correction) 0.253→0.462. `p2_hud_mask.json` |
| 3 | Wiggle probe battery | **GREEN** | (a) `p3_wiggle_battery.py` — Layer-2 spec battery (directionals x2 then component clicks, HUD-excluded, ≤16 actions): tu93 SELF mask 18 px with 9-px avatar blob; sb26 2/14, lp85 2/14, lf52 1/8 reactive clicks; DEAD masks everywhere; lf52 correctly yields the "no contingent body" fast negative. (b) `p3b_probe_core.py` (unmodified copy of `scratchpad/ideas/probe_core.py`, output path redirected): tu93 agency NMI **0.415**, lp85 click-AUC **0.962**, sb26 **0.759**, 800 steps, 2 s. |
| 4 | TransferExplorer resurrection | **GREEN — exact parity** | `p4_transfer.py`: historical 2026-06-29 config (DENSE, seed 0, budget 4000, same 8 games). TOTAL **16 = recorded 16**; tu93 **L4 with first level-up at action 431 = recorded "levelup@431"** to the action. Per-game: tu93 4, lp85 4, vc33 2, cd82 2, ar25/lf52/su15/m0r0 1. 65 s wall. `p4_transfer.json` |
| 5 | Human-replay budget extractor | **GREEN (rebuilt)** | The 08-08 `extract_budgets.py` + `budgets.json` are **not in the tree** (session-scratchpad casualty). `p5_budgets.py` re-derives the full 25-game table from `scratchpad/human_replays/extracted/` (340 replays, 34 s): actions/completed-level medians **ft09 24.0 (ref 24), vc33 44 (ref 44), ls20 91.5 (ref 91.5)** — exact; also matches memory on lp85 = 54 replays and cd82 apl 23.5. `p5_budgets.json` is the regenerated table; treat this script as the canonical in-tree extractor going forward. |
| 6 | true_score wiring | **GREEN** | `p6_true_score.py`: module imports by path; synthetic rows (max-over-clones, mean-over-games, string/None coercion) all correct; all 8 banked waves under `scratchpad/banked_waves_20260809/` score 28/28 rows and reproduce the documented values **pc_base 1.4751 ("1.475") and pc_cand 0.3134 ("0.313")**; struct 1.5198, w2_base 1.2039, etc. |

## Compatibility diffs made

**None to repo code.** No file in `src/arcagi3/`, `scripts/`, or `submission/`
was touched; no version drift blocked execution anywhere. New files are confined
to `scratchpad/engineered_stage0/`. The only derived file is
`p3b_probe_core.py`, a byte-identical copy of `scratchpad/ideas/probe_core.py`
except its JSON dump path, redirected so the run would not clobber the existing
`scratchpad/ideas/probe_core.json` artifact.

## Findings worth carrying into Stage 1

1. **Engine throughput** is far above need: 25-game load+step 0.6 s; a 4000-action
   TransferExplorer episode runs 3.5–15 s/game; probe_core 800 steps ≈ 2 s.
   Wall-clock will not bind the offline rig.
2. **TransferExplorer is deterministic-reproducible** across ~7 weeks of
   toolchain drift (level-up at the identical action index). The old per-game
   baseline numbers are trustworthy comparators for Stage-1 milestone (b).
3. **Battery target selection matters**: a color-deduped, centroid-clicking
   battery missed 100% of sb26's live buttons twice — (i) raw centroids land
   off-component on hollow shapes (fixed: click the component pixel nearest the
   centroid), (ii) the live answer buttons are the *second, smaller* instances of
   already-seen colors (fixed: distinct-colors pass, then duplicate instances,
   largest first). Both fixes are in `p3_wiggle_battery.py`; Stage-1's Layer-2
   should inherit them.
4. **Asset gap closed**: the budget extractor existed only as numbers in memory;
   `p5_budgets.py` is now the reproducible source of the budget table.
5. sb26 ground truth from the debug sweep: the only click-reactive region at L0
   is the answer strip (y 56–60, x 18–45); ACTION5/ACTION7 change nothing at L0
   open. Useful as a unit fixture for Layer-2 tests.
