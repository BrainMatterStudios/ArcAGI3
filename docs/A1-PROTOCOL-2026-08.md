# A1 PROTOCOL — August 2026 gate sequence (PRE-REGISTERED 2026-07-27)

**This document is written BEFORE any GPU spend or data collection. Thresholds
below are frozen. Any post-hoc change must be recorded in §7 as an amendment
with justification, and results under amended thresholds are exploratory, not
confirmatory.** Derived from the 2026-07-26 independent audit
(`docs/REVIEW-2026-07-26-independent-audit.md`) and its verified corrections.

## 0. Standing conditions (all runs)

- Duck tree: **`scratchpad/taaf_scored_ref`** (the downloaded scored dataset,
  Jul-03) via `TAAF_ROOT` — never the drifted local `_adopt` tree.
- Sampling: **temp 0.6 / top_p 0.95 / top_k 20** (the shipped
  `setup_commands.json` values; the temp-0.3 "defect" never shipped).
- Patches applied-but-toggled on every arm (`APPLY_DUCK_FIXES`,
  `APPLY_LEDGER_PATCH`, `APPLY_DOCTRINE_PATCH` = 1); arm identity is env
  toggles only. Applied-off is Stage-1-proven byte-identical.
- Instrument: `scratchpad/rl_gate/ab_driver.py` (paired arms per game/rollout,
  hard 45-min per-game kill; cost bounded by wall-clock, never action caps).
- Holdout: the pinned 13 games (commit 45cfb11): ff01 sy01 sq01 cs01 cx01
  dm01 lo01 sc01 / ic01 fs03 sk01 bd01 tp02. Official dev games appear in NO
  gate (they are training-corpus-contaminated for run-8).
- Metric: **paired per-game levels-completed deltas**. RHAE/efficiency on the
  holdout is NEVER a selection signal (padded baselines).

## 1. Gate 0 — serve-verify v2 (hard gate, ~3-4h GPU)

Kernel `arc-agi-3-serve-verify-k3` v2 (builder at commit 215394f+): in-kernel
merge with peft-prefix-safe delta assert, pre-flight base NLL < 6.0, merged
target-NLL gain ≥ 2% on the 43 val rows, duck-line vLLM serve of MERGED then
BASE, probes well-formed, outputs differ. Run for BOTH `checkpoint-8` and
`sft_adapter`. **Timed eval-time merge rehearsal recorded** (feeds the debut
kernel's 9h budget). FAIL on either artifact ⇒ that arm is dropped; FAIL on
both ⇒ SFT branch pauses, no-adapter stack proceeds (§4).

## 2. Gate 1 — base-duck pilot banding (R4, ~6-8h GPU)

1 rollout × 13 games × base arm. **Freeze rule:** keep games where base
completes ≥1 level AND < all levels. If <8 games survive, widen from the
verifier-proved pool (same rubric R1-R3, R5) rather than relaxing the band.
The frozen list is the panel for Gates 2-3; record it in §7 before Gate 2.

## 3. Gate 2 — A1 checkpoint sweep (~15-18h GPU)

- Server arms: **base / checkpoint-8 / sft_adapter(step-15)** (one vLLM
  session each; ab_driver run per session).
- 2 paired rollouts per game per arm on the frozen panel.
- **Adapter GO iff, vs base:** (a) total paired level-delta ≥ **+3** across
  the panel; (b) no single game regresses by ≥ **2** levels summed across
  rollouts; (c) behavioral scorecard non-inferior (first board-changing
  action median ≤ 2; HUD-anchoring game_overs not increased).
- Selection between ckpt-8 and step-15: higher total paired delta; ties →
  ckpt-8 (earlier checkpoint, OOD-peak literature).
- The −12.5% val gain is fit evidence only (100% episode overlap) and never
  enters this decision.

## 4. Gate 3 — doctrine A/B (~6-8h GPU, same endpoint as the selected arm)

Arms: base-toggles vs **D1+D2** (`DOCTRINE_FIELDGUIDE` + `DOCTRINE_PLAYBOOK`,
plus `LEDGER_TWO_TIER` as a third arm if hours allow), cross-tested against
the Gate-2 SELECTED server config. **GO iff** paired level-delta ≥ **+2**
across the panel AND no-regression conditions as §3(b,c). D3/ACTION7 is
**excluded** from holdout GO criteria (1/249 games use ACTION7) — decided
separately on official-dev evidence only, always coupled to patch_action7.

## 5. Debut (first slot after all gates green — promptly; earliest-submission tie-break)

One stacked config: selected adapter (if GO) + GO'd prompt arms + estimator
fix + adaptive budget + stall floor + **in-submission serving assert**
(SAFE mode, `assert_server_model_arg` + NLL chain; base fallback on failure).
If the adapter NO-GOes: debut the no-adapter stack (it must still pass Gate 3
on the base server). **Stop rule (frozen):** collect **7 debut draws** before
any verdict; revert iff their mean < (pinned base mean − 0.10). At sd 0.149
this false-kills a truly-equal config ~4% of the time and a true +0.10 config
~50%… asymmetric by design: the pinned config keeps the floor, and a real
improvement re-enters via Gate re-runs. Single-draw verdicts are banned.

## 6. Budget ledger (30h/week pool)

| Gate | hours | cumulative |
|---|---|---|
| 0 serve-verify v2 (both artifacts) | 3-4 | 4 |
| 1 pilot banding | 6-8 | 12 |
| 2 A1 sweep (3 server arms) | 15-18 | 30 |
| 3 doctrine A/B | 6-8 | *week 2* |
Overflow rule: Gate 3 slides to week 2; never compress rollout counts to fit.

## 7. Amendments / frozen-panel record

### 7.1 — A/A noise floor MEASURED; Gate 3 and B-track Gate B are under-powered (2026-07-31)

The thresholds in §3/§4 were pre-registered **before** anyone measured the
replicate-to-replicate variance they are supposed to exceed. That variance now
exists, computed from four same-config replicate pairs already in
`scratchpad/rl_gate/episodes` (`k3_sweep_<game>_a` vs `_a2`, `_b` vs `_b2`):

| pair | Δ levels | Δ score |
|---|---|---|
| ar25 a | +1 | +13.89 |
| bp35 b | 0 | −0.02 |
| cn04 a | 0 | +1.08 |
| g50t b | +1 | +1.73 |

**RMS per game: 0.707 levels, 7.019 score points.** (The score figure reproduces
the independent 2026-07-31 research sweep's ~7.0 estimate exactly, arrived at
separately.)

Propagated to a 13-game panel at §3's **2 paired rollouts per game**, the SD of
the panel total is `0.707 × √13 / √2 = 1.80` levels:

| gate | threshold | z | one-sided false-positive rate |
|---|---|---|---|
| Gate 2 (adapter) | +3 | 1.66 | **4.8%** — marginal but defensible |
| Gate 3 (doctrine) | +2 | 1.11 | **13.4%** — fires on noise 1 in 7 |
| B-track Gate B | +2 | 1.11 | **13.4%** — same |

**Minimum detectable effect at 2σ with 2 rollouts/game is +3.6 total levels, not
+2.** For +2 to sit at 2σ requires **6 paired rollouts per game (84 per arm)** —
three times the registered spend.

**Consequence.** Gate 3 and Gate B as written cannot distinguish a real +2 from
noise at any defensible confidence. Do not read a GO from either as evidence.
Before they run, choose one and record it here: (a) raise both thresholds to
**+4**, (b) raise to 6 paired rollouts/game and keep +2, or (c) declare both
exploratory. Gate 2's +3 stands.

**Caveats, stated so this is not over-read.**
- n=4 pairs. The 95% CI on RMS_levels is **0.40 to 2.63** — the estimate is
  directionally decisive but numerically loose. Widen it with every future
  same-config pair; this is cheap data that accumulates for free.
- These pairs are K3-teacher runs on **official dev games**, not the deployed
  Qwen on the pinned holdout. Direction should carry; magnitude may not.
- The A/A mean delta is **+0.50 levels / +4.17 score, not zero**. A true A/A
  should centre on zero. With n=4 this is within noise, but it is equally
  consistent with an order effect (every `_a2`/`_b2` ran after its partner).
  **Check this before trusting any paired design** — a systematic
  second-run advantage would bias every A/B that runs arms in a fixed order.
  Cheap fix: randomise arm order per game and record the order.
