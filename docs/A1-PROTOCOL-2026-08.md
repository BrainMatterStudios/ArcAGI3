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

*(empty at pre-registration — append here only)*
