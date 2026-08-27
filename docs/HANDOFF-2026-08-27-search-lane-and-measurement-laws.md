# HANDOFF 2026-08-27 — START HERE

Everything below was measured this week against the live Kaggle API, the
engine source, or `CompetitionArcadeServer` (the authoritative harness).
Memory index: `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/MEMORY.md`
(detail in `arcagi3-campaign-log-2026-08-22-25.md`, which holds the
search-lane laws).

## 1. TONIGHT (2026-08-28 00:01Z): v8 crack-or-nothing arm — ARMED

- Runner `scripts/submit_v8_20260828.py`, **pid 29105**, detached
  (`start_new_session`), marker `logs/v8_20260828.marker` (unclaimed).
- Kernel `ahmedmobasher86/arc3-duck38-v8` v1, **svid 345333330**, code hash
  `a6257e6dc1ebb927…`, local==remote, mock rc=0 verified.
- WHAT IT IS: v12 base (the bytes that flew 1.55) + the v8 search graft in
  crack-or-nothing config: zero-action frame-0 pre-screen, engage ONLY
  `ft09_gf2` (the one class measured to crack), specialist cap 4000 actions,
  **stall-grind OFF**, banking + stop-after-crack ON. No effort_medium.
- SMOKE (kernel `arc3-v8-smoke` v4, RTX Pro 6000): **5/5 bars**. ft09
  admitted → cracked 6/6 in 1233 actions → **banked in 75 → engine 100.0**
  (plays [3.512, 100.0]); vc33/dc22/sk48 declined at **0** probe actions;
  envelope maxima trivial (cumulative grind 0.6s of 2700s).
- EV +3.43 pts/game at p=4%, **floor measured at 0.00**. PRE-REGISTERED
  READING: lottery ticket with a zero floor — a null draw in the 0.69–1.30
  base band is EXPECTED if the public half holds no ft09-class game and is
  evidence about the hidden set, not against the mechanism; >1.74 re-banks
  the LB best (current best 1.74, banked 08-17).

## 2. Two measurement laws that govern everything now

1. **CV ≈ 0.20 of the mean** (measured in all three eras: 0.16 @mean 0.31,
   0.21 @0.93, 0.18 @1.43). At our mean ~1.4, per-draw sigma ≈ 0.28, so a
   2-draw read has SE 0.20 and can only detect effects **≥ ~0.55**. EVERY
   2-draw ladder read this campaign ran was underpowered, INCLUDING the
   effort_medium reversion (diff 0.43 < 0.55 — effort is neither confirmed
   harmful nor cleared). Only ≥0.55-class levers are slot-testable; smaller
   ones need offline/mechanism validation or bundling.
2. **Offline engine-action costs must be ×2.7 before comparison with a live
   cap** (tu93: 84,687 offline vs 225,511 on the competition server). The
   competition guard swallows resets at `_action_count==0` (api.py:316-334),
   billing an action without restoring state.

The seven search-lane laws (banking is the entire value; two scoreboards;
detection ≠ crack; failed engagement is a level-weighted step function;
partial grinding is net-negative; etc.) are in the memory campaign log.

## 3. Live series and what is settled

1.29 / 1.74 / 1.55 / 1.33 / 1.51 / 1.66 / 1.50 / 1.24 / 1.00 / 0.99 / **1.45**
(n=11, mean 1.38, best **1.74**).

- **DRIFT REFUTED**: duck-38 v2 identical bytes drew 1.29 / 1.74 / **1.45** —
  in range. Serving also flat (636 vs 642 tok/min/session at conc 28, decode
  fingerprint identical). The environment is exonerated; the 0.99/1.00/1.24
  stretch was low draws and/or arm effects.
- Killed by measurement this cycle: MTP, inspection-routing cascade,
  config-dispatch, sc25 token-threshold, dc22×bundle, flat triage-at-55,
  no-op guard, ALL LLM-judgment persona forms (menu recognition = chance),
  vLLM 0.27 lane (untestable on the Kaggle image).
- Surviving/validated but UNFLOWN: yield_carryover (mechanism telemetry PASS,
  confounded smoke — re-smoke WITHOUT effort), bugfix pack (9 patches),
  truthful_telemetry, expect_queue, archetype_triage, playbook_rider.

## 4. Next moves (in priority order)

1. **Read the 08-28 v8 draw (~09:30Z)** against the pre-registered bands.
   A null is expected and informative; do NOT treat it as a mechanism
   failure. Two nulls ⇒ the ft09-class frequency in the public half is
   likely <4% and the lane's value must come from MORE crackable classes.
2. **Grow the crack inventory** — this is the highest-value build. Only
   ft09_gf2 converts today. tn36/sc25/wa30 detect but fail after 345k–879k
   actions. Either make those solvers actually crack, or find new mechanic
   classes with cheap frame-0 screens. Each additional cracking class adds
   ~p×85.7 with a zero floor (the pre-screen pattern generalises:
   `submission/_search_core/prescreen.py`).
3. **Re-smoke yield_carryover without effort_medium** (its previous read is
   confounded) — the only remaining harness lever with a measured mechanism
   (0 duplicate snippets vs 71; max 5 slices vs 35).
4. Slot policy: safe redraws are a dead plan (E[best] ~2.0, P(≥2.39)≈0).
   Prefer zero-floor lottery arms and ≥0.55-class levers. ~34 slots remain
   to the Sep 30 milestone.
5. Watch: GLM-5.3 weights (~08-28) — the ONLY reopen condition for the
   persona/judge track (re-run the 38-case pack for cents).

## 5. Ops facts that cost time when forgotten

- The **competition-source attachment** gates the RTX Pro 6000 pool;
  `machine_shape` + `--accelerator` alone still bind P100.
- `kaggle kernels output` hangs on kernels with big outputs — fetch the
  results file by name from the REST output listing instead.
- Detached `start_new_session` processes survive session cutoffs; harness
  background tasks and in-session agents do NOT.
- **Verify the clock with `date -u` before treating a slot as missed** — a
  harness date notice ran ahead of the machine clock on 08-25 and I killed
  an armed runner over a phantom missed day (restored, no harm).
- Submission messages: keep under ~950 chars (a ~1100-char message 400s),
  and the honesty gate keyword-scans the message against notebook markers.
