# RESEARCH 2026-08-22 — slot-portfolio math + how the top scorers reach 3+

Two agent reports, cross-consistent. MC scripts: hunt3/slot_mc.py and
top3/arithmetic.md in the 08-22 session scratchpads (key numbers preserved
here). LB context at pull time: cstl 3.57, Tufa 3.04, 82 teams >=2.0, us 1.74
rank ~173.

## A. The public recipe is a DISTRIBUTION, not a floor: mu~2.05 sigma~0.37

Fit two independent ways (low-effort cohort ladder 2.39@3subs..2.72@13subs,
rmse 0.033; forum 736578 reports the same recipe drawing 1.4 and 2.2 on
different accounts). Consequences:
- Our pack-v22 1.66 is an IN-BAND draw of the recipe (p(<=1.66)~0.15).
  Recipe-parity is NOT refuted; parity redraws are legitimate high-mu fillers.
- Everything in the 2.39-2.76 cohort above ~2.05 is order statistics. LHS
  2.76@38subs = E[max38] of mu 2.2 exactly. Tufa 3.04@113 = mu~2.44 + volume.
- Podium bar for Sep 30 realistically 2.8-3.0+.

## B. Slot-portfolio math (~39 slots left; LB = MAX over draws)

E[max_n] grows ~log(n): n=2:+0.56sigma, 13:+1.67sigma, 39:+2.16sigma. MC
(40k trials, banked 1.74): nightly redraw of one composite E[best] 1.96-2.09,
P(>=2.39)<=0.005 — DEAD PLAN. Lever ladder (6 levers x 2-draw reads,
keep/revert) E[best] 2.32. Adding a deliberate high-variance arm (sigma 0.40)
2.41. Recommended mix (12 ladder slots -> exploit + variance alternation,
serving lever p=0.5): E[best] ~2.60 (2.45 if serving lever fails).
**The slot plan itself is worth +0.4-0.6 — largest single number found.**

KEY INSIGHT: under MAX scoring, run-level VARIANCE = MEAN in value
(+0.2 sigma at fixed mu ~ +0.43 E[best]). Run-level sigma needs CORRELATED
bets: budget reallocation (archetype-triage — hence PROMOTED: it raises mu
AND sigma), aggressive depth configs, temp 1.0. Discipline: 2-draw lever
reads never 3; no idle slots.

## C. What 3.57 requires (arithmetic verdict)

3.57 = 196 pts over ~55 public-half games. One full game win (played or
banked via post-WIN RESET replay) = +1.82 LB public-half / +0.91 expected.
cstl needs only floor-level filler + ~0.8-1 banked full win per run. Draw
luck excluded (needs mu>=3.09 for E[max31]=3.57; their 2.70->3.57 jump
p<0.01 as luck). Their edge is ADDITIVE and model-agnostic (+1.1-1.3 pre-3.8
edge survived the swap unchanged; 08-20 jump == the field's 3.8 delta) ⇒
hybrid: duck-class LLM + engine-speed search takeover + banking (team =
CF IM + game-bot programmers). IMPORTANT CALIBRATION: if search scaled on
the hidden set the score would be 8-20, not 3.6 ⇒ hidden set is largely
search-resistant; 3.57 is a MODEST mechanism (~1-2 extra full wins/run).

## D. Our gap to the mechanism (build list, offline-falsifiable)

Banking plumbing already validated (graft_bank) — it just never fires:
blind reset-replay BFS full-cracks 1/25 dev games (tu93). Binding constraint
is ALGORITHMIC (wa30: blind BFS intractable at 113k states; hand-built
grab-drag forward model + A* solves in 0.5s). Build = model-based
per-archetype search cores: avatar/coupled-movement forward model, grab-drag
generalization, object-click enumeration (4096 -> dozens via segmentation),
macro-actions from motifs.json, transposition tables, best-first. Dispatch
off public tags + frame-0 available_actions (both live on hidden games).
FALSIFIER (zero slots): full-crack count on our 25 fixtures in 70 engine-min
— current 1/25, target 3-5/25. Each 1/55 hidden crack rate ~ +0.9 E[LB].
Fund engagements from archetype-triage's reclaimed ~49wh (law #3 case
required). Filler mean co-equal: +0.1 filler == +0.11 crack rate.

## E. Hunt-3 lever verdicts (new, beyond the 08-22 verdict table)

- vLLM 0.27.1 + DFlash2 lane: OPEN, field-proven assets on Kaggle (saltb0x
  wheelhouse; bbucxi/qwen3-8-27b-dflash2 draft 08-20). 0.19-MTP-null does
  NOT transfer to block drafting + new scheduler. Probe kernel
  (serving-lab2) IN FLIGHT. Kill: <+15% tok/min/session conc28 or any
  parser failure.
- SM120 FP8 quality hypothesis (novel): all offline data was H100-served;
  quality on the competition GPU never measured; vLLM 0.20 notes carry
  Qwen3.5-FP8-on-Blackwell accuracy fixes. Battery rides serving-lab2.
  If null, live discount collapses to draw-optimism + set composition.
- Context-budget raise 48-64k: DOMINATED — naive raise costs 34-101% of the
  run in prefill at measured miss rates; bugfix patches 6+7 + carryover
  deliver the value inside 32k. Revisit only post-0.27 prefix-cache fix.
- Preserve-history-on-request-error: +0.02-0.05 rider ⇒ bugfix-pack patch 9
  (~20 LoC), never a slot.
- Cross-game in-run transfer: topology-capped (28x4 generations, 25% of
  plays unreachable) ⇒ ship the STATIC playbook prompt rider (playbook.json
  priors + frame-0 dispatch) instead; dynamic store only if static moves
  per-level actions.
- Bruggen anim bundle + jinbowang playbooks: fold into anim-digest variants
  and the static playbook rider respectively.

## F. Queue after this research (supersedes the 08-22 doc's ordering)

1. serving-lab2 probe (in flight, zero slots)  2. effort_medium smoke (in
flight)  3. yield_carryover smoke  4. bugfix pack incl. patches 6+7+new-9
5. archetype-triage (PROMOTED: mu+sigma)  6. expect-queue rider
7. truthful_telemetry + static playbook rider  8. SEARCH-CORE BUILD (the
cstl-class mechanism; offline falsifier first: 3-5/25 target)
Slot plan: 2-draw reads down this ladder; parity-arm redraws or the best
composite on non-read nights; add a high-variance arm once triage validates.

## Appendix: cross-game seam implementation cautions (Explore recap)

If the static playbook rider (or ever the dynamic store) is built: inject in
the USER prompt, not the system prompt — per-game system variants split the
shared vLLM prefix-cache KV, and the system message is never trimmed;
key any in-run signal on levels_completed (live _compute_final_score returns
0.0 when base_actions_per_level is None); play-wrapper seam = 
_HarnessGameSession.play (precedent duck_patches.py:6014), prompt seam =
_build_user_prompt (same seam carryover uses); import-by-value trap applies
(harness_mem.py:28-32). First 28 plays get zero benefit by construction;
same-game clones sit ~25 indices apart in-sim => no play-to-play transfer.
