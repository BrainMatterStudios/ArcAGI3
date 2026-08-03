# A/B round 3 — archetype playbook (2026-08-03)

Kernel ab-wmr v3, COMPLETE in 4.19h, waves A,B,B,A, diagnosis-targeted panel
(8 target games + ft09/re86 continuity). Playbook presence hard-proven per wave
(pre-wave probe + 10 sampled real prompts, 0 mismatches, stable sha256 per arm).

## Primary (pre-registered): NULL — zero B-only unlocks on any target game

Paired target results are ARM-IDENTICAL: cn04 A=[1,0] B=[1,0]; lf52 A=[1,1]
B=[1,1]; ls20/m0r0/wa30/g50t/tr87/sk48 all 0/0 in both arms. The playbook did
not convert a single diagnosed goal-inference failure into an unlock. The
diagnosis's "reachable with better priors" prediction is REFUTED for the
prompt-playbook implementation at 27B: the model receives the heuristics
(proven) but does not operationalize them.

Possible harm signal: ft09 (continuity game) collapsed in B (A=[3,2], B=[0,0],
with 12 antifreeze triggers in one B run) — the largest paired delta of the
round. ft09 is historically volatile across rounds, so this is suggestive, not
conclusive; but B totals 4 levels vs A's 9 almost entirely via ft09.

## Secondary findings (real value)

1. **lf52 and cn04 now unlock WITHOUT the playbook** — in both arms. lf52 went
   0/196 in the base-config audit to 4/4 waves here; cn04 2/4. Something in the
   current stack (WMR trio + HUD rotation fix + antifreeze, all shared by both
   arms — or the post-ACTION7 era generally) reaches games the July base never
   did. Not attributable within this design; the live v6 draws sample it.
2. **The HUD slow-tick fix works live**: m0r0's mask confirmed at 128 cells in
   all 4 waves (was 0 forever before the rotation fix).
3. **Antifreeze fires in production** (m0r0 9/5, ft09 12, tr87 2) and tr87
   still doesn't unlock — the breaker interrupts the freeze but the 27B cannot
   re-derive the symbolic mechanic. Perception-blocked confirmed.
4. **sk48 post-ACTION7: still 0/4** — the old ACTION7 bug was not sk48's
   binding blocker, as the diagnosis suspected.

## Consequences

- TAAF_PLAYBOOK stays pinned OFF for submissions; no further prompt-prior
  iterations without new evidence — prompting is not the lever at 27B.
- HUD rotation fix + antifreeze stay as defaults (verified live, no harm
  signal; ft09's collapse tracked the playbook arm, not these).
- The unlock lever standing after three A/B rounds: TRAINING the inference
  behavior (SFT v2 corpus with wrong-hypothesis→revision arcs), plus whatever
  capability the trace-license/Anthropic gates unlock. Prompt-side and
  harness-side menus are now both measured out.
