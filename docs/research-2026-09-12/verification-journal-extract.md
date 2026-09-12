# Raw verifier / skeptic / strategist returns — workflow wf_5507c1cc-d9e (2026-09-11/12)

Machine extract of each agent's structured return. Provenance only; the audited summary is R-validation-audit-0912.md.


---

## agent-0

```json
{
 "claim": "\"Sizing: the buy-calls family closed by arithmetic\" (offkaggle/REGIME_WAVE_STATUS.md:1492-1540, commit 636d85d): MTP-3 per-position acceptance 76.0/58.3/45.4% = 2.80 tokens/forward (ceiling 4.00); MTP-5 = 3.43 tok (1.16x), MTP-8 = 3.94 (1.22x); service 17.8 s of which decode 16.2 s; 48 levels on service alone needs 7.7 s service = 2.6x decode speedup \"which the ceiling settles independently of the decay assumption\"; KV 6.5 GiB worth +20-30% calls; elasticity 0.21 (decayed from KV10's 0.40); best live-legal stack = 42.6 levels vs a 48 bar, short by 5.4; therefore the whole buy-calls family is closed.",
 "verdict": "PARTLY_WRONG",
 "recomputed": [
  {
   "quantity": "MTP-3 per-position acceptance (marginal, per draft)",
   "claimed": "76.0% / 58.3% / 45.4%",
   "recomputed": "76.0% / 58.3% / 45.4% (pos0 508,0xx.. exact: per_pos 0/1/2 = 507,993? no — values: drafts 668,483; per_pos counts sum 1,201,417 = accepted_total). Same three numbers across 30 waves range 73.4-78.6 / 54.4-63.9 / 40.9-53.4; live-geometry base 0902 = 76.0/58.5/45.6",
   "source": "offkaggle/results/20260909-193735-regime-keith/metrics_after.prom minus metrics_before.prom (vllm:spec_decode_num_accepted_tokens_per_pos_total / vllm:spec_decode_num_drafts_total); python3 script over all */metrics_after.prom",
   "matches": true
  },
  {
   "quantity": "Which wave the acceptance came from",
   "claimed": "'a wave's own metrics' (implicitly the live-geometry base with service 17.8 s)",
   "recomputed": "Exact match only for 20260909-193735-regime-keith, which is NOT live geometry: 'geometry conc 6 | 1767 s/game | public25 geometry: False', inference 16.6 s, decode 15.0 s. Immaterial numerically (acceptance is stable across waves) but the write-up mixes sources.",
   "source": "offkaggle/results/20260909-193735-regime-keith/summary.txt (geometry, VLLM lines); metrics recompute",
   "matches": false
  },
  {
   "quantity": "Per-position metric marginal vs conditional",
   "claimed": "summed as if marginal → 2.80 tokens/forward",
   "recomputed": "Marginal confirmed: sum(per_pos[0..2]) == spec_decode_num_accepted_tokens_total exactly in 30/30 waves (e.g. 620452+482109+379894 = 1,482,455 for ctx16). tokens/forward = 1 + accepted/drafts = 2.797 (0909 wave), 2.800 (0902 base). Conditional acceptance pos1|pos0 = 0.768, pos2|pos1 = 0.779, so the '0.78 decay ratio' is a constant-conditional-acceptance (geometric) model — reasonable.",
   "source": "python3 over metrics_after/before.prom of all waves",
   "matches": true
  },
  {
   "quantity": "Service 17.8 s, decode 16.2 s (decode-dominated)",
   "claimed": "17.8 s service, decode 16.2 s",
   "recomputed": "0902 base: inference 17.84 s = prefill 1.61 + decode 16.23 (decode share 0.91); TPOT 12.50 ms x 1,332 gen tok/req = 16.66 s, consistent. keith_fx 09-10: 17.95/1.62/16.32.",
   "source": "offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/metrics_*.prom (vllm:request_inference/prefill/decode_time_seconds_sum/count, request_time_per_output_token, generation_tokens_total); runner definitions offkaggle/run_regime_wave.py:1005-1100",
   "matches": true
  },
  {
   "quantity": "MTP-5 / MTP-8 tokens per forward, speedup, service, calls",
   "claimed": "3.43 / 3.94 tok; 1.16x / 1.22x; 15.6 s / 14.8 s; 57 / 60 calls",
   "recomputed": "3.43 / 3.94 tok; 1.155x / 1.216x (3%/position cost); 15.63 s / 14.93 s; 57.0 / 59.6 calls (from the table's 50-call base)",
   "source": "python3 recompute with pos=[0.760,0.583,0.454], r=0.78, cost 1.03^(k-3), decode 16.2 + prefill 1.6",
   "matches": true
  },
  {
   "quantity": "Table 'levels' column and its base anchor",
   "claimed": "base 50 calls / 38.4 levels; KV6.5 65→40.6; MTP-5 57→39.5; MTP-8 60→39.9; KV6.5+MTP-5 74→42.0; KV6.5+MTP-8 78→42.6",
   "recomputed": "Every levels entry is reproduced EXACTLY by 39.33 x (1 + 0.21 x (calls/55.8 - 1)): 50→38.47, 57→39.51, 60→39.95, 65→40.69, 74→42.02, 78→42.62. I.e. levels are anchored at the MEASURED base (55.8 calls, 39.33 lv) while the calls column starts from a MODELLED 50. Mixed anchors understate every row by the 50/55.8 gap. Same multiplicative gains applied to 55.8: KV6.5 (+30%) 72.5→41.8; MTP-5 63.6→40.5; MTP-8 67.0→41.0; KV6.5+MTP-5 82.7→43.3; KV6.5+MTP-8 87.0→44.0 (not 42.6). Shortfall to 48 = 4.0, not 5.4. No file explains where '50 calls' comes from (searched STATUS, summaries) — origin UNVERIFIED.",
   "source": "python3 recompute; offkaggle/REGIME_WAVE_STATUS.md:1501-1524",
   "matches": false
  },
  {
   "quantity": "Service needed for 48 levels on service alone",
   "claimed": "7.7 s service, decode 16.2→6.1 s, 2.6x decode speedup",
   "recomputed": "Not reproducible from either consistent base. From the table base (50, 38.4): calls x2.19 → service 8.13 s, decode 6.53 s, 2.48x. From the measured base (55.8, 39.33): calls x2.05 → service 8.68 s, decode 7.08 s, 2.29x. 7.7 only appears by mixing bases (17.8 x 50 / 114.4 = 7.78).",
   "source": "python3 recompute, elasticity 0.21, prefill 1.6 s",
   "matches": false
  },
  {
   "quantity": "'Even perfect acceptance at MTP-8 reaches only 50.8 levels' / 'ceiling settles it independently of the decay assumption'",
   "claimed": "50.8 levels; ceiling alone closes the route",
   "recomputed": "Perfect MTP-8 = 9 tok/forward = 3.21x raw speedup (2.77x with 1.03^5 cost). Levels: 48.8-50.9 (anchored 55.8) or 49.6-52.0 (anchored 50) depending on the cost assumption. Every variant is >= 48, so the physical ceiling does NOT close the route; only the measured 0.78 conditional decay does. The notebook's own 50.8 > 48 contradicts 'settles it'.",
   "source": "python3 recompute",
   "matches": false
  },
  {
   "quantity": "Elasticity 0.21 (budget 3x) and 'decay from KV10's 0.40'",
   "claimed": "KV10 0.40 at +47%; budget3x 0.21 at +204% ⇒ elasticity decays",
   "recomputed": "Linear elasticity (L/L0-1)/(C/C0-1): budget3x vs pooled base = (56/39.33-1)/(169.8/55.8-1) = 0.207 ✓. KV10 = 0.412 ONLY against its same-session base (36 lv); against the same pooled base 39.33 it is 0.198. Consistently anchored, the two points read 0.20 and 0.21 — no decay. Noise: with sd 2.34 per 25-game total, KV10 elasticity = 0.20 ± 0.14 (± 0.22 using the empirical sd 3.74 of nine live-geometry keith-family draws 36,34,40,41,31,40,41,41,35); budget3x = 0.21 ± 0.03-0.05. KV10 cannot distinguish 0.2 from 0.4.",
   "source": "offkaggle/REGIME_WAVE_STATUS.md:365-376, 940-952, 1250-1260; results.json levels for each wave; python3",
   "matches": false
  },
  {
   "quantity": "Pooled six-draw base 39.33, sd 2.34",
   "claimed": "39.33 (sd 2.34)",
   "recomputed": "Six draws = Modal keith 36, Modal retry 40, Kaggle keith-copy v4 42, Modal yield900 41, Modal yield900-cb2 40, Kaggle yield900 commit 37 → mean 39.333, sample sd 2.338. Note 3 of 6 are yield900 draws (judged null) and 2 are Kaggle commit runs; the same doc estimates 'sd ≈ 6 levels' for a 25-game total, and nine live-geometry keith-family Modal draws have sd 3.74.",
   "source": "docs/research-2026-09-08/R-loss-ledger-3.md:3-5,9; results.json of each Modal wave; python3 statistics.stdev",
   "matches": true
  },
  {
   "quantity": "Route B: KV 6.5 GiB worth +20-30% calls",
   "claimed": "+20-30% calls ≈ 40-41 levels",
   "recomputed": "Only measured KV point: KV10 (2x KV) gave 82.16/55.8 = 1.47x calls, with inference rising 17.8→27.2 s and TPOT 12.5→18.8 ms — service is NOT constant as slots rise, contradicting the formula's fixed-service assumption. Calls ∝ KV^0.56; 1.3x KV → ≈ +16% calls, not +20-30%. Corrected best stack 55.8 x 1.16 x 1.216 = 78.7 calls → 42.7 lv at 0.21, 46.0 at 0.41.",
   "source": "offkaggle/results/20260903T0553-kv10-keith/20260903-075332-regime-keith/summary.txt and metrics_*.prom; 0902 base summary.txt",
   "matches": false
  },
  {
   "quantity": "Serving regime pins MTP-3 / KV 5 GiB",
   "claimed": "num_speculative_tokens=3, 5 GiB KV, 8 seqs",
   "recomputed": "SPECULATIVE_CONFIG {method: mtp, num_speculative_tokens: 3}; TAAF_VLLM_KV_CACHE_MEMORY_BYTES 5368709120; profile kv5-bf16-mtp3-c8-cg32",
   "source": "offkaggle/modal_flashnext_serve.py:16-17,105-113,148,165,177",
   "matches": true
  }
 ],
 "errors_found": [
  "Mixed base anchors in the sizing table: the calls column starts from a modelled 50 calls/game while every levels entry is 39.33 x (1 + 0.21 x (calls/55.8 - 1)), i.e. anchored to the measured 55.8/39.33. Re-anchoring the same multiplicative gains at 55.8 gives the best stack 87 calls → 44.0 levels (not 42.6); shortfall 4.0 not 5.4. The origin of '50 calls' is not documented anywhere I could find.",
  "'7.7 s service / decode 6.1 s / 2.6x' is not reproducible: consistent arithmetic gives 8.1 s / 6.5 s / 2.5x (table base) or 8.7 s / 7.1 s / 2.3x (measured base). 7.7 arises from mixing bases.",
  "'The ceiling settles it independently of my decay assumption' is false. Perfect acceptance at MTP-8 (9 tokens/forward, 2.8-3.2x speedup) projects to 48.8-52.0 levels under the notebook's own model — above the 48 bar — and the write-up's own '50.8' already exceeds 48. What closes Route C is the MEASURED conditional acceptance (~0.77-0.78 per position), not any ceiling.",
  "'Elasticity decays 0.40 → 0.21' is a base-choice artifact: KV10's 0.40 uses its same-session base (36 lv); against the pooled base 39.33 used for the 3x arm it is 0.20. Consistently anchored, the two points (0.20, 0.21) show no decay, and KV10's value carries ± 0.14-0.22 of draw noise anyway.",
  "Route B sizing assumes service time is independent of running slots ('calls = 32,400 x slots / (110 x service)'). The one measured KV point (KV10) shows inference 17.8 → 27.2 s and calls scaling as KV^0.56; 1.3x KV → ≈ +16% calls, not +20-30%. This over-sizes Route B (in the direction that favours the closure).",
  "The acceptance figures 76.0/58.3/45.4 come from the 09-09 conc-6 / 1,767 s wave (not live geometry; inference 16.6 s), while service 17.8 s / decode 16.2 s come from the 09-02 live-geometry base. Numerically immaterial (acceptance is stable 2.69-2.96 tok/forward across 30 waves) but mislabelled as one wave."
 ],
 "confounds": [
  "Elasticity is route-specific, not a property of 'calls': keith_ctx16 bought +47% calls (81.8/game) and lost 18 levels (21 vs 39.33). The sizing assumes MTP/KV-bought calls convert like clock-bought calls; plausible for MTP (context unchanged) and directly measured for KV (KV10), but it is an assumption for the stack.",
  "The 3%-per-speculation-position cost, whether vLLM + Qwen3.8-Flash-Next (single recursive MTP layer) accepts num_speculative_tokens 5/8 at all, and whether KV 6.5 GiB boots on the Kaggle card are all unverifiable from the repo — no wave has ever run any of them.",
  "Decode speedup from tokens/forward assumes step time is roughly batch-invariant; the KV10 wave shows per-request decode slows substantially as running sequences rise (TPOT 12.5 → 18.8 ms), so the MTP speedup and the KV slot gain do not multiply cleanly when stacked.",
  "The pooled base (39.33, sd 2.34) is 3 base + 3 yield900 draws (yield900 judged null) including two Kaggle commit runs; the empirical sd of nine live-geometry keith-family Modal draws is 3.74 and the loss-ledger doc itself says 'sd ≈ 6 levels' for a 25-game total. All the sd-based margins here are optimistic by ~1.6x.",
  "The single KV10 elasticity read (0.20-0.41 depending on base, ± 0.14-0.22) is the only measurement in the +20-56% call range that Routes B/C occupy; the precise 0.21 comes from the 3x arm at +204% calls, a different range."
 ],
 "does_conclusion_survive": "Yes at the notebook's own elasticity of 0.21, but with a smaller margin and on a different argument than stated. Correctly anchored, the most optimistic live-legal stack (KV 6.5 at +30% calls, MTP-8 at 1.22x) reaches 44.0 levels, 4.0 short of 48 (≈1.7 base sd, or ≈1.1 sd using the empirical 3.74). If Route B is corrected to the measured sublinear KV scaling (+16%), the best case is 42.7. The conclusion does NOT survive if the local elasticity is really the 0.41 the notebook itself attributes to KV10: then the +30% KV / MTP-8 stack reads 48.4 (≥ 48 bar), and even the corrected +16% KV stack reads 46.0 (redraw band). Because the KV10 point cannot distinguish 0.2 from 0.4, the closure rests on the 3x arm's 0.21 being the right local slope — a defensible but untested assumption. Two supporting claims are wrong: the 'ceiling settles it independently' argument (the ceiling permits ≥ 48; the measured 0.78 decay is what closes MTP) and 'elasticity decays 0.40 → 0.21' (a base-choice artifact). Net: the family is closed by the measured MTP acceptance decay plus elasticity 0.21, not by arithmetic that is independent of assumptions, and the margin is ~4 levels, not 5.4.",
 "what_would_change_verdict": "(1) A single KV 6.5 GiB rig wave (≈$9): if it reads ≥ +25% calls with levels ≥ 45, elasticity in the local range is > 0.21 and the stack projection crosses 48. (2) A second KV10 draw against a same-session base to pin the local elasticity to ± 0.1: a read of ≥ 0.35 reopens the stack; ≤ 0.25 confirms closure. (3) An MTP-5 serving profile test showing conditional acceptance at positions 3-4 above 0.85 (vs the extrapolated 0.78) would lift the MTP term. (4) Evidence that the 3x arm's 56 levels was itself a tail draw (a second 3x draw at ≤ 48) would lower 0.21 and harden the closure.",
 "commands_run": [
  "sed -n 1470,1560p offkaggle/REGIME_WAVE_STATUS.md",
  "git show --stat 636d85d; ls offkaggle/results/*regime*; ls offkaggle/results/20260911-*/",
  "grep -E 'spec_decode' <each 09-10/09-11 wave>/metrics_after.prom and metrics_before.prom",
  "grep -n -E 'elasticity|17\\.8|16\\.2 s|38\\.4|TPOT|service' offkaggle/REGIME_WAVE_STATUS.md; sed -n 225,250p / 365,382p / 940,960p / 1190,1215p / 1236,1330p / 1385,1445p / 600,622p",
  "cat offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/summary.txt",
  "python3: parse metrics_before/after.prom for every wave; per-position acceptance = per_pos_total[i]/drafts_total; verify sum(per_pos)==accepted_total; tokens/forward = 1+accepted/drafts; conditional ratios",
  "sed -n 1005,1100p offkaggle/run_regime_wave.py (telemetry definitions)",
  "python3: inference/prefill/decode/ttft/tpot/queue/e2e means from vllm:request_*_time_seconds_sum/count deltas for 0909 base, 0902 base, kv10, budget3x, ctx16, fx",
  "grep SCORE/CADENCE/CLIENT/VLLM/geometry/ARM KNOB lines from summary.txt of 0909 base, kv10, budget3x, ctx16, fx",
  "grep -n -E 'speculative|KV|MTP' offkaggle/modal_flashnext_serve.py; cat offkaggle/results/20260909-193735-regime-keith/arm_env.json",
  "python3: levels per wave from every results.json with arm/geometry/cadence",
  "python3: itertools search for six-draw combinations summing to 39.33; sd of nine live-geometry keith-family draws (3.74)",
  "python3: elasticity under each base anchor; noise bands; reproduce table levels column via 39.33*(1+0.21*(calls/55.8-1)); service-for-48; MTP-5/8 and perfect-MTP-8 projections",
  "python3: requests_shim.jsonl of 0902 base — completion mean 1,308, prompt mean 19,722, elapsed mean 138.7 s (n=1,419)",
  "grep -rn '39\\.33' memory/*.md docs/*.md; sed -n 1,30p docs/research-2026-09-08/R-loss-ledger-3.md (six-draw composition: 36, 40, 42, 41, 40, 37)"
 ]
}
```

---

## agent-1

```json
{
 "claim": "keith_budget 3x (offkaggle/results/20260910-192514-regime-keith_budget, conc 28 / 23,760 s/game): 56 levels vs pooled base 39.33 = +7.12 sd; 169.8 calls/game (3.04x base 55.8); median e2e 128.8 s so cadence held; actions/call 3.64; ft09 WON 6/6 at score 100 in 105 actions ('first outright win in this campaign's rig history'); stuck-at-one games 14 -> 8; 5-6-level games 0 -> 3; score 6.40 -> 13.76 (2.15x); implied elasticity 0.21 vs KV10's 0.40, so elasticity 'decays' but 'budget converts'; 'we are NOT at a capability ceiling on depth'.",
 "verdict": "CONFIRMED_WITH_CORRECTIONS",
 "recomputed": [
  {
   "quantity": "levels on 25 games",
   "claimed": "56",
   "recomputed": "56 (sum of levels_completed over 25 games; dist {0:2,1:8,2:7,3:2,4:3,5:2,6:1}; states 24 gave_up + 1 won)",
   "source": "results.json games[*].levels_completed via python3",
   "matches": true
  },
  {
   "quantity": "pooled six-draw base and sd",
   "claimed": "39.33, sd 2.34",
   "recomputed": "pool = Modal 09-02 keith 36, Modal 09-03 keith_retry 40, Kaggle base commit v4 42, Modal 09-06 yield900 41, Modal 09-07 yield900-cb2 40, Kaggle yield900 commit 37 -> mean 236/6 = 39.33, sample sd sqrt(27.33/5) = 2.34. Pool does NOT contain the KV10 draw. Three of six are yield900 draws and two are Kaggle-box runs, not Modal-rig base draws.",
   "source": "docs/research-2026-09-08/R-loss-ledger-3.md:3-5; per-draw levels re-derived from each directory's results.json (36, 40, 41, 40 for the four Modal draws)",
   "matches": true
  },
  {
   "quantity": "z-score vs pooled base",
   "claimed": "+16.67 = +7.12 sd",
   "recomputed": "(56-39.33)/2.34 = 7.12. Caveat: the same ledger states the sd of a 25-game total estimated from per-game variance is ~6 levels (R-loss-ledger-3.md:10), which would give +2.8 sd; 2.34 is an n=6 empirical sd.",
   "source": "arithmetic; docs/research-2026-09-08/R-loss-ledger-3.md:10",
   "matches": true
  },
  {
   "quantity": "calls/game",
   "claimed": "169.8 (3.04x base 55.8)",
   "recomputed": "4,244 status-200 rows / 25 = 169.76 (4,290 rows incl. 24 HTTP 500 + 22 ReadTimeout). 169.76/55.8 = 3.04. But 55.8 is the single 09-02 draw; the four Modal draws in the pool average (55.8+51.72+51.52+54.2)/4 = 53.3 calls/game -> 3.18x.",
   "source": "requests_shim.jsonl (status field; error rows lack it); base dirs' requests_shim.jsonl; telemetry.json aggregate.calls_per_game = 169.76",
   "matches": true
  },
  {
   "quantity": "median client e2e",
   "claimed": "128.8 s (cadence held, gate 120-175)",
   "recomputed": "pooled median 128.8 s, mean 132.2, p90 182.4 (n=4244). Split at the container restart (t=1789078957.04): pre-restart median 139.0 s (n=3083), post-restart median 104.2 s (n=1161) -- the tail 99 min ran BELOW the 120 s gate floor.",
   "source": "requests_shim.jsonl elapsed_s, t; metrics_after.prom process_start_time_seconds",
   "matches": true
  },
  {
   "quantity": "actions/game and actions/call",
   "claimed": "618/game, 3.64/call, 4.01x base 154",
   "recomputed": "15,457/25 = 618.3; 15,457/4,244 = 3.64. Base 154.2 is the 09-02 draw only; the four Modal pool draws average (154.2+119.2+102.2+124.0)/4 = 124.9 -> 4.95x.",
   "source": "results.json actions; base dirs' results.json",
   "matches": true
  },
  {
   "quantity": "score/game",
   "claimed": "13.76 vs base 6.40 = 2.15x",
   "recomputed": "343.99/25 = 13.76. 6.40 is the 09-02 draw (the lowest of the four Modal pool draws: 6.40, 8.62, 8.46, 7.90, mean 7.85) -> 1.75x vs the Modal-pool mean. The table mixes the pooled base for levels with the single lowest draw for score/actions/distribution.",
   "source": "results.json score; base dirs' results.json",
   "matches": false
  },
  {
   "quantity": "stuck-at-one games 14 -> 8; 5-6-level games 0 -> 3; zero-level 2 vs 2",
   "claimed": "14->8, 0->3, 2 vs 2",
   "recomputed": "This wave: one-level 8, >=5 levels 3, zero-level 2 (confirmed). Base side is the 09-02 draw only: one-level counts across the four Modal pool draws are 14/11/15/16, >=5-level counts are 0/1/1/1 (not 0), zero-level 2/4/1/1.",
   "source": "results.json level distributions for 20260902T2121-keith, 20260903T1014-keith_retry, 20260906T2251-keith_yield900, 20260907T0702-keith_yield900-cb2",
   "matches": false
  },
  {
   "quantity": "ft09 WON 6/6, score 100, 105 actions, 'first outright win in this campaign's rig history'",
   "claimed": "first outright win on the rig",
   "recomputed": "ft09 6/6 won, 105 actions, score 100.0, wallclock 9,905 s -- CONFIRMED for this wave. But ft09 ALREADY won 6/6 (state 'won', 108 actions, score 100.0, wallclock 7,686 s, inside a 1x clock) in the KV10 wave 20260903T0553; the notebook itself records it at REGIME_WAVE_STATUS.md:369 ('ft09 WON 6/6 = 100 pts'). ft09 also reached 5/6 in yield900-cb2 and carry75. 'First outright win' is FALSE. In this wave ft09's 5th clear landed at 8,013-8,375 s and the win after 9,367 s -- both past 7,920 s.",
   "source": "results.json (budget and kv10 dirs); artifacts/ft09-0d8bbf25_p0_events.jsonl; transcripts/ft09-0d8bbf25_p0.txt headers step 16 = 21:38:51, step 18 = 22:01:25; REGIME_WAVE_STATUS.md:369",
   "matches": false
  },
  {
   "quantity": "(b) levels cleared by 7,920 s within this wave",
   "claimed": "(implied) levels arrive late; base at 1x clock consistent",
   "recomputed": "55 of 56 clears dated by joining transcript headers to level_completed events (56th = ft09 win, after 9,367 s). 41 clears by 7,920 s (identical using the step's own header as lower bound or the next header as upper bound) = +0.71 sd vs 39.33; 15 clears after 7,920 s (14 dated + ft09 win). Clears per 3,960 s bucket: 35, 6, 6, 5, 0, 3. Calls/game in the first 7,920 s = 53.0 (min 51, max 55) vs base 55.8. So the base waves are NOT under-performing; the extra ~15-17 levels came from the extra clock.",
   "source": "scratchpad/clears.py over transcripts/*.txt + artifacts/*_events.jsonl + results.json; requests_shim.jsonl",
   "matches": true
  },
  {
   "quantity": "(c) implied elasticity",
   "claimed": "0.21 vs KV10 0.40 -> 'decays'",
   "recomputed": "Linear-arc form (L/L0-1)/(C/C0-1): 3x arm vs pooled = 0.4238/2.043 = 0.207 (confirmed); vs pooled calls 53.3 = 0.194; within-wave (41 lv @ 53.0 calls -> 56 @ 169.76) = 0.366/2.203 = 0.166. KV10 (43 lv, 82.16 calls; verified from its results.json) vs its own-session base 36/55.8 = 0.1944/0.4724 = 0.41; vs the SAME pooled base 39.33 = 0.0933/0.4724 = 0.198. Log-log form ln(L/L0)/ln(C/C0): 3x arm 0.318 (pooled) / 0.268 (within-wave); KV10 0.46 (vs 36) / 0.23 (vs pooled). On a common base the two points read ~0.20 and ~0.21: no decay is demonstrated. KV10's +7 (or +3.67 vs pooled) is 1-2 sd of a 25-game total, unresolvable.",
   "source": "arithmetic on results.json / requests_shim.jsonl of both waves; REGIME_WAVE_STATUS.md:942-950 for the KV10 derivation",
   "matches": false
  },
  {
   "quantity": "(d) container restart",
   "claimed": "request_success_total 524 -> 1182 vs 4,290 shim calls; 46 post errors; largest gap 215 s at +297 min",
   "recomputed": "before-snapshot process_start 1789057121.84 (66 min BEFORE the wave; the 524 requests and 0 preemptions are the 3-game smoke's), after-snapshot process_start 1789078957.04 = +297.3 min into the wave, 235 s after the before-container's 6 h mark (+293.4 min) -> the 6 h lifetime cap, not a crash. Last request before restart +297.0 min, first after +300.6 min, gap 215 s. Errors: 24 HTTP 500 at +294.8..296.7 min (one per game, 22 games; 101 requests overlapped the restart, 77 still returned 200) + 22 ReadTimeouts at +393.8..395.9 min (wave shutdown) = 46. Post-restart span 5,918 s: 1,161 ok calls vs ~1,005 expected at the pre-restart mean cadence 141.3 s -> ~+156 surplus calls (+3.7% of 4,244), offset by ~215 s outage (~-36 calls). 3 of 56 clears landed after the restart (cd82 L2 20,676 s, cd82 L3 22,246 s, g50t L2 20,825 s). Net bias: slightly UP, order +0.5 level; negligible against +16.67. The summary's VLLM row (661 requests, e2e 181.9, queue 162.9) is an artifact of subtracting a smoke-era snapshot from a fresh-container snapshot; the after-container alone served 1,185 requests at e2e 106.2 s / queue 90.9 s / inference 15.2 s with 266 preemptions in 99 min.",
   "source": "metrics_before.prom, metrics_after.prom (process_start_time_seconds, vllm:* counters); requests_shim.jsonl t/elapsed_s/status; clears.py output",
   "matches": true
  },
  {
   "quantity": "pre-registered band",
   "claimed": "band PARTIAL (46-59); headline 'BUDGET CONVERTS'",
   "recomputed": "Pre-registration (REGIME_WAVE_STATUS.md:1135-1143): >=60 = 'Budget converts', 46-59 = 'partial conversion (elasticity ~0.1-0.3), triage worth +10-30%, not a step on its own', <=45 = does not convert. 56 falls in 46-59. The band assignment in the write-up is correct; the headline borrows the >=60 band's phrase.",
   "source": "REGIME_WAVE_STATUS.md:1135-1143, 1242-1250",
   "matches": true
  }
 ],
 "errors_found": [
  "'ft09 WON OUTRIGHT ... the first outright win in this campaign's rig history' is false: ft09 won 6/6 (score 100, 108 actions, 7,686 s, inside a 1x clock) in the KV10 wave 20260903T0553, recorded at REGIME_WAVE_STATUS.md:369. ft09 is a game the base regime already wins occasionally and reaches 4-5/6 in most draws; in this wave its 5th clear and the win both came after 7,920 s.",
  "The comparison table mixes bases: levels use the pooled six-draw 39.33, but calls (55.8), actions (154), score (6.40), zero-level (2), stuck-at-one (14) and 5-6-level (0) all come from the single 09-02 draw, which is the lowest-scoring of the four Modal pool draws. Against the Modal pool means (53.3 calls, 124.9 actions, 7.85 score, 5+-level games 0/1/1/1) the ratios are 3.18x calls, 4.95x actions, 1.75x score (not 2.15x), and '5-6-level games 0 -> 3' should read '~0.75 -> 3'.",
  "'Elasticity decays 0.40 -> 0.21' is not supported: KV10's 0.40 is computed against its own-session base of 36 while the 3x arm's 0.21 is against the pooled 39.33. On the common pooled base KV10 reads 0.198 and the 3x arm 0.207 (log-log 0.23 vs 0.32). KV10's increment (+3.67 to +7 levels, one draw each side) is within 1-2 sd of a 25-game total and cannot resolve the curve's shape.",
  "Cadence gate 'held' only in the pooled median: after the 6 h container-lifetime restart at +297 min the last 99 min ran at median e2e 104.2 s (< the 120 s gate floor) and delivered ~156 surplus calls (+3.7%); 3 of the 56 levels arrived in that window. Small, but the write-up says nothing about the restart.",
  "The summary.txt VLLM row (661 requests, e2e 181.9 s, queue 162.9 s, 266 preemptions attributed to the whole wave) is unreadable: it is after-container counters minus a before-snapshot taken from the previous container that had served the 3-game smoke. The after-container alone: 1,185 requests, e2e 106.2 s, queue 90.9 s, 266 preemptions in 99 min.",
  "'+7.12 sd' rests on the n=6 empirical sd 2.34; the same loss-ledger doc (R-loss-ledger-3.md:10) puts the sd of a 25-game total at ~6 levels from per-game variance, which would make it +2.8 sd. The result is significant either way, but the headline sd overstates precision.",
  "Headline 'BUDGET CONVERTS' uses the pre-registration's >=60 band language; 56 lands in the 46-59 band the pre-registration itself labelled 'partial conversion ... not a step on its own'. The write-up assigns the band correctly in the PRIMARY line but the headline and the 'not at a capability ceiling' framing follow the >=60 script."
 ],
 "confounds": [
  "Base pool composition: 3 of the 6 pooled draws are yield900 (a different knob, judged null) and 2 are Kaggle-box commit runs, not Modal-rig base draws; the Modal base draws proper are 36 and 40. The 3x arm ran on the 60 s yield regime, so the cleanest comparator is the within-wave 41-by-7,920 s count, which lands at +0.71 sd of the pool anyway.",
  "Container restart at +297 min (6 h Modal lifetime cap, container had been up 66 min for the smoke before the wave): 215 s outage, 24 HTTP 500s, faster tail cadence (104 s median vs 139 s) -> ~+156 surplus calls; direction of bias is up but bounded at roughly +0.5 level.",
  "Not first-in-boot: the container had served 524 smoke requests before the wave; the earlier A/B order-control memory reports a +0.25 order effect (docs/EXPERIMENT-2026-09-02-ab-instrument-control.md). Sub-sd.",
  "Single draw: the +16.67 is one wave; the 3x-vs-1x within-wave delta (+15 over the same games, same boot) has no independent replication. The pre-registration did not require a redraw for the 46-59 band.",
  "The model does not see the clock (prompt only mentions in-game timer bars; no remaining-time variable), so early-phase play at 3x is distributionally identical to 1x play; this is what makes the within-wave 41-by-7,920 s comparison valid.",
  "ft09 alone contributes 6 levels and 100 score points (29% of the wave's score); it has won on a 1x clock before, so it is not evidence that extra clock unlocks it, only that this draw happened to do so late."
 ],
 "does_conclusion_survive": "Yes, in its narrow form; no, in its strong form. The primary numbers (56 levels, 169.76 calls/game, 3.04x, median e2e 128.8 s, 3.64 actions/call, 618 actions/game, 13.76 score/game, elasticity 0.21 on the pooled base) all reproduce from results.json and requests_shim.jsonl. The decisive check is (b): dating every clear from the transcripts shows 41 of 56 levels had arrived by 7,920 s (inside the base band at +0.71 sd) and 15 arrived in the extra 2x of clock. So the base waves are consistent and the extra clock genuinely bought ~15 levels on the same games in the same boot; the restart adds at most ~+0.5 level of upward bias. 'Budget converts sublinearly at 3x' therefore survives and is not an artifact. What does NOT survive: (1) 'first outright win in rig history' (ft09 won in KV10 on 09-03 within 7,686 s); (2) 'elasticity decays 0.40 -> 0.21' (same-base values are 0.20 and 0.21; KV10 cannot resolve curvature); (3) the 2.15x score and the 0 -> 3 five-level-game contrasts, which use the lowest single base draw rather than the pool; (4) the headline 'BUDGET CONVERTS' and 'not at a capability ceiling' framing, which is the pre-registration's >=60 band language applied to a 56 that the pre-registration itself placed in the 'partial, not a step on its own' band. The defensible statement is: tripling calls at (approximately) held cadence bought +37-42% levels (arc elasticity 0.17-0.21, log-log 0.27-0.32), which is the same exchange rate KV10 showed on a common base, and which the write-up's own triage re-simulation converts to only +12% inside the fixed live budget.",
 "what_would_change_verdict": "To REFUTED: if a re-dating of the clears showed most of the 56 levels landed before 7,920 s (it showed 41), or if the pooled base were found to be materially above 41 at the 1x clock. To fully CONFIRMED: if ft09 had never won on the rig before (it did, in 20260903T0553), if KV10's elasticity were recomputed on the pooled base and still read ~0.4 (it reads 0.20), and if the score/actions/distribution rows of the table were restated against the pool rather than the 09-02 draw. A second clean 3x draw (or a 2x draw) with no container restart would settle the functional form the write-up asserts; one point at 3x cannot.",
 "commands_run": [
  "ls -la offkaggle/results/20260910-192514-regime-keith_budget/; ls offkaggle/results/ | grep regime",
  "cat summary.txt, arm_env.json, git_status.txt (budget wave)",
  "sed -n 1200,1310p offkaggle/REGIME_WAVE_STATUS.md; sed -n 1111,1200p; sed -n 596,640p; sed -n 540,596p; sed -n 938,962p; grep -n for keith_budget/pooled/39.33/KV10",
  "git show --stat 4a7f22f",
  "python3: results.json -> levels, actions, score, states, level distribution, per-game rows, wallclock",
  "python3: requests_shim.jsonl -> status/error counts, calls/game (all vs status 200), e2e mean/median/p90, prompt/completion tokens, start-time gaps, error timestamps, pre/post-restart e2e, calls in first 7,920 s, post-restart call surplus, actions/call",
  "grep vllm:* counters and process_start_time_seconds from metrics_before.prom / metrics_after.prom; python3 arithmetic for after-container e2e/queue/inference",
  "python3 scratchpad/clears.py: parse '--- analysis_step=N | action=M | HH:MM:SS' headers in transcripts/*.txt (day-rollover handled), join to level_completed rows in artifacts/*_events.jsonl by analysis_step, date every clear, count clears <= 7,920 s and per 3,960 s bucket",
  "python3: ft09 events tail + grep analysis_step=16..21 headers in transcripts/ft09-0d8bbf25_p0.txt",
  "python3: results.json + requests_shim.jsonl for 20260902T2121-keith, 20260903T0553-kv10-keith, 20260903T1014-keith_retry, 20260906T1634-wall-keith, 20260906T2251-keith_yield900, 20260907T0702-keith_yield900-cb2, 20260908T0752-keith_probe, 20260908T2110-keith_carry75, 20260909-193735-regime-keith, 20260910-081906-regime-keith_fx (nested dirs) -> levels, calls/game, score/game, distributions; ft09 outcome across all draws",
  "grep -n prompts/lp85-305b61c3_p0.log for remaining/time budget/deadline (clock not exposed to the model)",
  "sed -n 1,30p docs/research-2026-09-08/R-loss-ledger-3.md (six-draw pool composition); python3 telemetry.json aggregate cross-check; grep run_regime_wave.py render_summary/calls_per_game"
 ]
}
```

---

## agent-2

```json
{
 "claim": "keith_nr16 (ctx16 window 16,384 + graft_noreason stripping reasoning from retained history) is DEAD at 20 levels = -8.26 sd; calls/game 59.2, actions/call 0.77, completion tokens/call 1,632 vs ctx16 1,118 (+46%), median prompt 9,435 vs 9,582 so 'the trimmer always fills the window' and dropping reasoning 'can never buy slots'; retained reasoning is load-bearing; e2e 90 -> 133 s; plus the comparison table's 'base (32,768)' row.",
 "verdict": "CONFIRMED_WITH_CORRECTIONS",
 "recomputed": [
  {
   "quantity": "nr16 levels (25 games) and sd vs pooled base",
   "claimed": "20 levels, -8.26 sd",
   "recomputed": "20 levels (sum of levels_completed), (20-39.33)/2.34 = -8.26; actions 1,133; 9 zero-level games; 25/25 gave_up",
   "source": "offkaggle/results/20260911-143210-regime-keith_nr16/results.json (python3 recompute.py)",
   "matches": true
  },
  {
   "quantity": "nr16 calls/game and actions/call",
   "claimed": "59.2 calls/game, 0.77 actions/call",
   "recomputed": "1,479 status-200 shim rows / 25 = 59.2; 1,133 / 1,479 = 0.77 (18 non-200 rows excluded)",
   "source": "offkaggle/results/20260911-143210-regime-keith_nr16/requests_shim.jsonl",
   "matches": true
  },
  {
   "quantity": "completion tokens/call nr16 vs ctx16",
   "claimed": "1,632 vs 1,118 (+46%)",
   "recomputed": "means 1,632 vs 1,118 (+46%); medians 1,201 vs 748 (+61%); 5%-trimmed means 1,460 vs 980 (+49%); per-game means higher in 25/25 games, per-game ratio median 1.48 (min 1.00 bp35/cd82, max 2.40 sc25); top-5% of calls carry 19.1% (nr16) vs 20.5% (ctx16) of all completion tokens -> NOT outlier-driven",
   "source": "requests_shim.jsonl of both waves, paired per game_id",
   "matches": true
  },
  {
   "quantity": "median prompt tokens nr16 vs ctx16",
   "claimed": "9,435 vs 9,582",
   "recomputed": "9,435 vs 9,582 (means 9,355 vs 9,510; p90 11,235 vs 11,094; max 15,818 vs 15,289)",
   "source": "requests_shim.jsonl of both waves",
   "matches": true
  },
  {
   "quantity": "e2e per call nr16 vs ctx16",
   "claimed": "90 -> 133 s",
   "recomputed": "client medians 90.1 -> 132.7 s (means 95.9 -> 132.6); vLLM /metrics deltas: e2e mean 95.2 -> 132.0 s, queue 76.7 -> 106.9 s, inference 18.3 -> 24.8 s per request",
   "source": "requests_shim.jsonl elapsed_s; metrics_before/after.prom deltas (vllm:e2e_request_latency_seconds_sum/count, request_queue_time, request_inference_time)",
   "matches": true
  },
  {
   "quantity": "graft actually removed reasoning from retained turns (served prompt)",
   "claimed": "current turn keeps reasoning, past turns lose it",
   "recomputed": "logs/nr16_20260911.log:3 'graft_noreason: noreason: _persistent_history_messages: OK'; prompt snapshots: nr16 25/25 files have 0 [REASONING] blocks under 83 [ASSISTANT] blocks, ctx16 has [REASONING] under every retained [ASSISTANT] (68/68, 2.72/snapshot). Served-prompt test on consecutive calls where n_messages grew by 2-3: nr16 delta_prompt = 1,442 + 0.11 x prev_completion_tokens (n=397, R2 0.01); ctx16 = 1,087 + 1.10 x prev_completion (n=407, R2 0.40); 09-09 base = 676 + 1.04 x (n=784, R2 0.73). Slope ~1 = prior reasoning is in the next prompt; slope ~0 in nr16 = it is gone from what the model sees",
   "source": "prompts/*.log grep counts; requests_shim.jsonl regression; logs/nr16_20260911.log",
   "matches": true
  },
  {
   "quantity": "'the trimmer always fills the window' / real prompt pinned near the budget",
   "claimed": "sequence length is set by the context window, full stop",
   "recomputed": "Real prompt tokens are NOT pinned near 16,384: nr16 median 9,435 / p90 11,235, ctx16 9,582 / 11,094; calls over the 15,360 budget: 0.1% (nr16), 0.0% (ctx16); calls over 12,000: 4.3% / 2.8%. What is filled is the trimmer's ESTIMATE: tool_agent.py:462-466 _estimate_tokens = (len(json.dumps(payload))+2)//3 over the whole request incl. tools schema and the base64 grid image (png_bytes 1,648 per logs/budget3x_20260910.log geometry line -> ~2.2k base64 chars -> ~730 estimated tokens vs 64 real vision tokens); budget = 16,384 - 512 reply reserve - 512 safety = 15,360 (tool_agent.py:945-948, matches harness-reported context_budget_tokens 15360). Real tokens sit at ~58% of the window in both arms",
   "source": "scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py:149,462-466,945-948,1672-1690; requests_shim.jsonl prompt_tokens distribution",
   "matches": false
  },
  {
   "quantity": "retained turn depth bought by the graft",
   "claimed": "pre-reg: ~12.5 turns at 16k (same as base ~12.6); result: 'bought more turns'",
   "recomputed": "Retained [ASSISTANT] blocks per latest-call snapshot: nr16 3.32 vs ctx16 2.72 (+0.6, +22%); base conc-28 (09-10 3x dir) 8.56, base conc-6 (09-09) 10.58. From n_messages: (n-2)/3 turns-in-prompt = nr16 3.15, ctx16 2.46, conc-28 stock base 6.83 (20260902-232148) / 6.78 (yield900-cb2). A reasoning-free turn costs ~1,440 real prompt tokens (nr16 regression intercept), not ~917; a with-reasoning ctx16 turn costs ~1,087 + 1.1 x completion (~1,630 at the median completion 497), not ~2,217",
   "source": "prompts/*.log per arm (grep -c '^\\[ASSISTANT\\]'); requests_shim.jsonl n_messages; regression above",
   "matches": false
  },
  {
   "quantity": "GPU decode budget spent (why calls fell)",
   "claimed": "fatter calls pushed e2e up and cut calls/game 81.8 -> 59.2",
   "recomputed": "vLLM deltas: request_inference_time_sum 36,771 s (nr16) vs 37,655 s (ctx16); decode_time_sum 35,355 vs 35,716 s; generation_tokens 2,416,939 vs 2,295,155 (+5%); e2e_sum 195,288 vs 195,705 s over 1,480 vs 2,056 requests; kv_cache_usage 98% both. Same total GPU decode seconds, spent on 28% fewer calls that each generate 46% more tokens (gen tok/req 1,633 vs 1,116). Causal chain confirmed",
   "source": "metrics_before.prom / metrics_after.prom deltas, both waves",
   "matches": true
  },
  {
   "quantity": "'base (32,768)' table row",
   "claimed": "median prompt 21,738 / completion 1,297 / e2e 26.6 s / calls 56.8 / actions/call 2.74 / levels 39.33",
   "recomputed": "Every telemetry number matches offkaggle/results/20260909-193735-regime-keith EXACTLY (median prompt 21,738; completion mean 1,297; e2e median 26.6; 1,419 calls/25 = 56.8; 3,891/1,419 = 2.74) — the 09-09 CADENCE arm: results.json concurrency_override=True, summary geometry 'conc 6 | 1767 s/game', public25 geometry False, and it scored 34 levels (-2.28 sd), not 39.33. A real conc-28/7,920 s stock base (20260902-232148-regime-keith, 36 lv): median prompt 21,421, completion 1,330, e2e median 144.8 s, 55.8 calls/game, 2.76 actions/call. The pooled 'six-draw' composition could not be reconstructed from disk: no six on-disk conc-28/7,920 s/32k draws sum to 236 levels unless the KV10 arm (43 lv, different KV profile) is included",
   "source": "offkaggle/results/20260909-193735-regime-keith/{results.json,summary.txt,arm_env.json}; offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/{results.json,requests_shim.jsonl}; brute-force over all regime dirs",
   "matches": false
  },
  {
   "quantity": "reasoning chars per call",
   "claimed": "(summary) mean 4143 median 3039",
   "recomputed": "nr16 mean 4,160 median 3,052 (status-200 only, n=1,479) vs ctx16 2,609 / 1,439 -> +59% mean; content_chars 146 vs 204; tool_calls 1.0 both",
   "source": "requests_shim.jsonl",
   "matches": true
  }
 ],
 "errors_found": [
  "MISLABELED BASE ROW (known error, CONFIRMED): the 'base (32,768)' row is the 09-09 conc-6 / 1,767 s cadence arm (34 levels, -2.28 sd) with 39.33 levels pasted in; its e2e 26.6 s and actions/call 2.74 are conc-6 numbers. The conc-28 stock base reads e2e ~145 s (144.8 median, 20260902-232148), calls 55.8, actions/call 2.76, median prompt 21,421. The sentence 'e2e 90 -> 133 s' is still correct because it compares ctx16 to nr16, but any reader comparing nr16's 133 s to the table's 26.6 s base would conclude a 5x slowdown vs base when nr16 is actually FASTER per call than the conc-28 base (133 vs ~145 s).",
  "OVER-CLAIM 'the trimmer always fills the window ... sequence length is set by the context window, full stop': real prompt tokens are at ~58% of the 16,384 window in both arms (median 9.4-9.6k, p90 ~11.1-11.2k, 0.0-0.1% of calls over the 15,360 budget). The pinning is in the stock trimmer's chars/3 estimate over the JSON payload including the tools schema and base64 image (tool_agent.py:462-466), which over-counts real tokens by ~1.6x. Correct statement: 'under the stock trimmer, real sequence length is set by the estimator's view of the window, so dropping reasoning refills the estimated budget with more turns'. The 'can never buy slots' conclusion holds for the shipped bytes but is not a property of the window; ctx16 independently shows shorter real sequences do not help anyway.",
  "PREMISE ARITHMETIC WRONG and 'bought more turns' is tiny: the graft restored +0.6 retained assistant turns per snapshot (3.32 vs 2.72; n_messages 11.5 vs 9.4), not the pre-registered ~12.5 vs ~5.2. Measured per-turn cost without reasoning is ~1,440 real tokens (not ~917) and with reasoning ~1,087 + 1.1 x completion (~1,630 at median), so reasoning was ~1/3 of a turn, not ~60%. The write-up says the result was 'exactly the failure mode I had already written down for the graft alone', but the pre-registration's turn-count projection was itself off before any refill effect.",
  "UNSUPPORTED INFERENCE 'this is the third pre-registered outcome: actions/call did not recover, so turn depth was NOT the cause of ctx16's collapse': branch 3 presupposed that turns were restored. They were not (+22%), so the turn-depth hypothesis was never actually exercised by this arm; the arm is uninformative about it rather than refuting it. What the arm does show is that stripping reasoning at 16k makes calls fatter (+46% completion, +59% reasoning chars) and slower.",
  "OVER-REACH 'independently corroborates OpenAI's retained-reasoning result (13.3 -> 38.3%)': the comparison is between two catastrophic 16k arms (21 vs 20 levels, indistinguishable at sd 2.34 for a 25-game total); it does not measure the value of retained reasoning at the 32k base window, and levels did not move at all between the two arms. 'Retained reasoning is load-bearing' rests on actions/call 0.77 vs 1.35 and the completion inflation, not on levels. 'It re-derives what it can no longer see' is an interpretation; nothing in the files tests it.",
  "VOID RULE SILENTLY WAIVED: the notebook's standing per-run VOID rule (STATUS line 470/622: request errors > 0 or preemptions > 0) flags all 25 runs VOID in both nr16 (preemptions 1,652, 18 request errors) and ctx16 (1,908; 24), per each summary.txt WALL line. The same is true of the 09-09 base (59), so the rule is routinely ignored on Modal, but the write-up does not say so."
 ],
 "confounds": [
  "Both arms ran in the same day on the same Modal container lineage back-to-back (ctx16 11:16-13:30 UTC, nr16 14:32-16:45); the A/B order-effect control (docs/EXPERIMENT-2026-09-02-ab-instrument-control.md) detected +0.25 lv/game order effects — negligible against a -8 sd read but relevant to the 20-vs-21 tie.",
  "Preemptions (1,652 / 1,908) and request errors (18 / 24) in both arms violate the notebook's own VOID rule; KV cache usage 98% at wave end in both, so both arms were KV-bound with preemption churn, which may itself inflate e2e independent of completion length.",
  "The +46% completion inflation coexists with a different retained-history format: stripped assistant messages carry only ~146 chars of content plus a tool call, so the model's prior turns look near-empty. Whether the extra tokens are 're-derivation' or a prompt-format effect (e.g. the model treating history as unfinished) is not separable from these files.",
  "The pooled six-draw base (39.33, sd 2.34) could not be reconstructed from on-disk waves (no six conc-28/7,920 s stock draws sum to 236 without the KV10 arm); the -8.26 sd arithmetic is right given the stated pool, but the pool's provenance was not verifiable here.",
  "nr16 yielded 75.6% of turns vs 58.3% (summary TURNS line) because e2e 133 s exceeds the 60 s yield; turns/game 57.9 vs 78.6. Actions/call therefore partly reflects the cadence collapse, not only the model's willingness to act."
 ],
 "does_conclusion_survive": "The operational conclusion survives: keith_nr16 is dead (20 levels, -8.26 sd; both mechanism gates failed), it is not better than ctx16 on any metric (calls 59.2 vs 81.8, actions/call 0.77 vs 1.35, levels 20 vs 21), and the causal chain 'strip retained reasoning -> +46% completion tokens per call (25/25 games, not outlier-driven) -> +36% per-request inference time on a fixed GPU decode budget -> 28% fewer calls' is confirmed from the shim and the vLLM counters. The graft demonstrably worked at the served-prompt level (regression slope 0.11 vs 1.10). What does NOT survive: (1) the 'base (32,768)' row is the conc-6 cadence arm and must be replaced by conc-28 numbers (e2e ~145 s, prompt ~21.4k, 36-41 levels per draw); (2) 'the trimmer always fills the window / sequence length is set by the window, full stop' is true only in the stock estimator's units — real prompts sit at ~58% of the window in both arms; (3) 'turn depth was NOT the cause of ctx16's collapse' is unsupported because the arm restored only +0.6 turns, so the pre-registered branch 3 never triggered in the sense intended; (4) 'corroborates OpenAI's retained-reasoning result' over-reaches from two indistinguishable catastrophic 16k arms. The 'buy calls family closed' bottom line is unaffected by these corrections.",
 "what_would_change_verdict": "A wave at the 32,768 base window with graft_noreason (no window change) that reads inside the base band (37-42 levels) with calls/game unchanged would show retained reasoning is NOT load-bearing at the real operating point, reversing the 'load-bearing' sentence. A trimmer that estimates tokens correctly (or a window sized so the real prompt lands at ~15k) paired with the graft would test the 'can never buy slots' sentence directly: if calls/game rose above 81.8 with actions/call >= 2.0 the composition route would reopen. Finding the six pooled draws' provenance showing a pool mean materially different from 39.33 would change the sd arithmetic but not the dead verdict (20 levels is below every stock draw on disk, min 31).",
 "commands_run": [
  "sed -n 1540,1621p offkaggle/REGIME_WAVE_STATUS.md; sed -n 1440,1540p ...; sed -n 596,626p ...",
  "cat offkaggle/results/{20260911-143210-regime-keith_nr16,20260911-111640-regime-keith_ctx16,20260909-193735-regime-keith}/summary.txt and arm_env.json",
  "git show --stat=200 72889ea; git show --stat a39b449",
  "cat submission/_throughput_v1/graft_noreason.py; sed -n 270,330p offkaggle/run_regime_wave.py",
  "sed -n 455,480p / 935,960p / 1590,1700p scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py; grep -n '_PERSISTENT_HISTORY_ASSISTANT_TURNS\\|_REQUEST_SAFETY_MARGIN_TOKENS' (same file)",
  "python3 scratchpad/recompute.py — levels/actions/calls/prompt/completion/e2e/n_messages/finish distributions from results.json + requests_shim.jsonl for nr16, ctx16, 09-09 conc-6 keith",
  "python3 paired per-game completion comparison (25 games), trimmed means, quantiles, top-5% share, n_messages->median prompt tables, reasoning/content chars",
  "grep -c '^\\[ASSISTANT\\]' / '^\\[REASONING\\]' over prompts/*.log for nr16, ctx16, 09-09 keith, 09-10 keith",
  "python3 regression of delta prompt_tokens on previous completion_tokens for consecutive calls (n_messages +2/+3) per arm",
  "python3 Prometheus delta extraction from metrics_before/after.prom (e2e, queue, inference, prefill, decode, generation_tokens, prompt_tokens, preemptions) + kv_cache_usage_perc, num_requests_running/waiting",
  "grep -n noreason logs/nr16_20260911.log; head -30 logs/budget3x_20260910.log; python3 inspection of offkaggle/results/20260910-132152-regime-keith/results.json (status running, 23,760 s geometry)",
  "python3 tabulation of every offkaggle/results/**/*-regime-* wave (levels, geometry, calls/game, actions/call, median prompt, completion, e2e) and brute-force of six-draw subsets summing to 236",
  "grep for 39.33 / six-draw / VOID rule / order effect across STATUS, docs and memory files"
 ]
}
```

---

## agent-3

```json
{
 "claim": "docs/research-2026-09-10/R-triage-family-bounded.md: early_predictability.py gives AUC 0.771/0.884/0.944 (20/40/60 actions) for final levels >= 3 from 'levels cleared so far' on 247 runs, passing the budget-controller prerequisite at the top of the published range; triage_simulation.py bounds an external budget controller at ~+11% (keep top 60% at 60 actions: +49% uplift, +0.99 on base 7.85) with a warning that the aggressive cell (+7.70) linearly extrapolates elasticity 0.4 to +289%; the family is below the step bar.",
 "verdict": "PARTLY_WRONG",
 "recomputed": [
  {
   "quantity": "corpus size",
   "claimed": "247 recorded runs",
   "recomputed": "247 runs = all offkaggle/results/*/results.json with mtime < 2026-09-10 09:10 (30 wall-kill + 100 + 25 probe + 12 carry-kill + 25 carry75 + 30 ws/wsd-kill + 25 base 09-09). Today the committed script reads 375 runs (glob is not pinned) and prints AUC 0.785/0.895/0.933, base rate 0.12.",
   "source": "scratchpad corpus.py load(cutoff) over offkaggle/results/**/results.json + artifacts/*_events.jsonl; .venv/bin/python src/modelaxis/early_predictability.py",
   "matches": true
  },
  {
   "quantity": "AUC final>=3, levels_cleared_by_k, K=20/40/60",
   "claimed": "0.771 / 0.884 / 0.944",
   "recomputed": "0.771 / 0.884 / 0.944 on the frozen 247-run corpus (n=244 after the <3-events filter); K=80: 0.977",
   "source": "features_at()/auc() from src/modelaxis/early_predictability.py:62-92 run on frozen corpus",
   "matches": true
  },
  {
   "quantity": "AUC final>=3, actions_on_current_level, K=20/40/60",
   "claimed": "0.766 / 0.816 / 0.825",
   "recomputed": "0.766 / 0.816 / 0.825",
   "source": "same run",
   "matches": true
  },
  {
   "quantity": "AUC final>=2 at 40 / 80; base rates",
   "claimed": "0.680 / 0.869; base rate 0.14 (>=3), 0.34 (>=2)",
   "recomputed": "0.680 / 0.869; base rates 0.135 and 0.340",
   "source": "same run",
   "matches": true
  },
  {
   "quantity": "P(final>=3 | levels cleared by K=60) — leakage check",
   "claimed": "(doc: 'partly self-fulfilling')",
   "recomputed": "levels_at_60 = 0: n=78, P=0.00; =1: n=128, P=0.031; =2: n=28, P=0.571; =3: n=10, P=1.0; =4: n=3, P=1.0. At K=40: 0.00/0.128/0.857/1.0. The target is almost fully determined by the feature.",
   "source": "scratchpad strata computation over frozen corpus events (level, action_num fields)",
   "matches": false
  },
  {
   "quantity": "AUC for the non-leaky target 'clears >=1 MORE level after K' (K=20/40/60)",
   "claimed": "not computed in the doc",
   "recomputed": "levels_cleared_by_k: 0.367 / 0.419 / 0.465 (anti-predictive); actions_on_current_level: 0.369 / 0.405 / 0.477; game_overs: 0.506 / 0.485 / 0.479; distinct_actions: 0.456 / 0.502 / 0.522. Within stratum levels_at_60=1: actions_on_current_level -> incremental AUC 0.651 (the only above-chance signal; n=128). Base rates 0.660 / 0.512 / 0.369.",
   "source": "scratchpad incremental-target computation",
   "matches": false
  },
  {
   "quantity": "AUC excluding the 6-game wall kill-sets (25-game waves only, n=175)",
   "claimed": "n/a",
   "recomputed": "0.772 / 0.874 / 0.937 — kill-set composition is not what drives the AUC; but the 247 runs are only 25 distinct games (cd82/dc22/lf52 have 31 runs each with zero >=3 outcomes; only 5 games have a majority of runs >=3). Per-game-mean AUC (n=25): 0.80/0.92/0.98.",
   "source": "scratchpad",
   "matches": true
  },
  {
   "quantity": "baseline score 7.85 / 100 and levels/game",
   "claimed": "base 7.85",
   "recomputed": "7.85 using game_score() = 100*L(L+1)/(N(N+1)) (the completion-share CEILING, parity efficiency); 1.377 levels/game. Actual mean results.json 'score' on the same 247 runs = 6.28 (efficiency term included).",
   "source": "src/modelaxis/triage_simulation.py:28-32; results.json 'score' field",
   "matches": true
  },
  {
   "quantity": "drop zero-level games at 40 actions",
   "claimed": "+72% uplift, net +1.36, +17%",
   "recomputed": "abandoned 110, lost 2.04, uplift +72%, gain 3.40, net +1.36 (+17%). K=60: +34%, +0.80 (+10%); K=80: +22%, +0.77",
   "source": "triage_simulation.py logic on frozen corpus",
   "matches": true
  },
  {
   "quantity": "keep top 60% at 60 actions",
   "claimed": "+49% uplift, net +0.99, +13%",
   "recomputed": "No code for any keep-top-X% policy exists in the repo (git ls-files src/modelaxis; grep for top 60/keep_frac/7.70 finds nothing). Reconstruction ranking by levels_at_k with stable order: abandoned 99, lost 1.24, uplift +49%, gain 2.50, net +1.26 (+16%); with actions-on-level tiebreak: +43%, net +1.19.",
   "source": "scratchpad reconstruction",
   "matches": false
  },
  {
   "quantity": "keep top 60% at 30 actions; keep top 75% at 60",
   "claimed": "+59% / +0.88 ; +24% / +0.42",
   "recomputed": "+59% / +0.88 ; +24% / +0.42 (both reconstructions match exactly)",
   "source": "scratchpad reconstruction",
   "matches": true
  },
  {
   "quantity": "aggressive cell keep top 25% at 30 actions",
   "claimed": "+289% uplift, net +7.70",
   "recomputed": "+282% uplift, lost 3.38, gain 11.40, net +8.02 (stable order); +243% / +6.96 (tiebreak variant). Same order of magnitude, not exact; warning that it is an extrapolation is correct.",
   "source": "scratchpad reconstruction",
   "matches": false
  },
  {
   "quantity": "KV10 elasticity (the sole input to GAIN)",
   "claimed": "+47% calls -> +19% levels, ~0.4",
   "recomputed": "KV10 wave 20260903-075332: 43 levels, 83.08 calls/game, 243.32 actions/game vs base 20260902-232148: 36 levels, 56.76 calls, 154.24 actions. Calls +46% -> levels +19.4% => 0.42 in CALLS. In ACTIONS (the unit the simulation's 'uplift' is in): +58% -> 0.34. n = one wave pair; +7 levels vs per-wave sd 2.34.",
   "source": "offkaggle/results/20260903T0553-kv10-keith/20260903-075332-regime-keith/{results.json,requests_shim.jsonl}; offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/; REGIME_WAVE_STATUS.md:944-950",
   "matches": true
  },
  {
   "quantity": "keith_budget 3x elasticity",
   "claimed": "0.21 at 3x (notebook :1260)",
   "recomputed": "20260910-192514-regime-keith_budget: 56 levels, 171.6 calls/game (notebook 169.8), 618.28 actions/game, score 13.76. vs pooled 39.33 / 55.8 calls / 154 actions: levels x1.424, calls x3.08, actions x4.01 => arc elasticity 0.204 in calls, 0.141 in actions, log-log 0.32.",
   "source": "offkaggle/results/20260910-192514-regime-keith_budget/results.json + requests_shim.jsonl; REGIME_WAVE_STATUS.md:1242-1262",
   "matches": true
  },
  {
   "quantity": "in-range rows re-run at elasticity 0.21 (doc's corpus, parity scoring)",
   "claimed": "notebook :1284-1286: 'best net +1.11 on a 9.12 base = +12%'",
   "recomputed": "247-run corpus, 0.21: K=60 keep60 +0.11 (+1%); K=30 keep60 -0.46 (-6%); K=60 keep75 -0.20 (-3%); drop-zero@40 -0.28 (-4%); drop-zero@60 -0.08 (-1%). With real results.json scores instead of parity: +0.57 (+9%), +0.19 (+3%), +0.25 (+4%), +0.41 (+7%). The '9.12 base' matches no corpus slice (all-300 = 8.57, 25-game-only = 9.63, excl. budget arms = 7.85); on the 300-run corpus that includes the 3x wave itself, best in-range cell is K=60 keep40 +0.91 (+11%) at uplift +95%.",
   "source": "scratchpad sensitivity runs",
   "matches": false
  },
  {
   "quantity": "efficiency term omitted by the simulation",
   "claimed": "'LOSS is exact ... scored properly through the depth-weighted rule'",
   "recomputed": "Depth weights correct (1..N, denominator N(N+1)/2, matches arc_agi/scorecard.py rule in memory). Efficiency term s_i = min(115,100*(b/a)^2) is omitted on both sides: real/parity score ratio = 0.88 (09-02 base), 0.85 (09-09 base), 0.85 (KV10), 0.82 (3x). Cleared-level actions/baseline median 0.89 (base) vs 0.84 (3x), share >1x 0.41 vs 0.36 — extra budget does not markedly degrade efficiency, so the omission inflates both loss and gain by ~15% and roughly cancels in relative terms.",
   "source": "results.json score/actions_per_level/baselines; memory arcagi3-scoring-is-convex-in-depth.md",
   "matches": false
  },
  {
   "quantity": "abandoned zero-level games that would have progressed",
   "claimed": "n/a (implicit in loss)",
   "recomputed": "K=40: 110 abandoned, 69 (63%) cleared >=1 level later, 92 levels forfeited, 23 reached >=2. K=60: 78 abandoned, 37 (47%) cleared later, 47 levels, 10 reached >=2.",
   "source": "scratchpad",
   "matches": true
  }
 ],
 "errors_found": [
  "TARGET LEAKAGE (decisive): 'levels cleared so far' vs 'final levels >= 3' is nearly a tautology — at K=60 P(final>=3 | levels so far) = 0.00/0.03/0.57/1.00 for 0/1/2/3+ levels. On the target a controller needs (>=1 MORE level after K) the same feature reads AUC 0.37/0.42/0.47 (below chance) and no eval-legal feature exceeds 0.52. 'Prerequisite PASSES at the top of the published range' is an artifact; the comparison to TextCraft 0.86 / WebShop 0.59 (which predict not-yet-observed success) is invalid. The doc's 'partly self-fulfilling' caveat understates this to the point of being wrong.",
  "The 'keep top 60%/75%/25%' sweep table has NO code in the repo (only the drop-zero-level policy at K=40/60/80 is committed in triage_simulation.py:80-106). Two rows reproduce exactly by reconstruction, two do not (K=60 keep60: net +1.26 not +0.99; K=30 keep25: +282%/+8.02 not +289%/+7.70).",
  "UNITS MISMATCH: elasticity is measured in calls (+47% calls) but the simulation applies it to an ACTIONS uplift (freed actions / survivor actions, triage_simulation.py:93-99). In actions units KV10 elasticity is 0.34 and keith_budget is 0.14, not 0.4 / 0.21.",
  "At the now-measured elasticity 0.21 the doc's own in-range rows on its own corpus read -6% to +1% (parity scoring) or +3% to +9% (real scores) — not +11%. The notebook's later claim (REGIME_WAVE_STATUS.md:1284-1286) of '+1.11 on 9.12 base = +12%' at 0.21 is not reproducible from any committed script or corpus slice; the closest corpus (300 runs) includes the 3x-budget wave itself, whose 'survivors' already had 4x actions — a confound.",
  "Simulation scores everything at the completion-share ceiling (parity efficiency) and omits the quadratic efficiency term; base 7.85 is the ceiling, the actual mean score on the same runs is 6.28. Depth weighting itself is correct.",
  "Corpus is not pinned: the script globs all result dirs, so the doc's numbers already differ today (375 runs, 0.785/0.895/0.933); reproduction required reconstructing the mtime cutoff.",
  "Doc says 'Predicting >= 3 (base rate 0.14)' — recomputed 0.135; rounding only."
 ],
 "confounds": [
  "Pseudo-replication: 247 'runs' are 25 public games played 7-31 times each under 10 different harness arms (wall kill-sets, yield900, probe, carry, ws, wsd, hypo, evid, up8, base); only 5 games ever have a majority of runs >= 3 and cd82/dc22/lf52 contribute 93 runs with zero >= 3 outcomes. The AUC largely encodes game identity; effective n ~ 25. Excluding kill-sets barely moves the AUC (0.772/0.874/0.937), so composition is not what drives it, but CIs are far wider than 247 suggests.",
  "The GAIN side rests on one wave pair (36 vs 43 levels, per-wave sd 2.34, so +7 levels ~ 2 sd) and a single functional form (linear in level count); the 3x measurement (0.21 calls / 0.14 actions) shows it decays, and applying a population-average elasticity to already-deep survivors assumes they progress like the average game, while the incremental AUC < 0.5 says deep-at-K games are LESS likely to clear another level.",
  "Freed 'actions' are not automatically freed wall-clock for survivors: 87% of each call is queue time (memory: queueing-not-compute); the real mechanism is lower queue depth -> more calls per survivor, which is what keith_budget measured (at constant e2e cadence, 128.8 s), so the elasticity is the right instrument but the actions-uplift arithmetic is a proxy.",
  "The verdict rule (>=48 levels = step) is in levels; the doc's bound is in score %; with convex weights a +11% score is less than +11% levels, so the family is even further from the 48-level bar than the score % suggests."
 ],
 "does_conclusion_survive": "YES, but on corrected grounds and with a corrected number. The qualitative conclusion — the external budget-controller family is not a step and sits far below the step bar (+22% levels = 48 vs 39.33) — survives and is strengthened: at the measured elasticity 0.21 (0.14 in the simulation's own action units) the in-range policies on the doc's own corpus net roughly -6% to +9% depending on scoring assumption, not +11%, and the aggressive cells remain unsupported extrapolations. Two of the doc's supporting claims do NOT survive: (1) 'the prerequisite passes at the top of the published range' is refuted — the 0.94 AUC is target leakage, and on the incremental target no eval-legal feature beats ~0.52 (the controller cannot tell which games will progress; it can only tell which already have); (2) the '+11% ceiling' should be restated as 'approximately 0 to +9% at measured elasticity, with sign depending on the scoring assumption'. The triage bound itself does not depend on the AUC (the policy ranks by observed levels-so-far, and the loss side is exact for the committed drop-zero policy), so the bound stands independently of the leaky AUC, but the AUC framing and the TextCraft/WebShop comparison should be retracted, and the sweep code producing the keep-top-X% table should be committed or the table marked unreproducible.",
 "what_would_change_verdict": "Finding the actual keep-top-X% sweep script and the corpus behind '+1.11 on 9.12' and showing they reproduce (would move the table rows from PARTLY to CONFIRMED but not change the leakage finding). A measured eval-legal feature with incremental-target AUC >= 0.7 on independent games (not repeated plays of the same 25) would revive the prerequisite. A measured elasticity at +50-100% uplift near 0.4 in ACTION units on survivors specifically (not the population average) would restore the +11-17% in-range rows. None of these would lift the family to the +22% levels step bar on the present evidence.",
 "commands_run": [
  "cat docs/research-2026-09-10/R-triage-family-bounded.md; cat -n src/modelaxis/early_predictability.py; cat -n src/modelaxis/triage_simulation.py",
  ".venv/bin/python src/modelaxis/early_predictability.py  (375 runs today: 0.785/0.895/0.933)",
  ".venv/bin/python src/modelaxis/triage_simulation.py  (375 runs today: base 7.43, drop-zero@40 +2.20)",
  "git show --stat 8614b7f; git diff 8614b7f -- src/modelaxis/*.py; git ls-files src/modelaxis/; grep -rn 'top 60|7\\.70|+0\\.99|keep_frac|+289' (no sweep code found)",
  "python inventory of offkaggle/results/**/results.json by mtime and *_events.jsonl count (247 runs before 2026-09-10 09:10; 375 total)",
  "scratchpad corpus.py + reruns of both scripts' logic on the frozen 247-run corpus (AUC 0.771/0.884/0.944; base 7.85; drop-zero@40 +72%/+1.36)",
  "strata P(final>=3 | levels_at_K), incremental-target AUC for levels/actions-on-level/game_overs/distinct_actions, within-stratum AUC, per-game-mean AUC, consistency checks (max(level)-1 == levels_completed: 0 mismatches; max(action_num) == actions: 0 mismatches)",
  "reconstruction of keep-top-frac policies at K=30/40/60 with two tie-break variants; elasticity sensitivity 0.4/0.21/0.1/0.0; real-score variant",
  "grep -n 'elasticit|+47|0\\.21|keith_budget' offkaggle/REGIME_WAVE_STATUS.md; sed -n 944,956p; sed -n 1242,1300p",
  "sed -n 1,80p ~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-scoring-is-convex-in-depth.md; grep -n 'def .*score|baseline' offkaggle/run_regime_wave.py",
  "per-wave recompute of levels/calls/actions/score for 20260902-232148 base, 20260903-075332 KV10, 20260909-193735 base, 20260910-192514 keith_budget, 20260910-081906 keith_fx (calls = requests_shim.jsonl rows / 25)",
  "AUC on 25-game waves only (n=175); 300-run corpus (cutoff 2026-09-11 02:05) rerun at elasticity 0.21 across K in {30,40,60,80} and keep in {0.25,0.4,0.5,0.6,0.75,drop-zero}; base-score variants (8.57/9.63/7.85)",
  "real/parity score ratio and cleared-level actions/baseline median per wave"
 ]
}
```

---

## agent-4

```json
{
 "claim": "\"The pooled six-draw base 39.33 sd 2.34 and the sigma verdict rule\": every verdict since 09-08 is levels vs a pooled six-draw base of 39.33 (sd 2.34); >=48 step candidate, 45-47 redraw, <=44 dead; engaged-but-flat class closed on four replications of +/-2 levels.",
 "verdict": "CONFIRMED_WITH_CORRECTIONS",
 "recomputed": [
  {
   "quantity": "Composition of the six-draw pool",
   "claimed": "six rig draws (implied by 'pooled six-draw base' in every verdict)",
   "recomputed": "Y1 41 = Modal 20260906T2251 yield900; Y2 40 = Modal 20260907T0702-cb2 yield900; YK 37 = Kaggle commit arc3-keith-yield900 (kf 347926973); M1 36 = Modal 20260902T2121 yield-60; M4 40 = Modal 20260903T1014-keith_retry (RETRY graft installed, fired 5x); K 42 = Kaggle commit arc3-keith-copy v4. Only 4 of 6 are rig draws; 3 of 6 are yield900 and 3 yield-60.",
   "source": "offkaggle/run_regime_wave.py:312-315 (comment + base_levels_six_draws [41,40,37,36,40,42]); docs/research-2026-09-08/R-loss-ledger-3-NOTES.md:3-10; offkaggle/results/20260903T1014-keith_retry/.../summary.txt RETRY line",
   "matches": false
  },
  {
   "quantity": "Pool mean and sd",
   "claimed": "39.33, sd 2.34",
   "recomputed": "mean 39.3333, sample sd 2.3381 (population sd would be 2.134) from [41,40,37,36,40,42]",
   "source": "python3 statistics.mean/stdev on the runner's list; the four Modal totals recomputed from results.json['games'] levels_completed: M1 36, M4 40, Y1 41, Y2 40 (scratchpad/tally.py, matrix.py)",
   "matches": true
  },
  {
   "quantity": "YK 37 and K 42 (the two Kaggle commit draws)",
   "claimed": "37 and 42 levels",
   "recomputed": "NOT VERIFIABLE from raw files: no per-game file for either Kaggle commit exists in the repo; the notes cite scratchpad/search/loss-ledger-3/cache.json, which does not exist (ls scratchpad/search, mdfind, grep -rl 347926973 all empty). Values rest on docs only.",
   "source": "docs/research-2026-09-06/R-loss-ledger-2-NOTES.md:5; docs/research-2026-09-08/R-loss-ledger-3-NOTES.md:7,10",
   "matches": false
  },
  {
   "quantity": "Geometry of each pooled draw",
   "claimed": "same geometry (conc 28, 7,920 s, window 32768)",
   "recomputed": "M1, M4, Y1, Y2 all print 'geometry conc 28 | 7920 s/game', CONTEXT_WINDOW=32768, identity profile kv5-bf16-mtp3-c8-cg32 on RTX PRO 6000 (M1 has no IDENTITY line). Yield differs by design: M1/M4 60 s, Y1/Y2 900 s. YK/K geometry (Kaggle, 'conc 28, 7,920 s') is asserted only in R-loss-ledger-3-NOTES.md:3 and cannot be checked from raw files; Kaggle serving showed 3 running/21 waiting (NOTES:61).",
   "source": "offkaggle/results/<dir>/summary.txt lines 'geometry conc' and 'ARM KNOB' for the four Modal dirs",
   "matches": true
  },
  {
   "quantity": "M4 'retries not engaged'",
   "claimed": "stock loop / retries not engaged",
   "recomputed": "RETRY fired 5 (0.20/game), games with retry 5, levels cleared after retry 0; knob override RETRY_K=2.5 vs the code's 3. The graft was installed and did fire; it is a non-stock arm labelled base.",
   "source": "offkaggle/results/20260903T1014-keith_retry/20260903-121426-regime-keith_retry/summary.txt RETRY and KNOB OVERRIDES lines; arm_env.json RETRY_*",
   "matches": false
  },
  {
   "quantity": "Chi-square 95% CI on sd (5 df)",
   "claimed": "roughly 0.62x to 2.45x the point estimate",
   "recomputed": "factors sqrt(5/12.8325)=0.624 and sqrt(5/0.8312)=2.453 -> CI on 2.34 = [1.46, 5.74]",
   "source": "scratchpad/stats.py",
   "matches": true
  },
  {
   "quantity": "Per-game draw variance across the four Modal base draws",
   "claimed": "(task premise) per-game sd ~1.2 -> 25-game total sd 5-6 if independent",
   "recomputed": "sum of per-game variances over 25 games (4 draws, 75 df) = 10.42 -> implied total sd under independence = 3.23; rms per-game draw sd 0.65; mean within-game draw variance 0.417 vs between-game variance of game means 1.013. Observed variance of the four totals = 4.92 (sd 2.22), i.e. no positive between-game covariance. Pairwise paired-diff sd between base draws 0.76-1.19 per game -> sd of a difference of two totals 3.8-6.0, consistent with a single-total sd of ~3.2.",
   "source": "scratchpad/stats.py on matrix.json built from the four results.json files",
   "matches": false
  },
  {
   "quantity": "Origin of loss-ledger-3's 'sd of a 25-game total ~6'",
   "claimed": "Per-run levels sd 1.19 => sd of a 25-game total ~6",
   "recomputed": "1.19 is the CROSS-SECTIONAL sd of levels across the 25 games within one run (Y2 = 1.190 exactly; M1 0.92, M4 1.32, Y1 1.15). 71% of that variance is fixed between-game heterogeneity (ft09/lp85/re86 vs one-level games), not draw noise. Multiplying it by 5 is a mis-derivation; the draw-noise sd of a total is ~3.2.",
   "source": "docs/research-2026-09-08/R-loss-ledger-3-NOTES.md:27 vs scratchpad decomposition (between 1.013 / within 0.417)",
   "matches": false
  },
  {
   "quantity": "AB-instrument 'per-game sd ~1.2'",
   "claimed": "identical-stock per-game sd ~1.2 levels",
   "recomputed": "The doc states the per-game PAIRED-DIFFERENCE sd is ~1.2 (D_AA paired sd 1.17, D_BA 0.88), on a different instrument (same-boot v22 Kaggle kernel, 09-02, stock 1.04-1.20 lv/game; the model/serving is not stated in the doc). A paired-diff sd of 1.2 implies a single-total sd of 1.2/sqrt(2)*5 = 4.2, not 5-6; the rig's own base draws give 3.2.",
   "source": "docs/EXPERIMENT-2026-09-02-ab-instrument-control.md:50-52,134,139,168,191",
   "matches": false
  },
  {
   "quantity": "Pooling yield900 with yield-60 draws",
   "claimed": "levels are a null between the regimes",
   "recomputed": "paired per game (Y1+Y2)/2 - (M1+M4)/2: mean +0.10 lv/game, sd 0.66, se 0.13 -> consistent with null; pooling defensible.",
   "source": "scratchpad/stats.py",
   "matches": true
  },
  {
   "quantity": "Excluded void yield900 draw (20260907T0106-cb)",
   "claimed": "VOID (server died at ~+120 min, 308 request errors)",
   "recomputed": "results.json total = 31 levels; matched-time read 30 at +120 min. Including it: five Modal draws mean 37.6, sd 4.16. Exclusion follows a pre-registered VOID rule, but the pool's sd is sensitive to it.",
   "source": "offkaggle/results/20260907T0106-keith_yield900-cb/.../results.json; REGIME_WAVE_STATUS.md:566-575",
   "matches": true
  },
  {
   "quantity": "Cadence arm 34 = -2.28 sd",
   "claimed": "34 vs 39.33 = -5.33 = -2.28 sd -> DEAD",
   "recomputed": "results.json total 34; (34-39.33)/2.34 = -2.28. But geometry was conc 6 / 1,767 s per game, 'public25 geometry: False' (by design); z at sd 3.23 = -1.65, at sd 4/5/6 = -1.33/-1.07/-0.89.",
   "source": "offkaggle/results/20260909-193735-regime-keith/summary.txt geometry + SCORE lines; offkaggle/read_cadence_arm.py:17-19,80",
   "matches": true
  },
  {
   "quantity": "fx 35 = -1.85 sd",
   "claimed": "-4.33 = -1.85 sd",
   "recomputed": "total 35, conc 28 / 7920 s; z = -1.85 at 2.34; -1.34 at 3.23; -1.08/-0.87/-0.72 at 4/5/6.",
   "source": "offkaggle/results/20260910-081906-regime-keith_fx/summary.txt; stats.py",
   "matches": true
  },
  {
   "quantity": "probe 41 and carry 41 = +0.71 sd",
   "claimed": "+1.7 = +0.71 sd each",
   "recomputed": "both totals 41 from results.json; +1.67/2.34 = +0.71; at sd 3.23 +0.52; at 4/5/6 +0.42/+0.33/+0.28.",
   "source": "offkaggle/results/20260908T0752-keith_probe/... and 20260908T2110-keith_carry75/... results.json; runner PROBE-PRIMARY line",
   "matches": true
  },
  {
   "quantity": "budget 56 = +7.12 sd",
   "claimed": "+16.67 = +7.12 sd",
   "recomputed": "total 56, but at 23,760 s/game (3x clock), wall 6.60 h, 'public25 geometry: False' -- not the live geometry; z = +7.12 at 2.34, +5.2 at 3.23, +4.17/+3.33/+2.78 at 4/5/6 -- stays significant.",
   "source": "offkaggle/results/20260910-192514-regime-keith_budget/summary.txt",
   "matches": true
  },
  {
   "quantity": "ctx12 12 / ctx16 21 / nr16 20",
   "claimed": "-11.68 / -7.83 / -8.26 sd",
   "recomputed": "totals 12, 21, 20 from results.json; z at 2.34 = -11.68/-7.83/-8.26; at sd 6 = -4.55/-3.05/-3.22 -- all remain dead.",
   "source": "offkaggle/results/20260911-*/results.json; stats.py",
   "matches": true
  },
  {
   "quantity": "48-level step bar in sigma units",
   "claimed": "implicitly >3.7 sd (48 = +8.67 at sd 2.34)",
   "recomputed": "+3.71 sd at 2.34; +2.68 at 3.23; +2.17/+1.73/+1.45 at 4/5/6. The 45-47 redraw band is +1.4 to +2.4 sd at 2.34 but only +0.95 to +1.4 at sd 6. Also the rule treats the pooled mean as exact; sd of (arm - pooled mean) = sd*sqrt(1+1/6) = 2.53 at 2.34, 3.49 at 3.23.",
   "source": "stats.py",
   "matches": false
  },
  {
   "quantity": "Rule thresholds recorded in code",
   "claimed": ">=48 step / 45-47 redraw / <=44 dead",
   "recomputed": "read_cadence_arm.py:19 uses 48/45/44; the runner's PROBE-PRIMARY summary line prints '>= 52 step candidate, 45-51 redraw, < 45 dead' (the probe pre-registration's rule). Two thresholds coexist; the notebook moved from 52 (probe, line 619) to 48 (carry onward, line 687) without a recorded justification.",
   "source": "offkaggle/read_cadence_arm.py:18-19; keith_probe summary.txt PROBE-PRIMARY line; REGIME_WAVE_STATUS.md:619,687",
   "matches": false
  }
 ],
 "errors_found": [
  "The 'six-draw base' is not six rig draws: two are Kaggle commit runs (YK 37, K 42) whose per-game data is not on disk anywhere (the cited scratchpad/search/loss-ledger-3/cache.json does not exist), so only 4 of the 6 totals can be recomputed from raw files.",
  "M4 (40) is the keith_retry arm with the retry graft installed and firing 5 times (0.20/game, RETRY_K override 2.5); calling it 'stock loop / not engaged' is a stretch. It is a non-stock arm inside the base pool.",
  "The pool mixes two yield regimes (3 x 900 s, 3 x 60 s). Pooling is defensible only because the paired per-game difference is null (+0.10 +/- 0.13 lv/game on the four Modal draws), which the notebook asserts but never shows per game.",
  "Internal contradiction on the same day: R-loss-ledger-3.md:11,57 and NOTES:27 state 'sd of a 25-game total ~6' and set the step bar at >=52, while the notebook/runner adopted sd 2.34 and bar 48 with no recorded reconciliation. Both numbers are wrong in opposite directions: the '~6' multiplies the cross-sectional per-run sd (1.19, 71% between-game heterogeneity) by 5, and the 2.34 is an n=6 point estimate whose 95% CI is [1.46, 5.74].",
  "Direct per-game decomposition on the four Modal base draws (75 df) gives within-game draw variance 0.417/game -> 25-game total sd ~3.2 under independence, with observed total variance (4.92) showing no positive covariance. The best-supported sd is ~3.2, not 2.34 and not 5-6.",
  "The task's own premise 'per-game sd ~1.2 -> total sd 5-6' is a mis-reading of the AB doc: 1.2 is the paired-DIFFERENCE sd (implying single-total sd ~4.2), and it was measured on a different instrument (09-02 v22 same-boot Kaggle kernel at 1.0-1.2 lv/game).",
  "Sigma statements on the cadence arm (conc 6 / 1,767 s) and the budget arm (23,760 s/game) compare arms at a different geometry to a base at the live geometry; those were deliberate designs, but '-2.28 sd' and '+7.12 sd' are not like-for-like draws of the same instrument.",
  "Two different rule thresholds coexist in code and summaries (runner PROBE line: 52/45-51/<45; read_cadence_arm.py: 48/45-47/<=44).",
  "'Patch 21' counted as one of the four replications is from the 08-27 27B-duck era (memory arcagi3-two-level-wall, dated 2026-08-27), before the Flash-Next regime existed (09-02); it is not a replication on this instrument."
 ],
 "confounds": [
  "Two of six pooled draws ran on Kaggle hardware (3 running / 21 waiting per the vLLM log cited in NOTES:61) rather than the Modal RTX PRO 6000; the notes claim matched e2e (~158 s/call) but this cannot be checked from raw files.",
  "Yield regime differs within the pool (60 vs 900 s); actions/game differ substantially (154/119 vs 102/124) even though levels do not.",
  "The excluded void draw read 31 (30 at matched time); including it moves the Modal-only pool to mean 37.6, sd 4.2. Its exclusion is by a pre-registered rule, but the pool's tightness partly reflects that exclusion.",
  "The rule ignores the pooled mean's own sampling error (se = sd/sqrt(6)); the correct sd for arm-minus-pool is sd*sqrt(7/6).",
  "Order/boot effects: the AB-instrument doc found a +0.25 lv/game order effect; the pooled draws are a mix of first-in-boot and later-in-session runs (Y1 was third in its session; Y2 first), which the sd estimate absorbs but does not control.",
  "M1 (36) is the only draw whose summary lacks an IDENTITY line, so its serving profile is not machine-attested in the file (the notebook asserts the same docker/profile)."
 ],
 "does_conclusion_survive": "Mostly yes, with the sigma language downgraded. Recomputed from the per-game data, the honest sd of a 25-game total is ~3.2 (95% CI on the n=6 estimate spans 1.5-5.7 and contains it), not 2.34 and not 5-6. Under sd ~3.2 (or the arm-vs-pool sd ~3.5): cadence 34 is -1.5 to -1.65 sd and fx 35 is -1.2 to -1.35 sd -- neither is 'outside 2 sd' any more, so 'not flat, WORSE' is over-claimed; both are still correctly DEAD because nothing below the base can be a step. Probe 41 and carry 41 stay flat (+0.5 sd). Budget 56 (+5 sd), ctx12 (-8), ctx16 (-5.7), nr16 (-6) keep their verdicts at any sd up to 6. The 48 bar becomes ~+2.5-2.7 sd (still a reasonable one-draw step screen), but the 45-47 'redraw' band is ~+1.6-2.4 sd and ~+1 sd if sd were 6 -- effectively noise, which is what the redraw rule already assumes. On the 'engaged-but-flat class is closed' inference: it is statistically justified only for STEP-sized effects (>= +8.7 levels): each single draw at 41 excludes a true +8.7 at roughly p~0.03-0.06, the yield900 pair (75 vs 75 runs, paired se 0.11 lv/game, i.e. +/-5.5 levels) is the one well-powered null, and pooling probe+carry+yield900 gives a class mean of ~+1.5 levels with a 95% CI of roughly [-2.5, +5.5] levels. That CI does NOT exclude +10-15% gains (up to ~+0.2 lv/game), which the AB-instrument doc already declared unmeasurable; and only three of the four cited replications are on this instrument (patch 21 is 27B-era). So: 'no behaviour-shaping lever has produced a step on Flash-Next' survives; 'the class is closed' as a claim about all effects does not, and 'four replications' should read three.",
 "what_would_change_verdict": "Producing the Kaggle per-game files for YK and K (re-fetch kf 347926973 and arc3-keith-copy v4 outputs) would let the six-draw pool and the 12-wall list be recomputed in full; if their per-game draw variance is materially higher than the Modal draws' 0.42/game (plausible given 3-running/21-waiting Kaggle serving), the total sd rises toward 4 and cadence/fx become clearly within-noise. Conversely, two or three more clean same-geometry base draws on the rig would tighten the sd CI from [1.5, 5.7] to something usable; if they land in 36-42 the 2.34 stands and the sigma reads are fine. A single base draw outside 33-45 would refute the 2.34 outright.",
 "commands_run": [
  "grep -n '39.33|six-draw|pooled' offkaggle/REGIME_WAVE_STATUS.md; sed -n 540-700, 905-930, 983-1000, 1076-1084, 1248-1262, 1351-1358, 1453-1468, 1584-1600 of the same file",
  "grep -rn '39\\.33' docs/ offkaggle/*.py memory; sed -n 295,330p offkaggle/run_regime_wave.py (LEDGER3_REFERENCE base_levels_six_draws)",
  "cat docs/research-2026-09-08/R-loss-ledger-3.md; sed -n 1,30p docs/research-2026-09-08/R-loss-ledger-3-NOTES.md; sed -n 1,14p docs/research-2026-09-06/R-loss-ledger-2-NOTES.md",
  "python3 scratchpad/tally.py (levels_completed summed from every offkaggle/results/**/results.json['games'])",
  "python3 scratchpad/matrix.py (25-game x 16-wave per-game level matrix -> matrix.json)",
  "grep -h 'geometry conc|SCORE|CADENCE|ARM KNOB|IDENTITY|RETRY' <each 25-game summary.txt>; cat offkaggle/results/20260903T1014-keith_retry/.../arm_env.json",
  "python3 scratchpad/stats.py (pool mean/sd, chi-square CI factors, per-game variances across M1/M4/Y1/Y2, pairwise paired-diff sds, z-scores at sd 2.34/4/5/6, sd*sqrt(7/6))",
  "python3 decomposition: cross-sectional per-run sd, between-game vs within-game variance",
  "sed -n 1,60p and 180,200p docs/EXPERIMENT-2026-09-02-ab-instrument-control.md; grep -n -i 'flash|27b|qwen|model' same file",
  "grep -n '48|45|44|2.34' offkaggle/read_cadence_arm.py; grep -h PRIMARY <summaries>",
  "search for the Kaggle commit per-game cache: ls scratchpad/search; grep -rl 347926973 --include='*.json' .; mdfind -name loss-ledger-3; find -iname '*yield900*' (none found besides docs/notebook)",
  "sed -n 1,12p memory/arcagi3-two-level-wall.md (patch 21 date/era)"
 ]
}
```

---

## agent-5

```json
{
 "claim": "keith_ctx16 (LOCAL_ANALYZER_CONTEXT_WINDOW 32768 -> 16384): calls/game 81.8 (1.47x) PASS, actions/call 1.35 FAIL vs gate 2.0, 21 levels = -7.83 sd -> DEAD; the pre-registered threshold theory (16k holds the longest observed turn so it is a different regime) is refuted; degradation is a smooth gradient 2.76 -> 1.35 -> 0.27 so there is no safe window and Route A (shrink sequences to buy calls) is closed. (offkaggle/REGIME_WAVE_STATUS.md ~1446-1490; wave offkaggle/results/20260911-111640-regime-keith_ctx16)",
 "verdict": "CONFIRMED_WITH_CORRECTIONS",
 "recomputed": [
  {
   "quantity": "calls/game",
   "claimed": "81.8 (1.47x base 55.8)",
   "recomputed": "2046 status-200 rows / 25 = 81.84; transcript parse 2043/25 = 81.7; 81.84/55.8 = 1.466x. Base 55.8 = 20260902T2121-keith/20260902-232148-regime-keith summary.txt CADENCE line",
   "source": "requests_shim.jsonl (python3 count of status==200); summary.txt CADENCE line",
   "matches": true
  },
  {
   "quantity": "actions/call",
   "claimed": "1.35 (base 2.76, ctx12 0.27)",
   "recomputed": "2752 actions (results.json sum) / 2046 calls = 1.345; base 3856/1384 transcript calls = 2.79 (154/55.8 = 2.76); ctx12 722/2709 = 0.267",
   "source": "results.json + requests_shim.jsonl; run_regime_wave.py:1603 actions_per_call = actions_total / n_calls",
   "matches": true
  },
  {
   "quantity": "levels and sd distance",
   "claimed": "21 levels = -7.83 sd vs pooled 39.33 sd 2.34",
   "recomputed": "sum(levels_completed) = 21; (21-39.33)/2.34 = -7.83. Caveat: the six-draw pool mixes yield-60 and yield-900 draws (STATUS:640-641 names M1 36, M4 40, Y1 41, Y2 40; remaining two not identifiable from the notebook). Same-knob yield-60 32k draws on disk: 36 (09-02) and 35 (fx 09-10) -> ctx16 is -14.5 levels vs their mean; per game ctx16 is below both draws in 5/25 games and above both in 0/25",
   "source": "results.json; STATUS:640-641; 20260902...-regime-keith/summary.txt; 20260910-081906-regime-keith_fx/summary.txt",
   "matches": true
  },
  {
   "quantity": "median prompt tokens",
   "claimed": "9,582 (base 20,061; ctx12 6,674)",
   "recomputed": "ctx16 median 9582 mean 9510 max 15289; ctx12 median 6674; fx median 21848",
   "source": "requests_shim.jsonl prompt_tokens over status==200 rows",
   "matches": true
  },
  {
   "quantity": "post errors / metrics coverage",
   "claimed": "24 post errors; metrics 1 -> 2057",
   "recomputed": "24 non-200 rows: 23 ReadTimeout (all at +131..132 min, one burst) + 1 HTTP 502 at +69 min; vllm:request_success_total{stop} 1.0 before -> 2057.0 after (2046 client-OK + 11 that finished server-side after the client timed out)",
   "source": "requests_shim.jsonl; metrics_before.prom / metrics_after.prom",
   "matches": true
  },
  {
   "quantity": "vLLM preemptions",
   "claimed": "1,908 (STATUS: ignored as before; ~65-181 in normal base waves)",
   "recomputed": "num_preemptions_total 0 -> 1908 on 2056 requests (0.93/request). Other waves (before->after): 09-09 keith 59, fx 65, budget 266 (3x clock), ctx12 0, nr16 (also 16k) 1652. The ctx16 write-up itself (STATUS 1446-1490) contains no mention of preemptions at all",
   "source": "grep vllm:num_preemptions_total offkaggle/results/*/metrics_*.prom; grep -n preempt REGIME_WAVE_STATUS.md",
   "matches": true
  },
  {
   "quantity": "history budget arithmetic",
   "claimed": "16384 -> 11,445; 12288 -> 7,349; 32768 -> 27,829 (system 3,170 + user ~745)",
   "recomputed": "context_budget_tokens 15360/11264/31744 (summary ARM KNOB lines) minus 3915 = 11445/7349/27829. First-call prompt (system+user only, n_messages==2) is 4052 tokens, not 3915, so the true history budgets are ~137 tokens smaller; immaterial",
   "source": "summary.txt ARM KNOB line; requests_shim.jsonl n_messages==2 rows",
   "matches": true
  },
  {
   "quantity": "longest observed completion",
   "claimed": "10,439 tokens (so 16k's 11,445 budget holds a max-length turn)",
   "recomputed": "10,439 is the ctx12 max. On disk before the pre-registration: 09-02 base 10,495; fx 10,673; wall-keith_evid 11,685; carry-kill 13,800. In ctx16 itself max 12,942 (2 calls > 11,445). A retained turn also carries the tool result (<=1024) and user prompt (~745), so a 10.4k completion turn is ~12.2k > 11,445. But only 0.1% of calls exceed the budget in either ctx wave, so this edge cannot drive a 2-10x actions/call change either way",
   "source": "python3 over every */requests_shim.jsonl completion_tokens",
   "matches": false
  },
  {
   "quantity": "retained history depth (the actual mechanism)",
   "claimed": "not reported in the write-up",
   "recomputed": "mean history_messages per turn: fx 21.2 (~7 exchanges), ctx16 9.15 (~3), ctx12 4.49 (~1.5). Turns with exactly one prior exchange (history_messages=3): fx 0%, ctx16 89/1942 = 4.6%, ctx12 926/2107 = 44%",
   "source": "grep ^history_messages: transcripts/*.txt per wave",
   "matches": true
  },
  {
   "quantity": "decomposition of actions/call",
   "claimed": "actions/call falls 2.76 -> 1.35 -> 0.27 'smoothly'",
   "recomputed": "P(call contains action()) x actions per acting call: 09-02 base 0.563 x 4.95 = 2.79; fx 0.517 x 5.01 = 2.59; ctx16 0.425 x 3.17 = 1.35; ctx12 0.117 x 2.22 = 0.26. Linear-in-window would predict 1.38 (obs 1.35) and 1.03 (obs 0.27): the 16k->12k step is a knee in acting propensity (x0.28), not a smooth continuation. Levels alone are close to linear in window (predicted 19.7/14.7 vs 21/12)",
   "source": "regex classification of every <parameter=code> block in transcripts/*.txt for the four waves",
   "matches": false
  },
  {
   "quantity": "preemption cost on e2e / calls (confound b)",
   "claimed": "not analysed in the write-up",
   "recomputed": "decode time per generated token: ctx16 17.37 s/1116 = 15.6 ms; fx 16.32/1350 = 12.1 ms; ctx12 13.22/1024 = 12.9 ms; 09-09 base 11.5 ms. Prefill 0.94 s. Using the ctx12 (0-preemption, same 8-seq batch) rate, ctx16 inference would be ~15.3 s not 18.3 s; e2e ~80 s not 95 s; ~99 calls/game not 81.8 (pre-registered projection was 103). Mechanism: kv_cache_size_tokens=105202 with max 8 seqs; ctx16 mean sequence 10,628 tokens (8x = 85k; p90 12,888 x 8 = 103k; 8.3% of sequences > 105202/8) so 8 seqs are admitted and overflow in decode; at 32k only 4.8 fit (max_num_seqs never binds); at 12k 8 x 7,625 = 61k never fills. nr16 (16k) reproduces the thrash (1652)",
   "source": "metrics_after.prom minus metrics_before.prom (request_decode_time/prefill/queue sums and counts); cache_config_info; requests_shim.jsonl prompt+completion",
   "matches": true
  },
  {
   "quantity": "what non-acting calls do (d)",
   "claimed": "implied: trimmer evicts constantly / thrashing",
   "recomputed": "ctx16: 2043 tool calls = 868 with action() + 1172 print/inspect analysis + 3 other; no-tool-call calls 6 (0.3%); verbatim consecutive-code repeats 111 (5.4%, fx 6.5%); no refusals. Transcripts show re-perception churn, not refusal or loops: lp85 turns 40-50 = 11 consecutive analysis-only turns at step 10 (12:34-12:49) re-deriving the same geometry/goal with 6-9 history messages; sp80 #34 'The world model isn't accumulating well', #40 'The world model hasn't been updated. Let me check the current state'; tr87 step 5 = 18 turns of per-transition diff attribution over 35 min",
   "source": "transcripts/lp85-305b61c3_p0.txt, sp80-589a99af_p0.txt, tr87-cd924810_p0.txt (parsed per turn); classification script over all transcripts",
   "matches": false
  }
 ],
 "errors_found": [
  "Longest-observed-completion premise is wrong on data that was already on disk: 10,439 is the ctx12 max, but 09-02 base had 10,495, fx 10,673, wall-keith_evid 11,685, carry-kill 13,800, and ctx16 itself hit 12,942; retained turns also carry ~1.8k of tool-result+user tokens. The threshold theory as written was not supported by tail statistics even before the $9 was spent (0.1% of calls exceed the budget in either ctx wave), so its 'refutation' is of a hypothesis that never had a plausible mechanism.",
  "'Degradation tracks context smoothly' is over-claimed: actions/call is linear from 32k to 16k (predicted 1.38, observed 1.35) and then collapses at 12k (predicted 1.03, observed 0.27); the decomposition shows acting propensity P(act) drops x0.79 then x0.28, coinciding with 44% of ctx12 turns retaining only one prior exchange. Three single-draw points cannot distinguish a gradient from a gradient-plus-knee. The defensible statement is 'monotone, and 16k is itself dead'.",
  "The ctx16 result section does not report the 1,908 vLLM preemptions (10-30x the 59-170 band of every 32k wave; 0 at 12k), nor that nr16 reproduced it (1,652). The 'ignored as before' rationale at STATUS:982-983 (base waves trip the rule harder) does not hold here. The arm therefore mixes two effects: the window (content/history) and a scheduler thrash that shipped with the window (8 seqs admitted at ~10.6k tokens overflow the 105k-token KV). The thrash costs ~17 calls/game (decode 15.6 vs 12.1-12.9 ms/token) and explains the shortfall vs the projected 103 calls; it cannot affect actions/call.",
  "The '-7.83 sd' uses a pooled base (39.33) that mixes yield-60 and yield-900 draws; the same-knob yield-60 32k draws on disk read 36 and 35. Against those ctx16 is -14.5 levels (below both draws in 5/25 games, above both in 0/25). Direction and verdict unchanged; the sd multiple is not the right headline.",
  "Minor: the history-budget arithmetic uses 3,170 + 745 = 3,915 for system+user, but the measured system+user first-call prompt is 4,052 tokens (budgets ~137 tokens smaller). Immaterial."
 ],
 "confounds": [
  "KV-scheduler thrash specific to the 16k window under the kv5/c8 profile (1,908 preemptions; reproduced by nr16 at 1,652). It inflates decode time ~25% and cost ~17 calls/game, so the calls gate passed with margin lost to the scheduler rather than the knob. Because the live kernel uses the same profile, the thrash would ship with the knob; a max_num_seqs=7 or slightly larger KV would remove it, but that only buys calls (elasticity 0.21 -> ~+1 level), not actions/call.",
  "Base pool heterogeneity: 39.33/2.34 pools yield-60 and yield-900 draws; the six members are not fully identifiable from the notebook (four named at STATUS:640-641). The between-wave sd of 2.34 is estimated from six draws; the campaign's per-game sd (~1.2 lv, A/B instrument memo) implies a 25-game wave sd nearer 6, which would make ctx16 ~-3 sd rather than -7.83 (still dead).",
  "All three windows have e2e > 60 s, so the 60 s yield fires on every analysis-only call in every wave (calls/turn 1.02-1.04); yield mechanics are therefore matched across arms and are not a differential confound, but they do mean each analysis-only call costs a whole turn.",
  "Single draw per window; the A/B instrument memo says 25-game A/B MDE is ~+0.5 lv/game (~12 levels), so the ctx16 vs ctx12 difference (9 levels) is itself inside noise; only the base-vs-either-ctx gap is clearly resolved.",
  "git_status.txt records the harness checkout as DIRTY at run time (ARC3-Inference b2607cd DIRTY); I could not verify from the wave directory that nothing beyond the CONTEXT_WINDOW env var differed (arm_env.json shows only the window changed among the env keys).",
  "Windows between 16k and 32k (e.g. 20-28k) were never measured; 'no safe window' and 'Route A closed' are interpolations from three points plus the calls elasticity (0.21), not measurements."
 ],
 "does_conclusion_survive": "Yes, the operative conclusion survives: keith_ctx16 is dead and the shrink-sequences route is not worth another wave. Every headline number reproduces from raw files, the level deficit is enormous under any comparator (-18 vs the pooled base, -14.5 vs the same-knob yield-60 draws, below both base draws in 5/25 games and above in none), and the actions/call collapse is a property of the model's emitted code (acting propensity 0.52-0.56 -> 0.425 -> 0.117; actions per acting call 5.0 -> 3.17 -> 2.22), which the scheduler cannot alter. The heavy preemption thrash is a real, unreported instrument anomaly, but it only depresses throughput (~17 calls/game) and if corrected would move levels by about one at the measured elasticity. What does not survive as written: (1) the pre-registered threshold mechanism was already contradicted by tail statistics on disk and affected 0.1% of calls, so its refutation is not informative about thresholds in general; (2) 'smooth gradient / no threshold' is over-claimed - the 16k->12k step shows a knee in acting propensity that coincides with history depth falling to one exchange, and three single-draw points cannot adjudicate shape; (3) 'no safe window' above 16k is an interpolation, not a measurement. Correct statement: context capacity is monotonically load-bearing between 32k and 12k, the effect at 16k alone is far larger than any calls gain can offset (levels ~proportional to window vs calls elasticity 0.21), and the mechanism visible in transcripts is re-perception churn as retained history shrinks from ~7 exchanges to ~3, not eviction of single over-long turns and not refusal or looping.",
 "what_would_change_verdict": "A 16k redraw with the scheduler thrash removed (max_num_seqs 6-7 or KV 5.5-6 GiB at the same window) reading >= 45 levels would show the window is not the cause - the arithmetic above says it should read ~22, so this is a cheap falsifier. A 24k window reading >= 45 would falsify 'no safe window'. Identifying the six pool draws and finding that they are all yield-60 32k runs would restore the -7.83 sd headline; finding the wave sd is ~6 rather than 2.34 would shrink it to ~-3 sd without changing the verdict. Finding a code diff in the DIRTY harness checkout beyond the env knob would void the arm.",
 "commands_run": [
  "sed -n 1380,1500p offkaggle/REGIME_WAVE_STATUS.md; sed -n 600,640p; sed -n 636,662p; grep -n 'preempt|void|39.33|55.8|10,439|3,170' offkaggle/REGIME_WAVE_STATUS.md",
  "cat offkaggle/results/20260911-111640-regime-keith_ctx16/{summary.txt,arm_env.json,git_status.txt}; ls the wave dir",
  "for d in offkaggle/results/*/; grep vllm:num_preemptions_total and vllm:request_success_total in metrics_before.prom/metrics_after.prom (all waves)",
  "python3: parse requests_shim.jsonl for ctx16/fx/ctx12/nr16/09-02 base - status counts, calls/game, prompt/completion stats, n_messages distribution, completions > history budget, error rows and their timestamps, e2e by 15-min bucket",
  "python3: results.json sums (levels, actions, zero-level) for ctx16, fx, ctx12, 09-02 base; per-game pairing of ctx16 vs both 32k yield-60 draws",
  "python3: metrics_after minus metrics_before deltas per wave - e2e/queue/inference/prefill/decode sums and counts, gen/prompt tokens, iteration_tokens, spec_decode drafts/accepted; cache_config_info (kv_cache_size_tokens=105202, block_size 8)",
  "python3: sequence-size (prompt+completion) distribution vs 105202/8 for fx, ctx16, ctx12",
  "python3: max completion_tokens across every */requests_shim.jsonl on disk before 09-11",
  "grep -E '^  TURNS|^  CADENCE|^  VLLM|^  SCORE|^  ARM KNOB|^  geometry' in summary.txt of 09-02 keith, 09-09 keith, fx, budget, ctx12, ctx16, nr16, yield900 x2",
  "grep ^history_messages: and ^message: across transcripts/*.txt for fx, ctx16, ctx12",
  "python3 regex classification of every <parameter=code> block in transcripts (action() vs print-only vs other; consecutive verbatim repeats) for 09-02 base, fx, ctx16, ctx12; decomposition into P(act) x actions per acting call",
  "python3 per-turn compact viewer (step, action count, history_messages, acted?, THINKING/ASSISTANT excerpt, code excerpt) on ctx16 transcripts tr87 (lines 3105-3310), sp80 turns 30-40, lp85 turns 40-50",
  "sed -n 1360,1400p and 1562,1700p offkaggle/run_regime_wave.py (wall_reads VOID rule, telemetry_for_game, aggregate_telemetry); grep -n yielded/outcome parse_transcript",
  "ls logs | grep ctx16; grep -c preempt logs/ctx16_20260911.log",
  "grep -rn 39.33 docs/*.md docs/research-2026-09-*/*.md memory/*.md; sed -n 618,634p docs/HANDOFF-2026-08-29-packs-in-flight.md"
 ]
}
```

---

## agent-6

```json
{
 "claim": "keith_ctx12 (offkaggle/results/20260911-075839-regime-keith_ctx12): window 12,288 is catastrophic — mechanism gate PASSED (median prompt 20,061 -> 6,674; calls/game 55.8 -> 108.4), 12 levels = -11.68 sd vs pooled base 39.33 (sd 2.34), 15/25 zero-level games, actions/call 2.76 -> 0.27; mechanism = history budget 11,264 - 3,170 system - ~745 user = ~7,349 tokens while single completions reach 10,439 so the trimmer thrashes. Write-up reports 'preemptions 0' and says nothing about a mid-wave server restart.",
 "verdict": "CONFIRMED_WITH_CORRECTIONS",
 "recomputed": [
  {
   "quantity": "levels on 25 games",
   "claimed": "12",
   "recomputed": "12 (sum of levels_completed; 25 games all state gave_up)",
   "source": "results.json ['games'] via python3",
   "matches": true
  },
  {
   "quantity": "zero-level games",
   "claimed": "15 (base 2)",
   "recomputed": "15; base draw keith0902 = 2, but the 8 valid same-regime draws range 1..7 (mean 3.25)",
   "source": "results.json of ctx12 and of the 8 base-regime waves",
   "matches": true
  },
  {
   "quantity": "z-score vs pooled base",
   "claimed": "-11.68 sd",
   "recomputed": "(12-39.33)/2.34 = -11.68; could NOT reproduce sd 2.34 from any 6-subset of the wave totals on disk (nearest subsets give sd 2.8-3.5); pooling all 8 valid conc-28/7920s/32k draws [36,34,40,41,40,41,41,35] gives mean 38.5 sd 2.98 -> z = -8.9; at sd 6 still -4.6",
   "source": "summary.txt of every offkaggle/results/**/ wave + python3 itertools",
   "matches": true
  },
  {
   "quantity": "paired per-game deficit (not in write-up)",
   "claimed": "n/a",
   "recomputed": "ctx12 - mean(8 base draws) = -1.06 lv/game, sd 0.74, se 0.15, t = -7.1; ctx12 >= base mean in only 2 of 25 games",
   "source": "results.json per game, ctx12 vs 8 base-regime waves",
   "matches": true
  },
  {
   "quantity": "calls/game",
   "claimed": "55.8 -> 108.4 (1.94x)",
   "recomputed": "2,709 status-200 shim rows / 25 = 108.36; base 1,395/25 = 55.8; ratio 1.943. Runner counts only successful calls (44 error rows excluded; 2,753 total rows)",
   "source": "requests_shim.jsonl (ctx12 and 20260902-232148-regime-keith)",
   "matches": true
  },
  {
   "quantity": "median prompt tokens",
   "claimed": "20,061 -> 6,674",
   "recomputed": "ctx12 median 6,674 (mean 6,581, max 10,827, min 4,042). Base '20,061' is the summary's MEAN prompt tok/call; the base MEDIAN is 21,421. Mislabeled row mixing mean (base) with median (arm); direction unchanged",
   "source": "requests_shim.jsonl prompt_tokens, status==200; base summary.txt line 'prompt tok/call 20061'",
   "matches": false
  },
  {
   "quantity": "actions/game and actions/call",
   "claimed": "154 -> 29; 2.76 -> 0.27",
   "recomputed": "722/25 = 28.9; 722/2,709 = 0.267; base 3,856/25 = 154.2, 3,856/1,395 = 2.764",
   "source": "results.json actions + shim call counts",
   "matches": true
  },
  {
   "quantity": "completion tokens",
   "claimed": "max 10,439, mean 1,044",
   "recomputed": "max 10,439, mean 1,044.3, median 773, p90 2,251, p99 4,589; only 2 of 2,709 completions exceed 7,349 and 18 exceed 5,000",
   "source": "requests_shim.jsonl completion_tokens",
   "matches": true
  },
  {
   "quantity": "context budget 11,264",
   "claimed": "11,264",
   "recomputed": "12,288 - reply_reserve 512 (MAX_OUTPUT=0 -> 512) - _REQUEST_SAFETY_MARGIN_TOKENS 512 = 11,264",
   "source": "tool_agent.py:937-948,149; summary.txt harness-reported context_budget_tokens 11264",
   "matches": true
  },
  {
   "quantity": "system prompt = 3,170 tokens",
   "claimed": "3,170",
   "recomputed": "Not directly verifiable (Qwen tokenizer not cached locally; no source for 3,170 anywhere in the repo before this write-up). Consistent with measured usage: first call (n_messages=2) = 4,052 real tokens = system + 3,497-char user text + tools + template + 64 vision tokens, leaving ~818 for user+tools+template if system is 3,170. System prompt is 12,358 chars (12,679 rendered)",
   "source": "shim first-call prompt_tokens 4,052..4,062; _build_system_prompt via .venv import; prompts/*.log",
   "matches": true
  },
  {
   "quantity": "history budget ~7,349 tokens",
   "claimed": "11,264 - 3,170 - 745 = 7,349",
   "recomputed": "Arithmetic correct but units wrong. The trimmer compares its budget against _estimate_tokens = len(json.dumps(payload))/3, which counts the base64 grid image and reasoning text. Reconstructed first-call payload estimate = 6,516 'estimator tokens' vs 4,052 real (real/est = 0.62). History room = 11,264 - 6,516 = 4,748 estimator tokens ~ 2,950 REAL tokens, i.e. ~1.05 typical retained turn blocks (assistant reasoning 2,489 chars + code + <=4,096-char tool result + next user text + image = 4,508 est). At 32k the same arithmetic gives ~5.6 turns, at 16k ~2.0",
   "source": "tool_agent.py:462-467 (_estimate_tokens), :1685-1688 (trim loop), :1146-1156 (image in user msg); .venv python reconstruction",
   "matches": false
  },
  {
   "quantity": "mechanism: 'one long-reasoning turn (10,439) does not fit'",
   "claimed": "trimmer thrashes because completions reach 10,439",
   "recomputed": "Rare-event framing is wrong: >7,349-token completions occur in 2/2,709 calls. The thrash is routine: n_messages distribution = 2:461, 4:282, 5:1,273, 6:14, 7:519, 8:89, 9:53, 10:15, 11:3 -> median 5 messages (one prior turn); 436 mid-game calls (16%) carry ZERO history (system+user only, prompt 4,042-4,680 tokens). Base n_messages typically 17-32",
   "source": "requests_shim.jsonl n_messages; base 20260902 shim",
   "matches": false
  },
  {
   "quantity": "what drives actions/call 0.27",
   "claimed": "trimmer thrash",
   "recomputed": "Calls whose python code contains action(): 12.0% (325/2,709) vs base 55.8% (778/1,395); refusals 0, no-tool-call 7 (0.3%) -> model is not refusing tools, it runs analysis-only python. Acting share by n_messages: 2 -> 0.0% (461 calls), 4 -> 6%, 5 -> 14%, 7 -> 15%, 8 -> 30%, 10 -> 67%; base with >=6 messages 47-71%. Acting share by call index within game: 22% -> 17% -> 10% -> 8% -> 6% -> 4%. Actions per acting call 2.2 vs base 4.96. Consistent with amnesia (each turn restarts exploration), which is a stronger form of the write-up's thrash claim",
   "source": "run_regime_wave.parse_transcript acts_code joined to shim rows by order (0 count mismatches)",
   "matches": true
  },
  {
   "quantity": "server restart",
   "claimed": "not mentioned; 'preemptions 0'",
   "recomputed": "Confirmed restart: process_start_time_seconds 1789106349 (before) vs 1789111341 (after) = new vLLM process at +79.7 min; all 25 games received a Modal 500 'Server has lost track of input' ending at +76.6 min; then 25 redirected (303 x2-3) calls waited 398-473 s; zero call starts in minutes 77-82. Wall lost: mean 464 s/game (409-524), total 11,591 s = 5.9% of the 7,920 s clock, ~6.3 calls/game (~159 total). The redirected calls returned 200 with completions (mean 982 tok), and request_error only reverts history to the previous turn (preserve_history=False), so no state wipe. All summary 'VLLM' numbers (requests 1,080, preemptions 0, e2e 67.5 s, queue 53.6 s, MTP 59.7%) cover only the post-restart process = 1,080 of 2,709 calls (40%); pre-restart preemptions unknown. Cause of the container death is not in any local file (not the 6 h cap: container was 80 min old)",
   "source": "metrics_before/after.prom; requests_shim.jsonl; logs/ctx12_20260911.log:334-360; tool_agent.py:1979-1996",
   "matches": false
  },
  {
   "quantity": "levels before vs after restart",
   "claimed": "n/a",
   "recomputed": "11 of 12 clears before the restart, ALL by 08:45 local (+43 min); zero clears from +43 to +76.6 min (pre-restart); 1 clear after (ka59 at +115 min). Log: levels=11 at +44 min and still 11 at +76 min. Base keith_fx by comparison: 18 levels at +42 min, 25 at +74, 34 at +130",
   "source": "artifacts/*_events.jsonl level_completed + transcript headers; logs/ctx12_20260911.log; logs/fx_wave_20260910.log",
   "matches": true
  },
  {
   "quantity": "pre-registered VOID rule",
   "claimed": "wave read as DEAD",
   "recomputed": "summary.txt tags all 25 runs VOID (request_errors 2-4 each). 25 of the 44 errors are genuine server 500s (container death), 18 are the usual end-of-clock ReadTimeouts at +131-132 min, 1 is a 502. The notebook's own rule (STATUS:470,622: request errors > 0 -> VOID, redraw once) is not addressed in the write-up",
   "source": "summary.txt WALL line; REGIME_WAVE_STATUS.md:470,622,630",
   "matches": false
  }
 ],
 "errors_found": [
  "Mislabeled comparison: base '20,061' is the MEAN prompt tokens/call (summary 'prompt tok/call 20061'); the base MEDIAN is 21,421. The arm's 6,674 is a true median. Direction and magnitude of the drop are unaffected.",
  "Token arithmetic is in the wrong units: the stock trimmer (tool_agent.py:462-467, :1685) budgets against len(json.dumps(payload))/3, which counts the base64 grid image (~2,200 chars/user turn) and reasoning text. Reconstructed first-call block = 6,516 estimator tokens (real 4,052; ratio 0.62), so history room is ~4,748 estimator tokens ~ 2,950 real tokens (~1 typical prior turn), not 7,349. Same error propagates to the ctx16 table (27,829 / 11,445 / 7,349).",
  "Mechanism over-attributed to rare long completions: only 2 of 2,709 completions exceed 7,349 tokens. The thrash is routine: median 5 messages per call, 436 mid-game calls (16%) sent with zero history, and acting share is 0% at n_messages=2, 6-15% at 4-7, 21-67% at 8-11.",
  "Undisclosed instrument event: a vLLM container death at +76.6 min (25 simultaneous Modal 500s, 398 s gap, new process_start_time), costing ~464 s/game (5.9% of clock, ~6 calls/game). 'preemptions 0' and every VLLM summary number cover only the post-restart process (1,080 of 2,709 calls). Cause not recoverable from local files.",
  "The notebook's own pre-registered VOID rule (request errors > 0) fires on all 25 runs and 25 of the 44 errors are genuine server failures, not end-of-clock timeouts; the write-up neither invokes nor waives it.",
  "'zero-level games (base 2)' cites the single best-case base draw; the 8 valid same-regime draws have 1-7 zero-level games (mean 3.25). 15 remains far outside that range.",
  "sd 2.34 for the six-draw base could not be reproduced from any 6-subset of the wave totals on disk (nearest 2.8-3.5); the 8 valid same-regime draws give sd 2.98. No 6-draw list is written anywhere in the repo."
 ],
 "confounds": [
  "Mid-wave container restart (+76.6 to +83.3 min): quantified at 5.9% of wall clock and ~6 calls/game; not the cause of the collapse because 11/12 clears happened by +43 min and none in the 33 pre-restart minutes after that (base clears ~7 levels in that same window).",
  "Base sd of 2.34 is unreproducible and likely understated (8-draw sd 2.98); the verdict is insensitive: z stays below -4.4 even at sd 6, and the paired per-game deficit is -1.06 lv/game (t = -7.1, 23/25 games below base).",
  "The experiment did not test 'a 12k window' in the intended sense: because the trimmer estimates tokens at chars/3 including base64 images, the effective retained history was ~1 turn (16% of calls with none). The result therefore measures near-total history removal, not a 3x capacity reduction; the smooth 'capacity' curve drawn in the ctx16 section inherits the same unit error.",
  "The 60 s yield with a 72 s mean e2e makes every turn ~1 call (calls/turn 1.26), so a single analysis-only call ends the turn ('yielded' 84.4%). Same structure exists in the base (e2e 142 s, 1.02 calls/turn, 43% yielded), so it is not a confound for the comparison, but it amplifies the cost of any analysis-only call."
 ],
 "does_conclusion_survive": "YES. The headline conclusion — window 12,288 under the stock trimmer is catastrophic (12 levels, 15/25 zero-level games, actions/call 0.27, mechanism gate passed with 108.4 calls/game) — is confirmed from raw files and survives every stress test: a 5.9%-of-clock server restart that post-dates the collapse (11/12 clears by +43 min, then a 33-minute pre-restart drought), an unreproducible base sd (z <= -4.4 even at sd 6), and a paired per-game test (23/25 games below base, t = -7.1). What does not survive is the stated MECHANISM: the history budget is ~2,950 real tokens (~1 turn), not 7,349, because the trimmer budgets chars/3 of the JSON payload including the base64 image, and the collapse is routine amnesia (16% of mid-game calls with zero history, 0% acting there) rather than rare 10,439-token completions. The write-up should also disclose the restart, note that all VLLM-line numbers are post-restart only, and address its own VOID rule.",
 "what_would_change_verdict": "A base-regime draw with a comparable mid-wave restart that also collapsed to ~12 levels would reassign the effect to the instrument (none exists: the 8 valid base draws span 34-41 with 0-1 restarts of this kind visible in their shims). A demonstration that the same-regime base sd is >= 8 levels would drop z below -3.5 but not rescue the arm given the paired per-game result. A rerun of ctx12 with the trimmer estimating real tokens (or images excluded from the estimate) that recovered actions/call >= 2 would show the 'capacity is load-bearing' inference was really 'the stock estimator strips history at small windows'; that would change the mechanism attribution, not the verdict on this arm as shipped.",
 "commands_run": [
  "sed -n 1330,1410p offkaggle/REGIME_WAVE_STATUS.md; ls -la offkaggle/results/20260911-075839-regime-keith_ctx12/",
  "cat summary.txt arm_env.json git_status.txt; head requests_shim.jsonl; ls prompts transcripts artifacts solver_analysis",
  "python3: json.load(results.json) -> sum levels/actions/zero-level, per-game table, elapsed/started_at",
  "python3: requests_shim.jsonl -> status/finish counters, prompt/completion token stats, n_messages/tool_calls distributions, per-game call counts, inter-call gaps, per-minute starts, error timeline, redirected rows, calls before/after gap",
  "grep process_start_time_seconds|vllm:request_success_total|num_preemptions_total metrics_before.prom metrics_after.prom",
  "sed -n 334,380p logs/ctx12_20260911.log; grep '[regime] +' logs/ctx12_20260911.log; grep '[regime] +' logs/fx_wave_20260910.log",
  "grep -n '39.33|2.34|3,170|10,439|20,061|55.8|2.76|VOID' offkaggle/REGIME_WAVE_STATUS.md; sed -n 605-632,978-984,1436-1470",
  "python3 glob offkaggle/results/**/summary.txt -> table of arm/geometry/levels/calls/actions/actions-per-call/prompt tok + recomputed median prompt_tokens from each requests_shim.jsonl",
  "python3: events.jsonl level_completed + transcript '--- analysis_step | action | HH:MM:SS' headers -> clear timestamps",
  "grep/sed offkaggle/run_regime_wave.py (outcome definitions 1275-1300, telemetry_for_game 1562-1610)",
  "grep/sed scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py (149, 151, 462-467, 930-955, 1146-1156, 1596-1622, 1653-1690, 1766-1800, 1975-2030)",
  "python3: itertools 6-subsets of wave totals -> which give mean 39.33 and their sd; z at sd 2.34/2.9/4/5/6/8",
  "python3: run_regime_wave.parse_transcript per call (acts_code, refused, tool_call_count) joined to shim rows by order for ctx12 and base 20260902 -> acting share by n_messages, wave minute, call index",
  "python3: paired per-game levels ctx12 vs 8 base-regime waves' results.json -> mean diff, sd, se, t, sign count",
  "python3: prompts/*.log section char sizes; zero-history mid-game calls; restart wall lost per game; redirected-call completions",
  ".venv/bin/python: from inference.agent import tool_agent; _build_system_prompt, _estimate_tokens on reconstructed first-call payload and a typical turn block; turns that fit at 12k/16k/32k"
 ]
}
```

---

## agent-7

```json
{
 "claim": "docs/research-2026-09-10/R-what-our-agent-actually-does.md: on the M0 corpus (2,047 tool calls made on levels the run went on to clear) the agent forward-simulates a next state 0 times (0.0%); 100% ascii, 53.1% segmentation, 27.5% diff, 2.2% search, 1.0% graph, 0.2% verify, 56.7% action(); median 9 lines; median action list 1; transitions in 3.2%; 43.3% no action; therefore \"our agent is a reactive perceive-and-act loop\" that \"never models live\".",
 "verdict": "PARTLY_WRONG",
 "recomputed": [
  {
   "quantity": "corpus size / reproducibility",
   "claimed": "2,047 records, 247 runs, 11 waves",
   "recomputed": "2,047 records (1,373 train + 674 holdout), 247 runs, 11,443 calls seen, 2,062 kept, 15 dups — rebuilt byte-for-byte from the committed builder with result dirs stamped <= 2026-09-09 21:10Z. BUT: 19 result directories (7 x 25-game waves + 12 x 6-game kill/wall runs), not 11 waves; 9 different harness arms (keith 578 kept calls, yield900 474, carry 328, probe 212, wsd 145, ws 136, hypo 65, up8 65, evid 59) — 72% of the corpus is NOT the shipped base arm. The corpus files themselves are absent from the checkout (scratchpad/modelaxis/ does not exist); only the manifest is committed. No classifier exists anywhere (grep src/modelaxis, scratchpad; commit e5225af adds only the .md).",
   "source": "python3 scratchpad/build_corpus_cut.py (copy of /Users/ahmed/Documents/ArcAGI3/src/modelaxis/build_corpus.py with CUTOFF_STAMP=20260909-211000) vs /Users/ahmed/Documents/ArcAGI3/docs/research-2026-09-09/corpus-manifest-m0.json; scratchpad/extract_all.py wave listing",
   "matches": false
  },
  {
   "quantity": "perception: ascii slicing / printing",
   "claimed": "2,047 (100%)",
   "recomputed": "`.ascii` appears in 1,298 (63.4%); `print(` in 2,047 (100%); `.ascii|segmentation` in 2,028 (99.1%). The 100% is only reachable by defining the category as 'prints anything' — vacuous.",
   "source": "regex over the 2,047 rebuilt records (scratchpad/corpus_cut/*.jsonl)",
   "matches": false
  },
  {
   "quantity": "segmentation objects",
   "claimed": "1,086 (53.1%)",
   "recomputed": "1,086 (53.1%)",
   "source": "regex `segmentation` over 2,047 records",
   "matches": true
  },
  {
   "quantity": "diff previous vs current",
   "claimed": "563 (27.5%)",
   "recomputed": "565 (27.6%) with previous_frame|before_frame|history[-2]|last_transition; previous_frame alone 411 (20.1%)",
   "source": "regex over 2,047 records",
   "matches": true
  },
  {
   "quantity": "search: bfs/dfs/queue/shortest path",
   "claimed": "46 (2.2%)",
   "recomputed": "46 (2.2%) exact — but at least 9 of the 46 are flood-fill component labelling (ka59 steps 11-18, `deque` used to group pixels), not planning search; `bfs|dfs` literally = 11 (0.5%)",
   "source": "regex bfs|dfs|deque|heapq|shortest|popleft|visited|frontier over 2,047; hand read of the 19 search-only hits",
   "matches": true
  },
  {
   "quantity": "search: builds a graph or map",
   "claimed": "21 (1.0%)",
   "recomputed": "could not reproduce: `\\bgraph\\b` = 0; graph|adjacen|networkx = 127 (6.2%) because the segmentation dict has an `adjacency_list` field. Definition unknown.",
   "source": "regex over 2,047 records",
   "matches": false
  },
  {
   "quantity": "forward-simulates a next state",
   "claimed": "0 (0.0%)",
   "recomputed": "REFUTED. Hand-confirmed >= 30 of 2,047 (1.5%): strong regex tier (def step/move/apply/slide/sim… or simulate/predict/next_state) flags 27, of which 16 are genuine on inspection (e.g. m0r0 step 9 `def step(s,a)` + BFS over the model; tu93 step 6 `apply()` traces a 14-move path and checks 'ends at goal'; tu93 step 7 `exp=apply(pos,a); r=action([a]); got==exp`; lp85 steps 9/11/14 cycle model checked against the observed diff then BFS); delta/copy tier flags 70 more, 11 genuine (+2 borderline: su15 predicted-position check, lf52 legal-move enumeration); search-only tier adds 3 (lp85 steps 16/18/20). The only regex that gives ~0 is the bare keyword `simulat|predict|next_state|expected|would` = 2 (0.1%), which is almost certainly what produced the 0 — it misses every hand-written transition function. Games with forward simulation: tu93, m0r0, lp85, lf52, dc22.",
   "source": "scratchpad/classify.py + hand labelling of all 27+70+19 flagged records (tool-results bho9v9agr.txt, bo5ezxqtc.txt, bg7z3dhyf.txt)",
   "matches": false
  },
  {
   "quantity": "verifies a prediction against reality",
   "claimed": "5 (0.2%)",
   "recomputed": "7 regex hits (0.3%); 6 genuine model-vs-observation checks (lp85 steps 7/9/11/14 'model mismatch', m0r0 steps 8/10 backtrack assert / 'mismatch'); tr87 assert is not one. Roughly consistent — but 5 verifications of predictions is internally inconsistent with 0 predictions.",
   "source": "regex + hand read of the 7 hits",
   "matches": true
  },
  {
   "quantity": "fires action()",
   "claimed": "1,161 (56.7%); 43.3% fire no action",
   "recomputed": "1,161 (56.7%); 886 no-action (43.3%)",
   "source": "regex (?<![\\w.])action\\s*\\( over 2,047",
   "matches": true
  },
  {
   "quantity": "median code length",
   "claimed": "9 lines",
   "recomputed": "9 (total and non-blank)",
   "source": "statistics.median over 2,047",
   "matches": true
  },
  {
   "quantity": "median action-list length / batches >= 5",
   "claimed": "1 / 0.3%",
   "recomputed": "median 1 (982 of 1,032 parsed literal call sites have length 1; 502 sites unparseable — variables/loops); calls containing a literal batch >= 5: 6/2,047 = 0.29%",
   "source": "safe AST eval of action(...) arguments incl. list * n and + expressions",
   "matches": true
  },
  {
   "quantity": "transitions touched",
   "claimed": "3.2%",
   "recomputed": "66 (3.2%) for the word `transitions`; 171 (8.4%) if `last_transition` (the same affordance) is counted",
   "source": "regex over 2,047",
   "matches": true
  },
  {
   "quantity": "30-call random hand sample",
   "claimed": "(implied) no forward simulation",
   "recomputed": "0/30 forward-simulate (seed 20260911). Consistent with a ~1.5% rate (expected 0.45 hits), so the sample cannot distinguish 0% from 1.5%; the targeted sweep above does.",
   "source": "scratchpad/sample30.txt hand read",
   "matches": true
  },
  {
   "quantity": "selection: first-of-turn only",
   "claimed": "(not stated in the doc; builder docstring says 'first-call-of-turn records only')",
   "recomputed": "True calls-per-turn (turn header = last header preceding each META block): 8,383 turns, 13,438 calls; 35.1% of turns have > 1 call and 62.5% of ALL calls sit in multi-call turns; base arm 1.39 calls/turn, yield900 1.78. The 4,573 dropped continuation calls have 3x the strong-tier forward-sim rate (129 = 2.8% vs 1.3%), 3x search (6.8% vs 2.2%), broad tier 15.5% vs 4.7%; a hand sample of 15 flagged continuation calls found 11 genuine transition-function + BFS programs (m0r0, sk48 `def sim(state, act)`, wa30 Sokoban `step()`, cd82, tu93, ft09 hypothesis sim vs actual).",
   "source": "scratchpad/extract_all.py, classify.py, per-turn count script; tool-results bg7z3dhyf.txt",
   "matches": false
  },
  {
   "quantity": "builder step/level mapping",
   "claimed": "records carry analysis_step and level; wall-level calls dropped (4,681)",
   "recomputed": "BUG: build_corpus.py line `step = steps[idx]` maps the idx-th CALL to the idx-th TURN HEADER, wrong from the first multi-call turn onward. Only 277 of 6,731 first-of-turn calls get the correct step; ~88% of corpus `analysis_step`/`level` fields are wrong. Because level is monotone in step the error only raises the level, so 1,288 genuine cleared-level first-of-turn calls were dropped as 'wall level'. Correct population = 3,350 (3,335 dedup), not 2,047. On the corrected set: strong-tier forward-sim 57 (1.7%), search 94 (2.8%), verify 21 (0.6%), action 59.5%, median 10 lines; the wrongly dropped 1,288 are richer in modelling (strong tier 2.3%, verify 1.1%).",
   "source": "scratchpad/remap.py against transcripts (e.g. offkaggle/results/20260908T2110-keith_carry75/20260908-232712-regime-keith_carry/transcripts/tu93-0768757b_p0.txt shows header at line 2 then 4 META blocks before the next header at line 432)",
   "matches": false
  },
  {
   "quantity": "events level semantics",
   "claimed": "level in events is 1-based and is the level the run is ON",
   "recomputed": "max event level = levels_completed + 1 for all 247 runs (238 gave_up, 4 cancelled, 3 crashed, 2 playing) — the wall filter itself is sound once the step mapping is fixed",
   "source": "events jsonl vs results.json across all corpus runs",
   "matches": true
  }
 ],
 "errors_found": [
  "Headline number is false: at least 30 of the 2,047 calls (1.5%) forward-simulate a next state with a hand-written transition function (tu93, m0r0, lp85, lf52, dc22), several of which then verify the prediction against the real frame (tu93 step 7 `exp=apply(pos,a)` vs `got`; lp85 'model mismatch' checks). The 0 is consistent only with a bare keyword regex (simulate/predict/next_state/expected/would -> 2 hits), which cannot see BFS over a hand-coded `step()`.",
  "Internal inconsistency: the table reports 5 calls that 'verify a prediction against reality' and 0 that make a prediction.",
  "The classifier is not committed and the corpus files are not in the checkout (scratchpad/modelaxis/ missing); only the manifest exists. The corpus IS reproducible from the committed builder (identical manifest), the classification is not.",
  "Builder bug: `steps[idx]` in src/modelaxis/build_corpus.py aligns calls to turn headers by index, which is wrong once any turn has >1 call (35% of turns do). Only 277/6,731 first-of-turn calls carry the right step; 1,288 cleared-level calls were dropped as wall-level; the true population is 3,350 not 2,047, and every record's analysis_step/level field is unreliable.",
  "Selection excludes all non-first calls of a turn (4,573 with code = 62.5% of calls live in multi-call turns). Those calls carry ~3x the forward-simulation and search rates, so the corpus was built from the sub-population least likely to contain modelling.",
  "'100% ascii slicing/printing' is vacuous: `.ascii` is used in 63.4%; only `print(` reaches 100%.",
  "'11 waves' is wrong: 19 result directories, 9 distinct harness arms; 72% of the corpus comes from non-base arms (yield900, carry, probe, ws, wsd, hypo, up8, evid), so it does not describe the shipped base agent in isolation.",
  "'search 2.2%' counts flood-fill component labelling (>= 9 of 46, ka59) as search; literal bfs|dfs is 0.5%.",
  "'builds a graph 21 (1.0%)' could not be reproduced under any reasonable regex (0 for `graph`, 127 with adjacency/networkx); definition unknown."
 ],
 "confounds": [
  "First-call-of-turn selection: the harness ends a turn on the first acting call, so first-of-turn calls are structurally the look-at-the-board calls; deliberation after a tool result (where simulation happens) is by construction in the excluded continuation calls.",
  "Cleared-level selection: modelling correlates with the games where it is tractable (tu93 slide puzzle, m0r0 twin-avatar maze, lp85 cycles, lf52 peg solitaire) and those are cleared; the write-up reads 'successful calls' as 'the agent's best behaviour', but the selection is by game type as much as by call quality.",
  "Arm mixing: 9 arms including ws/wsd (workspace + verifier), probe, carry, hypo, evid, up8 — arms whose prompts explicitly push modelling — are pooled with the base; base-only kept calls: 578, of which 27 broad-tier / 23 search hits.",
  "Step-mapping bug shifts levels upward, dropping later-in-run calls preferentially; later-in-run calls are the ones with a model already built (2.3% vs 1.3% strong-tier rate).",
  "Regex classification with no committed definition: whether BFS over a hand-written `step()` counts as forward simulation is definitional; under any definition that includes computing a hypothetical next state from an action, the count is > 0."
 ],
 "does_conclusion_survive": "Partly. The weak form survives: the live loop is predominantly reactive — roughly 1.5% of first-of-turn cleared-level calls (and ~3-15% of mid-turn calls depending on tier) forward-simulate, median 9-10 lines, one action per acting call, `transitions` rarely touched. The strong form in section 3 does not survive: 'forward-simulated a next state exactly zero times', 'never models live', 'the missing thing is an operation not in the live repertoire' are all contradicted by >= 30 hand-verified calls in the write-up's own 2,047 (and more in the 1,288 it wrongly dropped and the 4,573 it excluded), where the stock brain writes a transition function, searches over it, and in several cases checks the prediction against the real frame — unprompted, on tu93/m0r0/lp85/lf52/dc22. The right statement is 'the agent models live only on a handful of grid-puzzle games where a compact rule is obvious, and does so in a few percent of calls'; that is a frequency/coverage gap, not an absent operation, and the downstream inferences drawn from 'zero' (A2 failed because verification is not in the repertoire; keith_fx failed because it will not act on a model at all) are not established by this corpus.",
 "what_would_change_verdict": "Would move to CONFIRMED_WITH_CORRECTIONS if the author's classifier is produced and its 'forward-simulate' definition explicitly excludes search over a hand-written transition function AND the doc is reworded to that narrow definition (the numbers would then be reproducible but the 'never models' conclusion would still have to be dropped). Would move to REFUTED if, after fixing the steps[idx] bug and including mid-turn calls, the corrected corpus shows forward simulation in >= 5% of calls or on the majority of cleared games — the continuation-call regex rates (2.8% strong tier, 15.5% broad tier) make that plausible and it should be measured with the fixed builder before any further claim is built on this document.",
 "commands_run": [
  "cat docs/research-2026-09-10/R-what-our-agent-actually-does.md; ls docs/research-2026-09-10 docs/research-2026-09-09",
  "python3 json load of docs/research-2026-09-09/corpus-manifest-m0.json; ls src/modelaxis; git show --stat e5225af; grep -rn 'forward.sim|simulat' src/modelaxis scratchpad",
  "sed -n 1,400p src/modelaxis/build_corpus.py",
  "ls scratchpad/modelaxis (absent); ls scratchpad | grep -i model|corpus; sed -n 990,1110p offkaggle/REGIME_WAVE_STATUS.md",
  "for d in offkaggle/results/*/transcripts offkaggle/results/*/*/transcripts: count *.txt and stat results.json mtime",
  "python3 scratchpad/build_corpus_cut.py --out scratchpad/corpus_cut with CUTOFF_STAMP=20260909-211000 (manifest identical to committed one); same with no cutoff",
  "python3 scratchpad/extract_all.py scratchpad/all_calls.json (all 13,438 calls tagged kept/continuation/wall_level/zero_level_run; 19 wave dirs)",
  "python3 scratchpad/classify.py (tiered regex classifier over all tags + per-arm breakdown)",
  "python3 recount on deduped 2,047: print/.ascii/segmentation/previous_frame/transitions/bfs/graph/action/simulat|predict regexes; median lines",
  "python3 dump of fs_funcs|fs_keyword flagged kept calls (27) and fs_delta|fs_copy flagged (70) and search-only (19) for hand labelling",
  "python3 30-record random sample (seed 20260911) for hand labelling",
  "python3 safe-AST action() argument length recount; verify-regex hit listing",
  "python3 events-level check: max event level vs levels_completed across all 247 runs",
  "python3 scratchpad/remap.py scratchpad/remap.json (true turn = last header before each META block; compare with builder steps[idx]; recount kept)",
  "grep -n turn headers/section markers in offkaggle/results/20260908T2110-keith_carry75/.../transcripts/tu93-0768757b_p0.txt",
  "python3 classification of builder-kept vs TRUE-kept vs wrongly-dropped sets",
  "python3 true calls-per-turn histogram and per-arm calls/turn"
 ]
}
```

---

## agent-8

```json
{
 "claim": "Live ledger: base draw #4 (sub 56133282, 09-10) scored 2.73 on Kaggle but the ledger still lists it pending; ledger header stale (generated 2026-08-04 / 45 submissions); yield900 draws 4.31/2.45/4.01 mean 3.59 -> DEAD under the pre-registered 3-draw rule (>=4.0 step, <=3.6 dead); base family pooled 3.25/2.58/4.31/2.45/4.01 n=5 mean 3.32 sd 0.83; with 2.73 the family is n=6; pooling is justified by an off-Kaggle null (118 = 118 levels on 75 vs 75 runs).",
 "verdict": "CONFIRMED_WITH_CORRECTIONS",
 "recomputed": [
  {
   "quantity": "Sub 56133282 (base draw #4) Kaggle status/score",
   "claimed": "scored 2.73 on Kaggle; ledger says pending, no score, no verdict",
   "recomputed": "Kaggle CLI row: 56133282, 2026-09-10 00:11:04, SubmissionStatus.COMPLETE, publicScore 2.73. Ledger row: status 'pending', public_score '', no 'verdict' key (docs/submission-ledger.json rows 1-10; uncommitted working copy, last git touch f43f8c2 09-09).",
   "source": "kaggle competitions submissions -c arc-prize-2026-arc-agi-3 -v --page-size 200 ; python3 json load of docs/submission-ledger.json",
   "matches": true
  },
  {
   "quantity": "Ledger header staleness",
   "claimed": "generated 2026-08-04 / 45 submissions",
   "recomputed": "generated_utc 2026-08-04T17:42:32+00:00, counts.submissions 45, but the file holds 85 rows (06-15 .. 09-10 per Kaggle span 08-06..09-10 in ledger dates); Kaggle returns 86 rows (75 COMPLETE, 11 ERROR, 74 with a score).",
   "source": "docs/submission-ledger.json lines 1-17; kaggle CLI --page-size 200 (86 rows)",
   "matches": true
  },
  {
   "quantity": "yield900 3-draw mean",
   "claimed": "4.31/2.45/4.01 mean 3.59 -> DEAD (<=3.6)",
   "recomputed": "mean 3.5900, sd 0.9986, SE 0.577, CV 0.278; draw-3 dead line x<=4.04 and step line x>=5.24 both reproduce. 3.59 clears the dead line by 0.01 (a 4.04 draw would have read 3.60; a 4.05 draw would have fallen in the 'two more draws' band).",
   "source": "python3 statistics on the three Kaggle CLI scores for subs 56080757/56088705/56111215",
   "matches": true
  },
  {
   "quantity": "Base family n=5 mean/sd/SE/CV",
   "claimed": "mean 3.32 sd 0.83 SE 0.37; CV on identical bytes 0.25",
   "recomputed": "n=5 mean 3.3200 sd 0.8315 SE 0.372 CV 0.2505. But 'identical bytes' is wrong: the family mixes two byte-sets. v4 alone (3.25/2.58/2.73): mean 2.853 sd 0.352 CV 0.123. yield900 alone: sd 0.999 CV 0.278. Pooled within-config sd = 0.75.",
   "source": "python3 statistics on Kaggle CLI scores",
   "matches": true
  },
  {
   "quantity": "Base family n=6 with 2.73",
   "claimed": "n=6; SE 0.37 -> 0.34",
   "recomputed": "n=6 mean 3.2217 sd 0.782 SE 0.319 CV 0.243 (the ledger's 0.34 assumed sd stayed 0.83). 2.73 is inside the pre-registered 2.0-4.7 band.",
   "source": "python3 statistics; band from ledger row 56133282 description",
   "matches": true
  },
  {
   "quantity": "Off-Kaggle null 118 = 118 on 75 vs 75 runs",
   "claimed": "yield900 118 levels = base 118 levels; paired diff 0.00 +- 0.11",
   "recomputed": "Inventory per docs/research-2026-09-08/R-loss-ledger-3-NOTES.md:4-10: NEW = Y1 41 + Y2 40 + Kaggle commit YK 37 = 118; BASE = M1 36 + M4 40 (keith_retry graft arm, 'retries not engaged') + Kaggle commit K 42 = 118. Rig files reproduce 41/40/36/40 (results.json). The two Kaggle commit runs (37, 42) are NOT in the repo: scratchpad/search/loss-ledger-3/cache.json does not exist. Rig-only paired recompute: Y1+Y2=81 vs M1+M4=76 -> +0.10 lv/game se 0.13 (6 better/7 worse/12 same); vs pure-stock live-geometry base M1 36 + keith_fx 35 = 71 -> +0.20 lv/game se 0.12 (10 better/2 worse/13 same).",
   "source": "python3 over offkaggle/results/20260906T2251-keith_yield900/*/results.json, 20260907T0702-keith_yield900-cb2/*/results.json, 20260902T2121-keith/*/results.json, 20260903T1014-keith_retry/*/results.json, 20260910-081906-regime-keith_fx/results.json; ls scratchpad/search/loss-ledger-3 (absent)",
   "matches": true
  },
  {
   "quantity": "Rig reads yield900 41/40 vs base 36",
   "claimed": "41 and 40 on two independent boots vs 36 base",
   "recomputed": "41 (20260906T2251, 3rd in session), 40 (20260907T0702-cb2, cold boot), void draw 31 (20260907T0106-cb, server died at ~+120 min); base 36 (20260902T2121). A later pure-stock live-geometry base draw read 35 (20260910-081906-regime-keith_fx, yield 60 s, conc 28, 7920 s). Same serving profile kv5-bf16-mtp3-c8-cg32 in all.",
   "source": "results.json sums and summary.txt headers in the listed dirs",
   "matches": true
  },
  {
   "quantity": "'Pooled six-draw base 39.33 sd 2.34' used as the rig comparator",
   "claimed": "base 39.33 (sd 2.34), bar >=48 step",
   "recomputed": "[41,40,37,36,40,42] -> mean 39.33 sd 2.34 reproduces, but three of the six are yield900 runs and one is the retry graft arm; only M1 36 (and the later FX 35) are stock base at live geometry. The 48 bar was set at notebook:299 as base 36 + 0.5 lv/game; against 39.33 it is +8.7 levels = +3.7 sd.",
   "source": "offkaggle/run_regime_wave.py:313-315; offkaggle/REGIME_WAVE_STATUS.md:299,687",
   "matches": true
  },
  {
   "quantity": "Detectable LB delta at per-draw sd 0.8 (d)",
   "claimed": "not stated numerically in the ledger; '+1.0 one-draw detectability' in notebook:585",
   "recomputed": "sd 0.80: 1 draw: 1.57 (vs a known base mean, 1.96*sd/sqrt(n)), 1.69 (two-sample vs n=6 base), 2.42 at 80% power; 3 draws: 0.91 / 1.11 / 1.58; 5 draws: 0.70 / 0.95 / 1.36. With sd 0.83 add ~0.04-0.09. The yield900-vs-v4 live gap (3.59-2.85 = +0.74, Welch t=1.21) is below every threshold; the notebook's '+1.0 one-draw' figure is optimistic (it needs 1.6-1.7).",
   "source": "python3 (normal-approximation thresholds)",
   "matches": false
  },
  {
   "quantity": "Rig-to-LB mapping across flown arms (e)",
   "claimed": "memory: 'public-25 does NOT predict LB (5.0 local -> 1.4 LB)'; yield900 rig 41/40 vs base 36; live 3.59 vs 2.85",
   "recomputed": "Same serving regime, stock loop: base rig 36,35 (1.42 lv/game; local 6.40, 6.88) -> live 2.85 (n=3 identical bytes); yield900 rig 41,40 (1.62; local 8.46, 7.90) -> live 3.59 (n=3). LB per (lv/game) = 2.0 and 2.2; LB/local-score = 0.445 and 0.44. The third point ('flight' rig 22 lv, 0.88, local 2.34 -> live 1.87) is NOT a valid mapping point: the rig flight ran on the keith serving profile with our analyzer caps (summary.txt: CONTEXT_WINDOW=24576 MAX_OUTPUT=4096) while the live flight draws used our slow no-MTP 22-seq profile (ledger 55902917). The '5.0 -> 1.4' figure is other teams' harnesses (docs/HANDOFF-2026-08-29-the-model-axis.md:60-66) and was already marked 'overstated' in docs/STRATEGY-2026-08-29-independent-review-path-to-6.md:40. The 0.40 calibration (docs/research-2026-09-08/R-score-arithmetic-0908.md:33-41) pairs local 7.44 (= the keith_probe graft wave, notebook:643) and 8.46 (Y1) against a live mean that pools both configs; using matched pairs gives ~0.44. No document ties the 48-level bar to an LB number; at ~2.0-2.2 LB per lv/game, 48 levels (1.92/game) maps to ~3.9-4.2 LB, coincidentally near the live 'mean >= 4.0' step line.",
   "source": "summary.txt/results.json of 20260902T2121-keith, 20260910-081906-regime-keith_fx, 20260906T2251-keith_yield900, 20260907T0702-keith_yield900-cb2, 20260902T2334-flight; Kaggle CLI scores; the three docs cited",
   "matches": false
  }
 ],
 "errors_found": [
  "Ledger is missing a whole submission: Kaggle sub 55543514 (2026-08-16 04:42, duck-38 brain-swap first flight, COMPLETE 1.29) has no row in docs/submission-ledger.json (86 Kaggle rows vs 85 ledger rows; it is only mentioned inside other rows' descriptions). Every other id reconciles.",
  "Ledger row 56111215 (yield900 draw 3) carries status 'pending' together with public_score 4.01 and a verdict; Kaggle says COMPLETE. Row 56133282 is 'pending'/'' while Kaggle says COMPLETE 2.73 (the claim's own point, confirmed). Row 55634118 is 'pending' in the ledger while Kaggle says COMPLETE with an empty score (08-20 runtime-exceeded).",
  "'CV on identical bytes 0.25' (ledger row 56133282) is the CV of a mixed two-config pool. On truly identical bytes the v4 base reads CV 0.12 (sd 0.35, n=3) and yield900 CV 0.28 (sd 1.00, n=3). The 0.8 per-draw sd quoted for the family is dominated by the yield900 draws.",
  "The '118 = 118' null cannot be recomputed from the repo: the two Kaggle commit-run reads (yield900 37, base 42) came from a REST fetch cached in scratchpad/search/loss-ledger-3/cache.json, which no longer exists. On the four rig runs that do exist, yield900 is +5 levels (vs M1+M4, +0.10 lv/game, se 0.13) and +10 levels vs the two pure-stock live-geometry base draws (M1 36, keith_fx 35: +0.20 lv/game, se 0.12, 10 games better / 2 worse). The exact tie rests on one base commit run reading 42, the highest of all six, and on counting the keith_retry graft arm (40) as base.",
  "The 'pooled six-draw base 39.33' that every rig gate is read against is itself half yield900 runs (41, 40, 37) — so the rig comparator already embeds the pooling assumption the claim is trying to justify.",
  "The ledger's projected SE after draw #4 (0.34) assumed sd stayed 0.83; recomputed n=6 SE is 0.32 (sd 0.78, mean 3.22).",
  "The 0.40 live/local calibration in R-score-arithmetic-0908.md:33-41 uses local 7.44 (keith_probe graft wave) and 8.46 (yield900 Y1) — neither is the flown v4 base regime (6.40 / 6.88) — against a live mean pooling both configs; matched pairs give ~0.44-0.45."
 ],
 "confounds": [
  "Pooling into the base family: the three yield900 draws have sd 1.00 vs 0.35 for the three v4 draws (F=8.1 on 2,2 df, p~0.11, not significant at n=3 but a 3x sd ratio). If yield900 genuinely has higher draw-to-draw variance, pooling doubles the base's sd (0.35 -> 0.78) and raises every future candidate's detection threshold by ~2x. The pooled family also shifts the base mean from 2.85 to 3.22.",
  "Direction consistency: yield900 > base on the rig (41,40 vs 36,35), on live (3.59 vs 2.85), but < base on the two Kaggle commit runs (37 vs 42). None of these is individually significant; the 'null' is a pre-registered read, not a demonstrated absence of effect. The dead verdict (3.59 vs 3.6 line) is decided by 0.01.",
  "The base rig draws counted as 'base' are not homogeneous: M1 (09-02) is pure stock; M4 is the keith_retry graft arm with retries 'not engaged'; the later pure-stock keith_fx (09-10) read 35 and is not in the six-draw pool.",
  "Order/boot effects on the rig: Y1 ran third in its session (notebook:557-560 flags the order-effect control); Y2 was cold-boot first; base M1 was the only arm in its session. docs/EXPERIMENT-2026-09-02-ab-instrument-control.md records a detected +0.25 order effect.",
  "Live draws are on the hidden 110 with private per-game composition unavailable (R-loss-ledger-3.md:65); the rig is 25 public games. Only two config points (base, yield900) exist on the same serving regime for a rig-to-LB mapping; the third ('flight') mixes serving profiles.",
  "The ledger working copy is uncommitted and has whitespace-reindented the whole file (git diff 1428/1419 lines) while adding one row; the 45-submission header count is a template artifact, not a live count."
 ],
 "does_conclusion_survive": "Yes, with caveats. (1) Base draw #4 = 2.73 is real, in its pre-registered 2.0-4.7 band, and the ledger is stale exactly as claimed (plus one older submission, 55543514, is missing entirely). (2) The yield900 DEAD verdict survives as a pre-registered rule outcome: mean 3.59 <= 3.6 and draw 3 (4.01) <= 4.04, both reproduce to the decimal. But it is 'not a step at the >=4.0 line', decided by 0.01, not evidence of zero effect: every instrument where it can be compared on the same serving regime (rig 41/40 vs 36/35; live 3.59 vs 2.85) points the same way at +0.1-0.2 lv/game / +0.7 LB, and the exact 118=118 tie depends on two Kaggle commit runs whose data is no longer in the repo and on a graft arm counted as base. (3) Pooling the yield900 draws into the base family is defensible only as a pre-registered convention; statistically it inflates the base sd from 0.35 to 0.78 and shifts the mean up 0.37, which will make future one-draw and three-draw reads even less sensitive (thresholds 1.6 / 0.9 / 0.7 LB for 1 / 3 / 5 draws at sd 0.8). A cleaner practice is to keep v4 (n=3, 2.85 +- 0.35) as the identical-bytes base and report the six-draw pool separately. (4) The rig bar of 48 is not calibrated to any LB target in writing; it was set as +0.5 lv/game over the 36 read (notebook:299), and the 39.33 comparator it is now read against is itself half yield900 runs. Empirically the regime maps at ~2.0-2.2 LB per rig level/game (0.44 LB per local score point), so 48 levels ~ 3.9-4.2 LB, which happens to line up with the live >=4.0 step line.",
 "what_would_change_verdict": "Would flip to PARTLY_WRONG: recovering the two Kaggle commit-run logs (kf 347926973 and the v4 commit) and finding totals other than 37/42, or per-game data that does not give a 0.00 paired diff — the off-Kaggle 'null' is unverifiable locally today. Would flip toward REFUTED on pooling: one or two more identical-bytes draws of each config showing the yield900 sd (1.0) and v4 sd (0.35) are real and the means separate by >1.1 LB (the 3-vs-3 two-sample 95% threshold), which would make pooling illegitimate and the 'dead' read a threshold artifact. Would strengthen to CONFIRMED: finding cache.json or an equivalent per-game record of the commit runs reproducing 118=118, and a pure-stock rig base draw (like keith_fx 35) being added to the comparator pool in place of the retry arm.",
 "commands_run": [
  "kaggle competitions submissions -c arc-prize-2026-arc-agi-3 -v  (50 rows, default paging)",
  "kaggle competitions submissions -c arc-prize-2026-arc-agi-3 -v --page-size 200  (86 rows) -> scratchpad/kaggle_subs_all.csv; python3 csv/json reconcile against docs/submission-ledger.json",
  "python3 json inspection of docs/submission-ledger.json (header, counts, rows 56133282 / 56111215, status Counter, date span)",
  "git log / git diff --stat / git diff on docs/submission-ledger.json; stat -f %Sm docs/submission-ledger.json",
  "python3 statistics: means/sd/SE/CV for yield900 (n=3), v4 (n=3), family n=5, n=6; Welch t; detection thresholds for n=1/3/5 at sd 0.8 and 0.83",
  "python3 sum of levels_completed/score over every offkaggle/results/**/results.json (33 waves)",
  "python3 paired per-game recompute: Y1+Y2 vs M1+M4 and vs M1+keith_fx",
  "grep/sed on offkaggle/REGIME_WAVE_STATUS.md (lines 286-302, 533-600, 605-660, 1518-1530), offkaggle/run_regime_wave.py:308-320, docs/HANDOFF-2026-08-29-packs-in-flight.md:600-660, docs/research-2026-09-08/R-loss-ledger-3.md:1-12, R-loss-ledger-3-NOTES.md:1-30, R-score-arithmetic-0908.md:30-42, docs/HANDOFF-2026-08-29-the-model-axis.md:58-72, docs/STRATEGY-2026-08-29-independent-review-path-to-6.md:36-44, docs/PLAN-2026-09-08-revised-plan-to-7plus.md:8-24",
  "head of summary.txt for waves 20260902T2334-flight, 20260902T2121-keith, 20260906T2251-keith_yield900, 20260907T0702-keith_yield900-cb2, 20260903T1014-keith_retry, 20260909-193735-regime-keith, 20260910-081906-regime-keith_fx",
  "ls scratchpad/search/loss-ledger-3/ (absent); grep -rn 118 = 118 / 56133282 / 2.73 / 7.44 across docs and offkaggle"
 ]
}
```

---

## agent-9

```json
{
 "claim": "Commit 3743ab6 / memory arcagi3-model-axis-blocked-on-scale.md: the served Qwen3.8-Flash-Next is a 131B-parameter, 360 GB MoE (512 experts/layer, ~6B active), not the 8B dense model the plan assumed; LoRA needs the unquantised base resident (>=5x80GB), a realistic 8xH100 run at ~$40/h costs $400+, a 20-50x gap vs the $9-25 budget; the model axis is dead 'blocked on scale, not will'; only the M0 corpus survives.",
 "verdict": "PARTLY_WRONG",
 "recomputed": [
  {
   "quantity": "Architecture / experts / layers / hidden / vocab",
   "claimed": "Qwen4ExpForConditionalGeneration, 512 experts/layer top-10, 48 layers, hidden 2560, vocab 248,320",
   "recomputed": "architectures=['Qwen4ExpForConditionalGeneration'], num_experts=512, num_experts_per_tok=10, num_hidden_layers=48, hidden_size=2560, vocab_size=248320, moe_intermediate_size=640, 12 full-attn + 36 linear-attn layers",
   "source": "curl https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json (saved scratchpad/base_config.json); cross-checked offkaggle/flashnext_patches/radixark_nvfp4_ple_fp8.patch:32-42",
   "matches": true
  },
  {
   "quantity": "bf16 checkpoint on disk",
   "claimed": "360 GB across 131 shards",
   "recomputed": "131 .safetensors, 360,000,192,888 bytes = 360.000 GB (335.3 GiB); all tensors BF16",
   "source": "curl https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next/tree/main summed in python3 (scratchpad/base_tree.json)",
   "matches": true
  },
  {
   "quantity": "Total parameter count",
   "claimed": "~121 B (PREREG) / 131B (memory + commit title)",
   "recomputed": "180.0 B total (= 360 GB / 2 bytes). Routed experts 120.80 B; PLE n-gram table 51.23 B (20,000,000 x 2560 exactly); MTP 2.61 B; linear-attn 2.09 B; embed+lm_head 1.27 B; hyper-connections 0.64 B; full-attn 0.62 B; vision 0.45 B; shared experts 0.24 B; routers 0.06 B. Language core excluding PLE/MTP/vision = 125.7 B; everything except PLE = 128.8 B. '121B' is the routed-expert count alone; '131B' matches nothing; '131B ↔ 360 GB' is internally inconsistent (131B x 2 B = 262 GB).",
   "source": "HTTP range-read of all 131 safetensors headers (1,658 tensors) via scratchpad/gethdr.sh, bucketed in python3",
   "matches": false
  },
  {
   "quantity": "Active parameters per token",
   "claimed": "~6 B",
   "recomputed": "≈7.3 B: routed experts 10x3x2560x640x48 = 2.36 B plus always-on 4.9 B (linear-attn 2.09, embed/lm_head 1.27, hyperconn 0.64, full-attn 0.62, shared 0.24, router 0.06), excluding PLE lookups. Rough community figure; not load-bearing.",
   "source": "config.json fields + header counts",
   "matches": false
  },
  {
   "quantity": "NVFP4 checkpoint size and shard count",
   "claimed": "(not stated in claim; ~90 GB of weights on one 96 GB card)",
   "recomputed": "206 shards, 135,253,622,894 bytes total = 125.91 GiB; of which PLE fp8 = 51.2 GB (47.7 GiB, 10 files) and everything else = 84.0 GB = 78.2 GiB",
   "source": "curl HF tree of RadixArk/Qwen3.8-Flash-Next-NVFP4 (scratchpad/nvfp4_tree.json); matches offkaggle/modal_flashnext_serve.py:78 MODEL_TOTAL_BYTES=135_253_622_894 and offkaggle/KEITH_REGIME.md:32",
   "matches": true
  },
  {
   "quantity": "GPU-resident weights at serve",
   "claimed": "~90 GB; 'NVFP4 is the only reason it fits one 96 GB card'",
   "recomputed": "vLLM 'Model loading took 81.8 GiB' on Kaggle (= 87.8 GB) and 79.42 GiB on Modal (20 wave results.json). Reconciles with 78.2 GiB non-PLE NVFP4 bytes + runtime overhead; the 47.7 GiB PLE table is CPU-offloaded (VLLM_PLE_CPU_OFFLOAD=1, serve script :203). ~90 GB is fair. Note bf16 non-PLE would be 257.5 GB, so the 'only reason it fits' statement is true.",
   "source": "offkaggle/KEITH_REGIME.md:113; offkaggle/FLASHNEXT_MODAL_STATUS.md:23; docs/HANDOFF-2026-08-29-packs-in-flight.md:528; grep of offkaggle/results/*/results.json",
   "matches": true
  },
  {
   "quantity": "Which weights are quantized",
   "claimed": "'FP4-base LoRA training ... unproven'; implies LoRA targets are FP4",
   "recomputed": "quantization_config.ignore = [embed_tokens, mtp.*, *.self_attn.*, *.linear_attn.*, *.mlp.gate*, *.mlp.shared_expert.*, *hyper_connection*, *.ple.*, model.visual.*, lm_head]. Only routed experts are NVFP4; all attention/linear-attn weights (2.70 B, 5.4 GB bf16) — the normal LoRA targets — are bf16 in the served checkpoint.",
   "source": "curl https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json (scratchpad/nvfp4_config.json)",
   "matches": false
  },
  {
   "quantity": "Minimum GPUs to hold the base for LoRA",
   "claimed": ">=5 x 80 GB (360 GB unquantised base resident)",
   "recomputed": "bf16 base minus the frozen, offloadable PLE table = 257.5 GB -> 4 x 80 GB, not 5. Against the served NVFP4 checkpoint (attention already bf16, experts frozen NVFP4): 78.2 GiB of weights fits ONE 96 GB card; activations at long context push it to 2 x 96 GB or 2 x H100. A bnb-NF4 QLoRA of the 128.8 B non-PLE core would be ~65-72 GB and also fits 1-2 cards. Scale is not the binding constraint; the unverified part is a training path that loads modelopt-NVFP4 experts frozen (no evidence either way in the repo).",
   "source": "arithmetic on header counts and NVFP4 tree bytes; nvfp4 ignore list",
   "matches": false
  },
  {
   "quantity": "Modal 8xH100 hourly rate",
   "claimed": "~$40/h, so a short run is $400+",
   "recomputed": "Modal pricing page: H100 SXM5 $0.001097/s = $3.95/h -> 8 x H100 = $31.6/h; A100-80GB $2.50/h; RTX PRO 6000 $3.03/h; B200 $6.25/h. '$400+' implies >12.6 h on 8xH100. A 2-GPU 6-h run is ~$36-47; 4xH100 for 6 h ~$95.",
   "source": "WebFetch https://modal.com/pricing; offkaggle/modal_flashnext_serve.py:65 (RTX PRO 6000 $0.000842/s)",
   "matches": false
  },
  {
   "quantity": "Gap vs per-experiment budget",
   "claimed": "20-50x ($400+ vs $9-25)",
   "recomputed": "At a realistic 2-4 GPU configuration the GPU cost is ~$40-100 per attempt, i.e. ~2-5x the $9-25 wave cost; the real cost is engineering days on unproven tooling, not GPU scale.",
   "source": "derived from the two rows above",
   "matches": false
  },
  {
   "quantity": "'The plan implicitly assumed an 8B dense model'",
   "claimed": "Correction to a long-standing assumption",
   "recomputed": "No '8B dense' wording exists in docs/, offkaggle/, or memory outside the 09-10 correction itself. docs/PLAN-2026-09-08-revised-plan-to-7plus.md:70 (commit b42bb94, 09-08) already says 'fine-tune a 125B MoE off-box (Modal multi-GPU) ... 1-2 weeks and real money'; docs/research-2026-09-08/R-model-axis-0908.md:5 already says '125B MoE, 6B active + 51B n-gram/PLE + 4B MTP, NVFP4 ~80 GiB'. The premise being 'corrected' did not exist.",
   "source": "grep -rIn '8 ?b dense' docs offkaggle memory; sed PLAN:66-76; git log b42bb94",
   "matches": false
  },
  {
   "quantity": "M0 corpus numbers",
   "claimed": "2,047 records from 11,443 calls across 247 runs; 1,373 train / 674 holdout; holdout cd82, cn04, ft09, sk48, wa30",
   "recomputed": "records train 1373 + holdout 674 = 2047; calls_seen 11443; runs_seen 247; calls_kept 2062 - 15 duplicates = 2047; holdout_games exactly as claimed",
   "source": "docs/research-2026-09-09/corpus-manifest-m0.json",
   "matches": true
  }
 ],
 "errors_found": [
  "Parameter count: the model is 180.0 B parameters on disk (360 GB / 2 bytes), not 131 B or ~121 B. The write-up pairs a 131 B headline with a 360 GB size that only the 180 B total explains; 121 B is the routed-expert count alone. Source: per-tensor headers of all 131 shards.",
  "The 360 GB figure counts the 51.2 B-parameter (102.4 GB bf16) PLE n-gram embedding table, which serving already CPU-offloads and which is a frozen lookup during any LoRA training. Treating 360 GB as 'weights that must be GPU-resident' overstates the GPU requirement; non-PLE bf16 is 257.5 GB (4 x 80 GB, not >=5).",
  "'LoRA needs the unquantised base resident' is false as a general statement (QLoRA-style training against quantized frozen weights is the standard route) and specifically misdescribes this checkpoint: the NVFP4 quantization ignore-list keeps all attention/linear-attention weights (the LoRA targets, 2.70 B params) in bf16; only routed experts are FP4. The doc's 'untested alternative' is in fact the main path, and its 'still ~90 GB resident so multi-GPU at best' is wrong: 78.2 GiB of non-PLE weights fits one 96 GB card; activations, not weights, decide whether a second card is needed.",
  "Pricing: Modal lists H100 at $3.95/h, so 8 x H100 is $31.6/h, not ~$40/h; '$400+' needs >12 h on 8 GPUs. A defensible 2-4 GPU attempt is ~$40-100 in GPU time, a 2-5x gap to the $9-25 wave cost, not 20-50x.",
  "Framing: 'correction to a long-standing assumption that the model was 8B dense' has no antecedent; the 09-08 plan (b42bb94) and R-model-axis-0908.md already state '125B MoE ... Modal multi-GPU ... 1-2 weeks and real money'. The 09-10 note rediscovered the plan's own premise and then mis-stated the plan.",
  "Process: the pre-registered stop rule was 'if M2 is not passed within two working sessions, stop'. M1/M2 were never attempted; the lane was closed by a desk estimate with the wrong GPU count and price, and the closure was labelled as the stop rule firing.",
  "Minor: '~6 B active' recomputes to ~7.3 B (routed 2.36 B + always-on 4.9 B); 'the 5 GiB KV is what is left after the weights' is not how it is set — KV is pinned explicitly by --kv-cache-memory-bytes 5368709120 (KEITH_REGIME.md:93) and the profile is 8 seqs, with 3.21x being the 32k-token concurrency figure."
 ],
 "confounds": [
  "The doc cites no on-disk source for its config/size figures; the only local config fragment is the PLE patch (hidden 2560, 48 layers, 512 experts). I verified against Hugging Face over the network today (2026-09-11); the pinned NVFP4 revision 7b71922 matches the repo's MODEL_TOTAL_BYTES exactly, but the bf16 base repo could have changed since 09-10 (no revision was pinned in the claim).",
  "Whether any training framework can load modelopt-NVFP4 routed experts frozen alongside bf16 attention for PEFT, and whether the pinned vLLM fork (0.1.dev20073+g8e685d198, custom Qwen3_8FlashNext model class, MTP-3) can serve a runtime LoRA on a hybrid linear-attention model, is NOT verifiable from this repo. My memory estimate (1-2 x 96 GB) assumes such a path exists; if it does not, the fallback is bnb-NF4 QLoRA of the 128.8 B non-PLE core with 24,576 unfused expert modules, which fits in memory but would be very slow and requires the 360 GB download.",
  "Activation memory at the live 32k analyzer window was not computed; a 1-card fit at 8-16k tokens with gradient checkpointing is plausible, at 32k a second card is likely. This affects card count (1 vs 2), not the scale verdict.",
  "The 'Model loading took 79.42 GiB' Modal figure and 81.8 GiB Kaggle figure include vLLM's CUDA-graph/runtime allocations; the 78.2 GiB non-PLE checkpoint bytes are the pure weight footprint.",
  "Modal prices quoted from the public page today; the doc gave no source or date for '$40/h'."
 ],
 "does_conclusion_survive": "Partially, and on different grounds. The factual core — the served model is a very large MoE (180 B on disk, 120.8 B in routed experts, 512 experts top-10, 360 GB bf16 / 135 GB NVFP4) and not a small dense model — is confirmed. But the stated reason for closure ('blocked on SCALE: >=5x80GB, $400+/run, 20-50x over budget') does not survive: the 360 GB includes a 51 B CPU-offloadable PLE table; the served NVFP4 checkpoint's non-PLE weights (78 GiB) fit one 96 GB card and the LoRA-target attention weights are already bf16; Modal's real H100 price makes even 8 GPUs $31.6/h; a realistic 2-4 GPU attempt is tens of dollars, not hundreds. What actually blocks the lane is unproven software (training against modelopt-NVFP4 frozen experts; LoRA serving in the custom vLLM fork with MTP-3) plus calendar time before Nov 2 — exactly the '1-2 weeks and real money, no fine-tune ever served+scored' assessment the 09-08 plan already gave. The practical decision (do not sink the remaining weeks into it; Track D is the campaign) is still defensible, but 'blocked on scale, reopen only at 10x spend' is an over-close: the honest reopen condition is 'a demonstrated LoRA training + vLLM-serving path for this checkpoint', and the pre-registered M2 stop rule was never actually exercised.",
 "what_would_change_verdict": "To CONFIRMED: evidence that no framework can hold the NVFP4 routed experts frozen on GPU during training AND that bnb/NF4 QLoRA of this Qwen4Exp class fails to load, which would force the bf16 base (257.5 GB non-PLE, 4x80 GB) and restore a multi-GPU cost picture — though still ~$16/h on 4xH100, not $40/h. To REFUTED: a demonstrated attention-only LoRA trained on 1-2 x 96 GB against the served checkpoint and served by the pinned vLLM fork at the keith V14 regime, which would show the lane was never blocked by scale at all.",
 "commands_run": [
  "cat memory/arcagi3-model-axis-blocked-on-scale.md; git show --stat 3743ab6; git show 3743ab6 -- docs/research-2026-09-09/PREREG-model-axis.md; sed -n 1,200p docs/research-2026-09-09/PREREG-model-axis.md",
  "find . -iname config.json -o -iname '*flash*next*' -o -iname '*nvfp4*'; grep -rIl 'Qwen4Exp|num_experts|248320|360 GB|131 shards|131B' .",
  "python3 summarize submission/_flashnext_gate/FLASHNEXT_GCP_MODEL_INFO.json (10 PLE bf16 files = 102,400,512,032 bytes)",
  "sed -n 1,95p offkaggle/flashnext_patches/radixark_nvfp4_ple_fp8.patch; grep -n ... offkaggle/modal_flashnext_serve.py; sed -n 60,105p / 283,292p / 336,350p offkaggle/modal_flashnext_serve.py",
  "grep -rIh -o 'Model loading took ...' logs offkaggle | sort | uniq -c (79.42 GiB x20 Modal; 81.8 GiB Kaggle)",
  "grep -n GB/GiB/shards offkaggle/KEITH_REGIME.md offkaggle/FLASHNEXT_MODAL_STATUS.md; grep -rIn '81.8 GiB' docs offkaggle logs submission",
  "sed -n 1,60p docs/research-2026-09-08/R-model-axis-0908.md; python3 print docs/research-2026-09-09/corpus-manifest-m0.json",
  "WebFetch https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json",
  "WebFetch https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next/tree/main",
  "curl -sSL https://huggingface.co/api/models/{Qwen/Qwen3.8-Flash-Next,RadixArk/Qwen3.8-Flash-Next-NVFP4}/tree/main and .../raw/main/config.json -> scratchpad/{base,nvfp4}_{tree,config}.json; python3 sum of .safetensors sizes (360,000,192,888 B / 131 shards; 135,253,622,894 B / 206 shards)",
  "python3 print text_config / vision_config / quantization_config from the downloaded configs",
  "scratchpad/gethdr.sh: curl -r 0-7 then -r 8-(8+n-1) on each of 131 base shards -> scratchpad/hdrs/*.json; python3 bucket all 1,658 tensor shapes by module (180.0 B total; experts 120.796 B; PLE 51.233 B; MTP 2.607 B; linear_attn 2.087 B; embed/lm_head 1.271 B; hyperconn 0.641 B; full_attn 0.617 B; vision 0.449 B; shared 0.236 B; router 0.063 B)",
  "WebFetch https://modal.com/pricing (H100 $0.001097/s, A100-80 $0.000694/s, RTX PRO 6000 $0.000842/s, B200 $0.001736/s)",
  "grep -rIn -i '8 ?b dense|small dense' docs offkaggle memory; grep -n 'model axis|test-time|LoRA' docs/PLAN-2026-09-08-revised-plan-to-7plus.md; sed -n 35,38p / 66,76p docs/PLAN-2026-09-08-revised-plan-to-7plus.md; git log -1 -- docs/PLAN-2026-09-08-revised-plan-to-7plus.md docs/research-2026-09-08/R-model-axis-0908.md",
  "python3 split nvfp4_tree.json into PLE files (51,200,267,901 B = 47.68 GiB) vs rest (83,995,035,950 B = 78.23 GiB)"
 ]
}
```

---

## agent-10

```json
{
 "claim": "\"Cadence arm and effects graft: engaged-but-flat class closed\" — cadence wave 20260909-193735-regime-keith (conc 6 / 1,767 s) engaged on all gates (e2e 26.6 s, calls/turn 1.60, calls/game 56.8) and read 34 levels = -2.28 sd DEAD, closing the throughput/cadence family as a step; fx wave 20260910-081906-regime-keith_fx delivered a harness-computed effect table on 97.2% of turns and read 35 levels = -1.85 sd DEAD, showing the agent does not act on a dynamics model; memory file adds 87% of each call is queue and 62.5% of actions land after the last clear.",
 "verdict": "PARTLY_WRONG",
 "recomputed": [
  {
   "quantity": "cadence levels / zero-level games / actions per game",
   "claimed": "34 / 5 / 156",
   "recomputed": "34 / 5 / 155.6 (3891/25)",
   "source": "offkaggle/results/20260909-193735-regime-keith/results.json games[].levels_completed, actions",
   "matches": true
  },
  {
   "quantity": "cadence calls/game",
   "claimed": "56.76 (>= 40 gate; 'slightly exceeded base 55.80')",
   "recomputed": "56.76 = 1419 status-200 rows / 25; BUT tr87 ran alone in a 5th sequential batch (effective conc 1) with 181 calls; the other 24 games average 51.58 calls (-7.5% vs base 55.8)",
   "source": "requests_shim.jsonl (1434 rows, 15 ReadTimeouts); per-game t offsets show 5 batches starting at +0/+1767/+3535/+5302/+7069 s",
   "matches": false
  },
  {
   "quantity": "cadence median e2e per call",
   "claimed": "26.64 s (< 60 gate)",
   "recomputed": "26.64 s median, 30.07 mean, p90 54.7; 7.6% of calls >= 60 s; batch-5 (tr87 solo) median 5.7 s",
   "source": "requests_shim.jsonl elapsed_s over status-200 rows",
   "matches": true
  },
  {
   "quantity": "cadence calls/turn and turns/game",
   "claimed": "1.60 and 35.5 (base 1.02 / 54.7)",
   "recomputed": "1419 calls / 889 '--- analysis_step=' turns = 1.596; 35.56 turns/game; base wave 1396/1362 = 1.024, 54.48 turns/game",
   "source": "transcripts/*.txt parsed by turn header and [MODEL RESPONSE META] count; base 20260902T2121-keith/20260902-232148-regime-keith/telemetry.json",
   "matches": true
  },
  {
   "quantity": "cadence z-score",
   "claimed": "(34-39.33)/2.34 = -2.28 sd",
   "recomputed": "-2.28 arithmetically; but no six 25-game rig waves in offkaggle/results reproduce 39.33/2.34 (the four Modal draws named at STATUS:641 give 39.25/2.22; the nine keith-family live-geometry waves 36,34,43,40,41,31,40,41,41 give 38.56/3.97 -> cadence -1.15 sd, fx -0.90 sd)",
   "source": "python3 itertools search over results.json totals of every 25-game wave; offkaggle/REGIME_WAVE_STATUS.md:615,641",
   "matches": true
  },
  {
   "quantity": "cadence co-primary walls",
   "claimed": "1 of 12 (dc22 L2)",
   "recomputed": "1/12: dc22=2 passes; bp35 0, g50t 0, lf52 1, lp85 1, ls20 1, r11l 1, sb26 1, sp80 0, tn36 0, vc33 3 (<4), wa30 1",
   "source": "results.json vs WALLS dict in offkaggle/read_cadence_arm.py:24-25",
   "matches": true
  },
  {
   "quantity": "cadence 'healthier than base': requests and preemptions",
   "claimed": "1,434 vs 1,396 requests; preemptions 4.1% vs 10.2%",
   "recomputed": "vLLM e2e count 1434 vs 1396; preemptions 59/1434 = 4.1% vs 142/1396 = 10.2%",
   "source": "metrics_before/after.prom deltas for both waves",
   "matches": true
  },
  {
   "quantity": "vLLM queue share of e2e ('87% of each call is queue')",
   "claimed": "120.9 s of 139.9 s = 87%",
   "recomputed": "base wave 120.9/139.0 = 0.87; fx wave 122.2/140.4 = 0.87; cadence wave 12.6/29.5 = 0.43",
   "source": "vllm:request_queue_time_seconds_sum / vllm:e2e_request_latency_seconds_sum deltas, metrics_*.prom",
   "matches": true
  },
  {
   "quantity": "share of actions after the last level clear",
   "claimed": "62.5% (275 game-runs, 11 waves)",
   "recomputed": "base wave alone 62.1%; 9 keith-family live-geometry waves 60.0%; the only 11-wave set that yields 275 runs (9 + flight + halfconc) 63.1%; the doc does not list its 11 waves so the exact figure is not reproducible",
   "source": "artifacts/*_events.jsonl level_completed flags across offkaggle/results/*/ ; docs/research-2026-09-09/R-harness-teardown-and-scoring.md:36,71",
   "matches": true
  },
  {
   "quantity": "fx levels / zero-level / actions per game / calls per game / median e2e / calls per turn",
   "claimed": "35 / 7 / 141 / 55.0 / 147.7 s / 1.02",
   "recomputed": "35 / 7 / 140.8 / 55.04 (1376/25) / 147.66 s / 1376/1354 = 1.016",
   "source": "20260910-081906-regime-keith_fx results.json, requests_shim.jsonl, transcripts",
   "matches": true
  },
  {
   "quantity": "fx engagement: block share of turns; games with >= 2 action lines",
   "claimed": "97.2%; 25/25",
   "recomputed": "1316/1354 = 97.19%; 25/25 games reach >= 2 lines (max lines per game 2..8, cap EFFECTS_MAX_ACTIONS=8)",
   "source": "transcripts [USER PROMPT] sections containing 'Observed action effects'; arm_env.json EFFECTS_MAX_ACTIONS=8",
   "matches": true
  },
  {
   "quantity": "fx correction: '24 of 25 games saw a 7+-line block'",
   "claimed": "24/25",
   "recomputed": "16/25 (ar25 5, ls20 5, re86 5, tu93 5, g50t 6, m0r0 6, sk48 6, wa30 6, tr87 2 never reach 7 lines)",
   "source": "same transcript parse, regex '^- .+? x\\d+: ' on the block; STATUS:1161-1162 (commit c3437a1)",
   "matches": false
  },
  {
   "quantity": "fx correction: correlation block richness vs levels",
   "claimed": "-0.086",
   "recomputed": "+0.037 (Pearson, max block lines vs levels_completed, n=25) — both are ~0; the measure used in the note is not stated",
   "source": "statistics.correlation over results.json and transcript parse",
   "matches": false
  },
  {
   "quantity": "fx z-score",
   "claimed": "(35-39.33)/2.34 = -1.85 sd",
   "recomputed": "-1.85 arithmetically (same pooled-base caveat as above)",
   "source": "results.json",
   "matches": true
  },
  {
   "quantity": "fx 'agent does not act on a dynamics model even when handed one'",
   "claimed": "not measured in the write-up (telemetry uptake counts only [EVID]/[HYPO] markers, which are 0 here)",
   "recomputed": "model refers to the table ('observed action effects', 'moves 1 obj by (...)', etc.) in 95/1354 turns = 7.0%, in 23/25 games, always in THINKING, 0 times in TOOL CALL code",
   "source": "regex over transcripts THINKING / TOOL CALL: python / ASSISTANT sections; telemetry.json aggregate.wall.uptake",
   "matches": false
  },
  {
   "quantity": "cadence 'harness stamps time_remaining_seconds into every tool result as a pacing hint'",
   "claimed": "every tool result",
   "recomputed": "29 occurrences across all 25 cadence transcripts (~1/game), 38 in the base wave; tool_agent.py copies the key only when the payload carries it",
   "source": "grep -c time_remaining_seconds transcripts/*.txt; scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py:1446-1448",
   "matches": false
  },
  {
   "quantity": "per-game clock honoured in the cadence wave",
   "claimed": "1,767 s/game",
   "recomputed": "first-to-last call span 1,717-1,763 s for every game; sum of call e2e / span = 0.92-1.03 (model time saturates the clock); results.json wallclock_s (1768/3535/5302/7069/8839) is measured from wave start, not game start",
   "source": "requests_shim.jsonl t and elapsed_s per game; results.json wallclock_s",
   "matches": true
  },
  {
   "quantity": "fx SAFETY prompt tokens/call",
   "claimed": "expect ~ +250 vs base 20,061",
   "recomputed": "20,548 = +487",
   "source": "requests_shim.jsonl prompt_tokens mean; base summary.txt",
   "matches": false
  }
 ],
 "errors_found": [
  "STATUS:1161-1162 / commit c3437a1: '24 of 25 games saw a 7+-line block' is wrong — 16/25 by direct count of the block's action lines (cap is EFFECTS_MAX_ACTIONS=8); the 'no dose variation' argument built on it is weaker than stated.",
  "STATUS:979,985 and commit 32f11f7: 'calls/game 56.76 slightly exceeded base 55.80' is an artifact of the wave's batch structure: conc 6 over 25 games ran as five sequential batches (6,6,6,6,1); tr87 ran alone (effective conc 1, e2e median 5.7 s, 181 calls). The other 24 games averaged 51.6 calls, 7.5% below base. The >= 40 gate still passes; the 'matched call budget' framing is approximately, not exactly, true.",
  "STATUS:1006-1007 / memory arcagi3-queueing-not-compute.md: 'the harness stamps time_remaining_seconds into every tool result' — it appears ~29 times in 25 transcripts (about once per game); tool_agent.py:1446 only forwards it when the solver payload carries it. The rushed-pacing confound is real in principle but barely instantiated in the data.",
  "STATUS:1097-1099 / commit f94167e: 'the agent does not act on a dynamics model even when one is handed to it' was never measured — telemetry uptake tracks [EVID]/[HYPO] markers only (0 here). A direct grep shows the model quotes or reasons from the effect table in 95/1354 turns (7.0%, 23/25 games), always in THINKING and never in tool code. Defensible statement: it reads the table occasionally and never computes with it.",
  "Pooled six-draw base 39.33 (sd 2.34): its six constituent draws are not documented anywhere in the repo and no six 25-game rig waves reproduce it (36,40,41,40,42,37 would; 42 and 37 exist in no results.json). The nine same-regime keith-family live-geometry waves show sd 3.97 (31-43). Under that dispersion cadence is -1.15 sd and fx -0.90 sd; the 'engaged-and-negative' language (STATUS:987 'not flat — worse', 1094) is not supported by one draw, which c3437a1 partly conceded.",
  "STATUS:1016-1018 and memory arcagi3-queueing-not-compute.md: 'the throughput/cadence family is now closed as a source of a step' was contradicted the next day by the author's own keith_budget wave 20260910-192514 (conc 28 / 23,760 s: 169.8 calls/game, 56 levels = +7.12 sd, elasticity 0.21, STATUS:1242-1262), which STATUS:1289-1293 says 're-opens anything that buys effective calls per game'. The memory file still carries the closed verdict without this reversal.",
  "fx SAFETY line pre-registered prompt +250 tokens/call; actual +487 (20,548 vs 20,061). Not reported."
 ],
 "confounds": [
  "Cadence arm changed two knobs (conc 28->6 and clock 7,920->1,767 s) and, in practice, a third: it ran as five sequential batches with the last game alone at conc 1. Calls per game matched only because model time saturates the clock in both regimes (sum e2e / span ~= 1.0 per game).",
  "Level count vs per-game clock is monotone across the rig: 1,767 s -> 34, 3,960 s (halfconc) -> 32, 7,920 s -> ~39, 23,760 s (budget3x) -> 56. Every clock-shortened arm reads below base and the clock-lengthened arm reads far above, so 'cadence is negative' cannot be separated from 'clock cut is negative' with this data (the notebook admits this at STATUS:1003-1014 but the memory headline does not).",
  "The cadence deficit is concentrated in batch 1: tn36/lf52/cn04/bp35/wa30/lp85 read 4 levels vs an 8.38 per-game base mean; batches 2-5 read 30 vs 31.1. Batch-1 e2e (33.8 s mean) was ordinary, so no mechanism is identified; consistent with noise but the -5.33 is not a wave-wide effect.",
  "Engine/sandbox wall time is not a separate budget in either regime (model calls fill 92-103% of the clock), so 'less wall time for the engine' is not the issue; what differs is the magnitude of the rarely-shown time_remaining hint and the absence of ~120 s idle queue waits between turns.",
  "fx: the block is appended at the very end of the user prompt after the tool-format instructions (dc22 turn 5: char 4003 of 4343) and the system prompt never mentions it; 'engaged' means delivered, not attended to. 25 ReadTimeouts and 65 preemptions match the base regime.",
  "Base sd: same-regime 25-game draws on this rig span 31-43 (sd ~3-4), the instrument MDE is ~+0.5 lv/game (docs/EXPERIMENT-2026-09-02-ab-instrument-control.md:192-194), so single-draw reads only resolve step-sized effects; sub-step harm or gain is unreadable."
 ],
 "does_conclusion_survive": "Partly. What survives: both arms engaged by their pre-registered gates and neither reached the >= 48 step bar — both landed 4-5 levels BELOW base under every pooling tried, so 'not a step' is robust, and the narrow claim 'a behaviour-shaping or help-it-model graft that engages does not produce a step on this brain' is consistent with seven single-draw reads. What does not survive: (1) 'throughput/cadence family closed as a step' — refuted by the author's own next wave (budget3x 56 levels, +7.12 sd), and the memory file still states the dead version; (2) 'cadence itself is dead' — confounded with the 4.5x clock cut and the sequential-batch structure; (3) 'engaged-and-negative / levels went down' — one draw against a base whose sd is undocumented and likely understated (rig dispersion sd ~4), with the deficit sitting in one batch of six games; (4) 'the agent does not act on the table' — unmeasured in the write-up; it quotes the table in 7% of turns and never in code. Several supporting details are wrong (24/25 seven-line blocks is 16/25; time_remaining is not in every tool result; matched calls were 51.6 not 56.8 for the 24 non-solo games).",
 "what_would_change_verdict": "To CONFIRMED: (a) document the six draws behind 39.33/2.34 and show they are same-regime waves, or re-pool with the nine same-regime waves and show the reads still fall below the base; (b) run the queued diagnostic (conc 6 at the unchanged 7,920 s clock, or conc 6 with the clock held and calls capped at ~55) and read <= 44 — that would separate cadence from the clock cut; (c) a counterbalanced redraw of fx reading <= 44 with a measured uptake metric for the effects block (e.g., turns whose code uses an offset from the table) near 0; (d) amend the memory/STATUS headline to say the throughput family was reopened by budget3x. To REFUTED: a conc-6-at-7,920-s arm or a cadence redraw reading >= 45, or the four named Modal draws being the only real base (sd 2.22, four draws) with a redraw of either arm inside 36-42.",
 "commands_run": [
  "ls -la offkaggle/results/{20260909-193735-regime-keith,20260910-081906-regime-keith_fx,20260910-081755-regime-keith_fx}",
  "cat .../summary.txt, arm_env.json, git_status.txt for both waves",
  "grep -n '' offkaggle/REGIME_WAVE_STATUS.md | sed -n '505,515p;543,575p;596,660p;826,840p;880,1200p;1236,1300p'",
  "git show --stat 32f11f7 / f94167e / c3437a1; git show c3437a1 -- offkaggle/REGIME_WAVE_STATUS.md",
  "python3: results.json -> levels, actions, score, zero-level, wallclock, per-game levels, batch grouping (both waves)",
  "python3: requests_shim.jsonl -> status counts, calls/game (with and without tr87), e2e mean/median/p90, >=60 s share, prompt/completion tokens, finish reasons, per-game first-call offset and call span, per-batch stats",
  "python3: metrics_before/after.prom deltas -> e2e, queue, inference, prefill, preemptions, request_success for cadence, fx and base 20260902-232148-regime-keith",
  "python3: transcripts parse by '--- analysis_step=' header and section labels -> turns, calls/turn, 'Observed action effects' share (1316/1354), max block action lines per game, regex search of THINKING/ASSISTANT/TOOL CALL for references to the block (95 turns), block placement in the user prompt",
  "python3: Pearson(max block lines, levels) = +0.037",
  "python3: artifacts/*_events.jsonl level_completed -> share of actions after last clear for 12 waves and 3 candidate 11-wave pools",
  "python3: itertools search for any six 25-game rig waves giving mean 39.33 sd 2.34; alternative-pool z-scores",
  "python3: per-game levels of cadence/fx vs the per-game mean of 8 live-geometry keith-family waves; per-batch sums",
  "grep time_remaining_seconds / run_elapsed_seconds counts in transcripts of three waves; sed -n 1425,1465p scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py",
  "sed -n 1,120p offkaggle/read_cadence_arm.py; sed -n 1,80p and 140,262p submission/_throughput_v1/graft_effects.py",
  "cat memory/arcagi3-queueing-not-compute.md; sed -n 1,80p docs/research-2026-09-09/R-harness-teardown-and-scoring.md; sed -n 188,198p docs/EXPERIMENT-2026-09-02-ab-instrument-control.md",
  "python3: tabulate every offkaggle/results/*/results.json and */*/results.json (arm, n, levels, conc, per-game clock, grafts)"
 ]
}
```

---

## agent-11

```json
{
 "claim": "A2 (persistent workspace + transition log + verifier) is closed because the live loop refuses to build/verify executable world models (0 verifier calls in 358 python calls; harness-directed build fires 6/6 runs but 24/24 no_code inside Stage-1's 7k/25k window), while offline Stage-1 produced 4/4 backtest-green models on the agent's own stuck-at-a-wall transitions in 2-3 calls; green models score 100% on their own level and 0-45% on the neighbouring level; one verified model costs ~65k completion tokens = 79% of a game's decode; therefore 'the failure is the LOOP, not the brain and not the tooling', A2's conversion question dies untested, and Polyphony/A3/A4 are priced out.",
 "verdict": "PARTLY_WRONG",
 "recomputed": [
  {
   "quantity": "python calls / verifier (backtest) calls, keith_ws kill2",
   "claimed": "0 verifier calls in 358 python calls (877 transitions, 340 preambles)",
   "recomputed": "python_calls 358, backtests 0, transitions_logged 877, preambles 340; 0 [WS-BACKTEST] markers in 6 transcripts; prompt fix present 32-45x per transcript",
   "source": "offkaggle/results/20260909T0844-keith_ws-kill2/20260909-104418-regime-keith_ws/telemetry.json per_game/*/ws/graft summed; transcripts/*.txt grep",
   "matches": true
  },
  {
   "quantity": "directed build pass 2: fired runs / pairs / calls / no_code",
   "claimed": "fires 6/6 runs, 8 pairs, 24 calls, 24/24 no_code, 0 verified",
   "recomputed": "24 '[WS-DIRECT] ... no_code' markers, 8 (run,level) pairs, present in all 6 transcripts; 24 shim rows finish_reason=length; vLLM request_success_total{length} 0->24; backtests 0",
   "source": "offkaggle/results/20260909T1303-keith_wsd-kill3/20260909-150316-regime-keith_wsd/transcripts/*.txt, requests_shim.jsonl, metrics_before/after.prom",
   "matches": true
  },
  {
   "quantity": "directed build pass 2 prompt/output tokens",
   "claimed": "prompts 5.7-11.6k, output 21-27k",
   "recomputed": "prompt 5,676-11,577; completion 21,191-27,092; prompt+completion = 32,768 on all 24 rows; content_chars=0 on all 24; reasoning_chars 49,706-83,427; n_messages=2, has_tools=False on all 24",
   "source": "kill3 requests_shim.jsonl rows with finish_reason=length",
   "matches": true
  },
  {
   "quantity": "directed build pass 2 GPU seconds",
   "claimed": "5,016 s",
   "recomputed": "5,857.2 s (sum of elapsed_s over the 24 directed rows; same method reproduces pass 1's '2,824 s' as 2,823.8 s)",
   "source": "kill3 requests_shim.jsonl; kill2 (wsd) requests_shim.jsonl",
   "matches": false
  },
  {
   "quantity": "directed build pass 1 prompt/output ranges",
   "claimed": "prompts 11.0-14.8k, output 18.0-21.8k, 15/15 no_code, 5/6 runs",
   "recomputed": "15 length rows, prompts 8,114-18,912, completion 13,856-24,654 (dc22: 8,114/24,654 = already inside the 'Stage-1 window'); 15/15 no_code and 5/6 runs confirmed by summary",
   "source": "offkaggle/results/20260909T1146-keith_wsd-kill2/20260909-134649-regime-keith_wsd/requests_shim.jsonl, summary.txt",
   "matches": false
  },
  {
   "quantity": "Stage-1 prompt size",
   "claimed": "7,123-token prompt",
   "recomputed": "prompt0_tokens 7,506 (dc22L1) and 7,045 (dc22L2); 7,123 appears in no saved data file",
   "source": "docs/research-2026-09-09/stage1-own-transitions/s1a_dc22L1_summary.json, s1a_dc22L2_summary.json; grep of docs/, memory/, notebook",
   "matches": false
  },
  {
   "quantity": "Stage-1 greens and calls-to-green",
   "claimed": "4/4 green; dc22L2 3 then 2 calls; dc22L1 2 then 9",
   "recomputed": "green=True x4, best_matched 20/20 x4, best_call 2 (s1a L1), 3 (s1a L2), 9 (s1b L1), 2 (s1b L2); '2-3 calls' holds for L2 only",
   "source": "stage1-own-transitions/s1{a,b}_dc22L{1,2}_summary.json",
   "matches": true
  },
  {
   "quantity": "cross-level transfer",
   "claimed": "own level 100%; other level 10%/10%/0%/45%",
   "recomputed": "dc22L1_draw1 20/20 | 2/20; dc22L1_draw2 20/20 | 2/20; dc22L2_draw1 0/20 | 20/20; dc22L2_draw2 9/20 | 20/20",
   "source": ".venv/bin/python docs/research-2026-09-09/cross_level_transfer.py (re-run)",
   "matches": true
  },
  {
   "quantity": "cost of one verified model",
   "claimed": "65,468 mean completion tokens (40,886-118,221), 16,367/call, 12.7x stock, = 79% of a game's decode (83,094 @ 64.2 calls, 1,293/call), best 49%, worst 142%",
   "recomputed": "mean (47410+55356+118221+40886)/4 = 65,468; /16 calls = 16,367; 65,468/83,094 = 78.8%; 40,886/83,094 = 49.2%; 118,221/83,094 = 142.3%; 16,367/1,293 = 12.66. BUT 83,094/64.2/1,293 comes from the keith_carry75 arm's shim (1606 usage rows / 25 games), not a stock wave; true stock waves: 74,234 (09-02, 55.8 calls, 1,330/call) and 73,644 (09-09, 56.8 calls, 1,297/call) => 88-89%",
   "source": "docs/research-2026-09-09/GATE-polyphony-clock.md:44-51; offkaggle/results/20260908T2110-keith_carry75/.../requests_shim.jsonl; offkaggle/results/20260902T2121-keith/.../requests_shim.jsonl; offkaggle/results/20260909-193735-regime-keith/requests_shim.jsonl",
   "matches": false
  },
  {
   "quantity": "Stage-0 tn36 / cn04",
   "claimed": "tn36 green 12/12 in 3 calls, 187 s; cn04 green 18/18 converging 0,1,2,1,3,6,9,17,18 at call 11",
   "recomputed": "draw1_tn36: green, best_call 3, wall 186.7 s; draw2_cn04: green, best_call 11, scoreboard 0,0,1,2,1,3,6,0,9,17,18 (two zero calls omitted in the memory's sequence)",
   "source": "docs/research-2026-09-09/stage0-flashnext/draw1_tn36_summary.json, draw2_cn04_summary.json",
   "matches": true
  },
  {
   "quantity": "directed-build share of wave decode (pass 2)",
   "claimed": "(not stated)",
   "recomputed": "600,307 of 984,709 generated tokens = 61% of the whole wave's decode went to the 24 directed calls (vLLM generation_tokens_total 2 -> 984,711)",
   "source": "kill3 requests_shim.jsonl; metrics_before/after.prom",
   "matches": true
  }
 ],
 "errors_found": [
  "'5,016 s of GPU' for pass 2 is not reproducible from any file; the shim's elapsed_s over the 24 directed calls sums to 5,857 s (the identical method reproduces pass 1's 2,824 s).",
  "Pass-1 ranges in PREREG-a2-workspace-arm.md RESULT 3 ('prompts 11.0-14.8k, output 18.0-21.8k') are wrong: actual 8,114-18,912 / 13,856-24,654. dc22's pass-1 calls (8.1k prompt / 24.6k output) were already inside the claimed Stage-1 window and still emitted no code, so the 'narrow operating window' diagnosis was refuted by pass-1 data before pass 2 was run.",
  "'Stage-1's 7,123-token prompt' is unsourced; saved summaries give 7,506 (L1) and 7,045 (L2).",
  "GATE-polyphony-clock.md labels the 83,094-token / 64.2-call / 1,293-per-call row 'stock, per whole game', but it is computed from the keith_carry75 arm (A1 carry graft) shim. True stock waves are 73,644-74,234 tokens/game, so one verified model is 88-89% of a stock game's decode, not 79% (conclusion strengthens, label is wrong).",
  "'Same data, same contract, same verifier, same token window' (memory + RESULT 4) over-claims: contract/verifier/window match, data does not. Live used the run's own transitions capped at 12 (not the 20 Stage-1 used), a ~170-word DIRECT_SYSTEM instead of the ~600-word offline SYSTEM (which carries the explicit 'do not re-count characters of long rows by hand' guidance), per-cell decimal 'rRcC:old->new' lists instead of the offline row-grouped run-collapsed hex diff, and a hard 60-cell cap that truncates transitions with a '(+more changed cells...)' notice. On Stage-1's own dc22L2 data that cap truncates 2 of 12 transitions (129-cell switch toggles); the live kill3 dc22 L2 slice has a 129-cell transition truncated. A truncated transition cannot be reproduced exactly from the prompt, so pass 2 was partly ill-posed by construction. graft_workspace.py:780-832 vs stage0.py:150-214.",
  "The recorded 'plausible mechanism' (the directed call is issued inside a long-running agent whose 30+ turns of play history dominate) is contradicted by the shim: all 24 directed requests have n_messages=2 (system+user), has_tools=False - a fresh context with no play history, the same shape as the offline call. Play-history contamination cannot be the cause.",
  "'The model does not ELECT to build a world model' / '0 verifier calls' is numerically right but mischaracterises the behaviour: kill2/kill3 reasoning repeatedly states 'Decision: use the backtest tool...' / 'Let's use backtest to build a verified world model' and then issues a python call instead. The intent exists; the tool call is never emitted. Why (argument size, the 30 s WS_BACKTEST_TIMEOUT the model mentions, tool-call format habits) was not tested.",
  "Memory's '2-3 calls' for Stage-1 holds for dc22L2 only; the control dc22L1 took 2 and 9 calls, and one offline call (dc22L1 draw 2 call 4) also hit the length cap with no code - the offline instrument tolerated that with up to 20 calls and rich mismatch feedback, the live one allowed 3 calls with one-line feedback.",
  "Transfer memory's 'geometric memorisation of 20 transitions, not mechanic induction' overstates: dc22L2_green_draw1.py contains a WALKABLE colour set, a player finder scanning the grid, a base-terrain layer and a row-63 counter alongside hard-coded switch coordinates (52,22)/(52,40) and toggle rectangles. The models were asked for a one-level model with ENTRY_GRID given and never asked to generalise; the test conflates level-specific parameters with absence of mechanics. The economic point (no automatic amortisation) survives.",
  "Memory's cn04 convergence '0->1->2->1->3->6->9->17->18' omits two zero-score calls (scoreboard 0,0,1,2,1,3,6,0,9,17,18); '9 files of 11 calls' is unverifiable (no per-call log saved)."
 ],
 "confounds": [
  "Encoding/prompt confound: the live directed prompt is not the offline prompt (different system text, different diff encoding, lossy 60-cell cap, 12 vs 20 transitions, different feedback). The 0/24 result therefore cannot separate 'the live loop' from 'the live rendering'. The directed prompt and the model's 25k-token reasoning were NOT captured anywhere (prompts/*.log 'teacher-forced' hits are the stock tool-schema text at line 12; no 'Build the world model'/'ENTRY GRID' text saved), so what the model was doing inside <think> is unknowable after the fact.",
  "Sampling path differs: offline chat_chained used /v1/completions with prompt ids, temperature 0.6/top_p 0.95 and server-default top_k, in <=10k-token segments; live used the stock _chat_completion (chat/completions, temperature 0.6/top_p 0.95/top_k 20, thinking on, server-default max_tokens). Small but untested.",
  "Length-cap semantics: every directed call ended finish=length with content_chars=0, i.e. the model never exited <think>. Offline, output per green call was 13-24k tokens with the same 32k ceiling, so the live model had equal or more room. The difference is in how long it reasons, which points at prompt content, not the window - consistent with the write-up's 'window was never the explanation' but not with 'the loop'.",
  "The n=6-run kill geometry (cd82, dc22, lf52 x2 draws) cannot resolve level effects, as the pre-registration itself says; all level/safety numbers (7, 8, 8 vs base 8) are noise and were not used for the verdict - correct.",
  "Stage-1 'own transitions' came from a different run (carry75 wave dc22_p0) than the live kill3 runs; 'the same game's wall' is true, 'the same data' is not. Live cd82 fired at level 1 (not a wall).",
  "Cost denominator: 'a game's decode budget' is what the stock spends under the current clock/call geometry, not a fixed budget; using the carry75 arm (which includes compaction calls) inflated it. Also, in kill3 the 24 directed calls consumed 61% of the whole wave's generated tokens - the live wave itself confirms the tax at ~25k tokens/call.",
  "'0 in 358' pools two conditions; the control (kill1) also had 359 python calls and 0 backtests, so the true count is 0 in 717 across both conditions."
 ],
 "does_conclusion_survive": "Partly. What survives: (1) the counts - 0 verifier calls in 358 (and 359 more in the control), directed build fired in 6/6 runs, 24/24 no_code with every call exhausting the 32,768 window inside <think>, 0 verified models; (2) Stage-1's 4/4 offline greens and the 10/10/0/45% transfer table are reproducible; (3) A2's central question (does a verified model convert into cleared levels) is indeed untested; (4) the cost argument survives and strengthens (one verified model = 88-89% of a true stock game's decode, and the live directed calls ate 61% of kill3's decode). What does NOT survive: the headline inference 'the failure is the LOOP, not the brain and not the tooling'. The directed call was a fresh 2-message context with no play history, so the recorded loop mechanism is refuted by the shim; and the live tooling was NOT the offline tooling - a different system prompt, a different and lossy transition encoding (60-cell cap that truncates the very toggles Stage-1 modelled), 12 instead of 20 transitions, and 3 one-line-feedback calls instead of up to 20 rich ones. The correct statement is: 'the live directed-build instrument as implemented does not reproduce the offline result; the window is not the cause; play-history contamination is not the cause; the remaining candidates are the prompt/encoding/feedback differences and sampling variance, none of which was tested because the directed prompt and reasoning were not logged.' The decision to stop A2 under the pre-registered hard stop is defensible on budget grounds, but 'A2 closed: the loop refuses' should be downgraded to 'A2 closed: unread on the directed-build hypothesis after three instrument iterations'.",
 "what_would_change_verdict": "Upgrade to CONFIRMED if a directed-build run with the OFFLINE instrument bit-for-bit (stage0.py SYSTEM + build_data_block hex encoding, 20 transitions, no cell cap, up to ~10 calls with render_mismatch feedback) issued from inside the live harness ALSO produced 0 greens; that would isolate the live setting from the rendering. Downgrade to REFUTED if that same run produced greens (i.e. the 0/24 was the encoding), or if the '5,016 s' / '11-15k' / '7,123' figures turn out to come from a log I could not find that contradicts the shim. Secondary: a saved copy of one directed prompt plus its reasoning text (e.g. showing hand-counting of hex rows or reasoning about the '(+more changed cells' truncation) would settle which prompt difference drove the non-termination.",
 "commands_run": [
  "cat memory/arcagi3-a2-workspace-closed.md, arcagi3-stage0-reopened-flashnext.md, arcagi3-executable-models-dont-transfer.md, arcagi3-polyphony-priced-out.md",
  "ls offkaggle/results/20260909T{0804,0844,1128,1146,1303}-keith_ws*/*; cat each summary.txt",
  "python3: parse requests_shim.jsonl for all 5 runs (finish_reason counts, length rows with prompt/completion/reasoning_chars/content_chars/n_messages/has_tools/elapsed_s, prompt+completion sums, tool_calls field)",
  "python3: sum per_game/*/ws/graft counters from telemetry.json for all 5 runs (python_calls, backtests, saves, transitions_logged, preambles)",
  "grep -c 'WS-BACKTEST' / 'Tools: `python`' / 'backtest' on kill2 transcripts; grep -n WS-SAVE; grep -rn WS-DIRECT on kill3 transcripts",
  "grep -l 'Build the world model' / 'ENTRY GRID' / 'teacher-forced' / 'executable WORLD MODEL' across kill3 prompts/, transcripts/, benchmark.json, artifacts/; sed -n 11,13p prompts/dc22-fdcac232_p0.log",
  "grep vllm:request_success_total / generation_tokens_total / e2e_request_latency_seconds_sum / request_inference_time_seconds_sum in kill3 metrics_before.prom and metrics_after.prom",
  "sed -n 1,80p and 715,911p submission/_throughput_v1/graft_workspace.py; grep -n def/WS_DIRECT/no_code",
  "sed -n 1282,1342p scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py (stock _chat_completion params)",
  "git show --stat 09467d8; git log --oneline -- submission/_throughput_v1/graft_workspace.py",
  "cat docs/research-2026-09-09/PREREG-a2-workspace-arm.md, PREREG-stage1-own-transitions.md, GATE-polyphony-clock.md, cross_level_transfer.py; sed -n 100,215p stage0-flashnext/stage0.py; cat stage0-flashnext/llm.py; cat stage1-own-transitions/own_transitions.py",
  "python3: dump s1a/s1b dc22L1/L2 and draw1/draw2 cn04/tn36 summary.json (green, best_call, prompt/completion tokens, scoreboard, prompt0_tokens)",
  ".venv/bin/python docs/research-2026-09-09/cross_level_transfer.py",
  "grep -n WALKABLE/switch/range/def in stage1-own-transitions/dc22L2_green_draw1.py; grep -c ENTRY_GRID; wc -c green models",
  "grep -rn '7,123|7123|5,016|5016|358 python|keith_wsd|kill3' offkaggle/REGIME_WAVE_STATUS.md docs/research-2026-09-09 memory/*.md",
  "grep -l 'completion tok/call 1293 ' offkaggle/results/*/*/summary.txt; sed -n 855,875p offkaggle/REGIME_WAVE_STATUS.md; grep CADENCE/CLIENT/SCORE on carry75 and 25-game keith/keith_yield900 summaries",
  "python3: recompute calls/game and completion tokens/game from requests_shim.jsonl for keith_carry75, keith 09-02, keith 09-09, keith_yield900 09-07",
  ".venv/bin/python: render stage1 dc22L2_transitions.json through stage0.build_data_block and graft_workspace.render_transitions at (20,300),(12,60),(20,60); count truncations; changed cells per transition for Stage-1 data and for kill3 dc22-fdcac232_p0_events.jsonl level-2 slice"
 ]
}
```

---

## agent-12

```json
{
 "findings": [
  {
   "severity": "blocks_conclusion",
   "affected_claims": [
    "pooled six-draw base 39.33 sd 2.34",
    "every z-score/'sd' headline since 09-08 (cadence -2.28, fx -1.85, probe/carry +0.71, budget +7.12, ctx12 -11.68, ctx16 -7.83, nr16 -8.26)",
    "loss-ledger-3 'sd of a 25-game total ~6'"
   ],
   "finding": "The sd the data actually support for a 25-game wave total is ~3.5 levels, not 2.34 and not ~6. 2.34 is an n=6 sample sd of six totals (two of which, the Kaggle commit runs 37 and 42, have no per-game data on disk) with a chi-square 95% CI of [1.46, 5.74] (factors 0.624/2.453 at 5 df). The '~6' in R-loss-ledger-3-NOTES.md:27 multiplies the cross-sectional per-run sd (1.19 across the 25 games of one run) by 5; but 69-76% of that variance is fixed between-game heterogeneity, not draw noise. Direct decomposition on per-game data: four Modal pool draws [36,40,41,40] -> sum of within-game variances 10.42 -> total sd 3.23 (per-game draw var 0.417, within share 0.24); eight flat live-geometry draws [36,34,40,41,40,41,41,35] (M1, yield180, retry, Y1, Y2, probe, carry75, fx) -> sum of within-game variances 12.11 -> total sd 3.48, observed sd of the eight totals 2.98, mean 38.50; rms paired per-game difference sd across the 28 pairs of those draws 0.990/game -> single-total sd 0.99*5/sqrt(2) = 3.50, difference-of-two-totals sd 4.95. The AB-instrument doc's 'per-game paired sd ~1.2' (docs/EXPERIMENT-2026-09-02-ab-instrument-control.md:191) is a paired-DIFFERENCE sd on a different instrument and implies a single-total sd of 4.2, consistent with 3.5 given its own noise. Working figure: sd 3.5 per wave total; sd of (arm - pooled mean of 6) = 3.5*sqrt(7/6) = 3.78.",
   "evidence": "python3 scratchpad/stats.py over offkaggle/results/{20260902T2121-keith,20260903T0202-keith_yield180,20260903T1014-keith_retry,20260906T2251-keith_yield900,20260907T0702-keith_yield900-cb2,20260908T0752-keith_probe,20260908T2110-keith_carry75}/*/results.json and offkaggle/results/20260910-081906-regime-keith_fx/results.json (games[].levels_completed); pool list offkaggle/run_regime_wave.py:313-315 base_levels_six_draws [41,40,37,36,40,42]; docs/research-2026-09-08/R-loss-ledger-3-NOTES.md:27; docs/EXPERIMENT-2026-09-02-ab-instrument-control.md:190-194"
  },
  {
   "severity": "blocks_conclusion",
   "affected_claims": [
    "'>=48 step candidate / 45-47 redraw / <=44 dead' rule",
    "'engaged-but-flat class CLOSED' (probe 41, carry 41, cadence 34, fx 35)"
   ],
   "finding": "At sd 3.5 the minimum detectable effect of ONE wave against the pool is +/-7.4 levels (1.96*3.78) at 95% two-sided, +/-9.8 at 80% power; TWO waves averaged: +/-5.6 (1.96*3.5*sqrt(1/2+1/6)) at 95%, +/-6.9 at 80% power. The 48 bar is therefore only +2.29 sd above the pool (+3.43 under the claimed 2.34), and the 45-47 'redraw' band is +1.4 to +2.0 sd, i.e. inside noise. The kill side is far more permissive than the notebook treats it: a lever with a TRUE +10% effect (+4 levels, 43.3) reads 'dead' (<=44) 62% of the time from one wave and 66% from a two-wave mean; a true +15% (+5.9) reads dead 43%/40%; a true full step (+8.7, 48) reads dead 18% of the time from one wave (8% under sd 2.34). So single-wave reads of 41/41/34/35 do not distinguish 'flat' from '+10-15%'; the paired per-game t-tests against the 8-draw per-game mean give probe +0.10/game (t=+0.73), carry75 +0.10 (t=+0.86), cadence -0.18 (t=-1.07), fx -0.14 (t=-0.81) -- none beyond |t|=1.1 on 24 df. 'No step from a behaviour-shaping lever' is supported; 'class closed for all effect sizes' and 'not flat, WORSE' are not.",
   "evidence": "scratchpad/stats.py MDE and false-negative blocks: P(read<=44 | true 43.3, sd 3.5*sqrt(7/6)) = Phi((44.5-43.3)/3.78) = 0.62; P(read<=44 | true 48) = Phi(-0.93) = 0.18; per-game paired t vs 8-draw mean computed from the same results.json files; notebook verdicts at offkaggle/REGIME_WAVE_STATUS.md:639,815,987,1079"
  },
  {
   "severity": "blocks_conclusion",
   "affected_claims": [
    "~15 arms judged since 09-02 under the same 48/45/44 rule with no multiplicity control"
   ],
   "finding": "Fourteen 25-game arms (flight, yield180, kv10, retry, halfconc, yield900 x2 valid, probe, carry75, cadence, fx, budget, ctx12, ctx16, nr16) plus several 6-game kill sets have been read against one 48 bar. At sd 3.5 the per-arm false-positive rate at 48 is 1.5% (P(null >=47.5)), so the family-wise chance that at least one of 15 null arms would have produced a spurious 'step candidate' is 21% (0.9% under the claimed 2.34), and the chance at least one lands in the 45-47 redraw band by luck is 67%. That the campaign has produced ZERO spurious step candidates in 15 reads is consistent with the true sd being ~3.5 (expected 0.2 false hits) and not evidence the bar is tight. Conversely the multiplicity problem on the KILL side is the real one: with 15 single-wave kills, the expected number of genuinely +10-15% levers wrongly killed, if any existed, is 0.4-0.6 per such lever. The rule needs a redraw before 'dead' for any engaged arm reading 40-47, not only for 45-47.",
   "evidence": "scratchpad/stats.py multiplicity block: 1-(1-0.0153)^15 = 0.207; 1-(1-(Phi((44.5-39.33)/3.78)-0.0153)... = 0.665; RESULT headers offkaggle/REGIME_WAVE_STATUS.md:226-1580 enumerate the arms"
  },
  {
   "severity": "blocks_conclusion",
   "affected_claims": [
    "'elasticity decays 0.40 -> 0.21' (REGIME_WAVE_STATUS.md:1260)",
    "sizing-table closure of the buy-calls family at elasticity 0.21 (STATUS:1492-1540)",
    "triage bound built on 0.4 (R-triage-family-bounded.md)",
    "pre-registered budget bands STATUS:1135-1143"
   ],
   "finding": "Both elasticity points are single-wave ratios and the decay claim is a base-choice artifact plus noise. Delta-method SEs at sd 3.5 per total: KV10 vs its same-session base 36 = 0.41 +/- 0.32; KV10 vs the pool 39.33 = 0.20 +/- 0.21; budget3x vs pool = 0.21 +/- 0.05; budget3x within-wave (41 by 7,920 s -> 56) = 0.17 +/- 0.07. The KV10 point cannot distinguish 0.2 from 0.4 (or from 0.0); on the common pooled base the two points read 0.20 and 0.21. Separately, the pre-registered bands were mis-specified in a way that pre-baked 'decay': '>=60 = elasticity holds near 0.4 out to 3x' is wrong arithmetic -- 0.4 at 3.04x calls gives 39.33*(1+0.4*2.04) = 71.4 levels, and the 60 edge corresponds to elasticity 0.26; 46-59 maps to 0.08-0.25. A wave at exactly the KV10 rate would have read 71, so 56 being 'below 60' was always going to be labelled decay. The sizing-table closure rests on 0.21 +/- 0.05 being the local slope for +20-56% call increases, which the only measurement in that range (KV10) cannot pin better than +/-0.2-0.3; the shortfall to 48 is ~4 levels (about 1 sd of one wave) under the corrected anchoring the other verifier found.",
   "evidence": "scratchpad/stats.py elasticity block (d e/dL = 1/(L0 r), d e/dL0 = -L/(L0^2 r), r = C/C0-1; levels from results.json of 20260903T0553-kv10-keith, 20260910-192514-regime-keith_budget; calls 82.16/169.76/55.8 from the verifier reports); band arithmetic: (60/39.33-1)/2.04 = 0.258; offkaggle/REGIME_WAVE_STATUS.md:1135-1143, 1201, 1209, 1260"
  },
  {
   "severity": "weakens",
   "affected_claims": [
    "live 3-draw rule (mean >=4.0 step / <=3.6 dead) with per-draw sd ~0.8",
    "yield900 DEAD at 3.59",
    "'CV on identical bytes 0.25'"
   ],
   "finding": "With per-draw sd 0.8 the SE of a 3-draw mean is 0.46, so the 0.4-wide gap between the step line (4.0) and the dead line (3.6) is under one SE. P(a true-3.6 config reads 'step' >=4.0) = 0.19 and P(a true-4.0 config reads 'dead' <=3.6) = 0.19; a true 3.9 reads dead 26% of the time. Reaching 80% power on a +0.4 LB shift at sd 0.8 needs ~31 draws per arm. The yield900 3-draw mean 3.59 has a 95% CI of [2.68, 4.50], which covers both lines; the verdict was decided by 0.01. The 'identical bytes' base family is two byte-sets with very different dispersions: v4 [3.25, 2.58, 2.73] sd 0.35 vs yield900 [4.31, 2.45, 4.01] sd 1.00 (F = 8.1 on 2,2 df, not significant, but a 3x ratio); pooling to n=6 gives mean 3.22 sd 0.78, i.e. it more than doubles the base sd used for every future live read and lifts the base mean by 0.37. Every same-regime comparison points the same way for yield900 (rig Y1+Y2 vs M1+M4: +0.10/game se 0.13; vs M1+fx: +0.20/game se 0.12; live 3.59 vs 2.85): consistent with a small positive effect that no instrument on hand can resolve, not with a demonstrated null.",
   "evidence": "scratchpad/stats.py live block: se = 0.8/sqrt(3) = 0.462; Phi(-0.4/0.462) = 0.193; n = ((1.96+0.84)*0.8/0.4)^2 = 31.4; F = 0.998/0.124 = 8.07; live scores as given in the task brief and the ledger verifier (Kaggle CLI); paired per-game Modal recompute from the results.json files"
  },
  {
   "severity": "weakens",
   "affected_claims": [
    "pooling yield900 draws into the base family (rig pool 39.33 and live family n=6)"
   ],
   "finding": "On the rig the pooling is defensible: per-game paired Y1+Y2 minus M1+M4 is +0.10/game (sd 0.66, se 0.13), and adding the two yield900 draws changes the pool's implied total sd only from 4.18 (pure stock, n=2, poorly estimated) to 3.23 (n=4) / 3.48 (n=8). But the pool as constructed embeds the null it is meant to test (3 of 6 draws are yield900), includes a graft arm (retry, fired 5x) as 'base', and excludes the void 31 draw by a pre-registered rule whose inclusion would move the Modal-only mean to 37.6 and sd to 4.2 (n=5) -- so the pool's tightness partly reflects that exclusion. The rig reads yield900 41/40 vs pure-stock 36/35 (+5 to +10 levels) and the 118=118 tie rests on two Kaggle commit totals (37, 42) with no per-game data on disk. On the live side pooling is not defensible on the evidence: it merges the two most dispersed configs into one family and doubles the base sd (see previous finding).",
   "evidence": "scratchpad/stats.py pools A/B/C/D; offkaggle/results/20260907T0106-keith_yield900-cb/.../results.json total 31; summary.txt RETRY line of 20260903T1014-keith_retry (fired 5, 0.20/game); docs/research-2026-09-08/R-loss-ledger-3.md:3-6; ls scratchpad/search/loss-ledger-3 absent (per the ledger verifier)"
  },
  {
   "severity": "weakens",
   "affected_claims": [
    "the 48 bar's origin ('>= +0.5 lv/game') and its relation to any LB target"
   ],
   "finding": "The 48 bar was set at STATUS:297-300 as base 36 + 0.5 lv/game, justified by 'cross-boot local sd ~0.3 lv on this base per the three public commit runs 1.28-1.80' -- a live-LB dispersion figure applied to a rig-level count, i.e. wrong units; it was then carried onto the 39.33 pool without re-derivation (STATUS:687), and the probe pre-registration used 52 (STATUS:619) while read_cadence_arm.py:19 hard-codes 48. No document ties 48 to an LB number. Empirically the regime maps at ~2.0-2.2 LB per rig lv/game (base 1.42 -> 2.85; yield900 1.62 -> 3.59), so 48 levels (1.92/game) ~ 3.9-4.2 LB, which happens to coincide with the live >=4.0 line, but this is a two-point mapping.",
   "evidence": "offkaggle/REGIME_WAVE_STATUS.md:297-300, 619, 687; offkaggle/read_cadence_arm.py:17-19; rig/live pairs from the summary.txt SCORE lines and the ledger verifier's Kaggle CLI output"
  },
  {
   "severity": "cosmetic",
   "affected_claims": [
    "'four replications' closing the engaged-but-flat class"
   ],
   "finding": "Only three of the four cited replications (probe 41, carry 41, yield900 41/40) are on the Flash-Next rig; 'patch 21' is from the 08-27 27B era. Pooling the three on-instrument reads gives a class mean of ~+1.5 levels vs the 8-draw mean 38.5 with a 95% CI of roughly [-2.5, +5.5] levels (se ~2.0 for a mean of three totals at sd 3.5), which excludes a step (+8.7) but not +10-14%.",
   "evidence": "scratchpad/stats.py totals; memory arcagi3-two-level-wall.md dated 2026-08-27 (per the pooled-base verifier)"
  }
 ],
 "overall": "The verdict system's arithmetic is internally consistent but its dispersion input is wrong, and that error propagates into every sigma headline, the decay claim, and the class-closure inferences. Recomputed from per-game data across the eight flat live-geometry draws on disk (totals 36,34,40,41,40,41,41,35; results.json of each), the draw-noise sd of a 25-game total is ~3.5 levels (sum of within-game variances 12.1 -> 3.48; rms paired-difference sd 0.99/game -> 3.50; observed sd of the eight totals 2.98, mean 38.5). The claimed 2.34 is an n=6 point estimate with 95% CI [1.46, 5.74]; the rival '~6' (R-loss-ledger-3-NOTES.md:27) is a mis-derivation from cross-sectional per-run sd. Using sd 3.5 and the pool's own sampling error (x sqrt(7/6) = 3.78): one wave resolves +/-7.4 levels at 95% (+/-9.8 at 80% power), two waves +/-5.6 (+/-6.9 at 80% power). ROBUST under the corrected sd: budget3x 56 (+4.4 sd, paired t=+3.5), ctx12 12 (-7.2, t=-7.1), ctx16 21 (-4.9, t=-4.6), nr16 20 (-5.1, t=-5.5) -- all dead/alive as stated; and 'no arm has produced a step' (P of a spurious 48 across 15 null arms is 21% at sd 3.5, so zero hits is unremarkable). FRAGILE: cadence 34 (-1.4 sd, t=-1.07) and fx 35 (-1.15, t=-0.81) are 'not a step' but NOT 'worse than base'; probe/carry 41 (+0.44) are indistinguishable from +10-15% levers (a true +4-level lever reads <=44 62% of the time from one wave); 'elasticity decays 0.40 -> 0.21' is a base-choice artifact (0.20 vs 0.21 on a common base) on top of a KV10 SE of +/-0.2-0.3, with the pre-registered '>=60 = elasticity 0.4' band itself mis-computed (0.4 at 3x implies 71 levels; 60 implies 0.26); the live yield900 'dead' is decided by 0.01 on a 3-draw mean whose SE (0.46) exceeds the 0.4 gap between the step and dead lines (each line misclassifies the other's true value 19% of the time), and pooling yield900 into the live base doubles the base sd (0.35 -> 0.78) for every future read. The '118 = 118' rig null cannot be recomputed (two Kaggle commit totals have no per-game data on disk); the on-disk rig draws read yield900 +5 to +10 levels vs pure stock. RECOMMENDED RULE: (1) report every rig read as levels vs the 8-draw flat mean 38.5 with sd 3.5 (update the pool as clean base draws land; re-estimate sd from the per-game matrix, not from n totals); (2) one wave can only KILL below ~31 (>2 sd) or FLAG above ~46 (>2 sd); anything 31-46 that is ENGAGED gets a mandatory second counterbalanced draw before any verdict, and a two-wave mean >=45 (+1.9 sd) is the step-candidate bar, <=42 dead; (3) state elasticities with delta-method SEs and never infer curvature from two single-wave points; (4) live: keep v4 (n=3, 2.85 +/- 0.35) as the identical-bytes base, report pooled families separately, and replace the 4.0/3.6 lines with a pre-registered sequential test (e.g., continue until the 3-draw-mean 95% CI excludes 3.6 or 4.0, cap 5 draws) since 3 draws at sd 0.8 cannot separate them; (5) treat 'class closed' claims as closed for STEP-sized effects only, with the CI stated. Nothing in this audit revives a dead arm as a step; it downgrades 'worse'/'flat'/'decays'/'closed' language to what one wave at sd 3.5 can actually say.",
 "commands_run": [
  "ls offkaggle/results/; find offkaggle/results -name results.json | wc -l (33)",
  "python3 scratchpad/inv.py -- every results.json: n games, sum levels_completed, summary.txt geometry line",
  "sed -n 295,330p offkaggle/run_regime_wave.py (LEDGER3_REFERENCE base_levels_six_draws [41,40,37,36,40,42], 39.33, 2.34)",
  "sed -n 1,30p docs/research-2026-09-08/R-loss-ledger-3.md; sed -n 1,35p docs/research-2026-09-08/R-loss-ledger-3-NOTES.md; sed -n 1,40p offkaggle/read_cadence_arm.py",
  "grep -hE '^  (SCORE|ARM KNOB|IDENTITY|WALL|RETRY)' summary.txt for the 15 25-game waves (yield knob, levels, void tags, retry fired 5)",
  "python3 scratchpad/stats.py -- per-game 25x16 level matrix from results.json; sum of within-game variances and one-way between/within decomposition for pools A (4 Modal pool draws), B (M1+fx), C (8 flat live-geometry), D (C+void), E (C+KV10); 28 pairwise paired-diff sds; chi-square CI factors; MDE table for sd 2.34/3.2/3.5/4.0/4.4 (1 and 2 waves, with pool se); z-scores of 12 arms at sd 2.34 vs 3.5/4.4 with sqrt(7/6); P(null >=48) and family-wise over 15 arms; delta-method SEs for four elasticity definitions; live 3-draw rule power at sd 0.8, F-test yield900 vs v4, n=6 family; paired Y1Y2-M1M4 and Y1Y2-M1fx; paired per-game t vs the 8-draw mean for 12 arms",
  "sed -n 297,301p / 617,621p / 685,689p offkaggle/REGIME_WAVE_STATUS.md (rule origin: 36 + 0.5 lv/game, 'cross-boot local sd ~0.3', probe bar 52, carry bar 48)",
  "grep -n -iE 'decay|elasticity' offkaggle/REGIME_WAVE_STATUS.md; grep -n ' sd' verdict lines; grep -n '^## (RESULT|READ|VERDICT)' (arm inventory)",
  "sed -n 186,198p docs/EXPERIMENT-2026-09-02-ab-instrument-control.md (paired sd ~1.2, MDE +0.5 lv/game); sed -n 1198,1212p and 1258,1266p offkaggle/REGIME_WAVE_STATUS.md",
  "python3 inline: false-negative rates P(read<=44 | true 43.3/45.2/48) and P(read>=48) for 1 and 2 waves at sd 2.34 and 3.5; levels implied by elasticity 0.1-0.4 at 3.0x/3.04x calls; elasticity implied by band edges 46/59/60"
 ]
}
```

---

## agent-13

```json
{
 "overall": "INSTRUMENT TABLE (raw files; coverage = vLLM request_success_total delta ÷ shim status-200 rows; \"restart\" = process_start_time_seconds differs between metrics_before/after.prom; errors split into harness end-of-clock ReadTimeouts vs HTTP 5xx; largest gap = max difference between consecutive request starts in requests_shim.jsonl; wallclock from results.json games[].wallclock_s).\n\n| wave | arm / geometry | lv | shim ok/rows | coverage | preempt Δ | ReadTimeout (end-of-clock) | HTTP 5xx | 303 legs | largest gap | prompt med | per-game wallclock | restart | per-run VOID tags (summary.txt WALL) | write-up mentions fault? |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n| 20260909-193735-regime-keith | cadence, conc 6 / 1,767 s | 34 | 1419/1434 | 1434/1419 = 101.1% (same process) | 59 | 15 at +28/29/59/88/89/118 min = the five 1,767-s batch ends | 0 | 0 | 67.9 s @ +2.7 min | 21,738 | results.json wallclock_s is WAVE-relative (1768..8839); per-game first-to-last call span 1,766-1,813 s ⇒ all 25 got their clock | no | preempt 25/25, errors 15/25, length>1% 3 (r11l 4.3%, tu93 2.3%, vc33 4.0%) | yes, waived explicitly STATUS:981-984 |\n| 20260910-081755-regime-keith_fx | — | — | empty dir (no files) | — | — | — | — | — | — | — | — | — | aborted launch, nothing to read | n/a |\n| 20260910-081906-regime-keith_fx | conc 28 / 7,920 s | 35 | 1376/1401 | 1376/1376 = 100.0% | 65 | 25 at +130-132 | 0 | 613 (all 303 on calls >150 s, all returned 200) | 50.8 s @ +94.8 | 21,848 | 7,920-7,921 all 25 | no | preempt 25/25, errors 25/25, length>1% 4 (cn04 3.8, s5i5 5.7, su15 1.8, tu93 3.6) | STATUS:1086 notes \"the script printed VOID\" |\n| 20260910-132152-regime-keith | 3x attempt #1, conc 28 / 23,760 s | 0 finished | 2937/2937 | no metrics_after (never finished) | unknown | 0 | 0 | 868 | 64.9 s @ +189.9 | 21,582 | results.json status \"running\", 0 games | killed +276 min (ENOSPC, 17/25 games crashed) | — | correctly declared VOID STATUS:1174 |\n| 20260910-181717-regime-keith_budget | smoke, 3 games conc 3 / 3,600 s | 2 | 522/522 | 100.0% | 0 | 0 | 0 | 0 | 33.5 s | 20,969 | 3,602-3,631 | no | none | n/a (smoke) |\n| 20260910-192514-regime-keith_budget | 3x clean, conc 28 / 23,760 s | 56 | 4244/4290 | UNDEFINED pre-restart: metrics_before is the smoke container (process start 66.6 min BEFORE wave t0, 524 requests = smoke); metrics_after is a NEW process at +297.3 min; the 3,083 pre-restart calls have no server telemetry at all; post-container 1,185 vs 1,161 = 102.1% | 266 post-container only (pre unknown) | 22 at +394-396 | 24 × HTTP 500 \"modal-http: InternalFailure: Server has lost track of input\" at +295-297 min (24 games) | 1,210 (1,202 rows) | 215.2 s @ +297.0 | 21,363 | 23,761-23,768 (ft09 won at 9,905) | YES — old container age at new process start 363.9 min = 6.07 h ⇒ the serve script's own reaper (modal_flashnext_serve.py:293 MAX_LIFETIME_S = 6 h \"two 2.2 h arms + cold start\"; :451-459 os._exit(0) with no drain). Clock lost failed-call→first OK response mean 344 s/game (1.4%); e2e median 139.0 s pre / 104.2 s post | errors 24/25 (2-4 each), preempt 25/25, length>1% 6 (ka59, re86, sc25, sk48, su15, tu93) | NO — STATUS:1242-1342 has zero mention of restart/500/preemptions; summary VLLM row (661 req, e2e 181.9, queue 162.9) = new-container counters minus smoke-container snapshot, unreadable |\n| 20260911-075839-regime-keith_ctx12 | conc 28 / 7,920 s, window 12,288 | 12 | 2709/2753 | UNDEFINED pre-restart (1,669 calls); post-container 1,081 vs 1,040 = 103.9% | 0 post-container only (pre unknown; summary \"preemptions 0\" covers 40% of calls) | 18 at +131-132 | 25 × HTTP 500 \"Server has lost track of input\" at +75-77 (25/25 games) + 1 × 502 at +99 | 56 (27 rows; 25 of them are the calls that straddled the outage, 2-3 × 303, waited 398-473 s, returned 200) | 398.1 s @ +76.6 (zero call starts +76.6→+83.2) | 6,674 | 7,920-7,970 (clock ticked through the outage) | YES — new process at +79.7 min; old container age 83.2 min ⇒ NOT the 6-h reaper, not the 15-min idle scaledown; cause not in any local file (logs/ctx12_20260911.log:334-358 only shows the 25 500s). Clock lost mean 465 s/game = 5.9% | errors 25/25 | NO — STATUS:1342-1446 zero mention |\n| 20260911-111640-regime-keith_ctx16 | conc 28 / 7,920 s, window 16,384 | 21 | 2046/2070 | 2056/2046 = 100.5% | 1,908 = 0.93/request (every 32k conc-28 wave on disk: 65-386) | 23 at +131-132 | 1 × 502 at +69 | 153 | 73.8 s @ +95.1 | 9,582 | 7,920-7,942 | no | preempt 25/25, errors 23/25 | NO — STATUS:1446-1580 zero mention of preemptions |\n| 20260911-143210-regime-keith_nr16 | conc 28 / 7,920 s, 16k + noreason | 20 | 1479/1497 | 1480/1479 = 100.1% | 1,652 = 1.12/request | 18 at +130-132 | 0 | 510 | 84.5 s @ +89.5 | 9,435 | 7,920-7,964 | no | preempt 25/25, errors 19/25 | NO — STATUS:1580-1621 zero mention |\n\nQ1 — WHICH WAVES HAD A RESTART OR OTHER FAULT. Two mid-wave container replacements: keith_budget 3x (+297 min, the rig's own 6-h lifetime reaper — foreseeable from the code, since a 23,760-s wave launched on a container that had already served the 66-min smoke could never fit under a cap sized for \"two 2.2 h arms\") and keith_ctx12 (+76.6 min, container age 83 min, external cause not recoverable locally; same Modal signature as the events the notebook previously voided). Two waves with an out-of-band KV-scheduler thrash: ctx16 (1,908 preemptions, 0.93/request) and nr16 (1,652, 1.12/request) vs 0.05-0.3/request on every 32k conc-28 wave on disk. One wave killed by ENOSPC (132152, correctly VOID). One empty directory (081755). The cadence and fx waves are clean apart from the regime-constant class (preemptions 59/65; end-of-clock timeouts; coverage 100-101%).\n\nQ2 — DID A PRE-REGISTERED VOID RULE FIRE AND GET WAIVED. Yes, on every one of the seven readable waves, and the three clauses behave very differently. (a) \"request errors > 0\" is inoperable as written: every `error` row in all six 25-game waves is a harness-generated end-of-clock ReadTimeout (read timeout shrunk to remaining clock, values 0.78-147 s); genuine server faults are recorded as HTTP status 500/502, and those occurred in budget (24), ctx12 (25 + 1) and ctx16 (1). (b) \"preemptions > 0\" is also inoperable as written: it fires on every 32k conc-28 wave including all four Modal pooled-base draws (142/109/112/170, run_regime_wave.py:1375 applies it wave-wide to every run). The notebook's stated waiver rationale is \"~110-180 is a regime constant\" (STATUS:630, 695, 982-983); that rationale does not cover ctx16/nr16 at 10-30x the band, and the notebook itself flagged carry75's 386 as \"2-3x\" (STATUS:824). (c) \"length-finish > 1%\" fired on 3 (cadence), 4 (fx) and 6 (budget) runs and is never mentioned anywhere. Waivers were explicit for cadence (STATUS:981-984), fx (1086) and probe (630); silent for budget, ctx12, ctx16 and nr16 — a grep of STATUS 1242-1621 for restart/preempt/void/500/container/lost-track returns nothing.\n\nQ3 — IS THE WAIVER DEFENSIBLE / SELECTIVE. Uniform waiver of the two inoperable clauses is defensible and in fact necessary (a strict reading voids the base pool too). What is not defensible is that the one event class that IS a real fault — a container dying under in-flight requests — has been handled three ways with no written distinction: VOID + redraw for yield900-cb (STATUS:565-575, a draw tracking BELOW base) and for the kv10 launch (STATUS:361); silently read for ctx12 (negative) and silently read for budget 3x (the only positive arm in the set). The plausible distinction (harness recovered in 4-7 min vs did not) is never stated. So the data do not show \"selectively voiding positive arms\" — the asymmetry runs the other way: the positive arm's faults were waived without disclosure and a negative-tracking arm's fault was voided. The pattern is \"void when the read is ambiguous, stay silent when the verdict already looks clear\", which is discretion, not pre-registration.\n\nQ4 — VERDICTS TO DOWNGRADE. No level verdict flips. (1) keith_budget 3x: keep 56 in the 46-59 band (53 levels at +290 min pre-restart per logs/budget3x_clean_20260910.log:1345, 3 clears after), but every telemetry-based claim (cadence \"held\" at 128.8 s, the summary VLLM row, actions/call split, MTP) is a two-regime mixture or garbage, the pre-restart server telemetry is unrecoverable, and the wave must be tagged \"instrument fault, waived\"; any redraw must patch the 6-h cap and the O(n²) state-file writer first. (2) keith_ctx12: DEAD is robust (11/12 clears by +43 min, outage cost 5.9%), but under the notebook's own precedent this wave is VOID-redraw-once, and all VLLM-line numbers (e2e 67.5, queue 53.6, \"preemptions 0\", MTP 59.7) cover only the post-restart 40% of calls. (3) keith_ctx16: DEAD survives; the \"calls gate PASS at 81.8\", the elasticity/Route-A arithmetic and the \"smooth gradient\" claim all use a middle point measured under a scheduler thrash absent at 32k and 12k and undisclosed — downgrade those mechanism claims. (4) keith_nr16: DEAD survives; the e2e/calls chain is measured under 1,652 preemptions (relative to ctx16 it holds, absolute call counts do not). (5) pooled-base/sigma rule: add that the pool is all-VOID under the written rule, that the exclusion of the yield900-cb 31 is discretionary under the same rule that let ctx12 and budget through, and that the flight wave (09-03) also had a restart in this lineage. (6) sizing, (7) corpus, (8) triage, (9) model-axis, (10) cadence/fx, (11) A2 kill runs (all 6-game runs: 0 preemptions, 360/360 coverage), (12) live ledger: no instrument-based downgrade; the cadence wave's wallclock_s field is wave-relative but the per-game shim spans (1,766-1,813 s) confirm every game got its clock.\n\nRECOMMENDED RULE (so it can actually be enforced): VOID a wave if any HTTP 5xx or non-clock error > 0, or process_start_time_seconds differs between before/after, or preemptions per request fall outside 0.02-0.35, or any game's shim span < 0.99 × clock; count harness end-of-clock ReadTimeouts separately and never as errors; report preemptions per request on every non-32k window.",
 "findings": [
  {
   "severity": "weakens",
   "affected_claims": [
    "keith_budget 3x (verifier 1): cadence held / median e2e 128.8 s; VLLM row; 'BUDGET CONVERTS' framing",
    "sizing (verifier 6) wherever it borrows budget-wave telemetry"
   ],
   "finding": "The 3x clean wave was served by two containers: metrics_before.prom is the smoke container (process start 66.6 min before wave t0, 524 requests = the 3-game smoke), metrics_after.prom is a fresh process started at +297.3 min. The old container's age at that moment was 363.9 min = 6.07 h, i.e. the rig's own lifetime reaper fired (MAX_LIFETIME_S = 6 h, os._exit(0) with no drain). The restart was foreseeable from the code: a 23,760-s wave launched on a 66-min-old container cannot fit a cap sized for 'two 2.2 h arms + cold start'. Consequences: 3,083 of 4,244 calls have no server-side telemetry; the summary VLLM row (661 requests, e2e 181.9 s, queue 162.9 s, preemptions 266) is new-container counters minus a smoke-era snapshot and is unreadable; 24 HTTP 500 'Server has lost track of input' at +295-297 min across 24 games; 215-s gap; post-restart e2e median 104.2 s vs 139.0 s pre. Levels verdict (56, band 46-59) survives because 53 levels were on the board at +290 min. The write-up (STATUS:1242-1342) does not mention any of this.",
   "evidence": "offkaggle/results/20260910-192514-regime-keith_budget/metrics_before.prom process_start_time_seconds 1789057121.84 (identical to 20260910-181717-regime-keith_budget/metrics_after.prom) vs metrics_after.prom 1789078957.04; requests_shim.jsonl (24 rows status 500 at +295-297 min; max start gap 215.2 s at +297.0 min; 3,083 ok rows before / 1,161 after the new process start); offkaggle/modal_flashnext_serve.py:293 (MAX_LIFETIME_S = 6*3600, comment 'two 2.2 h arms + cold start'), :451-459 (_reaper -> os._exit(0)); logs/budget3x_clean_20260910.log:1345 (levels=53 at +290.1 min), :1349-1373 (24 x 500 'Server has lost track of input'); summary.txt line 12 (VLLM requests 661); awk 'NR>=1242&&NR<=1342' REGIME_WAVE_STATUS.md | grep -i 'restart|preempt|void|500|container' -> nothing."
  },
  {
   "severity": "weakens",
   "affected_claims": [
    "keith_ctx12 (verifier 2): 'preemptions 0'; all VLLM-line numbers; the DEAD-vs-VOID reading"
   ],
   "finding": "The ctx12 container died mid-wave: 25 HTTP 500 'Server has lost track of input' at +75-77 min (one per game, 25/25), zero call starts from +76.6 to +83.2 min (398-s gap), a new vLLM process at +79.7 min, plus one 502 at +99 min. The old container was only 83.2 min old, so this is NOT the 6-h reaper and not the 15-min idle scaledown; the cause is not in any local file. 1,669 pre-restart calls have no server telemetry; 'preemptions 0' and every VLLM summary number describe only the post-restart 1,081 requests (40% of calls). Clock lost per game (failed call -> first OK response) mean 465 s = 5.9% of 7,920 s, and the game clock kept ticking (wallclock 7,920-7,970). The dead verdict is robust on the verifier's evidence (11/12 clears by +43 min), but the notebook voided the identical event class on 09-07 (yield900-cb 'server died at ~+120 min' -> VOID, STATUS:565-575) and on 09-03 (kv10 launch preempted -> VOID, STATUS:361); the ctx12 write-up (STATUS:1342-1446) neither voids nor mentions it.",
   "evidence": "offkaggle/results/20260911-075839-regime-keith_ctx12/metrics_before.prom process_start 1789106349.32 vs metrics_after.prom 1789111341.28 (delta 83.2 min; wave t0 +79.7 min); requests_shim.jsonl: 25 rows status 500 at +75-77 min, 1 row 502 at +99, max start gap 398.1 s at +76.6 min, 25 redirected rows (303 x2-3) at +76.6 waiting 398-473 s all status 200; logs/ctx12_20260911.log:334-358 (25 x 'modal-http: internal error: status InternalFailure: Server has lost track of input'), :359-403 (calls frozen at 1669 from +78 to +82 min); results.json games[].wallclock_s 7920-7970; offkaggle/modal_flashnext_serve.py:293,614 (6-h cap; scaledown_window = 15 min idle); REGIME_WAVE_STATUS.md:565-575, :361 (precedents); awk 'NR>=1330&&NR<=1621' REGIME_WAVE_STATUS.md | grep -i 'restart|preempt|void|500|container|lost track' -> only an unrelated 'died'."
  },
  {
   "severity": "weakens",
   "affected_claims": [
    "keith_ctx16 (verifier 3): calls gate PASS at 81.8; 'smooth gradient 2.76 -> 1.35 -> 0.27'; Route A closed by interpolation",
    "keith_nr16 (verifier 4): e2e 90 -> 133 s and calls/game 81.8 -> 59.2 chain"
   ],
   "finding": "Both 16k-window waves ran under a KV-scheduler thrash that no 32k wave shows: ctx16 1,908 preemptions on 2,056 requests (0.93/request), nr16 1,652 on 1,480 (1.12/request), versus 65-386 per wave (0.05-0.24/request) on every 32k conc-28 wave on disk and 0 on the 12k post-restart container. Coverage is otherwise clean (100.5% / 100.1%, same process, no restart; ctx16 one 502 at +69 min). The per-run VOID tag 'preemptions=1908/1652' fires on 25/25 runs in both summaries; the notebook's standing waiver rationale is that ~110-180 preemptions are a regime constant (STATUS:630, 695, 982-983), which does not cover 10-30x that band, and the notebook itself treated carry75's 386 as notable (STATUS:824). Neither the ctx16 nor the nr16 write-up mentions preemptions at all. Dead verdicts survive (levels 21/20 are far below any comparator), but any inference that uses ctx16's calls/game as a clean 16k point — the calls gate, the elasticity arithmetic, the three-point 'smooth gradient' and 'no safe window' — has an undisclosed instrument confound in its middle point.",
   "evidence": "grep vllm:num_preemptions_total offkaggle/results/20260911-111640-regime-keith_ctx16/metrics_{before,after}.prom (0 -> 1908) and .../20260911-143210-regime-keith_nr16/... (0 -> 1652); sweep over all */metrics_*.prom (base 09-02 142, retry 109, yield900 112/181/170, probe 121, carry75 386, fx 65, cadence 59, kv10 4); summary.txt line 14 WALL void 25 {...'preemptions=1908'} / {...'preemptions=1652'}; REGIME_WAVE_STATUS.md:630, 695, 982-983, 824; awk 'NR>=1446&&NR<=1621' REGIME_WAVE_STATUS.md | grep -i preempt -> nothing."
  },
  {
   "severity": "weakens",
   "affected_claims": [
    "pooled six-draw base / sigma rule (verifier 5)",
    "live ledger pooling and the yield900-cb exclusion (verifier 12)",
    "every 'DEAD by rule' verdict that cites the VOID clauses as having been checked"
   ],
   "finding": "The pre-registered per-run VOID rule (STATUS:470, 622; run_regime_wave.py:1365-1378) is inoperable as written on this rig, so it has been applied by discretion. (a) 'request errors > 0': every `error` row in all six 25-game waves is a harness-generated end-of-clock ReadTimeout (the runner shrinks the read timeout to the remaining clock; recorded values 0.78-147 s; cadence's 15 fall exactly at its five 1,767-s batch ends). Real server faults are stored under `status` (500/502), not `error`. (b) 'preemptions > 0' is applied wave-wide to every run and fires on every 32k conc-28 wave including all four Modal pooled-base draws (142/109/112/170) — a strict reading voids the base pool. (c) 'length-finish > 1%' fired on 3 (cadence), 4 (fx) and 6 (budget) runs and is never mentioned. Waivers were explicit for cadence/fx/probe and silent for budget, ctx12, ctx16, nr16. The one clause-independent fault class (container death under load) was VOIDed for yield900-cb (a draw tracking below base) and the kv10 launch, but silently read for ctx12 (negative) and for budget 3x (the only positive arm in the set) — the opposite of selectively voiding positive arms, but still discretionary: the recovered-vs-not distinction that could justify it is nowhere written.",
   "evidence": "run_regime_wave.py:1365-1378 (void_reasons: request_errors>0, wave_preemptions truthy, length_share>0.01), :659 (VOID_LENGTH_SHARE=0.01); REGIME_WAVE_STATUS.md:470, 622, 630, 695, 981-984, 1086, 565-575, 361; requests_shim.jsonl `error` text in all six waves ('Read timed out. (read timeout=0.78..147.42)'; zero non-timeout errors); summary.txt line 14 WALL of each wave (length_finish_share tags: cadence r11l .043/tu93 .023/vc33 .040; fx cn04 .038/s5i5 .057/su15 .018/tu93 .036; budget ka59 .012/re86 .012/sc25 .017/sk48 .017/su15 .011/tu93 .029); preemption sweep of all */metrics_after.prom."
  },
  {
   "severity": "cosmetic",
   "affected_claims": [
    "cadence arm (verifier 10): per-game clock honoured",
    "fx wave: instrument cleanliness"
   ],
   "finding": "The cadence and fx waves are instrument-clean by every measure: same process before/after, coverage 101.1% (the 15 timed-out calls finished server-side) and 100.0%, preemptions 59 and 65 (in band), no 5xx, largest start gaps 68 s and 51 s. results.json wallclock_s in the cadence wave is wave-relative (1,768 / 3,535 / 5,302 / 7,069 / 8,839 s) rather than per-game, but per-game first-to-last call spans are 1,766-1,813 s, so every game received its 1,767-s clock (tr87 alone in batch 5 at +117.8 min). fx's 613 'redirected posts' are Modal 303 legs on calls over ~150 s that all returned 200 — a property of the e2e distribution (cadence has 0 because its e2e is 27 s), not a fault.",
   "evidence": "offkaggle/results/20260909-193735-regime-keith/{metrics_before,metrics_after}.prom (process_start 1788975586.55 both; request_success 1 -> 1435; preemptions 0 -> 59); requests_shim.jsonl per-game spans (python over t and elapsed_s); results.json games[].wallclock_s; offkaggle/results/20260910-081906-regime-keith_fx/requests_shim.jsonl (613 rows redirect_codes [303], final_path /v1/chat/completions, status 200, elapsed >= 150.5 s)."
  },
  {
   "severity": "cosmetic",
   "affected_claims": [
    "completeness of the 09-09..09-11 wave inventory"
   ],
   "finding": "Two directories in the glob are not readable waves and were handled correctly: 20260910-081755-regime-keith_fx is an empty directory (aborted launch, no files); 20260910-132152-regime-keith is the first 3x attempt, killed at +276 min after ENOSPC crashed 17/25 games (results.json status 'running', 0 games finished, no metrics_after.prom, 2,937 clean calls, 868 redirect legs), and the notebook declared it VOID with a root cause (O(n^2) state-file writes). The smoke 20260910-181717-regime-keith_budget (3 games, conc 3, 3,600 s) is clean (522/522, 0 preemptions) and is the container that the 3x clean wave then inherited at age 66 min.",
   "evidence": "ls -la of each directory; offkaggle/results/20260910-132152-regime-keith/results.json (status 'running', games []); requests_shim.jsonl row counts; REGIME_WAVE_STATUS.md:1174-1200; offkaggle/results/20260910-181717-regime-keith_budget/metrics_after.prom process_start 1789057121.84 = 20260910-192514.../metrics_before.prom."
  }
 ],
 "commands_run": [
  "ls -d offkaggle/results/20260909*-regime-* offkaggle/results/2026091*-regime-*; ls -la of each of the 9 directories",
  "head/python3 over offkaggle/results/20260911-075839-regime-keith_ctx12/requests_shim.jsonl (key union) and results.json (top-level keys, metrics block, shim block, one game record)",
  "grep -n -i void offkaggle/run_regime_wave.py; sed -n 410,420p / 655,662p / 1365,1400p offkaggle/run_regime_wave.py (wall_reads VOID rule, VOID_LENGTH_SHARE)",
  "grep -n -i -E 'void|preempt' offkaggle/REGIME_WAVE_STATUS.md; sed -n 466,476p / 618,632p / 690,696p / 976,990p / 1170,1200p / 1318,1330p / 565,580p / 358,364p / 302,308p",
  "grep -n -E '^  (geometry|SCORE|CADENCE|VLLM|WALL|IDENTITY|CLIENT|ARM KNOB|TURNS)' <each wave>/summary.txt",
  "python3 scratchpad/audit.py: per wave parse metrics_before/after.prom (process_start_time_seconds, vllm:request_success_total by reason, vllm:num_preemptions_total, running/waiting), requests_shim.jsonl (rows, status-200, error rows and kinds, non-200 codes and times, redirect rows/legs, largest and second-largest start gap, prompt_tokens median, finish reasons, length share, pre/post-restart splits and e2e medians), results.json (geometry clock/conc, per-game wallclock min/max/median, states, levels, started/ended, shim block) -> scratchpad/audit.json",
  "python3: redirect_codes / final_path / status / elapsed distribution of redirected rows (fx, budget, ctx12, nr16); raw status-500/502 rows and their games/elapsed; ctx12 redirected-row timing",
  "ls -la logs | grep 2026091|20260909|ctx|nr16|budget|fx|cadence; grep -n -i -E 'lost track|500|restart|Container|preempt|\\[regime\\] \\+7[0-9]|\\+8[0-9]' logs/ctx12_20260911.log; grep on logs/budget3x_clean_20260910.log for +29x/+30x min and 500s; grep logs/budget_smoke_20260910.log VLLM line",
  "python3 sweep of every offkaggle/results/*/metrics_before.prom and */*/metrics_before.prom vs metrics_after.prom: process_start, preemption delta, request_success delta, shim status-200 count, levels, clock, conc (33 waves)",
  "awk 'NR>=1330 && NR<=1621' and 'NR>=1242 && NR<=1342' offkaggle/REGIME_WAVE_STATUS.md | grep -n -i -E 'restart|preempt|void|500|container|lost track|died|outage|lifetime|gap'; grep -n '^## RESULT' headers",
  "grep -n -E 'timeout|scaledown|max_containers|lifetime|21600|6 h' offkaggle/modal_flashnext_serve.py; sed -n 451,470p offkaggle/modal_flashnext_serve.py (lifetime reaper)",
  "python3: per-game clock lost to each outage (failed-call start -> next OK call start, and -> first OK response) for budget and ctx12; old-container age at new process start; wave t0 UTC; cadence per-game first-call offset and first-to-last call span",
  "python3: classify every `error` row in the six 25-game waves by read-timeout value (all ReadTimeouts, 0.78-147 s; zero non-timeout errors); grep vllm:request_success_total in 20260910-192514-regime-keith_budget/metrics_before.prom (524 stop = smoke container)"
 ]
}
```

---

## agent-14

```json
{
 "angle": "MODEL AXIS AND WHAT THE BRAIN IS — which open model can run on the Kaggle 96 GB box, whether training/test-time specialisation of it is feasible and worth it, and a serving gate before any training.",
 "thesis": "The brain is fixed and it is the whole gap. The stock duck loop shape reaches 100 % RHAE on the public 25 with Opus 5 (docs/HANDOFF-2026-08-29-the-model-axis.md, §2 AVO/VISTA table) and ~7-8 with Flash-Next; the Kaggle leaders are confined to the same 96 GB RTX PRO 6000, and every public artefact ≥4 on the board is the same Flash-Next NVFP4 + June duck stack (docs/research-2026-09-08/R-field-refresh-0908.md; our own base 3.25/2.58/4.31/2.45/4.01/2.73 = verifier 12, Kaggle CLI). Nothing that fits ≤85 GiB beats Flash-Next per call: the 09-08 memory-blind sweep found none, and my scan today of September uploads confirms it — DeepSeek-V4.1-Flash NVFP4 is 386 GiB, GLM-5.3-Flash-NVFP4-MTP 181 GiB, GLM-4.5-Architect-106B-A12B 64 GiB but a prior generation, Qwen3.5-122B-A10B NVFP4 (just uploaded to Kaggle by a competitor) also prior generation, MiniMax-H3 is a video model; vLLM's expert-offload PR #37190 is still open, single-stream and LoRA-incompatible, so the bigger MoEs cannot be served at 8 seqs by Nov 2. So the 7.5-11 tier is not 'a better model we can get'; it is private harness work on this brain class (field refresh ranking 1-3), which is another strategist's angle. What the model axis CAN do is narrower than the plan assumed and cheaper than the 09-10 closure claimed: (a) the only live step this campaign ever produced (1.94→3.25) was the SAME model allowed to reason without caps — 53 long un-truncated calls vs 125 short ones (docs/SEARCH-2026-09-02-step-change-ranking.md:17-24) — so 'what the brain is' includes how much it is allowed to think, and any training that compresses deliberation (measured −18.9 % gen tokens under imitation on the 27B, docs/HANDOFF-2026-08-10-three-levers-closed.md:44-48) attacks the very lever that worked; (b) the served checkpoint keeps every attention/linear-attention tensor in bf16 (RadixArk model card; config.json quantization_config.ignore lists *.self_attn.*, *.linear_attn.*), and those 441 tensors live in 4 of 206 shards totalling 14.9 GiB (model.safetensors.index.json + HF tree, fetched today), so an attention-only LoRA merges into bf16 shards with NO requantisation and NO vLLM-LoRA support, ships as a ~15 GiB Kaggle overlay exactly like the competitor's ebinan92/qwen38-flash-next-nvfp4-mtp1-hybrid dataset (manifest read today), and the '≥5×80 GB / $400+ / 20-50× over budget' closure (memory arcagi3-model-axis-blocked-on-scale.md) was wrong on arithmetic (180 B on disk incl. a 51 B CPU-offloadable PLE table; 257 GB non-PLE = 4×H100 or 2×B200 at $12.5-16/h; verifier 8) and wrong on mechanism (the blocker is unproven training tooling for Qwen4Exp, not scale). Against that: three served-and-measured SFT negatives on the 27B (memory arcagi3-finetune-beats-base.md: 0.89→0.10 and 0.89→0.01 on dev with positive serve proof), the −18.9 % deliberation finding, yuran986's GRPO on Qwen3-8B ('reward rose, capability did not', docs/research-2026-09-10/R-field-and-harness-research.md:37-40), and one public +0.7 LB STaR LoRA at n=1 on a 27B base (field refresh). Honest prior for a step from training: ~15-20 %. The correct model-axis program is therefore one gated bet (serving gate → corrected corpus → attention-only LoRA → deliberation-signature kill → rig read), a $0 standing brain-watch with a fit rule, and explicit closure of TTT-per-game, in-context specialisation and CPU-offloaded giants by arithmetic, with the remaining weeks and quota reserved for Track D and the harness angle.",
 "plan": [
  {
   "step": "0. Serving gate M2 on the rig: prove a modified-attention Flash-Next checkpoint serves under the exact keith V14 argv",
   "what": "On Modal (offkaggle/modal_flashnext_serve.py, add a MODEL_DIR/local-snapshot override next to the pinned snapshot_download at :332; keep config.json byte-identical because _verify_model_dir only hashes config.json per offkaggle/KEITH_REGIME.md §1): (i) round-trip the 4 attention-bearing shards (model-bf16-00001/00010/00011/00012, 14.9 GiB) through safetensors with identical values and serve; (ii) apply a deliberate tiny edit (scale one layer's o_proj by 1.01) and serve again. Compare temperature-0 logprobs on 20 fixed prompts against the untouched base (the ab-wmr serving-identity method, docs/HANDOFF-2026-08-10-three-levers-closed.md:42-43). Also package (i) as a Kaggle overlay dataset in the ebinan92 mtp1-hybrid layout (MANIFEST with sha256 per file) and dry-run the mount on a CPU commit.",
   "why_it_should_move_the_lb": "It moves nothing by itself; it removes the reason every previous fine-tune died (memory arcagi3-silent-serving-failures-2026-07-31.md: 'an env toggle is not a shipped arm') and replaces the pre-registration's assumed 'merge + NVFP4 requantise' with a path that needs neither — routed experts are untouched, attention is bf16 (RadixArk model card; config.json ignore list). Without this gate any training spend is unreadable.",
   "cost": "$10-15 Modal (RTX PRO 6000 $3.03/h × ~3 h incl. two cold loads) + ~1 day; 0 Kaggle GPU hours, 0 slots.",
   "first_falsifiable_test": "Identity round-trip: 20/20 prompts give bit-identical temperature-0 logprobs vs base AND the vLLM startup lines match KEITH_REGIME.md (KV 105,202 tokens, 81.8→79.4 GiB load). Perturbed shards: logprobs differ on ≥19/20 prompts and the server still boots with MTP-3 acceptance in the 2.7-2.9 tok/forward band (verifier 6).",
   "kill_criterion": "The pinned vLLM fork refuses a re-saved shard set, or identity logprobs differ, or the overlay cannot be mounted alongside the keith datasets on Kaggle → the model axis is blocked at serving (report as engineering, not as evidence about training) and steps 2-5 are cancelled.",
   "expected_lb_gain": "0 directly; it is the precondition for steps 2-5."
  },
  {
   "step": "1. Rebuild the STaR corpus with the builder bug fixed, stock-prompt arms only, reasoning kept, held-out by game",
   "what": "Fix src/modelaxis/build_corpus.py:154 (`step = steps[idx]` maps the idx-th CALL to the idx-th TURN HEADER; only 277/6,731 first-of-turn calls got the right step, 1,288 cleared-level calls were dropped and 4,573 continuation calls excluded — verifier 6). Rebuild from the stock-prompt waves only (keith 09-02, retry, yield180/900, kv10, halfconc, fx, cadence, budget3x 56-level wave, ctx waves excluded) so the targets match the shipped prompt; keep the [THINKING] block verbatim in every target (≥75 % reasoning data per Unsloth's Qwen3.8 guidance; the −18.9 % deliberation compression is the known failure mode); apply the 'level field is the level AFTER the action' rule from memory arcagi3-stage0-reopened-flashnext.md; hold out cd82, cn04, ft09, sk48, wa30 as in corpus-manifest-m0.json; drop records >12k tokens. Commit the manifest with per-arm counts.",
   "why_it_should_move_the_lb": "Every prior fine-tune negative in this campaign had a data defect discovered afterwards (63 % failure runs labelled success in July, memory arcagi3-finetune-beats-base.md 'DATA-QUALITY DISCOVERY'; the current M0 has ~88 % wrong step/level fields and 72 % non-base-prompt records). Training on the shipped-prompt distribution with correct credit assignment is the only version of STaR whose failure would actually be informative.",
   "cost": "$0, ~1 day; no GPU.",
   "first_falsifiable_test": "Re-run the fixed builder and verify on 3 transcripts by hand that each kept call's level equals the level the events.jsonl shows at that analysis_step; expect 3-6k records (the corrected first-of-turn population alone is 3,350 per verifier 6). Report the exact count, tokens, and per-arm split.",
   "kill_criterion": "Fewer than ~2,000 records from stock-prompt arms after the fix (the pre-registration's M0 floor) → the corpus is too thin; do not train, fold the effort into Track D.",
   "expected_lb_gain": "0 directly; determines whether step 2 is worth running."
  },
  {
   "step": "2. Training feasibility gate M1: attention-only LoRA on the bf16 base with experts frozen and the PLE table on host RAM",
   "what": "Rent 2×B200 (180 GB each, $6.25/h per verifier 8's Modal pricing fetch) or 4×H100 ($3.95/h each); load Qwen/Qwen3.8-Flash-Next in transformers ≥5.8 (config.json transformers_version 5.8.0.dev0; image has 5.15.1) with device_map placing the 102 GB bf16 PLE embedding table on CPU and the 257 GB non-PLE weights across GPUs; PEFT LoRA r=16 on q/k/v/o and the linear-attention in-projections only (2.70 B target params, verifier 8), experts/routers/PLE/MTP frozen; gradient checkpointing, seq ≤12k, batch 1, 1 epoch over step 1's corpus; export the merged attention tensors as the 4-shard overlay from step 0. Frameworks to try in order: plain transformers+PEFT, then ms-swift/LLaMA-Factory (Qwen's README names Unsloth/Swift/Llama-Factory as the supported paths; Unsloth's docs cover only the 27B and say MoE 4-bit QLoRA is not recommended — so bf16 base, not the NVFP4 experts, is the training substrate).",
   "why_it_should_move_the_lb": "This is the one place the campaign has never been: a fine-tune of the model that actually flies, on its own winning trajectories, with credit assignment fixed. The mechanism is the same as the public STaR LoRA (+0.7 LB on a 1.25 base, n=1, docs/research-2026-09-08/R-field-refresh-0908.md) and Tufa's own 'better base models were the main driver'. Attention-only is what a frozen-expert MoE allows and is where behaviour (tool-call format, when to act) lives.",
   "cost": "$60-120 for one successful run including ~2 debug sessions (2×B200 $12.5/h × 4-6 h; the 360 GB HF pull is ~15-30 min on Modal); hard cap $250 for the lane. This is ~3-8× a rig wave, not 20-50×.",
   "first_falsifiable_test": "A forward+backward step completes on one 8k-token record with loss finite and PLE lookups served from CPU; then 50 optimisation steps with monotone-ish loss and an adapter that, merged into the 4 shards and served through step 0's path, changes logprobs on the 20 fixed prompts.",
   "kill_criterion": "The pre-registered stop rule from docs/research-2026-09-09/PREREG-model-axis.md, actually exercised this time: if no framework runs a training step on Qwen4Exp within two working sessions (≤$120 spent), stop and record 'blocked on training tooling for Qwen4Exp' — not 'blocked on scale'. Also kill if the merged overlay fails step 0's identity/perturbation check.",
   "expected_lb_gain": "0 directly; unlocks step 3."
  },
  {
   "step": "3. Behavioural read: 3-game held-out kill test with a deliberation-signature gate, then a 25-game rig wave under the corrected statistics",
   "what": "Serve the merged overlay under the V14 regime on the Modal rig (same conc 28 / 7,920 s geometry). Kill test first: the 5 held-out games minus two, 2 draws (~$3): read levels AND generated tokens per call and actions per call from requests_shim.jsonl against the base's 1,293-1,330 tok/call and 2.76 actions/call. Then one 25-game wave (~$9) read against the 8-draw flat base 38.5 with sd ~3.5 (cross-cutting statistics audit: sd 2.34 is unsupported), reporting trained-vs-held-out games separately. If the wave reads ≥46, a counterbalanced second draw before any 'step candidate' label (two-wave mean ≥45 = candidate; ≤42 dead) per the audit's recommended rule.",
   "why_it_should_move_the_lb": "Levels are 98 % of our loss (docs/PLAN-2026-09-08-revised-plan-to-7plus.md §2) and +1 level/game ≈ +2.8 LB at the 0.40 calibration; a STaR adapter that transfers to held-out games is the only model-axis lever with a published positive on this competition. The deliberation gate is there because imitation compressed reasoning by 18.9 % on the 27B and the clean 27B fine-tune regressed 90× by flailing (memory arcagi3-finetune-beats-base.md) — if that signature appears on Flash-Next we stop before spending a wave.",
   "cost": "$3 + $9 (+$9 for the redraw only if ≥46); 0 slots.",
   "first_falsifiable_test": "Kill test: held-out levels ≥ base on the same games, gen tokens/call within −10 % of base, no-tool-call share <1 %, actions/acting-call not collapsed (base ~5). Wave: total levels vs 38.5, with the held-out subset ≥ its base mean.",
   "kill_criterion": "Gen tokens/call down ≥10 % or actions/call down ≥25 % at the kill test = the deliberation-compression signature → DEAD, do not run the wave. Wave ≤42 → dead. Gain concentrated on trained games with held-out ≤ base → memorisation → dead (the pre-registration's own rule; public-25 does not predict the LB, memory arcagi3-review-2026-08-17-codex-audit / HANDOFF-2026-08-29).",
   "expected_lb_gain": "Most likely 0 (P(pass) ~15-20 %). If it passes the two-wave rule, the analog is +35-55 % levels → 3.2 → 4.3-4.9 live; EV ≈ +0.2 LB."
  },
  {
   "step": "4. Standing brain-watch with a fit rule, and explicit closure of the other model-axis ideas by arithmetic",
   "what": "A $0 weekly script (curl huggingface.co/api/models?search=NVFP4&sort=createdAt and kaggle datasets list -s nvfp4 / -s flash-next) applying the fit rule: safetensors ≤85 GiB after removing any CPU-offloadable embedding table, vLLM support in the pinned image or a newer vllm/vllm-openai tag, MTP available, and ≥ Flash-Next on Toolathlon/SWE-Pro/LCB. Any hit is measured on the rig for $9 by the byte-copy/regime method (offkaggle/absorb_kernel.py + modal_flashnext_serve.py). Today's scan: no hit. Record as CLOSED by arithmetic, with the numbers: per-game test-time training (12 h / 110 games = 6.5 min per game for everything; NVFP4 experts are not trainable in-box and no bf16 base fits the box); in-context specialisation (persistent prose knowledge flat at 41 lv, memory arcagi3-track-a1-carry-compact.md; window binding at 16k/12k, verifiers 3-4; Sensi v2 curriculum TTL solved 0 levels, arXiv 2603.17683); Qwen3.8-27B under the V14 regime (below Flash-Next on all 12 shared benchmarks, R-model-axis-0908; calls elasticity 0.2 means even +50 % calls from its 10× KV buys ≈+10 % levels, less than the per-call deficit); CPU expert offload of GLM-5.3-Flash/DeepSeek-V4.1-Flash (vLLM PR #37190 open, single-stream, LoRA-incompatible; 110 games × 55 calls × 1.3k tokens in 12 h needs ~180 tok/s aggregate, an order of magnitude above offloaded decode).",
   "why_it_should_move_the_lb": "The one live step in this campaign came from adopting a released model+regime within days of its appearance (keith V14 byte-copy → 3.25, memory arcagi3-step-change-search-2026-09-02.md). A better-fitting release between now and Nov 2 is the only model-axis event with step potential, and it is only capturable if someone is looking; the closures stop the campaign re-spending on ideas the arithmetic already kills.",
   "cost": "$0 standing; $9 per candidate wave; ~2 h to write.",
   "first_falsifiable_test": "The scan runs and reproduces today's result (DeepSeek-V4.1-Flash NVFP4 386 GiB, GLM-5.3-Flash-NVFP4-MTP 181 GiB, GLM-4.5-Architect 64 GiB, Qwen3.5-122B-A10B on Kaggle, Qwen3.8-Flash-Next-FP8 152 GB — none passes). A candidate that passes the fit rule is read on the rig with the step-3 statistics.",
   "kill_criterion": "A candidate wave ≤42 levels → dead; no candidate by Oct 20 → the watch simply expires (no cost).",
   "expected_lb_gain": "0 unless a release lands; if one does, the same-day rig read decides."
  },
  {
   "step": "5. (Conditional on step 3 two-wave pass and Ahmed's explicit go) Live 3-draw of the adapter overlay",
   "what": "Stage the 15 GiB attention-overlay dataset + the keith datasets exactly as ebinan92's hybrid mounts sonpham's part-a/b/c; push a byte-copy of the V14 kernel with only the dataset_sources delta (absorb_kernel.py diff shows the same serving regime); commit on the RTX PRO 6000 with the in-kernel serve proof (startup lines + a logprob canary vs base); fly three daily slots; read by the live sequential rule the statistics audit recommends (continue until the 3-draw-mean 95 % CI excludes 3.6 or 4.0, cap 5), against the identical-bytes v4 base 2.85 ± 0.35 (n=3), not the pooled 3.22.",
   "why_it_should_move_the_lb": "This is the only step that changes the leaderboard; everything before it is designed to make the three slots readable. Two selected duplicates of the best-mean config are what the private ranking scores (memory arcagi3-audit-2026-07-26-corrections.md).",
   "cost": "3-5 submission slots, ~2.5 h Kaggle GPU per commit, $0 Modal.",
   "first_falsifiable_test": "Commit log shows the overlay shards loaded (file list + 'Model loading took ~81.8 GiB') and the canary logprobs differ from base; draw 1 inside 2.0-5.5.",
   "kill_criterion": "Draw 1 < 2.0 (below every base draw) → stop and revert to base for remaining slots; 3-draw mean ≤3.6 → dead, base stays the final selection.",
   "expected_lb_gain": "If reached: +1.0 to +1.5 LB (to ~4.3-4.7) with ~50 % chance of being real given rig-to-live mapping ~2.0-2.2 LB per rig level/game (verifier 12); overall EV of the whole model-axis lane ≈ +0.2 LB."
  }
 ],
 "evidence_cited": [
  "/Users/ahmed/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/MEMORY.md (index) and memory files arcagi3-model-axis-blocked-on-scale.md, arcagi3-track-d-absorption-ready.md, arcagi3-polyphony-priced-out.md, arcagi3-kaggle-27b-lora-works.md, arcagi3-silent-serving-failures-2026-07-31.md, arcagi3-finetune-beats-base.md, arcagi3-claude-teacher-pipeline.md, arcagi3-stage0-reopened-flashnext.md, arcagi3-campaign-log-2026-08-10-15.md:20-32 (KAT probe / Track 1 closed / brain scout)",
  "docs/research-2026-09-09/PREREG-model-axis.md (gates M0-M5, stop rule, RESULT section); docs/research-2026-09-08/R-model-axis-0908.md (fit table, no ≤85 GiB model beats Flash-Next); docs/research-2026-09-08/R-field-refresh-0908.md (leaderboard, Tufa July writeup = 27B FP8, STaR LoRA 1.25→1.94 n=1, Fususu/Rakha notes); docs/PLAN-2026-09-08-revised-plan-to-7plus.md §1-§6 (Track C decision rule, 98 % loss = depth, +1 lv/game ≈ +2.8 LB); docs/SEARCH-2026-09-02-step-change-ranking.md:14-45 (the 1.94→3.25 step = uncapped long reasoning calls, 'model swaps dead', reasoning_effort medium wrong direction); docs/HANDOFF-2026-08-29-the-model-axis.md:40-110 (AVO/VISTA 100 % with Opus on our loop shape; public-25 does not predict LB); docs/HANDOFF-2026-08-10-three-levers-closed.md:42-56 (imitation compresses deliberation −18.9 %, paired levels 7 vs 11 p=0.453); docs/HANDOFF-2026-07-25-distillation-campaign.md:1-40 (K3 corpus 392 turns, serving-verification gate); docs/research-2026-09-10/R-field-and-harness-research.md:37-40 (yuran986 GRPO on Qwen3-8B: reward rose, capability did not)",
  "offkaggle/KEITH_REGIME.md §1-§4 (pinned HF revision 7b71922…, config.json sha gate, exact vllm argv, 81.8 GiB load, KV 105,202 tokens); offkaggle/modal_flashnext_serve.py:74-75,292-293,332 (HF_MODEL_REPO/REVISION, lifetime cap, snapshot_download pin); offkaggle/flashnext_patches/radixark_nvfp4_ple_fp8.patch:1-2 (patch targets vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py); src/modelaxis/build_corpus.py:150-154 (first-call-of-turn only; steps[idx] mapping bug per verifier 6); docs/research-2026-09-09/corpus-manifest-m0.json (2,047 records, holdout cd82/cn04/ft09/sk48/wa30); grep of offkaggle/REGIME_WAVE_STATUS.md for 27B/B1 (no 27B-under-V14 arm was ever run: only narrative hits at :845,:861)",
  "VERIFIED STATE OF THE EVIDENCE (this workflow): verifier 8 (180.0 B params on disk, PLE 51.23 B, non-PLE bf16 257.5 GB, NVFP4 non-PLE 78.2 GiB, Modal H100 $3.95/h, B200 $6.25/h, RTX PRO 6000 $3.03/h, quantization ignore list, no '8B dense' antecedent, M2 stop rule never exercised); verifier 6 (M0 builder bug, 72 % non-base arms, forward-simulation not zero); verifier 1 (budget3x 56 levels, elasticity 0.17-0.21); cross-cutting statistics audit (wave-total sd ~3.5 not 2.34; one wave resolves ±7.4 levels; two-wave rule); verifier 12 (live base v4 2.85 ± 0.35 n=3; rig→LB ≈ 2.0-2.2 LB per lv/game)",
  "Commands run today: `kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 --show` (Tufa 11.04 / Third Intelligence 8.21 / NVARC3 7.69 / Franzen 7.63 / mostik.ai 7.51 / Fususu 6.91 / Kyutai 5.84); `kaggle models list -s qwen` (keithtyser/qwen3-8-flash-next-nvfp4, foysalemonshanto/qwen3-8-27b-fp8-repacked-v1, qwen-lm/qwen3-next-80b); `kaggle datasets list -s nvfp4` and `-s flash-next` (mirzamilanfarabi/122b-a10b-nvfp4-{nvidia,unsloth} = Qwen3.5-122B-A10B uploaded 09-10/11; ebinan92/qwen38-flash-next-nvfp4-mtp1-hybrid 09-04; jakubluwierski/qwen38-flash-next-fp8 152.8 GB; sayandeepbhowmick/qwen38-27b-nvfp4; aigood/qwen38-flash-next-nvfp4 119 GB; sonphamorg serving parts a/b/c); `kaggle datasets files …` for each; `kaggle datasets download ebinan92/… -f MTP_HYBRID_MANIFEST.json / README.md` (overlay replaces 3 bf16 shards + adds NVFP4 mtp.* from Inferact/…, pinned by sha256; requires sonpham part-a/b/c); `kaggle kernels list --competition … --sort-by voteCount` (top public kernels: keithtyser V14 156 votes, wuliao0 anim-base 190, none with a fine-tune)",
  "curl https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json → quantization_config.ignore = [model.embed_tokens, mtp.*, *.self_attn.*, *.linear_attn.*, *.mlp.gate*, *.mlp.shared_expert.*, *hyper_connection*, *.ple.*, model.visual.*, lm_head], transformers_version 5.8.0.dev0, arch Qwen4ExpForConditionalGeneration; curl Qwen/Qwen3.8-Flash-Next config.json (same); WebFetch of the RadixArk model card ('Attention, QSA, GDN, mHC, shared experts, routers, embeddings, LM head, vision, and all 31 MTP tensors remain BF16 and byte-identical to the source'; NVFP4 scoped to routed experts only)",
  "curl model.safetensors.index.json (34,258,807 bytes) + HF tree API, analysed in python: 296,475 tensors, 441 self_attn/linear_attn tensors in 4 of 206 shards (model-bf16-00001 1.19 GiB, 00010 0.30, 00011 9.99, 00012 3.43 = 14.91 GiB), 2 routed-expert tensors in those shards, 193 shards carry routed experts",
  "curl huggingface.co/api/models?search=NVFP4&sort=createdAt (Sept 9-11 uploads) and ?search=Qwen3.8; HF tree sizes: msuiche/DeepSeek-V4.1-Flash-NVFP4 386.4 GiB, soyrsoyr/GLM-5.3-Flash-NVFP4-MTP 181.3 GiB, jfar-z/MiniMax-H3-…-NVFP4 46.7 GiB (H3 is a video model per search), primitive-ai/Nex-N2.5-mini-NVFP4 22.3 GiB, flukethoughts/GLM-4.5-Architect-106B-A12B-NVFP4 64.4 GiB; apetersson/Qwen3.8-Flash-Next-Abliterated-Adapter (a llama.cpp GGUF rank-one LoRA edit, not a trained adapter)",
  "WebFetch github.com/vllm-project/vllm/pull/37190 (MoE expert offload: OPEN, Triton-only, LoRA-incompatible, single-stream); WebFetch github.com/QwenLM/Qwen3.8-Flash-Next (125 B + 51 B n-gram, 6 B active; 'Unsloth, Swift, Llama-Factory' recommended for SFT/DPO/GRPO; vLLM example uses --tensor-parallel-size 4); WebFetch unsloth.ai/docs/models/qwen3.8/train (27B only: LoRA >36 GB, QLoRA 24 GB; ≥75 % reasoning data to preserve reasoning) and unsloth.ai/docs/models/qwen3.5/fine-tune (122B-A10B bf16 LoRA on 256 GB VRAM; MoE 4-bit QLoRA not recommended); WebFetch arxiv.org/abs/2603.17683 (Sensi v2 curriculum test-time learning solved 0 levels); WebSearch results on Qwen3.5-122B-A10B-NVFP4 (nvidia/unsloth/RedHatAI repos) and on ARC-AGI-3 TTT literature"
 ],
 "what_the_campaign_got_wrong": [
  "The 09-10 model-axis closure (memory arcagi3-model-axis-blocked-on-scale.md; PREREG-model-axis.md RESULT) is wrong on every load-bearing number: the model is 180.0 B on disk not 131 B (verifier 8: 360 GB / 2 bytes; '131 B ↔ 360 GB' is internally inconsistent); the 360 GB includes a 102 GB PLE n-gram table that serving already CPU-offloads and training can too, so the bf16 base needs 257.5 GB = 4×80 GB or 2×180 GB, not '≥5×80 GB'; Modal lists H100 at $3.95/h so 8×H100 is $31.6/h and a realistic 2-4 GPU run is tens of dollars, not '$400+'; the '8B dense' premise it 'corrects' appears nowhere before 09-10 (the 09-08 plan already says '125B MoE, Modal multi-GPU, 1-2 weeks and real money').",
  "The closure is also wrong on mechanism: M2 assumed 'merge + NVFP4 requantise' is required. It is not. The served checkpoint quantises routed experts only; every attention/linear-attention tensor is bf16 and byte-identical to the source (RadixArk model card; config.json ignore list), and those 441 tensors sit in 4 shards (14.9 GiB). A merged attention-only LoRA is a bf16 tensor edit served by the unchanged vLLM argv, shippable as a Kaggle overlay dataset — a competitor already does exactly this shard-overlay trick for MTP (ebinan92/…-mtp1-hybrid). The lane was closed 'on scale' when the real unknown is training tooling for Qwen4Exp, and the pre-registered two-session stop rule for M2 was never actually exercised.",
  "The pre-registration's prior for STaR was stated as 'genuinely mixed', but the campaign's own served-and-measured evidence is three negatives on the 27B (0.89→0.10; clean data 0.89→0.01 with positive LoRA serve proof) plus a measured −18.9 % deliberation compression from imitation, and the field's one positive is n=1 on a different base. The prior should have been written as ~15-20 %, and the deliberation signature should have been a pre-registered kill at the 3-game stage.",
  "Nobody connected the two model-axis facts the campaign already owned: the only live step (1.94→3.25) came from letting the SAME model reason without output caps (53 long calls vs 125 short), and imitation training measurably shortens reasoning. Any fine-tune plan must protect deliberation (full [THINKING] in targets, ≥75 % reasoning data, gen-tokens/call gate) or it re-runs the July regression.",
  "The M0 corpus that 'survives' is not usable as-is: build_corpus.py:154 aligns calls to turn headers by index, so ~88 % of records carry a wrong step/level, 1,288 cleared-level calls were dropped, 4,573 continuation calls (which hold 3× the modelling behaviour) were excluded, and 72 % of records come from non-base-prompt arms (verifier 6). Training on it would repeat July's '63 % failure trajectories labelled success' lesson.",
  "'Model swaps: no servable model beats the class' was right on 09-02 and is still right today, but it was recorded as a lane kill rather than a standing fit-rule scan; competitors uploaded Qwen3.5-122B-A10B NVFP4, a 152 GB Flash-Next FP8 and an MTP-1 hybrid to Kaggle this week, i.e. the field is actively working the model/serving axis while the campaign has no watch on it.",
  "Track B1 (Qwen3.8-27B under the V14 regime with large KV) in the 09-08 plan was never run and never formally closed; with the measured calls elasticity 0.17-0.21 and the 27B below Flash-Next on all 12 shared benchmarks it is dead by arithmetic and should be recorded as such rather than left open.",
  "Verifier disagreements with the notebook that bear on this angle: the pooled base 39.33/sd 2.34 is unsupported (wave-total sd ~3.5; two of six pool draws have no per-game data on disk), so any model-axis read must use the 8-draw flat mean 38.5 and a two-wave rule; 'elasticity decays 0.40→0.21' is a base-choice artefact (0.20 vs 0.21 on a common base); the yield900 live 'dead' was decided by 0.01 on a 3-draw mean whose SE exceeds the gap between the step and dead lines."
 ],
 "biggest_risk": "That the fine-tune 'works' on the loss and the trained games and still makes the live agent worse: SFT on its own trajectories compressed the 27B's deliberation by 18.9 % and the clean 27B fine-tune completed levels in 50× more actions (memory arcagi3-finetune-beats-base.md), while the only live step Flash-Next ever gave us came from long, un-truncated reasoning. On top of that, public-25 gains routinely fail to transfer to the hidden 110 (3-5× ratio across teams), so a trained-game lift is the most likely false positive. Second risk: Qwen4Exp training tooling (PLE offload, hybrid GDN/linear attention kernels, hyper-connections) does not run in transformers/PEFT within two sessions and the lane ends at ~$120 spent with nothing measured — which is why the stop rule is placed at M1, the serving gate is bought first for $10-15, and the whole lane is capped at $250 so it cannot displace Track D or the Oct-1 quota reserve.",
 "lb_target_by_nov2": "From the model axis alone: expected ~3.3 (base family mean 3.22, most likely branch is 'no change'); ~15-20 % chance of a real 4.2-4.8 if the attention-LoRA passes the two-wave rig rule and three live draws. The 7.5-11 tier is not reachable on the model axis with any open model that fits the 96 GB box today; getting there depends on the harness/absorption angles, to which this plan hands back every unspent dollar and GPU hour."
}
```

---

## agent-15

```json
{
 "angle": "LOOP SHAPE — change who executes and when the hypothesis is re-sampled, harness-side, with the model's bytes and prompts untouched",
 "thesis": "We sit at ~3.2 because the loop converts calls into levels at an exchange rate set by what happens at a wall, and the walls are comprehension-bound, not attempt-bound: 12 of 25 public games hit the same wall in every draw (docs/research-2026-09-08/R-loss-ledger-3-NOTES.md:29), and when the 3x-budget wave tripled calls and quadrupled actions only 3 of those 12 walls fell (dc22, sb26, vc33 — my recompute of offkaggle/results/20260910-192514-regime-keith_budget/results.json against offkaggle/read_cadence_arm.py:25-26); the other 13 extra levels came from games already progressing (ft09, lp85, re86...). So 'budget converts' (56 vs 41-by-7,920 s, elasticity ~0.2, verified) is a statement about progressing games, and the leaders at 7.5-11 (~2 levels/hidden game vs our 1.4, docs/research-2026-09-09/R-harness-teardown-and-scoring.md:47-48) are passing walls we never pass. Every prompt-content lever aimed at the wall was flat or negative (evid, hypo, probe 41, carry 41, fx 35 with the correct dynamics table on 97% of turns and quoted in ~7% of turns, never in code — offkaggle/REGIME_WAVE_STATUS.md:1069-1110), and the model does not use tools it is handed (0 verifier calls in 358, memory arcagi3-a2-workspace-closed.md). The 'reactive perceive-and-act loop, 0 forward simulations' diagnosis is overstated: on the stock base draw 51% of acting snippets already run action() under a loop with terminal checks in 82% of them, mean 5.1 actions per acting turn (my count on 20260902-232148-regime-keith transcripts/events; the verifier found >=30 hand-written forward simulations in the corpus that reported zero). The model already writes small closed-loop controllers; what it never does is abandon a wrong hypothesis (41/75 stuck tails hold the key idea and never execute it; the same game fails the same way in every draw — R-loss-ledger-3-NOTES.md:84,88; docs/research-2026-09-06/R-loss-ledger-2-NOTES.md:79-81). From the loop-shape angle the harness cannot give the brain understanding, but it can do three structural things that require no behaviour change: (0) settle for $3 whether this brain can act on a model it is handed at all (oracle probe, never run on Flash-Next — docs/research-2026-09-08/R-reopen-audit-0908.md 'Oracle-model probe' row), (1) re-sample the hypothesis at the wall with a full-history wipe, the one dose the fresh-mind graft was built with and never run (submission/_throughput_v1/graft_retry.py:53-55 RETRY_CLEAR_HISTORY=1 [0]; REGIME_WAVE_STATUS.md:426-434 'remaining variant on record (not run)'; ceiling E[max of 2] +0.33 lv/game, R-loss-ledger-3-NOTES.md:28), and (2) let a proven controller run several engine steps per model call under a harness-run surprise-halt built from the effect table graft_effects already computes (the 'batching pays only with a mid-flight abort detector' hypothesis nobody has tested — docs/research-2026-09-10/R-field-and-harness-research.md §9), which targets the 20% of tails that are time-limited and the 32% of acting turns that fire a single action per 145 s queued call. Honest ceiling of this whole angle at our compute is ~+0.4-0.6 lv/game (LB ~3.8-4.5); it cannot reach the 7+ tier on this brain, which needs a different loop that only Oct-1 absorption can supply.",
 "plan": [
  {
   "step": "0. Flash-Next oracle probe — can the deployed brain ACT on a correct model it did not build?",
   "what": "Port submission/_oracle_probe/oracle_probe.py (ollama chat API at :77) to the Modal OpenAI-compatible endpoint used by the rig (offkaggle/modal_flashnext_serve.py) and run ORACLE vs CONTROL on wa30 L3 (engine-verified 82-move solution, 100-move budget; docstring :1-57), 5 clones each, thinking on, Flash-Next NVFP4 in the kv5-bf16-mtp3-c8-cg32 profile. Only the 27B/ollama pilots were ever run (results.json: qwen3:8b 0/5, qwen3:14b control 0/3); the v2 push of 08-29 was never recorded (docs/research-2026-09-08/R-reopen-audit-0908.md, 'Oracle-model probe' row: NEVER PROPERLY TESTED).",
   "why_it_should_move_the_lb": "It does not move the LB itself; it decides which half of this angle is alive. fx showed the model ignores a handed effect table (35 lv; quoted in ~7% of turns, never in code — REGIME_WAVE_STATUS.md:1069-1110), but that was a weak dose (a table appended after the tool-format lines). The oracle hands a complete model, goal predicate and current state every turn. PASS reopens 'harness models, model executes' designs (effect-table planning, avatar tracking as a plan checker); FAIL kills that class by construction on this brain and stops any further $9 waves on it.",
   "cost": "~$3 GPU (one Modal container hour), half a day of engineering; zero submission slots.",
   "first_falsifiable_test": ">=3/5 ORACLE clones clear wa30 L3 inside the 100-move budget while CONTROL clears <=1/5. Re-verify each 'won' by replaying the recorded move list through the engine, not by reading the log.",
   "kill_criterion": "ORACLE <50% clears (the probe's own pre-registered line, oracle_probe.py:29-36) => every 'harness supplies dynamics for the model to plan with' design is dead on Flash-Next; steps 2's halt detector survives (it never needs the model to read anything) but no effect-table-in-prompt or harness-planner variant is built.",
   "expected_lb_gain": "0 directly (diagnostic). Saves ~$30 of waves and a week if it fails; if it passes it re-prices step 2's upside upward."
  },
  {
   "step": "1. Fresh mind at the wall, full-history dose (graft_retry with RETRY_CLEAR_HISTORY=1, no RESET, stuck-span trigger)",
   "what": "Re-use submission/_throughput_v1/graft_retry.py: when a level is stuck the harness wipes the carried chat history (graft_retry.py:441 `agent._history_messages = []`) and the six note fields (same set stock blanks at level transitions, tool_agent.py:1113-1126), keeps the board (RESET is swallowed live anyway, api.py:316-334 per memory arcagi3-reset-lever-refuted-competition-guard.md, and it collided with a cleared level in the note-only run: sp80 L1 base 1 -> 0, REGIME_WAVE_STATUS.md:418), and lets the next turn start from the stock action_num-style grounding prompt (tool_agent.py:1226-1230). Trigger is absolute because base_actions_per_level is None in submission mode (graft_retry.py:24-28): >=60 actions AND >=12 acting turns on the current level without a clear AND >=40% of the game clock remaining; max 1 wipe per level, 2 per game; the repeated-(state,action) loop detector (A-not-B, offline GO in memory arcagi3-research-2026-08-07-human-play-sweep.md) may fire it earlier. Write it as an in-memory graft installed from the notebook's customization hook (submission/_keith_copy/push_yield900/arc3-keith-yield900.ipynb cells 12-13; offkaggle/run_regime_wave.py:302-307,637-651) so it composes with any Oct-1 absorbed kernel.",
   "why_it_should_move_the_lb": "Stuck tails are dominated by one wrong hypothesis held to the clock (WRONG-HYPOTHESIS + ANALYSIS-PARALYSIS + LOOPING = 65% of 75 tails, R-loss-ledger-3.md Q4; 41/75 hold the key idea and never execute it, NOTES:84). Retained reasoning is load-bearing (ctx12/ctx16/nr16 all dead), which is exactly why the wrong hypothesis persists: the history keeps re-asserting it. A one-time wipe at the wall is a hypothesis re-roll, not amnesia — the window refills over the next ~10 turns. Nine public games are HIGHVAR across draws (spread >=2: ar25, cn04, ft09, lp85, sb26, sc25, tn36, tr87, tu93 — R-loss-ledger-2-NOTES.md:16-17), so a second mind has real variance to draw on there; E[max of 2 draws] on the wall level is +0.33 lv/game (NOTES:28) and max-of-3 whole-game draws +0.43 (R-loss-ledger-3.md Q5a). The note-only dose (K=2.5 x baseline, with RESET) fired 5x and cleared 0/5 (STATUS:414-434), but that dose left the history intact, so the old hypothesis was still in the prompt; the full-wipe variant was recorded as the remaining dose and never run.",
   "cost": "$2.5 kill test (6 runs: tn36, sb26, tr87, cn04, sc25, lp85 x draws 2, conc 3, 7,920 s — the same geometry as the carry kill test, REGIME_WAVE_STATUS.md:662-735) then $9 per 25-game wave, two waves minimum (~$21 total); ~1 day of graft work since the graft exists; zero slots until the rig bar is met.",
   "first_falsifiable_test": "Kill test on the six HIGHVAR-wall games: wipes fire in >=5/6 runs at the pre-registered trigger; paired levels on those six games >= the mean of the same games across the 8 flat live-geometry draws (36,34,40,41,40,41,41,35 — stats audit) minus 1; and >=2 of the wiped levels clear afterwards (note-only dose: 0/5).",
   "kill_criterion": "Kill test: 0 wiped levels clear, or paired levels <= base-1, or wipes fire and the post-wipe mind re-states the identical hypothesis in >=80% of wiped levels (read the first post-wipe reasoning; that would show failures are deterministic, not sampling). Waves: use the corrected instrument (sd ~3.5 per 25-game total): a two-wave mean <=42 or <=2 of the 12 never-passed walls passed => dead; >=45 on the two-wave mean AND >=3/12 walls => step candidate, counterbalanced redraw, then 3 live draws.",
   "expected_lb_gain": "+0.2 to +0.5 LB (+4 to +10 rig levels at ~2.0-2.2 LB per rig level/game; the base->live mapping from the ledger verifier). Probability it reaches the step bar ~15%; probability of a measurable positive on the 12-wall co-primary ~35%; on the 16 LOWVAR/CAPPED games I expect exactly zero."
  },
  {
   "step": "2. Proven-controller re-execution with a harness-run surprise-halt (graft_reexec: the LLM-written snippet runs N engine steps per model call)",
   "what": "Wrap ToolAgent._run_python_tool (tool_agent.py:1453-1590). After an acting snippet returns with every action executed, board_changed on each, no terminal flag, and the observed object displacement matching the modal effect-table prediction for those actions (reuse graft_effects' segmentation-hash matcher and per-action table as a checker, never as prompt content — submission/_throughput_v1/graft_effects.py docstring), the harness re-runs the SAME code against the refreshed runtime state up to K=4 times inside the same 30 s-per-call sandbox contract (prompts.py:111), halting on: no action() fired, executed=False, board unchanged, frame hash already visited on this level, predicted-effect mismatch, level_completed/game_over/run_complete, python error, or >K. The extra executions are summarised in one compact line inside the same tool result the model already reads; no prompt text changes. Excluded: snippets that fire MOUSE with computed coordinates only once, and any level where actions already exceed 3x the human avatar median (70 per level, memory arcagi3-human-replay-dataset.md).",
   "why_it_should_move_the_lb": "One turn = one acting call = one 145 s queued round-trip (memory arcagi3-queueing-not-compute.md), and 32% of acting turns (242/750 on the base draw) fire exactly one action even though 98% of acting snippets read state and 51% already loop — the model writes controllers but pays a call per iteration. Time-limited/PROGRESSING runs are 20% of tails (+0.21 lv/game at parity, R-loss-ledger-3.md Q5b) and the 3x-budget wave shows the same progressing games clear ~13 more levels given clock; re-executing a proven controller buys clock in the execution phase for zero calls, and the freed calls go to the next wall at the measured ~0.2 elasticity. The halt detector is what distinguishes this from patch 21 (x1.40 actions, -25% levels, ungated 26-action blind batches on sk48 — memory arcagi3-two-level-wall.md:116-124) and is the untested reconcilable hypothesis in the literature (R-field-and-harness-research.md §9: TTI's h=30 collapse vs ELHPlan's replan trigger). PRO-LONG's 'scale toward 20 for proven sequences' rule (memory arcagi3-prime-agent-findings.md) is the same doctrine done by the agent; here the harness enforces it.",
   "cost": "$0 for the first test (offline counterfactual replay, 1-2 days: replay each recorded run's action list through arcengine with ONLY_RESET_LEVELS to each acting turn, re-run the transcript's snippet in the real python_tool_sandbox against the live engine k times, log every halt reason); then $2.5 kill test on six progressing games (ft09, lp85, re86, ar25, vc33, r11l) and two $9 waves.",
   "first_falsifiable_test": "Offline replay over the base draw's 779 acting snippets: >=30% of re-executions reach a board never visited on that level without a death; deaths per re-execution <=3%; re-scoring cleared levels with the exact rule (arc_agi/scorecard.py:146-207 as reproduced in memory arcagi3-scoring-is-convex-in-depth.md) loses <=5% of the wave's score from added actions; and the halt detector stops >=90% of would-be deaths one step early (measure by continuing past the halt in simulation).",
   "kill_criterion": "Any offline number below its line => dead before a dollar is spent. Kill test: paired levels on the six games < base mean, or GAME_OVERs per run > 1.5x base (0.87-1.36/run). Waves: same corrected two-wave rule as step 1 (<=42 dead; >=45 candidate), with actions-per-cleared-level ratio to baseline as a co-primary that must stay <=1.0 median (base 0.83-0.86).",
   "expected_lb_gain": "+0.1 to +0.3 LB; bounded by the action elasticity the 3x wave implies (~0.14 in actions) and it can be negative (patch 21 precedent), which is why the offline replay comes first. Probability of a step alone ~5%; its value is mostly as a composable rider on step 1."
  },
  {
   "step": "3. Compose 1+2, redraw, fly under the corrected verdict rule; keep everything absorption-compatible",
   "what": "If either arm clears its two-wave gate, run the composed arm once (counterbalanced, first-in-boot), then 3 live draws on consecutive days. Read the rig with the stats audit's rule (sd ~3.5 per 25-game total; paired per-game t against the 8 flat draws; 12-wall co-primary; ENGAGED arms reading 40-47 get a second draw before any verdict) and the live with a sequential rule (continue until the running-mean 95% CI excludes 3.6 or 4.0, cap 5 draws) instead of the 0.01-margin 3.59/3.6 read that killed yield900. Before Oct 1, verify with offkaggle/absorb_kernel.py diff that the grafts still install on whatever kernel is absorbed (grafts rebind ToolAgent methods in memory, run_regime_wave.py:60,637-651), so the loop-shape gain, if any, stacks on Track D rather than competing with it.",
   "why_it_should_move_the_lb": "Both arms act on different tails (walls vs time-limited runs) and are additive in mechanism; the rig-to-LB mapping is ~2.0-2.2 LB per rig level/game on this regime (base 1.42 lv/game -> 2.85 n=3; yield900 1.62 -> 3.59 n=3, ledger verifier), so +8-12 rig levels is +0.7-1.0 LB — right at the one-draw live detectability floor (1.6 LB at sd 0.8), which is why the live rule must be sequential.",
   "cost": "$9 composed wave + 3-5 submission slots (out of ~52 remaining before Nov 2); one day for the verdict tooling changes.",
   "first_falsifiable_test": "Composed wave >= the better single arm's two-wave mean and >=3/12 walls; live draw 1 inside 2.6-4.9 (the pre-registered base band widened for a +0.7 shift).",
   "kill_criterion": "Composed wave below the better single arm by >4 levels (interference), or the live running mean's CI excluding 4.0 after 3 draws => fly the base family for the remaining slots and put all effort into Track D.",
   "expected_lb_gain": "Cumulative for the angle: +0.4 to +1.0 LB over the base family mean 3.22 (n=6, ledger verifier), i.e. 3.6-4.2. Nothing here reaches 5+."
  },
  {
   "step": "4. Rider (only if a wave slot is free): stub the retained user-prompt boilerplate to buy calls without losing history",
   "what": "~7k of the ~27k retained prompt tokens are the repeated per-turn user boilerplate (memory arcagi3-track-a1-carry-compact.md: 'rider, not built'; the user prompt text is tool_agent.py:1161-1257, ~745 tokens per turn). Replace every retained user message older than the last two turns with a one-line stub (keep the assistant reasoning and tool results untouched). Sequences shrink ~25% at zero history loss, so more sequences fit the 5 GiB KV (calls scale ~KV^0.56 per the sizing verifier) -> ~+13-16% calls/game.",
   "why_it_should_move_the_lb": "It is the only 'buy calls' route that does not cut history, which is what killed ctx12/ctx16/nr16; at the measured call elasticity ~0.2 it is worth ~+3% levels. Sub-step by construction; included only because it is one afternoon and composes with everything.",
   "cost": "$9 one wave, half a day.",
   "first_falsifiable_test": "Mechanism gate: median prompt tokens fall >=20% with n_messages unchanged and calls/game >=62 (base 55.8); vLLM preemptions/request stay in the 0.05-0.3 band (the 16k arms' 0.9-1.1 thrash must not appear).",
   "kill_criterion": "Calls/game <60, or actions/call <2.5, or preemptions/request >0.4 => drop it; it is not worth a redraw.",
   "expected_lb_gain": "+0.05 to +0.1 LB; undetectable live, kept only as a free composable rider."
  }
 ],
 "evidence_cited": [
  "offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/artifacts/*_events.jsonl — my recompute: 750 acting turns, actions per acting turn median 3 / mean 5.14 / p90 11; 242 (32%) fire exactly one action; 37 GAME_OVERs with ~1% of actions spent re-navigating",
  "offkaggle/results/20260902T2121-keith/20260902-232148-regime-keith/transcripts/*.txt — my recompute: 779 acting snippets, 51% action() under a loop, 82% check board_changed/terminal flags, 26% multiple action() sites, 98% read runtime state, 1% search",
  "offkaggle/results/20260910-192514-regime-keith_budget/results.json vs offkaggle/read_cadence_arm.py:25-26 (WALLS) — 3x budget passed only dc22/sb26/vc33 of the 12 never-passed walls; kv10 1, fx 1, base 0",
  "docs/research-2026-09-08/R-loss-ledger-3.md (Q4 tail classes AP 33% / WH 17% / LOOP 15% / PROG 20%; Q5 counterfactuals +0.91 lv/game for +1 level on started runs, max-of-3 +0.43) and R-loss-ledger-3-NOTES.md:16-17 (HIGHVAR list), :28 (E[max of 2] +0.33), :29 (12 never-passed walls), :84 (41/75 idea-in-note), :88 (recurring patterns)",
  "docs/research-2026-09-06/R-loss-ledger-2-NOTES.md:64-81 (same game fails the same way every draw; g50t x3, sk48 x4, sp80 x3)",
  "offkaggle/REGIME_WAVE_STATUS.md:414-436 (keith_retry note-only: 5 fires, 0/5 cleared, sp80 collision; full-wipe 'on record, not run'), :1069-1110 (keith_fx dead, 35 lv), :1242-1262 (budget 56), :565-575 (VOID precedent)",
  "submission/_throughput_v1/graft_retry.py:24-28 (base_actions_per_level None in submission mode), :37-56 (RETRY_CLEAR_HISTORY=1 [0]), :441 (agent._history_messages = [])",
  "submission/_throughput_v1/graft_effects.py (docstring: object-level effect table from segmentation hashes, zero model calls) and graft_sweep.py:80-115 (harness fires actions through step_env with no model call)",
  "scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/tool_agent.py:1113-1126 (note blanked on level transition AND game_over), :1147-1257 (user prompt builder), :1453-1590 (_run_python_tool / _handle_action -> step_env), :1770-1778 (yield)",
  "scratchpad/bundles/june_stock/src/ARC3-Inference/inference/agent/prompts.py:82 ('only tool is python'), :86 (BFS advice), :98-99 (batching advice), :111 (30 s per python call)",
  "scratchpad/bundles/june_stock/src/ARC3-Inference/inference/framework/solver.py:263-340 (play loop, auto-RESET on GAME_OVER at :276-281), :588-663 (step_env batches stop on level_completed/game_over), :663-665 (_execute_auto_reset)",
  "submission/_oracle_probe/oracle_probe.py:1-57 (design, >=50% pass line, wa30 L3 engine-verified 82-move solution), :77 (ollama endpoint), results.json / results_control.json (qwen3:8b 0/5, qwen3:14b control 0/3; never run on Flash-Next)",
  "docs/research-2026-09-08/R-reopen-audit-0908.md rows: 'Fresh-mind retry ... full-wipe NEVER TESTED (ceiling +0.33)', 'Oracle-model probe ... NEVER PROPERLY TESTED', 'Batching with expectation / abort-on-surprise ... NEVER BUILT', 'Persistent sandbox ... never measured'",
  "docs/research-2026-09-10/R-field-and-harness-research.md §4 (Tycho orchestrator 88.49 vs always-model 85.36), §9 (no ablation of 'LLM-written controller running N steps'; batching pays only with an abort detector), §1 (added tools hurt)",
  "docs/research-2026-09-10/R-what-our-agent-actually-does.md (corpus claims) with the verification-phase corrections (>=30 forward simulations; 62.5% of calls dropped; step-mapping bug)",
  "docs/research-2026-09-09/R-harness-teardown-and-scoring.md:36-48 (275-run level distribution; leader ~2 lv/game), :50-84 (queue chain, 62.5% of actions after last clear, 48% of games under baseline at the wall)",
  "memory arcagi3-two-level-wall.md:116-124 (patch 21 x1.40 actions -> -25% levels; ungated 26-action batches on sk48 never cleared L1)",
  "memory arcagi3-prime-agent-findings.md (PRO-LONG batching 1-20, 'scale toward 20 for proven sequences'; harness transfers no capability downward: GLM-5.2 8.6%)",
  "memory arcagi3-human-replay-dataset.md (avatar median 70 actions per completed level; humans reset freely; sub-second tempo)",
  "memory arcagi3-reset-lever-refuted-competition-guard.md (RESET swallowed live, api.py:316-334)",
  "memory arcagi3-track-a1-carry-compact.md (~7k of ~27k retained tokens are repeated user boilerplate; 'rider, not built')",
  "memory arcagi3-scoring-is-convex-in-depth.md (exact rule; efficiency above parity worth zero; inefficiency quadratic)",
  "submission/_keith_copy/push_yield900/arc3-keith-yield900.ipynb cells 12-13 (customization hook where a graft installs live) and offkaggle/run_regime_wave.py:302-307,637-651 (ARM_GRAFTS / install_grafts in-memory rebinding)",
  "Verification-phase cross-cutting audit: per-wave sd ~3.5 (not 2.34), one wave resolves only +/-7.4 levels, two-wave mean rule; 3-draw live rule cannot separate 3.6 from 4.0"
 ],
 "what_the_campaign_got_wrong": [
  "The 'reactive loop, 0 forward simulations, never models live' diagnosis is overstated and misdirected the angle: on the stock base draw 51% of acting snippets run action() under a loop with terminal checks in 82%, mean 5.1 actions per acting turn (my count, 20260902-232148-regime-keith), and the verifier found >=30 hand-written transition functions in the very corpus that reported zero (plus 3x that rate in the 62.5% of calls the corpus dropped, and a builder step-mapping bug). The model already writes small controllers; the loop-shape gap is that it never abandons a wrong hypothesis, not that it never simulates.",
  "Prompt-content arms (evid, hypo, fx, carry, up8) were read as tests of a 'model-based loop'; none changed who executes or when the hypothesis is re-sampled. The two structural loop arms that were run were dosed weakly: probe discipline (behaviour-shaping) and fresh-mind at the note-only dose with a RESET that collided with a cleared level — the full-history-wipe dose the graft was built with (RETRY_CLEAR_HISTORY) is still 'on record, not run' (REGIME_WAVE_STATUS.md:426-434).",
  "'BUDGET CONVERTS' hid the most important per-game fact: 9 of the 12 never-passed walls survived 3x calls and 4x actions (only dc22/sb26/vc33 fell). Walls are comprehension-bound; any lever that only multiplies attempts inherits budget's ~0.2 elasticity on progressing games only, so the 'not at a capability ceiling' framing is wrong for exactly the games that separate us from the leaders.",
  "The one decisive diagnostic for the whole 'harness models, model executes' class — the oracle probe on wa30 L3 — was pushed 08-29, never recorded, and never run on Flash-Next (reopen audit: NEVER PROPERLY TESTED), while ~$60 of prompt-content waves were spent testing weaker doses of the same question.",
  "Macro-batching was killed under the 27B regime with blind batches and no halt detector (patch 21; sk48 26-action batches) and never re-run under V14 with an abort-on-surprise, even though the campaign's own field sweep names 'batching pays only when coupled to a cheap mid-flight abort detector' as the reconcilable untested hypothesis and 'LLM-written controller running N steps' as an unpublished ablation.",
  "The verdict instrument discards exactly the effect size loop-shape levers have: single-wave kills at a claimed sd of 2.34 when the per-game data support ~3.5, so a true +4-level lever reads 'dead' 62% of the time and 'engaged-but-flat class closed' is closed only for step-sized effects; probe 41 and carry 41 are indistinguishable from +10-15% levers. Combined with the 3.59-vs-3.6 live read decided by 0.01, the campaign has been killing sub-step levers it cannot see and then declaring classes closed.",
  "Instrument faults were waived silently on the positive arm (6 h container reaper mid-way through budget3x; every VLLM number in that write-up is a two-container mixture) and voided on negative-tracking arms; the 'cadence family closed' memory file still contradicts the budget3x result recorded the next day."
 ],
 "biggest_risk": "The walls are deterministic comprehension failures of this brain — the same game fails the same way in every draw (g50t x3, sk48 x4, sp80 x3; 16 of 25 games are LOWVAR/CAPPED) — so hypothesis re-sampling yields nothing on most walls, re-execution's extra actions burn the quadratic efficiency term the way patch 21 did, and with a per-wave sd of ~3.5 a true +4-6-level gain is invisible in one wave. The failure mode is spending 4-6 rig waves ($40-60) and 2-3 weeks to end at a sixth 'engaged-and-flat' verdict with nothing new to fly by Oct 1, when the only lever that can reach the 7+ tier (absorbing a released model-based loop) needs the quota and the attention. Mitigation built into the plan: the $3 oracle probe and the $0 offline replay come first and can kill both bets before a wave is spent, and every arm is a composable in-memory graft so it stacks on an absorbed kernel rather than competing with it.",
 "lb_target_by_nov2": "3.6-4.2 from this angle alone (base family mean 3.22, n=6; +0.4 to +1.0 LB if step 1 converts at a third to all of its +0.33 lv/game ceiling and step 2 adds a rider), with a ~15% chance of a rig step candidate (>=45 two-wave mean) that would put a live 4.0+ within reach. Anything >=5 by Nov 2 has to come from Oct-1 absorption of a released model-based loop; loop-shape changes to the stock duck on Flash-Next at our compute cannot get there, and I would not spend more than ~$50 and 3 weeks of the remaining 52 days on this angle."
}
```

---

## agent-16

```json
{
 "angle": "FIELD INTELLIGENCE / ABSORPTION — what the 7.5-11 tier actually runs, what of it is public, and what can be byte-absorbed under Track D (offkaggle/absorb_kernel.py + docs/TRACK-D-absorption-checklist.md)",
 "thesis": "We sit at ~3.2 because we run the best PUBLIC artifact and nothing better is public: keith V14's code_cell_sha256 de24036a... re-pulled today is unchanged since our 09-02 attestation (submission/_keith_copy/ATTEST.json), the 190-vote 'Duck Qwen3.8 Anim Base' (wuliao0, 4.33) hashes identically, and every fork on the same serving regime is a prompt/guard no-op (lucifer19 C18 = one appended 'efficiency directive', anchor 2.84; ayodeji v124 = a preflight cell; amanatar = guard cells). Our 3.25 IS the V14 cohort mean (3.33, sd 0.55, docs/research-2026-09-06/R-field-refresh.md F3); the 4.0-4.6 public tier is that cohort's max statistic. The leaders are private code on the same 96 GB card: a scan of all 26 member accounts of the top-20 teams (leaderboard CSV pulled today) finds zero ARC-3 kernels/datasets/models since 09-01 for Tufa (6 accounts), Third Intelligence, Franzen, mostik.ai, Fususu, Kyutai (only a vLLM nightly wheelhouse), Reki (only a game-generator bundle). The one structural read the field gives is the NVARC3 roster: cpmpml (Puget, an AVO author), galkaplun and eladsarafian (NVIDIA OO-Agents/NOOA authors), sorokin (NVARC fine-tuning) — i.e. the strongest few-entry evidence on the board (7.69 in a handful of submissions; CPMP confirms 'you're solo, it is much harder' and denies GPU-scale) is an AVO/NOOA-lineage harness, possibly with a tuned model; AVO's code is not open (blog links only to arXiv), NOOA is Apache-2 but died on our clock contract (docs/research-2026-09-09/GATE-polyphony-clock.md:4,81), and the community CC0 reconstruction of AVO (jakobbrggen/taaf-kaggle-source 09-01, inference/avo/*, 733 lines: memory digest + stagnation supervisor + phased loop) scored 1.15 live on 27B-FP8 (yocybercode/thui-avo-v0, V1, svid 347495465) where plain duck+27B reads ~2.0-2.2 — the paradigm does not transfer to an open 27B at Kaggle compute, consistent with our own A2/Polyphony kills. So there is nothing to absorb before Sept 30, and the campaign's remaining score is almost entirely the Milestone-2 releases: open-sourcing by Sept 30 is the eligibility condition for the $25K/$10K/$2.5K milestone prizes (arcprize.org/competitions/2026/arc-agi-3; all three M1 winners released), and CPMP states the private 55-game score of every submission is computed at scoring time with no end-of-competition rerun (thread 697944, comment posted a day ago), so every daily slot after Oct 1 is a banked private draw of whatever we copy. The expected value of a byte-copy is the release's config MEAN, not its public max: the V14 cohort mean/max ratio is 0.77, so a Tufa release maps to ~7.7-9.4, NVARC3 (few entries) to ~6.5-7.5, Third Intelligence to ~6-7.",
 "plan": [
  {
   "step": "1. Freeze pre-Oct-1 spend and stand up the release watch (Sept 12, $0, half a day)",
   "what": "Stop flying V14-family redraws and stop $9 rig waves on behaviour-shaping grafts; both are zero-information for the final (docs/research-2026-09-06/R-field-refresh.md 'CAUTION' line; stats audit: one wave cannot see <+7 levels). Script a daily watch: `kaggle kernels list --user <u> --sort-by dateRun` and `kaggle datasets list --user <u> --sort-by updated` for the 26 top-20 member accounts (from the leaderboard CSV: Tufa dlorah/driessmit1/infinitecreativity/jeroencottaar/pressman1/stefano1283; TI seele1917/yukiokumura1; NVARC3 cpmpml/darraghdog/eladsarafian/galkaplun/sorokin; dfranzen; phuongncn; Kyutai hippolytepilchen/romainfabre; sonphamorg; ruichardliu/russcore), plus GitHub Tufalabs/duck-harness and NVIDIA-NeMo/labs-OO-Agents commit dates, plus the forum listing (Playwright works; WebFetch returns a JS shell). Diff against the baseline scan saved today (scratchpad users_scan.txt).",
   "why_it_should_move_the_lb": "It does not move the LB by itself; it converts Oct 1 from a 24-hour scramble into an hour, and it stops burning the 60 h/week GPU quota that the absorption needs (checklist: keep >=30 h unspent in the week of Oct 1).",
   "cost": "$0; ~4 h to script; 1 slot/day saved.",
   "first_falsifiable_test": "Run the watch against the known 09-01 jakobbrggen/taaf-kaggle-source release and the 08-29 sonphamorg Flash-Next serving datasets: it must flag both as 'new since baseline' on first run.",
   "kill_criterion": "If the watch misses a known release, fix it; there is no other kill — this step is prep.",
   "expected_lb_gain": "0 directly."
  },
  {
   "step": "2. Make the rig bundle-agnostic and build the serving-transplant kit (Sept 12-26, ~$27 rig, ~3 engineering days)",
   "what": "offkaggle/run_regime_wave.py is pinned to the june_stock agent tree (offkaggle/run_regime_wave.py:9,106,578) and to scratchpad/taaf_scored_ref pickles (:110,2283-2291); it cannot run a foreign TAAF bundle, so the checklist's 'measure the release on Modal the same day' (docs/TRACK-D-absorption-checklist.md H+1..H+6) is currently not executable. Build a runner that takes any released TAAF bundle (deploy_target.pkl + src/ + setup_commands.json, the format every Tufa/community release uses) and runs it against the Modal endpoint in Kaggle geometry (conc 28, 7,920 s). Validate on two bundles we already hold: (a) the V14 june_stock bundle (must land in the flat 8-draw band 34-41 levels), (b) jakobbrggen's avo-v2 bundle (HarnessSolver(avo_agent=True), a different agent class; live 1.15 on 27B-FP8 predicts a read well below base on Flash-Next). Second half: a 'serving transplant' — the keith V14 serving profile (RadixArk Flash-Next NVFP4, kv5-bf16-mtp3-c8-cg32, offkaggle/modal_flashnext_serve.py) exposed as a drop-in replacement for whatever vLLM stack a release ships, because the model swap 1.94 -> 3.25 is the ONLY live-measured step this campaign owns (memory arcagi3-step-change-search-2026-09-02).",
   "why_it_should_move_the_lb": "If the release ships on Qwen3.8-27B-FP8 or Gemma-4-31B (plausible for Tufa: driessmit1 re-hosted Qwen3.8-27B-FP8 on 08-18; Tufa's own writeup credits 'better base models' as the main driver), the transplant is the one composable lever with a measured +67% on this loop family; without a bundle-agnostic rig we cannot measure it before spending live slots.",
   "cost": "~3 waves ($27) + 3 days; no slots.",
   "first_falsifiable_test": "The foreign-bundle runner reproduces the V14 bundle inside 34-41 levels AND runs the avo-v2 bundle end-to-end (any level count) without hand edits to the bundle.",
   "kill_criterion": "If a foreign bundle cannot be run unmodified by Sept 26, abandon the generic runner and pre-write the hand-port recipe used on 09-02 (that path took one day and produced the live 3.25), keeping the serving-transplant kit.",
   "expected_lb_gain": "0 directly; enables step 4."
  },
  {
   "step": "3. Oct 1-3: byte-absorb the strongest Kaggle-runnable release, unmodified, and fly it (3 slots, ~27 h GPU quota)",
   "what": "Priority: Tufa (11.04, 132 entries, released at M1, six members with no newer public artifact) > Third Intelligence (8.21) > NVARC3 (7.69) > Franzen (7.63; released his 2024 and 2025 solutions, dfranzen/arc-prize-2025-by-the-architects) > Fususu. Per the checklist: `python3 offkaggle/absorb_kernel.py stage <owner>/<kernel> --as arc3-absorb-<name>`, record code_cell_sha256, `diff` against submission/_keith_copy/kernel-metadata.json (exit 0 = pure harness delta), read the loop shape and the completion-tokens-per-game cost gate (ours 73-83k/game; a loop that needs >2x will not reproduce on one RTX PRO 6000), push UNMODIFIED with Ahmed's go, three live draws. If the release is GitHub-only (no Kaggle datasets/models), re-host the weights the way raist321 re-hosted keith's (raist321/qwen38-flashnext-runtime-v1, 09-02) — budget one extra day.",
   "why_it_should_move_the_lb": "It is the mechanism that produced our only live step (keith V14 copy -> 3.25 in one day, sub 56042273). Every >=5 team is private code on the same card, so a release is the only path to their tier; the expected copy score is the config mean (V14 cohort mean/max = 3.33/4.33 = 0.77): Tufa ~7.7-9.4, NVARC3 ~6.5-7.5, TI ~6-7.",
   "cost": "3 daily slots, 2-3 Kaggle GPU sessions (~27 h of the 60 h quota), $0-9 rig.",
   "first_falsifiable_test": "Draw 1 of the byte copy reads >= 5.0 public — above any V14 draw ever observed (max 4.33; z > 3 on the cohort sd 0.55) — so one draw suffices to prove it is not V14-class.",
   "kill_criterion": "Draw 1 < 4.0, or the kernel cannot be made to run in the 12 h box (missing weights, CPU-only version, docker digest we cannot reproduce) within 48 h: move to the next release on the list. If no release reads >= 5.0 by Oct 8, revert to the V14 family for final selection and say so.",
   "expected_lb_gain": "+3.5 to +6 LB (3.2 -> 6.5-9.4) conditional on a Kaggle-runnable top-4 release; 0 otherwise."
  },
  {
   "step": "4. Oct 4-20: the one add-on we own — transplant the Flash-Next NVFP4 serving regime under the copied harness, only if the release runs a weaker model",
   "what": "If the absorbed kernel serves Qwen3.8-27B-FP8, Qwen3.6-27B or Gemma-4-31B, apply the step-2 transplant (keith V14 vLLM profile: NVFP4 weights, MTP-3, 8 seqs, 5 GiB KV, 32k analyzer window) as an in-memory graft that leaves the harness bytes attested (docs/TRACK-D-absorption-checklist.md 'composing our own work on top'). Measure on the rig against the copy's OWN rig read with the corrected statistics (two counterbalanced waves, step bar = two-wave mean >= copy + 6 levels; single waves resolve only +-7 levels), then 3 live draws. If the release already runs Flash-Next (as V14 does) there is nothing measured to add: skip to step 5 and do not graft behaviour-shaping levers (four replications flat on this instrument).",
   "why_it_should_move_the_lb": "The 27B-FP8 -> Flash-Next swap under an identical analyzer regime moved the duck 1.94 -> 3.25 live (+67%); it is the only lever in the campaign with a live-measured step, and it is orthogonal to whatever the released harness does.",
   "cost": "2-3 rig waves ($18-27) + 3 slots + ~2 days.",
   "first_falsifiable_test": "Rig two-wave mean of copy+transplant >= copy's two-wave mean + 6 levels.",
   "kill_criterion": "Two-wave mean below copy + 3 levels, or the released harness's prompts/tool schema break on the Flash-Next chat template (length-finish > 1% of calls), or completion tokens/game rise > 1.5x: drop the graft and fly the unmodified copy.",
   "expected_lb_gain": "+1 to +3 LB if the release is on a 27B/31B model; 0 if it is already on Flash-Next (probability ~50/50 on the Tufa evidence)."
  },
  {
   "step": "5. Oct 8 - Nov 2: bank private draws and select by config mean",
   "what": "CPMP (NVARC3) states the private 55-game score of every submission is computed at scoring time and there is no end-of-competition rerun (thread 697944). Therefore every remaining daily slot flies the best-MEAN config unchanged (n >= 5 draws), the ledger records public score + scriptVersionId per draw, and the two final selections are duplicates of the best-mean config chosen by MEAN over its draws — never the public max, which is a max statistic of an independent half.",
   "why_it_should_move_the_lb": "It does not raise the public max; it maximises the expected PRIVATE score, which is what is ranked. With per-draw sd ~0.8 at 3.2 (CV ~0.25, i.e. sd ~2 at 8), five draws pin a config's mean to +-0.9 LB.",
   "cost": "1 slot/day; $0.",
   "first_falsifiable_test": "After 5 draws the copied config's mean is >= 5.0 with all draws >= 4.0.",
   "kill_criterion": "If the copied config's 5-draw mean is below 4.5, or if a second release reads higher on a first draw, switch the daily slot to the better config and repeat.",
   "expected_lb_gain": "0 on the public max; protects the private mean (worth ~+1 LB versus selecting on public max at sd ~2)."
  }
 ],
 "evidence_cited": [
  "Leaderboard today: `kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 --show` (Tufa 11.04, Third Intelligence 8.21, NVARC3 7.69, Franzen 7.63, mostik.ai 7.51, Fususu 6.91, Kyutai 5.84, Chew Kok Wah 5.78, AFF AI CLUB 5.68, Son Pham 5.66, Reki 5.57); full CSV with TeamMemberUserNames via `kaggle competitions leaderboard -d` -> scratchpad/lb/arc-prize-2026-arc-agi-3-publicleaderboard-2026-09-11T20:45:47.csv (2,976 teams; NVARC3 = cpmpml,darraghdog,eladsarafian,galkaplun,sorokin; Tufa = dlorah,driessmit1,infinitecreativity,jeroencottaar,pressman1,stefano1283)",
  "Member-account scan: `kaggle kernels list --user <u>` / `kaggle datasets list --user <u>` for all 26 top-20 member accounts -> scratchpad/users_scan.txt: zero ARC-3 kernels/datasets since 09-01 for every >=5 team; only sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1 (08-29), romainfabre/muse-vllm-nightly-cu129-wheelhouse (08-17), russcore/arc-generator-bundle (08-27), phuongncn handoff (08-05)",
  "V14 unchanged and cloned: `kaggle kernels pull keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp -m` and wuliao0/duck-qwen3-8-anim-base both hash code_cell_sha256 de24036a43920cf0... = submission/_keith_copy/ATTEST.json (pulled 2026-09-02T15:49Z); metadata printout in this session (same datasets keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 + qwen38-flash-next-vllm-nvfp4-runtime-v1, model keithtyser/qwen3-8-flash-next-nvfp4/.../radixark-modelopt-fp4/1, docker 57e612b4..., NvidiaRtxPro6000)",
  "Same-regime forks are no-ops: unified diff vs V14 code cells — lucifer19/duck-qwen3-8-search-memory-c18 (+47 lines: one system-prompt directive, 'anchor_public_score 2.84'), ayodejiibrahimlateef/arc3-v124-observed-probe-memory-rtx6000 (+422 lines: read-only preflight asserting the exact V14 vLLM profile), amanatar/arc-agi-3-hybrid-repl-agent (+295: guard cells)",
  "Community AVO reconstruction: jakobbrggen/taaf-kaggle-source (updated 2026-09-01, CC0, git_status 'experiment/avo-v2 74ff3df'), files src/ARC3-Inference/inference/avo/{agent,memory,prompts,settings,supervisor}.py (733 lines), preamble avo_agent=True, setup pins Qwen/Qwen3.8-27B-FP8 + MULTIMODAL_UPSCALE=8; live score: yocybercode/thui-avo-v0 Public Score 1.15 (V1, scriptVersionId 347495465; commit-run public-25 = 21 levels/25 games, per-game [finished] lines in .playwright-mcp/page-2026-09-11T20-50-30-069Z.yml); docs/research-2026-09-06/R-field-refresh.md F5 ('users <=3.90')",
  "NVIDIA AVO: developer.nvidia.com blog (authors Terry Chen, Yeyin Zhu, Zhifan Ye, Jean-Francois Puget, Humphrey Shi; Aug 21 2026; Claude Opus 5; 100% on the public 25 only; no code link). NOOA Apache-2 (NVIDIA-NeMo/labs-OO-Agents, authors incl. Kaplun/Sarafian per web search); NOOA/Polyphony kills: docs/research-2026-09-09/GATE-polyphony-clock.md:4,26,81 and the 'Compute FAIL' table (65,468 completion tokens per verified model vs 83,094 per stock game)",
  "Forum (Playwright snapshots): thread 740812 'Only 6 submissions and jumped straight into the top 3' — CPMP (3rd) 'You're solo. It is much harder to be solo than in a team' and 'we are not the teams using most GPUs on Kaggle'; Fususu 'NvARC3 = Nvidia Arc3'. Thread 697944 'Update on Code Requirements' — CPMP, posted a day ago: 'There are no rerun at end of competition. Your final score is already computed but will stay hidden till end of competition. There are 100 games in test data. 55 are used to compute the score shown on public LB... 55 are used to compute a hidden score'. Thread 739707 — Ya Xu: Astra's 99% is on the 110-game set claim (unverified), 'The local 25-game set is much easier and can be solved using brute force'",
  "Milestone-2 rules: arcprize.org/competitions/2026/arc-agi-3 (Sept 30, $25K/$10K/$2.5K; 'Participants who open source their solutions by the milestone deadlines are eligible for milestone prize money'); memory arcagi3-competition-facts.md; docs/research-2026-09-08/R-field-refresh-0908.md 'Milestone 2 (pinned thread 713634)'",
  "Tufa: tufalabs.ai/research/ lists no ARC-3 post after 'Duck Harness' (July 1, 2026); Tufalabs/duck-harness GitHub still 2 commits; jeroencottaar kernels last run 07-01; driessmit1/qwen3-8-27b-fp8-hf-017b9c7a dataset updated 2026-08-18 (`kaggle datasets list --user driessmit1`)",
  "Son Pham: github.com/sonpham-org/arc-3 commits Sep 3-11 are viewer/docs/custom games (no Flash-Next/Kaggle recipe); memory arcagi3-sonpham-fork-findings.md (their Kaggle draw 1.12 in-band; no-impact policy fragile)",
  "V14 cohort statistics and 'redraws have zero information': docs/research-2026-09-06/R-field-refresh.md F3-F4 (n=9 V14 draws mean 3.33 sd 0.55; expected max of 50 draws 4.58) and CAUTION line; docs/research-2026-08-29/R1-kaggle-ecosystem.md §1.1 (Qwen3.6->3.8 swap moved identical code 1.2-1.6 -> 1.9-2.2); memory arcagi3-step-change-search-2026-09-02.md (byte-copy FLEW 09-05 -> 3.25, was 1.94)",
  "Rig runner is pinned to the june_stock bundle: offkaggle/run_regime_wave.py:9,106 (STOCK_AGENT_DIR), :110 (BUNDLE_PKL_DIR = scratchpad/taaf_scored_ref), :578,583-584,2283-2291 (sha pins + load_bundle); absorb_kernel.py REGIME_KEYS :50, stage :140, verify :190; docs/TRACK-D-absorption-checklist.md H+0..H+24 and the 'already known' table",
  "Campaign kills that this plan does not re-propose: memory arcagi3-track-d-absorption-ready.md (Track A closed), arcagi3-a2-workspace-closed.md, arcagi3-polyphony-priced-out.md, arcagi3-model-axis-blocked-on-scale.md (corrected by the verifier to 'blocked on unproven tooling', still not a 7-week bet), docs/research-2026-09-08/R-reopen-audit-0908.md:90,101 (AVO supervisor never engaged; animation digest stays dead), docs/research-2026-09-10/R-field-and-harness-research.md §1 (executable models, prose knowledge, added tools, small-model fine-tunes independently confirmed dead)"
 ],
 "what_the_campaign_got_wrong": [
  "It treated absorption as Track D, the fallback, and spent Sept 8-11 on Track A (A1 carry, A2 workspace, A3 NOOA, A4 Polyphony, ctx12/16, nr16 — all dead) while the field data already said no >=5 team had a single public artifact and none could be absorbed before Sept 30. The only Sept work with certain value — a bundle-agnostic rig runner so a release can be measured the same day — was never built; run_regime_wave.py is still pinned to june_stock (offkaggle/run_regime_wave.py:9,106,578).",
  "It read the paper harnesses (Astra, PRO-LONG, Polyphony, NOOA, AVO) as the mechanism behind the 7-11 tier and built A2/A3/A4 on that; the one public transplant of that paradigm onto an open 27B at Kaggle compute (jakobbrggen's CC0 AVO-v2 arm) scores 1.15 live where plain duck on the same model scores ~2.0-2.2, which the 09-06 field refresh recorded as '<=3.90' without checking the actual kernel (it is 1.15, yocybercode/thui-avo-v0). The field evidence points instead at NVIDIA's authors plus a fine-tuner (NVARC3 roster) and at Franzen's fine-tuning track record — private model work, not a public loop.",
  "It kept quoting '11.04' and '7.69' as configuration levels; they are max statistics (Tufa 132 entries). The campaign's own V14 cohort shows mean/max = 0.77, so the absorbable value of a Tufa release is ~7.7-9.4, not 11, and planning docs (PLAN-2026-09-08 §1, R-field-refresh-0908) should carry the mean estimate.",
  "It flew ~73 submissions of which the last ten are V14-family redraws with, by its own 09-06 analysis, zero information content; under the no-rerun rule (CPMP, thread 697944) those slots were still private draws, but of a 3.2 config that any Oct-1 release will dominate — they should have been held for the post-release banking period where each draw is worth ~+0.9 LB of selection precision.",
  "The absorption checklist reads a release against 'pooled base 39.33 sd 2.34, >=48 step'; the cross-cutting statistics audit puts the wave sd at ~3.5, so a single rig wave cannot distinguish a +10-15% release variant from the copy; the H+1..H+6 read rule must become a two-wave counterbalanced mean with a +6-level bar, or be skipped in favour of the one-draw live test (>=5.0 beats every V14 draw ever seen).",
  "Minor factual drift: docs cite 110 hidden games; CPMP (thread 697944) says 100 (55 public / 55 private). Neither changes the plan, but the ledger's expected-value arithmetic per level should use 55.",
  "The sonpham memory (08-10) and the 08-29 review over-weighted fork-level harness tricks (no-impact policy, grafts) that their own Kaggle draws never lifted (1.12 in-band); today's diff of every same-regime V14 fork shows the public field has converged on exactly zero harness deltas that score, which the campaign could have concluded on 09-06 and stopped grafting."
 ],
 "biggest_risk": "No top-4 team publishes a Kaggle-runnable kernel by Sept 30. Open-sourcing is required only to collect the milestone money ($25K/$10K/$2.5K), which a team defending a $40K final might forgo; a release may be GitHub code without the Kaggle model/dataset mounts (re-hosting a 135 GB checkpoint plus a custom vLLM fork took the community ~2 days for Flash-Next); and NVARC3-style entries may depend on a fine-tuned model whose weights are not released. If nothing absorbable appears, this angle has no owned lever left: the ceiling stays the V14 family (~3.2-3.5) and the honest recommendation is to bank duplicate V14 draws and select by mean. Secondary risk: a release that IS absorbable but whose per-draw mean is far below its public max (heavy resubmitters), so the copy reads 5-6 rather than 8-9; the one-draw >=5.0 test catches the worst case but not this one, hence the five-draw mean rule in step 5.",
 "lb_target_by_nov2": "Conditional on a Kaggle-runnable Tufa or NVARC3/Third-Intelligence release by Oct 1 (estimated ~60-70%: all three Milestone-1 winners released, and the M2 prizes require it): public mean 6.5-9.4 (point estimate ~7.5), plus 0 to +2 if the release runs a 27B/31B model and the Flash-Next serving transplant reads a step; private expected the same as the public mean. If no absorbable release appears: 3.2-3.5 (V14 family mean), with the two final selections chosen by mean over >=5 draws."
}
```
