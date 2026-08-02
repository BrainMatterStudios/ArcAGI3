# A/B kernel arc-agi-3-ab-wmr — analysis (2026-08-03)

Kernel COMPLETE in 4.16h (4 waves x ~3600s, 10 concurrent games/wave, one vLLM
session serving vrfai/Qwen3.6-27B-FP8). Serving assert passed all 4 checks
(model id, finite logprobs, on-disk FP8 snapshot in --model arg). Per-wave
toggle proof verified: arm A all three patches off, arm B all on. Waves A,B,B,A.

## Headline numbers (2 waves per arm, 10-game panel)

- Levels completed: **A = 16, B = 20** (+4, ~+25% relative)
- Local score sum: A = 34.5, B = 43.1 (ft09 alone swings 14.3 — do not read score)
- Per-game paired levels (A runs | B runs):
  sb26 1,1|1,1 · re86 0,1|1,2 · su15 1,1|1,1 · tu93 2,1|2,1 · vc33 2,0|1,1 ·
  bp35 1,0|1,1 · ar25 1,1|0,1 · ft09 0,2|2,2 · g50t 0,0|0,0 · ls20 1,0|1,0
- B never loses a game pairing outright; gains concentrate in re86 (+2) and ft09 (+2).

Significance: with per-game per-run level RMS ~0.7, the +4 delta is ~0.9 sigma —
DIRECTIONAL, not certified. Doctrine says do not ship claims off this alone.

## Event evidence (the reliable readout)

- **Watchdog fired 3x in arm B** (ft09 w1, g50t w1, tu93 w2 — one recovery RESET
  each, zero kills); 0 fires in arm A (disabled). ft09-B-w1 banked 2 levels after
  its recovery vs ft09-A-w0 = 0 levels (though ft09-A-w3 also got 2 unaided).
- **HUD mask engaged with zero false-mask trips**: suppressed_hud_only fired on
  bp35 (58), ls20 (10), sb26 (8); guard_dead_lines = 0 and dropped_cells = 0
  across all 20 B game-runs — the hardened detector held on live play.
- **Replay never fired** (status: skipped, all B runs): no game reached full WIN
  in any wave — exactly as the 196-run decomposition predicted. The replay path
  remains e2e-tested but behaviorally unexercised at eval-like conditions.

## Instrumentation gap

gen_tokens = 0 on every row — the token-accounting hook did not capture
(behavior + logprobs prove the model generated). Fix before the next rig run.

## Verdict

No-regression confirmed, weak positive unlock signal (+4 levels, concentrated
where the mechanisms actually fired), all three mechanisms proven live on the
real 27B in eval-like conditions. Certification would need ~4-6 more waves.
Patches are kill-switched and strictly-non-negative by design; inclusion in the
next experimental submission is a judgment call, not a measurement one.
