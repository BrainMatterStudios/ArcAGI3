# Judge — program on the 3.25 base (2026-09-06, fresh-context adversarial)

A. Free-capacity claims: estimator over-count reproduced (median 1.36×, n=25, rebuilt with the real
tokenizer); base64 PNGs in kept history messages are counted at ~500 "tokens" each vs 64 real vision
tokens. Replayed prior reasoning = 36% of REAL tokens (not 24%). BUT the "unused 30%" is the
generation headroom: prompt+completion hit 32,768 in 7 calls; filling prompts would push ~30% of
calls into length-truncation (the flight arm at 1.9% truncation lost 0.56 lv/game). Only the image
miscount is recoverable (~+3k tokens). "Use the real 32k" is NOT a lever; 0.88→1.44 is confounded
(window + 4096 output cap changed together). Riders only.
B. Candidate #1 (hypothesis enumeration): the 105 failure labels track the classifier, not the run;
2/5 flipped on full reads; the never-passed-wall list is 8 games (cd82, dc22, lf52, ls20, re86, s5i5,
vc33, wa30), not 12. The tails show NO hypothesis starvation (lf52 listed 12 candidates; cd82 had the
correct goal first) — they lose on EVIDENCE HANDLING: post-clear frame diffed as level data,
mis-transcribed references, misread batched logs, click at (16,25) vs the model's own (16,32), false
memory of when a key appeared, 30 s tool timeouts. Prior for a prompt rule being a step: <15%.
C. Throughput: the Kaggle box has no memory (OOM at 40 MB free during autotune); no profile knob is
×1.5; the one zero-memory ×2 is kv_cache_dtype=fp8 (→ the kv10 regime at 0 extra bytes), support
in this nightly's sparse-attention path unverified — $1 boot smoke; value capped at +0.28 (rider).
D. Field: nothing deserves $9 before the kill test; UPSCALE 8 is the only one with a mechanism
(64 → 256 vision tokens; the model does read the image) but no effect evidence.
E. SLOT POLICY: sd/draw ≈ 0.55 at 3.3. Detecting +0.5 live needs 15 draws/arm (30 slots) — impossible
with 56 left; +1.0 needs 4/arm. Rule: a slot goes only to an arm that passed the rig at ≥48 levels
(≥ +0.5 local); base arm = the V14 cohort (mean 3.33, n=9); 3 draws: mean ≥4.0 = step, ≤3.6 = dead,
else 2 more and decide at 5 with ≥3.9. Base redraws buy public rank only (final = 2 selected subs on
the private set); expected max of 50 more V14 draws = 4.6 with zero information. Leave slots unused
except one base redraw per ~10 days for drift.

## Ranked program
1. 3-WALL TURN-CAPPED INSTRUMENT (~$12): walls cd82 L2, dc22 L2, lf52 L2 (identical failure in ≥4/5
   draws, short human solutions); 2 draws × arm; --per-game-s ≈1500 in the queue-free regime (~55
   calls ≈ live turns). Arms: (a) EVIDENCE-INTEGRITY AID — harness auto-appends after each action
   batch an object-level before/after diff (moved/appeared/vanished with coordinates), a per-action
   position trace for batched moves, and flags the post-clear frame as the next level (targets the
   root cause in 3/5 tails); (b) hypothesis-enumeration + probe rule; (c) MULTIMODAL_UPSCALE 8.
   Engagement gate ≥80% of wall turns carry the aid/list. RULE: ≥3 of 6 wall-attempts passed → one
   25-game wave ($9) with rule ≥48 levels AND ≥3 of the 8 consistent walls passed; ≤1 → dead; 2 →
   one more round (P(≥3/6 | null) ≈ 1.6%).
2. Image-count fix as a rider on whatever ships next.
3. fp8-KV boot smoke ($1), information only.
DO NOT BUILD: exact estimator; drop-reasoning as a step bet; any Kaggle throughput arm; 27B-in-regime;
Polyphony; a 25-game protocol wave before the 3-wall test; V14 redraws for rank; memory rebuilds;
stronger-dose retry.
