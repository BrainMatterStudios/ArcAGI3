# Behavioral adapter eval — synth-corpus LoRA ckpt-10 vs base (2026-08-04)

Kernel ab-wmr v5, COMPLETE 4.39h, waves M,B,B,M (M = merged adapter served from
/tmp/merged_sft, B = FP8 base snapshot — per-wave serving identity logged; the
cross-arm logprob fingerprint gate passed, i.e. the arms provably served
different weights). Identical v7 duck config both arms.

## Result: adapter HURTS play

| arm | levels (2 waves) | median gen_tokens/game |
|---|---|---|
| B (base) | **17** | ~57k |
| M (adapter) | 9 | ~41k |

M is worse nearly everywhere it differs (ft09 0,0 vs 3,2; tu93 0,0 vs 0,2;
lf52/cn04 0 vs 1 each). The one bright anecdote: M unlocked m0r0 once (wave 0)
— a game with zero unlocks in all prior recorded runs — but B was not given
enough waves to rule out the current stack doing that anyway.

## The mechanism is visible in the token counts

M waves generate ~30% fewer tokens per game (and one g50t run emitted 1 action
/ 0 tokens). The synthetic corpus's mechanical teacher writes SHORT templated
targets (~265 text tokens vs corpus_v3's ~1,487); the adapter learned that
style and now under-deliberates in real play — exactly the "token diet" trap
measured in July (capped thinking looked fine on dev, cratered at 0.73 hidden:
thinking is essential on hard games). The NLL transfer signal was real but it
measured STYLE adaptation, not skill.

## Consequences (pre-registered: nothing ships)

1. Checkpoint-10 adapter: DO NOT SHIP. No further checkpoints worth sweeping
   from this corpus — the defect is the corpus's target style, not the step.
2. Stage-2 machinery: fully validated twice over (train pipeline AND
   merge+serve+behavioral-eval pipeline both work end-to-end now).
3. v2 corpus requirements sharpened, in priority order: (a) targets must
   PRESERVE long-form deliberation (teacher annotations at realistic thinking
   length, not templates); (b) revision arcs (wrong hypothesis → experiment →
   revise → win); (c) diagnosis-mapped families F2/F4/F5/F7. A corpus that
   fails (a) will make any (b)/(c) content actively harmful.
4. The behavioral gate earned its cost: NLL said +36% transfer; play said −47%
   levels. NLL gates alone are DISQUALIFIED as ship criteria for this track.
