# boardfix mechanism run — CONFOUNDED, and the confound may be the verdict

**Date:** 2026-08-02 · **Cost:** 2 GPU-hours (two 30-min sweeps + queueing)

## What was run

Paired mechanism sweeps through the rig: `rig-base` and `rig-boardfix`, 28 games,
1800s per-game box, 1 repeat each, identical behavioural probe injected into both
arms. Both completed cleanly, `stage: collect`, `error: none`, **zero config drift**.

## The arm activates — the judge's fatal defect is fixed

```
calls 620   corrected 90 (14.5%)   unresolved 0
raw_changed_frac 0.944 -> masked_changed_frac 0.798
masked games 18   unmasked 7 (ar25 ft09 g50t m0r0 sc25 su15 tn36 — correctly no HUD)
```
Before the fix this arm made **zero** corrections under the rig's clone ids and was
byte-identical to baseline. Verified now on real GPU, not just locally.

## The headline numbers, and why they are not a result

```
metric                          base   boardfix     delta    human
actions                       1119.0      620.0    -499.0
dead_reissue_of_actions        0.278      0.065    -0.213    0.034
noop_masked                    0.380      0.202    -0.178    0.084
immediate_repeat               0.578      0.400    -0.178    0.68
distinct_states_per_action     0.483      0.681    +0.197
```

Every delta favours boardfix and moves toward the human benchmark. **All of it is
explained by the boardfix arm taking 45% fewer actions.**

Two checks establish this rather than assert it:
- **Correlation between action count and dead_reissue within the base arm: r = +0.599.**
  The metric is mechanically action-count driven — more actions means more revisits
  means more opportunity to re-issue a known-dead pair.
- **On the 4 games where both arms took comparable action counts (within 20%), the
  delta REVERSES**: base 0.056, boardfix 0.092. n=4, so this is not evidence the arm
  is worse — only that the headline evidence it is better evaporates under matching.

Per-game action ratios show where the gap lives: sk48 166->23, ka59 104->17,
m0r0 89->17. The boardfix arm ran far shorter on exactly the games that drive the
metric.

The rig's own guard flags anything above 5% action divergence as uninterpretable.
This is 45%.

## The reframe: the gap may BE the verdict

The real competition is also wall-clock bound (7920s/game). If boardfix genuinely
makes the model generate more per turn, it gets fewer actions at eval too, and that
cost is real rather than an artifact. Depth dominates score, so a 45% action loss
would sink the arm regardless of how honest its signal is.

So the question is not "how do I remove the confound" but **"is the 45% action gap
real or noise?"** G0 gives a partial prior: actions/game across 5 identical repeats
had CV 11%, making 45% roughly 4 sigma — suggestive of real, but this is n=1 vs n=1.

## Next

One replication sweep per arm (~2 GPU-hours) to test whether the gap reproduces.
That is decision-relevant either way, and cheaper than the ~4-hour action-budgeted
comparison, which only matters if the arm is not already disqualified on cost.

`max_actions_per_game` exists (solver.py:745, checked in should_stop at :258) if an
action-budgeted run is wanted later.

## Standing conclusion

Do NOT ship boardfix to a submission slot on this evidence. The only mechanism
evidence is confounded and the one matched subset points the wrong way. This is the
second attractive result in two days to dissolve under checking — the first being a
transfer effect that survived neither a corrected substrate nor a sign test.
