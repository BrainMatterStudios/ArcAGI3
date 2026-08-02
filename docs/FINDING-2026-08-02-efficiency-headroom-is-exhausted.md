# Efficiency headroom is exhausted. 100% of remaining score is depth.

**Date:** 2026-08-02 · **Cost:** zero — computed from rig dumps already on disk

## The measurement

For every level our agent actually completed across two rig sweeps, compare the raw
efficiency term against the completed-share cap:

```
arm       game  nlev  L1base  L1act  ratio   rawEff    cap   binds
base      ar25     8      32     20   0.62x   3.194   2.778   CAP
base      sb26     8      18     12   0.67x   3.194   2.778   CAP
base      su15     9      22     19   0.86x   2.556   2.222   CAP
base      tn36     7      32     16   0.50x   4.107   3.571   CAP
base      ar25     8      32     31   0.97x   2.960   2.778   CAP
boardfix  sb26     8      18      9   0.50x   3.194   2.778   CAP
boardfix  su15     9      22     20   0.91x   2.556   2.222   CAP
boardfix  ar25     8      32     29   0.91x   3.194   2.778   CAP
boardfix  bp35     9      21     59   2.81x   0.282   2.222   eff
boardfix  tu93     9      19     24   1.26x   1.393   2.222   eff
```

**8 of 10 completed levels hit the cap.** On the levels it clears, the agent runs at
**0.50-0.97x the median human baseline** — faster than the median human — and
`min(weighted_mean, completed_share_cap)` truncates the reward to the cap regardless.

## What this invalidates

Marginal efficiency on a completed level is worth **exactly zero** to us today. Every
arm in the queue that targets action count targets a quantity with no value at our
depth:

- boardfix (fewer wasted actions)
- the context-starvation arm, which shipped and scored 0.76
- E5 geodesic replay / cycle elimination
- the token diet
- most of the "efficiency" half of the programme

It also corrects a figure this campaign has been reasoning from. `HANDOFF-2026-08-01`
says "the field at ~1.26 clears level 1 at ~1.7x human". **That is not our agent.** Ours
is at 0.5-0.97x. The inference "efficiency is bought by capability" does not apply to us
because we are already past the point where efficiency pays.

## The arithmetic that should drive everything

Cap for completing k levels of an n-level game is `sum(1..k)/sum(1..n) * 100`. For n=8:

```
levels cleared   1      2      3      4      5      6      8
max score      2.78   8.33  16.67  27.78  41.67  58.33  100.0
```

The level 1 -> 2 step is a **3.0x** multiplier. No efficiency gain on level 1 can produce
anything comparable, because level 1 is capped at 2.78 no matter how fast we clear it.

## Consequence for the programme

Rank every future arm by whether it plausibly increases LEVELS COMPLETED. If it only
reduces actions on levels already cleared, it is worth zero and should not take a GPU
sweep or a submission slot.

Related in-source mechanism, not yet tested: `tool_agent.py:1113-1126` wipes
world_model / goal_model / action_model / recent_findings / open_questions /
current_plan on every level transition, preserving only `cross_level_notes` — which the
model writes 0.00% of the time across 6,673 responses. At the exact moment depth
becomes possible, carried state is emptied into a channel that is always empty. The fix
is not "stop wiping" but "promote the populated fields into the surviving channel
mechanically", since the model demonstrably will not volunteer them.
