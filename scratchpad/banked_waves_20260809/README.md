# Banked screen-wave outputs, rescued 2026-08-09

The ONLY copy of these eight `patch_closure_result.json` files. They were living in a
different session's `/private/tmp` scratchpad and would have been destroyed by a tmp
sweep. Copied here verbatim.

| file | wave |
|---|---|
| pc_base.json / w2_base.json | closure-rig BASE waves 1, 2 |
| pc_cand.json / w2_cand.json | closure-rig CANDIDATE waves 1, 2 |
| pkg.json | package screen |
| struct.json / struct2.json / struct3.json | struct screen waves 1, 2, 3 |

Each has 28 rows (clones). `rows[]` carries `actions_per_level`, `levels_completed`,
`levels_total` AND a `score` field already computed by `pc_driver.pc_env_score` — the
TRUE weighted, completion-capped objective. `rows_by_source` aggregates that detail
away, and the classifiers only ever read `rows_by_source["levels"]`.

Re-scored 2026-08-09 (implementation validated against the stored `score` on 140/140
rows, zero mismatches > 5e-4):

| wave | raw levels | TRUE score (25 games) | ft09 |
|---|---|---|---|
| base_w1 | 11 | 1.4751 | 14.286 |
| base_w2 | 12 | 1.2039 | 14.286 |
| struct_w1 | 17 | 1.5198 | 0.000 |
| struct_w2 | 12 | 0.7866 | 0.814 |
| struct_w3 | 12 | 0.7667 | 0.000 |

base mean 1.3395 vs struct mean 1.0244 = **-23.5%** on the objective, versus the
reported "+2 levels/wave replicated positive". Excluding ft09 (the screen's own
convention) struct wins +32%, so the sign hinges entirely on ft09 — the largest
score term in the card, which `patch_closure_config.py:51` excludes from the
decision statistic on a stated rationale ("8 levels") that is factually wrong
(metadata and every row say 6).

Caveat: n=2 vs n=3, and struct's best wave (1.520) beats both base waves, so the
overall means are NOT statistically separated. The consistent part is ft09: base
2/2, struct 0/3.
