# Stage-2b — FULL-25 verdict: engineered agent vs duck BASE_ENV

Protocol: all 25 public games, budget 4000 actions/game, wall cap 900.0 s, effects ON (T0 + T1/T2 + move_blocked + win predicate + audit share). True objective throughout; duck = banked max-over-clones (pc_base + w2_base).

## Table 1 — per-game levels, efficiency, true score

| game | lv ours | lv duck | apl ours | apl human | x-human | score ours | score duck | hybrid | route | end |
|------|---------|---------|----------|-----------|---------|-----------|------------|--------|-------|-----|
| ar25 | 1/8 | 1 | 1728.0 | 69.0 | 25.04 | 0.001 | 0.5487 | 0.5487 | duck | budget |
| bp35 | 0/9 | 1 | None | 46.5 | None | 0.0 | 0.2634 | 0.2634 | duck | budget |
| cd82 | 1/6 | 1 | 625.0 | 23.5 | 26.6 | 0.0369 | 1.7784 | 1.7784 | duck | budget |
| cn04 | 0/6 | 0 | None | 90.5 | None | 0.0 | 0.0 | 0.0 | duck | budget |
| dc22 | 1/6 | 1 | 1882.0 | 84.0 | 22.4 | 0.0047 | 2.4652 | 2.4652 | engi | budget |
| ft09 | 2/6 | 2 | 1959.0 | 24.0 | 81.62 | 0.0233 | 14.2857 | 14.2857 | engi | budget |
| g50t | 0/7 | 0 | None | 86.0 | None | 0.0 | 0.0 | 0.0 | duck | frontier_exhausted |
| ka59 | 0/7 | 1 | None | 51.0 | None | 0.0 | 0.5714 | 0.5714 | duck | budget |
| lf52 | 1/10 | 1 | 167.0 | 109.0 | 1.53 | 0.0668 | 1.2893 | 1.2893 | duck | budget |
| lp85 | 1/8 | 1 | 305.0 | 35.0 | 8.71 | 0.0086 | 2.7778 | 2.7778 | engi | budget |
| ls20 | 1/7 | 0 | 589.0 | 91.5 | 6.44 | 0.005 **W** | 0.0 | 0.005 | duck | budget |
| m0r0 | 1/6 | 0 | 1788.0 | 106.0 | 16.87 | 0.0013 **W** | 0.0 | 0.0013 | duck | budget |
| r11l | 1/6 | 1 | 37.0 | 34.5 | 1.07 | 1.6835 | 4.7619 | 4.7619 | engi | budget |
| re86 | 0/8 | 1 | None | 110.0 | None | 0.0 | 2.7778 | 2.7778 | duck | budget |
| s5i5 | 1/8 | 0 | 1486.0 | 72.5 | 20.5 | 0.0005 **W** | 0.0 | 0.0005 | engi | budget |
| sb26 | 0/8 | 1 | None | 23.5 | None | 0.0 | 2.7778 | 2.7778 | duck | budget |
| sc25 | 0/6 | 1 | None | 40.0 | None | 0.0 | 0.4358 | 0.4358 | duck | budget |
| sk48 | 0/8 | 0 | None | 112.0 | None | 0.0 | 0.0 | 0.0 | duck | budget |
| sp80 | 1/6 | 1 | 113.0 | 46.0 | 2.46 | 0.5672 **W** | 0.4947 | 0.5672 | duck | budget |
| su15 | 0/9 | 1 | None | 30.5 | None | 0.0 | 1.8673 | 1.8673 | duck | budget |
| tn36 | 1/7 | 0 | 26.0 | 42.0 | 0.62 | 3.5714 **W** | 0.0 | 3.5714 | engi | budget |
| tr87 | 0/6 | 0 | None | 54.5 | None | 0.0 | 0.0 | 0.0 | engi | budget |
| tu93 | 4/9 | 2 | 513.2 | 28.0 | 18.33 | 0.0796 | 3.7358 | 3.7358 | engi | budget |
| vc33 | 2/7 | 1 | 158.0 | 44.0 | 3.59 | 0.1082 **W** | 0.0022 | 0.1082 | engi | budget |
| wa30 | 0/9 | 0 | None | 122.0 | None | 0.0 | 0.0 | 0.0 | engi | budget |

## Means (true objective, 25 games)

| engineered | duck BASE_ENV | hybrid (per-game max) | hybrid (dispatch rule) |
|------------|---------------|----------------------|------------------------|
| **0.2463** | 1.6333 | 1.7836 | 0.7314 |

## Stage-2 verdict (pre-registered criteria)

* (b) full-25 true mean >= duck mean: **FAIL** (engineered 0.2463 vs duck 1.6333; miss fraction 84.9%)
* (c) outright true-score wins >= 5: **PASS** (6 score wins; 6 level wins)
* KILL rule ((b) miss > 20% AND (c) < 3): not triggered

**VERDICT: MARGINAL**
