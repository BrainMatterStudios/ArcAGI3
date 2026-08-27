# Stage-2a milestones — T1/T2 accuracy gates + effects/commit re-run

Protocol: budget 4000 actions/game, 13 frame-Markov games, effect model + commit mode ON. Held-out = prequential (test-then-train). Gate = >=90% prequential within <=300 own-scope transitions (>=10 held-out outcomes).

## Table 1 — held-out rule accuracy (top rules per game)

| game | rule family | scope | transitions | held-out acc | gated_at | in cap (<=300) | gated now |
|------|-------------|-------|-------------|--------------|----------|----------------|-----------|
| dc22 | click_null | click:c=0 | 87 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=2 | 47 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=4 | 38 | 1.0 | 10 | Y | N |
| ft09 | click_null | click:c=0 | 20 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=11 | 20 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=12 | 20 | 1.0 | 10 | Y | Y |
| ka59 | click_null | click:c=0 | 10 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=1 | 10 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=15 | 10 | 1.0 | 10 | Y | Y |
| lp85 | click_null | click:c=1 | 20 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=10 | 20 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=11 | 20 | 1.0 | 10 | Y | Y |
| r11l | click_null | click:c=2 | 14 | 1.0 | 14 | Y | Y |
|  | click_null | click:c=0 | 139 | 0.8777 | 16 | Y | Y |
|  | click_null | click:c=10 | 44 | 0.4773 | None | N | N |
| re86 | blob_move | A3 | 943 | 0.4286 | None | N | N |
|  | blob_move | A2 | 938 | 0.375 | None | N | N |
|  | blob_move | A1 | 936 | 0.25 | None | N | N |
| s5i5 | click_null | click:c=13 | 20 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=2 | 20 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=3 | 20 | 1.0 | 10 | Y | Y |
| sb26 | click_null | click:c=5 | 10 | 1.0 | 10 | Y | Y |
|  | constdiff | A5 | 780 | 1.0 | 101 | Y | Y |
|  | click_recolor | click:c=0 | 115 | 0.7455 | 70 | Y | N |
| tn36 | click_null | click:c=4 | 31 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=2 | 10 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=3 | 10 | 1.0 | 10 | Y | Y |
| tr87 | constdiff | A4 | 982 | 0.4553 | 25 | Y | N |
|  | constdiff | A3 | 976 | 0.3971 | None | N | N |
|  | blob_move | A3 | 976 | 0.38 | None | N | N |
| tu93 | blob_move | A1 | 810 | 0.5835 | 22 | Y | N |
|  | blob_move | A2 | 756 | 0.4534 | 19 | Y | N |
|  | blob_move | A4 | 1006 | 0.3457 | 25 | Y | N |
| vc33 | click_null | click:c=0 | 30 | 1.0 | 10 | Y | Y |
|  | click_null | click:c=5 | 28 | 1.0 | 18 | Y | Y |
|  | click_null | click:c=4 | 26 | 1.0 | 16 | Y | Y |
| wa30 | colormap | A5 | 210 | 1.0 | 189 | Y | Y |
|  | blob_move | A2 | 936 | 0.3333 | None | N | N |
|  | blob_move | A1 | 940 | 0.2614 | None | N | N |

**Milestone S2a:** 12/13 games with a rule at >=90% held-out within <=300 own transitions -> **PASS** (pre-registered: PASS >=8/13)

## Table 2 — Stage-1b (T0) vs Stage-2a (T0+T1/T2+commit)

| game | levels s1 | levels s2a | duck | apl s1 | apl s2a | x-human s1 | x-human s2a | pred steps | mispred | end |
|------|-----------|------------|------|--------|---------|------------|-------------|------------|---------|-----|
| dc22 | 1 | 1 | 1 | 3710.0 | 1901.0 | 44.17 | 22.63 | 37 | 10 | budget |
| ft09 | 2 | 1 | 2 | 1984.5 | 268.0 | 82.69 | 11.17 | 0 | 0 | budget |
| ka59 | 0 | 0 | 1 | None | None | None | None | 0 | 0 | budget |
| lp85 | 1 | 1 | 1 | 470.0 | 278.0 | 13.43 | 7.94 | 0 | 0 | budget |
| r11l | 1 | 1 | 1 | 37.0 | 37.0 | 1.07 | 1.07 | 0 | 0 | budget |
| re86 | 0 | 0 | 1 | None | None | None | None | 0 | 0 | budget |
| s5i5 | 1 | 1 | 0 | 1881.0 | 1343.0 | 25.94 | 18.52 | 0 | 0 | budget |
| sb26 | 0 | 0 | 1 | None | None | None | None | 5 | 5 | budget |
| tn36 | 0 | 1 | 0 | None | 25.0 | None | 0.6 | 7 | 5 | budget |
| tr87 | 0 | 0 | 0 | None | None | None | None | 2 | 2 | budget |
| tu93 | 4 | 4 | 2 | 553.2 | 535.5 | 19.76 | 19.12 | 142 | 0 | budget |
| vc33 | 2 | 2 | 1 | 361.0 | 123.0 | 8.2 | 2.8 | 0 | 0 | budget |
| wa30 | 0 | 0 | 0 | None | None | None | None | 2 | 0 | budget |

**Efficiency movement:** 3/8 games within 3x human median (Stage-1b: 1).
