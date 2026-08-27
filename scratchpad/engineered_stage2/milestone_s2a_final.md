# Stage-2a FINAL — T1/T2 accuracy gates + effects/commit re-run

Protocol: 13 frame-Markov games, budget 4000 actions/game, wall cap 900 s (identical to Stage-1b). Three arms:
  * **s2a** — T0 + T1/T2 effect rules + predicted-edge planning + commit mode + per-level gate probation.
  * **t0c** — same code, effects OFF (fair control: Stage-1b numbers predate the determinism/commit fixes).
  * **s1b** — the frozen Stage-1b table.
Held-out = prequential (test-then-train). Gate (strict) = >=90% prequential within <=300 own transitions AND accuracy held (still gated or lifetime >=90%).

## Table 1 — held-out rule accuracy (top rules per game, s2a arm)

| game | rule family | scope | transitions | held-out acc | gated_at | meets gate |
|------|-------------|-------|-------------|--------------|----------|------------|
| dc22 | click_null | click:c=0 | 87 | 1.0 | 10 | Y |
|  | click_null | click:c=2 | 47 | 1.0 | 10 | Y |
|  | click_null | click:c=4 | 38 | 1.0 | 10 | Y |
| ft09 | click_null | click:c=0 | 20 | 1.0 | 10 | Y |
|  | click_null | click:c=11 | 20 | 1.0 | 10 | Y |
|  | click_null | click:c=12 | 20 | 1.0 | 10 | Y |
| ka59 | click_null | click:c=0 | 10 | 1.0 | 10 | Y |
|  | click_null | click:c=1 | 10 | 1.0 | 10 | Y |
|  | click_null | click:c=15 | 10 | 1.0 | 10 | Y |
| lp85 | click_null | click:c=1 | 20 | 1.0 | 10 | Y |
|  | click_null | click:c=10 | 20 | 1.0 | 10 | Y |
|  | click_null | click:c=11 | 20 | 1.0 | 10 | Y |
| r11l | click_null | click:c=2 | 14 | 1.0 | 14 | Y |
|  | click_null | click:c=0 | 139 | 0.8777 | 16 | Y |
|  | click_null | click:c=10 | 44 | 0.4773 | None | N |
| re86 | blob_move | A3 | 943 | 0.4286 | None | N |
|  | blob_move | A2 | 938 | 0.375 | None | N |
|  | blob_move | A1 | 936 | 0.25 | None | N |
| s5i5 | click_null | click:c=13 | 20 | 1.0 | 10 | Y |
|  | click_null | click:c=2 | 20 | 1.0 | 10 | Y |
|  | click_null | click:c=3 | 20 | 1.0 | 10 | Y |
| sb26 | click_null | click:c=5 | 10 | 1.0 | 10 | Y |
|  | constdiff | A5 | 780 | 1.0 | 101 | Y |
|  | click_null | click:c=8 | 6 | 1.0 | None | N |
| tn36 | click_null | click:c=4 | 31 | 1.0 | 10 | Y |
|  | click_null | click:c=2 | 10 | 1.0 | 10 | Y |
|  | click_null | click:c=3 | 10 | 1.0 | 10 | Y |
| tr87 | constdiff | A4 | 982 | 0.4553 | 25 | N |
|  | constdiff | A3 | 976 | 0.3971 | None | N |
|  | blob_move | A3 | 976 | 0.38 | None | N |
| tu93 | blob_move | A1 | 810 | 0.5835 | 22 | N |
|  | blob_move | A2 | 756 | 0.4534 | 19 | N |
|  | blob_move | A4 | 1006 | 0.3457 | 25 | N |
| vc33 | click_null | click:c=0 | 30 | 1.0 | 10 | Y |
|  | click_null | click:c=5 | 28 | 1.0 | 18 | Y |
|  | click_null | click:c=4 | 26 | 1.0 | 16 | Y |
| wa30 | colormap | A5 | 210 | 1.0 | 189 | Y |
|  | blob_move | A2 | 936 | 0.3333 | None | N |
|  | blob_move | A1 | 940 | 0.2614 | None | N |

**Milestone S2a (strict): 10/13 games -> PASS** (pre-registered: PASS >= 8/13).

## Table 2 — levels + actions-per-completed-level, three arms

| game | lv s1b | lv t0c | lv s2a | duck | apl s1b | apl t0c | apl s2a | x-h s1b | x-h t0c | x-h s2a | pred | mispred |
|------|--------|--------|--------|------|---------|---------|---------|---------|---------|---------|------|---------|
| dc22 | 1 | 1 | 1 | 1 | 3710.0 | 3710.0 | 1901.0 | 44.17 | 44.17 | 22.63 | 37 | 10 |
| ft09 | 2 | 2 | 1 | 2 | 1984.5 | 1984.5 | 268.0 | 82.69 | 82.69 | 11.17 | 0 | 0 |
| ka59 | 0 | 0 | 0 | 1 | None | None | None | None | None | None | 0 | 0 |
| lp85 | 1 | 1 | 1 | 1 | 470.0 | 470.0 | 278.0 | 13.43 | 13.43 | 7.94 | 0 | 0 |
| r11l | 1 | 1 | 1 | 1 | 37.0 | 37.0 | 37.0 | 1.07 | 1.07 | 1.07 | 0 | 0 |
| re86 | 0 | 0 | 0 | 1 | None | None | None | None | None | None | 0 | 0 |
| s5i5 | 1 | 1 | 1 | 0 | 1881.0 | 1881.0 | 1343.0 | 25.94 | 25.94 | 18.52 | 0 | 0 |
| sb26 | 0 | 0 | 0 | 1 | None | None | None | None | None | None | 5 | 5 |
| tn36 | 0 | 0 | 1 | 0 | None | None | 25.0 | None | None | 0.6 | 7 | 5 |
| tr87 | 0 | 0 | 0 | 0 | None | None | None | None | None | None | 2 | 2 |
| tu93 | 4 | 4 | 4 | 2 | 553.2 | 553.2 | 535.5 | 19.76 | 19.76 | 19.12 | 142 | 0 |
| vc33 | 2 | 2 | 2 | 1 | 361.0 | 361.0 | 123.0 | 8.2 | 8.2 | 2.8 | 0 | 0 |
| wa30 | 0 | 0 | 0 | 0 | None | None | None | None | None | None | 2 | 0 |

**Efficiency movement (games <= 3x human median):** s1b 1/7 -> t0c 1/7 -> s2a 3/8.
