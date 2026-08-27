# Stage-2a milestones — T1/T2 accuracy gates + effects/commit re-run

Protocol: budget 4000 actions/game, 13 frame-Markov games, effect model + commit mode ON. Held-out = prequential (test-then-train). Gate = >=90% prequential within <=300 own-scope transitions (>=10 held-out outcomes), and the accuracy HELD (still gated at end of run, or lifetime >=90%) — once-gated-then-collapsed does not count.

## Table 1 — held-out rule accuracy (top rules per game)

| game | rule family | scope | transitions | held-out acc | gated_at | meets gate | gated now |
|------|-------------|-------|-------------|--------------|----------|------------|-----------|
| dc22 | — | — | — | — | — | N | — |
| ft09 | — | — | — | — | — | N | — |
| ka59 | — | — | — | — | — | N | — |
| lp85 | — | — | — | — | — | N | — |
| r11l | — | — | — | — | — | N | — |
| re86 | — | — | — | — | — | N | — |
| s5i5 | — | — | — | — | — | N | — |
| sb26 | — | — | — | — | — | N | — |
| tn36 | — | — | — | — | — | N | — |
| tr87 | — | — | — | — | — | N | — |
| tu93 | — | — | — | — | — | N | — |
| vc33 | — | — | — | — | — | N | — |
| wa30 | — | — | — | — | — | N | — |

**Milestone S2a:** 0/13 games with a rule at >=90% held-out within <=300 own transitions -> **FAIL** (pre-registered: PASS >=8/13)

## Table 2 — Stage-1b (T0) vs Stage-2a (T0+T1/T2+commit)

| game | levels s1 | levels s2a | duck | apl s1 | apl s2a | x-human s1 | x-human s2a | pred steps | mispred | end |
|------|-----------|------------|------|--------|---------|------------|-------------|------------|---------|-----|
| dc22 | 1 | 1 | 1 | 3710.0 | 3710.0 | 44.17 | 44.17 | 0 | 0 | budget |
| ft09 | 2 | 2 | 2 | 1984.5 | 1984.5 | 82.69 | 82.69 | 0 | 0 | budget |
| ka59 | 0 | 0 | 1 | None | None | None | None | 0 | 0 | budget |
| lp85 | 1 | 1 | 1 | 470.0 | 470.0 | 13.43 | 13.43 | 0 | 0 | budget |
| r11l | 1 | 1 | 1 | 37.0 | 37.0 | 1.07 | 1.07 | 0 | 0 | budget |
| re86 | 0 | 0 | 1 | None | None | None | None | 0 | 0 | budget |
| s5i5 | 1 | 1 | 0 | 1881.0 | 1881.0 | 25.94 | 25.94 | 0 | 0 | budget |
| sb26 | 0 | 0 | 1 | None | None | None | None | 0 | 0 | budget |
| tn36 | 0 | 0 | 0 | None | None | None | None | 0 | 0 | budget |
| tr87 | 0 | 0 | 0 | None | None | None | None | 0 | 0 | budget |
| tu93 | 4 | 4 | 2 | 553.2 | 553.2 | 19.76 | 19.76 | 0 | 0 | budget |
| vc33 | 2 | 2 | 1 | 361.0 | 361.0 | 8.2 | 8.2 | 0 | 0 | budget |
| wa30 | 0 | 0 | 0 | None | None | None | None | 0 | 0 | budget |

**Efficiency movement:** 1/7 games within 3x human median (Stage-1b: 1).
