# Mechanic Coverage Report

Probe budget: 400 actions/game · 9 games · useful threshold: held-out pred_acc ≥ 0.7

| game | agent | acts | pred_acc | move | world | best primitives | plan? | cleared | cov | useful |
|---|---|---|---|---|---|---|---|---|---|---|
| collect | Y | [1, 2, 3, 4] | 0.7 | 0.7 | 1.0 | toggle,move | Y | 3 | 0.83 | **YES** |
| su15 | n | [] | 0.0 | 0.0 | 0.0 | — | n | 0 | 0.2 | no |
| sk48 | n | [] | 1.0 | 0.0 | 1.0 | — | n | 0 | 0.3 | no |
| re86 | Y | [1, 2, 3, 4] | 0.875 | 0.875 | 1.0 | toggle,move | Y | 0 | 0.782 | **YES** |
| wa30 | Y | [1, 2, 3, 4] | 0.0 | 0.0 | 0.0 | toggle | Y | 0 | 0.52 | no |
| m0r0 | Y | [1, 2, 3, 4] | 0.5 | 0.0 | 1.0 | paint_on_move | n | 0 | 0.45 | no |
| ls20 | Y | [1, 2, 3, 4] | 0.996 | 0.983 | 1.0 | move | Y | 0 | 0.759 | **YES** |
| tn36 | n | [] | 0.0 | 0.0 | 0.0 | — | n | 0 | 0.2 | no |
| tr87 | Y | [1, 2, 3, 4] | 0.0 | 0.0 | 0.0 | — | n | 0 | 0.26 | no |

**Useful transition models: 3/9 = 33%**
**KILL GATE (≥20-25%): PASS — proceed to Phase 2 (primitive library + model beam)**
