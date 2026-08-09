# EXP-B ground truth (derived from game SOURCE — NEVER shown to the model)

Source paths (read-only):
- cd82: environment_files/cd82/fb555c5d/cd82.py  (win fn `wvrremwltt`, lines 740-753)
- tu93: environment_files/tu93/0768757b/tu93.py  (win check lines 1257-1264)
- sb26: environment_files/sb26/7fbdac44/sb26.py  (win path `dbfxrigdqx` L926-927 -> `step` L800-805)

## cd82 — TEMPLATE / PATTERN MATCH
Win (`wvrremwltt`): take the target sprite `xytrjjbyib` (10x10) and the editable/player
sprite `eoqnvkspoa-*` (10x10). Build a mask that is True everywhere EXCEPT the two
diagonals (cells [i,i] and [i,9-i]). If the editable region's pixels equal the target's
pixels on that mask -> `next_level()`.
Plain: **make the editable 10x10 pattern match the target 10x10 pattern (all cells except
the two diagonals).** Mechanic: paint/rotate the editable copy (ACTION5 paints; moves
rotate) until it equals the reference.

## tu93 — COVER-ALL / SOKOBAN (all boxes on all exits)
Win (L1261-1264): let `ttauyadveo` = sprites tagged `0017unajnymcki` (the movable boxes)
and `onherbnfxo` = sprites tagged `0015msvpvzxhqf` (exit tiles). Win iff there is >=1 box
AND every box sits exactly on some exit (x==exit.x and y==exit.y). Else if no boxes / out
of steps -> lose.
Plain: **move every movable box so that each one lands on an exit tile.** Mechanic:
directional moves push the avatar/boxes through the maze; limited step budget.

## sb26 — PATTERN-MATCH COLOR FILL (replicate target into all slots)
Win path: in `dbfxrigdqx`, when the LAST slot (`pmygakdvy == len(wcfyiodrx)-1`) has been
filled with the correct color (`wrudcanmwy`), it sets `lmvwmlqtw=0`; `step` then counts
that timer to >15 and calls `next_level()` (L800-805).
Plain: **assign the correct color to every slot so the filled slots reproduce the target
color pattern; completing the final slot wins.** Mechanic: pick a color from the palette
(top row of colored 6x6 swatches) and click it into each slot (bottom row); ACTION5/7
confirm/submit.
