# EXP-B verdicts — MATCH / PARTIAL / MISS (strict; generic = MISS)

Judging rule: MATCH = names the true win predicate incl. the right objects. PARTIAL =
right structural frame AND correct key object(s), wrong detail. MISS = wrong object,
wrong frame, or generic. Model outputs are in `<game>__<model>_output.txt` (verbatim).

| game | model            | verdict | one-line justification |
|------|------------------|---------|------------------------|
| cd82 | qwen2.5:14b      | MISS    | "move color-15 objects onto background" — no template/target-match; wrong frame. |
| cd82 | qwen2.5-coder:14b| MISS    | "convert color-4 regions to background" — recolor-to-bg, not match-target. |
| cd82 | llama3.1:8b      | MISS    | "move color-3 object to top-left corner" — invented; color 3 is the top banner. |
| tu93 | qwen2.5:14b      | MISS    | "move all color-6 objects onto background" — color 6 = the 1x64 status/energy bar at row 63, NOT the boxes. Structural frame coincidentally cover-like (primed by the prompt example) but wrong object. |
| tu93 | qwen2.5-coder:14b| MISS    | "move color-6 onto color-0 tiles" — same status-bar confusion; boxes/exits never identified. |
| tu93 | llama3.1:8b      | MISS    | "move color-6 objects onto a single tile" — status bar again; and "single tile" is wrong (multiple exits). |
| sb26 | qwen2.5:14b      | MISS    | "recolor all color-2 to 3" — color 2 = the row-53 status bar (ACTION5 flipped 2->3 there); missed palette->slot fill entirely. |
| sb26 | qwen2.5-coder:14b| MISS    | "move color-0 cells onto background" — wrong; no palette/slot/target notion. |
| sb26 | llama3.1:8b      | MISS    | "move color-5 objects onto background" — generic move-to-bg; wrong. |

RESULT: 0 MATCH, 0 PARTIAL, 9 MISS across 3 games x 3 models.

## Why (honest root-cause, not just a score)
1. The compact object-list is SEMANTICALLY FLAT: it lists same-color connected blobs but
   does not mark which blob is the target/reference vs the editable copy (cd82), which are
   boxes vs walls vs avatar vs exits (tu93), or which is the palette vs the slots vs the
   target pattern (sb26). Two of three games' models locked onto a STATUS BAR (the 1x64
   bottom/mid-row strip) as the salient object because it is a large distinct blob and it
   visibly changed under trivial actions.
2. The shallow probe set does not reveal structure: ACTION5 in cd82 painted the editable
   region (0->15, a real clue) but the model could not connect "paint region" to "match the
   other 10x10 region". Clicks in sb26/tu93 mostly hit the status bar counter or a slot
   highlight, giving low-information deltas.
3. The prompt's own worked EXAMPLE ("all X objects moved onto Y tiles") likely biased the
   models toward a move-onto-tiles frame, yet even so they picked the wrong X.

These are BOTH a model-capability limit AND a description-quality limit. The pipeline as
tested — flat object-list + 4-10 shallow probes -> small local text model -> named win
condition — is NOT information-sufficient. A richer, role-typed scene description (target
vs editable, agent vs items vs goals, ignore volatile status bars) is a prerequisite before
this paradigm could be re-tested; that was NOT built here.
