# Unlock-failure diagnosis — 9 never-unlocked games (2026-08-03)

Sources:
- Ground truth: scripts/research_2026_07_01/ solvers, ALL RE-VERIFIED against current engine
  2026-08-03 (logs: verify_*.log in this dir). cn04 WON 14 acts, dc22 WON 20 acts,
  g50t L1 WON (clone-recorder), lf52 WON 8 clicks, ls20 WON 13 acts, tr87 WON 15 acts,
  wa30 WON 26 acts. m0r0/sk48 decoded fresh by subagents (see below).
- Duck behavior: bsm rig dump transcripts (base config, 27B, 1800s/game,
  /private/tmp/.../066c8134-.../scratchpad/bsm/taaf_harness_artifacts/transcripts/,
  clone map k003=cn04 k004=dc22 k006=g50t k008=lf52 k010=ls20 k011=m0r0 k017=sk48
  k021=tr87 k024=wa30) + round-2 kernel transcripts (kernel_ab_wmr/, k008=g50t 170-act
  wave-5 run, k009=ls20 107-act wave-5 run) + A/B per-game rows.
- Working-solve contrast: K3 teacher episodes scratchpad/rl_gate/episodes/k3_sweep_*.
  K3 solves L1: cn04, g50t(1/2), lf52, ls20, wa30. K3 FAILS L1: dc22, m0r0, sk48, tr87.

## Duck run anatomy (base config)
- bsm (1800s): 10-39 LLM turns/game, 4-166 actions; every run ends on wall-clock
  (vLLM read-timeout or stop_requested), never an explicit quit. ~1-3 min/turn.
- A/B rounds (g50t/ls20 only): 29-202 actions/run. ls20 unlocked L1 in 4/12 runs;
  g50t 0/12.
- So the real per-game budget is ~15-40 hypothesis-test cycles. Known solutions are
  8-26 actions for 7/9 games: raw depth is NOT the binding constraint; inferring the
  mechanic within ~20 experiments is.

## Per-game

### cn04 — verified GT: connect-the-endpoints
Two sprites carry color-8/13 connector markers; win = every 8-marker overlaps another
sprite's 8-marker and 13-on-13. Controls: arrows move SELECTED sprite, SPACE(A5)
rotates it, click selects. Solution 14 acts (3 rotates + 11 moves).
Duck (k003, 36 acts, 32 turns): perceives "containers and red blocks", pushes sprites
around/through each other, hypothesis drifts (put red blocks in green container),
fragments overlap visuals, never reads the 8/13 endpoint semantics. Movement+rotate
both exercised; goal never inferred.
BLOCKER: goal-inference. K3 solved it (~60 acts) → inferable from play at Claude level.

### dc22 — verified GT: click bridge-buttons + walk (interleaved)
Avatar and goal in disconnected floor clusters; clicking panel buttons toggles paired
bridges (intangible=walkable); win = avatar reaches goal cell; solution 20 acts
interleaving 3 button clicks with moves. Has GAME_OVER hazards.
Duck (k004, 13 acts in bsm; g0 0/7): does click panel crosses and observes bridge/
tile diffs, but the button→bridge pairing + the need to interleave movement is never
assembled into a plan; panel layout diffing swamps it.
K3 ALSO fails (2 episodes, 326/334 events, deaths). BLOCKER: goal-inference
(button→bridge pairing is a hidden relational mapping), depth secondary. Hardest
class: even the teacher fails.

### g50t — verified GT: clone-recorder (Talos-style), NO direct path
Avatar confined to a 12-cell region; goal physically unreachable by walking. Phase 1:
walk a path, press SPACE(A5) → banks the path as a ghost + resets avatar to start.
Phase 2: ghost replays your recorded moves in lockstep; park ghost on button (37,7) →
gate slides open → avatar can reach goal. Human baseline 78 acts. Blue bottom bar =
move-deadline timer.
Duck (k006 bsm + k008 kernel 170-act run, 702 thinking blocks): models it as plain
maze-navigation for the entire run; notices moves that "don't move" (confinement) and
sudden teleports to start (the A5 reset) but never forms the record/replay hypothesis
("clone" mentioned ~5x incidentally in 702 blocks). 0/12 across both A/B rounds.
BLOCKER: goal-inference (two-phase hidden mechanic; A5's effect looks like a
punishment/reset, not a tool). K3 solved 1/2 episodes → borderline inferable.

### lf52 — verified GT: peg solitaire by clicks
Click peg selects it (revealing direction markers at legal jump landings), click the
landing 2 cells past it jumps and removes the jumped peg; win when 1 peg remains.
Solution 8 clicks. Budget: lose at 64 clicks.
Duck (k008 bsm, 18 acts): clicks pegs, sees the selection/marker reveal, and
misreads it as "activating crosses" — plans to click each cross once ("activate the
remaining 4"), i.e. inverted goal (keep everything vs remove). Click budget then dies.
BLOCKER: goal-inference (feedback actively misleads: selection halo looks like
progress). K3 solved (~300 events incl. clicks at row 16/17) → inferable.

### ls20 — verified GT: rotation-gated maze goal
Navigate to static goal, but goal acts as wall unless avatar rotation matches goal
rotation; rotation changes ONLY by stepping on the rotation tile (cycles 90° per
visit); rotation state is rendered only on a small HUD indicator sprite, not on the
avatar. Solution 13 acts (1 rot-tile visit). 42 steps/life, 3 lives.
Duck (k010 bsm + k009 kernel): correctly does maze-BFS to the goal, arrives, nothing
happens, then hypothesis-thrashes ("move colored blocks", "match pattern"). "rotat"
mentioned 13x in 342 blocks but never linked to the tile/indicator. Unlocks 4/12 in
A/B rounds — consistent with accidental odd-parity rot-tile crossings en route.
BLOCKER: goal-inference of a hidden precondition (indicator visible but its semantics
illegible); perception secondary (avatar itself shows no rotation). K3 solved (85 ev).
Mask check (verified live 2026-08-03): the rotation indicator changes rows 57-58 on a
rot-tile visit — NOT inside the masked HUD band (61-62). The mask does not delete the
signal; but the indicator is a bottom-edge strip, exactly the shape our prompt tells
the model to discount as HUD, so attention-discounting is a plausible contributor.

### m0r0 — verified GT (fresh decode 2026-08-03): mirrored twin-dot merge
Two color-10 dots; every arrow moves BOTH, but the right dot's x-component is NEGATED
(m0r0.py lines 810-817) — LEFT/RIGHT push them oppositely, UP/DOWN together. Blocked
dot stays while twin moves (wall asymmetry = only way to change offset). Win = both
dots on the same cell (piznjwrdhn==0 → next_level, lines 863-865). 150-action lose
budget. BFS-minimal solution 15 acts (human baseline 30); state space tiny (596
states). Fully frame-visible. A5 = handler-less no-op; L1 clicks = pure no-ops.
HARNESS DEFECT (verified live 2026-08-03): rows 0+63 are a budget-bar HUD ticking
every ~2-3 actions, and m0r0 is ABSENT from hud_mask.py HUD_MAP (was classified
"correctly no HUD") → no-op/effect detection on m0r0 is corrupted ~every other action.
Duck (k011, 89 acts, 22 turns): perceives two chambers each with a 5x5 sky-blue
square inside a black figure; discovers both move together on arrows with different
clearances; tests and rejects hypotheses (return-to-start, equal-height). Systematic
but goal never found. K3 FAILED too (264 ev, deaths, score 0).

### sk48 — verified GT (fresh decode 2026-08-03): loading-arm order matching
Extendable striped arm on a left-edge rail: ACTION4/3 grow/shrink one segment,
ACTION1/2 slide the arm along the rail (only at notches). Blocks pinned against the
right wall get boarded when the arm slides under them; win = blocks riding the arm,
head→tip, match the bottom legend's color order (8,14,9) — Sk48.gvtmoopqgy, lines
842-859. ACTION6 = select arm (no-op on L1); ACTION7 = free UNDO (not needed to win).
196-move budget, no traps. BFS-minimal solution 14 acts (human baseline 61); fully
frame-visible incl. the goal legend; budget bar row 53 is the known HUD offender.
Duck (k017, 166 acts, 39 turns — most actions of the 9): models the arm as a drawn
"trail" and block-boarding as "collecting by touch"; watches bottom-panel legend
icons for confirmation but never reads them as an ORDER spec; pushes blocks without
ever pinning+sliding-under. Also tried ACTION7 and got "Unknown action at index 1"
(pre-fix harness bug — but UNDO is optional, so this did NOT gate the win).
K3 FAILED too (141 ev, score 0). BLOCKER: goal-inference (pickup rule + order-legend
goal), depth-of-search secondary (needs a coordinated ≥14-move plan; greedy pushes
look like progress and aren't).

### tr87 — verified GT: rewrite-rule matching (symbolic)
Top: six LHS=RHS glyph rules. Bottom: fixed TARGET row (A-glyphs) + editable WORKING
row (B-glyphs). Win = working[i] == rule(target[i]) for all 5 cells, matching by
rotation-invariant glyph identity. Controls: LEFT/RIGHT move cursor, UP/DOWN cycle
selected cell's glyph (7-cycle). 128-press energy budget. Solution 15 acts.
Duck (k021, 6 acts in bsm!): world model FREEZES — the identical "Layout:...
Selectors:..." text + "Plan: probe RIGHT" repeated verbatim 10+ turns; almost no
actions issued. The copy-forward world-model prompt instruction becomes a trap when
the model stops updating. K3 also failed (117 ev, DOWN x71 = blind cycling).
BLOCKER: perception (5x5 symbolic glyphs + rule table illegible in ascii/segmentation
reading) + harness-adjacent pathology (world-model verbatim freeze). Solution is
trivially short once the rule table is read.

### wa30 — verified GT: grab-drag sokoban
Arrows set facing then move (facing updates even when blocked); SPACE grabs the
faced-adjacent block / releases; win = all 3 blocks anchored in the goal region and
none held. Solution 26 acts.
Duck (k024, 39 acts, 12 turns): experiments with SPACE, sees blocks "destroyed"/
"appearing" (grab = block disappears into carry; release = reappears) and models it
as destroy/teleport, not carry. Facing-vs-movement conflation. Ran out of clock while
hypothesis still wrong. K3 SOLVED (268 ev, SPACE x57 — heavy grab use, some deaths).
BLOCKER: goal-inference (carry semantics misread), action-space secondary (grab needs
face-then-SPACE composition). Reachable: 26-act solution, teacher solves it.

## A/B round context (docs/test-artifacts-2026-08-02/)
g50t 0 levels in all 12 A/B runs; ls20 4/12. Round-2 C/D mechanisms (graph veto,
compaction) never fired / broke; nothing harness-side moved these games.

## ACTION7 caveat (important for sk48/lf52 history)
The bsm sk48 transcript shows the harness OFFERING ACTION7 ("Valid actions right
now: ... ACTION7") and then REJECTING it on execution ("Unknown action at index 1")
— the pre-fix round-trip bug. The 196-run g0 dump + bsm dump are pre-patch base
config; the current v6 submission applies patch_action7 (duck_patches.py
patch_action7, pinned on in build_duck_patched.py). The A/B panels did NOT include
sk48/lf52, so the never-unlocked verdict for sk48 under the CURRENT config is
unconfirmed under v6. HOWEVER the fresh decode shows sk48's ACTION7 is a free UNDO —
optional for winning — so the pre-fix bug wasted turns and contradicted the action
list (a trust hit) but was NOT the reason sk48 never unlocked.

## Final classification + synthesis

| game | L1 GT mechanic (verified vs current engine) | duck behavior | blocker (primary/secondary) | min solution |
|---|---|---|---|---|
| cn04 | move+rotate sprite until 8/13 connector markers overlap partner's | pushes "containers", goal drift, never reads markers | goal-inference | 14 |
| dc22 | click buttons toggling paired bridges, interleave with walking | clicks buttons, sees diffs, never assembles pairing→path plan | goal-inference / depth (K3 fails too) | 20 |
| g50t | record path, A5 banks ghost that replays; ghost on button opens gate | pure maze-nav 170 acts; A5 reset read as failure | goal-inference (two-phase tool) | ~78 |
| lf52 | click peg → markers reveal; click landing = jump, remove peg; 1 left wins | reads selection halo as "activation", clicks each peg once | goal-inference (misleading feedback) | 8 clicks |
| ls20 | maze + goal gated on rotation set by rot-tile visits (indicator rows 57-58) | BFS to goal, arrival fails, hypothesis-thrash; 4/12 accidental unlocks | goal-inference (hidden precondition) | 13 |
| m0r0 | mirrored twin dots (x-negated), wall-jam to desync, merge to win | found co-movement + asymmetry, never merge goal | goal-inference / harness (HUD unmasked) | 15 |
| sk48 | arm on rail, slide under wall-pinned blocks, match legend order | "trail-drawing + collect-by-touch" model; never reads order legend | goal-inference / depth | 14 |
| tr87 | rewrite rules: cycle working glyphs to rule(target[i]); cursor+cycle keys | world model FROZE verbatim 10+ turns, 16 actions total | perception (symbolic glyphs) / harness (freeze) | 15 |
| wa30 | face+SPACE grabs block, carry to goal zone, release; all 3 in zone | reads grab as "destroy/teleport", never carry model | goal-inference (carry semantics) | 26 |

Tally: goal/mechanic-inference primary on 8/9 (tr87 perception). ZERO games are
action-space-blocked under the current (post-ACTION7-fix) config; zero are
efficiency/budget blocked; depth secondary on 2-3. Solutions are 8-26 actions for
8/9 games — well inside even a 50-action run. The failure is interpreting composite
or hidden mechanics within the ~15-40 LLM turns a run affords, and specifically:
(a) arrival-without-win not triggering a precondition hunt (ls20, cn04),
(b) apparent setbacks not considered as tools (g50t reset, wa30 disappear, lf52
    removal),
(c) revealed markers/legends not read as affordances or specs (lf52, sk48, tr87),
(d) multi-entity coupling not hypothesized (m0r0, dc22).
K3 (Claude teacher) splits 5/9 solved vs 4/9 failed → about half the gap is
harness-agnostic capability, half is inferable-with-better-priors.

Interventions ranked:
1. Mechanic-archetype playbook (prompt): catalog of the 8 archetype families with
   disambiguation experiments + the 4 heuristics (a)-(d). Cheapest, immediate,
   A/B-able. Expected reach: lf52, wa30, cn04 (+ls20 rate ↑) — the K3-solvable set.
2. Synthetic SFT corpus (feeds the parallel generator) — see below.
3. Point fixes: add m0r0 (rows 0+63) to HUD_MAP; anti-freeze guard (detect verbatim
   world-model repetition, force re-derivation) — tr87-class.
4. Not worth targeting now: dc22 (teacher fails; relational depth), tr87 symbolic
   parsing at 27B.

Synthetic-corpus content for the generator (dominant class = mechanic-inference):
game families —
  F1 precondition-gate mazes (touch tile(s) → goal arms; small indicator changes)
  F2 record-replay / clone tools (action that seems to undo progress but banks it)
  F3 select-reveal-execute clicks (peg-jump; button→remote-toggle pairs)
  F4 grab-carry with facing (block vanishes into carry, reappears on release)
  F5 coupled multi-avatar (mirrored/negated axes, differential wall-blocking)
  F6 order/alignment matching vs a rendered legend or partner markers
  F7 symbolic rule tables (glyph rewrite, rotation-invariant identity)
trajectory SKILLS (more important than the families themselves) —
  wrong-hypothesis → targeted experiment → revision → win arcs (SCoRe-shaped);
  arrival-failure recovery; interpreting no-ops and setbacks; reading legends as
  goal specs; committing to 10-30-action plans once the mechanic is pinned.
