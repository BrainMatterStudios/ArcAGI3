# ffa7g — frame-full + action7-anim + goal-guidance + no-impact (band ∪ LLM world-model)

The integrated feature stack on top of the frozen baseline. `g` = goal-guidance
prompt; the no-impact detector (`n`) is layered on the same env-toggle loop. Stored
as a patch on a **copy of `../baseline-v12/`** (never edited in place), per the
harness rules.

- **Derives from:** `../baseline-v12/` (bundle-v12), composing three earlier variants
  as real source edits: `../frame-full/` (`ARC3_FRAME_MODE=full`), `../action7-anim/`
  (ACTION7 round-trip + animation metadata), plus goal-guidance + no-impact (new here).
- **Patch:** `patch/ffa7g-full-stack.patch` — 7 files, applies `-p1` to a fresh copy of
  `baseline-v12/src/ARC3-Inference` (dry-run clean). Reconstructs the exact source the
  in-flight `g4run-v12ffa7g*` bundles run.
- **Bundle:** `gs://cellens-ai-artifacts/arc3-duck/tufa-exact/bundle-v12ffa7g*.tgz`
  (`g` = without no-impact, `gn` = with). Launch: `gcp/launch_ffa7g.sh`.

## Feature layers (7 files)
1. **frame-full** (`frame_mode.py`, `runtime_state.py`, `action_names.py`, `prompts.py`):
   full-frame images per action. Env `ARC3_FRAME_MODE=full`; also runs in `ascii`.
2. **action7-anim** (`action_names.py`, `solver.py`, `tool_agent.py`): ACTION7 round-trip
   fix + always-visible per-action animation metadata (animated objects rendered as
   strings, per-frame cell counts, bbox top-left; `show_animation_by_objects` /
   `show_animation_by_bbox` sandbox methods).
3. **goal-guidance** (`prompts.py`): `game_overview` rewritten so the agent does not
   mistake a once-per-turn incrementing HUD for the goal; action-efficiency mantra.
4. **no-impact detection** (`solver.py`, `tool_agent.py`, `python_tool_sandbox.py`,
   `prompts.py`): stop wasted actions (wall-presses in exploration) whose only board
   change is deterministic housekeeping (a HUD/timer band). Default `intent="explore"`
   (stops on no-impact); agent opts out with `intent="solve"`. Detect-and-stop, reported
   as `stop_reason="no_impact_action"`. Two overlaid detectors, union'd (never subtract):
   - **Statistical band** (`_HousekeepingBand`): online per-row/col change frequency,
     `window=20, threshold=0.9, warmup=8`, reset on level-increment. Per-*cell* frequency
     fails (a bar's frontier changes each cell once); per-row/col is the fix.
   - **LLM code world-model** (`_HudCodeModel`, CEGIS): the agent writes `advance_hud(frame,
     action) -> frame` in a `HUD model:` note; harness runs it inline (SIGALRM-guarded,
     restricted builtins), verifies the bar prediction each action, and unions the
     predicted cells into the housekeeping mask **only when verified**. Write-once /
     run-free: dropped (`stale`) after 2 consecutive mispredictions → nudge to rewrite →
     falls back to the band. Handles no-HUD games (model never registers → pure band).
     `no_impact_source` ∈ {exact, band, model, model+band}.

## State-graph layer (`sg`, newest — bundle-v12ffa7gnsg)
Per-level graph of CANONICAL (HUD-excluded) game states + the actions between them, built on the
same HUD isolation as no-impact. Nodes = canonical boards (letter grid with the rolling housekeeping
band masked by an out-of-band `.` sentinel — NOT colour 0, which would collide); edges =
`(state, action_display) -> next_state` (multimap, determinism-tolerant). Reset per level; mask uses
the CURRENT rolling band (a monotonic union over-masked transient animations -> false self-loops).
- **Sequence-level predict-and-block** (explore): a whole plan is rejected *unexecuted*
  (`stop_reason=known_action_rejected`) iff EVERY action follows an already-tried deterministic edge
  (reaches no new `(state, action)`). Any untried/non-deterministic action in the plan -> allowed, so
  the agent may route through known states as long as the plan ends somewhere new. Self-loops stay
  with the per-action no-impact stop.
- **Intents:** `explore` (block all-known plans), `solve` (no checks), `re-explore` (traverse known
  states; **A/B toggle** `ARC3_REEXPLORE_STRICT` = soft flag vs hard-stop when a batch reaches nothing new).
- **Sandbox `state_graph` global:** `.current`/`current_state_id`, `.neighbors`/`.tried_actions`/
  `.frontier`, `.distance(a,b)`/`.path(a,b)` (BFS), and `frame_from_state(id)` -> a `FrameView`
  (`.ascii`/`.segmentation`) for any node (new sandbox↔host protocol + `graph_query` callback).
- **Validated:** dry-runs (0 false revisits/self-loops; mask=HUD; sequence block `[UP,UP]`→reject,
  `[UP,UP,RIGHT-untried]`→allow) + a live ls20 run (7 nodes/8 edges tracked, `state_node_id` flows,
  shipped to sandbox, coexists with no-impact, 0 tracebacks). Files: solver.py, tool_agent.py,
  python_tool_sandbox.py, runtime_state.py, prompts.py.

## State-graph LEAN surfacing revision (`sg` v2 — supersedes the shipped-graph view above)
Motivation: the gnsg/gnsg2 runs regressed (ex-`ft09` 1.11 vs baseline 1.624) at +334 tok/action over
baseline **with 0 graph-tool calls** — i.e. the tax was the graph being *carried*, not *queried*. The
old `light_view` serialized up to 200 nodes + all their edges into the runtime state **every turn**.
This revision ships only the current node's local frontier each turn and moves all deeper structure
behind on-demand host calls (user redesign: "just show immediate explored/unexplored actions, and have
methods for new actions"). Same 5 files.
- **Turn co-identity:** each node also carries the TURN it was first reached (`state_turn`); node refs
  surface as `{id, turn}` and `frame_from_state`/lookups accept `{'turn': N}`. `id_at_turn(t)` resolves
  a state by the turn it appeared. Dedup is still by canonical board (revisits collapse); turn is a
  label, not a key.
- **Per turn (tiny, always in the result):** `edges_here` = the current node's immediate edge map
  `{action: {to,turn} | 'unexplored' | {to:[ids],nondet}}`; `untried_here` = the `'unexplored'` actions
  (local frontier). MOUSE is always `'unexplored'` (open coordinate continuum).
- **Full graph NOT shipped.** `minimal_view()` (current id+turn, node count, immediate edges) replaces
  `light_view()`. Deeper queries are host-computed round-trips via a generalized `graph_query` protocol
  (`op` ∈ frame/chains/neighbors/distance/path/frontier/immediate_edges): `state_graph.chains(depth=k)`
  (agent-chosen depth; enumerates explored action chains, each ending in a node `{to,turn}` or
  `'unexplored'`), `.frontier()`, `.path(a,b)`/`.distance(a,b)`, `.neighbors(id)`. Cost only-when-called.
- **Validated:** all 5 files + sandbox bootstrap compile; patch dry-run clean; standalone unit test of
  `_StateGraph` (turn identity, `edges_here`, `chains(depth=2)` matching `[UP,LEFT]→2`/`[UP,RIGHT]→
  unexplored`, `frontier`, `distance`/`path`, `minimal_view`, `{'turn':N}` resolution). **Pending:**
  live end-to-end sandbox round-trip (running in the GCP A/B below).
- **`ARC3_STATE_GRAPH` opt-out master switch** (`_state_graph_enabled()`): default ON; `off` builds no
  graph, surfaces no graph fields, and drops the graph prompt bullets (`STATE_GRAPH_ADDENDUM` gated in
  `_build_system_prompt`) — a clean whole-feature ablation == ffa7gn, for the "does the graph help?" A/B.

## Subset-A/B iteration harness (`v12_run.py` + gcp scripts)
`ARC3_GAME_SUBSET="r11l tn36 ..."` (space/comma list of 4-char prefixes or full ids) filters the run to
a subset; no-op when unset (safe for every other run). Passed via GCP metadata `arc3-game-subset`; the
bundle is now metadata-parametrised too (`arc3-bundle`, default `bundle-v12ffa7gnsg.tgz`), plus
`arc3-state-graph`. Launch knobs: `GAME_SUBSET=`, `STATE_GRAPH=`, `BUNDLE_NAME=` in `launch_ffa7gnsg.sh`.
- **Rationale:** full-25 (×2, 4h budget) is ~9h; token-efficiency is a per-action property visible on a
  few games, and scoring signal is concentrated (ft09 swings the mean ±1.0; graph gains are lopsided).
  Iterate on a graph-heavy subset, confirm on full-25 only pre-submission (a subset can't catch a
  regression on an excluded game).
- **Graph-heaviness ranking** (distinct nodes / revisits, from the gnsg logs) picked the subset
  `r11l tn36 sb26 wa30 sk48 bp35 ka59` — overturning the ls20/tu93/vc33 seeds, which showed ~0 graph
  activity in the actual run.
- **Live A/B (2026-07-19):** `bundle-v12ffa7gnsglean.tgz` on the 7-game subset, one variable:
  arm A `arc3-g4-v12ffa7gnsglean-a` (graph ON) vs arm B `-b` (`ARC3_STATE_GRAPH=off` == ffa7gn).
- **RESULT — graph net-negative, so DEFAULT FLIPPED to OFF (opt-in):** final mean Kaggle score
  **1.533 (graph) vs 2.061 (no graph)**. Per game (A/B): r11l 4.76/0.27 (graph *far* more action-
  efficient: L1 in ~18 vs 92 actions), ka59 0.41/0.00, bp35 0.51/0.18 — graph wins on efficiency;
  but sb26 1.70/3.27 and tn36 3.36/10.71 — no-graph reached L2 on both, and depth (weight×2 + a 2nd
  capped-115 level) outweighs one efficient L1. Mechanism: the graph makes each level cheaper but its
  per-turn reasoning cost (~561 vs 455 gen tok/action) yields fewer total actions in the time budget
  (2,121 vs 2,494, −15%) → less depth, and the score rewards depth. Third consistent negative (gnsg,
  gnsg2, lean A/B). `_state_graph_enabled()` is now **opt-IN** (default OFF; `ARC3_STATE_GRAPH=on` to
  re-enable). Graph kept behind the flag for future improvement. Input-side tax fully removed by the
  lean surfacing; the remaining cost is the model's *reasoning* about the graph, which is the thing to
  fix before re-enabling.

## Verification
- All 7 files compile. Patch dry-run applies clean to a pristine baseline copy.
- **End-to-end extraction bug (found + fixed via a live local run):** the agent emits
  `advance_hud` inside a fenced code block, but `_extract_scientist_note` routed it through
  the prose extractor, which `.strip()`s each line and collapses newlines to spaces — so the
  code arrived as ONE line and `_HudCodeModel.register` raised `SyntaxError`, silently no-op'ing
  the whole model path (permanent fallback to the band). Fixed with `_extract_hud_model_code`,
  which pulls the fenced `def advance_hud` block raw (indentation + newlines preserved).
  Re-tested through the real `_extract_scientist_note` → register → predict path: registers,
  predicts, prose fields unaffected. The dry-run had missed this because it fed `register` a
  clean code string, skipping the extractor.
- **no-impact on ls20** (112 recorded actions, 36 true housekeeping-only, replay dry-run):
  | detector | detected | false-positive | miss |
  |---|---|---|---|
  | band only | 35/36 | **0** | 1 (inside warm-up) |
  | band ∪ verified model | **36/36** | **0** | 0 |
  The union is provably ≥ band (model can only add masked cells), so a partial/under-
  predicting model cannot regress detection; here it strictly improves (covers the
  band's warm-up window from action 1). 0 false-positives = never stops a real move.
- **advance_hud end-to-end (agent actually authors + uses a model):** pending local
  qwen3.6:27b (Ollama) ls20 run.

## Validated score (public 25 games, ex-`ft09`, 2 passes each — FINAL)
Clean 1-variable ablation: same bundle ± the no-impact band. Both runs completed
2026-07-18.
| arm | pass A | pass B | **mean ex-`ft09`** | actions/pass | levels/pass |
|---|---|---|---|---|---|
| **without** no-impact (ffa7g, `-1845`) | 0.946 | 1.145 | **1.046** | ~3353 | 15 |
| **with** no-impact (ffa7gn, `-1948`) | 2.061 | 1.187 | **1.624** | ~3476 | **21** |

Headline: the no-impact band lifts ex-`ft09` **1.046 → 1.624 (+0.58, ~55%)** and clears
**more levels (15 → 21)** at **~equal total actions** (both arms hit the ~132-min time cap,
so the wasted-action savings are reinvested into progress rather than booked as a lower
action count). Both with-passes beat both without-passes, and the gain is broad (14/24
games score >0; vc33 +4.96, r11l +3.72, tu93 +2.70, sb26 +11.0). n=2 with high pass-variance,
so treat the magnitude as indicative, but the direction is consistent. `ft09` excluded — it
swings the all-25 mean by ±1.0 on its own. (Earlier `−24%`/`−57%` action-reduction figures
were partial-snapshot artifacts — the with-runs were still climbing; these are the completed
numbers.)
