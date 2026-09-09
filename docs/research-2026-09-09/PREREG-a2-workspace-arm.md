# PRE-REGISTERED — arm `keith_ws` (A2: workspace + transition log + verifier), 2026-09-09 before data

**Chain of evidence this rests on.** A1 killed persistent *prose* knowledge (41 levels, +0.71 sd, 0/12 walls).
Stage-0 showed the deployed brain can emit a backtest-green executable world model where the 27B never emitted a
file at all (cn04 18/18). Stage-1 showed it does so from OUR OWN agent's transitions while STUCK at a real wall
(dc22 L2: 20/20 in 2-3 calls, both draws, models mechanistic and not memorised). All of that was OFFLINE. This arm
puts the same three capabilities in the live loop and asks whether they pay.

**What the arm adds** (`graft_workspace`, all host-side, no stock byte touched; `keith_ws` = `keith_yield900` +
exactly the `WS_*` keys): an automatic per-game transition log in the Stage-0/1 schema; a `backtest` TOOL that
replays a candidate against that log in an isolated subprocess and returns {matched, total, green, first_mismatch};
a `workspace` TOOL (save/load/list/delete) plus a preamble injecting WORKSPACE and TRANSITIONS into every python
call. Tools, not prompt text: A1 established that appended prompt blocks get <1 % uptake, and the tool schema is a
materially stronger channel. Verifier cross-validated — all four Stage-1 green models reproduce 20/20 through it,
and identity / no-contract / raising / infinite-loop controls fail with the right kinds.

**KILL TEST (first, fresh boot, ~$3):** `--arm keith_ws --games cd82,dc22,lf52 --draws 2 --concurrency 3
--max-calls 60 --per-game-s 7920` — the same 3-wall long-clock geometry used for the carry arms, whose stock base
is **8 levels / 6 runs** (cd82 1/1, dc22 2/2, lf52 1/1); carry 0.5 read 6 and carry 0.75 read 8 there.

The kill test's job is to check the MECHANISM works live, not to detect a level effect: 6 runs cannot resolve one
(carry 0.75 tied the base at 8 here and was flat at 41 in the wave). So the primary reads are mechanism reads and
levels is only a safety check.

* **GATE 1 — ENGAGEMENT: backtests >= 1 per game AND >= 50 % of runs make at least one.** Not met => the arm is
  **UNREAD, not refuted**: the model never tried the tools, and the next move is discoverability (tool description,
  a system-prompt line), NOT abandoning A2. Report uptake per run and what the model called instead.
* **GATE 2 — MECHANISM: at least one GREEN world model across the 6 runs.** Offline the brain does this in 2-3
  calls; if it cannot do it once in 6 runs x 60 calls with the same data and verifier, the ONLINE setting is the
  blocker (clock, interleaving with play, or self-generated data arriving in a worse order) and that is the thing
  to fix before any wave. Report best matched/total per level even when nothing goes green.
* **SAFETY: levels >= 6 of the base's 8** (no large regression) and the standing per-run VOID rules.
* Both gates met AND safety ok => the **25-game wave** (~$9, fresh boot, live geometry, with a Monitor).

**25-GAME WAVE READS, locked now:**
* ENGAGEMENT as above. PRIMARY = levels vs the pooled six-draw base **39.33 (sd 2.34)**: >= 48 step candidate ->
  counterbalanced redraw -> live 3-draw rule; 45-47 counterbalanced redraw; <= 44 or engaged-and-flat = DEAD.
* CO-PRIMARY = walls passed among the 12 six-draw-never-passed walls (target >= 3).
* **MECHANISM READ THAT DECIDES THE LANE — CONVERSION:** (run, level) pairs where a model went green, versus how
  many of those levels the run then actually CLEARED. A high green count with near-zero conversion is A1's
  engaged-and-flat pattern one level up the stack: the agent can model the world and still not use the model to
  play. That outcome kills A2 as a step even if levels drift up, and points at planning/search, not modelling.
* SAFETY: GAME_OVERs/run vs 0.87, live-cap vs 8.42. FIT-THE-CLOCK: play calls x e2e + verifier seconds per game
  <= 7,920 s; the verifier is host-side CPU (not a queued model call), so it should cost far less than A1's
  compactions did, and the summary reports it separately.
* VOID rules unchanged; first in a fresh boot.

**Prior, stated before data:** engagement is the real risk, not capability — Stage-1 settled capability. The
honest expectation is that a first live arm gets partial uptake and few greens, and that the first result is a
discoverability fix rather than a verdict on A2.

## RESULT 1 (CONTROL) 2026-09-09 — `keith_ws` pre-prompt-fix: ENGAGEMENT FAILED ⇒ **UNREAD, not refuted**

Wave `offkaggle/results/20260909T0804-keith_ws-kill` (fresh boot, identity ok, 0.59 h, ≈$2.5, 0 errors,
0 preemptions, 0 length-finishes).

**GATE 1 ENGAGEMENT: FAILED.** 360 model calls, **0 backtests in 0 of 6 runs** (gate: >= 1/game and >= 50 % of
runs), 1 workspace save. GATE 2 (>= 1 green model) unreachable in consequence. Per the pre-registration this makes
the arm **UNREAD, not refuted**: the model never exercised the capability, so nothing was learned about A2's idea.
SAFETY fine: levels **7 vs the base's 8**, zero-level games 0, GAME_OVERs 0.83/run vs 0.87, no VOIDs.

**CAUSE FOUND, and it is in the stock prompts rather than the model.** Both tools were correctly advertised in the
schema on every request (verified in the prompt logs and the shim), but the stock prompt asserts the opposite:
* `prompts.py:82` (system): "- The only tool is `python`; call it with one ephemeral `code` string."
* `tool_agent.py:1223` (USER prompt, re-sent EVERY turn — **44 times** in one 60-call run):
  "Only tool: `python`. It receives ..."
The harness told the model the capability did not exist while the schema offered it. The single workspace save
across 360 calls is the model noticing the schema in spite of the prompt, which is the shape one expects.

**FIX (committed eb71df3, in the graft only, flag-gated):** when WS is enabled both strings are rewritten to name
all three tools; flag-off leaves both stock, and a missing string is a recorded skip rather than an error so a
bundle change degrades quietly. The replacement ends "`python`." so the stock sentence that follows still reads
correctly (a grammar break the test caught). test_graft_workspace 12/12 both bundles, runner 32/32.

**RESULT 2 = the same kill test with the fix**, everything else identical (same games, draws, geometry, seed-free
sampling). That pair isolates discoverability: the control above is the zero-uptake baseline. Reads unchanged —
GATE 1 engagement, GATE 2 >= 1 green model, SAFETY levels >= 6. If engagement now passes, the arm becomes readable
and the wave decision follows the pre-registered wave reads; if it still fails with the prompt corrected, the
model is declining a capability it has been plainly told about, which IS a real (and much more interesting)
negative about A2 rather than an instrument artefact.
