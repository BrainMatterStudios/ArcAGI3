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

## RESULT 2 2026-09-09 — `keith_ws` WITH the prompt fix: the verifier is still UNUSED

Wave `offkaggle/results/20260909T0844-keith_ws-kill2` (fresh boot, identity ok, 0.59 h, ≈$2.5, 0 errors,
0 preemptions). Prompt confirmed corrected in the transcripts ("Tools: `python` ..., `backtest` ..., `workspace` ...").

| | control (prompt says python-only) | fixed (prompt names all three) | stock base |
|---|---|---|---|
| backtest calls | 0 in 6 runs | **0 in 6 runs** | n/a |
| workspace calls | 1 save | 3 calls / 2 saves, in 1 run | n/a |
| levels | 7 | 8 | 8 |
| score | 1.81 | 5.11 | — |
| never-passed walls present / passed | 2 / 0 | 2 / **2** (dc22, lf52) | 2 / 1 |
| GAME_OVERs per run | 0.83 | 0.33 | 0.87 |

**GATE 1 ENGAGEMENT: FAILED AGAIN — 0 backtests in 358 python calls and 877 logged transitions.** GATE 2
unreachable. The graft itself is healthy: 340 preambles injected, 877 transitions captured, 2 saves, 0 errors,
0 skips. The model used the *storage* affordance (one run saved `lib.py` 1,178 chars and `mechanics` 1,265 chars)
and never once used the *verifier*.

**This is now a substantive finding, not an instrument artefact.** Two independent discoverability conditions —
tools in the schema only, and tools in the schema WITH the prompt corrected to name them — both produced zero
verifier use. Combined with Stage-1, where the same brain built a green 20/20 model of this very game's wall in 2-3
calls when handed the data and asked, the gap is located precisely: **it is not capability and not discoverability.
The model does not spontaneously choose to invest in building a world model mid-game.** Its habitual
observe-hypothesise-act loop has a short payoff horizon; model-building is a large deferred-payoff detour it never
elects to take.

**Do not read the secondary numbers as a win.** Levels 8 vs the control's 7 vs the base's 8, on 6 runs, is noise;
score 5.11 vs 1.81 is a handful of efficient clears; both never-passed walls present were passed here (control 0/2,
carry-0.75 1/2), which is suggestive at n=2 and nothing more. None of it can be attributed to a verifier that was
never called.

**FORK (Ahmed's call; nothing started).**
1. **Harness-directed model building** — when a run is stuck at a wall for K calls, the harness itself asks the
   model to produce a world model from the recorded transitions and hands the verified result back. This makes the
   live loop match the Stage-1 setup that demonstrably works. Risk on record: five engaged-and-flat replications say
   harness-forced *behaviour* shaping does not move levels — but those forced things the model was already doing
   (act sooner, analyse less), whereas this directs it to do something it does WELL and never elects. ~1 day, ~$3.
2. **A3/A4 — port NOOA or Polyphony**, loops built around model-building rather than the duck's act-first loop, so
   the investment is structural rather than optional. 1-2 days each; the clock gate is the first read.
3. **Stop here on A2** and put the remaining time into Track D (Oct-1 absorption), which the plan calls certain value.
Recommendation: (1) first, because it is the cheapest test of the one hypothesis Stage-1 left open, and its result
also tells us whether (2) is worth the port.

## PRE-REGISTERED — arm `keith_wsd` (harness-DIRECTED model building), 2026-09-09 before data

Results 1 and 2 located the gap: not capability (Stage-1: green 20/20 model of dc22's wall in 2-3 calls), not
discoverability (two conditions, 0 verifier calls in 358 python calls). The model does not ELECT to build a world
model mid-game. `keith_wsd` = `keith_ws` + `WS_DIRECT_ENABLE=1 WS_DIRECT_AFTER_ACTIONS=40 WS_DIRECT_MAX_CALLS=3
WS_DIRECT_MAX_PER_GAME=2 WS_DIRECT_MAX_TRANSITIONS=24`: after 40 actions on one uncleared level the harness runs
the Stage-1 procedure itself and, only on a VERIFIED green model, tells the play loop it exists and how to use it.

**Stated risk, on the record:** this is harness-initiated, and the campaign has FIVE engaged-and-flat replications
(patch 21, yield900, probe discipline, carry x2) saying harness-forced behaviour does not move levels. The argument
for why this differs — those forced things the model already did, this directs the one thing it does well and never
chooses — is plausible, not proven. This kill test is what tests it. Prior: genuinely uncertain, and the most likely
failure is that a verified model gets handed over and the model still plays the way it always did (conversion 0).

**KILL TEST:** same geometry as every A2 read — `--games cd22,dc22,lf52 --draws 2 --concurrency 3 --max-calls 60
--per-game-s 7920`, fresh boot, ~$3. Comparators on this geometry: stock base 8 levels / 6 runs; keith_ws control 7;
keith_ws prompt-fixed 8.

**READS, locked:**
* **GATE 1 — DID IT FIRE:** directed attempts >= 1 in >= 3 of the 6 runs. If the 40-action trigger rarely fires,
  the threshold is wrong and the arm is unread (report actions-per-level distribution and re-tune, do not conclude).
* **GATE 2 — DID IT BUILD:** >= 1 VERIFIED (green) model across the wave. Offline this took 2-3 calls; if the live
  directed loop produces none in ~6-12 model-build calls, the difference is the live data (self-generated,
  possibly redundant or mid-level) and that is the thing to fix.
* **PRIMARY — CONVERSION, the read that decides A2:** among (run, level) pairs where a VERIFIED model was handed to
  the play loop, how many of those levels the run then CLEARED. >= 1 conversion with levels not worse => carry to a
  25-game wave. **0 conversions with >= 2 verified models handed over => A2 is DEAD as a step**: the agent can be
  given a correct, verified, executable model of the wall it is stuck on and still not clear it, which locates the
  remaining gap in planning/search rather than world-modelling, and Track A moves to A3/A4 + Track D.
* SAFETY: levels >= 6 (no large regression); GAME_OVERs/run vs 0.87; directed model-build calls counted against the
  clock (fit-the-clock reported: they are queued model calls, ~150 s each live, so 3 per level is ~6 % of a game).
* Secondary: best matched/total per directed attempt even when not green; whether the play loop then LOADS the saved
  model (workspace loads > 0) — a verified model it never opens is a different failure from one it opens and misuses.

## RESULT 3 2026-09-09 — `keith_wsd` directed build: GATE 1 PASSED, **GATE 2 FAILED** (0 models in 15 calls)

Two runs: `20260909T1128` ABORTED by me after 12 min (~$1) on an instrument fault — the entry grid was rendered as
space-separated ints, putting cd82's prompt at 27,999 tokens with 4,769 left for output. Fixed to the Stage-0 hex
encoding (a full 64x64 grid + 24 transitions now renders in ~1.8k tokens, with a test asserting < 12k).
`20260909T1146-keith_wsd-kill2` is the real read (0.88 h, ≈$3, 0 errors, 0 preemptions).

* **GATE 1 — DID IT FIRE: PASSED.** 5 (run, level) pairs in **5 of 6 runs** (cd82 L1 and L3, dc22 L1 and L2, lf52 L2).
  The 40-action threshold is right; the trigger is not the problem.
* **GATE 2 — DID IT BUILD: FAILED. 0 verified models in 15 model-build calls — 15/15 `no_code`.** Every call ended
  `finish=length` having spent its whole output budget reasoning: prompts 11.0-14.8k, output 18.0-21.8k, ~200-240 s
  each, **2,824 s of GPU** total across the wave.
* **PRIMARY (conversion): unreachable.** Nothing was ever handed to the play loop, so A2's deciding question is
  still unanswered.
* SAFETY: levels 7 (base 8, keith_ws 7-8), zero-level games 1, GAME_OVERs 1.00/run vs 0.87. The 2,824 s tax bought
  nothing; in live 25-game geometry those would be queued ~150 s calls, i.e. ~19 play calls per game surrendered.

**DIAGNOSIS, and it is a narrow operating window I had not identified.** Stage-1's greens came from a **7,123-token
prompt with ~25k output**. Live, the same procedure sees 11-15k prompts (self-generated transitions have far denser
changed-cell lists than the curated 20-transition slice) and therefore 18-22k output, and at that ratio the model
reliably overruns its budget inside `<think>` — the exact Stage-0 hard-game failure mode. So the directed build is
not refuted as an idea; the implementation is outside the window where the brain can do it.

**HONEST ACCOUNTING.** Four A2 runs so far: control (unread — stock prompt said python was the only tool), prompt-fixed
(unread — model will not elect to build), directed v1 (aborted — my int encoding), directed v2 (fires, cannot build —
my prompt/output ratio). Three of the four were faults in my instrument, not facts about A2, and the pre-registered
gates are the only reason that is visible rather than a tidy false story about the idea failing. Total ≈$9.

**DECISION (Ahmed's; nothing started).**
1. **One more parameter pass, ~$3:** `WS_DIRECT_MAX_TRANSITIONS` 24 -> 12 and a hard changed-cell cap (~60/transition)
   to land the prompt near 6-7k and output near 25k — Stage-1's proven window. This is a one-line change to an arm
   that already fires reliably, and it is the only untested version of the hypothesis. Risk: a fourth instrument
   iteration on my own diagnosis.
2. **Stop A2** and take the pre-registered fallback: A3/A4 (port NOOA or Polyphony, loops built around model-building)
   plus Track D (Oct-1 absorption), with three weeks to the milestone.
Recommendation: (1) exactly once, with a hard stop — if the directed build still cannot produce one verified model
inside Stage-1's own window, the live setting genuinely differs from the offline one and A2 goes to the fallback
without further tuning.

## PRE-REGISTERED — `keith_wsd` pass 2, with a HARD STOP (2026-09-09, before data)

One parameter change to an arm that already fires reliably: `WS_DIRECT_MAX_TRANSITIONS` 24 -> **12** and a new
`WS_DIRECT_MAX_CELLS` **60** (was an unflagged 300). Nothing else differs. Target: live prompt ~3-5k and output
~25-27k, i.e. inside the window where Stage-1 produced greens (7,123-token prompt, ~25k output, green in 2-3 calls).
Measured on the real dc22L2 transitions: 24/300 renders ~3,846 tokens, 12/60 renders ~2,483.

**THIS IS THE LAST TUNING PASS. The stop is binding and stated before data:**
* **GATE 2 — >= 1 VERIFIED model across the wave.** If the directed build produces even one, the CONVERSION read
  finally becomes answerable and A2 continues on its pre-registered wave path.
* **0 verified models => A2 GOES TO THE FALLBACK with no further tuning**, whatever the levels say. Three passes
  will then have shown that a procedure which works offline in 2-3 calls cannot be made to work inside the live
  loop, and the remaining time goes to A3/A4 (port NOOA / Polyphony, whose loops are built around model-building)
  plus Track D (Oct-1 absorption). I will not propose a fourth encoding change.
* Also read, whatever happens: prompt/output tokens per directed call (did the change land in the target window at
  all — if prompts are still > 8k the parameter did not do what the arithmetic says and that is a separate fact);
  directed seconds spent; levels vs base 8 as a safety check; GAME_OVERs vs 0.87.

Same geometry as every A2 read: `--games cd82,dc22,lf52 --draws 2 --concurrency 3 --max-calls 60 --per-game-s 7920`,
fresh boot, ≈$3. Running total on A2 after this: ≈$12.

## RESULT 4 2026-09-09 — `keith_wsd` pass 2: window hit, still 0 models ⇒ **HARD STOP APPLIED, A2 → FALLBACK**

Wave `offkaggle/results/20260909T1303-keith_wsd-kill3` (fresh boot, identity ok, 1.23 h, ≈$4, 0 errors, 0 preemptions).

**The parameter change did exactly what the arithmetic said, and it changed nothing.**

| | pass 1 | pass 2 | Stage-1 greens |
|---|---|---|---|
| directed prompt | 11.0-14.8k | **5.7-11.6k** | 7,123 |
| output available | 18.0-21.8k | **21.2-27.1k** | ~25k |
| verified models | 0 of 15 | **0 of 24** | green in 2-3 calls |

GATE 1 passed even more strongly than before: fired on **8 (run, level) pairs in 6 of 6 runs**. GATE 2 failed:
**24 model-build calls, 24 no-code, 0 verified models**, 5,016 s of GPU. Every call ended `finish=length` having
spent 21-27k tokens inside `<think>` without emitting a fence. CONVERSION remains unanswerable.
Levels 8 (= base), zero-level games 0, GAME_OVERs 0.17/run (base 0.87), 1 of the 2 never-passed walls present passed.

**The pre-registered hard stop applies: A2 goes to the fallback, with no fourth tuning pass.** Three passes, and the
decisive one put the live directed build inside the exact prompt/output window where the identical procedure went
green offline in 2-3 calls. The window was never the explanation.

**WHAT IS ESTABLISHED, AND WHAT IS NOT.** Established: (a) the brain builds correct verified executable world models
offline from our own stuck-at-a-wall transitions, reproducibly (Stage-1, 4/4 green); (b) live, inside the duck loop,
it will not elect to use a verifier (0 calls in 358, two discoverability conditions) and will not produce a file even
when the harness directs it with the same data, same contract, same verifier and the same token window (0 of 24);
(c) so the offline result does not transfer into this loop, and the failure is in the LOOP, not the brain and not the
tooling. NOT established, and now unanswerable by this route: whether a verified model would convert into cleared
levels. A2's central question dies untested.

**A plausible mechanism for the offline/online gap, recorded but NOT tested** (it would be a fourth pass): offline
each build was a fresh, dedicated process whose entire context was the task; live, the directed call is issued from
inside a long-running agent whose habits, notes and 30+ turns of play history dominate its behaviour, and it appears
to reason like a player rather than a modeller no matter what the system prompt of that one call says.

**A2 CLOSED. Fallback per the plan: A3/A4 (port NOOA or Polyphony — loops in which model-building is structural
rather than an optional detour) plus Track D (Oct-1 absorption).** A2 total ≈$13 over 5 runs.
