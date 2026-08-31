# J10 — Adversarial judge verdict: TP10 graft_memoryspine (2026-08-31)

Under review: `submission/_throughput_v1/graft_memoryspine.py` +
`test_graft_memoryspine.py`. Judge ran everything fresh: the 14-test suite, all
8 other graft suites, `dry_run.py` stock, a TP10-included dry run, both
install orders, exception injection at all six helpers, and four dedicated
reproduction scripts. Stock seams re-read line by line in
`submission/_inspect_replay/assets_build/ARC3-Inference/inference/agent/tool_agent.py`.

**VERDICT: SHIP-WITH-FIXES** (fix list at the bottom; F1 and F2 before arming,
F4's cap change before the A/B is interpreted).

---

## Stock-delta truth check (checklist 1) — builder claims verified, one erratum

- Per-turn live reinjection: CONFIRMED. `_summarized_knowledge_lines` :1358,
  consumed in `_build_user_prompt` :1472. Stock already re-serves live notes
  every turn; the graft correctly does NOT duplicate this.
- The three wipes: CONFIRMED. `_update_summarized_knowledge_from_step_summary`
  :1343-1356 blanks all fields except `cross_level_notes` on
  `level_transition` / `run_complete` / `game_over`; `_ensure_session`
  :1140-1152 resets `_summarized_knowledge = _empty_world_model()` on a
  runtime-dir change. The delta (persistence across those wipes) is real —
  the A/B tests something stock does not do.
- `_describe_last_outcome` :1300 dead code: CONFIRMED. Grep over the whole
  bundle: definition only, zero call sites.
- `requested_count`/`stopped_early` recorded then dropped: CONFIRMED
  (recorded :329-338 aggregate, :1674-1682 compact; absent from the
  `_summarize_step_sequence` summary dict :1284-1295).
- ERRATUM (NIT, F7): the docstring claims `stop_reason` is also dropped —
  false. Stock's summary keeps it (`"stop_reason": last.get("stop_reason")`,
  :1295). Harmless — the graft in fact relies on that kept field.

## Test results

- `test_graft_memoryspine.py`: **14/14 green**, re-run by the judge. Tests
  exercise the REAL bundle `ToolAgent` through the real wrapped methods (no
  reimplementations); the suite fails without install (test_01 + every
  content assertion). Two tests (10, 13) would pass uninstalled but are
  legitimately absence tests.
- All 8 other graft suites: green. `dry_run.py`: PASS (243 s, 3 games).
- Fail-open: PROVEN. Injected `RuntimeError` into each of `sync_journal`,
  `render_notes`, `accounting_line`, `harvest_intent`, `_game_key`,
  `_summary_level` one at a time; all five wrapped seams ran, nothing
  escaped, stock prompt intact every time.
- Stacking: installed all 8 grafts with TP10 LAST and with TP10 FIRST; both
  orders install clean and produce well-formed prompts (test_12 additionally
  proves the deaths wrapper chain).

---

## Findings

### F1 — CONFIRMED-BUG: the `Next:` echo instruction collides with stock's label-block continuation; "consumed after one serve" is false in practice

`_extract_labeled_blocks` (:375-408) appends ANY unlabeled line to the
current labeled block. `Next:` is not a stock label (`Next test:` is, and
does not prefix-match `next:`), so the graft-taught note-to-self line gets
glued into whatever labeled block precedes it. Reproduced on the real seams:

```
input:  "World model: keys open doors\nPlan: hug the left wall\nNext: try the blue door with UP"
current_plan == 'hug the left wall Next: try the blue door with UP'
turn 1: intent text appears 2x (echo header + live Plan block)
turn 2: echo header gone, but intent STILL served inside "- Plan: ... Next: try the blue door with UP"
```

Consequences: the intent is re-served every turn as the standing Plan
(defeating "reconsider, don't just obey"), it is journaled, and after a wipe
it comes back inside YOUR NOTES as a stale imperative. The graft's own
ECHO_HOWTO makes this the COMMON case, since stock prompts the model to end
turns with labeled blocks. `test_05` masks this: it asserts only that the
"YOUR PRIOR INTENT" header is absent on turn 2 while the intent text is in
fact still in the prompt (judge measured 1 occurrence on turn 2).
Repro: scratchpad `repro_a_next_leak.py`.

### F2 — CONFIRMED-BUG (co-install): "committed N" is POST-clamp under graft_throughput's batch cap — the accounting line asserts a falsehood on exactly the axis it exists to make honest

`requested_count = len(normalized_actions)` (tool_agent.py:1760, :1890) is
computed AFTER `_normalize_python_actions`, which graft_throughput wraps to
silently truncate to the per-call budget (`normalized = normalized[:remaining]`,
graft_throughput.py:268). Reproduced with the shipped stack (all grafts,
TP_BATCH_CAP=3):

```
model requested 10 actions; post-cap normalized batch = 3
accounting line: LAST TURN: committed 3 action(s), 3 executed, ended level=1, state=NOT_FINISHED.
```

The model committed 10; TP10 tells it it committed 3 and everything ran —
teaching false batch math and hiding the truncation TP performed. The
builder's claim that N is "the model's requested batch size" is wrong under
the intended co-install. Repro: scratchpad `repro_cdef.py` §C.

### F3 — CONFIRMED-BUG (minor): superseded pre-first-summary notes resurface forever next to their own correction

Notes harvested before the first step summary get `level=None`
(`_summary_level` returns None); later revisions land under `level=1,2,...`.
`render_notes` dedupes by `(key, level)`, so the `(key, None)` entry is never
superseded. Reproduced:

```
YOUR NOTES (persistent): ...
- World model: WRONG early guess: the goal is the red square
- [L1] World model: CORRECTED: the goal is the exit door, red square is a trap
```

The refuted early guess is re-served on every turn for the rest of the game.
Repro: scratchpad `repro_a_next_leak.py` §B.

### F4 — DESIGN-RISK (measured): the notes block is "empty on healthy turns" only until the FIRST wipe; afterwards it costs ~1.1k tokens on EVERY turn and measurably halves retained history depth

After any level transition or game over, the journal permanently differs from
the live block, so the tail rides every subsequent user message — and each
retained history message carries its own copy. Measured under full co-install
with a realistic 3-level journal (fields at ~280 chars, cap 4000):

```
stock(+other grafts) prompt: 3219 chars; +TP10: 7700 chars; delta = 4481 chars (~1120 tok)
user turns surviving the 32k trim: stock 7  ->  TP10 3
```

On the axis the campaign says is binding (prefill-bound throughput, 88
actions/game), TP10 trades history depth for cross-wipe memory. That may be
the right trade — it is the R11 Schema recipe — but R11 §4 itself prescribes
**≤2KB** for A/B #1, and the default here is 4000 chars. Also `ECHO_HOWTO`
(~150 chars) is appended to EVERY turn even when the echo feature is never
used. If the A/B reads null/negative, this is the first suspect; it must be
controlled (lower cap arm) rather than discovered post hoc.
Repro: scratchpad `repro_cdef.py` §E.

### F5 — DESIGN-RISK (dev-only): concurrent passes of the same game share one journal — sync thrash and cap-eviction

The solver runs each (game, pass) as its own thread (`solver.py:1056-1078`,
ThreadPoolExecutor); nothing serializes two passes of the same game, and
`_game_key` strips `_pN_` so they share a journal. When both are in flight,
each pass's sync re-appends its own text because `_latest_for_key` alternates:
measured 301 journal entries after 150 turn-pairs with the world model
changing only every 10 turns (~2 entries/turn-pair regardless of content).
A long dual-pass run hits the 500 cap and the oldest-first trim can evict
genuine early cross-level notes. Cross-pass note visibility itself only
surfaces after a wipe (each agent's pre-render sync masks its own live keys)
and is a legitimate feature for score=max-over-plays — not a rules problem
(our harness, our memory). At eval runs=1, so none of this fires on Kaggle;
it contaminates dev A/Bs run with passes ≥ 2. Dict mutation itself is
GIL-safe (per-op atomic; list trim during a concurrent `reversed()` walk
cannot raise). Repro: scratchpad `repro_cdef.py` §D.

### F6 — EVIDENCE-GAP: the claimed "full-stack co-install smoke" is not in the repo, and no deployment path installs TP10

`dry_run.py` installs tp/tc/te/tem/t6/t7 — no TP10 (nor TP9); no marker or
artifact of a TP10 smoke exists. The judge ran one (TP10 innermost +
`dry_run.py`): **PASS**, with live accounting lines in transcripts (ls20: 55,
vc33: 38, incl. `ended level=2` after the transition; sb26: 0 because no
batch executed there — fail-open behaving as designed). Separately:
`_duck38_flight/build_flight.py` embeds only graft_throughput/control/
explore/emission — TP10 (and deaths/economy/pipeline) are not wired into any
flight bundle. TP10 cannot reach an eval as things stand; the smoke and the
wiring both need to become real artifacts before any slot is spent on it.

### F7 — NIT: docstring errata

`stop_reason` is not dropped by `_summarize_step_sequence` (see stock-delta
section). All other cited line numbers check out.

### F8 — NIT: the WIN/GAME_OVER fallback in `accounting_line` is near-unreachable

It fires only when no executed payload carried a `state` (stock always
records one, :1668/aggregate :321). When ALL items are non-executed the stock
summary is None and no line is printed at all — so a wrong "state=WIN" print
requires a summary with flags but no state, which production does not
produce. Also: when the only drops happen on fully-non-executed payloads
(terminal refusals / single blocked no-op), the drop count is reported but
the recorded reason sits on the non-executed item and `summary["stop_reason"]`
(last EXECUTED item) may be empty — the line then omits the reason. Honest,
just occasionally reason-less.

### F9 — NIT: `_game_key` over-strip is theoretical

Only a game id containing a literal `_p<digit>_` after `artifact_stem`
sanitization could over-merge; real ids (`vc33-5430563c`, ...) cannot produce
one.

---

## Verdict: SHIP-WITH-FIXES

Fix before arming:
1. **F1**: strip trailing `Next:`/`Suggestion:` lines from the assistant text
   before it reaches the stock harvest (the wrapper already sits on that
   seam), so the intent lives ONLY in the one-shot echo; then strengthen
   test_05 to assert the intent text count is 0 on turn 2, not just the
   header.
2. **F2**: report pre-clamp committed count — e.g. graft_throughput stashes
   the pre-truncation length on the agent for TP10 to read, or TP10 drops the
   word "committed" when TP's cap truncated the batch. As shipped the line
   teaches false feedback whenever the cap fires.
3. **F6**: add TP10 to `dry_run.py`'s install block (and to the flight
   builder when armed) and commit the smoke — the claimed smoke does not
   exist in the repo (judge's own run passed, so this is bookkeeping plus
   real wiring, not a rebuild).

Fix before interpreting the A/B:
4. **F4**: default `TP10_NOTES_CAP` to ≤2000 per R11's own prescription (or
   run a capped arm), and stop appending `ECHO_HOWTO` on turns where it
   cannot matter (e.g. fold it into the notes header or emit it only every
   few turns) — ~1.1k tokens/turn against a 32k window measurably drops
   retained history from 7 to 3 turns.
5. **F3**: retire `(key, None)` entries once a levelled entry for the same
   key exists (one-line change in `render_notes` dedupe).
6. **F5**: for dev A/Bs with passes ≥ 2, either key the journal by full stem
   (env-gated) or run arms at passes=1; eval (runs=1) is unaffected.

What holds up: the stock-seam analysis is accurate (one doc erratum), the
delta vs stock is real (the A/B is not testing a no-op), fail-open survived
per-seam exception injection, both stacking orders work, all 9 suites and
both dry runs pass, and the accounting line is honest in the un-capped case
including in-batch known_noop drops (committed 5 / executed 3 /
stop_reason=known_noop verified). The design is worth flying once F1/F2 are
closed and the budget arm is controlled.

Judge artifacts: scratchpad `repro_a_next_leak.py`, `repro_f2.py`,
`repro_cdef.py`, `dry_run_with_tp10.py`; TP10 dry-run transcripts under
`scratchpad/tp_dry_run/230005/`.

---

# Re-verification (2026-08-31, second pass) — fixes for F1/F2/F3/F4

Everything below re-run by the judge from the updated files (graft 592 lines,
suite 20 tests); nothing taken from the builder's reports.

## Fix verification — all four fixes CONFIRMED

- **F1 (intent strip): FIXED.** `split_intent()` removes `Next:`/`Suggestion:`
  lines before the stock harvest. Judge's original repro re-run:
  `current_plan == 'hug the left wall'` (no glue), intent text appears
  exactly ONCE on turn 1 (echo header) and ZERO times on turn 2. The suite's
  new `test_05c` covers the scenario against the real seams.
- **F2 (pre-clamp committed): FIXED, both orders.** Judge's repro §C re-run in
  BOTH install orders with TP_BATCH_CAP=3, batch 10:
  `LAST TURN: committed 10 action(s), 3 executed, ... 7 committed action(s)
  were dropped before execution (7 truncated by the harness batch cap).`
  The capture scope is correct: `action_results`/`_summarize_step_sequence`
  are per-python-tool-call (:1938-1967 inside `_run_python_tool`), and the
  raw-count list resets at exactly that boundary — no cross-call bleed.
  `test_14` runs both orders in fresh subprocesses against the real seams.
- **F3 (level-None retirement): FIXED.** Original repro re-run: the refuted
  "WRONG early guess" is no longer served; the correction is. `test_16`
  covers it.
- **F4 (budget): FIXED as specified.** Default cap 2000; ECHO_HOWTO on
  prompt 1 / every 8th / first after wipe. Judge's §E measurement re-run:
  TP10 delta 2,565 chars (~641 tok) on a howto+accounting+intent turn
  (steady-state lower), retained user turns under the 32k trim now 7 → 4
  (was 7 → 3). Live cadence confirmed in the dry run: howto on 7/56, 6/46,
  79/627 prompts (~1 in 8).
- **F6 (wiring): CONFIRMED committed.** `dry_run.py` now installs t9+t10
  behind the hard assert; judge ran it: PASS (240 s), accounting lines live
  in transcripts (ls20 55/56 prompts, vc33 45/46; sb26 0 — no executed
  batches there, fail-open as designed).
- **F7 erratum**: docstring now correctly says stop_reason/level are kept by
  the stock summary. Round-1's combined fail-open injection re-run: now
  passes with all six helpers raising (the round-1 "False" was confirmed to
  be F1 pollution of the judge's own check, not a fail-open failure).
- Full re-run: TP10 suite 20/20; all 9 other suites green; dry_run PASS.

## New regression probes (all run, not reasoned)

- **(i) Strip aggressiveness — two NITs, no blocker.**
  (a) With multiple `Next:` lines only the LAST is echoed and the earlier
  ones are removed entirely (neither harvested nor echoed) — measured:
  `"Next: A\n...\nNext: B"` → A is destroyed. The howto teaches "one line",
  so exposure is self-inflicted; still a small silent data loss.
  (b) With `TP10_ECHO=0` the strip is off and the original F1 glue returns
  (`current_plan == 'go up Next: glue me'`) — acceptable because the howto
  is also off (the model is never taught the phrase), but an echo-off A/B
  arm carries the residual hazard if the model emits `Next:` spontaneously.
- **(ii) Normalize raising: CLEAN.** Stock `ValueError` (empty action)
  propagates with NOTHING appended (append sits after the inner call); the
  next call records correctly (`raw=[2]`, committed 2, no phantom
  cap_dropped). No stale capture.
- **(iii) Early-error tool call: CLEAN.** A syntax-error tool call resets the
  list to `[]` and never reaches `_summarize_step_sequence` (:1965-1967 gate
  on step_executed), so no phantom "committed 0" line; the prompt keeps
  describing the last EXECUTED call — same staleness semantics as stock's
  own executed-count header.
- **(iv) TP5 reasoning-channel bypass: CONFIRMED RESIDUAL (low severity).**
  Driven through the real `_chat_completion` wrapper with a faked HTTP
  response under TP5_ENABLE=1: content with no note + reasoning
  `"World model: ...\nNext: press the top button"` → emission's fallback
  writes `world_model = 'buttons cycle colors Next: press the top button'` —
  glued, served in the live block (1 line/turn), journaled, never echoed.
  Requires the model to put its ONLY labeled note in the reasoning channel
  AND a `Next:` line there — a narrow slice of turns; consequence is the old
  F1 behavior scoped to those turns. Right fix lives in graft_emission
  (route its fallback text through `split_intent`); not a TP10 blocker.
- **(v, judge-added) Cap-REFUSAL order asymmetry — NIT.** When a second
  `action()` call is refused outright (remaining=0 → ValueError): TP10-last
  (deployed order) does not count the refused call (`committed 3, 3
  executed` + the model got the explicit in-band error); TP10-first counts
  it (`committed 5, ... 2 truncated`). Both are defensible; the docstring's
  "recorded only when the chain returns" is exact only for the deployed
  TP10-last order. Cosmetic.

## FINAL VERDICT: **SHIP**

All four fixes verified by re-running the original reproductions; no new
blocking defect found. Remaining non-blockers, in priority order:
1. graft_emission's reasoning fallback should route through `split_intent`
   (residual iv) — a one-line change in TP5, best done next time TP5 is
   touched.
2. F5 (dev-only same-game-pass journal sharing) stands deferred; keep dev
   A/B arms at passes=1 or accept the contamination knowingly.
3. NITs (i-a), (i-b), (v) need no action beyond awareness.

Round-2 judge artifacts: scratchpad `reverify_probes.py`,
`probe_refusal.py`; re-run `repro_a_next_leak.py`, `repro_cdef.py` (both
orders); dry-run transcripts under `scratchpad/tp_dry_run/232659/`.
