# J9 — Adversarial judge review of TP9 (graft_pipeline.py)

Judge: fresh-context adversarial review, 2026-08-31 (late). Under review:
`submission/_throughput_v1/graft_pipeline.py` + `test_graft_pipeline.py`.
Everything below was verified by reading the anim bundle
(`submission/_inspect_replay/assets_build/ARC3-Inference`) and by running
probes; probe scripts live in the session scratchpad
(`probe_stack.py`, `probe_retry.py`, `probe_dry_run_t9.py`).

**Verdict: SHIP-WITH-FIXES** (fixes listed at the end; none are crash-class).

---

## Checklist verdicts

### 1. Seam truth — CLAIM CONFIRMED (with one precision nit)

The builder's central claim for (a) is correct, verified against the bundle:

- `tool_agent.py:2365` sets `preserve_history = False` on
  `requests.RequestException`; the `finally` at `:2403` reverts
  `self._history_messages = previous_history_messages` (snapshot taken
  `:2138`); the return at `:2380` is
  `AnalyzerTurnResult(step_executed=False, retryable_failure=True,
  reasoning=captured_reasoning)`. `captured_reasoning` is assigned from each
  reply's reasoning (`:2273-2275`), so on a timed-out turn the reasoning
  survives only in `result.reasoning`.
- The solver loop (`solver.py:352-363`) handles `retryable_failure` (1 s
  backoff + retry), `yielded_control`, and idle by `continue` — and
  `grep -n '\.reasoning' solver.py` returns **zero** hits: the field is
  dropped on the floor, exactly as claimed.
- Clean-yield case: after a no-tool-call reply the reasoning-only assistant
  message is appended (`:2287`) and kept (`preserve_history` stays True), so
  the "kept but trimmable, and regenerates the same doomed turn" caveat in
  the graft docstring is also accurate.
- NIT: "the ONLY surviving copy" — the analyzer transcript log on disk also
  holds the THINKING text (`append_transcript("THINKING", ...)`), but nothing
  that feeds back into model context reads it, so the operative claim stands.
  Minor line-cite drift: the context-length seam is `:2221` (builder said
  :2220), the solver block is `:352-363` (builder said 352-364).

Retry seam (c) is also correctly aimed: context-length errors are raised as
bare `requests.RequestException` (from `raise_for_status`, `:1546-1553`) —
not Timeout/ConnectionError — so they pass the graft untouched and reach the
in-analyze recovery at `:2221` (`_is_context_length_error`). test_10 covers
this; confirmed.

### 2. Wrapper safety / stacking — PASS (verified by execution)

- House law honored: stock callables live in module-level `_STOCK`
  (graft_pipeline.py:86, 262-264). (Aside, not under review: graft_deaths and
  graft_economy hold their stock prompt builders only in closures.)
- Signatures checked against the bundle: `analyze` (`tool_agent.py:2090`),
  `_build_user_prompt` (`:1391`), `_chat_completion` (`:1518`) — all wrapper
  signatures bind the real call sites (`solver.py:333`, `tool_agent.py:2111`,
  `:2214`).
- `graft_memoryspine.py` (TP10) **appeared mid-review** (22:44 tonight) and
  wraps `_build_user_prompt` too, via its own module-level `_STOCK` — chains
  cleanly.
- Executed proof, both install orders:
  - `probe_stack.py` — t6, t7, t8, t10 installed, **then t9 last**: RESUME +
    ACTION ECONOMY + stock prompt coexist in one prompt; consumed-once holds.
    PASS.
  - `probe_dry_run_t9.py` — **t9 (+t8, t10) installed first**, then the
    committed `dry_run.py` main() stacks tp/tc/te/tem/t6/t7 on top; 3 real
    games on the real engine, all 13 dry-run checks PASS, no crash.

### 3. State leaks / concurrency — PASS

- The solver runs games on a `ThreadPoolExecutor` sized to `concurrency`
  (`solver.py:17, 1042-1043`) behind an asyncio semaphore (`:1056`) —
  blocking calls (including the graft's `time.sleep` backoff, matching the
  solver's own `time.sleep` at `:356`) block only that game's worker thread.
- Each game gets its **own** ToolAgent (`solver.py:1345-1347`), and TP9 state
  rides the agent instance (`agent._tp9`), so per-game isolation is
  structural; the runtime-dir keyed reset additionally covers a shared
  analyzer_factory. Probe: 2 agents × 200 interleaved turns in 2 threads —
  zero cross-game contamination, zero exceptions.
- `_STATE` counters are shared ints incremented without a lock across up to
  28 threads — can undercount. Telemetry-only. NIT.

### 4. Fail-open — PASS (verified by injection)

- `hashlib.sha1` patched to raise → wrapped `analyze` still returns the stock
  result. PASS.
- `RESUME_BLOCK.format` patched to raise → `_build_user_prompt` returns the
  stock text (other grafts' blocks intact). PASS.
- The retry wrapper's flag reads are try/except'd; disabled paths
  (`TP9_ENABLE=0`, per-behaviour flags) verified by tests 11-12 and rerun
  here. PASS.

### 5. Semantic traps

**(i) Stale RESUME — LOW RISK.** Between an idle turn and the next
`analyze`, no actions execute (the solver only retries), so the tail is never
board-stale at injection time. The injected block does persist in kept
history after a clean turn (until trimming), duplicating up to 1500 chars of
reasoning already in history — a token cost (~375 tok/idle turn at a 48%
idle rate against a 32k window), not a correctness issue. One real loss mode
(DESIGN-RISK, low): the tail is consumed at prompt build; if that very
request then dies with no new reasoning (timeout before any reply), history
is reverted and the tail is gone for good — the resume nudge silently
evaporates in exactly the flaky-server window it was built for.

**(ii) LIVELOCK false positives — DESIGN-RISK, HIGH. This is the main
finding.** Two demonstrated failure scenarios:

1. **Server outage arms it.** Three consecutive `retryable_failure` turns
   with **no model output at all** (reasoning "") hash identically → LOOP
   BREAKER armed. The solver retries failures at 1 s intervals
   (`solver.py:352-357`), so a **3-second vLLM hiccup** is enough; the next
   healthy turn is told *"You have produced identical output 3 times in a
   row"* — factually false to the model — and ordered to act NOW without
   investigation. Demonstrated in `probe_stack.py` (pure-failure streak and
   mixed yield/failure streak both arm). The 13 measured read-timeouts
   cluster at run end, so this will fire in production.
2. **No de-arm ⇒ prompt flooding.** Once `idle_streak >= K`, every further
   identical-hash idle turn re-arms `perturb_pending` — there is no throttle
   or per-episode cap. In the integrated dry run TP9 injected the LOOP
   BREAKER into **594 of ~700 prompts** (`perturbations_injected: 594`),
   each block persisting in kept history. The dry run's stall regimes are
   deliberately pathological, but the mechanism transfers: any game stuck in
   an idle phase with empty extracted reasoning gets a LOOP BREAKER stapled
   to essentially every prompt.

   On "are tool-only quiet turns idle": yes — `step_executed` is True only
   when an `action()` actually executed (`tool_agent.py:1965`), so an honest
   investigation turn (tool calls, no action) that yields with empty
   reasoning text joins the streak. With thinking enabled (default) reasoning
   is rarely empty on real turns, which is why the builder's r11l signature
   ("no assistant output", transcript lengths 44934/44925/44922 — R9 §2.2)
   is legitimately caught — but the failure-turn and flooding paths above
   are not r11l, and they share the same empty hash.

   False-negative flank (DESIGN-RISK, low): served decode is temperature
   0.6 / top_p 0.95 (`tool_agent.py` defaults) — a genuine livelock that
   still emits sampled reasoning text will rarely be byte-identical, so the
   detector catches only the empty/deterministic-replay subspecies. Fine as
   scoped, but the name overpromises.

**(iii) RETRY vs wall-clock — DESIGN-RISK, CONFIRMED by measurement.** The
retry has no `should_stop`/wall awareness and reuses the turn's
`request_timeout_seconds` (computed once per turn as min(analyzer timeout,
wall remaining, soft remaining) — `solver.py:267-284`). Real-socket probe
(`probe_retry.py`, black-hole server): request timeout 2.0 s → stock burns
~2 s; graft burned **5.0 s** (2 + 1 backoff + 2), second connection opened
with the wall possibly already exhausted. Stock behavior would instead
return to the solver, which checks `should_stop()` before its retry
(`:354-355`). Overshoot per timeout event is bounded by
request_timeout + backoff, but it is spent at exactly the moments (run-end
contention) the R9 evidence says timeouts happen. A ConnectionError retry is
near-free; a ReadTimeout retry always doubles the spend.

**(iv) graft_durable interference — NONE (verified).** TP8 rewrites the
sandbox timeout *error text* only; `step_executed` is computed from
`action_results` regardless of error (`tool_agent.py:1965`), so a tool
timeout whose actions DID execute yields `step_executed=True`, TP9 clears
its state, and the RESUME claim "nothing was applied" is never emitted for a
turn whose actions actually landed. The two grafts' messages cannot
contradict each other.

### 6. Test honesty — MOSTLY HONEST, one gap

- The suite installs on the **real** bundle classes and calls the **real**
  wrapped `_build_user_prompt` (real stock prompt text). test_01 would fail
  on a real install failure (a `SKIP (import failed…)` status is not in its
  accepted set). test_10 genuinely protects the context-length seam.
- EVIDENCE-GAP: every turn simulation mocks `t9._STOCK["analyze"]` — no test
  drives the real `analyze()` against a timing-out HTTP server to prove the
  load-bearing premise of (a): that stock analyze actually populates
  `result.reasoning` on a RequestException. Tests 02-07/11-12 would still
  pass if the bundle's exception path stopped returning reasoning. (I
  verified the premise by code reading — `:2273-2275`, `:2380` — and the
  retry path end-to-end with a real socket, so the premise holds *today*.)
- EVIDENCE-GAP (was): the committed `dry_run.py` does **not** install TP9
  (installs tp/tc/te/tem/t6/t7 only), so the claimed integrated proof never
  included this graft. Filled by this review: `probe_dry_run_t9.py` runs the
  full stack including TP9/TP8/TP10 — PASS, and it is what surfaced the
  594-injection flood.

---

## Findings summary

| # | Class | Finding |
|---|-------|---------|
| F1 | CONFIRMED (claim) | Seam analysis (a) correct: reasoning survives only in `result.reasoning` (tool_agent.py:2365/:2380/:2403); solver never reads it (solver.py:352-363). |
| F2 | CONFIRMED (claim) | Retry (c) correctly scoped; context-length errors reach `:2221` untouched; real ReadTimeout retried end-to-end (probe). |
| F3 | DESIGN-RISK (high) | LOOP BREAKER false-positives: a 3 s server outage arms it with a false "identical output" message; no de-arm/throttle → 594/700 prompts injected in the integrated dry run. |
| F4 | DESIGN-RISK (med) | Timeout retry ignores wall-clock/should_stop: measured 5.0 s burn for a 2.0 s budget; doubles run-end strand time per event. |
| F5 | DESIGN-RISK (low) | Detector misses non-empty-text livelocks at temp 0.6 (hash-identity too strict); catches only the r11l empty/deterministic subspecies. |
| F6 | DESIGN-RISK (low) | Consumed RESUME tail is lost forever if the injected request itself dies with no new reasoning. |
| F7 | EVIDENCE-GAP | dry_run.py omits TP9; integrated proof did not cover the graft as committed (filled by this review — stack PASSES). |
| F8 | EVIDENCE-GAP | No test exercises real analyze() → result.reasoning capture; suite tests wrapper logic against mocked _STOCK. |
| F9 | NIT | `_STATE` counter races (telemetry); `resume_reason` never cleared; line-cite drift (:2220→:2221, 352-364→352-363); RESUME/LOOP blocks persist in kept history (token cost at 48% idle). |

## Verdict: SHIP-WITH-FIXES

Required before arming on a slot:

1. **F3a** — exclude `retryable_failure` turns from the livelock streak (count
   only yielded/no-action turns), so a server outage can never arm the
   breaker with a false accusation.
2. **F3b** — throttle the breaker: inject at streak K, then at most every K
   further turns (or cap injections per game), so stall regimes don't get a
   LOOP BREAKER stapled to ~every prompt.
3. **F4** — don't blind-retry a ReadTimeout: keep the (cheap, safe)
   ConnectionError retry as-is; for Timeout either skip when the attempt
   consumed a wall-sized budget or shrink the retry's
   `request_timeout_seconds`. At minimum, document the ≤ timeout+backoff
   overshoot against the wall-clock law.
4. **F7** — add `graft_pipeline` (and `graft_durable`/`graft_memoryspine`) to
   `dry_run.py`'s install block so the committed integration proof covers
   what actually flies.

Recommended (not blocking): one test that drives real `analyze()` against a
mock HTTP server that times out, pinning the (a) premise (F8); clear
`resume_reason` with the tail (F9).

---

# Re-verification (same judge, after the builder's F3/F4/F7 fixes)

Scope: targeted re-check of the four fixes plus regression hunting on them.
Everything re-run by me (suites, probes, committed dry run); probe scripts in
the session scratchpad (`probe_fixes.py`, re-run `probe_retry.py`). All 9
test suites pass (test_graft_pipeline.py grew 13 → 19 tests).

## Fix verdicts

**F3a (outage exclusion) — VERIFIED.** `_note_turn_result` returns before any
hash work when `result.retryable_failure` is set (graft_pipeline.py:298-301).
Probes: 6 consecutive server-failure turns → streak 0, nothing armed, no
LOOP BREAKER in the next prompt. Regression (i) — alternating
real-empty-yield / failed / real-empty-yield / failed / real-empty-yield —
the streak survives the interleaved failures and arms exactly on the 3rd
genuine empty turn (failures neither build nor reset). No suppression.

**F3b (once-per-streak + cooldown) — VERIFIED.** Injection resets
streak+hash and starts a 5-turn cooldown (:381-385, :302-307). My
independent 100-turn persistent-livelock sim: **13 injections** (1 per
K+cooldown = 8 turns), matching the builder's number. Committed dry run:
`perturbations_injected: 67` over ~636 prompts (**10.5%**, was 594/~700 =
85% pre-fix; builder reported 78/724 — run-to-run stochastic, same order).
Regression (ii) — cooldown keying: after an injection in game A
(cooldown=5), switching to game B gets a fresh PipelineState (cooldown 0,
streak restarts at 1); no cross-game leak. Failed turns do not consume
cooldown (checked at :298 before :302) — cooldown counts only real turns,
as documented.

**F4 (wall-clock guard) — VERIFIED by measurement.** Black-hole-socket
re-run, request budget 2.0 s: **1 connection, 2.0 s burn** (was 2
connections / 5.0 s), and the original `requests.ReadTimeout` propagated
with its full traceback intact — not swallowed, not wrapped (regression
(iv)). ConnectionError probes: with a 30 s budget the instant failure IS
retried (2 attempts, retry given the shrunken leftover ≈29.995 s, counter
bumped) then propagates; with a 3 s budget the retry is skipped and the
original ConnectionError propagates after 1 attempt.
*Consequence worth stating plainly (by design, not a bug):* since the
budget IS the per-request timeout, a ReadTimeout always consumes ~the whole
budget, so **the Timeout retry effectively never fires when
`request_timeout_seconds` is supplied** (leftover ≈ −backoff < 5 s floor).
In solver contexts TP9_RETRY is now a ConnectionError-blip retry; the R9
"13 read-timeout" events fall back to the solver's own should_stop-guarded
retry loop — which is exactly what my F4 asked for.

**F7 (dry_run coverage) — VERIFIED.** `dry_run.py:166-170` installs and
asserts t8/t9/t10 alongside the rest; I ran the committed script once:
all 13 checks PASS, no crash, tp9 counters reported.

**F6 (optional parking) — WORKS, with one spec/impl divergence.**
Executed-step correctly discards the parked tail (no zombie resume). While
requests keep dying with no new reasoning, the parked tail re-injects every
rebuilt prompt (measured 10/10) — unbounded in count but benign: each such
turn's history is reverted (`tool_agent.py:2403`), so nothing accumulates
and the model sees the nudge once when the server recovers.
**Divergence:** the docstring scopes restoration to "the very request it
was injected into *dies*", but the implementation (:292-296) restores on ANY
idle empty-reasoning turn — including a **clean yield** where the model saw
the RESUME block and produced nothing (demonstrated: re-injected after a
clean empty yield). In an r11l-style empty-output livelock those turns
preserve history, so duplicate RESUME blocks (≤1500 chars each) can
accumulate until trimming, each claiming "interrupted last turn"
(increasingly false). DESIGN-RISK (low): fires only when the model emits
literally nothing, where re-nudging is arguably desirable; one-line narrow
fix if wanted: restore only when `result.retryable_failure`.

## Final verdict: SHIP

All three required fixes hold under adversarial probing; no blocking
regressions found. Two non-blocking notes for the record: (1) the F6
parking restores on clean empty yields too (broader than its docstring —
either amend the docstring or gate the restore on `retryable_failure`);
(2) TP9_RETRY should be understood as a ConnectionError retry in practice —
if Timeout-retry value is ever wanted back, it needs a budget source other
than the per-request timeout itself.
