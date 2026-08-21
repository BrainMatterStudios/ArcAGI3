# RESEARCH 2026-08-21 — bug-lever hunt (4-agent deep exploration)

Four parallel agents: harness code audit (June stock src), live-trace forensics
(13 smokes / 47 game-runs / 8,621 actions / ~3,480 requests / 6.65M tokens),
harness↔engine mismatch audit, serving-stack external sweep. Findings below are
CROSS-ADJUDICATED — where one agent's claim was refuted by another's
measurement, the measurement wins (law #1).

Corpus caveat: forensics ground = smoke runs (4h-class), not full 9h live runs;
frequencies scale with run length. 40/47 game-runs died `cancelled` at the
clock mid-level, 0 finished — wall reclaimed IS depth.

## TIER 1 — measured live cost, cheap single-variable fix

### 1. Yield-resume context wipe (the biggest single wall sink found to date)
- 2,234 turn-slices: 43.3% ended with NO env action ("Yielded control:
  turn_time_budget"); **51.5% of ALL wall (130,631s/253,830s) elapsed in
  slices that ended without an action**.
- Mechanism: on yield-resume the turn conversation is rebuilt with only ~2–4
  messages (history_messages 14→11→4→2 observed inside one turn) — every tool
  result + reasoning from the previous slice is discarded; the model re-grounds
  from zero. **71 byte-identical python snippets re-issued within single
  turns**; worst turns: 35 slices (packv22/sk48-dup step 12), 18 slices /
  3,584s / 118k tokens for ONE action (depthdiag/tn36 step 30).
- Priced: if half the yield-slice wall is restart redundancy ≈ 2.2h of the 9h
  envelope ≈ ~90 median turns of real play.
- Fix: carry turn conversation (or structured digest of the turn's tool
  results) across yield-resume; hard-cap slices/turn (~3) with forced
  best-available action.
- Falsifier: one smoke with carryover; duplicate-snippet count and max
  slices/turn should collapse.

### 2. Sampling + reasoning-effort cluster (three findings, one root)
- Harness pins temp 0.3 / top_p 0.9 forever at eval: `_pass_sampling`
  (tool_agent.py:153–169) bucket pass_index%3==0; runs=1 ⇒ always pass 0.
  `LOCAL_ANALYZER_TEMPERATURE` is dead code while ANALYZER_PASS_DIVERSITY=1.
- Official chat template defaults `reasoning_effort` to **xhigh** (verified by
  rendering our snapshot's template); harness sends only enable_thinking, so
  every scored call carries "Reasoning effort is set to xhigh" with
  max_tokens=None.
- Measured live symptom (forensics): **122 thinking-only dead completions**
  (finish_reason=stop, zero tool calls, 111 with zero content) = 2.62M
  reasoning chars ≈ 650–750k tokens ≈ **~6h of decode producing literally
  nothing**, with near-verbatim paragraph repetition loops — the exact
  low-temp degeneracy signature reported externally.
- External effect size on THIS model (zephel01 bench): temp 0.2 → 15.3%
  aborts/degenerate loops; official temp 1.0/top_p 0.95/top_k 20 → 2.1%
  aborts, +18.7pt benchmark. Corroborated: QwenLM#145, official card rec.
- Fix (zero code): env `ANALYZER_PASS_DIVERSITY=0`,
  `LOCAL_ANALYZER_TEMPERATURE=1.0` (top_p 0.95/top_k 20 already defaults);
  one line in openai_compat.py:68 → `"reasoning_effort": "medium"`.
  `--generation-config vllm` disables checkpoint defaults — keep explicit
  harness values.
- Retry hardening (same cluster): on stop-with-no-tool-call retry, force
  tool_choice=python + "your previous reasoning produced no action — act now".
- Falsifier: offline dev A/B, single variable each (sampling; effort). This is
  a BEHAVIOR BET (prompt/sampling levers nulled twice at 27B/3.6) — but here
  backed by measured live pathology + external measurement on 3.8.

### 3. Tool-API friction pack (~153 failed snippets, 4.4% of requests)
- `animation()` documented as returning dicts, actually returns STRINGS →
  30× `'str' object has no attribute 'get'` across ≥8 runs (doc/impl
  mismatch, not model error).
- 25× NameError from the "every call starts fresh" sandbox vs the model's
  persistence instinct; 23× ImportError (`sys`, `difflib` blocked); 78
  tool-result truncations (1024-token cap) each forcing a re-query.
- Cost ≈ 2.5–5h corpus-wide (each failure ≈ wasted decode + correction
  decode); concentrated in bad games (tn36: 19% of requests errored).
- Fix: make animation() return dicts (or fix the prompt line + example);
  whitelist difflib; persist tool namespace within a turn; raise result cap.
- Falsifier: one smoke post-fix; `.get` AttributeErrors → exactly 0.

## TIER 2 — structural, priced, needs rig validation

### 4. hard_noop_guard doesn't guard
8.2% of duck-v12 env actions (265/3,230) changed ZERO pixels; 55 whole turns
(225k tokens, 7,390s) emitted only no-effect actions; repeated same-coordinate
no-op clicks ×6. Guard keys on something weaker than frame-identity. Fix:
frame-unchanged ⇒ refuse same action/coord + loud "NO-OP" in tool result.
Falsifier: replay the 265 recorded cases through the new predicate.

### 5. Analyzer side-channel blocks on decaying triple-digit timeouts
42 unique analyzer read-timeout events (289s → 2s decaying deadline);
~2,800s of blocked waiting measured in just 3 turns. Fix: fixed ≤10s timeout
or async. Falsifier: grep next run for `read timeout=[0-9]{2,}`.

### 6. Truthful-telemetry prompt patch (mismatch pack, one combined arm)
- RESET is hidden from the model (solver.py:117–118 strips it; zero prompt
  mentions) while the engine keeps it always-legal — soft-locked levels have
  no exit today. `action(['RESET'])` already executes with zero code change.
- Time keys (`run_elapsed_seconds`/`time_remaining_seconds`) sent every call
  but never explained; no endgame mode possible.
- Total level count known (initial.win_levels) but never surfaced — model
  can't value depth/weights; public `tags` never used at play.
- Payload traps: `"score"` field = levels_completed; `reward` =
  completion-fraction delta; neither defined anywhere.
- All one-line prompt/payload fixes; ship as ONE arm, offline
  CompetitionArcadeServer A/B. Caution: prompt levers nulled twice at 27B —
  pre-register, read per-level actions not score.

### 7. Stale-image history accumulation
Every retained user turn keeps its board PNG captioned "Current grid image:"
(~20–30 stale boards/request, each with textual license to be read as
current). Misgrounding channel + real vision tokens. INTERACTION: today the
len/3 estimator overcharge accidentally trims old images early; once the
estimator patch ships, MORE stale images survive — ship this WITH or BEFORE
the estimator patch. Fix: strip image parts from prior turns (~10 lines).

### 8. Runtime-state O(n²) serialization + 30s sandbox kill
Full unbounded history re-written to disk EVERY action (65/129MB at
n=1000/2000) and re-shipped into the sandbox per action() call (18/37MB
pipe); sandbox hard-capped 30s, on timeout ALL printed output discarded;
crash stderr replaced by a constant string in BOTH code branches
(_sanitize_host_error_text returns the same constant either way).
Smoke-measured timeout frequency LOW (2 events) — but smokes are 4h-class
and n rarely exceeds ~500; live 9h runs go deeper. Fix: cap serialized
history to last ~50 entries, drop indent=2, return real stderr, flush
partial stdout. Falsifier: replay recorded runtime_state at n∈{100,1000,2000}
with a 40-action batch; grep live logs for timeout strings.

## TIER 3 — pre-register carefully (measured-negative neighbors)

- **game_over world-model wipe** (tool_agent.py:1113–1126 clears world/goal/
  action model on mid-level death, then auto-RESETs the same level while the
  action meter runs). One-token fix BUT duck-mem P1–P4 pooled −0.29; check
  overlap with queued duck-p3 before building. dc22-L2 is the regression
  test: 5/5 xpl runs entered, 55–215 actions each vs base 102, 0 completions,
  ~25% of xpl-family wall for 0 points.
- **Scoring-contract prompt** (model is score-blind: no (b/a)², no level
  weights, no action accumulation, no "actions on uncompleted levels are
  score-free"). At 3.6 priced ~0 (unlock-limited); at 3.8 the completed-level
  surface is materially larger — re-price, offline A/B only.
- **graft_bank rider on xd**: xd bundles explorer+digest but NOT graft_bank —
  a sloppy LLM full win in xd doesn't get replay-banked; 3.8 makes full wins
  real. Cheap coverage add for Branch A/B tomorrow.
- **Multi-tool-call turn erasure** (tool_agent.py:1965–1973 discards the whole
  turn when a non-final tool call acts): frequency NOT yet measured (forensics
  counted markup-recovery = 0, which is a different signature). Grep first.

## CLOSED / ADJUDICATED by cross-check

- vLLM 0.19.0 think-tool-call drop (#39056): real upstream bug, but measured
  **0 occurrences in 3,482 live requests** (tool_call_markup_in_text: no on
  all) → CLOSED for our stack, do not spend on it.
- fp8 KV cache: keep OFF — checkpoint ships no calibrated KV scales (HF disc
  #10 unanswered), corruption reports exist (#42179). Watch-list item RESOLVED.
- Prefix caching: NOT dead on our workload — measured **41.3% hit rate** in
  smoke serve logs (refutes vLLM #43587 applying to us as written), though
  modest vs ideal (hybrid 528-token granularity #40696 plausible). **MTP
  LANDMINE CONFIRMED RELEVANT**: cache hits occur, and MTP+prefix-cache on
  GDN corrupts generations (fix PR #47861 is post-0.19) — **MTP ships only
  with --no-enable-prefix-caching**.
- Blank-think history bug: not a bug on v0.19.0 (chat_utils maps
  reasoning→reasoning_content; verified at tag).
- 2048-token tokenizer truncation: unsloth-NVFP4 packaging only; our snapshot
  verified `truncation: null`.
- Coordinate mapping row/col↔x/y↔ascii↔PNG: audited end-to-end, CONSISTENT —
  no off-by-one exists. (Out-of-range clicks ARE silently clamped to borders
  though — small fix bundled in Tier 2 #6 territory.)
- Negative checks from forensics: zero malformed tool calls, zero
  finish_reason=length, zero post-win actions, batches correctly truncated at
  level transitions.

## packv22 smoke context for tomorrow's reading (08-22)

The pack-recipe smoke on OUR rig: 0 levels in all 4 game-runs; sk48 390
actions never clearing L1 (base 61) ending in blind identical 26–28-action
batches; sk48-dup frozen 35 slices in one turn with a 20-chain of
thinking-only completions. The pack recipe's failure mode on our rig is
macro-batch flailing + thinking stalls, NOT serving defects. Hold this when
reading the live band: a ~1.5 draw would be consistent with these pathologies
persisting live; it does not by itself indict serving.

## Suggested order of fire (post-08-22 reading, either branch)

1. FREE greps (today): multi-tool-call frequency; live-log timeout strings.
2. Offline A/B #1: sampling (temp 1.0/top_p 0.95/top_k 20, diversity off).
3. Offline A/B #2: reasoning_effort medium.
4. Build+smoke: yield-resume carryover (Tier-1 #1) — biggest priced lever.
5. Tool-API friction pack (Tier-1 #3) — near-zero risk, ship as rider.
6. Truthful-telemetry arm (Tier-2 #6) on the offline rig.
All obey law #2 (safe default armed first) and law #3 (envelope arithmetic
for anything touching duration).
