# yield_carryover — envelope arithmetic (law #3)

Neither mechanism can extend run duration; both can only redistribute or
reclaim wall inside the existing per-game clock.

## Digest (YIELD_CARRYOVER) — bounded prompt-token cost, zero wall extension

- The digest is injected into the resumed slice's initial user prompt only.
  It adds **prompt** tokens, never decode obligations, and never touches
  `max_runtime_s_per_game`, `_yield_seconds`, timeouts, or the retry cadence.
- Hard bounds (graft constants): per tool call 200 chars code + 400 chars
  result, ≤10 entries, one 500-char reasoning tail, whole block capped at
  **6,000 chars ≈ 1,500 tokens** (len/4; the harness's own estimator at
  len/3 prices it ≈ 2,000, so the context trimmer sees it conservatively).
- Typical slice (3-6 tool calls): ~2,300-4,100 chars ≈ **600-1,100 tokens**
  per resumed slice. Only slices that ARE resumes pay it (43.3% of slices
  ended yielded in the smoke corpus; the digest rides only on their
  successors), and it competes for the same 32,768-token context budget —
  the stock trimmer (tool_agent.py:1672-1690) drops the OLDEST history first,
  so the block displaces stale turns, not the fresh frame.
- **Exactly one digest is alive at a time**: before each slice the graft
  scrubs prior digest/final blocks out of carried user messages, so the cost
  never compounds across resumes (verified by the injected-exactly-once
  test: slice 3's request contains one digest block, not two).
- Priced against the measured redundancy it targets: 71 byte-identical
  re-issued snippets and ~2.2h of restart redundancy per 9h envelope. One
  avoided re-issued probe (decode + tool round-trip, tens of seconds) pays
  for many digests' worth of prompt tokens (prompt prefill at these sizes is
  sub-second).

## Slice cap (YIELD_SLICE_CAP) — strictly turn-shortening

- The cap adds one bounded instruction line (≈340 chars ≈ 90 tokens) and
  never adds slices, sleeps, or retries. Its only behavioral effect is to
  push the model to emit `action(...)` earlier in a turn that has already
  burned ≥3 slices — turns can only get shorter, never longer.
- It fabricates no actions: if the model still refuses to act, behavior
  degrades to exactly stock (the solver's own yield-resume loop).

## Flags

- `YIELD_CARRYOVER=0` — disables the digest (byte-identical prompts).
- `YIELD_SLICE_CAP=0` — disables the cap; any positive integer overrides
  the default cap of 3.
- Both off ⇒ install() SKIPs (no patch applied at all).
