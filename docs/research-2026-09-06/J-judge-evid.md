# Judge — 3-wall instrument build (EVID / HYPO / UP8), 2026-09-06 — SHIP-WITH-FIXES

Coverage of the three walls' root causes ≈ 1 direct (dc22: the per-action TRACE contradicts the
"ice-slide" misread ten times over), 1 partial/one turn late (lf52: the APPEARED marker at (15,32)
contradicts the (16,25) click; the model's arithmetic slip was in its own table), 0 (cd82: the wrong
belief came from reading the last visible L1 frame as the completion state — the engine never returns
the completed board; the flag must say so). Coordinate convention VERIFIED on a real click.
Prompt cost ≈ one fewer kept history turn (~10%); HYPO block rides every persisted user message (10
identical copies) — strip from older messages. ±15% "regime in kind" cannot gate a prompt aid; gate on
identity + truncation share only.
Defects: MOVED matcher runs before in-place overlap → stationary tiles reported as swapped movers at the
interaction site (dc22, lf52); EVID_TRACE_MAX=12 cut the informative blocked action; ~30% noise lines.
Instrument: 6 concurrent runs on a 3.21x server → e2e ~37 s → ~40 calls in 1500 s (not 55); cd82 first
reaches L2 at turn 14–34 in the base; base pass rate on the three walls 0/14 attempts. Fix: concurrency 3
(or --max-calls 60), attempts count only if L2 reached with ≥30 calls left (else VOID, one redraw), base
arm first at the same geometry ($2), void rule for preemptions/errors/truncation >1%, and pre-registered
MECHANISM reads (uptake = share of wall turns whose thinking quotes the block ≥30%; dc22 contradicted-
movement-model turns, base 4; lf52 own-table-vs-sent-click disagreements, base 4).
UP8: expect ≈ +192 prompt tokens on the first call; +0 = processor downscaled (no-op); ≫ = geometry differs.
Corrected plan: fixes (matcher order, trace cap + last action, noise/flag wording, HYPO strip, --max-calls
or concurrency 3, void rule) → keith base → keith_evid → keith_hypo → keith_up8, each cd82,dc22,lf52 ×2
draws, ≈$3/arm. Decision: ≥3/6 pass AND base ≤1/6 → 25-game wave (≥48 lv AND ≥3 of 8 consistent walls);
2/6 → one more round; ≤1/6 with uptake ≥30% and mechanism counts → ≤1 → EVID = RIDER candidate; low
uptake → dead. Realistic prior for a step: low; the instrument mostly decides rider vs dead.
