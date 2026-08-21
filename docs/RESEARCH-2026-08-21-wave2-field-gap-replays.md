# RESEARCH 2026-08-21 wave 2 — field refresh, gap localization, replays, tier re-price

Second 4-agent exploration wave (after the bug-lever hunt,
`docs/RESEARCH-2026-08-21-bug-lever-hunt.md`). Cross-adjudicated; corrections
first because several change standing beliefs.

## CORRECTIONS to standing record (verified by settings-diff / artifacts)

1. **Our live arms do NOT run temp 0.3.** The `_pass_sampling`/
   `ANALYZER_PASS_DIVERSITY` code exists only in `submission/_adopt/taaf-src`
   (reference copy) and retired `_prim4`. June stock — and therefore duck-38
   v2, v12-lane arms, pack-v22, xd — defaults **temp 0.6 / top_p 0.95 /
   top_k 20** (tool_agent.py env defaults; neither shipping notebook overrides).
   The bug-hunt doc's Tier-1 #2 sampling claim is amended accordingly: the
   remaining sampling question is 0.6 vs official thinking-rec 1.0 (modest),
   NOT 0.3 vs 1.0. The public floor cohort runs the same 0.6 — sampling does
   not explain our −0.7 vs the floor. reasoning_effort=xhigh (template
   default) DOES still apply to every call — ours and the field's — and stays
   a candidate edge-over-floor lever.
2. **The triage-at-55 evidence base is mislabeled.** The "35 hidden-run
   observations" are off-Kaggle Modal sims of 7 PUBLIC games at 60-min boxes
   (`triage_study.py` reads qwen38-factorial screens; k000–k006 = clone
   aliases of ar25/cd82/sc25/vc33/ft09/dc22/tr87 per pc_driver.py:622). The
   3% FN knee must be re-derived on 132-min competition-geometry runs before
   the 55-min threshold is trusted live.
3. **2.5291 is the right live-comparable offline stat** (mean over 28
   single-session rows; per-game-max mean is 2.7215; live n_passes=1 makes
   the row mean the correct comparator). Premise was right by accident.
4. **Hidden rerun plays ~110 games** (both halves; LB = public-half mean).
   Envelope 110/28×7920s = 8.8h + model load vs 9h wall — zero structural
   slack (explains the 08-20 wall death class).
5. **OPINE code IS public** (github.com/david-courtis/opine-world; corrects
   07-14 "no code released"). No LICENSE — reimplement only.
6. **08-15 sweep's Franzen per-technique numbers (+5.1pp PoE etc.) do not
   appear in his sources** — superseded by this wave's audit.
7. **08-20 blank-score sub 55634118 = the known runtime-wall death** (CLI
   shows COMPLETE + blank publicScore; errorDescription confirms). No mystery.

## 1. Field state (LB snapshot 2026-08-21T20:24Z, full CSV in wave-2 scratch)

- cstl **3.57** (08-20 jump +0.87 ≈ their 3.8 rebase; team = Stepanov
  [tehnar, CF IM 2348] + Gadaev; elite private harness, nothing published,
  nothing adoptable). Tufa Labs 2.97 (their own jump is private; public
  duck-harness repo dead since Jul 1). Lord Han Solo 2.76 ≈ max-draws profile
  (38 subs, entered at the 3.8 release, no footprint). Franzen 2.58 unchanged.
- Density: 16 teams ≥2.39, 112 ≥1.86, 219 ≥1.60. **We are 1.74, rank ~155.**
- **The public free floor ≈ 2.1–2.3 single-draw**: the whole 2.4+ low-effort
  cohort runs one assembly — June duck + FOYSAL Qwen3.8 repack + driessmit1
  vLLM wheelhouse (+ optional thtennant grafts; his fork dataset has 1,256
  downloads). Zero-effort accounts land 2.39–2.47 in 3–5 subs. The 2.5–2.76
  tier is consistent with draw-count order statistics alone.
- Tonight's pack-v22 = the parity test of exactly this floor. We sit ~0.7
  UNDER it — after this wave, the surviving explanations are (a) offline-draw
  optimism + set makeup, (b) live throughput at conc 28 (unmeasured), (c)
  something else in our runtime; NOT sampling (correction #1).
- Adoptable field artifacts (pulled to wave-2 scratch `srcpull/`, `kpull/`):
  keithtyser serving bundle (max-model-len **262144**, kv fp8, zero solver
  changes, hash-verified) — NOTE our fp8-KV caution stands (no calibrated
  scales) and 262k alone is inert while LOCAL_ANALYZER_CONTEXT_WINDOW=32768
  caps harness-side, so the real lever is the context-budget raise, which is
  envelope-priced work; ataraxian prompt deltas (game_over anti-paralysis +
  tried-actions checklist — diff hunks saved); saltb0x publishes a vLLM
  **0.27.1** wheelhouse (rank-4 team) → a newer-vLLM lane is field-proven,
  relevant because the MTP/GDN/prefix-cache fixes are post-0.19.
- Forum: no host/rule changes; PCIe-multitenancy noise hypothesis (unconfirmed)
  fits our variance band; thread 732854 (community per-game public-25 scores)
  comments unreadable by proxy — **5-min manual browser read recommended**.
- Models since 08-15: nothing ≤96GB credibly beats Qwen3.8-27B. Watch:
  GLM-5.3 weights promised ~08-28. FOYSAL repack retains `mtp.*` tensor keys
  (keith's setup asserts them) — MTP head present in the Kaggle-mounted model.

## 2. Live-vs-offline gap decomposition (~1.0)

- Ratio live/offline stable across models: 0.65 (3.6) and 0.59 (3.8) with
  disjoint jackpot games ⇒ the discount is a property of live conditions/set,
  not any game.
- Split: offline-draw optimism −0.2..−0.5 (true offline mean plausibly
  2.1–2.4; geometry repeat spread 0.271); set composition −0.3..−0.6 but
  CAPPED (public LB proves the hidden half supports ≥2.4 means); **live
  RTX-Pro-6000 throughput at concurrency 28 never measured** (Modal H100
  baseline 445 gen-tok/min/session; Kaggle smokes ran conc 3–5) — the one
  open branch with the right multiplicative shape.
- Live per-game observables: NONE exist (rerun logs closed; only score,
  parquet `totalBytes` (±60B; zero-runs byte-identical 3411B), and
  runtime-death events). Every circulating live zero-rate number is inference.
- Offline per-game structure: 3.8 jackpot games sc25/vc33/cd82/ar25 carry 59%
  of the mass; keyboard-only games near-dead (0.48 mean vs 3.59 click-only);
  7 games at 0.00 in both full waves (dc22/ft09/g50t/m0r0/sk48/tr87/wa30) —
  though xd/explorer smokes unlock several (ft09 15.12!).

**New levers from this branch:**
- **Concurrency-28 throughput probe** (zero slots, quota only): 28 synthetic
  decode streams against in-kernel vLLM; ≥~400 tok/min/session kills the
  throughput hypothesis; far below promotes MTP to the gap-closing lever.
- **Parquet-size telemetry side channel** (⚠ AHMED'S CALL REQUIRED before any
  build): re-write submission.parquet with byte-identical data + a metadata
  blob whose LENGTH encodes zero-count/levels; read back via REST totalBytes.
  One draw yields the first live zero-rate. It is deliberate submission-file
  shaping — needs rules/honesty-gate review first. Not built, not tested.

## 3. Human-replay corpus (340 traces, all 25 dev games, mined this wave)

- **25/25 dev games now have engine-verified full winning traces** (19 human
  extracted tonight + 6 BFS fixtures), written in explorer-fixture schema to
  wave-2 scratch `human_fixtures/`. Human traces contain mid-trace RESETs ⇒
  offline instruments only (live gateway swallows RESET); per-level segment
  extraction needed for live use.
- **Efficiency is NOT our gap**: on levels our 3.8 arm completes, we play at
  0.915× human-median actions (n=18). The deficit is DEPTH (humans 6–10
  levels; our wave-1 capped at 2). `baseline_actions` ≈ human median
  (104/183 within ±1) ⇒ RHAE 100 = human-median play.
- **Frame-0 `available_actions` is a free archetype dispatch channel**
  ([6]=click-hammer, [1–4]=avatar; agrees with tags) — the wiggle battery
  spends live actions probing what the menu states. Rider: menu-based
  dispatch (+0.05–0.15).
- **Per-archetype takeover triggers**: explorer waits fixed 120 actions/level;
  human p90 probe-before-L1 on click games is 37. Archetype-p90 triggers
  convert dead decode into search (+0.1–0.3, rig falsifier). Pairs with
  (re-derived) triage.

## 4. Tier-list re-price (post-3.8) + queue

- **Franzen PoE + DFS: KILLED** (need trained-in 64-token grid vocabulary /
  raw forward(); no analog on served FP8 chat model). **In-kernel TTT:
  infeasible** on memory arithmetic (BF16 upcast ~54GB can't co-reside with
  FP8 serve; pausing vLLM kills 28 games); offline variant = existing gated
  fine-tune lane.
- **Retrodict → expect-queue rider** (+0.1–0.25, ~250 LOC, envelope-safe by
  construction — only truncates batches): per-step expected-cells contract,
  halt-on-mismatch + diff in tool result, "batches >3 require expects".
  Zero-GPU falsifier: replay recorded batches, count saved actions. Its
  99.86% needed a frontier model at ~1000× our token budget — protective
  value here, not their score. Predict-before-act recurs across every public
  saturation (VISTA, Schema, arc-skill, NVIDIA AVO).
- **OPINE probe-prior**: gated on tonight's read (clickmap flying tonight is
  its live proxy); ≤1 day reimplementation if ≥2.0.
- **Upscale-8: downgraded** (VISTA has no upscale ablation; misgrounding
  claim unsourced; our coordinate chain verified consistent). 1h probe only,
  quota-slack.
- **Queue order unchanged at top: MTP > triage-at-55** (triage now requires
  the FN re-derivation per correction #2; MTP must fly with
  `--no-enable-prefix-caching` per the wave-1 GDN rule; MTP head confirmed
  present in the mounted repack; saltb0x's 0.27.1 wheelhouse = a field-proven
  upgrade lane if 0.19-with-flag disappoints). Wave-1 Tier-1 bug fixes
  (yield-resume carryover ~2.2h; xhigh→medium; tool-API friction) out-price
  everything else on this list.

## 5. 48h actionables (ranked)

1. Read pack-v22 vs the ~2.2 floor (bands in HANDOFF §2); the config-delta
   audit is now DONE — sampling excluded, remaining named deltas: none found
   between our harvest arm and the floor recipe besides grafts/serving detail
   ⇒ a low read points at rig/throughput, exactly what actionables 2–3 probe.
2. Concurrency-28 throughput probe (commit-run, quota only).
3. Decide on the parquet telemetry channel (Ahmed's rules call).
4. Manual browser read of forum thread 732854 comments.
5. Offline A/Bs in order: reasoning_effort medium; temp 0.6→1.0; then
   yield-resume carryover build.
6. Diary: GLM-5.3 weights ~08-28; re-derive triage FN curve at 132-min.

Wave-2 scratch (session 9a35ebe3): `lb_0821/` (full LB CSV), `kpull/`,
`srcpull/` (diffable bundles), `human_fixtures/` (19 verified traces),
`mine_replays.py` outputs (budgets_v2/motifs/dispatch/playbook/fixtures).
