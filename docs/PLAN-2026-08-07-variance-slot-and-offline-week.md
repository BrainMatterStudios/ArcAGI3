# Plan 2026-08-07 — tonight's variance slot + the offline adjudication week

Approved by Ahmed 2026-08-07 (evening session): use tonight's slot; best-tested
submission; proceed with parallel offline work; no slot may be lost to a buggy
submission. Supersedes nothing; extends `CAMPAIGN-PLAN-2026-08-04.md` and the
2026-08-05 pre-reset amendment (whose Friday variance branch was displaced by the
v8 auto-submit and is executed tonight instead).

## 1. Tonight (2026-08-08 00:01 UTC): frozen variance v1

**Arm.** `arc-agi-3-duck-variance` v1 — plain base duck, analyzer sampling only:
temp/top_k/top_p 0.6/20/0.95 → **0.9/50/0.98**, unseeded. Registered hypothesis:
large-effect / rank-upside screen on draw variance. Reading rule (pre-registered,
unchanged): only a score outside 0.69–1.27 is individually actionable; IN_BAND is
inconclusive by design; ERROR/≤3500 bytes/fast-COMPLETE = infrastructure, not a
model result.

**Why this arm.** (a) Kernel pushes are quota-blocked until the Saturday reset, so
tonight can only be an already-COMPLETE published version; variance v1 is the only
such arm with an unrun registered hypothesis. (b) Under the real scoring mechanics
(final = best-of-2 selected duplicates ≈ mean + 0.56σ; public rank = max-over-draws)
draw variance is nearly as valuable as mean. (c) The 35B sparse arm is deferred to
its capability gate (see §3) — public evidence (sonpham fork 0.000, 27B > 35B-A3B on
agentic benchmarks) makes an ungated live draw a 3–8% shot with real burn risk.

**Execution path (all steps completed and logged this session).**
1. Identity re-attested live: remote v1, canonical code-cell hash
   `71c25dc2…` (method: `"\n".join(code cells)+"\n"`), remote == tracked local
   notebook, status COMPLETE, sampling fingerprint 0.9/50/0.98 verified in-source.
2. `submit_gated.py --dry-run` passed all pre-submit gates with the exact
   registered message.
3. One-shot runner `scripts/submit_variance_20260808.py` armed under caffeinate.
   Guards: one-shot UTC window (00:01–02:00 Aug 8 only), full identity re-attest at
   fire time, same-day race abort, then the sanctioned gated path (builder honesty,
   COMPLETE+settle, never-played watch with `SSL_CERT_FILE` set so the watch is not
   SSL-blind). Full chain tested end-to-end in mock mode before arming.
   Log: `scratchpad/variance_submit_20260808.log`.
4. After scoring: record ref/bytes/version/hash/classification in both ledgers.

**Known leftover automations audited:** the 2026-08-06 retune launchd job is a
spent one-shot (`runs=1`, no calendar trigger) and cannot re-fire; the earlier
duck-sparse `auto_submit_on_reset.py` is dead (not running); the AgentSecurityComp
watchdogs target a different competition.

## 2. Zero-live-action test battery (running / queued)

From `docs/RESEARCH-2026-08-07-human-play-idea-sweep.md` — five of the Top-8
mechanisms adjudicate offline before any build is trusted:

| Test | Decides | Bar | Status |
|---|---|---|---|
| A-not-B log mining | perseveration brake (rank 7) | >15% repeat-fraction in zero-bank runs; winning traces must not need repeats | RUNNING (agent) |
| Human-replay artifact check | behavioral budgets wildcard | dataset downloads + parses | RUNNING (agent) |
| Contingency classifier on logged triples | wiggle/masks (rank 2) | ≥80% avatar ID on avatar games; no false SELF on click games | queued |
| Change-list vs raw-grids probe | differ/comparator (rank 4) | raw-grid change detection materially worse | queued (needs local 27B or gate kernel) |
| Undo-fidelity audit on offline envs | peek()/residue wildcard | per-game act-undo fidelity table | queued |
| HUD counter census | smuggled-reward wildcard | ≥4 games with non-clock counters | queued |
| Archetype classifier accuracy | dispatch (rank 3) | ≥80% on 25 public games | queued (needs rank-2 probe features) |

## 3. Weekend GPU (quota resets Sat 2026-08-08): the two gates

1. **Track B capability gates** (existing frozen design): Qwen3.6-35B-A3B kernel —
   amended with `--max-num-seqs 512` (known Blackwell cache-block failure; our
   setup lacks it) and a first-class protocol-compliance metric (sonpham-style
   silent collapse must be measured, not inferred) — then the AgentWorld candidate.
   GO bar unchanged: ≥95% protocol success, ≥2/4 target first unlocks, no ≥2-level
   control regression. Only a GO earns the live duck-sparse v7 slot.
2. **Track A patch closure** (existing frozen design), with two sonpham-informed
   candidate additions *behind env pins*: no-impact detection (their +55%) and the
   60s-yield act-look-act tempo. Reminder: patched-family live draws now read
   directionally BELOW base (n=5 mean 0.788 vs base 0.929, ~1.75 SE) — if closure
   fails, the duplicate config reverts to byte-reconstructed base v2.

## 4. Next-week build spine (gated on §2 results)

Ship order from the sweep: differ/comparator (independent plumbing) → wiggle
battery + masks → `run_probe` macro-action → two-way archetype dispatch. Each build
lands with unit tests + an offline dev-game A/B at eval geometry before any live
slot. Hypothesis ledger and forward-model builds follow only after their offline
bars clear. Every future live slot keeps the n=1 reading discipline: fingerprint in
the message, classification against the 0.69–1.27 band, ledger row on completion.

## 5. Slot-safety doctrine (restated, now enforced in code)

One sanctioned path (`submit_gated.py`), one armed automation at a time (audit
`pgrep` before arming), identity re-attest at fire time, one-shot windows on all
scheduled submitters, and no live arm without either a passed gate or a
pre-registered frozen identity. A failed/buggy submission is a process defect, not
bad luck.
