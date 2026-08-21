# HANDOFF 2026-08-21 — pack-parity flight, big-lever queue, standing laws

START HERE for a fresh session. Everything below was verified this week against
live Kaggle API, engine source, or run artifacts — not inherited from memory.
Memory index: `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/MEMORY.md`
(the 08-17 audit memory holds the day-by-day detail).

## 1. Scoreboard reality (as of 2026-08-21 ~18:00Z)

- **Us:** LB best **1.74**. Live Qwen3.8 series (all attested arms):
  1.29, 1.74 (duck-38 v2 = June stock + model swap), 1.55 (v12/anim bundle),
  1.33 (explorer v2), — (08-20 KILLED at the 9h wall), 1.51 (explorer v7).
  Mean 1.48, sd ~0.18, n=5. Single-draw resolution is ±0.4: every lever so far
  reads null live.
- **Leaders:** cstl 2.81, Lord Han Solo 2.76, Franzen 2.58; six teams at
  2.39–2.72 with ≤12 submissions — the central unexplained fact.
- Milestone-2 **Sep 30** (public LB BEST). Final Nov 2 (private set MEAN,
  2 selections; harvest luck does not transfer — mean levers carry).

## 2. TONIGHT (2026-08-22 00:01Z): pack-v22 recipe-parity arm — ARMED

- `scripts/submit_packv22_20260822.py` running (nohup). Kernel
  `ahmedmobasher86/arc3-pack-v22` v1, svid 343925396, code hash `a2b29022…`.
  Landing check detached → `scratchpad/packv22_landing.out` (00:30Z).
- WHAT IT IS: faithful fork of public `thtennant/arc3-duck-v22` (the nightly
  kernel the 2.4+ cohort plausibly forks): byte-identical June stock harness +
  Qwen3.8 (FOYSAL Kaggle-Model repack — in-kernel attestation matches our
  official-weights fingerprints) + `taaf_grafts` armed exactly as published:
  `{efficiency, retry_guard, shortcircuit, goalkeep, hudmask, clickmap,
  searchmap}` — prompt-information-only fixes; banking/transfer/schema OFF.
- Fresh audit (08-21): 4-file delta vs our 08-17 line-by-line audit; clickmap +
  searchmap are info-only, fail-open, blanket-guarded; base bundle byte-equal
  to June stock. SAFE-TO-FLY verdict in the audit agent report.
- **PRE-REGISTERED READING:** ≥2.0 ⇒ the public recipe explains the 2.4 cohort
  → it becomes our floor; every lever re-tests on top; next slots = re-draws +
  one gated lever each. ~1.5 ⇒ recipe theory dead → their edge is draw-count +
  private tweaks; pivot to structural levers (MTP, triage) + composite harvest.
  >1.74 re-banks LB best regardless.

## 3. Standing LAWS (Ahmed directives — binding)

1. **Skepticism:** no assumption proceeds as fact. Validate it or flag it
   explicitly in the safety case. (This week: my settings-diff refuted the
   hunt's "-1.0 our-stack leak" claim; the diff is the law working.)
2. **Never lose a submission day.** Safe default armed FIRST each day
   (exact-byte duck-38 v2 harvest runner pattern); upgrades replace it only
   after full validation + re-clear + Ahmed's go for new-mechanism arms.
3. **Envelope law:** any change that can extend per-game/per-run duration
   needs a WRITTEN worst-case arithmetic case (`docs/ENVELOPE-2026-08-20-v7.md`
   is the template). The 9h envelope only closes via early finishes (proven
   fatally 08-20, sub 55634118 killed at the wall).
4. **Re-clear rule:** every evening (~23:00Z latest) the armed arm re-clears
   against anything that went green that day; swap by default.
5. **Ops:** verify mock rc EXPLICITLY before arming (a piped `tail` masks
   rc — bit us 08-21); never launch two runners for one slot (double-submit
   race); pkill by script name works; detached `nohup bash -c` survives
   harness kills, plain background tasks do not; machine sleep kills long
   `sleep` waits — use detached file-writing watchers + short harness wakes.

## 4. Component inventory (all committed on `winning/duck-patched`)

| Component | Path | Status |
|---|---|---|
| Safe harvest runner pattern | `scripts/submit_38_20260822.py` (retarget dates) | proven 3/3 envelope completions |
| Boot attestation cell | inside `submission/_duck38_v12/build_duck38_v12.py` | flown 6×; discriminates official 3.8-FP8 vs vrfai 3.6 (fp8/e4m3 + tf 5.8.0.dev0) |
| Explorer graft v7 | `submission/_explorer_floor/graft_explorer.py` | live-flown (1.51); envelope-guarded; reset-replay BFS validated 4/5 zero-games offline, 4/4 live smoke |
| Grind-to-win + self-bank | same file (v5/v6.1 path) | offline official 100.00 on tu93 (`validate_v5_tu93.py`, `validate_v6_tu93.py`); bounded takeover validated |
| Banking graft | `submission/_duck38_v12_bank/graft_bank.py` | validated vs competition server; trigger (full wins) never fires live — rider only |
| Animation digest | `submission/_anim_digest/` (`digest.py`, `graft_digest.py`) | falsifiable sb26 test PASS; seam-validated 35/35; smoke: ft09 best-ever 3/6 15.12 (in xd combined) |
| Retry guard | `submission/_retry_guard/graft_retry.py` | 16/16 tests; fixed community bug (authed-vLLM 401 misread) |
| Combined xd arm | `submission/_duck38_v12_xd/` | best smoke profile (ft09 15.12, vc33 17.73); staged, unflown |
| pack-v22 arm | `submission/_pack_v22/` | ARMED tonight |
| Fixtures (6 mechanical wins) | `submission/_explorer_floor/fixtures/` | triple-verified replayable traces (tu93 full 9-level 185-act win) |
| Envelope safety-case template | `docs/ENVELOPE-2026-08-20-v7.md` | law #3 artifact |

Killed by measurement: yield-180 (vc33 regressed 2×), batching (priced +0.5
lvl/box max), transfer-as-slot (premise ~20%, worst-cased by Ahmed), forward-
walk grinding (can't crack anything), stacking unvalidated arms at night.

## 5. Big-lever queue (hunt wf_2e50ef2e, adversarially scored)

1. **[FLYING TONIGHT] pack-recipe parity** — see §2.
2. **MTP speculative decoding** — 95% of wall is decode (~50s/call, ~42
   calls/level-up). Qwen3.8 ships an MTP draft head:
   `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` on our
   pinned vLLM 0.19. Sibling-measured 2.4–2.5× decode. LANDMINES: FP8+MTP
   crash (vllm issue #40756); quantized-head acceptance collapse (5–11%);
   community data says speedup collapses at concurrency 28 → pair with
   concurrency 8–16. NO MODAL (Ahmed) — validate via Kaggle smoke: tok/s +
   acceptance at conc 8/16/28, parser round-trip, then dev-game envelope smoke.
   Priced +0.2 to +0.6.
3. **Triage-at-55 + TimeBank** — kill zero-level sessions at minute 55
   (measured FN 3%, n=77 incl. 35 hidden-run observations — see hunt journal);
   reclaims ~99 worker-hours per run; structurally fixes the 08-20 death mode
   and funds 55-min grind-to-win banks (each ≈ +0.9) the v7 caps exclude.
   Build: zero-GPU queue simulator first, then rig night, then slot.
4. Watch list: fp8 KV cache + prefix-cache audit (rider on MTP session);
   clone-fingerprint probe over our own logs (free); estimator phantom-token
   patch; temp-1.0/top-k-20 sampling alignment (offline A/B only); upscale-8
   grounding probe. All in the hunt journal:
   `~/.claude/.../subagents/workflows/wf_2e50ef2e-300/journal.jsonl`.

## 6. Measured facts a fresh session must not re-derive

- Scoring: per level `min(115,(baseline/actions)^2*100)`, weight level_index+1,
  play cap 100, completion-share cap; game = MAX over plays; post-WIN RESET
  opens a fresh play (engine + live verified). Efficient levels SUBSIDIZE junk
  levels inside a play; partial runs have no subsidy pool (grind self-harm
  measured 100→1.2).
- Engine quirks: `GameAction(value)` raises — use `from_name` only. Offline
  wrapper play-accounting differs from the gateway — validate scorecard
  mechanics ONLY through `taaf.competition_arcade.CompetitionArcadeServer`.
  Slow-tick HUD bands (10–40% change rate) beat threshold volatility masks —
  band rule: border band changing while interior unchanged (2 events ⇒ mask).
- Turn economics (measured): 95% wall = decode; 67% of calls inspection-only;
  yield-overruns are re-orientation, not lost generation (60s budget checked
  at boundaries only; raising it VERIFIED harmful at 180s).
- Quota: scored reruns do NOT bill user GPU quota; commits/smokes do
  (~30h+/week). Kaggle 403 on `SaveKernel` = stale foreign `id_no` in copied
  kernel-metadata.
- The honesty gate (`scripts/submit_gated.py` PACK_MARKERS) keyword-scans the
  submission MESSAGE — words like banking/replay/hud/frontier/depth require
  matching notebook markers; extend the map honestly, never reword to deceive.
- Baseline-actions exposure probe: HIDDEN on offline commit runs; still
  unresolved for the live gateway (probe rides in y180 arm cell, reusable).

## 7. Tomorrow morning (08-22) checklist

1. Read pack-v22 score (~09:30Z) against §2's pre-registered bands. Update
   ledger (`docs/submission-ledger.json`) + memory.
2. Branch A (≥2.0): pack-v22 = new floor. Next slots: re-draw it while adding
   ONE gated lever per slot (first candidates: animation digest — its `xd`
   smoke profile was best-ever; then explorer as rider; then MTP).
3. Branch B (~1.5): recipe theory dead. Ship composite harvest nightly
   (xd arm is the staged candidate) and push MTP + triage-at-55 validation
   hard — they are the remaining ≥+0.5 candidates.
4. Either branch: safe default harvest runner armed FIRST for 08-23 before
   any experimentation (law #2).
5. Check quota before smoking (heavy use this week).

## 8. Working-state pointers

- Armed runner: pid in `ps aux | grep submit_packv22`; log
  `logs/pack_v22_20260822_runner.log`; marker `logs/pack_v22_20260822.marker`.
- Scratchpad (session-scoped, may vanish): bundle copies + smoke outputs at
  `/private/tmp/claude-501/.../scratchpad/` — bundles/{june,anim,banking,banking_v22},
  smoke outputs {v12smoke,banksmoke,depthdiag,xpl2,xpl7,y180,y180b,digest1,xd,packv22},
  workflow journals under `~/.claude/projects/.../subagents/workflows/`.
- Git: all work committed on `winning/duck-patched` (NOT pushed anywhere).
  Ledger current through sub 55656535 (v7 = 1.51); tonight's sub appends via
  submit_gated auto-ledger.
- Kaggle kernels (ours): arc-agi-3-duck-38, arc3-duck38-v12{,-bank,-xpl,-y180,
  -digest,-xd}, arc3-pack-v22, arc3-depth-diag.
