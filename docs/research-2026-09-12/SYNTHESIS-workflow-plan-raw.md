# Recommended plan: 2026-09-12 → Nov 2

Synthesis of five plans and three judges. Judges' totals: Absorption 112, Loop Shape 110, Outsider 104, Compute 102, Model Axis 97. Two judges ranked Absorption first, one ranked Loop Shape first; all three named the same "best idea to keep" from each runner-up, so those are grafted below. Numbers come from the verification phase (cited as "verifier N" or "instrument/stats audit") unless marked "checked today", which means I ran the command myself this session.

Checked today: leaderboard reads Tufa 11.04 / NVARC3 **8.40** / Third Intelligence 8.21 / Franzen 7.63 / mostik.ai 7.51 / Fususu 6.91 (`kaggle competitions leaderboard --show`, entries dated 09-11); 52 submission slots remain including today and Nov 2; `offkaggle/run_regime_wave.py:313-315` still hard-codes the six-draw pool `[41,40,37,36,40,42]` / 39.33 / 2.34 and `offkaggle/read_cadence_arm.py:17-19` the 48/45/44 rule; `offkaggle/modal_flashnext_serve.py:293` still has the 6 h container reaper; `docs/submission-ledger.json` header still says generated 2026-08-04 / 45 submissions over 85 rows, row 56133282 is now `complete` / 2.73, and there is no row for 55543514.

---

## 1. The one-paragraph verdict

We are at ~3.2 because we fly the best public artifact (a byte-identical copy of keith V14) and nothing public scores higher; every in-house lever measured since 09-02 reshaped behaviour without moving the per-game comprehension distribution. The per-game matrix says the leaders are roughly "our brain on its best draw of every game" (single draw 38.5 levels, E[max of 3] 50.8, union of 10 draws 64 — Outsider plan, reproduced by all three judges), and the 3x-clock wave passed only 3 of the 12 never-passed walls (Loop Shape plan, reproduced by judges 1 and 3). So the harness side tops out at roughly +0.3–1.0 LB, the model axis was closed on wrong numbers but remains a 10–15% shot that collides with the calendar, and the only path to the 7+ tier is absorbing a Milestone-2 release on Oct 1. The plan is therefore: **make Sept the preparation month for Oct 1 (bundle-agnostic rig, serving-transplant kit, release watch, fixed instrument), run a capped ~$50 in-house lane of cheap, composable grafts alongside it, and spend Oct on absorption plus mean-based selection.**

---

## 2. STOP doing (effective today)

| Stop | Why (evidence) |
|---|---|
| Behaviour-shaping prompt grafts (probe/carry/evid/hypo/fx class) | Three on-instrument replications flat; class closed for step-sized effects (stats audit). |
| Window-shrink and reasoning-strip arms (ctx12/ctx16/nr16 family) | Catastrophic: 12/21/20 levels; retained history is load-bearing (verifiers 2-4). |
| Executable-world-model lane (A2 / Stage-1 / Polyphony / NOOA ports) | One verified model costs 88-89% of a stock game's decode; models score 0-45% on the next level (verifier 10, corrected denominator). |
| Per-game crack/specialist solvers | Hidden games are different games; deployable inventory was 1 (memory two-level-wall). |
| Single-wave "dead" verdicts on arms reading 31-46 levels | At the honest sd (~3.5) a true +4-level lever reads "dead" 62% of the time from one wave (stats audit). |
| Routine V14 redraws "to read the band" | Zero information about the final; the base band is known (2.0-4.7). Every remaining slot must test a hypothesis or bank a draw of a config that could be a final. |
| Buy-calls arms priced in levels at elasticity 0.21 as if closed | The sizing table mixed anchors (44.0 not 42.6, verifier 6); reopened only as loop-agnostic serving riders inside a stack (step 5). |
| 3x-clock waves as a level instrument | Illegal live; keep only as a trajectory generator. |
| Silent waivers of instrument faults | Budget3x and ctx12 both had mid-wave container replacements that the write-ups never mention (instrument audit). |

---

## 3. Corrections that must be written back (do this on day 0, $0)

### 3a. `offkaggle/REGIME_WAVE_STATUS.md`

1. **Verdict instrument (lines ~615-641, 687; plus `run_regime_wave.py:313-315`, `read_cadence_arm.py:17-19`).** Replace pool 39.33/sd 2.34 with the 8-draw flat live-geometry mean **38.5, sd ~3.5** (from per-game variance: within-game var 0.417-0.485 per game; the n=6 sd 2.34 has 95% CI [1.46, 5.74]). Record the pool's composition: two Kaggle commit totals (37, 42) with no per-game data on disk, M4 is the retry-graft arm (fired 5x), three of six are yield900. New rule: one wave kills only below ~31 or flags above ~46; any ENGAGED arm reading 31-46 gets a mandatory counterbalanced second draw; two-wave mean >=45 = step candidate, <=42 dead; arm-minus-pool sd = 3.5 x sqrt(7/6). Restate: cadence 34 and fx 35 are "not a step", not "worse than base" (-1.2 to -1.5 sd); probe/carry 41 are "flat, indistinguishable from +10-15%"; "engaged-but-flat class closed" holds for step-sized effects only and rests on three replications (patch 21 is 27B-era).
2. **VOID rule (lines 470, 622; `run_regime_wave.py:1365-1378`).** Rewrite to fire on: any HTTP 5xx, `process_start_time_seconds` differing between before/after, preemptions per request outside 0.02-0.35, or any game's shim span < 0.99 x clock. Never on end-of-clock ReadTimeouts. Record that the old rule fired on every wave including the base pool and was waived explicitly for cadence/fx/probe and silently for budget/ctx12/ctx16/nr16.
3. **keith_budget 3x (lines 1242-1342).** Delete "first outright win in rig history" (ft09 won 6/6 in KV10 on 09-03, STATUS:369). Restate the table against the pool: calls 3.18x, actions 4.95x, score 1.75x (not 2.15x), 5+-level games ~0.75 -> 3. Add: 41 of 56 clears landed by 7,920 s (+0.71 sd), so the base is consistent and the extra clock bought ~15 levels. "Elasticity decays 0.40 -> 0.21" is a base-choice artifact (0.20 vs 0.21 on a common base). 56 sits in the pre-registration's own 46-59 "partial, not a step" band; drop the "BUDGET CONVERTS / not at a capability ceiling" headline. Disclose the container replacement at +297 min (the rig's own 6 h reaper, `modal_flashnext_serve.py:293`), 24 HTTP 500s, the unreadable VLLM row, and that pre-restart server telemetry is unrecoverable. Tag "instrument fault, waived"; note only 3 of 12 pre-registered walls fell (dc22, sb26, vc33). Sigma: ~+4.4-5 at sd 3.5.
4. **keith_ctx12 (lines 1342-1446).** Disclose the container death at +76.6 min (25 x HTTP 500, 398 s gap, 5.9% of clock lost, cause not in local files); all VLLM-line numbers and "preemptions 0" cover only the post-restart 40% of calls. Mechanism correction: history budget is ~2,950 real tokens (~1 turn) because the trimmer estimates chars/3 of the JSON payload including the base64 image (`tool_agent.py:462-467`); the collapse is routine amnesia (16% of mid-game calls with zero history, 0% acting there), not rare 10,439-token completions. "20,061" is the base MEAN (median 21,421). Address the VOID rule explicitly.
5. **keith_ctx16 (lines 1446-1580).** Disclose 1,908 preemptions (0.93/request; every 32k wave is 0.05-0.3). The longest-completion premise was already contradicted on disk (10,495 / 10,673 / 11,685 / 13,800). "Smooth gradient / no safe window" is over-claimed: actions/call is linear 32k->16k and then collapses at 12k (a knee); the visible mechanism is retained history falling ~7 -> 3 -> 1.5 exchanges. Sigma ~-5.
6. **keith_nr16 (lines 1580-1621).** The "base (32,768)" row is the 09-09 conc-6 cadence arm (34 levels, e2e 26.6 s) with 39.33 pasted in; replace with conc-28 base numbers (e2e ~145 s, prompt ~21.4k, 55.8 calls, 2.76 actions/call). "The trimmer always fills the window" is true only in estimator units (real prompts ~58% of the window). The graft restored +0.6 retained turns, not +7, so branch 3 never triggered; drop the OpenAI-corroboration sentence. Disclose 1,652 preemptions.
7. **Sizing (lines 1492-1540).** Re-anchor: best live-legal stack 44.0 levels (shortfall 4.0, ~1 sd); service-for-48 is 8.1-8.7 s not 7.7; "the ceiling settles it" is false (perfect MTP-8 projects 48.8-52); Route B scales as calls ∝ KV^0.56 (+16% at 1.3x KV). The family is closed by the measured acceptance decay plus elasticity 0.21 +/- 0.05, with KV10's own value at 0.20 +/- 0.21.
8. **Cadence/fx (lines 979-1018, 1069-1110, 1161-1162).** Calls 56.76 includes tr87 running alone (24 others: 51.6). "24/25 games saw a 7+-line block" -> 16/25; correlation +0.037; `time_remaining_seconds` appears ~1/game, not in every tool result. "The agent does not act on the table" was unmeasured: it quotes the table in 7% of turns, never in code. "Throughput family closed" was contradicted by budget3x the next day.
9. **A2 / PREREG-a2 (docs/research-2026-09-09).** "5,016 s" -> 5,857 s; pass-1 ranges 8,114-18,912 / 13,856-24,654; "7,123-token prompt" -> 7,506/7,045; cost denominator: one verified model is 88-89% of a stock game's decode (83,094 was the carry75 arm). Downgrade "the failure is the LOOP" to "unread after three instrument iterations": the directed calls were fresh 2-message contexts (play-history contamination refuted) and the live encoding differed from offline (60-cell cap truncating the very toggles Stage-1 modelled, 12 vs 20 transitions, different system prompt, 3 one-line-feedback calls).
10. **R-what-our-agent-actually-does.md.** "0 forward simulations" is false (>=30 hand-verified in the 2,047; ~3x that rate in the excluded mid-turn calls). Builder bug `src/modelaxis/build_corpus.py:154` (`steps[idx]`); 1,288 cleared-level calls wrongly dropped; true population ~3,350; 19 result dirs / 9 arms, 72% non-base; "100% ascii" is vacuous (63.4%).
11. **R-triage-family-bounded.md.** AUC 0.94 is target leakage (P(final>=3 | 2+ levels by 60 actions) = 0.57-1.0); on the incremental target no eval-legal feature beats 0.52; the keep-top-X% table has no code in the repo; at elasticity 0.21 the in-range rows read -6% to +9%; the efficiency term is omitted.
12. **PREREG-model-axis.md RESULT.** 180.0 B on disk (not 131 B); 51.2 B PLE table is CPU-offloaded and frozen in training; the served NVFP4 checkpoint keeps every attention/linear-attention/hyper-connection tensor in bf16 (the LoRA targets); ms-swift publishes an 8-GPU LoRA recipe at 64-78 GiB/GPU; Modal H100 is $3.95/h (8x = $31.6/h); no "8B dense" antecedent exists; the M2 stop rule was never exercised. Closure reason becomes "unproven tooling + calendar", not scale.

### 3b. `docs/submission-ledger.json`

- Regenerate the header (`generated_utc`, `counts.submissions` 45 -> 85; Kaggle returns 86).
- Add the missing row **55543514** (2026-08-16, duck-38 first flight, COMPLETE 1.29).
- Row 55634118: status `pending` -> complete / no score (runtime exceeded).
- Row 56111215: strike "Matches the off-Kaggle ledger's NULL (118 = 118)" (cache.json is gone; rig-only recompute reads yield900 +5 to +10 levels vs pure stock); add "decided by 0.01; 3-draw SE 0.46 exceeds the 0.4 gap between the step and dead lines; for FINAL SELECTION the higher-mean config is yield900 (3.59, n=3) pending further draws".
- Row 56133282 (already 2.73/complete — confirmed today): replace "CV on identical bytes 0.25" with v4 CV 0.12 (sd 0.35, n=3) and yield900 CV 0.28; report v4 (2.85 +/- 0.35) and the pooled family (3.22, n=6, sd 0.78) as separate rows.

### 3c. Memory files

- **MEMORY.md** is over the size limit (25.7 KB > 24.4 KB); trim old entries and add one line: "VERDICT INSTRUMENT CORRECTED 2026-09-12 — base 38.5 / sd 3.5 / two-wave rule / VOID rewrite / instrument-fault register".
- New file `arcagi3-instrument-faults-2026-09-12.md`: budget3x reaper restart (+297 min), ctx12 container death (+76.6 min), ctx16/nr16 preemption thrash (0.93 / 1.12 per request), 132152 ENOSPC, all with the coverage table from the instrument audit.
- `arcagi3-queueing-not-compute.md`: cadence is -1.2 to -1.5 sd not -2.28 and confounded with a 4.5x clock cut; the throughput family was reopened by budget3x the next day.
- `arcagi3-model-axis-blocked-on-scale.md`: rewrite per 3a-12.
- `arcagi3-a2-workspace-closed.md`, `arcagi3-polyphony-priced-out.md`, `arcagi3-executable-models-dont-transfer.md`, `arcagi3-stage0-reopened-flashnext.md`: apply 3a-9 (numbers, denominator 88-89%, "memorisation" softened — the models were never asked to generalise; cn04 scoreboard is 0,0,1,2,1,3,6,0,9,17,18).
- `arcagi3-step-change-search-2026-09-02.md`: base draw #4 = 2.73; family n=6 mean 3.22; yield900 "dead by 0.01" with the selection note above; NVARC3 8.40.
- `arcagi3-two-level-wall.md`: "four replications" -> three on this instrument.

---

## 4. Honest LB target

- **In-house only (no usable release):** 3.3-4.0 public mean. Base family 3.22; the restart lever is worth +0.2-0.6 if it converts and the serving stack +0.2-0.5 if it stacks; neither is individually detectable in one live draw (per-draw sd 0.8, one-draw MDE ~1.6 LB), which is why they fly only as a composed arm under a sequential read.
- **With a Kaggle-runnable release from a >=7 team absorbed on Oct 1-3:** 6-9. The 0.77 mean/max ratio is the V14 cohort's and is a heuristic, not a measurement; a heavy resubmitter's copy may read 5-6. Judges put P(such a release) at 45-55%, the plan author at 60-70%.
- **Probability-weighted expectation: ~4.5-5.** The 7.5-11 tier is not reachable from in-house work on this brain by Nov 2, and nobody on the panel claims otherwise.

---

## 5. Ordered steps

Each step is gated; nothing pushes or deploys without Ahmed's go (his standing rule). Budget cap for the whole in-house lane: ~$60 rig + <=10 h Kaggle quota before Oct 1, leaving >=30 h quota unspent in the week of Oct 1.

### Step 0 — Fix the instrument and the record (Sept 12, $0, half a day)
- **What:** Apply section 3. Edit `run_regime_wave.py` (pool, VOID rule) and `read_cadence_arm.py` (thresholds). Re-read all 12 existing verdicts under 38.5/sd 3.5.
- **Why:** Every later read depends on it; the old rule kills +10-15% levers and cannot see instrument faults.
- **First test:** No dead arm flips to a step candidate under the new base (expected: cadence/fx become "not a step", probe/carry "flat", budget/ctx12/ctx16/nr16 unchanged).
- **Kill:** None; if Ahmed rejects sd 3.5, every later threshold is restated at his sd.
- **LB:** 0 directly.

### Step 1 — Release watch (Sept 12, $0, ~4 h)
- **What:** Script a daily `kaggle kernels list` / `kaggle datasets list` diff over the 26 top-20 member accounts plus the Tufa and NVIDIA GitHub repos and the forum listing; baseline it today. (Absorption plan step 1.)
- **Why:** Converts Oct 1 from a scramble into an hour and stops burning quota.
- **First test:** It flags the known 09-01 jakobbrggen release and the 08-29 sonpham serving datasets as "new since baseline".
- **Kill:** n/a.
- **LB:** 0 directly.

### Step 2 — Two class-level diagnostics before any wave (Sept 12-13, ~$5)
- **2a. Flash-Next oracle probe** (Loop Shape step 0; all three judges' "best idea to keep"): port `submission/_oracle_probe/oracle_probe.py` from ollama (:77) to the Modal endpoint; ORACLE vs CONTROL on wa30 L3, 5 clones each, wins replay-verified through the engine.
  - **Test:** >=3/5 ORACLE clears within 100 moves, CONTROL <=1/5.
  - **Kill:** ORACLE <50% => every "harness supplies dynamics, model plans" design is dead on this brain (no effect-table planner, no harness planner is built). The restart lever (step 4) does not depend on this.
- **2b. Prefix-caching smoke on the pinned build** (Compute plan step 2; judge 3's pick): add `ARC3_PREFIX_CACHING` / `--mamba-cache-mode align` overrides to `modal_flashnext_serve.py`, boot, run 20 recorded prompts at temperature 0 with caching off/on/on, read `vllm:prefix_cache_hits`.
  - **Test:** boots; >=19/20 byte-identical completions; hit rate >=30% on the 3-game smoke.
  - **Kill:** boot failure, >1/20 divergence, or hit <20% => caching closed for this build (no vLLM rebuild before Nov 2).
- **LB:** 0 directly; each decides whether a later lane is alive.

### Step 3 — Bundle-agnostic rig runner + serving-transplant kit (Sept 13-26, ~$27, ~3 eng days) — THE SPINE
- **What:** `run_regime_wave.py` is pinned to the june_stock tree and the taaf pickles (:9, :106-110, :576-586, :2286-2291, checked by all three judges), so a foreign TAAF bundle cannot be measured on Modal today and the checklist's H+1..H+6 step is not executable. Build a runner that takes any released bundle (`deploy_target.pkl` + `src/` + `setup_commands.json`) and runs it in Kaggle geometry (conc 28, 7,920 s). Validate on the V14 bundle and on jakobbrggen's avo-v2 bundle. Package the keith V14 serving profile (RadixArk NVFP4, kv5-bf16-mtp3-c8-cg32) as a drop-in replacement for whatever vLLM stack a release ships. (Absorption plan step 2.)
- **Why:** The 27B-FP8 -> Flash-Next swap (1.94 -> 3.25) is the only live-measured step this campaign owns and is orthogonal to whatever a released harness does; without the runner a release cannot be read before spending slots.
- **First test:** V14 bundle reads inside 34-41 levels through the new runner; avo-v2 runs end-to-end without hand edits.
- **Kill:** Not runnable unmodified by Sept 26 => pre-write the 09-02 hand-port recipe (one day, it produced the live 3.25) and keep only the transplant kit.
- **LB:** 0 directly; enables steps 7-8.

### Step 4 — Full-history-wipe restart at the wall (Sept 14-22, <=$25) — NEEDS AHMED'S EXPLICIT GO (STATUS:432-434 says not to spend on this without it)
- **What:** `submission/_throughput_v1/graft_retry.py` with `RETRY_CLEAR_HISTORY=1` (the dose recorded as "not run", STATUS:426-434), no RESET, trigger on **turns on the same level** (T ~10-12 steps; the Outsider's fix for why the action-multiple trigger fired only 5 times), max 1 wipe per level / 2 per game, post-wipe prompt carries a compact factual ledger (actions tried, one-line observed effects, refuted hypotheses). Installed as an in-memory graft so it composes with any absorbed kernel.
- **Why:** Per-wall pass probability is 0.12-0.62 per draw and passes usually come early; E[max of 2] = +0.33 lv/game is the ceiling. Nine HIGHVAR games give a second mind real variance to draw on; 16 LOWVAR/CAPPED games will give nothing.
- **First test:** $2.5 six-game kill test (tn36, sb26, tr87, cn04, sc25, lp85 x 2 draws, conc 3): wipes fire in >=5/6 runs; paired levels >= those games' 8-draw mean minus 1; >=2 wiped levels clear afterwards (note-only dose: 0/5); the post-wipe reasoning must state a different hypothesis in >20% of wiped levels.
- **Kill:** Kill test fails => dead, $2.5 spent. Waves: two-wave mean <=42 or <=2 of the 12 walls passed => dead; >=45 and >=3/12 walls => step candidate, then 3 live draws under the sequential rule. Actions/call <2.0 = amnesia signature, dead.
- **LB:** +0.2 to +0.5 if it converts; P(step) ~15%.

### Step 5 — Loop-agnostic serving riders, flown only as a stack (Sept 13-24, <=$12 rig + ~6 h Kaggle quota)
- **What:** (a) KV 6.0 then 6.5 GiB in a Kaggle **commit** run of the byte copy (the boundary is Kaggle-only: 81.8 GiB load there vs 79.4 on Modal; 8 and 10 GiB OOMed, 6-6.5 never tried); (b) MTP-5 rig smoke reading per-position acceptance at positions 4-5; (c) capture the fp8-KV boot traceback ($0.5). If 2b passed, one $9 rig wave of prefix caching with the actions/call and reasoning-chars gates.
- **Why:** Each is +6-16% calls at most (verifier 6, calls ∝ KV^0.56; acceptance decay 0.78), undetectable alone, but they compose with each other and with any absorbed kernel; the serving profile is the one thing we transplant under a release regardless.
- **First test:** (a) vLLM log prints the reserved KV and the smoke completes without OOM; (b) acceptance at positions 4-5 >= 0.35; (c) traceback names the failing path.
- **Kill:** (a) OOM at 6.0 => KV headroom dead on this box; (b) engine refuses nst>3 or tokens/forward <3.1; prefix wave: calls/game <65 or actions/call down >15% or two-wave mean <=40.
- **LB:** +0.1 to +0.4 for the surviving stack; not a step. Do NOT build the lean-history graft before Oct 7 (boilerplate share did not reproduce for two judges: 59-66% vs the claimed 92%; it does not port to a foreign loop). Do NOT build the proven-controller re-execution arm (patch-21 precedent); its $0 offline replay is backlog only.

### Step 6 — Optional, only if step 3 is done by Sept 22: the $15 serving gate for the model axis (Sept 23-30)
- **What:** Model Axis S1: round-trip the served RadixArk NVFP4 checkpoint's bf16 attention/linear-attention/hyper-connection tensors (identity and 1e-6-perturbed variants), experts byte-identical, serve on the rig profile, 3-game smoke; one Kaggle commit with `model_sources` swapped. **No training before Oct 8.**
- **Why:** It converts "blocked on tooling" from an assertion into a measurement for $15 and zero training; it is the gate that killed every previous fine-tune. Training itself (ms-swift 8xH100 recipe, $130/attempt, $300 cap) is deferred to Oct 8+ and only if Track D produced nothing and S1 passed; P(step) 10-15%, and all three judges rated feasibility lowest.
- **First test:** Variant A smoke telemetry inside the smoke's band (MTP acceptance ~76/58/45, tool-call parse, 0 refusals) with expert sha256 unchanged; the Kaggle commit boots and plays.
- **Kill:** Re-sharded checkpoint fails to load in the pinned fork, or telemetry leaves the band => lane dead for engineering reasons, reported as such.
- **LB:** 0 (gate).

### Step 7 — Oct 1-3: byte-absorb the strongest Kaggle-runnable release, unmodified, and fly it (3 slots, ~27 h quota)
- **What:** `docs/TRACK-D-absorption-checklist.md` as written: `absorb_kernel.py stage` / `diff` / cost gate (completion tokens per game <= ~1.3x our 74k), read it on the new runner the same day, push UNMODIFIED with Ahmed's go. Priority: Tufa > Third Intelligence > NVARC3 (8.40 today) > Franzen > Fususu. GitHub-only release: budget one extra day to re-host weights.
- **Why:** The mechanism that produced our only live step (V14 copy, 1.94 -> 3.25 in one day). Every public >=4 today is our regime plus stock duck.
- **First test:** Live draw 1 >= 5.0 (above any V14 draw ever seen; z >3 on the cohort sd 0.55).
- **Kill:** Draw 1 <4.0 or not runnable in the 12 h box within 48 h => next release on the list; no release >=5.0 by Oct 8 => revert to the V14 family and say so.
- **LB:** +3 to +6 conditional; 0 otherwise.

### Step 8 — Oct 4-20: compose only what is measured
- **What:** If the release runs a 27B/31B model, lay the Flash-Next serving transplant under it (two counterbalanced rig waves, bar = copy's two-wave mean + 6 levels, then 3 live draws). If the release runs Flash-Next, add only the surviving in-memory grafts from steps 4-5, one at a time, under the same two-wave rule. No behaviour-shaping grafts.
- **Kill:** Two-wave mean below copy + 3, length-finish >1%, or completion tokens/game >1.5x => fly the unmodified copy.
- **LB:** +1 to +3 if the release is on a weaker model; 0 if it is already on Flash-Next.

### Step 9 — Oct 8 - Nov 2: endgame by config mean
- **What:** Every slot is a draw of a surviving config (v4, yield900, best in-house composed arm, absorbed copy, absorbed + grafts), read sequentially (continue until the running-mean 95% CI separates from the current best, cap 5 draws per config). Select the two final entries as two duplicate draws of the config with the highest MEAN, never the public max. Note the CPMP "no rerun, private score fixed at scoring time" ruling is from a forum snapshot and unverified from the repo; the memory's own contract (best of two selected, private locked at run time) gives the same selection rule.
- **Kill:** A config whose 3-draw mean is >1 SE below the leader stops drawing. If nothing beats the base family by Oct 25, select two draws of the higher-mean base config and stop.
- **LB:** +0.1-0.2 from selecting by mean; prevents a -0.5 to -1.0 mistake.

---

## 6. Calendar

| Window | Spine | In-house lane (capped) | Quota |
|---|---|---|---|
| Sept 12-13 | Step 0 write-backs; step 1 watch | Step 2 diagnostics ($5) | ~0 |
| Sept 13-20 | Step 3 runner + transplant kit | Step 4 kill test then wave 1; step 5 KV 6.0/6.5 commits, MTP-5 smoke | ~6 h |
| Sept 21-26 | Step 3 validation (V14 + avo-v2 bundles) | Step 4 wave 2 if alive; prefix wave if 2b passed; composed stack rig read | ~3 h |
| Sept 27-30 | Freeze; hand-port recipe pre-written; step 6 only if idle | No new waves | keep >=30 h unspent |
| Oct 1-3 | Step 7 absorb + fly | — | ~27 h |
| Oct 4-20 | Step 8 compose | LoRA S2 only if no release and S1 passed | as needed |
| Oct 8 - Nov 2 | Step 9 selection | — | — |

---

## 7. The next 7 submissions (each needs Ahmed's go; an ERROR costs no slot)

| # | Date (target) | Config | Hypothesis it tests | Prereq | Read |
|---|---|---|---|---|---|
| 1 | Sept 13 | Byte copy + `KV_CACHE_MEMORY_BYTES` 6.0 GiB (push_kv10 mechanism) | Boots on the Kaggle card at 6.0 GiB; score in 2.0-4.7 | Commit run Sept 12 (~3 h quota) | ERROR = OOM, KV headroom dead; in band = alive, banked draw |
| 2 | Sept 14 | 6.5 GiB if #1 booted; else prefix-caching arm if 2b passed; else **hold** | Boot boundary / caching correctness live | Commit run | Same as #1 |
| 3 | Sept 16-17 | Prefix caching + `--mamba-cache-mode align` (new attested dataset version with the `serving_setup.py` edit) | Live-legal on Kaggle; score in band | 2b passed AND rig wave >= base band | In band = keep for the stack; <2.0 = pull |
| 4 | Sept 19-20 | yield900 draw #4 (svid 347926973) — **flag: this is the one "redraw" and needs Ahmed's OK under his no-routine-resubmission rule** | The two fallback finals: yield900 mean - v4 mean >= 0.4 (today +0.74, Welch t = 1.21, n = 3 vs 3) | None | Feeds the sequential test for final selection if no release lands |
| 5 | Sept 23-25 | Restart-graft arm if its two-wave rig mean >= 45; else the composed serving stack if its rig read >= band; else v4 draw #4 | First live draw of the best surviving in-house arm | Step 4/5 rig gates | Draw 1 must land >= 2.85 (v4 mean); below = stop and diagnose, not redraw |
| 6 | Sept 27-28 | Second draw of whatever flew in #5 (sequential rule), or hold | Running-mean CI vs v4/yield900 | #5 in band | Continue or stop |
| 7 | Oct 1 (or the day the release is stageable) | Absorbed release, unmodified byte copy, draw 1 | Draw 1 >= 5.0 = not V14-class | Step 7 stage/diff/cost gate; runner read | >= 5.0 = draws 2-3 follow; < 4.0 = next release |

Sept 29-30: hold slots and quota for the absorption week. Slots not listed above (e.g. Sept 15, 18, 21-22, 26) are held unless a step-4/5 gate lands early; an idle slot is a lost draw, but a routine V14 redraw carries no information and Ahmed's rule forbids it.

---

## 8. What I did not adopt, and why

- **Absorption plan's total freeze of Sept work** — forfeits the $5 class-level diagnostics and the composable restart graft that all three judges wanted kept; the lane is capped at ~$60 so it cannot crowd out Oct 1.
- **Loop Shape step 2 (controller re-execution with surprise-halt)** — original, but inherits the patch-21 precedent (x1.40 actions, -25% levels) and is priced at +0.1-0.3; backlog as a $0 offline replay only.
- **Compute plan's lean-history graft and 4.0-4.6 target** — the 92% boilerplate figure did not reproduce (59-66% in two judges' recounts), the score elasticity 0.5-0.74 rests on two single draws with +/-0.2-0.3 noise, and the graft does not port to a foreign loop. Kept only the loop-agnostic serving riders.
- **Model Axis S2 training in September** — feasibility scored 4-5/10 by every judge; it collides with the Oct 1 quota reserve and the operator's attention. Kept the $15 S1 gate as optional and moved training to Oct 8+ conditional on no release.
- **Outsider plan's three parallel branches and its "cracked only cd82/dc22/ka59" sentence** — the sentence is wrong under either wall definition (budget passed dc22/sb26/vc33 of the 12 pre-registered walls); its instrument fix, turns-trigger and select-by-mean endgame are all adopted.