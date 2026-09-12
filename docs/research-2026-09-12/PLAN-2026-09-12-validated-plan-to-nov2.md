# Plan 2026-09-12 → Nov 2, built on the validated record

Inputs: the validation audit (`R-validation-audit-0912.md`), five independent strategist plans and three
judge scorecards (`plans-and-judges-raw.json`), the workflow synthesis (`SYNTHESIS-workflow-plan-raw.md`)
and the completeness critique (`CRITIQUE-workflow-completeness-raw.md`). This document is the synthesis
with the critique's twelve gaps applied. Judges' totals: Absorption 112, Loop shape 110, Outsider 104,
Compute 102, Model axis 97; two of three ranked Absorption first.

## 1. Verdict in one paragraph

We sit at ~3.2 because we fly the best public artifact (a byte copy of keith V14) and nothing public
scores higher. Every in-house lever since 09-02 reshaped behaviour without moving the per-game
comprehension distribution: the leaders look like "our brain on its best draw of every game" (single draw
38.5 levels; E[max of 3 draws] 50.8; union of 10 draws 64), and the 3x-clock wave passed only 3 of the 12
never-passed walls. The harness side therefore tops out at about +0.3 to +1.0 LB; the model axis was
closed on wrong numbers but stays a 10-15 % shot that collides with the calendar; the only route to the
7+ tier is absorbing a Milestone-2 release on Oct 1. So September is the preparation month for Oct 1,
with a capped in-house lane of cheap composable grafts, and October is absorption plus selection by mean.

Honest targets: in-house only 3.3-4.0; with a Kaggle-runnable release from a ≥7 team absorbed Oct 1-3,
6-9 (heavy-resubmitter max-statistics mean the copy may read 5-6); probability-weighted ≈4.5-5.
P(absorbable release) is closer to P(Tufa releases a runnable kernel) than to "any ≥7 team": only the
top-3 public at Sept 30 have the milestone-prize incentive, and in Milestone 1 only the #1 release was
absorbable.

## 2. Stop doing

| stop | why |
|---|---|
| behaviour-shaping prompt grafts (probe/carry/evid/hypo/fx) | three on-instrument replications flat; closed for step-sized effects |
| window-shrink / reasoning-strip arms | 12 / 21 / 20 levels; retained history is load-bearing |
| executable-world-model ports (A2, Stage-1, Polyphony, NOOA) | one verified model = 88-89 % of a stock game's decode; 0-45 % next-level transfer |
| per-game crack solvers | hidden games differ; deployable inventory was 1 |
| single-wave "dead" verdicts on 31-46 levels | at sd 3.5 a true +4-level lever reads dead 62 % of the time |
| routine V14 redraws "to read the band" | the band is known (2.0-4.7); every slot must test a hypothesis or bank a final-candidate draw |
| silent waivers of instrument faults | budget 3x and ctx12 both had mid-wave container replacements never mentioned |
| 3x-clock waves as a level instrument | not live-legal; keep only as a trajectory generator |

## 3. Decisions that need Ahmed (blocking)

1. **Verdict rule.** Adopt the 8-draw flat mean 38.5 / sd 3.5, two-wave rule (mean ≥45 step candidate,
   ≤42 dead, mandatory counterbalanced second draw for any ENGAGED arm reading 31-46), and the rewritten
   VOID rule (5xx, process restart, preemptions/request outside 0.02-0.35, shim span < 0.99 × clock).
   Then edit `offkaggle/run_regime_wave.py:313-315` and `offkaggle/read_cadence_arm.py:17-19`.
2. **Restart-at-the-wall graft (step 4).** STATUS:432-434 says not to spend on it without an explicit go.
   It also needs a code change first: `graft_retry.py:411,454` always issues RESET; a no-RESET branch
   does not exist yet.
3. **yield900 draw #4** as a fallback-final sizing draw (conflicts with the no-routine-resubmission rule).
4. **Model-axis serving gate (step 6, $15, optional).**

## 4. Ordered steps

**Step 0 — write-backs and instrument ($0, done/pending).** Audit written; STATUS corrections appended;
ledger row 56133282 recorded; checklist cost gate fixed to 73.6-74.2k tokens/game; memory updated.
Pending Ahmed's decision 1: the runner constants. Still to add: ledger row 55543514; a license gate at the
checklist's H+0 step (milestone releases must be CC0/MIT-0 to claim the prize, but a GitHub-only release
can ship under anything, and our own eligibility requires open-sourcing what we fly).

**Step 1 — release watch ($0, ~4 h).** Daily diff of `kaggle kernels list` / `kaggle datasets list` /
`kaggle models list` over the 26 member accounts of the top-20 teams, plus Hugging Face (a fine-tuned
checkpoint from NVARC3 or Franzen would land there), the Tufa and NVIDIA GitHub orgs, sonpham-org/arc-3,
and the forum. Test: it flags the 09-01 jakobbrggen release and the 08-29 sonpham datasets as new vs
baseline.

**Step 2 — two $0-5 diagnostics before any wave.**
2a. Oracle probe on Flash-Next (all three judges' "best idea to keep"): can the served brain ACT on a
correct model it did not build? Before porting `submission/_oracle_probe/oracle_probe.py` to Modal, read
its own history: `results*.json` show 0 wins in 24 runs across control / oracle / *guided* on the 8B and
14B, every run ending at exactly 3 blocks left — which points at the harness/action channel, not the
brain — and the 27B v2 kernel result (08-29) was never recorded: `kaggle kernels output
ahmedmobasher86/arc3-oracle-probe` first ($0). Only then port. Test: ≥3/5 ORACLE clears of wa30 L3
within 100 moves, CONTROL ≤1/5. Kill: ORACLE <50 % ⇒ every "harness supplies dynamics, model plans"
design is dead on this brain. Also define the success branch: if ORACLE passes, the effect-table planner
(harness-run search over `graft_effects`' table) becomes step 5's first graft.
2b. Prefix caching — but NOT alongside MTP. Standing law (`docs/RESEARCH-2026-08-21-bug-lever-hunt.md`,
memory bug-lever-hunt): MTP ships only with `--no-enable-prefix-caching`; MTP + prefix cache corrupts on
the GDN layers. The real arm is caching WITHOUT MTP (losing 2.80 tokens/forward): boot with
`--mamba-cache-mode align`, 20 recorded prompts at temperature 0, off/on/on. Test: boots; ≥19/20
byte-identical; hit rate ≥30 % on the 3-game smoke; net calls/game vs the MTP profile ≥ +10 %. Kill: any
of those fails ⇒ caching closed for this build.

**Step 3 — bundle-agnostic rig runner + serving-transplant kit (Sept 13-26, ~$27, ~3 days). The spine.**
`run_regime_wave.py` is pinned to the june_stock tree and the taaf pickles (:9, :106-110, :576-586,
:2286-2291), so a foreign bundle cannot be measured on Modal today and the checklist's H+1..H+6 step is
not executable. Build a runner that takes any released bundle (`deploy_target.pkl` + `src/` +
`setup_commands.json`) and runs it in Kaggle geometry; validate on the V14 bundle (must read 34-41
levels) and jakobbrggen's avo-v2 bundle (must run unmodified). Package the keith V14 serving profile
(RadixArk NVFP4, kv5-bf16-mtp3-c8-cg32) as a drop-in under whatever vLLM stack a release ships, and
TIME its commit run against the 9 h box (organizer-confirmed; ~12 min margin at 110 games; a 135 GB
model load vs a 27B's 31 GB eats that margin). Kill: not runnable unmodified by Sept 26 ⇒ pre-write
the 09-02 hand-port recipe and keep only the transplant kit.

**Step 4 — full-history-wipe restart at the wall (Sept 14-22, ≤$25; needs decision 2).** Add a no-RESET
branch to `graft_retry.fire()`; trigger on turns-on-the-same-level (~10-12), max 1 wipe/level, 2/game;
post-wipe prompt carries a compact factual ledger (actions tried, observed effects, refuted hypotheses).
Why: per-wall pass probability is 0.12-0.62 per draw and passes come early; E[max of 2] ceiling is
+0.33 lv/game on the 9 HIGHVAR games; the 16 LOWVAR/CAPPED games give nothing. Kill test ($2.5, six
games × 2 draws, conc 3): wipes fire in ≥5/6 runs; the FIRST post-wipe call's acting rate ≥ base
(verifier 2 measured zero-history calls acting 0.0 % — a wipe manufactures that shape); ≥2 wiped
levels clear afterwards (note-only dose: 0/5); read levels AND per-game score (RESET and re-exploration
actions accumulate; the efficiency term is quadratic). $0 first: check in
`scratchpad/human_replays/index_by_game.json` whether humans who clear the 12 walls use resets/retries.
Waves: two-wave mean ≤42 or ≤2/12 walls ⇒ dead; ≥45 and ≥3/12 ⇒ step candidate → 3 live draws.
LB +0.2 to +0.5 if it converts; P(step) ~15 %.

**Step 5 — loop-agnostic serving riders, flown only as a stack (Sept 13-24, ≤$12 + ~6 h quota).**
(a) KV 6.0 then 6.5 GiB in a Kaggle COMMIT run of the byte copy (boundary is Kaggle-only: 81.8 GiB load
there vs 79.4 on Modal; 8 and 10 OOMed, 6-6.5 never tried). (b) MTP-5 rig smoke reading acceptance at
positions 4-5 (kill: engine refuses nst>3 or <0.35). (c) capture the fp8-KV boot traceback ($0.5).
(d) if 2b passed, one rig wave of no-MTP prefix caching with actions/call and reasoning-chars gates.
Each is +6-16 % calls (calls ∝ KV^0.56), undetectable alone, but they compose with any absorbed kernel.
Do NOT build the lean-history graft (the 92 % boilerplate figure did not reproduce: 59-66 %) or the
controller re-execution arm (patch-21 precedent) before Oct 7.

**Step 6 — optional $15 model-axis serving gate (Sept 23-30, only if step 3 is done by Sept 22).**
Round-trip the served RadixArk NVFP4 checkpoint's bf16 attention tensors (identity and 1e-6-perturbed),
experts byte-identical, serve on the rig profile, 3-game smoke, one Kaggle commit with `model_sources`
swapped. No training before Oct 8; training (ms-swift 8×H100 recipe, ~$130/attempt, $300 cap) only if
no release lands and the gate passed. P(step) 10-15 %.

**Step 7 — Oct 1-3: byte-absorb the strongest Kaggle-runnable release, unmodified (3 slots, ~27 h quota).**
`docs/TRACK-D-absorption-checklist.md` as written plus the license gate: stage / diff / cost gate
(completion tokens per game ≤ ~1.3 × 74k) / rig read on the new runner the same day / push unmodified
with Ahmed's go. Priority: Tufa > NVARC3 (8.40) > Third Intelligence (8.21) > Franzen > Fususu. Kill:
draw 1 <4.0 ⇒ next release; 4.0 ≤ draw 1 < 5.0 ⇒ one more draw before deciding (a foreign config's
per-draw sd may be 1.0, at which a true 5.0 reads <4.0 16 % of the time); ≥5.0 ⇒ draws 2-3. No release
≥5.0 by Oct 8 ⇒ revert to the V14 family and say so.

**Step 8 — Oct 4-20: compose only what is measured.** If the release runs a 27B/31B model, lay the
Flash-Next transplant under it (two counterbalanced rig waves; bar = copy's two-wave mean + 6; then 3 live
draws). If it already runs Flash-Next, add only surviving in-memory grafts from steps 4-5, one at a
time. Kill: two-wave mean below copy + 3, length-finish >1 %, or completion tokens/game >1.5× ⇒ fly the
unmodified copy.

**Step 9 — Oct 8 → Nov 2: endgame by config mean.** Every slot draws a surviving config (v4, yield900,
best composed arm, absorbed copy, absorbed + grafts), read sequentially (continue until the running-mean
95 % CI separates from the current best, cap 5 draws per config). Select the two final entries as two
draws of the config with the highest MEAN, never the public max. The private score is fixed at scoring
time with no end-of-competition rerun (organizer-confirmed, memory kaggle-submission-contract §1-2).

## 5. Calendar and quota (Kaggle week resets Saturday; Oct 1 is a Thursday)

| window | spine | in-house lane (cap ~$60 rig, ≤10 h quota before Oct 1) |
|---|---|---|
| Sept 12-13 | step 0 write-backs; step 1 watch | step 2 diagnostics ($5) |
| Sept 13-20 | step 3 runner + transplant kit | step 4 kill test then wave 1; step 5 KV 6.0/6.5 commits, MTP-5 smoke |
| Sept 21-25 | step 3 validation (V14 + avo-v2) | step 4 wave 2 if alive; prefix wave if 2b passed; composed stack read |
| Sat Sept 26 → Fri Oct 2 | reserve week: ≥30 h quota untouched; hand-port recipe pre-written; step 6 only if idle | no fresh commits (re-flying an existing svid costs no quota) |
| Oct 1-3 | step 7 absorb + fly | — |
| Oct 3 (new quota week) → Oct 20 | step 8 compose | LoRA training only if no release and step 6 passed |
| Oct 8 → Nov 2 | step 9 selection | — |

## 6. Next 7 submissions (each needs Ahmed's go; an ERROR costs no slot)

| # | target | config | hypothesis | read |
|---|---|---|---|---|
| 1 | Sept 13 | byte copy + KV 6.0 GiB (push_kv10 mechanism) | boots on the Kaggle card; score in 2.0-4.7 | ERROR = KV headroom dead; in band = alive, banked |
| 2 | Sept 14 | 6.5 GiB if #1 booted; else hold | boot boundary | same |
| 3 | Sept 16-17 | no-MTP prefix caching (`--mamba-cache-mode align`, new attested dataset version) | live-legal; in band | only if 2b passed AND rig wave ≥ band |
| 4 | Sept 19-20 | yield900 draw #4 (svid 347926973) — needs decision 3 | yield900 mean − v4 mean ≥ 0.4 (today +0.74, Welch t 1.21, 3 vs 3) | feeds final selection if no release |
| 5 | Sept 23-25 | restart graft if its two-wave rig mean ≥45; else composed serving stack if ≥ band; else v4 draw | first live draw of the best surviving in-house arm | must land ≥2.85 (v4 mean); below = stop and diagnose |
| 6 | Sept 27-28 | second draw of #5 (existing svid, no quota) or hold | running-mean CI vs v4/yield900 | continue or stop |
| 7 | Oct 1 (or when stageable) | absorbed release, unmodified byte copy, draw 1 | draw 1 ≥5.0 = not V14-class | ≥5.0 → draws 2-3; 4.0-5.0 → one more; <4.0 → next release |

Unlisted slots are held: a routine V14 redraw carries no information.

## 7. Not adopted, and why

Absorption plan's total September freeze (forfeits $5 diagnostics and the composable restart graft);
Loop-shape's controller re-execution with surprise halt (patch-21 precedent; backlog as $0 offline
replay); Compute plan's lean-history graft and 4.0-4.6 target (boilerplate share did not reproduce; score
elasticity rests on two single draws ± 0.2-0.3); Model-axis training in September (feasibility 4-5/10 by
every judge; collides with the Oct 1 reserve); Outsider's three parallel branches and its "cracked only
cd82/dc22/ka59" sentence (wrong: the 3x wave passed dc22/sb26/vc33 of the 12 walls) — its instrument fix,
turns trigger and select-by-mean endgame are all adopted.

## 8. Known and unclosable

Hidden games expose `tags` only inside the scored rerun and no channel exists out of one, so the rig-to-live
mapping stays a two-point line (≈2.0-2.2 LB per rig level/game); no step here tightens it. Every graft is
tuned and read on the same 25 public games (effective n ≈ 25); the arc-interactive 249-game corpus and the
340 human replays are unused out-of-sample instruments.
