<!-- Produced by research workflow wf_613be1c1-a3e (83 agents, 3.5M tokens, 2026-07-26).
     Journal: ~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/0ba5f0b9-392e-404c-824f-ade3803cc524/subagents/workflows/wf_613be1c1-a3e/journal.jsonl -->

# ARC-AGI-3 Campaign Report — 2026-07-26

## 1. TL;DR

- **Plan verdict: SOUND-WITH-AMENDMENTS, two blocking defects.** The gate architecture (serve-verify → behavioral sweep → stop-lossed debut) survives audit, but as sequenced the A1 sweep is holdout-contaminated (run 8 trained on the holdout games) and the debut's serving path is unspecified — the exact failure mode that produced the "1.26 = base" retraction has no in-submission assert.
- **Biggest new finding:** arc-interactive (CONFIRMED-3-0) supplies ~249 MIT-licensed, offline-runnable ARC-AGI-3-format games — free, disjoint game supply that fixes the holdout-contamination problem and feeds SCoRe v2 without burning official dev games or dollars.
- **Top recommended experiment:** Prompt doctrine pack (anti-HUD field guide + K3 win-behavior playbook + guided ACTION7 probe-once), judge 8.2/VIABLE — $0, ~6-8 GPU-h, attacks the 27B's #1 verified killer, and prompt bytes cannot repeat the adapter-not-served disaster.
- **Biggest risk:** the arithmetic. SFT's own expected value is +0.05–0.07 against +0.50 needed for public 1.58; only a *stack* (SFT + prompt doctrine + serving retune + memory/token fixes, +0.30–0.40 combined) plausibly clears the private bar (~1.30–1.45). Scarce GPU is currently concentrated on the branch that cannot get there alone.
- **Updated win probability:** top-5 private with SFT-only: <5%. With the full stack executed and gates green: roughly 15–25% (private bar estimate is a model, not an observation — one strong-true-mean rival moves it). Milestone-2 money (>1.61 public by Sept 30) is a best-of-draws game our farming plan can partially play (E[max public] ~1.38–1.47).

## 2. Plan & Progress Assessment

**Auditor 1 — plan-soundness: SOUND-WITH-AMENDMENTS.** Three hard problems:

1. **Holdout contamination (blocking).** The A1 sweep selects run-8 checkpoints by wins on vc33/sc25/lp85 — games *in* run 8's 435-sample training corpus. As sequenced, the sweep measures memorization and the debut GO is invalid. The judge-amended holdout-excluded retrain, demoted in the handoff to "$0 anytime", must precede the sweep. (Better: use arc-interactive games as the holdout — see §4.)
2. **Deployment gap (blocking).** Serve-verify proves the merge in a *gate* kernel; the *scored* kernel's path is unspecified. The 20GB output cap blocks publishing the 55GB merged model, so debut must merge in-kernel at eval time — unbudgeted, against a notebook cap that is itself unresolved (6/8/9/12h conflict). No in-submission weight-delta/NLL assert exists. This is byte-for-byte the 1.26=base failure mode.
3. **Resource inversion.** The costed A/B (~36–48 GPU-h) exceeds the 30h/week pool; the "week-1 sequence" silently spans 2+ weeks. Meanwhile the universal floor-safe mean-raisers (stall detection, adaptive budget, serving tuning, anti-HUD lines) — which raise *every* config's mean including base — are unscheduled or last. Also: both sweep arms (ckpt-8, step-15) sit inside the literature's early band, so the sweep can't answer "does deeper help."

**Auditor 2 — ev-math: directionally sound, arithmetically short.** Pooled identical-bytes base mean is **0.98**, not 1.26 (1.26 is best-of-draws). 1.57-public-plausible requires true mean ~1.49–1.50 = +0.50; SFT's EV is +0.05–0.07 — a ~7x shortfall. The saving reframe: rivals' scores are equally draw-inflated, the 1.44–1.61 plateau is inert forks with true means ~1.0–1.26, Kojima regresses to 1.42–1.53 — so the effective *private* top-5 bar is ~1.30–1.45, reachable with +0.30–0.40 stacked. Other corrections: the "5 draws < base+0.05 → revert" stop-loss falsely kills a true +0.10 config ~23% of the time (widen to 7–8 draws); the corpus (62% L1) trains where marginal score value is lowest — depth beats breadth 3–10x per unit capability under level-index weighting and squared efficiency; slots should be spent as farm-plus-aggregate-measurement (30 draws resolve ±0.05), never single-draw inference.

**Auditor 3 — untapped-assets:** three revival-grade assets sitting idle: (1) the ~340-token mechanic field guide in `scratchpad/mechanics_compendium.md` §4, written, never A/B'd, judge-mandated; (2) adaptive per-game budget in `submission/_duck_retune/` — the un-impugned half of the shelved retune, fail-safe raise-only, worth up to 2x per-game wall time if eval=55 games; (3) Layer-2 fixes (`submission/_duck_fixes/duck_fixes.py`) built and tested, but **sub 54914567's score is recorded nowhere** — a free, blocking recovery action. Also documented-but-unbuilt: fp8 KV cache serving fix; the offline A/B rig (`rl_gate/run_rollout.py` + `trace_forensics/`) is the enabling asset for everything.

**Auditor 4 — state verification (verified claims + competitor scan): three contradictions with our recorded state.**

- The memory fact "no public fork has EVER touched serving/concurrency/sampling or ACTION7" is **stale**: at least two Jul-26 forks touch concurrency, sampling, and ACTION7 (see §3). Update the memory file.
- Tufa's "1.6002" is now explained: it is their official mean ±0.4475 over 25 *public* games, 20 tries each — consistent with our refutation of it as a hidden-LB figure. Their own diagnosis (context management + perception weakest) independently matches our game_over-wipe and token-estimator defect list.
- The dataset-version hazard stands: poisoned v1 and valid v2 share `arc3-sft-k3-ckpts` and both contain a checkpoint-8; a wrong-version mount silently reintroduces poisoned weights.

**What stands:** the gate discipline itself, run 8's validity (base NLL 0.911, val −12.5%), the $0 doctrine, the offline-A/B-first law, checkpoint selection by behavioral wins never loss. **What must amend:** retrain-before-sweep (or arc-interactive holdout), in-submission serving assert, eval-time-merge budget measured, stop-loss widened to 7–8 draws, select-2 override verified *before* debut, cheap mean-raisers scheduled in parallel not contingent, corpus re-weighted toward deep-level win-turns before SCoRe v2.

## 3. Competitive Picture

Changes since Jul 25:

- **Top-5 cutoff crept 1.57 → 1.58.** Kojima #1 at 1.86 (still zero public footprint, still best-of-48). **Tecnod8.AI is a new #2 at 1.61** (submitted Jul 26, no writeup found) — watch whether it holds or is a lucky draw. Plateau 1.46–1.61 is dense: a real +0.1 jumps ~10 ranks.
- **ericmao is running our campaign in parallel:** 7 LoRA adapters on HF (control, SFT v1–v3, GRPO, human100, spatial-continuation), all updated ~Jul 21. Critical caveat: all are on **text-only** Qwen3.6-27B, not the VL duck base — they either run a text-only duck or have a base-mismatch problem. The `human100` arm (human demonstrations as ToS-clean teacher data) is an idea worth stealing.
- **Our VL lane is publicly uncontested:** zero public adapters exist for Qwen3.6-VL-27B. First mover on a correctly-merged VL adapter still holds.
- **ACTION7 knowledge is going public:** wethepeople918's fork ships the mapping fix (and their "score-stability rollback" *stripped the prompt guidance* because "emphasizing it can encourage unnecessary probes" — independent reproduction of our unguided-ACTION7 own-goal finding, supporting our guided-probe framing). prvsiyan's fork adds the mapping fix plus an "evidence ledger" memory block. The compat fix alone will stop differentiating within weeks.
- **EWM-flavored and alternative-brain forks appearing** (obirdy's verified-world-model grafts; pranshubahadur's Gemma-4-26B NVFP4 vision-policy kernel). All unscored; we already measured EWM dead at 27B-class. React only if any scores >1.26.
- **Clock:** Milestone 2 closes 2026-09-30 ($25K/$10K/$2.5K, open-source required); ~66 daily slots to then, ~99 to Nov 2.
- **Unread but targeted:** arXiv 2605.25931 "Explore Before You Solve" — formalizes the exploration/efficiency tradeoff central to RHAE; directly relevant to stall-detection and probe budgeting. Worth one read.

## 4. Confirmed New Findings

1. **arc-interactive: 249 MIT-licensed offline ARC-AGI-3 games** — CONFIRMED-3-0. Count grep-verified (exactly 249 rows in GAMES.md), MIT license verbatim, runs fully offline from `environment_files/`, built on the official arcengine packages, 64x64 frames + ACTION1-7, disjoint from all official dev games. **Why it matters:** solves the holdout-contamination blocker and the SCoRe-v2 environment supply at $0 without burning dev games. Caveats: community quality varies (run the bundled solvability verifier first; lo05/BP01 had winnability bugs); a thin frames-to-duck adapter is needed (`scratchpad/arc3_adapter/` groundwork exists); community games skew toward conventional WASD semantics — fine for relative ckpt comparisons, mildly weaker for transfer claims. https://github.com/theredbluepill/arc-interactive
2. **schema-harness GPT-5.6 Sol trajectories: numbers real, corpus unusable as-is** — CONFIRMED-2-1 on the facts (24/25 wins, 182/183 levels, 95.35% mean RHAE self-scored, up to 10 levels deep), but the L2 applicability refutation is the operative finding: the traces are the residue of an execution-verified harness we already proved a 27B cannot emulate (12 EWM gate runs, 0 levels), 20/25 games show *zero* logged exploration (cloning them trains exploration out — the token-diet failure shape), coverage is maximal-memorization on public games, and the format is scaffold-alien. Selective salvage only: per-level optimal action counts as evaluation baselines; possibly shallow-segment mining after student-capability filtering. https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces
3. **That dataset has NO license of any kind** — CONFIRMED-3-0. No HF license tag, no LICENSE file, nothing in the README; the usable half is OpenAI model output. Any training use is blocked pending a license/ToS review and Ahmed's explicit call. The Claude half (25 traces, 98.98%) is ToS-forbidden regardless. https://huggingface.co/api/datasets/schema-harness/arc-agi-3-schema-traces
4. **Competitor rollback independently validates our guided-ACTION7 framing** (from the competitor scan, not 3-vote verified): wethepeople918 shipped ACTION7 prompt guidance, then stripped it for score stability while keeping the bare mapping fix — the same probe-spam mechanism that killed our unguided variant. Weak evidence (one unscored fork's revision history), but it is the only external data point on ACTION7 guidance and it points our way. https://www.kaggle.com/code/wethepeople918/agi-duck-harness-fast-eval-b25-score-optimized

## 5. Ranked Experiment Portfolio

All **eight** judged proposals came back VIABLE (no fatal flaws from any judge). Sorted by mean judge score; top 4 in detail, 5–8 compact.

**1. Prompt doctrine pack (anti-HUD field guide + K3 playbook + guided ACTION7 probe-once) — 8.2**
- *Mechanism:* inject the ready ~340-token field guide (HUD-is-state-not-reward, all-quantified wins, modal controls, free level-reset) + 9 K3 win-behaviors as imperatives + guided ACTION7 (probe-once budget, persistent ACTION7_SEMANTICS memory field). Targets HUD-as-reward anchoring, the 27B's #1 verified killer, with deaths concentrated at 9x-weight deep levels.
- *Gain:* +0.06–0.12 true mean; ceiling +0.3. *Cost:* ~1.5 days + 6–8 GPU-h sharing the A1 endpoint; ACTION7 is a config bit on the same arm.
- *Kill gate:* offline scorecard — first_board_changing_action median ≤2, HUD game_overs strictly down, levels-won ≥ base, actions/level rise ≤10%; ACTION7 sub-gate: ≤1 probe/game in control games, separately NO-GO-able.
- *Judge concern:* dev-to-hidden transfer risk (thinking-cap precedent: 2x dev → 0.73 hidden crater) and the game_over-count criterion is underpowered on ~8 games/1 run — add a second run or a count margin.

**2. Serving retune bundle (fp8-e4m3 KV + max-num-seqs 32 + prefix-caching strike + MTP spec=3) — 8.0**
- *Mechanism:* four flag-level fixes: fp8-e4m3 KV with checkpoint scales (never e5m2-default — corrupts uniform-color vision inputs, i.e. exactly ARC grids; never --calculate-kv-scales) roughly doubles the ~663K-token KV pool; cap scheduler admission; strike prefix caching (0% hits on the GDN hybrid, up to ~11% throughput cost); wake the dormant MTP head at spec=3 (spec=4 crashes the engine). More delivered turns per game inside the fixed 2.2h share; raises every config's mean including base.
- *Gain:* +0.06–0.14 combined; ceiling +0.25. *Cost:* 7–9 GPU-h, $0, zero scored slots.
- *Kill gate:* uniform-color-grid canary answer-exact vs bf16; /metrics preemptions down AND tok/s + TPOT up ≥8% on replayed traffic; prefix-strike non-inferior; MTP 2h soak, zero engine errors, zero temp-0 divergence — any instability = permanent NO-GO.
- *Judge concern:* the one "independent confirmation" of 28-lane thrash was REFUTED, so preemption severity is assumed, not evidenced — the floor case is near-zero if the duck is turn-limited rather than wall-clock-limited; MTP is the first arm to drop under time pressure.

**3. Death-safe two-tier memory (confirmed/guessed ledger surviving game_over) — 7.8**
- *Mechanism:* extend built FIX A: CONFIRMED mechanics (verified by observed transition) survive death and eviction; GUESSED hypotheses are demoted by the death that falsified them. Converts deep-level deaths from "restart learning" to "retry with knowledge." Hosts the ACTION7_SEMANTICS field.
- *Gain:* +0.04–0.08; ceiling +0.2. *Cost:* 1–2 days + 0 GPU-h for Stage-1 regression + 6–8 GPU-h as a third arm on the doctrine panel. First action free: recover sub 54914567's score.
- *Kill gate:* Stage 1 mock-replay byte-identical on death-free traces; Stage 2 on death-observed holdout games: post-death actions-to-previous-depth down ≥25%, levels-won ≥ base, NO-GO on any death-free-game regression.
- *Judge concern:* the "confirmed-vs-guessed cleared 182/183 levels" attribution is unverified frontier-harness lore, and the cited ReplayMockLLM rig does not exist in the repo — Stage 1 harness must be built first.

**4. Token-budget refit (image-aware estimator + analyzer window 32768→~49152) — 7.8**
- *Mechanism:* built FIX B kills the len/3 estimator's 3.7–4.0x image overcount (~29% phantom budget evicting real history); conditional on the serving bundle's GO, raise the analyzer window into the freed KV headroom for ~50% more true history per turn. Fewer repeated probes of known mechanics, rewarded by squared efficiency.
- *Gain:* +0.03–0.08; ceiling +0.15. *Cost:* estimator ~$0 (CPU regression, rides any kernel); window raise ~3 GPU-h on the serving session.
- *Kill gate:* replay byte-identical with patch; retained-real-history/turn up ≥25%, zero context-overflow errors at the raised window; scorecard non-inferiority. On failure: ship estimator alone, keep 32768.
- *Judge concern:* "fp8 KV doubles the pool" overstates for a hybrid GDN model (only attention layers hold KV); window raise is strictly hostage to the serving bundle's GO.

**5. Depth-reweighted holdout-clean retrain (corpus v2a) + BFS optimal-path augmentation — 7.7**
- *Mechanism:* re-sample the 435 K3 samples with weight ∝ level index (corpus is 62% L1 while the metric pays level L at L× with squared efficiency), de-dup near-identical L1 turns, and augment with provably-optimal win trajectories harvested $0 from arc-interactive's mechanical BFS solver (patch `engine_bfs_single_level` to record action paths) — exact shortest-path exemplars teaching squared-efficiency behavior no LLM teacher demonstrates reliably. Also converts the A1 sweep from invalid to valid (holdout-clean).
- *Gain:* +0.08–0.12 true mean (vs +0.05–0.07 for the current shallow corpus); ceiling +0.35. *Cost:* 2–3 days CPU + ~4–6 GPU-h retrain (run-8 recipe).
- *Kill gate:* corpus gate (≥35% samples at L≥2, ≥60 BFS-optimal trajectories over ≥10 games, else don't train); behavioral gate (beat run-8's best checkpoint by ≥2 level-wins at L≥2 on holdout-excluded games, no first-strike regression).

**6. Stall-detection + progress-weighted budget donation — 7.5**
- *Mechanism:* stall classifier (zero level progress AND saturated frame-hash set AND no score delta over N turns) + raise-only wall-clock donation: stalled games keep a reduced heartbeat (never killed — single play preserved), surrendered hours extend the deadlines of games actively clearing levels. Deepening one game 2 levels at half human efficiency ≈ +0.30 LB.
- *Gain:* +0.10–0.15 mean; ceiling +0.35 (plateau-escape class); raises base AND tuned configs. *Cost:* 2–3 days build + 8–10 GPU-h calibration/A/B.
- *Kill gate:* false-hopeless rate <5% on replays of games the duck eventually progressed in; 2-arm A/B ≥1 net additional level with ZERO levels lost on any game — either fails → ship stall-floor only.

**7. Multi-action plan commits with surprise-abort (predict-commit-verify) — 7.3**
- *Mechanism:* each duck turn pays a multi-thousand-token prefill to emit 1–2 actions. Once a game's world-model block matures, have the 27B commit `[[action,x,y],...]` + one-line predicted effect per action; a ~30-line harness shim executes until the first mispredicted frame transition, then aborts and re-models (design pattern from the GPT-5.6 schema traces — ls20 7/7 levels at 533 actions over 106 LLM turns; scaffold ported, no trace data used).
- *Gain:* +0.05–0.12; ceiling +0.30 if turn-starvation is the binding mid-game constraint. *Cost:* ~2 days + 6–10 GPU-h A/B.
- *Kill gate:* instrumentation PRE-gate — if prefill <40% of per-action cost, cancel the build; then A/B: levels/wall-clock-hour +≥20%, per-level RHAE non-inferior, actions-per-turn ≥2x in mature-model games.

**8. probe_components(): scripted per-component click-sweep as a duck REPL tool — 7.3**
- *Mechanism:* deterministic Python tool called once per new level/screen: segment the frame into connected components, click each centroid in one engine-speed batch (~34 probes, ACTION7 excluded), diff frames, return a ground-truth affordance table straight into the world-model memory block. This is the online-probing approach that empirically DOMINATED the dead learned click predictors; probes are spent at level-1 weight while the knowledge pays at deep-level weight.
- *Gain:* +0.05–0.10; ceiling +0.3 if it unlocks first-level chains on CLICK-gated games the duck currently zeroes. *Cost:* 2–3 days CPU + 6–8 GPU-h A/B.
- *Kill gate:* CLICK-heavy dev panel A/B — GO iff probe arm clears strictly more total level-weight AND median RHAE on shared-cleared levels drops <10%; harness must be a byte-level no-op when the tool is never called.

**Killed / not funded**
- *Schema-traces as SFT teacher corpus* — killed at verification (L2 applicability refutation: supervises a harness, not a policy; plus no license; plus half is ToS-radioactive).
- *Copying wethepeople918's 4-lane concurrency cap* — killed by arithmetic: 4 lanes x 9h ≈ 16 game-slots < ~28 games, unplayed = 0; consistent with that team's 0.34.
- *Standing dead list unchanged* — best-of-N/shadow/banking, local-brain EWM, Claude distillation, unguided ACTION7, thinking caps, learned no-op predictors, appearance-based affordances, kernel self-mount resume. No new proposals resurrected any of these.

## 6. Recommended Amended Plan

**Jul 27–31 (pre-reset, GPU blocked — $0 CPU/browser week):**
- Ahmed browser actions (blocking, ~15 min total): recover **sub 54914567's score** from the Kaggle UI and record it; verify the **select-2 manual override** exists; read the rules page to resolve the **notebook wall-clock** (6/8/9/12h conflict). All three gate later decisions; none costs GPU.
- Build: in-submission **weight-delta/NLL assert** (a silently-unserved adapter can never score again); prompt doctrine pack text + ACTION7_SEMANTICS plumbing; two-tier ledger Stage-1 mock harness; adapter for arc-interactive → duck frames (extend `scratchpad/arc3_adapter/`).
- Curate: run arc-interactive's solvability verifier, pick ~12–16 clean games spanning mechanic families → **new uncontaminated behavioral holdout** (kills the retrain-before-sweep dependency: run 8 never saw these games). Keep vc33/sc25/lp85 only as a memorization probe, never a selection signal.
- Jul-27 farm slot stays NOT ARMED per standing instruction; resume 1/day base farming Jul 28–31 if desired (ev-math: ~6 base draws sharpen the baseline mean estimate to SE ~0.06).

**Aug week 1 (quota resets Aug 1; budget ≤30h):**
- Serve-verify v2 (~2.5h) including a timed **eval-time in-kernel merge rehearsal** — measure the merge's wall-clock cost against the now-known notebook cap before any debut design is locked.
- A1 checkpoint sweep on the arc-interactive holdout: 3 arms (base, ckpt-8, step-15) x ~8 games ≈ 18h at the 45min/game estimate — **instrument per-game wall-clock in the first kernel and re-budget**, since 45min is an assumption. Selection by behavioral wins with a pre-registered minimum win-margin; note the sweep cannot answer "does deeper help" (both arms are early-band) — it only picks the debut adapter.
- Remaining ~9h: prompt-doctrine A/B rides the same vLLM endpoint (2 arms, cross-tested against the *selected* adapter, not base, so the shipped combination is what was tested).

**Aug week 2:**
- Serving retune bundle (7–9h): canary + dual-metric gate + prefix strike + MTP soak (drop MTP first under pressure). Token-window raise (~3h) piggybacks iff serving GOes; estimator fix ships regardless.
- Death-safe ledger Stage-2 A/B as a third arm if week-1 GPU allowed pre-work; else here (6–8h).
- Fold adaptive per-game budget + stall bound + Layer-2 fixes into the debut kernel (fail-safe, zero GPU to include).

**Aug week 3 — gated debut:**
- One stacked config: selected adapter (if sweep GOed) + doctrine pack + serving retune + estimator/ledger fixes + adaptive budget, with the in-submission serving assert. Stop-loss: **7–8 draws** (not 5) before revert — the 5-draw rule false-kills a true +0.10 config ~23% of the time. If the sweep NO-GOes, debut the no-adapter stack anyway — it raises the base mean and is the fallback track the audit demanded.
- Do not debut anything before the select-2 override is confirmed. A fluky draw of a bad-mean config must be evictable from the final pair.

**Sept (milestone 2 closes Sept 30):**
- SCoRe v2 corpus, re-weighted **toward deep-level win-turns** (mechanics-compendium §3 targets: ft09 L5, ar25 L6, sk48, lf52-class depth) — teacher spend is Ahmed-gated; the arc-interactive supply provides student environments at $0. Retrain + re-sweep on the uncontaminated holdout (~20–24h across two weeks).
- Farm the best-validated config daily: double duty — public max chases milestone-2 (>1.61 needed; E[max] climbs toward 1.38–1.47) and every draw tightens the true-mean estimate (30 draws → SE ~0.03). Read arXiv 2605.25931 for probe-budgeting refinements before committing further stall-detection design.

**Oct–Nov (final, ~99 slots total from today):**
- No new unvalidated configs after mid-October — the select-2 winner's-curse risk dominates any upside. All remaining slots farm the best true-mean config. Lock the final pair by validated true mean, not public draw rank. Publish CC0/MIT-0 ahead of deadline.

**Standing rules:** every effect < ±0.3 decides offline, never by scored draws; pooled base mean is 0.98, so quote uplifts against that, not 1.26; verify the ckpts-dataset version string on every mount (v1 is poisoned and shares a checkpoint-8 filename with v2).

## 7. Refuted / Do-Not-Cite (new items only)

- **"wethepeople918 fork independently confirms 28-lane oversubscription degrades the baseline" — REFUTED (2 of 3 lenses).** The "repeated analyzer timeouts" exist only as a code comment; the observable timeouts occurred *at 4 lanes* under a slashed 1500s budget and are end-of-budget artifacts; the fork changes sampling/tool-steps/grafts so nothing isolates concurrency; the author's team best is 0.34 over 42 subs. The concurrency lever stays alive on our own KV-math grounds only. Never copy the 4-lane config.
- **"No public fork has ever touched serving/concurrency/sampling or ACTION7" — STALE, retire from memory.** Two Jul-26 forks touch all of these. The differentiator is now *doing it correctly with offline gates*, not doing it at all.
- **Schema-traces as a teacher corpus for our student — do not cite as an SFT asset.** Facts confirmed, applicability refuted (harness residue, zero-exploration traces, no license, half Claude-generated). Permissible uses only: per-level action-count baselines; possible shallow-segment salvage after license review and Ahmed's call.
- **Tufa "1.6002" — now provenance-resolved, still do-not-cite as a hidden-LB number.** It is their public-25-games, 20-tries figure (±0.4475); citing it as a hidden-set benchmark remains wrong.