# Revised plan 2026-09-08 — from 3.2 to the 7–11 tier

Inputs (all written today, memory-blind where marked): `docs/research-2026-09-08/R-field-refresh-0908.md` (field, memory-blind), `R-model-axis-0908.md` (models, memory-blind), `R-score-arithmetic-0908.md` (our own 75 rig rows vs the official scorer), `R-reopen-audit-0908.md` (every kill in our history re-examined), plus `R-loss-ledger-3.md` and the probe-discipline result (`offkaggle/REGIME_WAVE_STATUS.md`). Deadline Nov 2; Milestone 2 Sept 30 (public LB, open-sourcing required for the prize).

## 1. Where we stand (facts)

| | |
|---|---|
| Our live family (four byte-equivalent draws) | 3.25 / 2.58 / 4.31 / 2.45 → mean **3.15**, sd 0.85; rank 26 on the max |
| Best public notebook | 4.33, a byte-identical copy of what we fly; nothing public ≥ 5 |
| Top | Tufa 11.04 (132 entries), Third Intelligence 8.21, Franzen 7.63, mostik.ai 7.51, **NVARC3 5.96 in 3 entries** |
| Final ranking | private set, two selected submissions → the config's **mean** is what counts, not the public max |
| Rig (Modal, exact regime) | 25-game total 39.3 ± 2.3 levels; 52 calls/game at 150 s; 60 % of every game's clock is spent stuck at one wall with a baseline-sized action budget |

## 2. What the arithmetic says the lift must be (R-score-arithmetic)

- 98 % of our lost points are **levels not reached**. Efficiency on completed levels is already at the cap (median 0.86 × baseline actions); total efficiency headroom ≈ 0.8 LB points, ever.
- Marginal values on the public 25 (LB-equivalent at the 0.40 calibration): **+1 level per game ≈ +2.8 LB**; L1 on every zero-level game ≈ +0.2; perfect efficiency ≈ +0.8. Efficiency must be *protected* (2 × baseline actions multiplies the score by 0.25), not pursued.
- A **7.0** is our public-25 play (1.5–1.8 levels/game at near-baseline actions) generalised to all 110 games; today the 85 unseen games imply ≈ 1.7/game. An **11.0** is ≈ 2.0–2.2 levels/game at near-baseline actions.
- Under the current cadence (52 calls × 3 actions), perfect comprehension caps at ≈ 2.9 levels/game ≈ 9 LB. Comprehension is binding today; cadence becomes binding only past ≈ 9.

## 3. What the field says the top tier does (R-field-refresh)

Three independent, public, dated sources point at the same mechanism, and it is exactly what the stock duck does *not* do:

1. **ARC Prize, Sept 3:** GPT-6 Astra 62.7 % → **99.9 %** on the same games with a harness whose only change is *preserving reasoning state between requests + compaction*. Astra's replays: compact symbolic world-model notes in a self-invented notation, and (in a sandbox) a per-game toolkit it writes itself — parser, state model, search, planner, persistent notes.
2. **NVIDIA's open agents** (DreamTeam, NOOA — Apache-2, vLLM-ready): an *executable, verified world model* that persists; their authors are on NVARC3, the team with 5.96 in three draws.
3. **Polyphony** (open, Qwen3.6-27B): "grows a verified per-game heuristic system as executable Python files" — 19.8 % on the community board where the duck lineage reads ≈ 7–8.
4. Tufa's own writeup names the two levers they had not finished: **context compaction/memory and perception**.

The stock duck, verified in `scratchpad/bundles/june_stock/.../tool_agent.py` and `python_tool_sandbox.py`: the sandbox rebuilds its globals in a fresh subprocess **every call**; the Qwen chat template drops earlier-turn reasoning, so at 1.7 calls/turn the model's thoughts are wiped roughly every second call; cross-turn memory is a one-line "World model:" note plus eviction of the oldest turns. Nothing the model computes or infers survives, except what it re-types.

## 4. What our own history says (R-reopen-audit)

- Behaviour-shaping levers on this loop are **closed** (patch 21, yield900, probe discipline: engaged and flat on the exact instrument; retry/evidence/hypothesis/upscale/half-concurrency dead).
- The **executable-world-model / protocol lane was killed on the wrong brain**: every failure (EWM gates, Tycho-27B, Stage-0) was thinking non-termination on Qwen3.6/3.8-27B. Flash-Next terminates (length-finish 0.4 %) and won ft09 6/6 on the rig. The kill does not transfer.
- The **model axis was marked closed on a confounded comparison**; the only live step the campaign ever produced (1.9 → 3.2) *was* a model swap under equal analyzer regimes. But R-model-axis finds nothing ≤ 85 GiB that beats Flash-Next per call; the only mechanistic alternative is Qwen3.8-27B (10 × KV, ≈ 1.5 × throughput, weaker per call).
- Never built: persistent workspace + transition log + backtest + commit-with-halt (TP12), the park-walled-games scheduler, the oracle-model probe on Flash-Next.

## 5. The thesis

**The agent is not action-starved or time-starved at the wall; it is model-starved.** It has enough actions (median 1.0 × baseline at the wall) and 60 % of the clock, and it fails because nothing it learns persists in a form it can query or execute. Every public 7–11 signal is a version of "make the game model persistent, verified and executable". That lever is orthogonal to everything we have killed, it composes with whatever gets open-sourced on Oct 1, and our own kill of it was on a brain that could not finish a thought.

## 6. Program (three tracks, sequenced by information per dollar)

Gates, unchanged: a 25-game rig wave vs pooled base **39.3 (sd 2.3)**: ≥ 48 → step candidate → counterbalanced redraw → live 3-draw rule (mean ≥ 4.0 adopt). New **fit-the-clock gate**: calls/game × measured e2e ≤ 7,920 s in Kaggle geometry on the rig, else the arm is not live-eligible whatever it scores. Kill tests first: 3 games × 2 draws (≈ $3) before any 25-game wave (≈ $9).

### Track A — persistent, verified game knowledge (the thesis; start now)

| id | arm | what it changes | build | first read |
|---|---|---|---|---|
| A1 | **Carry + compact** (graft on duck) | keep the model's own reasoning/notes across turns instead of dropping them; replace oldest-turn eviction with a model-written compaction of older turns (the Astra provider-adapter condition, done for an open model) | 2–3 days: graft in `submission/_throughput_v1/` convention; verify on the rig prompt logs that prior-turn reasoning actually reaches the model (Qwen3.8 template) | 3-game kill test, then 25 games; read levels, wall passes among the 12 never-passed walls, prompt tokens/call (must stay ≤ 32k) |
| A2 | **Persistent workspace** (TP12, never built) | sandbox globals survive across calls within a game; a queryable transition log (action, before, after, diff); files the model writes persist (its own parsers/solvers); `backtest(hypothesis)` over the log | 3–4 days | same; plus "files written per game" and "backtests run" as engagement counters |
| A3 | **NOOA on Flash-Next** (NVIDIA-NeMo/labs-OO-Agents, Apache-2) | a different loop entirely: object-oriented CodeAct agents with an executable world model | 1–2 days port to our Modal endpoint + Kaggle geometry (calls/game budget is the risk) | 3-game kill test with the fit-the-clock gate first |
| A4 | **Polyphony on Flash-Next** (open) | grown per-game heuristic system as Python files | 1–2 days | same |

A1 and A2 compose (A2 is where A1's compacted knowledge becomes executable). A3/A4 are alternatives to the duck; if either passes the clock gate and reads ≥ 45, it becomes the base to graft onto.

### Track B — throughput/model axis (cheap, parallel, low prior)

| id | arm | rationale | cost |
|---|---|---|---|
| B1 | **Qwen3.8-27B NVFP4 under the V14 analyzer regime, large KV, high concurrency** | the only model with a mechanism (kv10 wave = 43 levels, the max we ever drew, from 1.6 × calls/game); weaker per call | one wave, $9 |
| B2 | Flash-Next + prefix caching on vLLM ≥ 0.27 with PR #51113, MTP kept | if the hit rate holds, calls get cheaper → more calls/game; must measure accuracy | one wave, $9 |

B matters only once comprehension improves (cadence binds past ≈ 9 LB), so B1/B2 run in idle rig sessions, not ahead of A.

### Track C — behaviour-clone our own winning trajectories (STaR LoRA)

One public LB-measured lift (1.25 → 1.94, n = 1, noisy, base-model-specific). We hold 150+ scored Flash-Next trajectories (six rig waves + two Kaggle commits) and can filter to level-completing turns. Cost: fine-tune a 125B MoE off-box (Modal multi-GPU), re-quantise NVFP4 with ModelOpt, serve, verify identity — 1–2 weeks and real money, and no fine-tune has ever been served + scored in this campaign. **Decision rule:** start C only if Track A's first two reads (A1, A2) are ≥ 45 (so the adapter trains on the improved loop's trajectories) or if A is flat by Sept 18. If B1 shows the 27B within reach, C on the 27B is a one-GPU job (`arcagi3-kaggle-27b-lora-works`).

### Track D — Oct 1 absorption (certain value; prepare now)

The byte-copy pipeline (copy → committed → 3.25 live in one day) and the Modal regime rig are proven. Prepare: (1) a 24-hour checklist for reproducing a released harness's serving regime on Modal and attesting it on Kaggle; (2) keep ≥ 30 h of the weekly GPU quota unspent in the week of Oct 1; (3) our Track-A grafts written to the duck's graft convention so they can be re-applied to a released duck descendant in hours. If a top team releases a different loop, the Modal rig measures it in Kaggle geometry the same day.

## 7. Slot policy until Oct 1

- Daily slots fly **only pre-registered candidates** that passed the rig gate, or base draws that size something specific. No public-max chasing: the final is the private mean.
- Yield900 draw 3 (09-09 02:00Z) completes the 3-draw rule and pools into the base family; after it the base is v4 (byte-copy) for any control draw.
- Every live candidate needs ≥ 3 draws before selection; the two final selections are duplicates of the best-mean config.

## 8. Timeline

| dates | work |
|---|---|
| Sept 9–12 | A1 build + kill test; A3 port + clock gate; B1 wave in an idle session |
| Sept 12–16 | A2 build + kill test; A4 port; A1 25-game wave |
| Sept 16–22 | 25-game waves of survivors; counterbalanced redraws for anything ≥ 45; C decision on Sept 18 |
| Sept 22–30 | live 3-draw candidates from A; Track-D checklist; keep quota |
| Oct 1–7 | absorb Milestone-2 releases on the rig; compose with A |
| Oct 8–Nov 1 | live draws of the composed config; select two duplicates of the best mean |

## 9. What would change this plan

- A1 and A2 both flat on the rig (≤ 42) → the thesis is wrong for this brain; fall back to A3/A4 as the base and to D.
- A3 or A4 ≥ 45 in Kaggle geometry → they become the base; A1/A2 become grafts on them.
- A released Milestone-2 harness ≥ 7 that fits the box → absorb first, then re-apply A on top; our grafts are the differentiator, not the base.
