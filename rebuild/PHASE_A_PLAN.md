# Phase A — Online Action-Effect Model (GraphRanker)

> Software-factory T2 design (orchestrator->tier->team->judge), 2026-06-16. Gate: combine([REVISE,REVISE,REVISE])=REVISE folded into synthesis; no security_block; GraphRanker ranked #1 by all 3 judges. HALTS for human approval before any model code.

Everything is verified. Here is the final plan.

---

# Phase A — T2 Implementation Plan: Online Action-Effect Model (GraphRanker)
**Status: AWAITING HUMAN APPROVAL — halts before any model code.**

All three independent judges ranked the same approach #1 and returned REVISE (no security block). This plan adopts that winner, grafts the runner-up's faithful-port fidelity, and bakes in every blocking required-change. Code citations below are verified against the live repo.

---

## 1. Recommended Approach: **GraphRanker** (graph backbone + online action-effect CNN as a within-tier re-ranker)

**Unanimous judge verdict.** All three judges ranked GraphRanker #1 (overall 7.5 / 8.5 / 7.0), OnlineActionEffectExplorer #2, SalienceRanker #3.

**What it is.** A new class `OnlineLearningExplorer` that **composes** (never subclasses, never edits) `SalienceExplorer`. The explorer's object-graph, BFS-to-frontier, suspicion filter, HUD/border mask, exploit-reward, and reset logic remain the sole execution backbone and memory. An online, per-game CNN does exactly one thing: when the explorer reaches its **uniform-random tie-break** (`src/arcagi3/salience_explorer.py:250`, `choices[int(self.rng.integers(0, len(choices)))]`) among >1 equal-tier untried candidates, the model re-orders that *already-sanctioned* set by predicted frame-change probability — replacing both the random tie-break and the hand-crafted click salience from `perception.salient_click_targets` (`salience_explorer.py:113`). It never picks a tier, never overrides `reward_action()` (`:233`), a `plan` replay (`:237`), a reset, or BFS; it never adds or invents a candidate; it never plans over predictions.

**Why GraphRanker over the runners-up (judge-cited):**
- **Correct label source.** GraphRanker specifies the frame-change label as `explorer._key(after) != explorer._key(before)` — the masked `object_state_key` (verified at `salience_explorer.py:118-126`: `_key` zeroes volatility + border cells before hashing). SalienceRanker (C1) used raw `(after != before).any()`, which all three judges + the perception specialist independently flagged as a **systematic label-space mismatch** — it fires on HUD counters, animation, and monotonic progress bars the graph deliberately masks, training the click head to chase useless frame changes. This is the single most concrete correctness bug in the field and it is **fatal to C1, blocking for any candidate**.
- **Correct normalization.** GraphRanker uses GroupNorm (batch-size-independent). OnlineActionEffectExplorer (C2) used BatchNorm, a silent correctness bug at batch=1 inference (train/eval running-stat mismatch → noisy logits → non-deterministic confidence gate), confirmed by both the perception and performance specialists.
- **Spatially faithful click head at lowest cost.** 1×1 conv on a 64×64 feature map (9-px receptive field, per-pixel, no upsample), ~70K params — vs C1's stride-2 + bilinear-upsample head that blurs adjacent candidates across 4×4 regions (violating StochasticGoose's "spatially-aware, NOT flattened" principle) and vs C2's 98K. Performance specialist: 567 vs 794 MFLOPs.
- **Only candidate that confronts the integration hazard.** GraphRanker explicitly names the `prev_action` desync risk and supplies the correct fix: reuse the **proven** `wm_policy.py` seam (verified `wm_policy.py:165` "always call base.decide() first", `:231` `self.base.prev_action = ... out` write-back) — the same composition pattern locked by `test_wm_passthrough_equals_hybrid` (`tests/test_wm_policy.py:137`).

**Grafts from the runners-up (judge-directed):**
- From C2: stay **faithful to StochasticGoose** — 16-channel one-hot, 4-layer conv, two heads (5-way action-effect + 64×64 conv click map), online BCE, per-level reset. **Reject** C2's BatchNorm and **reject** C2's "global argmax of the score map as an injected candidate" (Judge #3: that widens the set beyond what the graph sanctioned — the exact brute-force move the research says collapses on hardened games).
- From C1: its cheaper stride-2 backbone is the **documented fallback base** *only if* the A.0 probe reveals a weaker-than-expected GPU — but with a finer click head than C1's.

**Binding integration contract (Section 7 of acceptance — verified non-negotiable):**
1. **Do NOT modify** `salience_explorer.py` or `policy.py`. Compose only. (Judges unanimously reject GraphRanker's own "option-B" callback-into-`_choose` last resort and its "option-A" external `_choose` branch reconstruction.)
2. `base_token = self._explorer.decide(...)` is **line 1, unconditional**, every step (mirrors `wm_policy.py:165`) — the graph/volatility/level-detection always stay warm. This structurally prevents the C1–C7 regression where the planner intercepted control before the graph recorded a transition.
3. Override fires **only** when the returned token is a provable step-3 tie-break pick: `node.reward_action() is None` AND `self._explorer.plan` empty before+after AND token is not a reset AND token ∈ `node.untried_le(active_group)` with `len(choices) > 1`. Otherwise return `base_token` verbatim.
4. On override, write back `self._explorer.prev_action = override` (mirrors `wm_policy.py:231`) so the graph records the edge actually taken.
5. Add a **sticky per-level circuit-breaker** (Judge #2): after N consecutive model-caused no-change steps, disable the model for the rest of that level → pure graph. Fourth independent guard.

---

## 2. FIRST DE-RISKING STEP (blocking, no model code merges until done): verify torch + GPU in the offline eval image

**Why first.** The entire plan assumes torch + CUDA at eval. The submission-contract memory records "GPU T4 x2, torch available" — but that is **secondhand from the sample notebook**, never empirically confirmed for *this* image. Locally we have torch 2.9.1 with **CUDA = False**. The performance specialist proved GraphRanker training on CPU runs at **8–32 act/s — a real levels-per-12h regression below the v6 floor that the per-action firewall does NOT catch**. So CUDA presence is a true go/no-go.

**How (probe, exact mechanics):**
1. Add a probe cell to `submission/notebook.ipynb` guarded by `if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):` printing: `torch.__version__`, `torch.cuda.is_available()`, `torch.cuda.device_count()`, `get_device_name(i)` + total memory per device, `numpy.__version__`, and `os.listdir('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels')` filtered for `torch`. Wrap the torch import in try/except so the probe itself never crashes the run.
2. Rebuild: `python submission/build_notebook.py`. Push: `kaggle kernels push -p submission`. **Submit one scored slot** (costs one daily submission).
3. Read the rerun stdout (Kaggle captures stdout even for private kernels) and record the literal output in a **new memory file** `memory/arcagi3-eval-image-facts.md`.

**Go / No-Go ladder (decided by the recorded probe, not assumption):**
- **GO (full plan):** `torch` present AND `cuda.is_available()` True with ≥1 T4 → GPU path is primary. **Note on wheels:** do not attempt to embed torch (CPU wheel >200 MB, GPU wheel >700 MB — infeasible in a notebook cell). Most likely torch is in the Kaggle base image, not the competition wheels. If — and only if — the probe shows a `torch-*.whl` in `arc_agi_3_wheels/` and torch is otherwise absent, extend the existing offline install cell with `pip install --no-index --find-links <wheels> torch`.
- **CONDITIONAL (CPU-only torch, no CUDA):** model runs **inference-only, training DISABLED** (per all three judges: raising `TRAIN_EVERY` is proven insufficient; CPU training is a net regression). Re-evaluate whether Phase A has any lift in this mode before proceeding past A.3.
- **NO-GO (torch absent entirely):** the numpy fallback becomes the only path — it degenerates to v6 by construction, so Phase A ships nothing and **v6 stays live**. Re-scope Phase A.

This gate matches the acceptance criteria A.0 and is the first item in all three judges' required-changes lists.

---

## 3. Phased Build Plan — each phase small, default-OFF, independently measured vs the no-regression gate

**No-regression gate definition (Judge #2 fix — verified):** the repo has **161 `def test_` functions but 174 collected** (parametrization; confirmed `pytest --co` reports `174 tests collected`). The gate is therefore: **"all currently-collected tests pass, zero failures, zero deletions/renames; new tests additive"** — not a brittle hardcoded 174 count. Plus 8 local games and the 25-game @30k sweep ≥33.

| Phase | Deliverable | Measurement vs gate | Default state |
|---|---|---|---|
| **A.0** | Probe cell + `memory/arcagi3-eval-image-facts.md` | Probe recorded; go/no-go decided | n/a — no model code |
| **A.1** | `perception.encode_onehot` (pure numpy) + `src/arcagi3/online_model.py` (lazy torch, `TORCH_AVAILABLE`, GroupNorm CNN, NumpyFallbackModel, replay buffer) + `src/arcagi3/online_explorer.py` (`OnlineLearningExplorer`, `enabled=False`) + `tests/test_online_policy.py` lockstep identity | Full suite passes; lockstep byte-identical on all 8 local games (flag off); module imports with NO torch at import time | OFF (byte-identical to v6) |
| **A.2** | All fail-safe paths wired (Section 4 below) | All Section-2-acceptance unit tests pass; `ARCAGI3_ONLINE_LEARNING=1` never crashes on 8 local games; `--agent salience` still 8/8 wins ≥24 | OFF; flag-on tested, not shipped |
| **A.3** | BCE training loop, masked-`_key` labels, hash-dedup buffer, `TRAIN_EVERY=10`, confidence gate, circuit-breaker | Per local game @2000 budget flag-on: ≥50 unique transitions; train loss decreases between step 10→100; local wins ≥24 hold; **btnc ≤50 actions, vc33 re-verified** (click head sanity) | OFF; validated flag-on |
| **A.4** | Add `online` branch to `scripts/ab_salience.py` `make_policy()` (`scripts/ab_salience.py:36-41`) | 25-game A/B (below) | OFF until A.6 |
| **A.5** | `online_model.py` + `online_explorer.py` added to `CORE` in `submission/build_notebook.py`; flag-gated branch in `submission/my_agent.py` (`_Policy`, `my_agent.py:28-39`); 6-thread parallel + latency tests | Notebook builds, runs end-to-end, valid `submission.parquet`; `test_online_latency.py` median ≤5ms CPU / ≤1ms GPU; 6-thread test no OOM/deadlock | OFF in shipped notebook |
| **A.6** | Leaderboard submission with flag ON | Public score **strictly > 0.33 (min 0.34)** → PROMOTE | Promote only on pass |

**Exact A/B commands (A.4 — back-to-back, same session, same key, same 25 games, to control server variance):**
```
PYTHONPATH=src python scripts/ab_salience.py salience 30000   # baseline; must confirm >=33 first
PYTHONPATH=src python scripts/ab_salience.py online 30000     # Phase A; must be STRICTLY >=34
```
*(The harness already dispatches on `sys.argv[1]`; `scripts/ab_salience.py:36` `make_policy()` needs one added `online` branch — the only change to the harness.)*

**A.4 promotion rule (Judge-hardened):** `online` ≥ 34 **strictly** (a 33=33 tie does NOT promote) **AND** ≥1 game that scored 0 @v6 now scores ≥1 (genuine new capability, not variance) **AND zero per-game regressions**. If +1 with no new-game win → treat as noise, re-run 3×, require mean > 33.5.

**Local game commands (A.1–A.3):** existing `LocalGamesCollector` path for `--agent salience` (must hold 8/8 ≥24), plus the new `--agent online` (must also hit 8/8 ≥24 at budget 16000).

---

## 4. Banked-0.33 Firewall + Fail-Safe-to-SalienceExplorer Contract

**Firewall (downside provably zero):**
- **Default OFF is byte-identical.** `OnlineLearningExplorer.decide()` with `cfg.enabled=False` (env `ARCAGI3_ONLINE_LEARNING=0`) returns `self._explorer.decide(...)` from the same object, same rng, same call order. Proven by `tests/test_online_policy.py::test_lockstep_identity_all_games`, cloned from the verified `test_wm_passthrough_equals_hybrid` (`tests/test_wm_policy.py:137`).
- **No partial ships.** v6 (`SalienceExplorer`, public 0.33) remains the **live submission** until A.6 clears the full ladder. No judgment-call promotions — the 25-game sweep number + leaderboard are the arbiters.

**Seven-layer fail-safe (every catch terminates in "return `base_token` = SalienceExplorer unchanged"):**
1. **Import safety** — `online_model.py` top level: `TORCH_AVAILABLE=False`; `try: import torch … except Exception:` logs `[online_model] torch unavailable -> numpy fallback`. Torch imported **lazily**, never at module import time (A.1 import-safety test).
2. **Master gate** — `if not cfg.enabled: return self._explorer.decide(...)`.
3. **Torch/model gate** — `if not TORCH_AVAILABLE or self._model is None: return base_token`.
4. **Untrained delegate** — `if len(buffer) < MIN_TRANSITIONS (10): return base_token` (zero model forward passes early in every level).
5. **OOM recovery** — catch `torch.cuda.OutOfMemoryError` / `RuntimeError("CUDA out of memory")` → `empty_cache()`, log once, switch that **game** (not just level) to CPU for the remainder; no silent retry.
6. **Outer scoring/override catch** — any exception after `base_token` is computed → log traceback, return `base_token`.
7. **NaN/Inf guard** — `if torch.isnan(logits).any() or torch.isinf(...): return None` (→ base_token); plus the confidence gate `top_score < CONFIDENCE_THRESHOLD (0.6) → None`.

**Numpy fallback contract (precise, not a "dumb stub"):** `predict()` returns 0.5 for every `('S',aid)` and 0.0 for every `('C',x,y)` — all below the 0.6 gate → override never fires → `decide()` degenerates **exactly** to `SalienceExplorer` ordering. `encode_onehot` is pure numpy so it runs in this path. Verifiable: force `TORCH_AVAILABLE=False`, run the 25-game sweep → must equal 33 ±2.

**Buffer storage (Judge fix — avoids a 52 GB OOM):** store raw `int8[64,64]` grids (4 KB), key on `(state_bytes, action)` with hash-dedup (latest-wins), `encode_onehot` at sample time. **Never** store float32 one-hot (200k × 256 KB = 52 GB, instantly fatal). Per-level cap **20k–50k** (buffer resets each level; 200k unreachable except in a non-leveling game where memory isn't binding).

**Per-game GPU assignment (6 parallel Swarm games, 2× T4):** `device = cuda:(hash(game_id) % torch.cuda.device_count())` → 3 games/GPU; `device_count()==0` → CPU/numpy. Models are **per-`MyAgent`-instance, never shared, never persisted** (Section 7.2/7.3).

---

## 5. Generalization + Overfit Guards

**Reframed headline (Judge #3, citing `arcagi3-research-findings.md`).** The documented StochasticGoose 12.58%→0.25% collapse is attributed to **"full observability / non-resetting / brute-forceable"** tricks — NOT to frozen weights overfitting a public set. So "reset weights per-game" answers a *different* failure mode. The **primary** generalization argument is the **structural-caps firewall**: the explorer's graph / BFS / suspicion / reset-bounce bound exploration, so the model can only re-rank *within a graph-sanctioned candidate set* and can never turn the agent into a brute-forcer. This is why C2's "inject the score-map argmax as a candidate" is rejected.

**Secondary guard (still valid):** the model is re-initialized per-level and per-game and **never persisted** — no weights ship. What must generalize is the *online-learning procedure* ("which untried action/click most reliably changes the masked game state"), a game-agnostic affordance prior, not memorized content.

**Acceptance hardened against in-sample overfit (Judge #3):**
- The 25-game sweep is the **public** set; a threshold tuned on it can overfit. Tune `CONFIDENCE_THRESHOLD` (sweep 0.5–0.8) and `TRAIN_EVERY` on a **held-out dev subset**, not the full sweep.
- Require ≥1 game 0→≥1 (genuine capability), zero per-game regressions, treat single-game spikes as noise (re-run 3×, mean > 33.5).
- The **hidden leaderboard >0.33** is the **only honest generalization arbiter** — the sweep is necessary-not-sufficient.
- **btnc / vc33** re-verified explicitly in A.3 (reversible toggles can be confidently-frame-changing-but-useless; frame-change is necessary-not-sufficient for progress, and the squared action-efficiency metric punishes wasted clicks). Keep the explorer's **tier as the primary sort key**; model re-orders **within-tier only**.

---

## 6. Honest EV: score band, effort, kill criteria

**Realistic score band (PM/charter, judge-endorsed).** Phase A is a within-tier tie-break re-ranker. Evidenced anchor: the preview CNN at 12.58% beat the directed-graph class (~6.7%) ~2:1; we are that class at 0.33.
- **Realistic: 0.45–0.70** (1.3×–2.1×).
- **Stretch: ~0.8.** Anything near 1.21 is a Phase B (MBRL + planning) outcome — **not promised here**.
- **Floor: ≥0.33**, firewalled to zero downside. Judge #3's sober read: the *most likely* honest outcome is **+1 to a few levels (≈0.34–0.4×)**; the 0.45–0.70 band is the optimistic-but-credible case.

**Why EV is positive despite modest upside.** The downside is firewalled to zero (default byte-identical until A.6). The +0.12–0.37 lift is the most evidence-backed increment available, and it builds the **online-training harness inside the agent** that Phase B (DreamerV3-style MBRL) requires. We buy the cheapest evidenced step plus the scaffolding for the next.

**Effort estimate.**
- A.0 probe: **~0.5 day** + 1 calendar day for the scored rerun to return.
- A.1 skeleton + lockstep: **~2–3 days.**
- A.2 fail-safe paths + tests: **~2–3 days.**
- A.3 training loop + local validation: **~3–4 days** (most iteration-heavy).
- A.4 25-game sweep + threshold tuning: **~2–3 days** (sweeps are slow; multiple re-runs likely).
- A.5 notebook embed + parallel/latency tests: **~1–2 days.**
- A.6 leaderboard: 1 daily slot + read.
- **Total: ~2.5–3.5 weeks** of focused work, well inside the Nov 2 deadline.

**Kill criteria (stop and keep v6 live):**
1. **A.0 NO-GO:** torch absent at eval → numpy fallback is the only path → Phase A ships nothing; re-scope.
2. **A.0 CONDITIONAL with no GPU AND inference-only shows no local lift in A.3** → kill (CPU training is a net regression).
3. **A.3:** training loss does not decrease, or btnc/vc33 regress, or local wins drop below 24 → fix or kill before touching real games.
4. **A.4:** `online < 34`, OR any per-game regression, OR no 0→≥1 game with mean ≤ 33.5 over 3 runs → do not ship; tune `CONFIDENCE_THRESHOLD`/`TRAIN_EVERY` once, re-measure; if still flat → **kill, bank the harness for Phase B.**
5. **A.6:** leaderboard ≤ 0.33 → do not promote; v6 stays live.

---

## Files touched (verified to exist)
- **New:** `src/arcagi3/online_model.py`, `src/arcagi3/online_explorer.py`, `tests/test_online_policy.py`, `tests/test_online_latency.py`, `memory/arcagi3-eval-image-facts.md`
- **Modify (additive only):** `src/arcagi3/perception.py` (add `encode_onehot`), `scripts/ab_salience.py` (add `online` branch at `make_policy()`, line 36), `submission/build_notebook.py` (add 2 modules to `CORE`), `submission/my_agent.py` (flag-gated branch at `_Policy`, line 28), `submission/notebook.ipynb` (A.0 probe cell via `build_notebook.py`)
- **Never modified (Section 7, binding):** `src/arcagi3/salience_explorer.py`, `src/arcagi3/policy.py`

**Proven seam reused:** `wm_policy.py:165` (base.decide first) + `:231` (prev_action write-back), locked by `tests/test_wm_policy.py:137`. **Target tie-break:** `salience_explorer.py:250`. **Label source:** `salience_explorer.py:_key` (`:118-126`, masked `object_state_key`).

---

**This plan halts here for human approval. No model code will be written until (a) you approve, and (b) the A.0 probe returns a GO.** The first action on approval is the A.0 probe submission — nothing else.