# Genre-Perceiver Program — human-style goal perception for ARC-AGI-3

**Date:** 2026-06-23
**Branch:** factory/experiment
**Status:** design — awaiting user review (program-level spec; phase specs to follow)
**Builds on:** Exp 50 (solvability audit), Exp 51 (pipeline death-point map), the two VLM probes (small-VLM genre 57%, strong-VLM genre ~83%), eval-image facts (T4×2, no internet), and the validated discover→plan back-half (collect 14×).

---

## 1. Why this program exists

Across 51 experiments the wall has been isolated with unusual precision. The campaign mechanised the
parts of human play that are **search and logic** — and proved they are *not* the bottleneck:

- **Search is not the wall** (Exp 50): true-model BFS clears every valid game inside the 200k
  production budget (ls20 L1 in 445 nodes). The ~7,961-action ls20 result was never a search problem.
- **The planner is capable** (Exp 50): given a correct model it clears `collect` at perfect efficiency.
- **The gap is perception** (Exp 51): the discover→plan loop dies in an even three-way split —
  **FRONT-END** (no avatar perceived: su15/sk48/tn36), **ONTOLOGY** (avatar but no frameable goal:
  m0r0/tr87), **TERMINAL** (plans+executes but the win-condition is mis-modelled: ls20/re86/wa30).

This is the human-play gap stated precisely. Humans clear an unseen level in ~100 actions because
they **perceive the goal from priors** (objectness, agency, genre literacy) — not because they
out-search the machine. Our agent has a tiny hand-coded prior (one directional-avatar family), so it
clears the one game that matches (`collect`, 14×) and is blind elsewhere.

**Thesis.** Give the agent the missing human faculty — *genre/goal perception from broad priors* —
and feed its output to the validated induction+planning back-half. The faculty cannot be hand-coded
(brittle; overfits — every prior-injection experiment cratered) nor taken off-the-shelf (small VLMs
are too weak — 57%, this session). It must be **acquired the way humans acquire genre literacy: from
broad simulated experience**, then distilled into a model small enough to run offline.

## 2. The decisive prior evidence (what is already known, not assumed)

| Finding | Source | Implication for this program |
|---|---|---|
| Search/planning solved | Exp 50 | Build only the front-end; reuse the solver. |
| Death split FRONT-END/ONTOLOGY/TERMINAL ≈ 3/2/3 | Exp 51 | Front-end fixes ≤5/8 holdouts; terminal stays bespoke. |
| Strong VLM reads genre ~83% + agent | Claude probe (frames) | A capable teacher exists. |
| Small VLM reads genre 57%, agent ~0% | qwen2.5vl:7b probe | Off-the-shelf insufficient → fine-tuning required. |
| Symbolic interaction-probe nails agent-ID | Exp 31 | Front-end should be hybrid: VLM=genre, symbolic=agent. |
| Eval = T4×2 16GB, torch 2.10, NO internet | eval-image-facts | Embed open weights; call sparingly; no hosted API. |
| Dev wins have repeatedly not moved Kaggle | v15 0.28; HOLDOUT-inert levers | Gate on HOLDOUT proxy; submit cautiously; firewall always. |

## 3. Architecture (the converged pipeline)

```
SYNTHETIC GENERATOR ──labels by──▶ STRONG VLM (teacher, dev-only)
   (Track A)                          │ distill / fine-tune
      │ priors source                 ▼
      └────────────▶ small EMBEDDABLE VLM  ── genre ──┐
                     symbolic interaction-probe ── agent + movement ──┤
                                                                      ▼
                                                              GOAL-TEMPLATE (per genre)
                                                                      ▼
                              VALIDATED discover→plan solver (genre-parameterised) ─▶ execute ─▶ revise
```

The back-half (everything from GOAL-TEMPLATE rightward) **exists and is validated**. This program
builds the front-half and the data engine behind it. Tracks A (synthetic priors) and B (embed a VLM)
are **not alternatives** — A is the fuel B needs, because off-the-shelf B is too weak and the teacher
cannot run at eval.

## 4. Genre taxonomy

A closed set covering the public families (Exp 48/49/51 + frame inspection). Hidden games are assumed
to draw from the same designer/engine distribution (plausible, unverifiable — see §10 risk).

| genre | win-template | public exemplars |
|---|---|---|
| NAVIGATE | reach target cell(s) | ls20 |
| COLLECT | reach-all items | collect, sp80? |
| PUSH | move block(s) onto marker(s) (sokoban) | wa30 |
| AIM | align a cursor/crosshair onto target(s) | su15, re86 |
| MATCH | arrange/align pieces to a shown pattern or color sequence | sk48, tr87 |
| SYMMETRY | mirror the two halves | m0r0 |
| CLICK | select the special element; no moving avatar | vc33, tn36, lp85 |

## 5. Component specs

### 5.1 Synthetic multi-genre generator (Track A — the linchpin)
- **Output per sample:** `(frame 64×64 uint8 grid, genre_label, agent_pos|none, goal_descriptor)`.
- **Genres:** the seven in §4, each a parametric template (board size, palette, agent shape/color,
  target count/placement, obstacles, optional UI meter/border bar).
- **Visual fidelity is the whole ballgame.** Frames must match the real rendering: the 16-colour ARC
  palette (reuse `vlm_prior_probe.PAL`), connected-component object shapes, bordered boards, edge
  progress bars (the HUD elements that broke earlier explorers). Domain gap here is *the* failure mode.
- **Scale:** start ~2–5k labelled frames (≈700/genre) for the gate; expand for fine-tuning.
- **Module:** `src/arcagi3/synthgen/` (generator + per-genre templates + renderer). Pure-numpy,
  deterministic by seed (no `Math.random`/wallclock issues; seed in).

### 5.2 Genre classifier + THE DECISIVE GATE (Phase 1)
- **Model:** a small CNN (embeddable, CPU-feasible) trained to classify the 7 genres from a frame.
  Cheapest possible test of the generalization thesis — no VLM needed for the gate itself.
- **Train:** synthetic frames + the 8 TUNE public games (hand-labelled genres).
- **Test:** genre top-1 accuracy on the 8 HOLDOUT public games (the unseen-distribution proxy).
- **GO/NO-GO threshold:** HOLDOUT genre top-1 **≥ 75%** (comfortably above small-VLM 57% and chance
  ~14%). Below that, priors-from-simulation does not transfer (the StochasticGoose overfit trap) and
  the program **stops at Phase 1** with a negative result — cheaply.
- **Harness:** `scripts/genre_gate.py` → prints synthetic-val acc, TUNE acc, **HOLDOUT acc** (the
  number that matters), confusion matrix, and the per-game miss list.

### 5.3 Hybrid front-end (Phase 2)
- **Genre:** the fine-tuned embeddable VLM (or the CNN classifier, if the gate shows it suffices —
  preferred, because a CNN is smaller/faster/CPU-safe than a VLM).
- **Agent + movement model:** the **existing symbolic interaction-probe** (Exp 31 / discovery
  PROBE_MOVEMENT) — it already identifies the agent and fits deltas where a VLM said "none visible".
- **Output:** `GenrePerception{genre, agent_color, deltas, goal_template}`.

### 5.4 Teacher labelling + embeddable-VLM fine-tune (Phase 2, only if §5.2 needs a VLM)
- **Teacher:** a strong VLM (Claude via dev API, or a large local model) labels real + synthetic
  frames with genre (+ agent) to (a) augment training with the real distribution and (b) provide
  distillation targets.
- **Student:** LoRA-fine-tune a T4-fitting open VLM (qwen2.5vl-2B/7B or Moondream2) on the corpus for
  genre. Ship LoRA weights as a Kaggle dataset. **Call once per level**, not per step (latency budget).
- **Note:** if the §5.2 CNN already clears the gate on HOLDOUT, the VLM fine-tune is *optional* — the
  CNN is the simpler, cheaper, safer front-end. The VLM path is the fallback if a CNN can't generalize.

### 5.5 Goal-template → solver integration (Phase 3)
- Each genre instantiates a goal-template that parameterises the discover→plan terminal: NAVIGATE →
  reach-slot; COLLECT → reach-all (already works); PUSH → block-to-marker; AIM → cursor-to-target;
  MATCH → match-reference; SYMMETRY → minimise mirror-error.
- Wire into `DiscoveryExplorer` behind a default-OFF flag; the existing induction+planner executes.
- **Measure:** first-level acquisition on HOLDOUT (budget 6000, live), vs the 0.375 salience baseline.

## 6. Phased plan with kill-early gates

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| **0 — probes** ✅ | small/strong VLM genre, search-not-the-wall | done; teacher exists, off-shelf insufficient |
| **1 — GATE** | generator + CNN classifier + HOLDOUT genre acc | **HOLDOUT top-1 ≥ 75%** else STOP |
| **2 — front-end** | hybrid GenrePerceiver (genre model + symbolic agent) | agent-ID + genre correct on ≥6/8 holdouts |
| **3 — integration** | goal-template → solver, first-level on HOLDOUT | beats salience HOLDOUT (>0.375 mean-levels) |
| **4 — terminals** | per-genre win-condition handling (the bespoke part) | each genre clears ≥1 holdout end-to-end |
| **5 — ship** | offline packaging (embed weights/CNN, T4, sparse calls) | beats banked 0.33 dev-safely → submit |

Each phase is its own design doc before code. The program **halts at the first failed gate** and is
written up as a result — Phase 1 is the cheapest make-or-break and must run before any VLM work.

## 7. Offline-feasibility constraints (hard requirements)

- **No internet at eval.** No hosted-API calls in the shipped agent. The teacher is dev-only.
- **GPU:** push the kernel with `--accelerator NvidiaTeslaT4` (T4×2, cap 7.5, torch-compatible). A
  quantized 2–7B VLM fits 16GB; a CNN runs on CPU. Fail-safe: detect a real CUDA op succeeds, else
  fall back to CPU/transfer-dense.
- **Latency:** front-end model called **once per level** (genre is stable within a level); symbolic
  probe + planner handle every step. Keeps within the 8h/game budget.
- **Weights shipped as a Kaggle dataset**, materialised at runtime (same pattern as the embedded
  modules in `build_notebook.py`).

## 8. Firewall discipline (non-negotiable)

Every component lands behind a default-OFF flag; the shipped `my_agent.py` stays **byte-identical to
transfer-dense (0.33)** until a phase is measured a strict HOLDOUT improvement. The GenrePerceiver
path degrades to transfer-dense on any failure (model missing, CUDA broken, low confidence). No phase
can regress the banked score. This is the same discipline that held across all 51 experiments.

## 9. Honest ceiling (stated up front, must hold through the program)

- A perfect front-end addresses **FRONT-END + ONTOLOGY** games (≤5/8 holdouts reach induction with the
  right template). The **TERMINAL-bespoke** wall (ls20 moving-goal, re86 match, wa30 carry) persists —
  Exp 51 showed each needs a different win-condition that does not transfer. Phase 4 tackles these
  per-genre, with no guarantee of generality.
- Realistic upside = **first-level breadth** (the capability the 0.58–0.70 cluster has), which moves a
  level-weighted score **modestly** because first levels weight least. This is a credible shot at
  rising off 0.33 toward the cluster via breadth — **not** a guaranteed win, and not a path to the 1.21
  leader (which uses hosted-scale vision/compute outside the offline box).

## 10. Risks & what would falsify the program

- **Synthetic↔real domain gap** (primary): the generator's frames may not capture the real games'
  visual idiom → the classifier overfits synthetic and misses HOLDOUT. *Phase-1 gate measures exactly
  this; it is the kill switch.*
- **Hidden-distribution gap:** even good HOLDOUT-public accuracy may not transfer to the truly hidden
  games. Unmeasurable offline — acknowledged, not solved. (This is the AGI-hard core.)
- **Terminal bespoke-ness** (Exp 51): front-end success ≠ cleared levels; Phase 4 may stall per-genre.
- **Dev-misleads-Kaggle** (durable lesson): a HOLDOUT win may still not move publicScore. Submit only
  with the firewall and treat dev gains skeptically.
- **Big-build-comes-back-negative** (the campaign's recurring outcome): mitigated by the kill-early
  gates — the most we lose is Phase 1's ~1–2 days.

## 11. Definition of done (program level)

Either: (a) a shipped, firewalled GenrePerceiver agent that beats 0.33 on a clean HOLDOUT proxy and on
Kaggle; **or** (b) a documented negative result naming the exact gate that failed (most likely Phase-1
domain gap or Phase-4 terminal bespoke-ness), folded into the experiment overview as Exp 52+. Both are
acceptable; the gates ensure we learn the answer cheaply.

## 12. Immediate next step

Build **Phase 1** only: `src/arcagi3/synthgen/` + `scripts/genre_gate.py`, and report the single
decisive number — HOLDOUT genre top-1 accuracy. Everything downstream is conditional on it.
