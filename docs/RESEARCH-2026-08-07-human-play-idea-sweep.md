# RESEARCH 2026-08-07 — Human-Play Idea Sweep (high-temperature, cross-domain)

**Provenance:** 17-agent workflow (8 blind domain-lens ideators + 4 web scouts -> 4 adversarial
feasibility filters -> 1 synthesis). Ideators received ONLY a clean problem statement (no CLAUDE.md,
no memory, no campaign docs) per Ahmed's independence requirement. 93 raw ideas -> 93 verdicts ->
45 novel+feasible survivors -> 9 merged mechanisms. Filters judged against the hard eval constraints
and the full tried-list, so the Top 8 below is deduplicated against everything shipped to date.

**Companion facts from the same day's research:** ARC Prize open-sourced 342 step-by-step human
replays across the 25 public environments (458-participant study) — the first real human-behavior
ground truth available to this campaign; see the Human Replays wildcard.

---

# Closing the Human Gap: Final Research Report
## What humans do in minute one that the agent never does — and the eight builds that close it

**Framing fact that governs the ranking:** on ~70% of runs the agent banks ZERO levels. That is not an execution failure — it is a *mechanic-decoding* failure. Humans decode the mechanic in the first 1–3 minutes via a stereotyped ritual: wiggle the controls → identify the self → classify the genre → poke the anomaly → commit to one theory and test it. Every top-ranked item below is one stage of that ritual, made architectural (enforced by the harness/sandbox) rather than advisory (suggested in the prompt), because the sweep's verdicts repeatedly show prompt-level advice gets satisfied performatively.

**A note on duplicates:** the sweep's 30+ survivors collapse into ~9 real mechanisms. Rankings below merge each cluster into one build; the merge sources are named so nothing gets built twice.

---

## Part 1 — Ranked Top 8

### 1. Probe Battery as Macro-Action (`run_probe`) — change the exchange rate
*Merges: TAS Lua harness (8/8), Blicket Battery, tertiary-circular-reaction bursts, the probe halves of the babble ideas.*

**What it is.** A sandbox primitive `run_probe(script) -> effect_table`: the LLM authors a short Python policy that autonomously executes up to N environment actions (raster clicks over connected components, direction sweeps, act-observe-undo chains), and returns one compressed table of `{input → changed_region, delta_summary, reversible?}`. The LLM then gets ONE deliberation over the whole atlas.

**Human basis.** TASers write Lua scripts that fire hundreds of inputs and reason over the logged table; casual humans do the motor version — ten seconds of wiggling absorbs dozens of micro-experiments before a single verbal thought forms. The agent's current contract — one full 27B deliberation per keypress — is a catastrophic exchange rate that no other fix can compensate for. This is both the information-rate bottleneck and the wall-clock bottleneck (25+ games against one server).

**Implementation.** (a) Hard per-script action cap (30) and max 2 batteries per level; (b) whitelist of safe script patterns (click-sweep, direction-sweep, act-undo pairs); (c) batteries preferentially run in SCOUT mode / on the sacrificial early level (the campaign's own action-accounting shows early-level sacrifice is net-positive, and depth dominates efficiency in scoring — this guard is load-bearing); (d) undo/reset assumptions verified per game before any script relies on them (see wildcard: undo-fidelity audit).

**Cheapest test.** Offline harness A/B on 6 dev games spanning click-puzzle and avatar families: current agent vs `run_probe` agent, same wall-clock. Primary metric: levels banked; secondary: LLM calls per banked level. One afternoon, fully offline.

**Expected effect.** Largest single lever. If mechanic-decoding fails because the agent gathers evidence at 1 bit per expensive thought, a 30-action battery per level is a ~10–30× increase in early-game evidence rate. Plausible movement of zero-bank runs from ~70% toward ~45–55% on its own; it also enables ranks 2, 3, 5, and 6 as consumers of its output.

---

### 2. Empowerment Wiggle + Controllability Masks — find the body before the goal
*Merges: Empowerment Wiggle (9/7), Contingency-Probe Self-Localization, Motor-babbling (verbatim dup), Sensorimotor Contingency Map, Controllable-entity detection, Common-Fate Probe, ICM controllability filter. One build, seven submissions.*

**What it is.** A hard-coded, LLM-free opening battery (each directional twice, then clicks on k distinct color blobs — directions first, clicks last, early-abort if 4 directional presses yield zero diff) whose output is three persistent per-pixel masks: **SELF** (moves with directionals), **REACTIVE** (changes on click), **DEAD** (never changes), plus an autonomous-dynamics estimate (changes uncorrelated with action — the principled generalization of the hand-built HUD mask). Clustering pixels by change-signature across the battery additionally yields rigid-body segmentation (common fate): avatar vs same-colored decor, pushables flagged by "moves only when avatar adjacent."

**Human basis.** Infants localize their own limbs by contingency detection within minutes (Rovee-Collier; Watson); any adult's first 5 seconds in a new game is pressing arrows to see what moves. The agent instead guesses "the avatar" from a static frame — and the sweep's diagnosis is that avatar-misidentification plausibly explains a large share of zero-bank navigation runs. Equally valuable is the **negative proof**: no contingent component after the battery = this is not an avatar game; stop spending directional actions forever (19/25 dev games are click-dominant, so the fast negative fires on the majority).

**Implementation.** Runs as the first `run_probe` battery of every game (12–16 scored actions, bounded); masks written to notes as a structured legend prepended to every prompt; avatar tagged `@` in the text grid; a standing service re-scores contingency continuously to catch control-transfer mechanics; 2-action mini-probe after each level-up (bodies change between levels). Masks feed rank 4's differ, rank 5's predicates, rank 7's state hashes.

**Cheapest test.** Zero live actions: run the contingency classifier over logged early-run (frame, action, next_frame) triples from ~10 dev games against hand-labeled avatars. Go/no-go: ≥80% avatar ID on avatar games, no confident SELF on pure click puzzles, <1 destructive click per game.

**Expected effect.** Directly converts "goal reasoning about a scene whose controllable element was never isolated" — the canonical zero-bank texture on navigation games — into a solved sub-problem for ~15 actions. Expect most of its gain concentrated on the avatar/Sokoban minority of games, but the fast click-mode negative saves dozens of wasted actions on the majority.

---

### 3. Probe-then-Dispatch Archetype Portfolio — genre recognition as a measurement
*Merges: Return42 portfolio (7/7); consumes rank 2 as its probe stage; subsumes the Athey schema vocabulary as its classification language.*

**What it is.** From the opening battery's measured feature vector — {controllable component found?, deterministic (identical action from identical state-hash reproduces)?, click-reactive?, undo-functional?, objects moving while idle?} — a decision tree commits the run to one of ~4 archetype scaffolds: AVATAR-NAV, SOKOBAN-PUSH, CLICK-PUZZLE, TURN-BASED-HIDDEN-STATE. Each scaffold is a different prompt, macro vocabulary, and weighting of the other mechanisms.

**Human basis.** Humans complete novel games in minutes because most of the "learning" is a single genre classification plus a wholesale library import ("this is a Sokoban" → entire control scheme and objective schema arrive at once). Novice studies show human performance collapses exactly when a game defeats genre priors — i.e., genre lookup IS the human speed advantage.

**Implementation.** Build the probe+classifier first and gate everything on its measured accuracy. Escape hatch is mandatory (the sweep's sharpest objection): misclassification is sticky for 2.2 hours, so a scaffold that produces zero new effect-table rows for K actions triggers re-classification with the offending archetype excluded. Start with two scaffolds only (CLICK-PUZZLE vs AVATAR-NAV) — the four-scaffold build is weeks; the two-way split captures most of the variance since CLICK dominates 19/25.

**Cheapest test.** Build only probe+classifier, run offline on all 25 public games, score archetype accuracy against known ground truth. One day; <80% accuracy kills it before any scaffold is written.

**Expected effect.** High if classifier accuracy holds: every game currently pays hundreds of actions to implicitly discover facts the probe gets in 12. The compounding effect — right macro vocabulary from action ~15 onward — plausibly rescues a meaningful slice of zero-bank runs in both major genres.

---

### 4. Action-Locked Transient Map + SURPRISE Comparator — never ask the LLM to spot the difference
*Merges: anti-change-blindness diff objects (9/6), VoE-dwell, precision-weighted prediction-error router; provides the trigger for rank 5's interrupt.*

**What it is.** Two coupled disciplines. **(a)** Never hand the model two raw grids: a deterministic pass computes the HUD-masked diff, labels connected changed components, detects translations via cross-correlation, and prepends 3–10 structured lines ("CHANGE #2: 4×3 region moved (+1,0)") *above* the grids. **(b)** Corollary discharge: before each action the model must emit a machine-checkable `predicted_diff` (region + change-type — vacuous predictions rejected by the validator); the sandbox compares prediction to reality and on mismatch injects a SURPRISE block with a mandatory notes/schema edit before the next action.

**Human basis.** Rensink's flicker paradigm: human change detection is carried almost entirely by the motion transient, not by memory comparison — the agent's two-static-grids condition is literally the setup used to *induce* change blindness in humans. And efference copy (von Holst & Mittelstaedt): self-caused changes are checked against a forward prediction, so mismatches pop out for free. The agent has neither the transient nor the comparator.

**Implementation.** Pure plumbing over existing diff primitives plus prompt architecture; feasibility 9. HUD/autonomous masking from rank 2 is a hard prerequisite or SURPRISE fires every turn. Don't aggressively strip confirmation context on surprise=0 turns (the token-diet hidden-crater finding warns against it).

**Cheapest test.** Zero live actions: 100 logged transitions, ask the current model "what changed?" with raw grids vs injected change-list, score against ground truth. Falsified if raw-grid change detection is already accurate. Secondary check: fraction of predictions that are region-specific and wrong at least once (<30% ever-wrong = the model is sandbagging and the signal is dead).

**Expected effect.** Broad, moderate per-game, near-universal coverage: every downstream mechanism reasons over correct percepts instead of hallucinated ones. Its real payoff is as the sensory front-end that makes ranks 5 and 6 trustworthy.

---

### 5. Hypothesis Ledger with Cheapest-Discriminator Rule + Unexpected-Observation Interrupt — one implementation for the whole EIG family
*Merges (the sweep's four-way family, judged as one bet): Hypothesis Ledger (6/7), Explicit EIG experiments (6/7, best-specified), Model-Disagreement Probing, Plan2Explore, EMPA-exploration, Reconstruct-the-Manual, Dueling mechanism programs, Superstition audit (as its evidence-accounting columns).*

**What it is.** A JSON ledger in the sandbox: each entry = {hypothesis, weight, *sandbox-evaluable predicted-diff predicate*, interventional_confirmations, contradictions, contrast_tested}. Harness rules: every action is tagged EXPLOIT or PROBE (probes name the hypotheses they discriminate, with the predicted diff written BEFORE acting); the Python differ auto-adjudicates predictions and flips statuses — the scoreboard, not the LLM's narrative confidence, decides which theory the planner may condition on; probe selection = argmax(hypotheses discriminated / action cost). Plus the single sharpest atom in the sweep: an **unexpected-observation interrupt** — any outcome no live hypothesis predicted (fired by rank 4's SURPRISE) forces a hypothesis-revision turn before any further progress actions. Include the EMPA fragment: a small closed win-predicate library (all-targets-filled / reach-X / match-template / …) tested against the one observed level-up — the only supervision bit the benchmark gives, spent deliberately.

**Human basis.** Children design deconfounding interventions spontaneously (Cook, Goodman & Schulz); Platt's strong inference; and the measured 2026 LLM-agent finding that agents *discover* task-relevant information at 79–98% rates but *act* on it at 0.5–50% — the discover-but-ignore gap is plausibly the single largest identified share of the zero-bank mode. Humans poke the anomaly immediately; the agent notes it in prose and wanders off.

**Implementation.** No executable world model required — hypotheses need only one-step observable predictions. Guards: bounded-retry escape on tag rejection (no deadlock); predicates must be cell/region-level falsifiable claims (anti-vacuity validator shared with rank 4); superstition-audit columns gate multi-step plans (≥2 interventional confirmations + 1 contrast) while letting provisional rules guide probes (bootstrap tier, avoiding opening deadlock).

**Cheapest test.** Offline on 3 logged games: seed a hand-written ledger, replay the transcript, measure the differ's auto-adjudication rate vs hand judgment; then measure on zero-bank transcripts what fraction of turns had an action available that would have discriminated ≥2 live hypotheses — that fraction upper-bounds the gain before any live run. Diagnostic in the A/B: fraction of surprises that trigger revision vs get ignored.

**Expected effect.** This is the mechanism that turns "acting consistent with many world-models and learning nothing" into one falsification per action. Highest leverage on hidden-state and conjunctive-mechanic games — exactly the ones that zero-bank hardest — with the known caveat that 27B models degrade on conjunctions, mitigated by the closed win-predicate library and schema-level (rank 3) priors shrinking the hypothesis space.

---

### 6. Learned Local Forward Model with Accuracy Gate — simulate before you spend
*Distinct from the LLM-authored world-model cluster (see rank 8): this one is fitted by classical code, not written by the model.*

**What it is.** A standing sandbox module that fits a per-cell local transition model (cell value ← 3×3 neighborhood + action) from all logged triples, reports its own held-out accuracy per action type, and exposes `rollout(state, action_sequence)`. Policy: once accuracy for movement actions clears ~99% of cells, the LLM must plan multi-step sequences by simulated rollout and execute only the winning sequence — BFS/A* to a named target executed as an uninterrupted macro (the YOLOBOT discipline), monitored only for prediction mismatch, abort-and-replan on surprise.

**Human basis.** Humans build a predictive simulation within seconds ("press right, I slide to the wall") and plan against it rather than paying real actions for consequences. The agent re-queries the scored environment for every consequence a model would give free.

**Implementation.** Dockhorn & Lucas: one observed transition yields ~4096 local training rows, so deterministic movement dynamics converge in a handful of frames. Scope honestly: movement actions only at first (ACTION6's 4096 click parameterizations need click-relative encoding — defer); the accuracy gate is load-bearing (planning against a plausible-but-wrong model is worse than none); the model will be confidently wrong exactly on hidden-state/global-toggle games — the gate keeps it off those. Rank 4's SURPRISE machinery doubles as the mismatch monitor during macro execution.

**Cheapest test.** Entirely offline: fit per dev game from existing logs, report held-out per-action cell accuracy. The accuracy table *is* the per-game enable list. Falsified if <99% movement accuracy on fewer than ~8 games.

**Expected effect.** On the games where the gate opens (likely most avatar/Sokoban games), converts N scored exploratory actions into 1–2 verified plans — attacking depth AND efficiency simultaneously. Also the natural substrate for rank 5's rollout-based probe selection.

---

### 7. A-not-B Inhibitor — a hard perseveration brake, in the architecture not the prompt
*Merges: A-not-B Inhibitor (8/6), DCCS set-shift (as the schema-level tier), IW(1) novelty droughts (as the trigger bookkeeping), MaastCTS2 cycle-blacklist (soft form only).*

**What it is.** A response-strength ledger per (masked-state-signature, action): any pair producing k=3 consecutive null diffs is *stripped from the action menu shown to the model* (with a one-line explanation), decaying when the scene hash changes. Plan-level tier: a level attempt ending in reset with zero relational novelty bans its own opening 5-gram next attempt, forcing a genuinely different opening. Overlay IW(1) atom-novelty bookkeeping (coarse-region atoms over rank 2's masks): a novelty drought ≥ k forces a macro-family switch. Hazard memory ships as a **soft** veto only (warn string, overridable with stated reason) — the sweep's check on winning traces showed hard vetoes on revisits/repeats would block required moves in hidden-state games.

**Human basis.** The A-not-B error: infants reach where the toy *was*, despite watching the switch — knowledge present, prepotent response wins anyway (Diamond: an inhibition failure, not an ignorance failure). LLM transcripts show the identical dissociation: "this isn't working" followed by the same action, because autoregressive n-gram habit beats stated knowledge. Development fixes this with external scaffolding of the switch; advice-level anti-perseveration is known to fail — hence menu filtering, a prosthetic PFC.

**Implementation.** ~100 lines of wrapper. Two documented sharp edges: state-signature hashing needs rank 2's masks or "same state" never (or always) hashes equal; delayed-effect actions look null for k steps — so vetoes decay and the drought trigger must be overridable when a live ledger hypothesis (rank 5) explicitly predicts delayed effect (e.g., charge/cycle mechanics that *require* repetition).

**Cheapest test.** Pure log mining, $0: fraction of all scored actions in zero-bank runs that were repeats of a (state-hash, action) pair already at ≥3 consecutive null diffs (>15% = real headroom), and how often winning traces contain such repeats (false-positive bound). Also: do novelty droughts discriminate zero-bank from banking runs in past logs? If not, the trigger has no signal and dies free.

**Expected effect.** Bounded but cheap and near-certain: reclaims the budget currently burned on oscillation loops and dead repeats — likely 10–20% of scored actions in stalled runs — and converts it into rank 1/5 probes. Doesn't decode mechanics itself; buys the budget that does.

---

### 8. LLM-Induced Executable World Model — the capped, highest-ceiling bet
*Merges (three submissions of one idea + a fourth variant): symbolic forward model + search (5/8), theory-based world modeling (5/8), executable Python world model with verifier-replay (5/8), Sleep Replay consolidation (5/7).*

**What it is.** On level-death, reset, or every ~50 actions: a SLEEP turn where the model's only task is to write/refine `step(grid, action)`, graded by the sandbox against the *entire episode log* in Mattar-Daw priority order (largest error, nearest to level-ups first), with a pytest-style **first-divergence counterexample report** (the best engineering detail in the cluster — 27Bs debug against concrete counterexamples far better than against "accuracy dropped"). Prose notes are regenerated from the simulator ("facts my code encodes"), purging unfalsified folklore. At ~90% replay accuracy, unlock vicarious trial-and-error: free simulated rollouts, only winning sequences spent for real.

**Human basis.** Hippocampal replay between lives consolidates episodes into a model that supports mental simulation (Foster & Wilson; Mattar & Daw); sleep triples hidden-rule discovery (Wagner et al. 2004). The novice→competent transition IS the moment the action budget flips from exploration currency to execution currency.

**Implementation — the guards are the design.** Hard per-game abandonment budget (token-capped; on expiry revert to baseline policy permanently for that game); partial simulators (70% accuracy) are never used in VTE mode; expect the gate to simply never open on the 13/25 games where the frame is not a sufficient statistic — that is priced in, and rank 5's ledger with history-conditioned hypotheses is the fallback organ there. Token economics under 25-way concurrency cap sleep rounds at 2–3 per 50 actions.

**Cheapest test — run once for the whole cluster, this week, $0.** Feed the actual frozen 27B (same serving config) logged episodes from 10 dev games, sleep prompt + grading loop, 3 rounds each; report per-game transition accuracy and token cost. If the model can't clear ~80% on simple deterministic games, the entire cluster's ceiling is unreachable *with this brain* and dies for free. If it clears, separately verify the planner half (BFS finds known winning sequences inside a hand-verified model) — the published baseline's own failure mode was good-model/weak-planner.

**Expected effect.** Bimodal by design: zero on gate-never-opens games (cost capped), step-change on games where it opens — search replaces groping, which is the zero-bank pathology itself. The published sibling recipe solved 7/25 with mean RHAE 32.6%; treat that as the realistic, not the hoped-for, ceiling.

---

### Build order and dependency spine

Rank 2 (masks) → rank 1 (`run_probe`) → rank 3 (dispatch) form one probe-infrastructure package and should ship together; rank 4 (differ/comparator) is independent plumbing and can ship first; rank 5 consumes 4's SURPRISE as its interrupt; ranks 6 and 8 are gated consumers of the logs everything else generates; rank 7 is a standalone wrapper. Five of the eight cheapest tests require **zero live actions** — a single offline week adjudicates most of this report.

---

## Part 2 — Honorable-Mention Wild Cards

**Undo-fidelity audit → `peek()` + reversibility-residue probe** *(merged pair)*. The reframe is genuinely original: undo is a free 1-ply forward-model query, and — sharper — changes that *survive* act-then-undo are a hidden-state detector, directly aimed at the frame-insufficiency killer. Everything hinges on unknown per-game undo semantics, so the instrument needs a probe of the instrument: one offline script (5 random states × every dev game, act-undo, compare masked hashes) yields a per-game undo-fidelity table for free. If fidelity holds broadly, `peek()` becomes a probe primitive inside rank 1's batteries and the residue map feeds rank 5's ledger; if not, it demotes to opportunistic use at UNCERTAIN decision points in depth-weighted levels.

**Declared Sacrifice Run (scout dirty, execute clean).** Pre-declared mode split where completing the level during SCOUT is a discipline failure: run batteries, catalog effects, deliberately observe loss semantics, then RESET and EXECUTE from the distilled route only. The campaign's own action-accounting arithmetic (attempts accumulate; early-level sacrifice net-positive) already endorses it. Folded partially into rank 1 as its "where batteries run" guard; the full mode split with the take-the-free-win escape hatch is worth its own dev-game A/B — 5 seeds, fixed budget, free on the local engine.

**Coin-Operated Death.** Buy the lesson: when discovery stalls, trigger the suspected failure once with a pre-registered question, capture the terminal frames, write the rule, reset (cost ≈ 1 action). Sound insight — confusion is the most expensive state in the system — but it presupposes an articulable hazard hypothesis, which the audit suggests exists in a minority of fully-confused runs. Ship as a 2-per-game-capped trigger inside rank 5 once the ledger exists; the offline check (do death frames identify the killer on known-hazard games?) costs an afternoon.

**HUD Literacy (counters as the smuggled reward channel).** The masking half is done; the novel half — parsing non-clock HUD scalars whose jumps coincide with board events as a proto-reward — is one offline census away from a verdict: count dev games with a non-monotone counter. Fewer than ~4 kills it; more, and it hands rank 5 a genuine intermediate objective in a benchmark that claims to offer none.

**Human replays (342 traces).** The only idea grounded in real human play on this exact benchmark, but per-game opening books are worthless on the private 55-set and the LB can't measure an SFT delta (sd ≈ 0.17–0.195 needs ~45 draws/arm). The defensible extraction is genre-level behavioral *budgets* — median actions-to-first-state-change, probe-before-commit ratio, error-recovery latency — injected as runtime gates ("humans median 12 actions here; at 3× that, reset and re-hypothesize"). Hour one: verify the artifact actually downloads and parses before spending anything else on it.

**PSRL commit-per-attempt.** One sampled theory per level attempt, no mid-attempt vacillation unless falsified, resample on reset. Entirely parasitic on rank 5's ledger, but the scored-action economics genuinely favor it (a clean attempt-long test of one theory is cheap under accumulating attempts) and it attacks per-step dithering and first-hypothesis tunnel vision at once. ~30 lines once the ledger exists; the scripted no-LLM offline comparison decides it.

**Gestalt Scene Grammar.** Lattice-pitch detection + deviant-tile + broken-symmetry captions: on pattern-click games "tile (2,3) differs from the other 18" often *is* the solution, and it converts the model's weakest faculty (global structure over 4096 cells) into 15 deterministic lines. Held out of the Top 8 only because a confident-wrong structural caption is worse than silence; the zero-action test (caption all 25 initial frames, hand-score hit rate vs confident-wrong rate, >50%/<10% gate) should run in the first offline batch, and on a pass it slots into rank 3's CLICK-PUZZLE scaffold.

---

## Part 3 — Cross-Cutting Themes: what humans do that this agent structurally cannot

**1. Humans settle ontology before teleology; the agent does the reverse.** Which pixels am I, what can I move, what moves by itself, what's UI — humans answer these in seconds via self-generated contingency evidence, *before* any goal reasoning. The agent goal-reasons from frame one over an unparsed scene, so on bad runs every subsequent inference inherits a wrong ontology. Half the sweep's survivors (all of the contingency/wiggle/common-fate/dispatch cluster) are one message: the first 15 actions of a game are for measurement, not for hope.

**2. The exchange rate between thought and act is inverted.** Humans run dozens of micro-experiments per verbal thought; the agent runs one expensive thought per single act. This is the deepest structural gap — not intelligence but architecture — and it's why `run_probe` ranks first: it's the only idea that changes the units of deliberation rather than the content.

**3. Humans have a comparator; the agent has a narrator.** Efference copy gives humans free, automatic prediction-mismatch signals, and surprise *seizes* their next actions (Stahl & Feigenson's infants probe the specific violating object). The measured LLM-agent pathology is discover-at-98%, act-at-1%: anomalies are noticed in prose and abandoned. Three independent lenses (perception, developmental, LLM-agent-empirics) converged on the same fix — machine-checked prediction before every action, and a mandatory interrupt that binds surprise to the next action. That convergence is the sweep's strongest signal.

**4. Human "learning" a novel game is mostly classification against a closed prior vocabulary.** Genres, play schemas, HUD conventions, win-condition archetypes — humans search a short list, not open mechanic-space. The benchmark is designed to defeat *specific* priors but its games still sit inside the genre/schema vocabulary. Closed vocabularies (archetype dispatch, win-predicate library, schema scoreboard) buy the same search-space collapse — at the priced-in cost of confident misclassification, which is why every one needs an escape hatch.

**5. Knowing is not doing: stated knowledge loses to habit strength, so fixes must be architectural.** The A-not-B dissociation appears verbatim in transcripts ("this doesn't work" → same action), and the sweep's verdicts document every prompt-level compliance rule being gamed — ritual schema citation, vacuous predictions, boilerplate anomaly rejections. The recurring design law: **enforcement lives in code the model cannot argue with** — menu filtering, harness-rejected untagged actions, sandbox-adjudicated predicates, scoreboards that outrank narrative confidence. Anything enforced by asking will be satisfied performatively.

**6. Humans accumulate across deaths; the agent's beliefs are episode-local prose that never meets its own evidence.** Human deaths are tuition; a wrong belief written on turn 5 of an agent run survives to turn 200 because nothing replays notes against the log. Executable, self-verifying knowledge — ledger predicates, route-file assertions, graded micro-simulators — is the antidote across three clusters: notes that cannot rot because every replay re-validates them.

**7. The unclosed residual: hidden state.** Frames are insufficient statistics in 13/25 games, and every mechanism above degrades there — forward models won't gate open, state hashes mislead, frame-conditioned hypotheses falsely converge to "noise." The partial answers (history-conditioned hypothesis features, undo-residue probing, turn-based-hidden-state as a first-class archetype) are the weakest part of this portfolio, and honest accounting says so: if the Top 8 lands, the zero-bank mode should compress substantially on visible-mechanic games, and hidden-state games become the next dominant failure — which is where the following sweep should aim.
