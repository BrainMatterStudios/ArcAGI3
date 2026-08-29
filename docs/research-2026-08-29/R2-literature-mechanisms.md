# ARC-AGI-3: how the big jumps were made, and what transfers to an offline 27B agent

Independent public-source review, 2026-08-29. No project files or memory were read. Every number below has a URL; where I computed something myself (human-baseline stats, budget arithmetic) I say so.

Constraint frame used for every "transferability" rating: Kaggle rerun, no network, one ~96 GB GPU, 9 h wall clock for ~110 games (about 295 s per game, my arithmetic), Qwen3.8-27B FP8 on vLLM as a per-action text agent, currently ~1.7 on the public Kaggle LB vs 5.99 for the leader.

Rating scales:
- Evidence: Strong = controlled ablation or verified score; Medium = self-reported with details/traces; Weak = blog claim, no ablation; None = no published evidence.
- Transfer to our setting: High = model-agnostic and cheap enough for 295 s/game; Medium = plausible but depends on 27B capability or costs wall clock; Low = needs frontier reasoning or a compute budget we do not have.

---

## 0. The scoring and budget facts that shape everything

Sources: docs.arcprize.org/methodology ; technical report arXiv 2603.24621 ; Schema traces scorer README (huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces) ; Psyho on the April rule change (x.com/FakePsyho/status/2044140048019276093).

- Per level: score = min(115, 100 * (h/a)^2); level weight = 1-indexed level number; game score = min(weighted mean, completion cap) where completion cap = 100 * sum(weights of completed levels) / sum(all weights); benchmark = mean over games. (Exact formula reproduced in the Schema score_trajectories.py, lines 293-318.)
- Human baseline = upper-median first-run human per level (moved from "2nd-best human" to median in April 2026; per-level cap raised 100 -> 115 at the same time).
- Hard cutoff for agents: 5x the human median actions per level, then the agent is terminated on that level (tech report).
- Human calibration: 486 participants, 414 candidate environments, 2,893 attempts, median attempt 7.4 min; sessions with <30 actions or >30 min excluded; participants could reset a level at any time; 90-minute paid sessions ($130 + $5/env). Public demo set: 342 replays, 145 solves (arcprize.org/blog/arc-agi-3-human-dataset).
- Per-level human baselines for the 25 public games (downloaded from the Schema dataset baseline_actions.csv; stats are my computation): 183 levels, 17,135 total baseline actions, mean 93.6 actions/level, median 60, min 6 (sc25 L2), max 578 (dc22 L6). Per-game totals range 171 (cd82) to 1,843 (wa30). Level-1 baselines are small (7-78) and later levels are large; combined with linear level weights, this is why depth (clearing levels 3+) dominates efficiency for any agent under ~30.
- RESET accounting: the methodology page and the tech report do not state whether RESET counts as an action; recordings log RESET as a distinct action entry with a full_reset flag (docs.arcprize.org/recordings). Retrodict's prompt treats RESET as expensive and warns "never issue two RESETs in a row" because a second RESET on a fresh attempt restarts the whole campaign in their runner. Flag: sources are silent/inconsistent on reset counting; treat it as costing at least one action and possibly the whole level's action history.
- Public-25 is a weak proxy: AERA shows every public game is reachable by non-intelligent strategies (Section 5), and ARC Prize's own harness experiment shows per-game bimodality (Section 6). The Kaggle public LB is scored on a subset of the private games.

Budget arithmetic for us (mine): 9 h / 110 games = ~295 s/game. Top harnesses spend ~600-1,060 LLM calls per game (Tycho), 2 h per game (NOOA), 45 min per game (Duck default). We cannot afford an LLM call per environment action; a 27B at a few seconds per call gets roughly 75-150 calls per game. Every mechanism below is rated against that.

---

## 1. Tycho (NIMI Research, Apache-2.0)

Sources: arxiv.org/html/2607.28287v1 ; github.com/NIMI-research/Tycho ; arcprize.org/leaderboard/community (100.0%, $2,986, verified on semi-private).

What the loop does:
- Roles: Actor (the only thing that acts; gets the current "decision frame" + action mask each turn), World-Model Builder (a focused subagent invoked conditionally), Workspace API (disk, level_L/turn_NNN, a wmlib library that parses history into typed records), Verifier (replays the program against recorded transitions: accepted-match rate, coverage, outcome classification), Planner (BFS from replayed model states to level_complete; returns routes with frame hashes).
- Frame typing is the key model-agnostic idea: every frame is tagged decision / transient (animation) / terminal / reset / next-level-init. Transient frames are not decision points.
- Actor prompt contents: typed interaction record, accumulated task memory, current executable model if any, earlier-level summaries. Tycho clears the conversation at each level boundary; level summaries, actor beliefs, builder notes and the model survive in the workspace.
- Builder prompt: filtered transition evidence, prior hypotheses, verifier feedback, and a Python contract (init_state, transition, render, outcome, optional observation_variants and planner hooks).
- Plan gating: a builder-surfaced plan is only followed if the observed frame's hash matches the plan's stored hash and the action is in the mask; otherwise the actor deliberates. One action per deliberation; no queued actions.
- Four policies: no_world_model / single (actor writes models itself) / orchestrator (actor asks for the builder) / trigger (harness auto-invokes the builder on verifier failure, new level, fatal reset).

Numbers (matched study, Opus 4.8 xhigh, 25 games, $750/game cap, 3,500 LM calls/game cap, 24k-token answer budget, 40 tool steps per turn):

| Policy | Builder calls | Accepted transition match | Levels | RHAE |
|---|---|---|---|---|
| no model | - | - | 157 | 79.07 |
| single | 0 | 13.9% | 162 | 85.36 |
| orchestrator | 147 | 16.2% | 166 | 88.49 |
| trigger | 1,192 | 88.1% | 162 | 83.07 |

Frontier orchestrator runs: GPT-5.6 Sol 100.00 RHAE, 7,766 actions, ~1,060 calls/game, ~$179/game; Opus 5 100.00, 6,641 actions (61% fewer than the aggregate human baseline), ~604 calls/game, ~$119/game.

What the ablations say:
- Simulator accuracy does not buy score. The trigger policy has 88.1% accepted transition match vs 16.2% for orchestrator, yet scores lower (83.07 vs 88.49). The paper: "accurate transition prediction alone does not ensure strong gameplay." Terminal-outcome recall (predicting win/loss) is weak under every policy.
- The world model is worth ~9 RHAE on top of a strong actor with typed history (79 -> 88). The typed evidence workspace + tools is worth far more: Opus 4.8 scores 1.5% on the official harness (benchlm.ai/benchmarks/arcagi3) and 79 with Tycho's no-model policy.
- "Pre-action executable model" rates: single 89.3%, orchestrator 83.5%, trigger 52.0% -- automatic repair spends the budget rebuilding models instead of acting.

Context sizes: not stated per call; the system relies on 24k-token answers and ~600-1,000 calls per game, and clears context per level. Repo supports only Anthropic and OpenAI transports; no local-model path documented.

Model-agnostic parts: typed-history protocol, Moore-machine formalization, wmlib evidence interface, verifier, BFS planner, plan gating by frame hash, per-level context clearing with summaries, the four policy patterns.

Ratings: Evidence Strong (matched 4-policy ablation, verified score). Transfer: the full system Low (600-1,000 frontier calls/game; the builder needs a model that writes correct simulators). Individual parts High: frame typing (transient vs decision), plan gating by frame hash, level-boundary context reset with summaries, and the lesson to not over-invest in model repair.

---

## 2. NVIDIA AVO

Sources: developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-... ; AVO paper arxiv.org/pdf/2603.24517 (the paper is about CUDA-kernel evolution; ARC-AGI-3 is only in the blog).

Mechanism (as much as is public):
- Input is text-only: "an exact 64 x 64 text grid, with no images or image tokens sent to the model." Built on Claude Opus 5 (bare model ~30% official).
- Loop: form hypothesis, act, observe, update state, continue. The agent is NVIDIA's general-purpose coding agent (planning, tool use, persistent memory via conversation history plus a committed lineage; in the kernel setting each committed version is a git commit with its score).
- Persistent memory carries "prior implementations, evaluation results, ... and accumulated reasoning, allowing the agent to resume from the current state rather than repeatedly reconstructing the search."
- Supervisor: from the paper (Section 3.3), a "self-supervision mechanism" that detects two failure modes -- stalling after exhausting a line of exploration, and unproductive edit cycles that repeatedly fail to improve -- then "reviews the overall evolutionary trajectory and steers the search toward several candidate optimization directions." No thresholds or metrics disclosed.
- Result: 183/183 levels, 6,624 actions (VISTA 7,542). The blog itself says the 12% action difference "should not be interpreted as a controlled ablation." No code, no ablation.

Ratings: Evidence Weak-Medium (100 on public, but no ablation, no code). Transfer: the stagnation-supervisor idea is High as a cheap rule-based component (no new frame hash for N actions, no level-up for M actions, or repeated action loops -> inject a redirect); the quality of the redirect on a 27B is untested.

---

## 3. VISTA (MIT)

Source: vista-research.github.io (authors Han, Hu, Qiu, Wu, He; no arXiv link found).

Mechanism:
- Perception: 512x512 PNG = 8x nearest-neighbour upscale of the 64x64 frame with 1-px grid lines. The model is never told the world is a 64x64 grid. Tools: inspect (re-view any earlier state, intermediate animation frames, or a region) and read_pixels (exact colors of a region). Read-only tools are free under the scoring rules.
- Reasoning: Observe -> Reason -> Act; the model is asked to predict the expected visual outcome before acting; one environment action per turn.
- Memory: "lossless visual memory" -- every returned frame is stored and indexed by turn/frame. Two markdown notes: GUIDE.md (durable, cross-level understanding) and WORKING.md (level scratchpad). Near the context limit the model writes a "concise continuation state" and resumes in a fresh context with notes + visual memory intact.
- Prompt: <50 words of explicit instruction.
- Results: Opus 5 (via Claude Code CLI) 100.00 RHAE, 25/25 games, 7,542 actions (56% fewer than first-time humans); GPT-5.6 Sol (Codex CLI) 98.27, 10,063 actions. Official no-harness GPT-5.6 Sol on the public set: 13.33.
- Ablation: only a modality ablation on two games (s5i5, cd82): text grid, 2D image and 3D render all succeed. No memory/tool ablations. No smaller models.

Ratings: Evidence Medium (self-reported perfect score; a thin ablation). Transfer: image input Low-Medium for us (needs a VLM; Kaggle-milestone entries Reki/forge did use Gemma-4-31B vision locally, so it is feasible, but VISTA's own ablation says the text grid works). Notes files + predict-before-act + continuation-state compaction: High and cheap.

---

## 4. OPINE-World

Source: arxiv.org/html/2607.01531v2 ; community leaderboard 78.4%, $1,040, self-reported.

Mechanism:
- Two agents, coupled only through a shared replay buffer and structured handoffs. Action agent plays live, reads logs via filesystem tools, runs short analysis scripts, picks actions. Synthesis agent maintains game_engine.py (transition_function, reward_function, extract_objects, optional planner) and never touches the action agent's artifacts.
- Objects are (key, type, attributes); sprite-based when the engine exposes sprites, otherwise a synthesized extract_objects(frame). Pairing across timesteps: exact (name,x,y) match, then greedy same-name matching.
- CEGIS: synthesize on init, then only on a live prediction mismatch; acceptance requires exact replay of every recorded transition; two independent runs to detect hidden state; a deferral window batches counterexamples; a stall guard stops thrashing. No optimism constraint, no likelihood scoring.
- Ontology error (the exploration driver, and model-free): map each object's changed-attribute set to an effect signature ("no_change", "x", "pixels,x", "gone"); keep a Dirichlet count table per row (type, action, context); row uncertainty = normalized entropy of the effect distribution; type uncertainty = entropy of the type posterior; aggregate error = mean. The action agent probes high-entropy objects; the synthesizer refines types or adds context features when a row stays high-entropy.
- Planner: bounded forward search over the verified model; activates only after a level has been completed; validated offline on completed-level entry states; executes one step at a time; any mismatch becomes a counterexample and returns control to the action agent.
- Results (Opus 4.8): 20/25 games, 160/183 levels; on 16 of 20 wins uses fewer actions than the human reference; beats baseline1 (GPT-5.5) 78.4 vs 63.8 on their metric, and solves six games baseline1 abandons in about a quarter of the actions (2,578 vs 10,874). Context discipline: per-game token thresholds, structured handoffs, session resets (Appendix A.6; budgets not disclosed). No ablations ("heavy interdependency").

Ratings: Evidence Medium (self-reported, detailed, no ablations). Transfer: the full CEGIS loop Low-Medium (needs a model that writes exact simulators; 27B unproven). The effect-signature/entropy table is High: it is pure counting over connected components and can drive exploration and dead-action detection with zero LLM calls.

---

## 5. AERA / "Explore Before You Solve" (arXiv 2605.25931, Liew Keong Han)

Source: arxiv.org/pdf/2605.25931 (PDF text extracted locally). Code CC0 at github.com/farmountain/aera-arc3-paper.

What is exploitable on the public set (Tables 8-9):
- 10 games win with one blind action: ACTION6 depth-1: FT09, CN04, M0R0, LF52, BP35; other single action: R11L, VC33, LP85, TN36, S5I5.
- 5 games win with ACTION6 after one probing observation: SB26, CD82, AR25, SK48, DC22.
- Repeated single action with enough budget (no LLM): TU93 ACTION1 x50, RE86 ACTION1 x100, TR87 ACTION1 x128, KA59 ACTION6 at (32,32) x100, LS20 ACTION2 x129, SC25 ACTION6 at (24,48) x52, G50T ACTION1 x130, WA30 ACTION1 x200, SP80 ACTION1 x30+. SU15 needs "diverse exploration" (ReAct 50 steps).
- Note these are first-level wins; the paper's "solved" is level-level, not game-level. It also reads hidden-game metadata tags ("keyboard", "keyboard click") and human baseline totals (350-1,843) to size budgets.
- Library bug: ACTION6 with {"x": None, "y": None} raises a TypeError in the engine's camera-coordinate code; the arc_agi wrapper's exception handler returns it as a WIN. 18/25 public games "win" this way on local arc_agi v0.9.8. The paper says it is "not confirmed to work on the Kaggle competition server, which may use a different library version or have patched the null-coordinate path," calls RHAE undefined for crash-wins, and recommends reporting it. Do not build on it.
- Their small-model results: Qwen2.5-0.5B on Kaggle CPU FP32, EXPLORE/VERIFY/PLAN with an explore budget max(5, min(30, floor(0.4 * H_level1))): RHAE 0.2116 on 25 games (their nonstandard per-level average, 4/25 first levels), random and no-explore 0. Forcing ACTION6 as the first exploratory action turned CN04/M0R0/LF52 from 0/5 to 5/5 runs -- the failure was action-selection bias toward ACTION1, not reasoning. 1.5B did worse than 0.5B (more concentrated action distribution). Showing 6 structured trajectories before acting solved 0/8 at both sizes because the text observations ("state=NOT FINISHED lv=0 win=0") were too sparse.
- Disagreement flag: the paper and one Medium summary describe their Kaggle BFS+pre-solve-cache entry as "RHAE = 0.30 (30%)" and as beating the 12.58% community best. The live Kaggle public LB (fetched via the kaggle CLI on 2026-08-29) tops out at 5.99, so "0.30" is almost certainly 0.30 on the Kaggle percentage scale, not 30%. Treat that claim as a unit error.

Ratings: Evidence Strong for the taxonomy (deterministic search, reproducible), Weak for the agent. Transfer: High for two lessons -- (a) at the start of every level, deterministically try each available action once and probe ACTION6 on salient objects before asking the LLM anything; (b) the 5x-human action budget on private levels makes "repeat the right action" a real strategy for keyboard games, so a cheap "repeat last progress-making action" policy is worth having.

---

## 6. ARC Prize's own harness experiments and official analysis

Sources: technical report arxiv.org/html/2603.24621v1 ; arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis ; arcprize.org/blog/arc-agi-3-preview-30-day-learnings ; docs.arcprize.org/llm_agents ; arcprize.org/leaderboard/community ; symbolica.ai/blog/arc-agi-3 ; openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/ (403 for me; numbers via gigazine.net/gsc_news/en/20260731-how-enabling-two-settings-tripled-arc-agi-3-benchmark/).

- Official harness: system prompt "You are playing a game. Your goal is to win. Reply with the exact action you want to take." No tools. The template LLM agent keeps a 10-message history. Launch scores on the semi-private set: Gemini 3.1 Pro 0.37, GPT-5.4 0.26, Opus 4.6 0.25, Grok 0.00; later Opus 5 30.16 (verified), GPT-5.6 Sol 7.8, Opus 4.8 1.5, GPT-5.5 0.4.
- The 0% -> 97.1% experiment: ARC Prize sponsored Duke to build a harness on ls20/ft09/vc33 and then tested it on the full public set. Duke's harness lets the LRM "execute arbitrary Python code to selectively retrieve and transform information from its action history." On TR87 Opus 4.6 went 0.0 -> 97.1; on BP35 0.0 -> 0.0. ARC Prize's conclusion: harness gains are bimodal per game and do not generalize, so the official leaderboard only counts bare API calls. Symbolica's Arcgentica (orchestrator that never touches the env; subagents return compressed text summaries): 36.08% public, 113/182 levels, 7/25 games, $1,005 with Opus 4.6.
- OpenAI's two settings (GPT-5.6 Sol, public set): keep the model's reasoning across turns instead of discarding it, and compaction instead of rolling truncation. 13.3 -> 38.3 with ~6x fewer output tokens. The stated failure of the default loop: the model "keeps starting over" every action.
- Official failure-mode analysis (GPT-5.5 0.43, Opus 4.7 0.18 with reasoning audits): (1) "true local effect, false world model" -- the model sees what an action does but never abstracts a rule; (2) wrong abstraction imported from training data (calls games Tetris/Sokoban/Frogger from a visual resemblance and plays that game); (3) "solved the level, didn't learn the game" -- lucky early wins carry wrong theories into later levels. Opus 4.7 = "wrong compression" (confident wrong theory, aggressive execution); GPT-5.5 = "failure to compress" (many hypotheses, no commitment).
- Preview lessons: StochasticGoose (CNN frame-change predictor, RL) 12.58% with 255,964 actions; on the full benchmark at launch it fell to 0.25%. Humans show "minimal exploration phases followed by efficient execution."
- Community leaderboard (public set, 2026-07/08): Tycho 100.0 ($2,986, verified), Retrodict 99.9 ($654, verified), baseline1 99.0 ($400, verified), NOOA 85.1 ($332), OPINE-World 78.4 ($1,040), "Vision - Continual Learning v1" 63.1 ($4,788; OPINE notes it pre-trains on the evaluation set, so not zero-shot), Read-Grep-Bash Agent 50.2, TELL 43.9 ($1,406; single conversation compounding a memory file), DreamTeam 38.1 ($18,000), Continual Harness 20.5 ($774), Polyphony 19.8, a-evolve 12.3, OpenClaw 5.2 ($2,912). Human Intelligence Harness baseline 95.3.

Ratings: Evidence Strong (official). Transfer: the "retain reasoning + compaction instead of eviction" result is the single most transferable finding in this report -- it is exactly the shape of the Duck-style eviction loop we run, and it tripled a frontier model's score at lower token cost. The Duke result is a warning: per-game bimodality means public-25 numbers cannot validate a harness change.

---

## 7. Test-time training on ARC-AGI-3

Sources: sethkarten.substack.com/p/continual-harness-an-efficient-self ; arxiv.org/html/2605.09650 (DreamTeam) ; arxiv.org/html/2607.20709 (NOOA) ; arxiv.org/abs/2603.17683 (Sensi) ; blog.huikang.dev/2026/05/31/autoresearch-hackathon.html ; github.com/AR6420/arc-agi-3-agent ; github.com/DriesSmit/ARC3-solution ; Kaggle public LB (kaggle CLI, 2026-08-29).

Is there any published evidence that weight-level TTT helps on the interactive benchmark? No. I found none.
- ARChitects/Franzen, MindsAI, Tufa, Tong Hui Kang are all on the Kaggle LB (Franzen 3.15, "Abstraction Lab & MindsAI" 2.94, Tufa 4.67, Kang 4.27) with no public ARC-3 TTT write-up. Tufa's public method is a pure inference harness. Kang's public post: autoresearch over architectures trained on "modified games" (same engine, altered shapes) with ideal traces; validation loss 4.5 -> 3.5 but "no evidence that the trained model is performing better than ... random actions," even on games it trained on; he concludes curve-fitting (history, action) is probably not on the critical path.
- The one entry with "continual-learning weights" (63.1%, $4,788) pre-trains on the evaluation set per OPINE-World, so it is not evidence for TTT.
- StochasticGoose is genuine online learning (CNN trained during play on a hash-deduplicated replay buffer, reset per level): 12.58 in the preview, 0.25 on the full benchmark.
- AR6420 (behavior cloning on 180,144 human-replay transitions + frame-change head): no score reported.
- Sensi (curriculum test-time learning, in-context, database-as-control-plane): v1 2 levels, v2 0 levels; they diagnose a "self-consistent hallucination cascade originating in the perception layer."
- Everything that does work as "learning at test time" is workspace/memory learning with frozen weights: Continual Harness refines prompt notes, a confidence-scored memory, Python skills and subagent specs after game-overs/level-ups/stagnation (Gemini 3.1 Pro, 20.5%; reusable skills account for 62% of executed actions). DreamTeam frames it explicitly as "artifacts in place of parameters, counterexamples in place of losses": 36 -> 38.4 with 31% fewer actions/game (444 vs 643) at ~$18k. NOOA's ablation: replacing its typed SQLite memory with markdown files costs 11.8 RHAE (50.2 vs 38.4, GPT-5.5); PRO-LONG's ablation: read-only 23.1 -> +grep 27.2 -> +python 38.3 -> +write/edit 41.2 (GPT-5.5) -- "programmatic tools drive gains, not auxiliary memory mechanisms."

How weight-TTT would even fit a 9 h offline budget: it would have to be a per-game LoRA on self-generated transitions (predict next frame / predict frame-change) during the first minutes of a game, with the base model frozen. Nothing published does this on ARC-3; the closest analogue (StochasticGoose) collapsed on the private set; a 27B LoRA step costs GPU time that competes with inference in the same 295 s/game. Evidence None; Transfer Low. The defensible version of "TTT" in this domain is in-context: a per-game playbook/skills file refined on evidence.

---

## 8. Open-source agents with reported numbers, and what matters for small models

Sources: primeintellect.ai/blog/prime-agent ; arxiv.org/html/2608.23552 ; arxiv.org/html/2607.20064v2 (PRO-LONG) ; github.com/ryanbbrown/Retrodict ; github.com/dolphin-in-a-coma/arc-agi-3-just-explore ; arxiv.org/html/2512.24156v1 ; github.com/Tufalabs/duck-harness ; tufalabs.ai/research/duck-harness/ ; arcprize.org/blog/arc-prize-2026-milestone-1 ; arxiv.org/html/2605.05138v2 ; schema-harness.github.io ; news.ycombinator.com/item?id=48935905 ; huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces.

Frontier-model harnesses (for mechanism, not for numbers we can reach):
- Prime Agent (MIT license): Opus 5 95.5 best@1 (runs 95.0/95.2/95.5, best@3 99.97, 183/183). RLM = context is a variable, subagents are function calls inside a persistent IPython kernel (the only built-in tool); compaction replaces a prefix with a summary but keeps the original events retrievable from the REPL; the ARC-3 prompt is "adapted from PRO-LONG." Self-reported; they note Claude Code / Codex runs score below the vendors' self-reports. Their own caveat, important for us: "many harness capabilities remain underused because current models were not trained to operate them."
- PRO-LONG: append-only logs.txt (action number, level, plan, action, board), ~30-line prompt, tools Read/Write/Bash/Grep + Python; the prompt asks for short action sequences while testing hypotheses and longer ones once proven. Ablation above. Fable 5 97.4 best@2, Opus 4.6 42.4, GPT-5.5 41.2.
- Retrodict: 99.86 RHAE, 7,703 actions, 660M tokens, $654 (GPT-5.6 Sol max). No images after priming. log.txt with a [DIFF] line per step plus parsed boards and connected components via an arclog library; Python only (numpy/scipy/networkx, no engine imports). Prompt disciplines: every hypothesis must retrodict the recorded history before a live test; every action must have a predicted outcome; a full-width/height edge strip that changes on most steps is a timer, not gameplay; prefer undo to RESET, never two RESETs in a row; searches are built incrementally with iteration/wall-time caps. playbook.md (working model marked log-verified vs assumed, plus working memory) survives context drops at 150k tokens. Escalation: after 300 actions (or two RESETs) on one level a binding directive is appended: inventory unexplained log content, promote checked rules into a step(state, action) simulator, verify it retrodicts every frame, then search; a second 300-action tier steers search toward never-seen states. Plan executor runs one step per turn, validates the predicted board, and re-invokes the model on mismatch or plan exhaustion.
- Executable World Models / baseline1: Codex CLI fills world_model_engine.py / state_io / main_planner templates; observe-model-verify-refactor-plan-execute; 1,500-action cap per level; GPT-5.5 15/25 games 58.12, GPT-5.4 8/25 41.29; "substantial variation between runs"; no ablations.
- Schema (Impossible Research): ~99 with Opus 4.8 / Fable 5, 95.35 with GPT-5.6 Sol, ~$25k per full run per HN; state grounding + mechanism program + backtest + plan-in-model. Self-reported. Their released traces give per-level action counts (e.g. Claude on ft09: 4,7,14,19,21,13 vs human 43,12,23,28,65,37).

Model-free and small-model agents (the relevant comparison set):
- just-explore (3rd, preview; zero LLM): segment frames into single-colour connected components, mask status bars, hash masked frames as graph nodes; five click-priority tiers by segment size/colour salience; always take the untested action from the highest tier, tracking distance to the nearest frontier; 12 private levels in the challenge, median 16 private levels in 8-h runs, 19 levels under a 4,000-interaction cap; random agent 6 levels, GPT-4.1+DSL 5 levels. Breaks on ft09/ls20 later levels (state space) and when the status bar differs.
- StochasticGoose: 4-layer CNN (32-256 ch) on 16-channel one-hot 64x64, predicts whether each action changes the frame; hierarchical sampling (action type, then click coordinate via convolution); hash-deduplicated buffer; weights and buffer reset on each new level; ~350 actions wasted per level discovering clickables.
- The Duck (Kaggle Milestone 1 winner, 1.21 private; 1.6002 +/- 0.4475 public over 25 games x 20 tries): Qwen 3.6 27B FP8 on vLLM; a Python REPL tool is the only interface; the model sees current_frame.ascii, current_frame.segmentation (object hashes, boundaries, containment, adjacency), history, previous_frame, transitions, valid_actions, last_action_result (board_changed, level_completed, game_over, reward); the raw numeric grid is deliberately hidden; multiple action(...) calls per tool invocation allowed; each tool call is a fresh Python process; context kept short by evicting oldest messages (Kaggle write-up: target 32k, max 64k); default 45 min per game. Tufa's stated lessons: "hand-crafted tools actually hurt the model; letting it improvise worked better," and "the main driver of improvement ... came from better base models and introducing multi-modality."
- Reki (2nd, Milestone 1): Gemma-4-31B vision, local. Renders recent frames as labelled images; asks for one JSON with what changed, a short plan and the next 1-4 actions; reflection memory refreshed every ~10 steps; numpy click heuristic (prefer small, rare-coloured, button-like shapes); "dead-signature" notices that stop clicking object types that never change anything for the rest of the level; legal-action and JSON-repair guards; ablation via env vars. forge (3rd): same core with a generator/arbiter that was mostly disabled in the winning config.
- Kaggle public LB on 2026-08-29 (kaggle CLI): cstl 5.99, Lord Han Solo 4.99, Tufa Labs 4.67, Tong Hui Kang 4.27, rfbr 3.37, Tony G 3.17, Daniel Franzen 3.15, OzanM 2.98, Abstraction Lab & MindsAI 2.94. No public write-ups found for cstl or Lord Han Solo.

What the evidence says matters for small models specifically:
1. Loop shape: retained state + compaction beats eviction (OpenAI 13.3 -> 38.3 with 6x fewer output tokens). The Duck evicts; Reki keeps a reflection memory refreshed every 10 steps. Evidence Strong on frontier, Medium on 31B (Reki placed 2nd with it).
2. Programmatic tools over the history (grep/python over a log) are the largest single lever in the only tool ablation that exists (PRO-LONG, +18 points for GPT-5.5) -- but Tufa found hand-crafted tools hurt a 27B, and Prime says models are not trained to operate rich harnesses. Read together: give the small model a minimal, general interface (a log + Python), not many bespoke tools.
3. Observation format: diff lines and connected components (Retrodict, Duck) rather than raw grids; hide the timer strip. AERA's small-model failures were "text observations too sparse"; Sensi's were perception hallucination. Evidence Medium.
4. Action-selection priors: force the probes (each action once, ACTION6 on salient objects) rather than trusting the LLM's first pick (AERA 0/5 -> 5/5; Reki heuristics). Evidence Strong at 0.5B, Medium at 31B.
5. Action batching: Reki emits 1-4 actions per call; Duck allows many action() per call; PRO-LONG asks for short sequences until a strategy is proven, then long ones. This is the only way to fit 295 s/game. Evidence Medium (no ablation), Transfer High.
6. Structured memory beats markdown notes (+11.8, NOOA) and skills reuse cut exploration cost (Continual Harness) -- both on frontier models; unverified at 27B.
7. Program-synthesized world models: the 100-RHAE systems all have one, but Tycho shows the actor+evidence does most of the work and over-building the model hurts; no evidence a 27B can write a simulator that retrodicts. Medium-Low for us.

---

## 9. Human baseline dynamics and agent-vs-human action counts

- Humans: 17,135 baseline actions over 183 public levels (mean 93.6/level, median 60; my computation from the Schema baseline file). Level 1 is cheap everywhere (7-78 actions); later levels routinely 100-578.
- Top harnesses are faster than humans: VISTA 7,542 (56% fewer), AVO 6,624, Tycho/Opus 5 6,641 (61% fewer), Retrodict 7,703, Tycho/GPT-5.6 7,766. Mid-tier: Tycho matched Opus 4.8 orchestrator 10,354 actions for 166 levels; DreamTeam 444 actions/game and 54.1 per solved level vs 100.1 for its baseline; OPINE 1.7x human on wins. Weak agents are one to two orders slower: StochasticGoose 255,964 actions in the preview; Executable World Models capped at 1,500 per level (16x the median human level).
- Because the level score is squared and capped at 1.15, being 2x slower than the human median on a level yields 25%; being 5x slower ends the level. For an agent at ~2-6%, almost all score comes from which levels are completed at all, weighted by level index -- efficiency polish is second-order until levels 3+ are being cleared.
- Resets: humans could reset freely and it "typically increased total interaction time"; agents log RESET as an action; no source states the accounting precisely (see Section 0).

---

## Where sources disagree or should be discounted

- "Harness-only" 100% scores (VISTA, AVO, Tycho frontier, Retrodict, Schema, Prime) are all on the public 25, which AERA shows is reachable by non-intelligent strategies and ARC Prize shows is bimodal per game; only Tycho/Retrodict/baseline1 are verified on a semi-private set. None are Kaggle-eligible.
- GPT-5.6 Sol's "official" score appears as 7.8 (semi-private, BenchLM/Context Studios) and 13.33 (public, VISTA/OpenAI); both are right for different sets.
- AERA's "RHAE = 0.30 (30%)" Kaggle claim conflicts with the live LB maximum of 5.99; treat as 0.30 on the LB scale.
- Human baseline: tech report (2nd-best human, cap 100) vs current methodology (median, cap 115) -- the April 2026 change; all 2026-07+ harness numbers use the new rule.
- The null-coordinate WIN is a local-library artifact, unconfirmed on the hosted/Kaggle engine and explicitly excluded from RHAE by its discoverer.
- Tufa ("hand-crafted tools hurt") vs PRO-LONG (tools are the whole gain): the reconciling reading is that generic programmatic tools help, bespoke game tools confuse a small model.

---

## Top 10 mechanisms most likely to move an offline 27B agent from ~1.7 to 6+

Ordered by (evidence x transferability x fit to 295 s/game). "Cost" is wall-clock/LLM-call cost inside the Kaggle budget.

1. Retain state across turns with compaction instead of eviction. Keep a per-game playbook (verified rules / assumptions / current plan / ruled-out hypotheses) and a compact continuation summary; drop raw old turns, never the notes. Evidence Strong (OpenAI 13.3 -> 38.3, 6x fewer output tokens; Retrodict playbook; VISTA GUIDE/WORKING; Reki reflection every 10 steps). Cost near zero. This is the loop-shape change that every 30+ system shares and the Duck-style eviction loop lacks.
2. Action batching with a code-executed plan and interrupt-on-surprise. The LLM emits a short action sequence (1-4 while hypothesizing, longer once a rule is verified); harness executes, stops on board_changed == False, level change, game over, or predicted-frame mismatch, then re-invokes. Evidence Medium (Reki 2nd place at 31B; Retrodict/Tycho plan executors; PRO-LONG prompt). Cost negative (fewer calls per action) -- it is what makes items 3-8 affordable.
3. Deterministic level-start probe policy before any LLM call: each available action once; ACTION6 on the top-k salient components (small, rare-coloured, button-like); record the effect signature of each; feed the LLM a table of "action -> what changed." Evidence Strong at small scale (AERA 0/5 -> 5/5 by forcing ACTION6; StochasticGoose wasted ~350 actions/level learning clickables; Reki heuristics). Cost: ~10-30 actions per level, zero LLM calls.
4. Observation as diffs + connected components, timer strip masked, raw grid hidden. Per step give [DIFF] cells, moved/appeared/vanished components, board_changed, and keep the full grid only on request. Evidence Medium (Retrodict, Duck, just-explore, Tycho frame typing); direct fix for the "observations too sparse / perception hallucination" small-model failures (AERA, Sensi). Cost near zero.
5. Model-free stagnation supervisor with escalation tiers: track frame-hash novelty, actions since last level-up, repeated action loops; on stall inject a binding directive (inventory unexplained effects; switch from exploit to explore; try never-seen states; consider RESET) and, if still stuck, hand control to the graph explorer (item 6). Evidence Medium (AVO supervisor, Retrodict 300-action tiers, Continual Harness refine-on-stagnation). Cost near zero.
6. Graph-based frontier exploration as the fallback policy: hash masked frames, keep untested (state, action) edges with visual-priority tiers, walk to the nearest frontier. It scored 3rd in the preview with no LLM and cleared 16 private levels in 8 h. Use it when the LLM is stuck or when per-game wall clock is nearly spent. Evidence Strong for a model-free floor. Cost: CPU only.
7. Effect-signature tables (OPINE's ontology error, without the synthesizer): Dirichlet counts of effect signatures per (component type, action, context); high entropy = probe target, zero-effect rows = dead actions/objects to ban for the level (Reki's dead-signature). Evidence Medium (OPINE 20/25; Reki 2nd). Cost: counting.
8. Level-boundary context reset with a carried summary plus per-level ruled-out list. Tycho clears conversation at each level and carries summaries/beliefs/model; VISTA keeps GUIDE.md across levels. Directly targets the official failure mode "solved the level, didn't learn the game." Evidence Medium. Cost near zero.
9. Predict-before-act with cheap verification: require a predicted effect ("board changes", "player moves right by 1", "no change") for each batched sequence; on mismatch, mark the rule unverified and re-plan. Evidence Medium (VISTA, Retrodict, NOOA retrodiction as the "sole refinement signal"; Tycho warns not to over-invest in full simulators). Cost: one extra field per call. Do not require a full Python simulator from the 27B; verify predicates, not frames.
10. Test-time learning of the workspace, not the weights: a per-game skills/rules file the model appends to and reads first on every re-invocation, plus generic tools (log grep, python) rather than bespoke game tools. Evidence Medium-Strong on frontier (PRO-LONG +18, NOOA +11.8, Continual Harness skills = 62% of actions), Medium on small models (Tufa: bespoke tools hurt). Cost low. Weight-level TTT stays off this list: no published evidence it helps on ARC-AGI-3, and the one online-learning system that placed (StochasticGoose) collapsed on the private set.

Explicitly not recommended for this budget: full program-synthesized simulators with CEGIS (Tycho shows the model is worth ~9 RHAE on top of good evidence handling and that repair-heavy loops hurt; needs ~600-1,000 frontier calls/game), image-based perception as a first move (VISTA's own ablation says text works; Tufa credits multimodality but has no ablation), and anything built on the null-coordinate crash.

---

## URL index

- Tycho: https://arxiv.org/html/2607.28287v1 ; https://github.com/NIMI-research/Tycho
- AVO: https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-demonstrating-a-frontier-level-general-purpose-architecture-for-long-horizon-autonomous-agents/ ; https://arxiv.org/pdf/2603.24517
- VISTA: https://vista-research.github.io/
- OPINE-World: https://arxiv.org/html/2607.01531v2
- AERA: https://arxiv.org/pdf/2605.25931
- Tech report: https://arxiv.org/html/2603.24621v1 ; methodology https://docs.arcprize.org/methodology ; human dataset https://arcprize.org/blog/arc-agi-3-human-dataset ; GPT-5.5/Opus 4.7 analysis https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis ; preview learnings https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings ; LLM templates https://docs.arcprize.org/llm_agents ; recordings https://docs.arcprize.org/recordings ; community LB https://arcprize.org/leaderboard/community ; Milestone 1 https://arcprize.org/blog/arc-prize-2026-milestone-1
- OpenAI two settings: https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/ (summary: https://gigazine.net/gsc_news/en/20260731-how-enabling-two-settings-tripled-arc-agi-3-benchmark/)
- Symbolica: https://www.symbolica.ai/blog/arc-agi-3
- Prime Agent: https://www.primeintellect.ai/blog/prime-agent ; https://arxiv.org/html/2608.23552 ; PRO-LONG https://arxiv.org/html/2607.20064v2
- Retrodict: https://github.com/ryanbbrown/Retrodict
- Executable World Models: https://arxiv.org/html/2605.05138v2
- Schema: https://schema-harness.github.io/ ; https://huggingface.co/datasets/schema-harness/arc-agi-3-schema-traces ; https://news.ycombinator.com/item?id=48935905
- NOOA: https://arxiv.org/html/2607.20709 ; DreamTeam https://arxiv.org/html/2605.09650 ; Continual Harness https://sethkarten.substack.com/p/continual-harness-an-efficient-self ; Sensi https://arxiv.org/abs/2603.17683
- just-explore: https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore ; https://arxiv.org/html/2512.24156v1
- StochasticGoose: https://github.com/DriesSmit/ARC3-solution ; https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db
- Duck: https://github.com/Tufalabs/duck-harness ; https://tufalabs.ai/research/duck-harness/ ; Kaggle write-up (403 to me) https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133
- huikang autoresearch: https://blog.huikang.dev/2026/05/31/autoresearch-hackathon.html ; AR6420 BC agent https://github.com/AR6420/arc-agi-3-agent
- Scores aggregators: https://benchlm.ai/benchmarks/arcagi3 ; https://www.contextstudios.ai/blog/arc-agi-3-measured-the-harness-not-just-the-model ; https://dev.to/p0rt/the-model-scored-30-the-harness-scored-100-which-one-did-you-benchmark-3mp4
- Kaggle public LB: fetched with `kaggle competitions leaderboard arc-prize-2026-arc-agi-3 --show` on 2026-08-29.
