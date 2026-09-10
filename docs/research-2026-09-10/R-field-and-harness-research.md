# Field + harness research, 2026-09-10 (two parallel sweeps)

## 0. THE PREMISE NEEDS CORRECTING FIRST

"OpenAI's new model scored very high on ARC-AGI-3 and it was the harness" is **half right, and the
half that is right is one we have already partly tested.**

* The high scores are on the **25-game PUBLIC preview set**, not the Kaggle hidden set we compete on.
  Between 07-30 and 09-08 **at least six independent systems reported 100.00 RHAE** there (Tycho,
  NVIDIA AVO, VISTA, Kepler, arc-skill, agno's GPT-5.6 Sol). An independent audit (arXiv:2605.25931)
  finds **10 of 25 solvable in a single blind step**, 5 after one probing action, and a library-level
  null-coordinate bug that **bypasses 18 of 25 games in one step**. That set is saturated and cannot
  discriminate. **Every 100.00 in this field is public-set-only; no paper reports a private number.**
* The offline Kaggle track we are in tops out at **11.04** (Tufa, 09-09). **There is no published
  ARC Prize analysis of why that gap exists.**
* The specific OpenAI harness claim IS real: with **retained reasoning + compaction**, GPT-5.6 Sol
  went **13.3 % -> 38.3 % with ~6x fewer output tokens** (07-31; OpenAI's own page 403s to fetchers,
  corroborated by four secondary outlets and ARC Prize's reply). **But our A1 work already showed our
  stock CARRIES reasoning (1,268 request pairs, slope 1.0) and the loss is EVICTION (27 % of
  requests) — and we tested the compaction knob at 0.5 and 0.75 and got dead / tie.** So the lever
  behind the headline is one we have already reached for. The operative variable it points at is the
  eviction fraction, not carry.

## 1. INDEPENDENT CONFIRMATIONS OF OUR OWN KILLS (three, all measured)

* **Executable world models.** Rodionov (arXiv:2607.15439, 4 variants x 2 models x 2 efforts):
  **textual BEAT the executable variant** on gpt-5.5 (+7.69 RHAE at high, +2.81 at xhigh); verification
  costs 1.82-3.26x the tokens; conclusion *"the three imposed mechanisms are not required"* — from an
  author whose earlier paper championed the lane. Corroborates our Polyphony/Stage-1 kill.
* **Prose knowledge.** Skill-as-Pseudocode (arXiv:2605.27955) bundle ablation: contract-only 13 wins,
  template-only (executable snippets) 25, full 30, **length-matched prose control 18**. Prose is
  near-worthless; executable snippets ~2x an abstract spec. Corroborates A1.
* **Added tools.** ARC Prize's own Milestone-1 writeup: *"hand-crafted tools actually hurt the model;
  letting it improvise worked better."* Plus Cohere/Poolside (arXiv:2604.17609): **restricting to
  bash-only roughly DOUBLED interaction rates** — richer toolsets push agents to default patterns.
  Both predict A2's 0 verifier calls in 358.
* **Fine-tuning at our scale.** `yuran986/arc-agi-3-agent-post-training` ran stateful multi-turn GRPO
  on Qwen3-8B: **reward rose, capability did not** — 16/16 sampled trajectories found the same single
  oracle-progress action then produced 144/160 no-progress actions; **level completion zero.**
  Corroborates the model-axis blocker from a different direction.

## 2. THE ONE STRUCTURAL FACT WORTH INTERNALISING

ARC Prize technical report (arXiv:2603.24621), measured: on environment **TR87, Opus 4.6 scores 0.0 %
with no harness and 97.1 % with the Duke harness**; on **BP35 it scores 0.0 % both ways**. Harnesses
built against three environments then run on the full set produced *"extreme bimodal performance...
controlling for the same frontier model."* The harness is worth up to ~97 points on some games and
exactly zero on others. That is the shape of our own per-game data (7 of 25 games never exceed one
level in any run; ft09 averages 3.55).

Their trace analysis names three failure modes; the third is ours exactly: **"victory without
understanding"** — clearing a level without internalising the mechanic, so misread primitives harden
into confident-but-wrong strategies **at the next level**. That is a depth problem and it maps onto
our L1 hazard 0.87 / L2 hazard 0.40 cliff.

## 3. THE LARGEST UNCLAIMED IDEA IN THE LITERATURE (untested by anyone)

**AutumnBench** (arXiv:2510.19788), 43 interactive grid worlds, 517 humans vs frontier models:
reasoning models *"use less than 7 % of their actions for resets and no-ops combined"* versus humans
at **~12.5 % each**, and *"do not treat resets as special actions, unlike humans"* — they fail to use
**no-ops and resets as experimental instruments**. Models also *"often fail to update their
understanding when faced with contradictory evidence."*

Nobody has tested the intervention. It costs **actions, not calls** — and our scoring makes actions on
levels we never clear **free**. This is the best-fitting untested idea found.

## 4. THE ONLY ABLATION THAT RAISES SCORE WITHOUT RAISING SPEND

**Tycho** (arXiv:2607.28287), Opus 4.8 at FIXED budget: no world model **79.07** -> actor-controlled
**85.36** -> **orchestrator choosing per-situation whether to build / repair / use / bypass a model
88.49** -> automatic repair trigger **83.07**. It beats always-model AND always-repair. And separately:
*"automatic repair produces models that reproduce observed transitions much more accurately, yet
reaches only 83.07"* — **world-model fidelity does not buy action quality**, which is the same lesson
as our cross-level transfer result (green models, 0-45 % transfer).

## 5. BUDGET ALLOCATION — strong evidence, one hard design constraint

ReD (arXiv:2601.21522): optimal retry count tau=1, round-robin; **81 % vs 34 % coverage** first round.
CLEAR (arXiv:2606.03092): **+11.6 to +24.0 pts** at tight budgets, gains *diminish* as budget grows.
ZEBRA: uniform split costs **-5.14 to -5.92 pts**, and it tolerates a **50 % noisy** difficulty signal
(-1.4 pp, n.s.). BAGEN: early stopping saves **28-64 % of tokens on failed trajectories for 1.6-4.2 pp**.

**HARD CONSTRAINT, three independent results: the triage decision must NOT live in the model.**
TRIAGE (arXiv:2605.13414) finds triage efficiency negative for most of 20 model configs with no
scaling in parameter count; BAGEN finds models predict feasibility only after **60 % of budget is
burned**; ZEBRA finds an LLM allocating its own budget costs **-4.2 to -4.3 pts**. Every positive
result used an EXTERNAL controller.

**Two caveats that are ours specifically:** all of this assumes binary per-task success, and nothing
addresses a **level-weighted quadratic score**, which moves the abandonment threshold. And the
prerequisite — *is per-game outcome predictable from the first K calls?* — is **answerable offline on
our existing corpus at zero cost**. Published AUCs range 0.86 (TextCraft) to 0.59 (WebShop) to
<0.60 (deep research); our domain is unmeasured.

## 6. CONVERGENCE WITH WHAT WE BUILT TODAY

**PRO-LONG's harness appends to a log after every action AUTOMATICALLY — the agent never decides to
write, only to read** — with a full-log vs no-log ablation of **41.2 % vs 24.0 % pass@1**. That is
precisely the design principle behind `graft_effects` (built this morning, before this was found):
supply the derived information rather than asking the agent to derive it. Independent convergence.

## 7. DOES NOT FIT OUR BOX (stated so it is not re-proposed)

Twin (224k tokens per scored action, ~1,250x our budget); Tycho's frontier config (3,500 calls vs our
55); Kepler (34M tokens/game, and its 97 % cache discount is unavailable under our MTP=>no-prefix-cache
law); Rodionov's verification variant (2.2-8.9M tokens/game); DreamTeam (11.5 calls and 693k input
tokens **per environment step**); WorldCoder (~400k tokens/env, same order as our own 79 %-of-budget
measurement); Go-Explore/GLoW (depends on replaying to archived states — our RESET is swallowed in
competition mode); RAPOA (1.6-10M tokens per task family, and prompts do not transfer); agno's seeded
runs (per-game manuals for the 25 public games — nothing to seed on 110 hidden games).

**Skill libraries: two 2026 results say don't.** arXiv:2607.07504 (7,560 runs): **-0.8 pp, 95 % CI
[-3.8, +2.0], p=0.644**, with a real skill statistically indistinguishable from a token-matched
IRRELEVANT one at 4.5x the input tokens.

## 8. WORTH A CHEAP CHECK

**CodeAct** (ICML 2024, 17 LLMs): code as the action format vs JSON vs text is worth **up to +20.7 pp
and 30 % fewer actions**. Our action channel already is python code, so this is likely already banked —
but confirming costs 20 minutes.

## 9. GAPS NOBODY HAS FILLED (i.e. genuinely open)

* No study reallocating a global budget across a batch of interactive **game** episodes.
* No measured evidence on early failure predictability for grid/game agents.
* Nothing on early abort interacting with **level-weighted quadratic scoring**.
* **No ablation of "LLM-written controller running N steps" vs "one action per call" at a fixed small
  call budget.** Nobody publishes an actions-per-call curve at all.
* No replication of our macro-batching kill in either direction. The closest same-sign analogue is
  TTI's h=30 collapse (**-14 pp vs a shorter horizon**); ELHPlan reports the opposite sign but pairs
  chains with a **proactive replan trigger** — the reconcilable hypothesis, untested by anyone, is that
  **batching pays only when coupled to a cheap mid-flight abort detector**. That is a different
  experiment from the one we ran and killed.
* No controlled ablation isolating which harness component drives the 30 % -> 100 % public-set jump.
