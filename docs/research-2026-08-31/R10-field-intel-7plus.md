# R10 — Field intel: how is cstl at 7.51, and what moves a duck-harness to 7+?

Date: 2026-08-31. Fresh-eyes competitive-intelligence pass, primary sources only (no reliance on this repo's prior conclusions). All claims graded **VERIFIED** (source given) or **SPECULATIVE**.

---

## 1. Verified leaderboard state (Kaggle API, pulled 2026-08-31)

Via `.venv/bin/kaggle competitions leaderboard arc-prize-2026-arc-agi-3 --show`:

| # | Team | Score | Entries | Last sub |
|---|------|-------|---------|----------|
| 1 | cstl | 7.51 | 41 | 2026-08-30 19:46 UTC |
| 2 | Lord Han Solo | 4.99 | 48 | 2026-08-30 22:22 |
| 3 | Tufa Labs | 4.71 | 124 | 2026-08-31 06:39 |
| 4 | Son Pham & Mark Barney | 4.42 | 34 | |
| 5 | Tong Hui Kang | 4.27 | 58 | |
| 6 | Daniel Franzen | 4.05 | 57 | |
| 7 | Ebi | 3.85 | 4 | |
| 8 | Kyutai | 3.37 | 35 | |

Entry counts are from the rendered Kaggle leaderboard page (browser). **VERIFIED.** Note Ebi at 3.85 with only 4 entries — a strong method can land high with almost no LB iteration; conversely Tufa's 124 entries bought 4.71.

The public LB is computed on ~50% of the private test data ("This leaderboard is calculated with approximately 50% of the test data" — leaderboard page. **VERIFIED**).

## 2. WHO IS CSTL

- The cstl team resolves to two Kaggle accounts, in leaderboard row order: **`/tehnar` ("Tehnar")** and **`/gatamaz` ("TG")** (rendered leaderboard DOM, profile links immediately preceding rank-2 `/lordhansolo`). **VERIFIED** (rendered page; member-to-team mapping inferred from strict row ordering — treat as high-confidence, not absolute).
- **Tehnar**: "Software Engineer", Amsterdam, NL. Joined Kaggle 11 years ago, 11 followers, 6 competitions total. Best prior results: NeurIPS 2024 Lux AI S3 **116/701**, TalkingData 2958/3943, plus two invitation-only **St. Petersburg (СПб) AI hackathons** (2/16, 4/21) 8 years ago. One kernel, one discussion post ever. (kaggle.com/tehnar, /tehnar/competitions. **VERIFIED**)
- **gatamaz ("TG")**: San Francisco, joined 9 years ago, **1 competition** (this one), all profile sections hidden. (kaggle.com/gatamaz. **VERIFIED**)
- Neither is a Kaggle GM or a known ARC researcher. Web searches for "cstl" + ARC across X/HN/Reddit/GitHub/Medium returned **nothing** attributable — no writeups, no talks, no repos. **VERIFIED (absence of evidence after multiple searches)**.
- The St. Petersburg hackathon history suggests a Russian competitive-programming background ("технарь" ≈ "techie"). **SPECULATIVE.**
- Rank 2 "Lord Han Solo" is the same archetype: anonymous software engineer, Białystok, Poland, 2 competitions, 3 followers (kaggle.com/lordhansolo. **VERIFIED**).

**Conclusion:** the top of this board is not held by famous Kagglers or labs — it's held by anonymous, extremely strong systems engineers who publish nothing. There is no public cstl method trail to copy. Their edge must be reconstructed from the public field (§4).

## 3. Trajectory analysis

- cstl: **41 entries**, last sub 1 day ago → they have submitted roughly daily for ~6 weeks. **VERIFIED** (leaderboard page).
- The jump 5.99 → 7.51 landed ~08-30 (prompt premise; consistent with submissionDate 08-30 on the API row). Historical per-day LB values are not exposed by the API; I could not verify the exact step structure. **Partially VERIFIED.**
- Kaggle discussion "**Sudden increase in top 3 teams?**" (Drona Bajaj, posted ~08-26, discussion/737617): "almost simultaneously, the top 3-5 teams have increased by a very good amount." Community guesses in-thread: (a) a public new idea — Jakob Brüggen searched and *failed to find* a recent public release explaining it, pointing only to the Polyphony Agent (27B, but scored just 19.8% on public — weak candidate); (b) Halla Yang points to the **NVIDIA AVO** post as an idea source; (c) OverfitOracle (14th): "they didn't share any code." **VERIFIED thread content; causes unresolved.**
- The second milestone prize (open-source deadline) is **September 30**, not August (staff post, discussion/713634 — **VERIFIED**). So the late-August surge is NOT milestone-snapshot timing; but a September 30 milestone does create an incentive for leaders to bank score now and open-source at the last hour (precedent set at Milestone 1, per Tong Hui Kang in discussion/705043. **VERIFIED**). **Expect the leaders' code to become public ~Sept 30** — a hard date worth planning around.
- A simultaneous rise of several independent teams in mid/late August is most parsimoniously explained by a shared enabler. The obvious shared enabler in the timeframe: the **Qwen 3.8 27B** model release wave (public kernels "LB-9 arc3 duck v12 with Qwen 3.8 27B" updated 08-18, 264 votes; "Duck Qwen3.8 27B FP8" 08-17 — kernels list via API, **VERIFIED**) — i.e., base-model upgrade + harness rework on top. Attribution of cstl's specific jump to this: **SPECULATIVE.**

## 4. Field methods since June (primary sources)

### 4a. The public-25 set is SOLVED by executable-world-model agents

ARC Prize community leaderboard (arcprize.org/leaderboard/community, **VERIFIED**):

| Agent | Authors | Public-25 RHAE | Model | Code |
|---|---|---|---|---|
| **Tycho** | Jens Lehmann et al. (NIMI) | **100.0%** | GPT-5.6 Sol / Claude Opus 5 | github.com/NIMI-research/Tycho (Apache-2.0) |
| **Retrodict** | Ryan Brown | **99.9%** | GPT-5.6 Sol (max reasoning) | github.com/ryanbbrown/Retrodict |
| **baseline1 (EWM)** | Sergey Rodionov (SingularityNET) | **99.0%** | GPT-5.5 high | github.com/astroseger/arc-3-agents-baseline1, arxiv 2605.05138 |
| **NOOA** | Gal Kaplun et al. (NVIDIA) | 85.1% | ? | github.com/NVIDIA-NeMo/labs-OO-Agents, arxiv 2607.20709 |
| **OPINE-World** | Courtis/Li/Sanner (U Toronto) | 78.4% | ? | arxiv 2607.01531 |
| Polyphony Agent | Ruiyang Yu et al. | 19.8% | 27B-class | github.com/Mininglamp-AI/polyphony-arc-3 |

Plus, on arXiv (2608.14490, **VERIFIED**): **Twin** (Stanford/Cornell/USC/Yeshiva) — 93.3/100, 23/25 games, **0.61× human actions**, using GPT-5.6 Sol. Method: grow an executable Python "digital twin" by counterexample-guided refinement; **validate the twin against every observed transition before any scored action**; treat goal inference as hypothesis testing (correct goal inferred pre-reward on 87.2% of levels); BFS inside the validated twin; 92.9% of scored actions execute pre-validated plans. Cost: 2.6B tokens / 91.4 wall-clock hours for 25 games.

The shared recipe of the entire ≥78% tier (**VERIFIED across the four writeups**):
1. **Executable world model** — a Python `step(state, action)` program, built and repaired from observed transitions.
2. **Validation gate** — the model must reproduce the transition log before the agent is allowed to spend real actions (Retrodict: "only a hypothesis that survives the log earns real actions").
3. **Engine-side search** — BFS/planning inside the verified model; the LLM proposes, cheap deterministic code verifies.
4. **Persistent memory across context resets** — Retrodict's `playbook.md` survives its 150k-token resets; AVO carries persistent memory of implementations/results/reasoning.
5. **Structured perception** — Retrodict's `arclog` (boards, diffs, connected components); Tufa's segmentation tool.

Crucial caveat: all of the ≥90% agents run on **frontier API models with enormous budgets** (Retrodict: $654/660M tokens; Tycho: $2,986; Twin: 91h). None runs in the Kaggle box. The open question of the competition is compressing this loop into 9h × 110 games × one 96GB GPU.

### 4b. NVIDIA AVO (developer.nvidia.com blog, **VERIFIED**)

AVO = general-purpose long-horizon coding-agent architecture (Terry Chen, Eva Zhu, Zhifan Ye, **Jean-Francois Puget**, Humphrey Shi). **100.00 RHAE on all 25 public games / 183 levels in 6,624 actions** on Claude Opus 5. Notably **no explicit programmatic world model** — instead: persistent memory, a **supervisor that watches trajectories and redirects the agent when progress stalls**, and a hypothesize→act→observe→preserve-state→revise loop. Their thesis: "long-horizon capability is a property of the full system," not the model. (JF Puget + Ivan Sorokin also won ARC Prize 2025's Kaggle LB at 27.64% — NVIDIA blog, **VERIFIED**.) No evidence links AVO's authors to cstl; NVIDIA people (Scott Le Grand posts as 83rd) are visibly *mid-pack* on this LB. **VERIFIED.**

### 4c. Tufa's Duck (the harness we run) — from their own milestone writeup (discussion/717133 + tufalabs.ai/research/duck-harness, **VERIFIED**)

Qwen 3.6 27B FP8, REPL with game-as-Python-variables, 64k context cap / evict-oldest at 32k, image + ASCII + segmentation, "world model" = a self-copied note tag. Milestone score: **1.21% LB** (June 30). Their own listed weaknesses: context management ("could be improved by compaction or adding memory"), perception, no optimal prefix caching. Same-submission variance: 0.77–1.30 LB on identical code; public-game σ up to 0.4. Tufa has since climbed 1.21 → 4.71 over 124 entries with the model/harness axis (their exact current internals are private).

### 4d. Public-vs-private gap (discussion/732854 + /736578, **VERIFIED**)

Self-reported local-25 → LB pairs: 5.0–5.4 → 1.4–1.8 (Pellegrin); 6.8 → 1.19 (daoviet); 3.8 → 0.9 (Scott Le Grand); 5.0+ → 1.6 (OverfitOracle); 2.8 → 2.4 (mikelou1). **The public 25 do not rank-order harnesses on the hidden 110.** Son Pham (4th): private is harder AND public-set familiarity inflates local scores. Also Scott Le Grand: Claude-built **game-specific solvers** got 7/25 games at ~38% RHAE but took 5 days + Claude access — "utterly impractical here"; his generalized version collapsed to 8/183 levels. Hand-crafting per-game does not survive the box.

## 5. Ranked explanations for cstl's 7.51

1. **In-box port of the executable-world-model / validate-then-act loop (the ≥90% public-tier recipe) onto a small local model, by elite systems engineers.** Evidence: the recipe is fully public since July (Tycho/Retrodict/baseline1 code released); cstl are exactly the profile (SWE, not ML-Kaggler) that wins by systems compression; 7.51 ≈ 7.5% RHAE over 110 games is what you'd expect from a loop that solves a modest number of extra levels at near-human action efficiency rather than grinding levels inefficiently (Twin's 0.61× human actions shows the efficiency headroom is real). Grade: method existence **VERIFIED**; attribution to cstl **SPECULATIVE** (they've published nothing).
2. **Base-model upgrade wave (Qwen 3.8-class) + harness redesign on top.** The whole top-5 rose near-simultaneously in late August (thread **VERIFIED**); Qwen 3.8 27B duck kernels appeared 08-17/08-18 (**VERIFIED**); OverfitOracle (14th) explicitly says his jump was "a very different approach in the harness + model" (**VERIFIED** quote). cstl-specific: **SPECULATIVE**.
3. **Compounding daily iteration + variance selection.** 41 entries at 1/day; with same-submission σ of ±0.3–0.4 (**VERIFIED**, Tufa), a leader's displayed best-of-many overstates their mean by several tenths — but not by 2.5 points over #2. This explains polish, not the gap. Grade: mechanism **VERIFIED**, sufficiency **REFUTED by magnitude**.
4. **Fine-tuning the local model on frontier-agent trajectories (distillation) or replays.** Rules-compatible (fine-tuned weights must be published for milestone eligibility — discussion/705043 **VERIFIED**); zero direct evidence anyone in the top 5 does it. **SPECULATIVE.**
5. Ruled out: milestone-snapshot timing (M2 = Sept 30, **VERIFIED**); famous-researcher identity (profiles **VERIFIED** anonymous); public-set overfitting (impossible — the LB is the hidden set); the NVIDIA AVO team being cstl (their known members post mid-pack, **VERIFIED**).

## 6. Directions to move a duck-harness (local 27B, one RTX Pro 6000, 1 sub/day) from ~2.5 to 7+

Ranked by evidence strength:

**D1 — Validate-before-act: no real action without a log-consistent hypothesis.** [Strongest evidence: shared by every ≥78% public agent — Twin/Retrodict/Tycho/baseline1, all **VERIFIED**] Keep a full transition log as data; force the agent's next move to come from a hypothesis (rule or plan) that is first replayed against the log in cheap Python. Twin's numbers show why this wins the *efficiency-weighted* score: 92.9% of its scored actions execute pre-validated plans and it beats human action counts (0.61×). For a 27B, the LLM only has to *propose* rules; verification and search are deterministic code — exactly the part that doesn't need a frontier model.

**D2 — Executable world model + engine-side BFS as the escalation path.** [**VERIFIED** design in Retrodict ("stuck-level escalation" builds a `step()` simulator) and Twin (counterexample-guided repair, BFS in the twin)] Don't build the WM up front on every game (token cost); build it when stuck, from the already-collected log, then plan offline and replay the plan. This converts wall-clock GPU time into free deterministic search.

**D3 — Persistent curated memory that survives eviction.** [**VERIFIED**: Retrodict's `playbook.md` across 150k-token resets; AVO's persistent memory; Tufa *names* context/memory as duck's top weakness in their own writeup] Duck's evict-oldest loses learned mechanics mid-game. A structured, agent-maintained playbook (mechanics learned, level layouts, failed hypotheses) re-injected after every eviction — and carried **across levels of the same game** — is the cheapest port of the frontier-tier recipe.

**D4 — Supervisor / stall governor.** [**VERIFIED** in AVO: a supervisor monitors the trajectory and redirects when progress stalls] On a 27B this matters more, not less: detect action-loops, no-frame-change streaks, and per-level budget burn with plain code, then force a strategy switch (explore-differently / build-WM / skip-level). Protects the 9h budget across 110 games.

**D5 — Structured perception instead of raw grids.** [**VERIFIED**: Retrodict's `arclog` (boards/diffs/connected components); Tufa reports segmentation "significantly helps" and that raw-grid dumps dilute attention] Feed diffs and object-level summaries, not full boards; do vision priming once per game, not per turn.

**D6 — Variance/logistics hygiene (cheap, small, real).** [**VERIFIED** variance ±0.3–0.4 per identical sub] With 1 sub/day and ~60 slots left, every submission should be an intended best config; expect and plan for draw noise; don't burn slots on A/B reads the noise floor can't resolve. Also: prefix caching is explicitly left on the table by Tufa's writeup ("we currently do not optimally use prefix caching") — throughput is score under an efficiency-capped metric.

**D7 (option, weaker evidence) — Distill frontier-harness trajectories into the 27B.** No public evidence a top team does this; rules allow it if weights are published for milestones. High effort, unproven on this LB. **SPECULATIVE.**

**Hard date to exploit: September 30.** Milestone 2 forces prize-seeking leaders to open-source by 23:59 UTC that day (staff post **VERIFIED**; Milestone-1 precedent: winner published a full writeup + code). If cstl or Lord Han Solo want the milestone money, their method becomes public in 30 days. Plan the campaign so that whatever they release can be absorbed and re-served within days — and do not over-invest in guessing what a last-hour release will reveal.

**Measurement warning** (repeatedly **VERIFIED** in the forum): local public-25 scores do not predict LB rank across harnesses (5.0→1.4, 6.8→1.19, 3.8→0.9). Evaluate D1–D5 by *mechanism metrics* that transfer (fraction of actions from validated plans, actions-per-level vs. human baseline, levels lost to loops/stalls), not by public-25 totals.

---

## Appendix: primary sources

- Kaggle API leaderboard pull, 2026-08-31 (local CLI)
- Kaggle leaderboard page (rendered): entries, members, 50%-split note
- kaggle.com/tehnar, /tehnar/competitions, /gatamaz, /lordhansolo (rendered)
- Discussion threads (rendered): 737617 (sudden increase), 732854 (public-25 self-reports), 736578 (public/private gap), 717133 (Tufa milestone writeup + comments), 705043 + 713634 (milestone timing; M2 = Sept 30), 697944 (9h runtime)
- arcprize.org/leaderboard/community (Tycho 100.0, Retrodict 99.9, baseline1 99.0, NOOA 85.1, OPINE 78.4, Polyphony 19.8)
- github.com/NIMI-research/Tycho; github.com/ryanbbrown/Retrodict (READMEs)
- arxiv 2608.14490 (Twin), 2605.05138 (EWM/baseline1), 2607.01531 (OPINE-World)
- developer.nvidia.com blog: AVO 100 RHAE on public-25; NVIDIA KGMoN ARC Prize 2025 win (Sorokin/Puget)
- arcprize.org/blog/arc-prize-2026-milestone-1; tufalabs.ai/research/duck-harness
- Searches that came up empty (negative results): "cstl" identity on X/HN/Reddit/GitHub/Medium; kaggle.com/cstl (404 — team name, not a username)
