# Field refresh 2026-09-08 (memory-blind; external evidence only)

Method: Kaggle CLI (leaderboard, kernels), Kaggle forum via browser, arcprize.org, GitHub, arXiv. Labels: **verified** (seen), **stated by team**, **inferred**.

## Leaderboard (Kaggle CLI, 15:47Z)

| rank | team | score | entries | note |
|---|---|---|---|---|
| 1 | Tufa Labs | 11.04 | 132 | authors of the open duck harness; nothing public after July 1 |
| 2 | Third Intelligence | 8.21 | 40 | Yuki Okumura (Comp. GM) + seele1917; nothing public |
| 3 | Daniel Franzen | 7.63 | 65 | ARC Prize 2024 winner (TTT + augmentation + custom decoding); nothing public on ARC-3 |
| 4 | mostik.ai | 7.51 | 47 | latent-bridge start-up (12 PhDs); declines to describe the ARC entry |
| 5 | NVARC3 | 5.96 | **3** | NVIDIA KGMs (Puget, Sorokin, Deotte) + the two authors of NVIDIA's ARC-AGI-3 papers (DreamTeam / NOOA) |
| 6 | Son Pham & Mark Barney | 5.66 | 42 | instrumented duck fork on GitHub (sonpham-org/arc-3); recipe not disclosed |
| 7–15 | Fighting Gold 5.55, jinbo wang1 5.49, Fususu 5.47, Kopiczko 5.33, MindMatrix 5.07, Nader 5.05, Lord Han Solo 4.99, Kyutai 4.90, Slavin 4.82 | | | |
| 24 | wuliao_0 | 4.33 | | byte-identical copy of keithtyser V14 (verified by cell diff) |
| 26 | us | 4.31 | 73 | byte-copy of keithtyser V14 + yield 900 |
| 33 | keithtyser | 4.17 | | V14 author |

**No public notebook scores ≥ 5.** Everything ≥ 4 is the Flash-Next serving stack + stock June duck. Public copies of V14 draw 2.33–4.33 (the author's own list), i.e. the 4.0–4.6 band is the max statistic of daily resubmits (Ya Xu, 19th: "95% of teams copy the best public notebook and submit 5 days in a row to negate the variance"). NVARC3's 5.96 in three entries is the strongest per-draw evidence on the board.

## Milestone 2 (pinned thread 713634, verified)

Deadline 11:59 pm UTC **Sept 30**; milestone ranking uses the **public** LB; open-sourcing is required for milestone-prize eligibility (not for other entrants). Tufa open-sourced at Milestone 1. Expect prize-claiming teams' code on or after Oct 1.

## What the top tier says (little) and what the foundation says (a lot)

- **Tufa** writeup (717133, July 1): duck = REPL with game state as variables, REPL **reset between calls**, evict oldest turns to hold ~32k, one "World model:" note carried between turns, segmentation tool, UNDO hidden. "Main driver of improvement came from better base models and multi-modality"; hand-crafted tools hurt; **the two levers they call unfinished: context compaction/memory and perception.** Nothing newer is public (GitHub last commit July 1).
- **ARC Prize, Sept 3 (arcprize.org/blog/astra, verified):** GPT-6 Astra scores 62.7% on the standard harness and **99.9%** with the provider adapter whose only differences are *preserving opaque reasoning state between requests* and *compaction*; 3.66× faster, 49% fewer tokens. Astra's winning behaviours: compact symbolic world-model notes carried forward (a self-invented DSL), and in PRO-LONG a per-game toolkit it writes itself (board parser, state model, search, planner, persistent notes; e.g. maze_solver.py / patrol_solver.py / sync_state.py on tu93). Under the standard harness it beat the human action baseline on 96% of levels.
- **NVIDIA (NVARC3 lineage):** DreamTeam (arXiv 2605.09650): six-agent harness that builds an *executable world model*, 36→38.4% public with 31% fewer actions. NOOA (arXiv 2607.20709, github NVIDIA-NeMo/labs-OO-Agents, Apache-2, vLLM/open-weight supported): CodeAct object-oriented agents, 85.1% on the community board at $332.
- **Polyphony Agent** (github Mininglamp-AI/polyphony-arc-3, July 7): "grows a verified per-game heuristic system for state, dynamics, planning and action selection as executable Python files"; **19.8% on the community board with Qwen3.6-27B** at $115 (duck+Flash-Next reads ~7–8 local on the public 25).
- **Prime Agent** (arXiv 2608.23552, open): persistent IPython REPL + recursive sub-agents + continual harness, 30→95.5% (frontier API). **Twin** (2608.14490, open): test-time executable digital twin with counterexample repair, 179/183 levels (frontier API). **PRO-LONG** (2607.x): programmatic memory = structured log the coding agent searches, 97.4% best@2.
- **STaR LoRA** (forum 739047, Sept 2, verified): LoRA-SFT of Qwen3.6-27B on the duck's own level-completing trajectories; LB **1.25 → 1.94** (one team, n=1, high variance, more data hurt, adapter is base-model-specific). Three CC0 datasets released.
- **Fususu (9th)**, forum 739938: tried and rejected Gemma-4-31B (vision better, code worse), a symbolic "system decoder" for GPT-OSS (game-specific bias), 5-role multi-agent (better reasoning, 5× calls, too slow for Kaggle), thinking on/off (little change), concurrency > 16 saturates, self-play training data (no gain), a strategy library (bias). Now "fewer tokens per step is more effective".
- **Rakha (152nd)**, same thread: deterministic serving profile beat prompt tweaks; single-step verified recovery; memory of *verified transition facts only*; adaptive compute allocation; suspects continuous-batching nondeterminism drives variance. Sits at 2.86–3.48.
- **mostik.ai** read-more (verified): trained bridge passing hidden states from frozen GLM-5.2 753B to frozen Qwen-3.5 4B; small model closes 50% of the gap. Needs the sender at inference; cannot be their Kaggle entry as published (inferred).

## Ranked: what the 7–11 tier appears to do that stock duck + Flash-Next does not

1. A private harness, not the public notebook (**verified by elimination**).
2. Persistent memory / compaction instead of eviction, with reasoning state carried forward (**stated** by Tufa as unfinished, **measured** by ARC Prize on Astra, **stated** by PRO-LONG and two forum practitioners).
3. An executable, verified world model / per-game toolkit that persists across calls (NVIDIA lineage → NVARC3 5.96 in 3 draws, **inferred** from author overlap; Polyphony **stated** with a 27B result; Astra PRO-LONG **verified**).
4. Fine-tuning on the harness's own winning trajectories (one public +0.7 LB, caveated; Franzen's track record, **speculative**).
5. Serving determinism and fewer tokens per step (**stated** by competitors).
6. Heavy resubmission inflating the visible max (**verified**: Tufa 132 entries, MindMatrix 93, us 73).

Gaps: Third Intelligence, Franzen, Kyutai, Fighting Gold, jinbo wang1, Kopiczko have zero public method disclosure; X was not accessible.
