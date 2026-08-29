# ARC Prize 2026 / ARC-AGI-3 Kaggle — how the top teams and top public notebooks score

Independent, public-sources-only research. Date: 2026-08-29. Working dir: `scratchpad/r1/` (pulled notebooks in `nb/`, downloaded TAAF source bundles in `ds/`, leaderboard history in `lb_history.json`).

Evidence labels: **[E]** = directly evidenced by a cited notebook, dataset, thread, or page I read; **[I]** = my inference from the evidence; **[C]** = an unverified claim made by a participant.

---

## 0. Headline answers

1. **No 3+ scorer has published code.** The best *public* notebook on the competition Code tab scores **2.23** (FOYSAL, "LB-9 arc3 duck v12 with Qwen 3.8 27B"); every public notebook above ~1.9 is Tufa's June "Duck" harness with the model swapped from Qwen3.6-27B-FP8 to Qwen3.8-27B-FP8 [E, Kaggle Code tab sorted by score]. Tufa's own public notebooks were last updated 2026-07-01 (`taaf-duck-harness-kaggle` v26, best 1.21; the milestone-winner notebook v6, best 1.25) — **nothing after 08-15** [E].
2. **The 4–6 band appeared in one 12-day window (08-16 → 08-28), immediately after Qwen3.8-27B was released (~08-14).** Tong Hui Kang's leaderboard-history mirror shows: cstl 1.59 (08-09) → 2.52 (08-11) → 3.57 (08-18) → 5.99 (08-24); Lord Han Solo 1.65 → 2.76 (08-16) → 3.36 (08-22) → 4.99 (08-24); Tufa 1.62 (08-10) → 2.07 (08-19) → 2.97 (08-20) → 4.58 (08-23) → 4.67 (08-25); Tong Hui Kang 0.80 (since June) → 1.06 (08-22) → 2.24 (08-23) → 3.39 (08-25) → 3.88 (08-27) → 4.27 (08-28) [E, `lb_history.json` from https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run/]. 160 teams crossed 2.0 and every single first crossing is dated 08-11 or later [E].
3. **What separates 3–6 from 1–2 (best-supported ranking, details in §8):** (a) Qwen3.8-27B-FP8 with thinking on — worth roughly ×1.5–2 by itself; (b) more actions per 9-hour run (every Duck game dies on the wall clock, never on an action cap; tokens/sec is the currency), so serving speed (newer vLLM, speculative/MTP decoding) and prompt-token economy translate directly to score; (c) memory that survives instead of the Duck's rolling eviction — retained reasoning + compaction/notes (OpenAI measured ×3 on GPT-5.6 Sol from exactly this change; a Kaggle user measured that 66.8% of Qwen's tool-call turns put the "World model:" update in hidden reasoning where the Duck never captures it); (d) verified plans / expectation checks before spending actions; (e) perception fixes (animation frames, upscaled images, no-op guards); (f) best-of-N selection over many noisy submissions (the same notebook ranges 0.7–1.3; cstl 39, Tufa 121 entries).
4. **Local-25 is not a leaderboard proxy.** Multiple posters: 5.0–5.4 local → 1.4–1.8 LB; 6.8 local → 1.19; 3.8 → 0.9–1.8; 2.8 → 2.4; Tufa 1.6 local → 1.21 LB; "5.0+ on 25 games, 1.6 on 110" (OverfitOracle, now #9 at 2.94) [E, threads 732854/736578]. The organizer: "Public Demo was made to be easier than semi private (what is used for the public leaderboard). Going from 1.56 > 0.05 is in line with expectations" [E, thread 703990].

---

## 1. Public notebooks (Task 1)

### 1.1 Score-sorted listing (Kaggle Code tab, 2026-08-29) [E]

| Public score | Notebook | Author | Fork lineage | What it is |
|---|---|---|---|---|
| 2.23 | `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` (254 votes) | FOYSAL (#70) | copied from SpeedSci `caoyupeng/arc3-duck-v12-1d7d88` ← Tufa share notebook | Tufa Duck harness, source bundle `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`, model swapped to `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (Kaggle Model, hf-fp8, 18 shards incl. `mtp.safetensors`) |
| 2.14 | `samrishb/sam-solver` | Samrish B | copied from FOYSAL | Same, plus "grafts" that are **disabled** in the scoring build ("CONTROL RUN … only graft A, the inert level probe, is active. The single live change versus the run that scored 2.14 is that 2.14's own model is restored") |
| 2.13 | `keithtyser/duck-qwen3-8-27b-fp8` (95 votes) | ktyser | from Tufa milestone notebook | June Tufa bundle (`keithtyser/taaf-duck-qwen38-serving-v1`, byte-identical `src/` to Tufa's June share except wheels) + Qwen3.8 |
| 2.02 | `yocybercode/thui-v1-1-r2` | yocyber-code | from Tufa | "Thuitanium": solver untouched; pins `LOCAL_ANALYZER_SEED=20260825`, temperature 0.6 unchanged |
| 1.99 | `yanggod/arc-agi-3-duck-fork-baseline-reproduction` | Boopathi Raja | Duck | baseline reproduction |
| 1.93 | `thtennant/arc3-duck-v22` | Teddy Tennant | Tufa share + `thtennant/taaf-kaggle-source-share-fork` (adds `src/taaf-grafts`) | Duck + Qwen3.8 + report-only "grafts" (see 1.3) |
| 1.91 | `kunaldesale2408/duck-harness-fast-eval` | Kunal Desale | Jakob anim bundle + Qwen3.8 | same shape as FOYSAL |
| 1.84 / 1.83 / 1.81 | Sangita Bannore 2 / Rahul Jiwane 8 / nabidnur | — | Duck forks | — |
| 1.71 | `jakobbrggen/taaf-model-20260815-q38-p1` (91 votes) | Jakob Brüggen (#310) | his own anim bundle + `jakobbrggen/qwen3-8-27b-fp8-hf-snapshot` | Duck + animation awareness + Qwen3.8 (the only diff vs his 1.61 notebook is the model dataset) |
| 1.61 | `jakobbrggen/taaf-anim-arc-agi-3-solver` (122 votes) | Jakob Brüggen | Duck (Qwen3.6) | animation frames + hard no-op guard (write-up in thread 734369) |
| 1.53 | `gktrkakman/taaf-duck-sub-20260805-share` | Göktürk | Duck (Qwen3.6) | adds `compaction.py` |
| 1.25 | `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` (286 votes) | Tufa Labs | root | Qwen3.6-27B-FP8, June bundle |
| 1.21 (best V21) | `jeroencottaar/taaf-duck-harness-kaggle` (243 votes) | Tufa Labs | root | the actual milestone-winning notebook |
| 0.86 | `mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone` | Md Boktiar Mahbub Murad | Gemma-4-31B "forge" | milestone 3rd |
| ≤0.54 | pre-Duck era (StochasticGoose 0.25, FORGE BFS, Persistent-memory BFS 0.46, "version 47" 0.54, Gemma-4-31B reflection agent by Reki = milestone 2nd) | — | — | search/CNN/VLM-policy agents |

Takeaways: the entire public frontier is one code base (Tufa's TAAF Duck) plus a model swap; the swap Qwen3.6→3.8 moved the same code from ~1.2–1.6 to ~1.9–2.2 on the LB [E]. The reported "Qwen3.8 is a consistent 2x on the local 25" [C, Ya Xu #45, thread 735243] is consistent with that.

### 1.2 The Duck harness itself (read from the June source bundle `jeroencottaar/taaf-kaggle-source-share`, and the 2026-08-07 bundle) [E]

Files: `ds/jeroencottaar_taaf-kaggle-source-share/src/ARC3-Inference/inference/agent/{tool_agent.py (2.4k lines), prompts.py, python_tool_sandbox.py, runtime_state.py, vision_context.py}`, `inference/framework/{solver.py, kaggle.py}`, `inference/utils/segmentation.py`, `configs/inference.json`, plus TAAF (`tufa-arc-agi-framework/src/taaf/*`).

* **Model / serving**: `vrfai/Qwen3.6-27B-FP8` on vLLM 0.19 (`driessmit1/arc3-vllm-h100-wheelhouse-v3`), `--max-model-len 65536`, prefix caching on, `reasoning-parser qwen3`, `tool-call-parser qwen3_coder`, `preserve_thinking: true`, `gpu_memory_utilization 0.92`, TP=1. Analyzer sampling: thinking on, temperature 0.6, top_p 0.95, top_k 20, `max_output 0` (= uncapped output), `LOCAL_ANALYZER_CONTEXT_WINDOW=32768` (agent trims its own prompt to 32k inside a 64k server window).
* **Action selection**: LLM per turn, one tool (`python`), sandboxed 30 s / 1024-token output cap; the REPL is preloaded with `current_frame` (`.ascii`, `.segmentation`, `.level`, `.step`), `previous_frame`, `history`, `transitions`, `last_action_result`, `valid_actions`, and an `action([...])` function that executes one or a batch of real actions from inside Python (loops allowed). Raw numeric grid deliberately hidden; segmentation (4-connected components with shape hash, boundary, containment, adjacency) is the "primary view". Actions exposed: UP/DOWN/LEFT/RIGHT/SPACE/MOUSE(row,col)/RESET; **UNDO (ACTION7) is not exposed** ("the model fails to use it efficiently" — write-up 717133).
* **Tool steps per turn**: `LOCAL_ANALYZER_TOOL_STEPS=0` in the Kaggle setup = unlimited python calls per turn (config default 12). The prompt says "Do not ration tool calls when the state is unclear."
* **Memory**: (1) rolling eviction — `_trim_messages_for_context` drops the oldest user+assistant+tool block until the estimated input ≤ 32k tokens, always keeping the system prompt; (2) an optional text note — the assistant is invited to write `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, `Cross-level notes:` lines in its *visible* content, which `_update_summarized_knowledge_from_assistant` copies into the next user message. Nothing reads hidden reasoning for this.
* **Perception**: a 4× upscaled PNG of the current frame is attached to every user turn (`multimodal.upscale=4`, chosen because Qwen's encoder uses 16×16 patches); ASCII crops; segmentation tool. Only the *last* frame of an animated action is shown (bug/omission fixed by Jakob's fork).
* **Level transitions**: TAAF `HarnessSolver` runs each game as one continuous run; on `level_completed` the next user message says "You have progressed to a new level!"; on `GAME_OVER` the runner auto-RESETs before the next turn (`_execute_auto_reset`); `ONLY_RESET_LEVELS=true` is set in the notebook so RESET keeps the level; `_level_number = levels_completed+1`.
* **Budgets/concurrency**: `concurrency=28` games in parallel, `max_runtime_s_per_game=7920` (132 min), `analyzer_timeout=900` s per model call, `max_actions_per_game=None`. 110 games / 28 × 2.2 h ≈ 8.6 h, i.e. the whole 9-hour budget. In competition mode the game list is read from the Kaggle gateway (`http://gateway:8001/`).
* **Reward signal seen by the model**: `reward = (levels_completed − previous)/number_of_levels`; `board_changed` is a whole-frame diff (so HUD/timer bars make it always-true on some games).
* **Claimed numbers**: 1.6002 ± 0.4475 on the 25 public games × 20 tries; LB 1.21 (a 1.30 was retracted as an excess submission); "very unstable, ranging from 0.7 to 1.3" for the same notebook [E, 717133, 716696].

### 1.3 Notable public forks and what they changed [E]

* **Jakob Brüggen, "TAAF Anim" (1.61 / 1.71 with Qwen3.8), write-up thread 734369**: (i) *hard no-op guard* — the harness blocks re-executing an action already proven to have no effect in the identical (level, board) state, costing one LLM turn but no environment action; (ii) *animation awareness* — the API returns a frame list per action; the Duck only read `raw.frame[-1]`; 13/25 public games return multi-frame responses (sp80: 22 frames, 624 transient pixels; bp35: every action animates). He adds compact per-action animation metadata, an `animation()` diff-timeline tool, and a stuck-hint. Honest measurements: public A/B +1.4% (p=0.92, 6 games × 4 passes); tokens per action rose 384 → 449 (+17%) and "**every run in both arms hits the 132-minute wallclock cap. Nothing ends because of an action limit** … more tokens per action meant fewer actions"; the model called the tool at the wrong moments (2 of 181 calls informative); the single most informative call died at the 30 s sandbox timeout. His 08-26 bundle (`jakobbrggen/taaf-kaggle-source`, label `poly-20260826-smoke`) is a new **"Polyphony" arm: Observe → Edit → Plan → Act** with a workspace (`policy.py`, `goal.py`, `notes.md`), a trace log, a verifier that certifies a policy by retrodicting the observed transitions, a bounded BFS planner over the certified policy, a bootstrap probe of every action class at level start, compaction that summarises rather than evicts, and a policy deadline that falls back to direct play. No score yet.
* **Göktürk `taaf-duck-sub-20260805`** (1.53): adds `inference/agent/compaction.py`.
* **thtennant `arc3-duck-v22/v28`** (1.93 for v22): Tufa share bundle + `taaf-grafts` composite; v28's cell-12 comments contain a long list of *measured* Duck pathologies worth quoting: the stock harness carried a non-empty world model on only 33 of 481 turns; every archived game ends on the 7920 s wall clock and none on the action cap; the model generates ~30.7 tok/s so the per-game budget is ~240k generated tokens, 97.6–98.3% of them reasoning; 12/12 games ended mid-analysis without an action; RESET restores the level opening exactly (306/306) and the model chose it deliberately 3 times in ~1980 actions; `board_changed` is true 100% of the time on 10/25 public games because of HUD bars; the win frame is mis-read (on a level-completing step `frame[-1]` is already the *next* level's opening board); only 41% of runs ever pressed every declared action.
* **jacquesbuis `taaf-src-mc-r4-trt-s20260829`** (pushed today, branch `apex`): vLLM 0.24.0 + transformers 5.13, Qwen3.8-27B-FP8, `reasoning_effort: "xhigh"` + `preserve_thinking`, an optional `--speculative-config` MTP arm ("spec-decode OFF mandatory (N14)" in its campaign note, with automatic fallback if vLLM fails to start with it), `LOCAL_ANALYZER_TOOL_STEPS=2`, a "control_memory" (required structured memory object + retraction), an "output_budget" (256–1024 tokens), a vLLM watchdog/relauncher, a "Duck-CEGIS" shadow-evidence plane (lossless per-action frame ledger, transition graph, object-effect matcher, prediction contracts that halt a batch on mismatch), and a "Duck-Retrodict Lite" gate. Uses Jack Cole's `jcole75/arc3-qwen36-runtime-wheels` stamp. Score unknown.
* **BlackCat Dual-Mind Router C05** (lucifer19): Qwen2.5-1.5B critic + Qwen3.8 primary; states its parent anchor scored 1.47.
* **boristown / kunaldesale "fast-eval"**: ACTION7 round-trip patch, animation metadata, then a "score-stability rollback" restoring the original prompt; boristown is #41 at 2.41.
* **Jason Feng, thread 734843**: "Tool-call responses with hidden reasoning but zero visible content: 1,723 / 2,580 = 66.8%" across all 25 games — i.e. the Duck's `World model:` note mechanism silently loses most updates with Qwen; DeepSeek-V4-Flash improved 9% → 11% (local) when forced to write visible updates.
* **Community datasets that reveal directions being tried** (Kaggle dataset search, all Aug 2026): `justforgags/arc3-duck-lora-sft` + `arc3-sft-trajectories` (LoRA SFT on Duck trajectories, 08-20/26); `mikedan7/qwen2-vl-2b-arc3-schema-qlora-bf16` (QLoRA on public "schema" traces); `mariogemoll/arc-prize-2026-arc-agi-3-vlm` ("Qwen3.6-27B VLM with ARC-AGI-3 perception LoRA merged in"); `ataraxian/arc3-duck-prompt-v24a…v36a` (Ya Xu, #45 — prompt-only iterations); `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` (Akhil Tolani, #12 — vLLM 0.27.1); `jcole75/arc3-qwen36-runtime-wheels` ("ARC3 Qwen3.6 NVFP4 Runtime Wheels", MindsAI); `romainfabre/muse-vllm-nightly-cu129-wheelhouse` (rfbr, #5, 08-17); `sonphamorg/arc3-flashnext-gcp-runtime-exact-v1` (08-28) and `alisalmanrana/qwen3-8-flash-next-gguf` (OverfitOracle/Abstraction Lab, #9) — both targeting **Qwen3.8-Flash-Next** (125B MoE, 6B active, released 08-26); `dangkhoa2016/z-lab-qwen3-8-27b-dflash2` (DFlash2 speculative-decoding drafter, ~2–3.4× decode speed).

---

## 2. Discussions (Task 2) — concrete claims

Threads read via headless browser (Kaggle is JS-rendered): 717133, 716696, 725002, 732854, 736578, 734369, 734843, 703990, 705043, 687655, 696615, 691696, 699208, 697407, 735243, 737617, 709355, 736540, plus the votes/recent listings.

**Local-vs-LB ratios** [E, 732854 "What are your agents scoring on the 25 public games?", 736578, 703990]
* Nick Pellegrin (#242): Duck+Qwen3.8 local ~2.1 → LB ~1.4 (public notebook shows 2.2); *his own* harness 5.0–5.4 local → still 1.4–1.8 LB.
* daoviet: 6.8 local → 1.19 LB. Fususu: 3.8 local → 0.9–1.8. Scott Le Grand (#59): 3.8 local → ~0.9; separately, a Claude-built game-specific solver reached 38% RHAE on public (7/25 games, 116/183 levels) but "utterly impractical … needed Claude access and 5 days".
* mikelou1 (#37): 2.8 local → 2.4 LB. Son Pham (#40): 2.8 local. donk666 (#18): 3.5–7.5 local. OverfitOracle (#9): "stable at 5.0+ on 25 games and 1.6 on the 110 games after submission"; later "a very different approach in the harness + model … completely redesigned"; will not share.
* Akhil Tolani (#12): 1.56 local (9/20 L0 clears) → 0.02–0.05 on Kaggle in May; Greg Kamradt: public demo is easier than semi-private; scoring runs all 110 games.
* Reki's framing question: "Are people's scores mostly limited by not beating levels, or by beating them with far more actions than humans?" — PRASHANT SHUKLA91: "Anyone who broke the bottlenecks of won games: 0/25 so far."

**What the top teams have said** [E]
* Tufa (717133): design principles (fit 96 GB, generic, lightweight, "keep the model in the driver seat"); "the main driver of improvement … came from better base models and introducing multi-modality. Against our intuition, hand-crafting specific tools did not help"; "context management could be improved by compaction or adding memory"; "we currently do not optimally use prefix caching". Jeroen: score for the same notebook ranges 0.7–1.3.
* Tong Hui Kang (705043, 2 days ago): asks whether a **fine-tuned model's dataset and fine-tuning script** must be published for the milestone/final prizes ("If a fine-tuned model is used, the model has to be published before the deadline") — the only public hint about his method [I: suggests he is fine-tuning].
* Jakob Brüggen on "Sudden increase in top 3 teams?" (737617): could not find a public release explaining the simultaneous jump; points at the community leaderboard's Polyphony Agent (27B). OverfitOracle: "they didn't share any code".
* Ya Xu (#45): Qwen3.8-27B-8bit is "a consistent 2x on the local 25"; "3 × 2.0+ scores in 12 hours is not a coincidence".
* Scott Le Grand / Drona Bajaj: overnight ("nightly") submissions score 30–50% lower than daytime ones; suspected shared-PCIe/multi-tenancy contention on the RTX Pro 6000 hosts [C] — consistent with runs being throughput-bound.

**Hardware / runtime facts** [E]
* Rerun: one **RTX Pro 6000 (96 GB)**, **9 hours**, 110 games; earlier a bespoke ARC-3 setting capped runs at 6 h (some at 3 h) until Kaggle fixed it (699208). H100 sessions were "effectively unavailable" in May–June; RTX queues long in August (696615). The leaderboard-history mirror shows top teams' runs at 8h50m–9h15m; Tong Hui Kang's June runs were 5–6 h.
* Submission limit issues: 5/day; a 1.30 Tufa score was voided as an excess submission (705405).
* Milestone rule of thumb (Tong Hui Kang): winners publish 1–2 h before the deadline; copies cannot score in time; ranking is by the snapshot at the deadline.

**Scoring gotchas / exploits** [E]
* Jeroen (687655, unresolved): `environment_info.baseline_actions` (the human baseline used for scoring) "are available to us for the 110 test games during submission"; asked whether it is intentional. Tufa's June bundle even ships `re-arc-3` with `metadata_baseline_actions`.
* "It is 0.66%" (CPMP, 691696): LB numbers are percentages; the human ceiling is 100.
* Per-game runtime is the binding constraint (Jakob, thtennant); per-level cap 115; score capped by completion share, so a level-1-only clear on a 6-level game is worth at most 100·1/21 ≈ 4.8 for that game regardless of efficiency.
* Reki on a notebook copy scoring 0.00: "almost certainly an infra/timeout failure".

**Pure search / no-LLM** [E]: pre-Duck public agents (BFS/MCTS/graph exploration/CNN StochasticGoose) top out at 0.25–0.54; "Explore Before You Solve" (arXiv 2605.25931) shows 10 public games are reachable in one blind step and 5 after one probe, but its Qwen2.5-0.5B agent scored 0.21 public / 0.30 on the private 55. Nothing search-only is anywhere near 2.

**Human baseline** [E, arcprize.org/blog/arc-agi-3-human-dataset; arXiv 2603.24621]: 458 participants / 342 replays on the 25 public games (145 solves); baseline changed from "2nd-best player" to **median player per level**; per-level cap raised to 115%; the 25 public games have 183 levels and a 17,135-action human total (≈94 actions/level, from VISTA's table); 135 environments total (25 public / 55 semi-private / 55 private); score = min(1.15, h/a)² per level, linear level weights, mean over games.

---

## 3. Top-team footprint (Task 3)

Leaderboard members (Kaggle leaderboard page, member avatars) [E]:

| Rank | Team | Members (Kaggle handle) | Public footprint | Approach evidence |
|---|---|---|---|---|
| 1 (5.99) | **cstl** | `tehnar` ("Tehnar", software engineer, Amsterdam), `gatamaz` ("TG" = Tamaz Gadaev, ML R&D, MIPT, Lead ML Eng at Jhourney, GitHub `tamazgadaev`) | zero ARC-related public notebooks/datasets/posts; tehnar's one Kaggle discussion post and a 2015 Theano notebook | **None.** History: 0.13 (06-17) → 0.52 (07-08) → 1.21 (07-19) → 1.59 (08-03) → 2.52 (08-11, *before* Qwen3.8) → 2.70 → 2.81 (08-17) → 3.57 (08-18) → 5.99 (08-24); every run 8h50m–9h13m; 39 entries [E]. [I] Already ~1.6× Duck on Qwen3.6, then roughly ×2.2 within 10 days of Qwen3.8. |
| 2 (4.99) | **Lord Han Solo** | `lordhansolo` (software engineer, Białystok, Poland; solo) | nothing public; 2 competitions | History: 1.25 (07-08) → 1.41 → 1.47 → 1.65 (08-04) → 2.76 (08-16) → 3.36 (08-22) → 4.99 (08-24); 46 entries; every run ~9h [E]. |
| 3 (4.67) | **Tufa Labs** | `dlorah` (Harold Bessis), `driessmit1` (Dries Smit), `pressman1` (Isaiah Pressman), `stefano1283` (Stefano Viel) (+ Jeroen Cottaar, Michal Tešnar) | Duck harness (Kaggle notebooks, GitHub `Tufalabs/duck-harness`, tufalabs.ai/research/duck-harness, MLST episode). **Repo last commit 2026-07-01; no post after July** [E] | History: 1.21 (June) → 1.45 (07-18) → 1.62 (08-10) → 2.07 (08-19) → 2.97 (08-20) → 3.04 → 4.58 (08-23) → 4.67 (08-25); 121 entries [E]. Their public philosophy: better base model + multimodality; compaction/memory named as next step. |
| 4 (4.27) | **Tong Hui Kang** | `huikang` (SUTD; Notebooks Master; AIMO/Nemotron veteran) | blog.huikang.dev (autoresearch post: NN policy over 384×64×64 tensors on modified games — "no evidence … better than random", disclaimer added Aug 2026); historical LB monitor arc3.huikang.dev; asks about publishing fine-tuned models | History: 0.35 (05-31) → 0.80 (06-29, 5–6 h runs) → idle → 1.06 (08-22) → 2.24 (08-23) → 3.39 (08-25) → 3.88 (08-27) → 4.27 (08-28); 56 entries [E]. [I] Fine-tuning likely; daily +0.4–1.2 steps suggest a fast iteration loop. |
| 5 (3.37) | **rfbr** | `romainfabre` (Romain Fabre, Research Scientist at Kyutai, Paris) | dataset `romainfabre/muse-vllm-nightly-cu129-wheelhouse` (7.5 GB, 08-17) | History: 0.66 (08-12) → 0.71 (08-18) → 1.03 (08-19) → 1.81 (08-20) → 2.19 (08-24) → 3.37 (08-26); 15 entries [E]. [I] The "muse … nightly vLLM" wheelhouse one week after Meta's open-weight **Muse Glimmer 30B** (Apache-2.0, multimodal, agentic, released 08-10) and immediately before his climb suggests a non-Qwen model axis. |
| 6 (3.17) | **Tony G** | `junvalue` ("Tony") | nothing public | 0.24 (08-05) → 1.09 (08-20) → 1.65 → 2.45 (08-23) → 3.17 (08-24); 14 entries [E]. |
| 7 (3.15) | **Daniel Franzen** | `dfranzen` (ARChitects; Mainz) | GitHub `da-fr` has no ARC-3 repo; Kaggle datasets are 2024–25 ARC-2 assets | 0.35 (June) → 0.96 (07-17) → 1.24 (07-27) → 2.58 (08-14) → 2.88 (08-22) → 3.15 (08-28); 55 entries [E]. |
| 8 (2.98) | **OzanM.** | `analyticaobscura` (Ozan Mohurcu, Istanbul; 2× Notebooks GM) | no ARC-3 public work | 1.04 (07-01) → 1.21 → 1.41 (08-13) → 2.17 (08-24) → 2.98 (08-27); 99 entries [E]. |
| 9 (2.94) | **Abstraction Lab & MindsAI** | `jcole75` (Jack Cole), `ultsaza`, `kimura0415`, `alisalmanrana` (OverfitOracle, "Research Scientist at Abstraction Lab"), `sumirinn` | `jcole75/arc3-qwen36-runtime-wheels` (Qwen3.6 **NVFP4**, 07-10); OverfitOracle's `qwen3-8-flash-next-gguf` model; his posts: "harness + model completely redesigned, 5.0+ stable on public, 1.6 on 110" | 1.17 (07-09) → 1.91 (08-15) → 2.05 (08-16) → 2.94 (08-26); 131 entries [E]. |
| 10 (2.80) | **Tony Li** | `tonylica` (Toronto; Competitions Master #16) | many other competitions; no ARC-3 assets | 0.76 (07-06) → 1.01 → 1.20 (08-18) → 2.07 (08-19) → 2.39 → 2.80 (08-23); 14 entries [E]. |

Other names: #12 Akhil Tolani (`saltb0x`) publishes a vLLM 0.27.1 wheelhouse and Qwen3.8 FP8 mirror; #40 Son Pham & Mark Barney run the instrumented Duck fork `sonpham-org/arc-3` (README: best validated config 1.62 local; "no-impact detection" +55%; state-graph regression; "the 25-game mean exhibits 95% noise range 0.45–2.67"; ft09 alone swings ±1.0; model swaps to 35B-A3B, GLM-4.6V, Gemma-4-31B all failed) [E].

**June 30 milestone winners** [E, arcprize.org/blog/arc-prize-2026-milestone-1; thread 725002]: 1st Tufa "The Duck" (Qwen3.6-27B-FP8, REPL, eviction, multimodal, segmentation; $25k); 2nd Reki (Gemma-4-31B vision-LLM-as-policy, labeled frame images → JSON plan + 1–4 actions, reflection memory every ~10 steps, numpy click heuristics preferring small rare-colored button-like shapes, "dead-signature" to stop clicking inert object types; notebook `ruichardliu/milestone1-2nd-solution`); 3rd Md Boktiar Mahbub Murad "forge" (same VLM pattern, generator+arbiter, "top-scoring run used a profile that turns off all of the extra machinery"; 0.86). Second milestone ends 09-30.

---

## 4. Official ARC Prize material and the harness literature (Task 4)

* **ARC-AGI-3 human dataset** (04-14): 458 people; baseline = median per level; cap 115% [E].
* **GPT-5.5 & Opus 4.7 analysis** (05-01): 0.43% / 0.18% on the semi-private set with the official harness; three failure modes — local perception without a global model, misapplied game priors (Tetris/Sokoban…), "success without comprehension" (a cleared level does not lead to the next) [E].
* **Milestone #1 post** (07-06): "what works: multimodality; flexible model choice (27B–31B); reflection/running memory; heuristic fallbacks" [E].
* **Official leaderboard, frontier models, no harness** (BenchLM mirror, 08-29): Opus 5 30.2%, GPT-5.6 Sol 7.8%, Opus 4.8 1.5%, GPT-5.5 0.4% [E].
* **OpenAI (07-31)** "two settings tripled our score": GPT-5.6 Sol 13.3% → 38.3% public by (1) *retaining reasoning* across tool calls and (2) *compaction* instead of rolling truncation; output tokens ÷6; "less time re-deriving the rules of each game" [E].
* **VISTA (MIT)**: Opus 5 → 100.00 RHAE, 25/25, 7,542 actions vs 17,135 human; 512×512 (8×) upscaled PNGs, lossless visual memory with `inspect`/`read_pixels`, `GUIDE.md` (cross-level notes) + `WORKING.md`, state expected visual outcome before acting, fresh context with continuation notes; "models were released after the public games … the private set remains the real test" [E].
* **NVIDIA AVO (08-21)**: 100.00 RHAE with 6,624 actions on Opus 5, text-only 64×64 grid; persistent memory + supervisor that flags stagnation + inspect/plan/implement/evaluate loop; public set only [E].
* **Retrodict** (Ryan Brown): 99.86% for $654 on gpt-5.6-sol: replays hypotheses in Python against the recorded log before acting; every action carries an expected board; a plan queue halts on mismatch; `playbook.md` survives 150k-token context resets [E].
* **Tycho**: programmatic world models; the *orchestration policy* (when to build/consult a model) mattered more than model accuracy; 88.49 RHAE Opus 4.8 → 100 on Opus 5/GPT-5.6 Sol [E].
* **Executable World Models** (2605.05138): GPT-5.5 15/25 games, 58.12% mean RHAE with verifier-driven executable models [E].
* **Community leaderboard** (arcprize.org/leaderboard/community): Tycho 100, Retrodict 99.9, baseline1 99.0 ($400), NOOA 85.1, OPINE-World 78.4, Polyphony Agent 19.8 ($115), Continual Harness 20.5 [E].
* **Prime Intellect "Prime Agent"** (08-05): 95.5% with Opus 5, single persistent IPython kernel, self-refining prompt/skills/memory; MIT [E, secondary sources].

All of these are frontier-model, public-set results; none is served offline on a 96 GB GPU. Their transferable lessons (memory that persists, verify-before-act, stagnation supervision, upscaled vision, action budgeting) are exactly the Duck's admitted gaps.

---

## 5. Answers to the coordinator's specific questions

**(a) Tufa public notebook version history after 08-15?** No. `taaf-duck-harness-kaggle` is at version 26 (best 1.21, V21) and the milestone-winner notebook at version 6 (best 1.25), both last run 2026-06-30/07-01; `kaggle kernels list --user jeroencottaar` shows no ARC-3 notebook newer than 07-01; GitHub `Tufalabs/duck-harness` has two commits, both 07-01; tufalabs.ai/research has no ARC-3 post after 07-01 [E]. Tufa's 4.67 code is private (121 entries, still rising).

**(b) Newer `taaf-kaggle-source` bundles?** Tufa's own datasets (`jeroencottaar/taaf-kaggle-source`, `-share`) are dated 06-10/06-12. Everything newer is community forks: `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (08-07), `jakobbrggen/taaf-kaggle-source` (08-26, Polyphony arm), `thtennant/taaf-kaggle-source-share-fork` (08-28, grafts), `jacquesbuis/taaf-src-mc-r4-trt-s20260829` (08-29, apex/CEGIS/metacontrol), `gktrkakman/*` (08-05/07), `ronitagarwal1/tiger-ii-*` (08-21), `poby7722/taaf-kaggle-source-qwen3-8-rtx-pro-6000` (08-16), `iseesmth/duck-harness-prolong/nca-*` (08-11), `autumndyer/taaf-kaggle-source-share-base-v54` (08-01) [E]. Note that Jakob Brüggen is not a Tufa member (his bundles are built from `/Users/jakobbruggen/Desktop/duck-harness`; he is #310).

**(c) Who is Lord Han Solo?** Kaggle `lordhansolo`, "Software Engineer", Białystok, Poland, he/him, joined 6 years ago, 2 competitions, 3 followers, no notebooks/datasets/discussion posts; single-member team; nothing on the web under that name [E]. Only footprint is the score history above.

**(d) thtennant's notebooks and scores.** `arc3-duck-v12` (08-09), v18, v19, v20, v21 (Qwen3.8 pin), v22 (**1.93** public), v23–v27, v28 (08-28, score not shown on the first page of the score-sorted listing, so < 1.48 or unscored) [E]. They are Tufa-share forks with a report-only "grafts" layer (efficiency note, retry guard, goalkeep, hudmask, clickmap, searchmap, clockwatch, lawbook, winframe, carryover, undo, untried) — all instrumentation/prompt injection, no change to the model or serving.

**The main question — what takes the same model class from ~1.5 to 4–6?** Nobody in the 3+ band has published; the honest answer is a reconstruction (§8). Two hard facts bound it: (1) the identical Duck code with Qwen3.8 lands at 1.9–2.2 on the LB, so the model swap accounts for roughly +0.6–1.0 of the +3 to +4.5 the leaders gained; (2) cstl was already at 2.52–2.70 on 08-11/12 *before* Qwen3.8 existed, so their harness alone was ~1.6× the Duck, and it then roughly doubled again with the new model — model and harness gains multiply.

---

## 6. Key numbers to keep

* Rerun: RTX Pro 6000 96 GB, 9 h, 110 games (55 semi-private used for the public LB; 55 private for the final). Public demo 25 games / 183 levels; humans 17,135 actions total.
* Duck defaults: 28 concurrent games, 132 min/game, 900 s per model call, 32k prompt budget in a 64k window, thinking on, T=0.6, 4× image upscale, unlimited python calls/turn, 30 s sandbox.
* Duck measured: ~30.7 tok/s per game stream, ~240k generated tokens per game, 98% of them reasoning; 384–449 tokens per environment action; every game ends on the clock.
* Variance: same notebook 0.7–1.3 (Tufa); public-25 95% range 0.45–2.67 (sonpham); ft09 swings the 25-mean by ±1.0.
* Field (08-29, 2,603 teams): median 0.27, mean 0.65; ≥1.0: 766; ≥1.5: 411; ≥2.0: 160; ≥2.5: 29; ≥3.0: 7; ≥4.0: 4. First 2.0 crossings: 1 on 08-11, then 2/1/8/4/13/19/16/13/11/16/12/10/16/13/5 per day from 08-14 to 08-28.

---

## 7. Where the evidence is thin

* No source code, write-up, tweet or blog exists for cstl, Lord Han Solo, Tong Hui Kang's current agent, rfbr, Tony G, Daniel Franzen's ARC-3 entry, OzanM, or Tony Li (searched web, GitHub, Kaggle notebooks/datasets/discussion; profiles read). Everything about their methods is inference from timing, member backgrounds, and uploaded infrastructure datasets.
* OverfitOracle's "5.0+ stable on public 25" and Ya Xu's "2x" are self-reports.
* The `baseline_actions`-visible-at-eval question was never answered publicly.
* thtennant's measurements are from that author's own run archives (quoted from notebook comments), not independently reproduced.

---

## 8. Ranked: concrete mechanisms that appear to separate the 3–6 scorers from the 1–2 scorers

1. **Qwen3.8-27B-FP8 as the brain, thinking on, run at the highest reasoning effort you can afford.** Strongest evidence of anything here: every top-10 jump is dated 08-14 → 08-28; the unmodified Duck moved 1.2–1.6 → 1.9–2.2 on the LB with only the model changed; "consistent 2x locally". Qwen3.8 ships `reasoning_effort` (xhigh/medium/low), `preserve_thinking` on by default, native vision, 262k context, and MTP heads. [E for the effect; the exact settings used by leaders are I]
2. **Throughput, i.e. actions per 9 hours.** Every Duck game dies on the 132-minute wall clock, never on an action budget; ~30 tok/s ⇒ ~240k tokens ⇒ a few hundred actions per game; adding 17% tokens per action removed up to 54% of actions on some games; overnight submissions (slower shared hosts) score 30–50% lower. Anything that raises tokens/s or lowers tokens/action — newer vLLM, MTP/DFlash2 speculative decoding, prefix-cache-friendly prompts, shorter tool outputs, no full-board dumps, fewer redundant probe turns, right concurrency for the GPU — converts into levels. The leaders' runs are all 8h50m–9h15m (using the whole budget) while Tong Hui Kang's 0.8-era runs were 5–6 h. [E for the constraint; I for what the leaders did]
3. **Memory that survives the game.** The Duck evicts the oldest turns to fit 32k and captures its "World model:" note only from visible text — which Qwen leaves empty in ~67% of tool-call turns. OpenAI tripled GPT-5.6 Sol (13.3 → 38.3) by retaining reasoning and compacting instead of truncating; VISTA/Retrodict/AVO all keep a persistent notes file or playbook across context resets and levels. Replacing eviction with summarised compaction plus an explicit, harness-owned per-game memory (controls, mechanics, goal, per-level plan, what has been tried) is the single most-cited fix and the cheapest. [E for the mechanism elsewhere; I that the leaders did it]
4. **Verify before spending an action.** Retrodict's plan queue halts on the first mismatch between predicted and observed board; Tycho found the *policy for when to build/consult a model* matters more than model accuracy; Polyphony/CEGIS forks add verifiers and prediction contracts. In the Duck the model can already batch actions from a search; the missing piece is a harness-level expectation check that stops a bad batch after one action. This raises levels per action and protects the (h/a)² term. [E for the literature; I for the leaders]
5. **Perception fixes that are pure information gain:** feed the animation frames (13/25 public games hide state in intermediate frames; ft09/sb26 show the effect only mid-animation), mask HUD/timer bands out of `board_changed`, upscale images more aggressively (VISTA uses 8×), read the winning frame correctly at a level transition, and block/flag proven no-ops (hard guard, dead-signature). Each is small; together they stop the agent from wasting turns on non-events. [E]
6. **Play for depth, not efficiency.** The per-game score is capped by the completion share, so level-1-only clears saturate at a few points per game; a 4–6 average across 110 games requires clearing multiple levels on a meaningful fraction of games. Cross-level carry-over of mechanics (controls persist 93% across level pairs; layouts do not), exploiting RESET (restores the level opening exactly), and not giving up early are the levers; efficiency polish on level 1 is worth almost nothing. [E from thtennant/TAAF measurements and the scoring formula]
7. **Selection over noisy runs.** The LB shows a team's best of up to 5 submissions/day; the same notebook spans 0.7–1.3; leaders have 39–131 entries. Part of the 4–6 is the maximum of many draws of a ~3–4 mean system — expect the private-set score of the same code to be lower. [E for the variance; I for the magnitude]
8. **(Speculative, weakly evidenced) alternative or adapted models.** rfbr's nightly-vLLM "muse" wheelhouse (Meta Muse Glimmer 30B, Apache-2.0, multimodal, open weights 08-10) right before his climb; Abstraction Lab/MindsAI and Son Pham staging Qwen3.8-Flash-Next (125B MoE, 6B active, 08-26); Tong Hui Kang asking about publishing fine-tuned weights; multiple public LoRA/QLoRA-on-trajectories datasets. None of these is yet tied to a scored notebook. [I/C]

What does **not** appear to matter: hand-built game-specific tools (Tufa: they hurt), search-only/no-LLM agents (≤0.54), swapping to other ≤35B open models on the same harness (sonpham: all regressed), and efficiency tuning on already-cleared levels.

---

## Sources (URLs / refs)

Kaggle notebooks: jeroencottaar/taaf-duck-harness-kaggle · jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner · foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b · keithtyser/duck-qwen3-8-27b-fp8 · samrishb/sam-solver · yocybercode/thui-v1-1-r2 · thtennant/arc3-duck-v22, v28 · jakobbrggen/taaf-anim-arc-agi-3-solver, taaf-model-20260815-q38-p1 · kunaldesale2408/duck-harness-fast-eval · boristown/agi-duck-harness-fast-eval · lucifer19/blackcat-dual-mind-router-c05 · ruichardliu/milestone1-2nd-solution · mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone
Kaggle datasets/models: jeroencottaar/taaf-kaggle-source-share · jakobbrggen/taaf-kaggle-source-anim-20260807-anim · jakobbrggen/taaf-kaggle-source · thtennant/taaf-kaggle-source-share-fork · jacquesbuis/taaf-src-mc-r4-trt-s20260829 · gktrkakman/taaf-kaggle-source-duck-sub-20260805 · keithtyser/taaf-duck-qwen38-serving-v1 · driessmit1/arc3-vllm-h100-wheelhouse-v3 · foysalemonshanto/qwen3-8-27b-fp8-repacked-v1 · jcole75/arc3-qwen36-runtime-wheels · romainfabre/muse-vllm-nightly-cu129-wheelhouse · saltb0x/arc3-vllm-wheelhouse-v0271-cu129 · sonphamorg/arc3-flashnext-gcp-runtime-exact-v1 · alisalmanrana/qwen3-8-flash-next-gguf · justforgags/arc3-duck-lora-sft · dangkhoa2016/z-lab-qwen3-8-27b-dflash2
Kaggle discussions (competition arc-prize-2026-arc-agi-3): 717133, 716696, 725002, 732854, 736578, 734369, 734843, 703990, 705043, 687655, 696615, 691696, 699208, 697407, 735243, 737617, 709355, 736540; leaderboard page; user profiles tehnar, gatamaz, lordhansolo, romainfabre, analyticaobscura, junvalue, tonylica, jcole75, alisalmanrana, dfranzen, dlorah, huikang
History: https://arc3.huikang.dev/leaderboard (data: https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run/)
External: https://tufalabs.ai/research/duck-harness/ · https://github.com/Tufalabs/duck-harness · https://github.com/sonpham-org/arc-3 · https://arcprize.org/blog/arc-prize-2026-milestone-1 · https://arcprize.org/blog/arc-agi-3-human-dataset · https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis · https://arcprize.org/leaderboard/community · https://arxiv.org/html/2603.24621v2 · https://openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores/ (via gigazine mirror) · https://vista-research.github.io/ · https://developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-… · https://github.com/ryanbbrown/Retrodict · https://arxiv.org/html/2607.28287v1 (Tycho) · https://arxiv.org/abs/2605.05138 · https://arxiv.org/abs/2605.25931 · https://blog.huikang.dev/2026/05/31/autoresearch-hackathon.html · https://benchlm.ai/benchmarks/arcagi3 · https://huggingface.co/Qwen/Qwen3.8-27B-FP8 · https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model · https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2
