# Model-swap scan — 2026-08-30

Question: is there an open-weights model swap (one 96GB RTX Pro 6000, vLLM, agentic) that plausibly beats Qwen3.8-27B-FP8 in the duck per-action harness, and what is the cheapest reliable A/B?

**Answer: yes, one serious candidate — Qwen3.8-Flash-Next 125B-MoE (6B active) via the RadixArk NVFP4 checkpoint that sonpham has already packaged into three Kaggle datasets with the exact runtime. The case is throughput (we are prefill/wall-clock-bound, and 6B-active ≈ 4.5x less compute per token than 27B dense), not benchmark quality (≈ parity). The A/B costs zero submission slots: a serving-gate kernel (~2-4h GPU) then one 25-game smoke (~4.6h), both against baselines we already measured this week.**

---

## 0. Leaderboard movement since 08-29 (verified via `kaggle competitions leaderboard`, 08-30 09:35 UTC)

| # | team | score | note |
|---|---|---|---|
| 1 | cstl | 5.99 | sub 08-29 19:25 (unchanged from the 08-29 11:40 read) |
| 2 | **Lord Han Solo** | **4.99** | **NEW at #2**, sub 08-29 22:18 — not in our 08-29 notes |
| 3 | Tufa Labs | 4.67 | sub 08-29 17:42 |
| 4 | Tong Hui Kang | 4.27 | sub 08-29 23:54 |
| 5 | rfbr | 3.37 | sub 08-29 14:52 |
| 9 | Liao Zixu | 3.13 | |
| 11 | Abstraction Lab & MindsAI | 2.94 | dropped out of top-10 |

Us: 1.74 (rank ~291). No sub yet dated 08-30. Discussion posts since 08-28: **not retrievable** — Kaggle discussions have no CLI endpoint and the web page is a JS shell (WebFetch returns title only). The dataset-activity signals below are the substitute evidence.

---

## 1. Ranked shortlist

### #1 — Qwen3.8-Flash-Next (RadixArk NVFP4) — **GO for the A/B**

**What it is (verified):** Alibaba's open-weight Qwen4-architecture preview, released **08-26**. Multimodal ultra-sparse MoE: **125B main params, 6B active/token** (48 layers, 512 experts, top-10+shared), plus a **separate 51B n-gram "PLE" table** (offloadable) and a 4B MTP head — ~180B total, 360GB BF16. Hybrid GDN + QSA sparse attention (small KV state). 262k native context. Day-0 vLLM support "verified on NVIDIA and AMD GPUs" (vLLM tweet x.com/vllm_project/status/2092600887873286157; MarkTechPost 08-26; huggingface.co/Qwen/Qwen3.8-Flash-Next).

**Fit on 96GB (verified math + third-party demo):**
- Official FP8 = **172.78 GiB → does NOT fit**; BF16 335 GiB (recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next: min validated FP8 deploy is TP2).
- **RadixArk/Qwen3.8-Flash-Next-NVFP4** (released 08-25, ModelOpt NVFP4 W4A4 on routed experts only; attention/GDN/shared/embeddings/MTP byte-identical BF16; PLE tables FP8): **135GB checkpoint**. With the 51B PLE table offloaded to host RAM (`VLLM_PLE_CPU_OFFLOAD=1`, needs ≥51GB host RAM — documented on the vLLM recipe page), GPU-resident ≈ 83GB. A community variant demonstrates exactly our target: "**one 96 GB Blackwell GPU with 88.8 GiB of VRAM, the 51B n-gram table in host RAM**" and a "validated single RTX PRO 6000 profile" (huggingface.co/garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream — **single-GPU claim is from search-result text; not yet reproduced by us — that is what the serving gate is for**).
- Kaggle host RAM: our own env captures show **MemTotal 185,473,264 kB (~177 GiB)** on the RTX Pro 6000 hosts → PLE offload fits.
- KV headroom is **~7-13GB, not the 40GB the ideal asked for** — but the hybrid GDN/QSA architecture keeps only a fraction of layers as full attention, so per-token KV is several times smaller than a dense 48-layer model, and the duck runs a 24k window anyway. NVFP4 W4A4 is Blackwell-native — RTX Pro 6000 is exactly the right silicon.

**Quality (verified numbers, mixed picture):**
- NVFP4 vs BF16: GSM8K 97.27, AIME26 pass@1 98.75 — in-band with BF16 reference (RadixArk card). sonpham independently reproduced 0.9875 on their snapshot (`aime26_metrics.json` in the dataset).
- vs our 27B (benchlm.ai/models/qwen3-8-flash-next, datacamp.com/blog/qwen3-8-flash-next, orcarouter.ai/blog/qwen-3-8-27b-benchmarks): SWE-bench Pro **62.5 vs 61.7** (parity-plus), LiveCodeBench v6 91.9 vs 90.3, Agentic category #7/140 (70.5). **Terminal-Bench 2.1: Qwen publishes no figure for Flash-Next** (27B = 73.0); OSWorld numbers are on different benchmark versions (19.4 on "OSWorld 2.0" vs 27B's 84.3 on "OSWorld-Verified") and not comparable. So the quality proxy that predicted the 3.8-27B jump is **unavailable** — quality parity on grid-agentic work is the hypothesis the A/B must test, not a given. 6B-active reasoning depth is the main risk.
- RadixArk behavioral note: "long agentic generations tend to run longer than BF16" — mildly adverse for a wall-clock-bound harness; measure it.

**Throughput case (the actual lever):** Every duck game dies on the 9h wall clock (R4 audit: 27B does ~30.7 tok/s/stream, 297 tok/s aggregate at conc 28, 47.4 actions/game stock). 6B active vs 27B dense ≈ 4.5x less FLOPs/token for prefill and decode, plus an MTP head for speculative decode. RadixArk measured 440 tok/s output on their (GB-class) rig. If even 1.5-2x survives on our GPU, that is the "more actions per run" axis the 08-29 independent review named binding.

**vLLM/serving (verified):** vLLM **0.29.0+** required; tool parser `--tool-call-parser qwen3_xml`, `--reasoning-parser qwen3`; `VLLM_PLE_CPU_OFFLOAD=1`; pipeline parallel unsupported (vLLM recipe page). Our own 0.27.1 wheelhouse boot **failed twice** on Kaggle (`submission/_serving_lab2/v1-boot-fail-flashinfer-sm75.log`, `v2-boot-fail-cuda129-nvcc.log`) — do not build our own runtime; **sonpham ships the exact pinned runtime** (`flashnext-gcp-container-site-packages.tar.zst`, 6.0GB, inside part A). Note the RadixArk card itself serves via SGLang (`qwen4_exp`, TP2 on GB300/B300); sonpham's README says "pinned vLLM runtime" — whichever engine is inside their tarball is the one to reuse verbatim.

**License:** Qwen Community License 1.0 per the HF LICENSE file [weakly verified — from search snippets; read the file before shipping]. Not Apache; confirm competition acceptability.

**Direct ARC-AGI-3 evidence:** No one has scored it on the LB yet (released 4 days ago). But the field is tooling up fast:
- `sonphamorg/arc3-flashnext-serving-part-{a,b,c}-v1` (53.3 + 40.8 + 41.6 GB, updated **08-29 18:41-19:50**) — a complete offline 3-dataset serving package: RadixArk NVFP4 rev `7b71922`, ZIP64-STORE transport pre-expanded by Kaggle, zero-copy symlink view, exact runtime + source bundle. Provenance names a Kaggle **model** too: `sonphamorg/qwen3-8-flash-next-plebf16-gcp-exact/PyTorch/gcp-exact-serving/1`. Plus `sonphamorg/arc3-flashnext-gcp-runtime-exact-v1` (6.0GB, 08-28).
- `xugeger/qwen3-8-flash-next-ud-q2-k-xl` (Unsloth UD Q2_K_XL GGUF, 78GB, 08-27) + a llama.cpp source dataset — a second team going the llama.cpp route (Q2 quality + llama.cpp ≠ our lane).
- I could **not** find an Abstraction-Lab-attributed Flash-Next dataset via CLI search (the claim in the tasking is unconfirmed; the two publishers found are sonpham and xugeger).

**Go/no-go: GO — run the zero-slot A/B now.** A ready-made snapshot exists (saves the 135GB upload entirely); the binding constraint (throughput) is exactly what a 6B-active MoE attacks; quality is at benchmark parity where measurable.

### #2 — Meta Muse Glimmer 30B — **NO-GO as primary; cheap backup probe only**

Verified: dense 30B multimodal agentic model, **Apache 2.0**, released **08-10**, `meta-models/Muse-Glimmer-30B` (research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model; marktechpost.com 08-10). FP8 ~30GB → fits with huge KV headroom. vLLM serves it only via `--model-impl transformers` (no native impl → slow path) on recent nightlies.

**This is almost certainly what rfbr (#5, 3.37) runs**: their `romainfabre/muse-vllm-nightly-cu129-wheelhouse` (7.5GB, 08-17) pins `vllm==0.27.2rc1.dev122+g8efa13b70.cu129` + `transformers==5.15.0` + `torch 2.13.0+cu129` — a nightly vLLM named "muse", dated a week after Muse's release and before Flash-Next existed. [Inference from the dataset name/pins; rfbr's kernel is private.]

Why no-go: on the proxy that predicted the 3.8 jump it **loses to our current model** — Terminal-Bench **-21.3 pts (-41%)** and SWE-bench Pro -10.5 vs Qwen3.8-27B (kingy.ai/blog/muse-glimmer-30b-benchmarks-hardware-run; orcarouter.ai/blog/qwen-3-8-27b-vs-muse-glimmer); it wins only on agentic-orchestration benches (tau3-Banking 23.5 vs 16.7). And rfbr's 3.37 sits *below* Tong Hui Kang's 4.27 (presumed 27B private harness) — Muse is not outscoring the 27B in the field. Ready snapshots exist if we ever want the probe (`nick2187/muse-glimmer-30b-nvfp4-preyazz` Kaggle model, `sergiodefreitas/muse-glimmer-30b` 17.6GB dataset, several GGUF mirrors).

### #3 — NVIDIA Nemotron 3.5 Lightning-30B-A3B — **NO-GO on quality; keep as throughput floor**

Verified: released ~08-11, 30B MoE / **3B active**, official **NVFP4** checkpoint (+DFlash/DSpark speculative decode variants), OpenMDW license, first-class vLLM recipes (huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4; build.nvidia.com modelcard). Fits trivially (~17GB) with maximal KV headroom and would be the fastest option by far. But SWE-bench Verified **51.6%** — clearly below the 27B class; a 3B-active brain on ARC grids is a comprehension downgrade, and comprehension (not budget) is our measured constraint.

### #4 — Same-model serving upgrade: RadixArk/Qwen3.8-27B-NVFP4 — **not a model swap; cheap insurance arm**

huggingface.co/RadixArk/Qwen3.8-27B-NVFP4 exists. NVFP4 W4A4 on Blackwell ≈ 2x GEMM throughput vs FP8 for the *same* brain — same binding constraint, zero quality-identity risk. If the Flash-Next gate fails to boot, this is the fallback experiment on the same serving-lab kernel (needs a quant-quality check; W4 on a dense 27B is lossier than experts-only W4 on a 125B MoE).

**Everything else in the 08-14..08-30 window** (checked): DeepSeek V4-Pro GA, GLM-5.2, MiniMax M3, Nemotron 3 Ultra — all far beyond 96GB single-GPU; Cohere's new 30B agentic coding model [existence from one search snippet, unverified] — Cohere licenses typically non-commercial, skip. Nothing else with an agentic profile above the 27B fits one 96GB card.

---

## 2. The exact A/B recipe for the top pick (zero submission slots until phase 2)

**Baselines already banked (no new stock run needed):** `arc3-tp-smoke` v1 stock phase, 08-29, 25 games x 7,920s on the real Kaggle RTX Pro 6000: **47.4 actions/game, 1.00 levels/game, 8 zero-level games, score 3.86, 1,213 requests, prefix-hit 0.29, 331 gen tok/s**; plus `_serving_lab3/results-2026-08-26.json` `remeasure_conc28`: 297 tok/s synthetic duck load on the 27B.

**Phase 0 — serving gate (one GPU session, ~2-4h, no slot).**
Clone the `submission/_serving_lab2` kernel pattern → `arc3-serving-lab-flashnext`:
- `machine_shape: NvidiaRtxPro6000`, internet off, `dataset_sources`: `sonphamorg/arc3-flashnext-serving-part-a-v1`, `-part-b-v1`, `-part-c-v1` (runtime tarball + source bundle inside part A; total attach ~136GB — sonpham's README explicitly says "Attach Parts A, B, and C to the notebook", so the mount budget works).
- Follow their setup wrapper: symlink view over the pre-expanded shards, untar `flashnext-gcp-container-site-packages.tar.zst`, launch their pinned runtime exactly (do not substitute our wheels — our 0.27.1 boot failed twice; theirs is the only Kaggle-proven ≥0.29-class runtime). Env: `VLLM_PLE_CPU_OFFLOAD=1` (host RAM 177GiB ≥ 51GB requirement); parsers `qwen3_xml` / `qwen3`; `chat_template_kwargs: {"thinking": true}`; if the tarball turns out to be SGLang, use the card's launch line adapted to `--tp 1`.
- Record: boot success + KV-cache-size line from the engine log (the single-GPU 88.8GiB profile is the one unreproduced claim); duck-shaped synthetic load at conc 28 (same harness prompts as `_serving_lab3`) → aggregate and per-stream gen tok/s, prefill tok/s, MTP on vs off (standing law: MTP ⇒ no prefix cache; stock prefix-hit is only 0.29, so the loss is bounded — measure both); tool-call parse rate over ≥100 duck turns; the standing grid instruments (`probe_boundary` / `probe_plans`) as a fixed quality battery.
- **Gate (pre-registered):** proceed iff boots on TP1 AND aggregate duck-shaped throughput ≥ 1.5x the 27B's 297 tok/s AND tool-call parse ≥ 95% AND grid battery not directionally worse. Fail → fall back to candidate #4 on the same kernel.

**Phase 1 — 25-game eval-geometry smoke (~4.6h, no slot).**
Clone `arc3-tp-smoke` with the *stock* duck harness (no packs — one variable at a time), endpoint swapped to the Flash-Next server; only harness deltas: served-model name, tool parser, thinking flag, and re-check the vision path (duck attaches a 4x-upscaled PNG tuned to Qwen's 16x16 patches; Flash-Next has a different vision encoder — verify patch size before the run).
- **Read (pre-registered, CV-aware — 25-game levels/game noise ≈ ±0.3):** PASS = actions/game ≥ 1.5x stock (≥71) AND levels/game ≥ 0.85; STRONG PASS adds levels/game ≥ 1.15; FAIL = levels/game < 0.7 at any throughput (comprehension downgrade — the 6B-active risk realized; the two-level-wall law says extra actions without comprehension buy nothing).
- Actions/game and tok/s are near-deterministic; read levels for direction only.

**Phase 2 — one live slot** only on Phase-1 PASS, as the day's falsifiable experiment (hypothesis: throughput x parity-comprehension moves the 12-draw 1.41 mean; one slot resolves ~±0.6).

Total cost: ~9h of the 60h GPU quota, 0-1 submission slots, no 135GB upload (sonpham's snapshot is public).

---

## 3. Caveats / unverified list

1. Single-GPU (TP1) boot of the NVFP4 checkpoint on RTX Pro 6000: claimed by a community HF variant, **not reproduced** — Phase 0's whole purpose.
2. rfbr ⇒ Muse Glimmer: inference from dataset name + pins; their kernel is private.
3. "Abstraction Lab published Flash-Next datasets": **not found** via CLI search; only sonpham and xugeger confirmed.
4. Qwen Community License 1.0: from search snippets; read `Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE` before a scored run.
5. Kaggle discussions since 08-28: unreadable by tooling; nothing cited from them.
6. benchlm's 73 tok/s / 30s TTFT for Flash-Next are hosted-API figures, irrelevant to local NVFP4 serving.
7. sonpham's datasets say "pinned vLLM runtime" while the RadixArk card documents SGLang only — the engine inside their 6GB tarball is whichever they proved; reuse it verbatim, don't assume.

## 4. Sources

- LB / datasets: `kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 --show` (08-30 09:35 UTC); `kaggle datasets list/files/download` for the sonphamorg, romainfabre, xugeger refs above; downloaded metadata in this directory (`README.dataset.md`, `PRECONVERTED_MODEL_PROVENANCE.json`, `hf_quant_config.json`, `config.json`, `README.md`, `aime26_metrics.json`, `rfbr/requirements.lock`).
- Flash-Next: https://huggingface.co/Qwen/Qwen3.8-Flash-Next · https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next · https://x.com/vllm_project/status/2092600887873286157 · https://www.marktechpost.com/2026/08/26/alibabas-qwen-team-releases-qwen3-8-flash-next-a-125b-multimodal-moe-with-6b-active-parameters-previewing-the-qwen4-architecture/ · https://benchlm.ai/models/qwen3-8-flash-next · https://www.datacamp.com/blog/qwen3-8-flash-next
- NVFP4 checkpoints: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 · https://huggingface.co/garnermccloud/Qwen3.8-Flash-Next-NVFP4-SSD-Stream · https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4
- Muse Glimmer: https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model · https://huggingface.co/meta-models/Muse-Glimmer-30B · https://kingy.ai/blog/muse-glimmer-30b-benchmarks-hardware-run/ · https://www.orcarouter.ai/blog/qwen-3-8-27b-vs-muse-glimmer · https://www.marktechpost.com/2026/08/10/meta-ai-releases-muse-glimmer/
- Nemotron: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 · https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/modelcard
- Repo internals: `docs/research-2026-08-29/{R1-kaggle-ecosystem.md,R4-harness-throughput-audit.md,R5-pack3-vllm024-boot-recipe.md}`, `docs/HANDOFF-2026-08-29-packs-in-flight.md`, `submission/_serving_lab2/kernel-metadata.json` (+ boot-fail logs), env captures with `mem_total: 185473264 kB`.
