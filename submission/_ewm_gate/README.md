# EWM local-brain viability gate (Qwen3.6-27B)

**What it answers:** Everything proven on-host used *Claude* as the coding brain (mean ~25/100 across 5 hard dev games, incl. tu93 80, sb26 27.8, cd82 14.3). At eval there is no Claude — a **local model must drive the loop**. This notebook runs the *identical* verification-by-execution agent loop (`ewm_agent.py`) driven by a local **Qwen3.6-27B-FP8** on 3 blind dev games and reports solve rate + RHAE + cost.

**This is a measurement gate, not a submission.** It does not touch the competition gateway or submit a score.

## GPU: MUST be a modern accelerator (NOT the default P100)
The Qwen FP8 model needs compute capability >=7.0 AND ~30GB. The default Kaggle GPU is a **P100 (cap 6.0,
16GB)** which FAILS with `compressed-tensors is not supported ... Minimum capability: 70. Current: 60`.
`kernel-metadata.json` now pins `machine_shape: nvidiaL4x4` (Ada L4x4, cap 8.9, FP8-native, 96GB) and the
serve cmd uses `--tensor-parallel-size 4`. Do not remove these.
Note: interactive Kaggle GPU is capped at **30h/week** — the gate run + a failed run can exhaust it; check
your quota/reset before pushing. (The competition RTX Pro 6000 is provisioned at official scoring, separate.)

## Run it
```bash
# from the repo root; kernel-metadata.json pins nvidiaL4x4; --accelerator not needed (baked into machine_shape)
kaggle kernels push -p submission/_ewm_gate
# watch it (GPU, internet-off). Serving FP8 weights takes several minutes.
kaggle kernels status ahmedmobasher86/arc-agi-3-ewm-gate
kaggle kernels output ahmedmobasher86/arc-agi-3-ewm-gate -p /tmp/ewm_gate_out
```
Attached automatically (already in metadata): `driessmit1/arc3-vllm-h100-wheelhouse-v3`, `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`, competition `arc-prize-2026-arc-agi-3`. Needs GPU on, internet OFF.

## Read the output — the final block
```
================ EWM LOCAL-BRAIN GATE SUMMARY (Qwen3.6-27B) ================
game   levels   RHAE/100  out_tok  sec
tu93   ?/9      ?         ?        ?
sb26   ?/8      ?         ?        ?
cd82   ?/6      ?         ?        ?
MEAN RHAE over 3 games: ?/100
```
**Decision:**
- **Qwen mean ≳ 10/100** → local-brain deployment is VIABLE. Next: build the selective-EWM eval submission (time-box per game, fail-fast, duck-baseline the rest) and submit a real EWM score.
- **Qwen mean ≈ 0** → 27B is too weak to drive the loop. Next: try `gpt-oss-120b` (stronger coder; needs the harmony-offline-vocab fix), or a leaner prompt / more turns, before committing to a submission.

Also note **tok/s** (printed after serve) and **out_tok/game** — these size the 9h/110-game budget for the selective scheduler.

## If the vLLM serve fails
The notebook prints the tail of `/kaggle/working/vllm.log`. Most likely causes + fixes:
- FP8 needs a flag: try adding `--quantization fp8` or removing `--enforce-eager` (edit the serve cmd in the last cell / `build_ewm_gate.py`).
- 32K context OOM on the KV cache: lower `--max-model-len` to 16384 or `--gpu-memory-utilization` to 0.90.
- Qwen served-name/rope: the duck stack serves this exact snapshot, so cross-check its serve args if needed.

## Rebuild after edits
```bash
python submission/_ewm_gate/build_ewm_gate.py   # regenerates ewm-gate.ipynb from the scaffold
```
The notebook is self-contained (base64-embeds the scaffold + `ewm_agent.py` + 3 dev games); only the model + wheels come from datasets.
