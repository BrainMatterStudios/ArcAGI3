# QUANT SWAP PROBE — upload plan (SCOUT VERDICT: no Kaggle dataset exists)

Date: 2026-08-13. Status: **BLOCKED on a ~31 GB dataset upload** — per the
build instruction, this plan is written INSTEAD of attempting the upload.
Nothing below has been executed past the scout.

## The hypothesis

sonpham's public 5-way quant A/B found the RedHatAI FP8 quant of
Qwen3.6-27B best among the quants he tested. Our entire pipeline serves the
**vrfai** FP8 quant (`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`,
29,052,908,203 bytes — every scored submission and every rig arm; see
`scratchpad/taaf_scored_ref/setup_commands.json` cmd 0:
`MODEL_OWNER = 'driessmit1'`, `MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'`,
`SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'`). RedHatAI-vs-vrfai is
UNMEASURED for us. CAVEAT: sonpham's result is external and unreplicated
here; treat it as a hypothesis source, not evidence.

## Scout evidence (2026-08-13)

**Kaggle: NO existing dataset.** `python3 -m kaggle datasets list -s <terms>`
run with 10 term variants ("RedHatAI Qwen3.6", "qwen3.6 27b fp8",
"redhat qwen3", "qwen3-6-27b", "qwen3.6-27b", "RedHatAI", "redhatai",
"qwen fp8 27b redhat", "Qwen3.6-27B-FP8", "qwen3 6 27b fp8 redhat") plus
`--user RedHatAI`. The only FP8 27B snapshot on Kaggle is our own vrfai one
(`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`). No RedHatAI-derived
dataset exists under any spelling tried.

**Hugging Face: the exact repo id is `RedHatAI/Qwen3.6-27B-FP8`.**
Verified via `HfApi().model_info(..., files_metadata=True)`:

- public, not gated, last modified 2026-06-16 12:38:59 UTC
- **total 30,890,041,440 bytes (30.89 GB) across 80 files**
- weights layout: `outside.safetensors` (6.0 GB) + `layers-{0..63}.safetensors`
  (0.372-0.384 GB each) + `mtp.safetensors` (0.477 GB), with
  `model.safetensors.index.json` present (vLLM loads via the index, so the
  non-standard shard naming is fine)
- configs present: `config.json`, `generation_config.json`,
  `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`,
  `preprocessor_config.json`, `video_preprocessor_config.json`
- `RedHatAI/Qwen3.6-27B-FP8-dynamic` / `-Dynamic` do **NOT** exist
  (RepositoryNotFoundError both casings) — the 35B has a `-dynamic` variant,
  the 27B does not. So "RedHatAI FP8 27B" can only mean this repo.

## Step 1 — download (~31 GB disk, hours on home bandwidth)

```bash
mkdir -p ~/models/redhatai-qwen3.6-27b-fp8
.venv/bin/python -m huggingface_hub.commands.huggingface_cli download \
    RedHatAI/Qwen3.6-27B-FP8 \
    --local-dir ~/models/redhatai-qwen3.6-27b-fp8
# (or: hf download RedHatAI/Qwen3.6-27B-FP8 --local-dir ... with a newer hub CLI)
# sanity: 80 files, 30,890,041,440 bytes total
find ~/models/redhatai-qwen3.6-27b-fp8 -type f | wc -l
du -sb ~/models/redhatai-qwen3.6-27b-fp8
```

Optionally drop `.gitattributes`/`README.md`; keep EVERYTHING else —
especially `model.safetensors.index.json`, `mtp.safetensors`, and all
config/tokenizer files (the serve's smoke test will fail fast if the layout
is broken, `setup_commands.json` cmd 0 `run_vllm_api_smoke_test`).

## Step 2 — create the Kaggle dataset (~31 GB upload)

Mirror the vrfai snapshot convention (dataset root IS the HF snapshot root —
`resolve_kaggle_dataset_path` returns the dataset mount directory and vLLM
is pointed straight at it):

```bash
cd ~/models/redhatai-qwen3.6-27b-fp8
cat > dataset-metadata.json <<'EOF'
{
  "title": "RedHatAI Qwen3.6 27B FP8 Hugging Face Snapshot",
  "id": "ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
python3 -m kaggle datasets create -p . --dir-mode tar
# poll until processing completes (large datasets take a while server-side):
python3 -m kaggle datasets status ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot
```

Notes: `--dir-mode tar` avoids per-file upload flakiness at this size; if it
fails, retry with plain `python3 -m kaggle datasets create -p .` (per-file).
The vrfai precedent (29 GB) proves Kaggle accepts this size.

## Step 3 — the rig arm (build AFTER the dataset lands; not built now)

A `quant` arm identical to the shipped arm (`build_shipped_screen.py`
pattern, distinct slug e.g. `arc-agi-3-patch-closure-quant`) with exactly two
deltas, both at build time:

1. **kernel-metadata `dataset_sources`**: replace
   `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` with
   `ahmedmobasher86/redhatai-qwen3-6-27b-fp8-hf-snapshot`
   (`build_patch_closure.py:387-391` writes the list; the builder would take
   a dataset-swap parameter).
2. **serve override**: the model resolves inside `setup_commands.json` cmd 0
   (mounted read-only from the taaf-src bundle dataset), executed by
   duck-base cell 8's loop — which the rig already rewrites (`SERVE_GUARD`
   force at `build_patch_closure.py:311-315`). The quant builder must rewrite
   the command string in-memory before `subprocess.run`, replacing the three
   anchors:
   - `MODEL_OWNER = 'driessmit1'` -> `MODEL_OWNER = 'ahmedmobasher86'`
   - `MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'` ->
     `MODEL_SLUG = 'redhatai-qwen3-6-27b-fp8-hf-snapshot'`
   - `SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'` ->
     `SERVED_MODEL_NAME = 'RedHatAI/Qwen3.6-27B-FP8'`
   with a hard assert that all three replacements fired (anchor drift must
   abort the build, not silently serve vrfai). `SERVED_MODEL_NAME` flows to
   `LOCAL_ANALYZER_MODEL_ID`, which `pc_driver._pc_serving_probe`
   (`pc_driver.py:288-301`) then verifies against `/models` — so a
   wrong-model serve is caught before any game minute. Also update duck-base
   cell 6's baked `DATASET_SOURCES` list literal for the
   `TAAF_KAGGLE_INPUT_PATHS` mapping (belt and braces; `resolve_kaggle_dataset_path`
   falls back to `/kaggle/input/<slug>` anyway).

Comparator: the existing shipped kernel wave from the same session. Reading:
same `classify_shipped`-style contract with primary
`true_score_all_games` delta at the 0.2712 banked-spread threshold (one wave
per arm cannot resolve less — power honesty as in
`shipped_screen_config.SHIPPED_READING`).

## Risks / honest caveats

- **~31 GB is above the vrfai dataset's 29 GB** — same order, should pass,
  but Kaggle-side processing failures at this size are common; budget a
  retry.
- **`mtp.safetensors` (MTP module)**: the vrfai snapshot serves fine on the
  wheelhouse vLLM 0.19.0; whether that vLLM build loads/ignores RedHatAI's
  MTP weights is UNVERIFIED. The in-command smoke test fails fast if not.
- **GPU quota**: the arm costs a full rig wave (~2.5 h RTX Pro 6000) plus
  the shipped comparator wave.
- **sonpham provenance**: his 5-way A/B was on his own harness/games, not
  ours; RedHatAI-best may not transfer.
