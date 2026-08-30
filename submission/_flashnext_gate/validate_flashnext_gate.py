#!/usr/bin/env python3
"""validate_flashnext_gate.py — offline structural + syntax validation of the
arc3-flashnext-gate notebook and its kernel metadata. Never touches Kaggle.

Checks:
  1. notebook shape (nbformat 4, phase order by section markers)
  2. per-cell syntax via compile(); top-level `await` is wrapped in an async
     function before compiling (Kaggle's IPython allows it, plain compile does not)
  3. serve-chain / safety invariants (their env + argv verbatim, no games,
     no online installs, no SGLang)
  4. kernel-metadata.json contract (id, GPU shape, four datasets, internet off)
  5. the on-disk notebook matches a fresh in-memory build (no stale .ipynb)
  6. optional: pyflakes over the concatenated code cells (undefined names)

Usage:  .venv/bin/python submission/_flashnext_gate/validate_flashnext_gate.py
"""
import hashlib
import json
import runpy
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "arc3-flashnext-gate.ipynb"
META_PATH = HERE / "kernel-metadata.json"
BUILD_PATH = HERE / "build_flashnext_gate.py"

SECTION_ORDER = [
    "DRIVER + HOST + MOUNTS FIRST",
    "Phase 1 — ASSEMBLE",
    "SERVING LAB LIBRARY (flashnext-gate)",
    "Phase 2 — BOOT",
    "Phase 3 — SMOKE",
    "Phase 4 —",
    "Phase 5 —",
]
REQUIRED_TOKENS = [
    # their serving_env verbatim (the single-GPU mechanism)
    '"VLLM_PLE_CPU_OFFLOAD": "1"',
    '"VLLM_PLE_OFFLOAD_READY_TIMEOUT": "1800"',
    '"TORCH_CUDA_ARCH_LIST": "12.0f"',
    '"VLLM_ENABLE_CUDA_COMPATIBILITY": "0"',
    '"PYTORCH_ALLOC_CONF": "expandable_segments:False"',
    '"HF_HUB_OFFLINE": "1"',
    # their launch_server argv verbatim
    '"--tensor-parallel-size", "1"',
    '"--distributed-executor-backend", "mp"',
    '"--gpu-memory-utilization", "0.96"',
    '"--max-model-len", "32768"',
    '"--max-num-seqs", "22"',
    '"--max-num-batched-tokens", "6144"',
    '"--kv-cache-dtype", "auto"',
    '"--enable-prefix-caching"',
    '"--no-enable-flashinfer-autotune"',
    '"--tool-call-parser", "qwen3_xml"',
    '"--reasoning-parser", "qwen3"',
    # pins
    "c06a78d59a74ac278dc2278d26dde6c70c48a4e28bb91fd4fbbefff4484e10f3",
    "7c5468b370f7c47eda07281e3437fafc568f95d10420051e3aa522709f9342c5",
    "0.1.dev20073+g8e685d198 2.13.0+cu130 5.15.1 13.0",
    "RadixArk/Qwen3.8-Flash-Next-NVFP4",
    "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
    # assembly + provenance
    "serving-part-000",
    "serving-part-001",
    "serving-part-002",
    "source-bundle/zstd",
    "FLASHNEXT_GCP_MODEL_INFO.json",
    "ple-bf16-conversion.json",
    "model-plefp8-",
    "model.safetensors.index.json",
    # measurement
    "flashnext_gate_results.json",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:generation_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "reset_prefix_cache",
    'regime="stable"',
    "churn_conc28",
    "run_tool_battery",
    "image_smoke",
    "GATE_CONC28_TOK_S",
    "BASELINE_27B",
    "start_new_session=True",
    "WRONG-GPU",
    "driver_version",
]
BANNED_TOKENS = [
    "submission.parquet",
    "GameAgent",
    "scorecard",
    "arc_agi_3_wheels",
    "pip install",
    "sglang",
    "@@",
]
EXPECTED_META = {
    "id": "ahmedmobasher86/arc3-flashnext-gate",
    "code_file": "arc3-flashnext-gate.ipynb",
    "enable_gpu": True,
    "enable_internet": False,
    "is_private": True,
    "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": [
        "sonphamorg/arc3-flashnext-serving-part-a-v1",
        "sonphamorg/arc3-flashnext-serving-part-b-v1",
        "sonphamorg/arc3-flashnext-serving-part-c-v1",
        "sonphamorg/arc3-flashnext-gcp-runtime-exact-v1",
    ],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": [],
}


def compile_cell(src: str, name: str) -> None:
    try:
        compile(src, name, "exec".join(("", "")))
        return
    except SyntaxError as exc:
        if "await" not in (exc.msg or "") and "await" not in src:
            raise
    wrapped = "async def __cell_wrapper__():\n" + textwrap.indent(src, "    ")
    compile(wrapped, name + "(wrapped-await)", "exec".join(("", "")))


def main() -> int:
    problems = []
    nb = json.loads(NB_PATH.read_text())
    assert nb.get("nbformat") == 4, "nbformat must be 4"
    cells = nb["cells"]
    assert cells[0]["cell_type"] == "markdown", "cell 0 must be the markdown header"
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(code_cells) == 7, f"expected 7 code cells, got {len(code_cells)}"

    # 2. per-cell syntax
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            compile_cell(src, f"cell{i}")
        except SyntaxError as exc:
            problems.append(f"cell {i} syntax: {exc}")

    joined = "\n".join("".join(c["source"]) for c in cells)
    code_joined = "\n".join("".join(c["source"]) for c in code_cells)

    # 1. section order (code cells only)
    pos = -1
    for marker in SECTION_ORDER:
        at = code_joined.find(marker)
        if at < 0:
            problems.append(f"section marker missing: {marker!r}")
            continue
        if at < pos:
            problems.append(f"section out of order: {marker!r}")
        pos = at

    # 3. invariants
    for tok in REQUIRED_TOKENS:
        if tok not in joined:
            problems.append(f"required token missing: {tok!r}")
    for tok in BANNED_TOKENS:
        if tok in joined:
            problems.append(f"banned token present: {tok!r}")
    # every server/load call must be failure-wrapped: no bare calls at cell top level
    for cell in code_cells:
        src = "".join(cell["source"])
        for line in src.splitlines():
            if line.startswith("start_server(") or line.startswith("run_load_phase("):
                problems.append(f"unwrapped top-level call: {line[:60]}")

    # 4. metadata
    meta = json.loads(META_PATH.read_text())
    for key, want in EXPECTED_META.items():
        if meta.get(key) != want:
            problems.append(f"kernel-metadata {key}: {meta.get(key)!r} != {want!r}")

    # 5. freshness: rebuild in memory and compare code-cell sha
    built = runpy.run_path(str(BUILD_PATH))["build_notebook"]()
    fresh = "\n".join("".join(c["source"]) for c in built["cells"] if c["cell_type"] == "code")
    sha_disk = hashlib.sha256(code_joined.encode()).hexdigest()
    sha_fresh = hashlib.sha256(fresh.encode()).hexdigest()
    if sha_disk != sha_fresh:
        problems.append(f"stale notebook: disk {sha_disk[:12]} != fresh build "
                        f"{sha_fresh[:12]} — rerun the builder")

    # 6. pyflakes (optional)
    try:
        from pyflakes import api as pyflakes_api
        from pyflakes import reporter as pyflakes_reporter
        import io
        out, err = io.StringIO(), io.StringIO()
        rep = pyflakes_reporter.Reporter(out, err)
        pyflakes_api.check(code_joined, "flashnext-gate-cells", rep)
        flakes = [ln for ln in out.getvalue().splitlines()
                  if "undefined name" in ln and "__cell_wrapper__" not in ln]
        for ln in flakes:
            problems.append("pyflakes: " + ln)
        print(f"pyflakes: {len(out.getvalue().splitlines())} messages, "
              f"{len(flakes)} undefined-name")
    except ImportError:
        print("pyflakes not installed — undefined-name check skipped")

    print(f"cells: {len(cells)} ({len(code_cells)} code) | code sha256 {sha_disk}")
    if problems:
        print("FAIL")
        for p in problems:
            print(" -", p)
        return 1
    print("OK — arc3-flashnext-gate notebook + metadata validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
