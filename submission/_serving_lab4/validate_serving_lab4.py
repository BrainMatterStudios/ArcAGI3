#!/usr/bin/env python3
"""validate_serving_lab4.py — offline structural + syntax validation of the
arc3-serving-lab4 notebook and its kernel metadata. Never touches Kaggle.

Checks:
  1. notebook shape (nbformat 4, cell order by section markers)
  2. per-cell syntax via compile(); top-level `await` is wrapped in an async
     function before compiling (Kaggle's IPython allows it, plain compile does not)
  3. recipe / safety invariants (R5 §6.2 flags + env, no fp8 KV, no games)
  4. kernel-metadata.json contract (id, GPU shape, sources, internet off)
  5. the on-disk notebook matches a fresh in-memory build (no stale .ipynb)
  6. optional: pyflakes over the concatenated code cells (undefined names)

Usage:  .venv/bin/python submission/_serving_lab4/validate_serving_lab4.py
"""
import hashlib
import json
import runpy
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "arc3-serving-lab4.ipynb"
META_PATH = HERE / "kernel-metadata.json"
BUILD_PATH = HERE / "build_serving_lab4.py"

SECTION_ORDER = [
    "DRIVER + GPU FIRST",
    "INPUTS —",
    "INSTALL —",
    "SERVING LAB LIBRARY (lab4)",
    "Q1 — BOOT",
    "weight attestation",
    "Q2 — parser",
    "Q3 — MTP",
    "FINAL —",
]
REQUIRED_TOKENS = [
    '"VLLM_USE_FLASHINFER_SAMPLER": "0"',
    '"VLLM_USE_DEEP_GEMM": "0"',
    '"--mamba-cache-mode", "align"',
    '"--enable-prefix-caching"',
    '"--no-enable-prefix-caching"',
    '"method": "mtp"',
    '"num_speculative_tokens": 2',
    'SITE_PACKAGES / "nvidia" / "cu13"',
    "requirements-runtime.txt",
    "WHEELHOUSE_MANIFEST.json",
    "wheel_count",
    "mtp.safetensors",
    "--only-binary",
    "--no-cache-dir",
    "--ignore-installed",
    "start_new_session=True",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:generation_tokens",
    "reset_prefix_cache",
    "TORCH_CUDA_ARCH_LIST",
    "driver_version",
    "Initializing a V1 LLM engine",
    "speculative_config=",
    "flashinfer/jit",
    "/kaggle/input/arc3-qwen36-runtime-wheels",
    "/kaggle/input/datasets/jcole75/arc3-qwen36-runtime-wheels",
    "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
    "serving_lab4_results.json",
    'regime="churn"',
    'regime="stable"',
    "battery_mismatches",
    "MTP_LADDER",
    "WRONG-GPU",
]
BANNED_TOKENS = [
    '"--kv-cache-dtype"',
    "submission.parquet",
    "GameAgent",
    "scorecard",
    "arc_agi_3_wheels",
    "@@",
]
EXPECTED_META = {
    "id": "ahmedmobasher86/arc3-serving-lab4",
    "code_file": "arc3-serving-lab4.ipynb",
    "enable_gpu": True,
    "enable_internet": False,
    "is_private": True,
    "machine_shape": "NvidiaRtxPro6000",
    "dataset_sources": ["jcole75/arc3-qwen36-runtime-wheels"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
}


def compile_cell(src: str, name: str) -> None:
    try:
        compile(src, name, "exec")
        return
    except SyntaxError as exc:
        if "await" not in (exc.msg or "") and "await" not in src:
            raise
    wrapped = "async def __cell_wrapper__():\n" + textwrap.indent(src, "    ")
    compile(wrapped, name + "(wrapped-await)", "exec")


def main() -> int:
    problems = []
    nb = json.loads(NB_PATH.read_text())
    assert nb.get("nbformat") == 4, "nbformat must be 4"
    cells = nb["cells"]
    assert cells[0]["cell_type"] == "markdown", "cell 0 must be the markdown header"
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(code_cells) == 9, f"expected 9 code cells, got {len(code_cells)}"

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

    # 1. section order (code cells only — the markdown header mentions the phases too)
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
    if "reasoning_effort" in code_joined.replace('reasoning_effort "xhigh" (policy', ""):
        problems.append("reasoning_effort must not be set (pack3 xhigh policy is NOT copied)")
    # every load phase / battery / boot must be wrapped: no bare start_server at cell top level
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
        problems.append(f"stale notebook: disk {sha_disk[:12]} != fresh build {sha_fresh[:12]} — rerun the builder")

    # 6. pyflakes (optional)
    try:
        from pyflakes import api as pyflakes_api
        from pyflakes import reporter as pyflakes_reporter
        import io
        out, err = io.StringIO(), io.StringIO()
        rep = pyflakes_reporter.Reporter(out, err)
        pyflakes_api.check(code_joined, "lab4-cells", rep)
        flakes = [ln for ln in out.getvalue().splitlines()
                  if "undefined name" in ln and "__cell_wrapper__" not in ln]
        for ln in flakes:
            problems.append("pyflakes: " + ln)
        print(f"pyflakes: {len(out.getvalue().splitlines())} messages, {len(flakes)} undefined-name")
    except ImportError:
        print("pyflakes not installed — undefined-name check skipped")

    print(f"cells: {len(cells)} ({len(code_cells)} code) | code sha256 {sha_disk}")
    if problems:
        print("FAIL")
        for p in problems:
            print(" -", p)
        return 1
    print("OK — arc3-serving-lab4 notebook + metadata validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
