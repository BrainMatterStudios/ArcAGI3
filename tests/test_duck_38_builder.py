from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO / "submission/_duck_38/build_duck_38.py"
BASE = REPO / "submission/_duck_base/duck-base.ipynb"
CANDIDATE = REPO / "submission/_duck_38/duck-38.ipynb"
METADATA = REPO / "submission/_duck_38/kernel-metadata.json"
PUBLIC_REF = "mustangliu/qwen38-27b-fp8-hf-snapshot"
FALLBACK_REF = "ahmedmobasher86/qwen3-8-27b-fp8-hf-snapshot"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_duck_38", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_builder_names_primary_and_fallback_dataset_ownership() -> None:
    builder = load_builder()

    assert builder.PRIMARY_SNAPSHOT_REF == PUBLIC_REF
    assert builder.FALLBACK_SNAPSHOT_REF == FALLBACK_REF
    assert builder.NEW_SNAPSHOT_REF == builder.PRIMARY_SNAPSHOT_REF
    assert PUBLIC_REF in builder.__doc__


def test_built_notebook_and_metadata_bind_the_same_q38_identity() -> None:
    notebook = json.loads(CANDIDATE.read_text())
    metadata = json.loads(METADATA.read_text())
    source = code_source(notebook)

    assert PUBLIC_REF in metadata["dataset_sources"]
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaRtxPro6000"
    assert PUBLIC_REF in source
    assert "Qwen/Qwen3.8-27B-FP8" in source
    assert "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot" not in source
    assert "_38_missed" in source
    assert "_d38_wheel_dirs" in source
    assert "_d38_env_candidates" in source


def test_builder_documents_the_required_rtx_push_flag() -> None:
    builder = load_builder()

    assert builder.MACHINE_SHAPE == "NvidiaRtxPro6000"
    assert "--accelerator NvidiaRtxPro6000" in builder.__doc__


def test_builder_changes_only_the_four_declared_cells() -> None:
    base = json.loads(BASE.read_text())
    candidate = json.loads(CANDIDATE.read_text())

    assert len(candidate["cells"]) == len(base["cells"])
    changed = {
        index
        for index, (left, right) in enumerate(zip(base["cells"], candidate["cells"]))
        if "".join(left.get("source", [])) != "".join(right.get("source", []))
    }
    assert changed == {4, 6, 8, 14}
    assert '_os.environ["TAAF_' not in code_source(candidate)
    assert "def apply_all" not in code_source(candidate)
    assert "patch_struct_channel" not in code_source(candidate)


def test_builder_is_deterministic_and_clears_execution_state() -> None:
    before = CANDIDATE.read_bytes()
    metadata_before = METADATA.read_bytes()

    proc = subprocess.run(
        [sys.executable, str(BUILDER_PATH)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )

    after = CANDIDATE.read_bytes()
    assert after == before
    assert METADATA.read_bytes() == metadata_before
    assert "canonical hash: 417c1eedb7e3cfae" in proc.stdout
    assert hashlib.sha256(after).hexdigest() == "308d3a42b45ff144b6c25114a31a98180193ac3413317ecc47483174f2e386e1"
    notebook = json.loads(after)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
