"""Builder tests for the patch-closure arm kernels (Task 2 of the 2026-08-04 plan).

Run:  .venv/bin/python -m pytest submission/_ab_patch_closure/test_build_patch_closure.py -q
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from build_patch_closure import ARM_SLUGS, build_arm, notebook_contract  # noqa: E402
from patch_closure_config import (  # noqa: E402
    BASE_ENV,
    CANDIDATE_ENV,
    GEOMETRY,
    HYPOTHESIS,
)

REPO = Path(__file__).resolve().parents[2]


def test_built_notebooks_share_base_hash_and_differ_only_in_arm_env(tmp_path):
    base = build_arm("base", output_root=tmp_path)
    candidate = build_arm("candidate", output_root=tmp_path)
    assert notebook_contract(base)["source_base_sha256"] == notebook_contract(candidate)["source_base_sha256"]
    assert notebook_contract(base)["patch_sha256"] == notebook_contract(candidate)["patch_sha256"]
    assert notebook_contract(base)["arm_env"] == BASE_ENV
    assert notebook_contract(candidate)["arm_env"] == CANDIDATE_ENV
    assert notebook_contract(base)["hypothesis"] == HYPOTHESIS
    assert notebook_contract(base)["geometry"] == GEOMETRY


def test_metadata_is_private_commit_kernel():
    for arm in ("base", "candidate"):
        build_arm(arm)  # regenerate into the repo dirs (idempotent, deterministic)
        meta = json.loads(
            Path(f"{REPO}/submission/_ab_patch_closure/{arm}/kernel-metadata.json").read_text())
        assert meta["is_private"] is True
        assert meta["enable_internet"] is False
        assert meta["machine_shape"] == "NvidiaRtxPro6000"
        assert meta["enable_gpu"] is True
        assert meta["id"] == f"ahmedmobasher86/{ARM_SLUGS[arm]}"
        assert meta["code_file"] == f"patch-closure-{arm}.ipynb"
        assert "arc-prize-2026-arc-agi-3" in meta["competition_sources"]


def test_arm_slugs_are_distinct_and_never_the_competition_kernel():
    assert ARM_SLUGS["base"] != ARM_SLUGS["candidate"]
    for slug in ARM_SLUGS.values():
        assert slug != "arc-agi-3-duck-patched"


def test_every_code_cell_compiles(tmp_path):
    for arm in ("base", "candidate"):
        nb_path = build_arm(arm, output_root=tmp_path)
        nb = json.loads(nb_path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            # IPython executes cells per-statement, so mid-cell future imports
            # from the inlined modules are runtime-legal (the COMPLETE ab-wmr
            # kernels share the byte pattern); validate with ast.parse and the
            # notebook-only top-level awaits stripped, falling back to the
            # interactive-await grammar (the ab/rig builder validation).
            src = "".join(cell["source"])
            for marker in ("await pc_main", "await bm.run_and_score", "await bm.run"):
                src = src.replace(marker, "_ = " + marker.split(" ", 1)[1])
            try:
                ast.parse(src)
            except SyntaxError:
                compile(src, f"{arm}-cell{i}", "exec",
                        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)


def test_arm_env_is_pinned_immediately_before_apply_all(tmp_path):
    for arm, env in (("base", BASE_ENV), ("candidate", CANDIDATE_ENV)):
        nb = json.loads(build_arm(arm, output_root=tmp_path).read_text())
        hook = next(
            "".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code" and "PC_ARM_ENV" in "".join(c["source"]))
        env_pos = hook.index("PC_ARM_ENV = {")
        apply_pos = hook.index("_pc_patch_results = apply_all()")
        assert env_pos < apply_pos, "arm env must be pinned before apply_all()"
        pin_block = hook[env_pos:apply_pos]
        for key, value in env.items():
            assert f'"{key}": "{value}"' in pin_block, (arm, key, value)
        assert f'PC_ARM = "{arm}"' in hook
