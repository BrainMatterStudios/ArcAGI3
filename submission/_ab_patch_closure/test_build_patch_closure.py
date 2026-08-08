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
        # fc12c29: candidate ERRORed on unpinned images — the pin must survive
        # every rebuild (the builder owns kernel-metadata.json).
        assert meta["docker_image"].startswith("gcr.io/kaggle-private-byod/python@sha256:")


def test_install_cell_probes_both_competition_mount_forms(tmp_path):
    """2026-08-08 root cause: the competition data mounts at either
    /kaggle/input/competitions/<slug> or /kaggle/input/<slug> per machine;
    duck-base hardcodes the first and discards pip stdout. The built kernels
    must probe BOTH at runtime and keep pip's failure visible."""
    for arm in ("base", "candidate"):
        nb = json.loads(build_arm(arm, output_root=tmp_path).read_text())
        install = [
            "".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code" and "arc_agi_3_wheels" in "".join(c["source"])
        ]
        assert len(install) == 1, f"{arm}: expected exactly one wheels install cell"
        src = install[0]
        assert "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels" in src
        assert '"/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"' in src
        assert "either mount form" in src           # loud failure names both paths
        assert "DEVNULL" not in src                 # pip output must be visible
        assert "capture_output=True" in src and "stderr" in src


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


def test_dry_run_emits_classifiable_pair(tmp_path):
    """GPU-free end-to-end: both arms through the REAL harness + mock brain,
    emitting the exact result schema the classifier freezes. Runs dry_run.py
    in a SUBPROCESS (the standalone form the runbook uses) so its kernel-like
    module surgery — synthetic duck_patches, scored-ref sys.path — cannot
    pollute this shared pytest process. Equal levels are expected (identical
    mock policy in both arms), so the verdict must be NO_GO — the dry run
    must never encode a forced GO."""
    import os
    import subprocess

    from patch_closure_config import classify_result

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "dry_run.py")],
        env={**os.environ, "PC_DRY_WORKDIR": str(tmp_path)},
        capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, f"dry run failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    assert "state=NO_GO" in proc.stdout, proc.stdout[-2000:]

    workroot = next(p for p in sorted(tmp_path.iterdir()) if p.is_dir())
    base = json.loads((workroot / "base" / "patch_closure_result.json").read_text())
    candidate = json.loads((workroot / "candidate" / "patch_closure_result.json").read_text())
    assert base["schema_version"] == candidate["schema_version"] == 1
    assert base["arm"] == "base" and candidate["arm"] == "candidate"
    assert base["stage"] == candidate["stage"] == "done"
    out = classify_result(base, candidate)
    assert out["state"] in {"GO", "NO_GO"}
    assert out["state"] == "NO_GO"


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
