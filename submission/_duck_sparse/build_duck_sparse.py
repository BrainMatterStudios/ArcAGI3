#!/usr/bin/env python3
"""Build submission/_duck_sparse/duck-sparse.ipynb.

Takes the base duck-patched notebook (submission/_duck_patched/duck-patched.ipynb) and builds
the Track B Sparse Model + arc3kit.py REPL sandbox notebook for Kernel Version 9 submission.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_NOTEBOOK = REPO / "submission/_duck_patched/duck-patched.ipynb"
SPARSE_MODULE = Path(__file__).parent / "duck_sparse.py"
SPARSE_PATCHES = Path(__file__).parent / "duck_patches_sparse.py"
OUT_DIR = Path(__file__).parent
OUT = OUT_DIR / "duck-sparse.ipynb"


def main() -> None:
    if not BASE_NOTEBOOK.is_file():
        raise SystemExit(f"Base notebook not found: {BASE_NOTEBOOK}")

    nb = json.loads(BASE_NOTEBOOK.read_text())
    sparse_code = SPARSE_MODULE.read_text()
    sparse_patches_code = SPARSE_PATCHES.read_text()

    landed = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and "# In-memory harness patches" in src:
            # Check idempotency
            if "apply_sparse_sandbox_patches()" in src:
                landed = True
                break

            new_source = (
                "# ============================================================================\n"
                "# Track B Sparse Model & arc3kit.py REPL Sandbox Injection\n"
                "# ============================================================================\n"
                f"{sparse_patches_code}\n\n"
                f"{sparse_code}\n\n"
                'print(apply_sparse_sandbox_patches(), flush=True)\n'
                'print(apply_sparse_patches(), flush=True)\n\n'
                f"{src}"
            )
            cell["source"] = [line + "\n" for line in new_source.splitlines()]
            landed = True
            break

    assert landed, "CRITICAL ERROR: customization hook cell '# In-memory harness patches' not found in base notebook!"

    # Track B model swap: point the notebook's attached-model source at the
    # Qwen3.6-35B-A3B FP8 snapshot (cell 6 DATASET_SOURCES) in place of the 27B.
    OLD_MODEL_REF = "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot"
    NEW_MODEL_REF = "cmechevalier/face-of-agi-qwen36-35b-fp8-weights"
    swapped = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if OLD_MODEL_REF in src:
            new_src = src.replace(OLD_MODEL_REF, NEW_MODEL_REF)
            cell["source"] = [line + "\n" for line in new_src.splitlines()]
            swapped += 1
    assert swapped >= 1, "CRITICAL ERROR: 27B model ref not found in any notebook cell!"

    # Bundle swap: replace the third-party TAAF bundle (jeroencottaar, 27B
    # setup_commands) with our own 35B-bundle dataset that carries the edited
    # setup_commands.json (MODEL_OWNER=cmechevalier, 35B-A3B-FP8).
    OLD_BUNDLE_REF = "jeroencottaar/taaf-kaggle-source-share"
    NEW_BUNDLE_REF = "ahmedmobasher86/arcagi3-bundle-35b"
    swapped_bundle = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if OLD_BUNDLE_REF in src:
            new_src = src.replace(OLD_BUNDLE_REF, NEW_BUNDLE_REF)
            cell["source"] = [line + "\n" for line in new_src.splitlines()]
            swapped_bundle += 1
    assert swapped_bundle >= 1, "CRITICAL ERROR: third-party bundle ref not found in any notebook cell!"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_text = json.dumps(nb, indent=2)
    
    assert "apply_sparse_sandbox_patches()" in out_text, "CRITICAL ERROR: sparse sandbox patch did not land in output notebook!"
    assert NEW_MODEL_REF in out_text, "CRITICAL ERROR: 35B model ref did not land in output notebook!"
    assert OLD_MODEL_REF not in out_text, "CRITICAL ERROR: stale 27B model ref still present in output notebook!"
    assert NEW_BUNDLE_REF in out_text, "CRITICAL ERROR: 35B bundle ref did not land in output notebook!"
    assert OLD_BUNDLE_REF not in out_text, "CRITICAL ERROR: stale third-party bundle ref still present in output notebook!"
    assert "connected_components" in out_text, "CRITICAL ERROR: connected_components primitive not found in output notebook!"

    OUT.write_text(out_text)
    print(f"wrote {OUT}  ({len(nb['cells'])} cells, verified sparse patches landed & idempotent)")


if __name__ == "__main__":
    main()
