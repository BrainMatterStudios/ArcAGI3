"""Assemble the Kaggle dataset payload (the arcagi3 package) under submission/dataset/.

Run:  python submission/build_dataset.py
Then: kaggle datasets create  -p submission/dataset   # first time
  or: kaggle datasets version -p submission/dataset -m "update"
"""

import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
SRC_PKG = ROOT / "src" / "arcagi3"
OUT = HERE / "dataset"
OWNER = "ahmedmobasher86"

# Fresh copy of the package under a wrapper dir `lib/`. We upload with `--dir-mode zip`,
# which strips exactly one top folder level on extraction — so `lib/arcagi3/...` extracts to
# `arcagi3/...` at the dataset root (preserving the package dir). Uploading `arcagi3/`
# directly would flatten the package to the root and break `import arcagi3`.
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "lib" / "arcagi3").mkdir(parents=True)
for p in SRC_PKG.rglob("*.py"):
    rel = p.relative_to(SRC_PKG)
    dst = OUT / "lib" / "arcagi3" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)

(OUT / "dataset-metadata.json").write_text(json.dumps({
    "title": "arcagi3-agent",
    "id": f"{OWNER}/arcagi3-agent",
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))

n = len(list((OUT / "arcagi3").rglob("*.py")))
print(f"wrote {OUT} with {n} python files + dataset-metadata.json")
