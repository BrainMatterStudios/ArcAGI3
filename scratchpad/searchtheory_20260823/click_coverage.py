"""Click-abstraction coverage: replay each human winning trace; at every
ACTION6, segment the current masked frame and measure (a) whether the human
click lands ON a non-background component, (b) distance to the nearest compact
component centroid, (c) candidate-set sizes. Scratch only."""
from __future__ import annotations

import glob
import json
import logging
import os
import sys

import numpy as np

logging.disable(logging.CRITICAL)
ROOT = "/Users/ahmed/Documents/ArcAGI3"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scratchpad/ideas"))

from arc_agi import Arcade, OperationMode          # noqa: E402
from arcengine import GameAction                   # noqa: E402
from hud_mask import mask_frame                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import components, settled              # noqa: E402

FIXES = sorted(glob.glob(os.path.join(
    ROOT, "scratchpad/testing_20260822/human_fixtures/*.json")))
GAMES = {}
for stem in os.listdir(os.path.join(ROOT, "environment_files")):
    ver = [v for v in os.listdir(os.path.join(ROOT, "environment_files", stem))
           if not v.startswith(("_", "."))]
    GAMES[stem] = f"{stem}-{ver[0]}"

only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
out = {}
for f in FIXES:
    if "_verify" in f:
        continue
    d = json.load(open(f))
    stem = d["game"]
    if only and stem not in only:
        continue
    n6 = sum(1 for a in d["actions"] if a["name"] == "ACTION6")
    if n6 == 0:
        continue
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(ROOT, "environment_files"))
    env = client.make(GAMES[stem])
    obs = env.reset()
    on_comp = 0
    near_centroid = 0
    within5 = 0
    cand_sizes = []
    total = 0
    for a in d["actions"]:
        if a["name"] == "ACTION6":
            g = mask_frame(settled(obs), stem)
            vals, counts = np.unique(g, return_counts=True)
            bg = int(vals[np.argmax(counts)])
            comps = [c for c in components(g, bg) if 2 <= c["size"] <= 200]
            x, y = int(a["x"]), int(a["y"])
            total += 1
            if 0 <= y < 64 and 0 <= x < 64 and g[y, x] != bg:
                on_comp += 1
            if comps:
                dmin = min(abs(c["cx"] - x) + abs(c["cy"] - y) for c in comps)
                if dmin <= 2:
                    near_centroid += 1
                if dmin <= 5:
                    within5 += 1
                cand_sizes.append(len(comps))
        # step
        if a["name"] == "ACTION6":
            obs = env.step(GameAction.ACTION6, data={"x": a["x"], "y": a["y"]})
        else:
            obs = env.step(GameAction.from_name(a["name"]))
    out[stem] = dict(
        clicks=total, on_component_frac=round(on_comp / total, 2),
        near_centroid2_frac=round(near_centroid / total, 2),
        near_centroid5_frac=round(within5 / total, 2),
        cand_median=int(np.median(cand_sizes)) if cand_sizes else None,
        cand_p90=int(np.percentile(cand_sizes, 90)) if cand_sizes else None)
    print(stem, out[stem], flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "click_coverage.json"), "w") as fh:
    json.dump(out, fh, indent=1)
