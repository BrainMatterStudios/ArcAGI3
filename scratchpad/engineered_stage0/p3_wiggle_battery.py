"""Stage-0 Pillar 3 — wiggle/contingency probe battery on the current engine.

Implements the design-doc Layer-2 opening battery (each directional twice, then
clicks on k distinct component centroids, HUD rows excluded) and confirms it
produces the three per-pixel masks — SELF / REACTIVE / DEAD — plus a per-action
affordance summary, on >= 3 games spanning both archetypes:

  tu93  avatar-nav (avail 1-4)   -> expect a nonempty, compact SELF mask
  sb26  click puzzle             -> expect REACTIVE cells on clicked components
  lp85  click puzzle             -> expect REACTIVE cells on clicked components
  lf52  click game w/ HUD ticker -> battery must not be fooled by the HUD row

This is the mask-producing half; scratchpad/ideas/probe_core.py (T1 agency NMI,
T2 hidden state, T3 click AUC) is executed separately as p3b.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ideas"))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from hud_mask import hud_mask, settled_frame

GAMES = ["tu93", "sb26", "lp85", "lf52"]
K_CLICKS = 8
_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def components(frame, bg):
    """(click_y, click_x, color, size) per non-background component.

    The click target is the component pixel NEAREST its centroid — a raw
    centroid of a hollow/non-convex component can land off-component (this
    exact failure nulled all sb26 clicks on the first battery run).
    """
    out = []
    for col in np.unique(frame):
        if col == bg:
            continue
        lab, n = ndimage.label(frame == col, structure=_S4)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            cy, cx = ys.mean(), xs.mean()
            k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            out.append((int(ys[k]), int(xs[k]), int(col), len(ys)))
    return out


def run_battery(arcade, gid, stem):
    env = arcade.make(game_id=gid, scorecard_id=f"s0wig-{stem}")
    o = env.reset()
    board_mask = ~hud_mask(gid)  # True where board (non-HUD)
    prev = settled_frame(o).copy()
    avail = set(o.available_actions or [])
    h, w = prev.shape
    self_mask = np.zeros((h, w), dtype=bool)
    reactive_mask = np.zeros((h, w), dtype=bool)
    touched = np.zeros((h, w), dtype=bool)
    actions_spent = 0
    directional_diffs = 0
    per_action = {}

    # phase 1: each directional (ACTION1-4, if offered) twice
    for aid in (1, 2, 3, 4):
        if aid not in avail:
            continue
        diffs = 0
        for _ in range(2):
            if o.state in (GameState.WIN,):
                break
            if o.state == GameState.GAME_OVER:
                o = env.step(GameAction.RESET)
                prev = settled_frame(o).copy()
            o = env.step(GameAction.from_id(aid))
            actions_spent += 1
            cur = settled_frame(o)
            d = (cur != prev) & board_mask
            touched |= d
            if d.any():
                diffs += 1
                self_mask |= d
            prev = cur.copy()
        per_action[f"A{aid}"] = diffs
        directional_diffs += diffs

    # phase 2: clicks on k distinct component centroids (largest-first, distinct colors preferred)
    click_results = []
    if 6 in avail:
        bg = int(np.bincount(prev.ravel()).argmax())
        comps = sorted(components(np.where(board_mask, prev, bg), bg),
                       key=lambda c: -c[3])
        # No directionals -> the whole 12-16 action budget belongs to clicks, and
        # SAME-color duplicate blobs must each be probed (sb26: the top display
        # and the bottom answer buttons share colors; deduping by color missed
        # every live button on the first run).
        k_budget = K_CLICKS if any(f"A{a}" in per_action for a in (1, 2, 3, 4)) else 14
        # Pass 1: one blob per distinct color (largest first). Pass 2: fill the
        # remaining budget with further instances of already-seen colors, again
        # largest first — sb26's live answer buttons are the SECOND, smaller
        # instances of the display colors and a dedup-only policy misses them.
        seen_colors, targets, leftover = set(), [], []
        for cy, cx, col, size in comps:
            if col not in seen_colors:
                seen_colors.add(col)
                targets.append((cy, cx, col))
            else:
                leftover.append((cy, cx, col))
        targets = (targets + leftover)[:k_budget]
        rng = random.Random(0)
        for cy, cx, col in targets:
            if o.state == GameState.WIN:
                break
            if o.state == GameState.GAME_OVER:
                o = env.step(GameAction.RESET)
                prev = settled_frame(o).copy()
            o = env.step(GameAction.ACTION6, data={"x": cx, "y": cy})
            actions_spent += 1
            cur = settled_frame(o)
            d = (cur != prev) & board_mask
            touched |= d
            if d.any():
                reactive_mask |= d
            click_results.append({"y": cy, "x": cx, "color": col, "changed": bool(d.any()),
                                  "cells": int(d.sum())})
            prev = cur.copy()

    dead_mask = board_mask & ~touched
    # SELF compactness: largest connected blob share of the self mask
    self_blob = 0
    if self_mask.any():
        lab, n = ndimage.label(self_mask, structure=_S4)
        self_blob = int(max(np.sum(lab == i) for i in range(1, n + 1)))
    return {
        "actions_spent": actions_spent,
        "avail": sorted(a for a in avail if a),
        "per_directional_diffs": per_action,
        "self_px": int(self_mask.sum()), "self_largest_blob": self_blob,
        "reactive_px": int(reactive_mask.sum()),
        "dead_px": int(dead_mask.sum()),
        "clicks": click_results,
        "clicks_reactive": sum(1 for c in click_results if c["changed"]),
    }


def main() -> int:
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid_of = {e.game_id.split("-")[0][:4]: e.game_id for e in arcade.get_environments()}
    out = {}
    t0 = time.time()
    for stem in GAMES:
        r = run_battery(arcade, gid_of[stem], stem)
        out[stem] = r
        print(f"{stem}: {r['actions_spent']} actions  avail={r['avail']}  "
              f"SELF={r['self_px']}px(blob {r['self_largest_blob']})  "
              f"REACTIVE={r['reactive_px']}px  DEAD={r['dead_px']}px  "
              f"clicks {r['clicks_reactive']}/{len(r['clicks'])} reactive")

    ok = (out["tu93"]["self_px"] > 0                       # avatar game: body found
          and out["sb26"]["clicks_reactive"] > 0            # click game: reactive cells
          and out["lp85"]["clicks_reactive"] > 0
          and all(r["actions_spent"] <= 16 for r in out.values()))  # bounded battery
    print(f"\nP3a VERDICT: {'PASS' if ok else 'FAIL'}  elapsed {time.time()-t0:.0f}s")
    json.dump(out, open("scratchpad/engineered_stage0/p3_wiggle_battery.json", "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
