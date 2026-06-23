# Phase 3 — Perception/Coordinate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make engine #1's induced plans executable by snapping scene-graph target centroids to the agent's measured motion lattice, and surface collect items as targets — so the bake-off finally clears L1 on ls20 (paint) and collect.

**Architecture:** A pure `motion_lattice` helper derives the agent's pixel pitch from `_deltas` and snaps any pixel position to the nearest reachable lattice cell. `_build_plan` routes target centroids through `snap()` (so off-lattice targets become reachable) and falls back to small non-agent objects as targets when the scene graph finds none (collect). The planners are unchanged — only slot coordinates change.

**Tech Stack:** Python 3.12, numpy, pytest (`pythonpath=["src"]`), offline `arc_agi`/`arcengine`. No GPU/LLM.

**Reuse (current signatures, verified in `src/arcagi3/discovery_explorer.py`):**
- `_build_plan(self, grid)`: builds `start_pos = self._agent_pos(grid)`, `start_attrs`, sources `slot_positions` from `SG.extract(grid, self._bg)["target_candidates"]` (each `t["centroid"]` → rounded `(int,int)`), computes `paintable_cells`/`true_walls`, enumerates `reqs` (reach-only + attainable color/shape), and plans via `plan(model, start, slots)` (A1) or `plan_painted_set(...)` (A2). `self._deltas` is `{action_id: (dr,dc)}` in pixels (ls20 → ±5). `self._agent_color`, `self._bg` set during probing.
- `P.connected_components(grid, background)` → list of `Obj(color, cells, bbox, size, centroid)`.

**Guardrails:** pure helper, no game constants; snapping only changes slot coordinates (not the action vocabulary or the planners); collect fallback fires ONLY when scene-graph returns no candidates (ls20 unaffected).

---

## File Structure

- Create `src/arcagi3/motion_lattice.py` — `lattice_pitch(deltas)`, `snap(pos, origin, pitch, bounds)`.
- Create `tests/test_motion_lattice.py` — unit tests for both.
- Modify `src/arcagi3/discovery_explorer.py` — `_build_plan`: snap slot positions (Task 2); broadened target fallback (Task 3).
- Modify `tests/test_discovery_explorer.py` — snapping + broadened-extraction unit tests; update gated guards (Task 4).
- Modify `docs/experiment-overview.html` — Experiment 45 (Task 5).

---

## Task 1: motion_lattice helper

**Files:** Create `src/arcagi3/motion_lattice.py`; Test `tests/test_motion_lattice.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_motion_lattice.py
from arcagi3.motion_lattice import lattice_pitch, snap

def test_lattice_pitch_from_deltas():
    assert lattice_pitch({1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}) == (5, 5)

def test_lattice_pitch_zero_axis():
    # only vertical motion -> column pitch 0 (no snapping on that axis)
    assert lattice_pitch({1: (-3, 0), 2: (3, 0)}) == (3, 0)

def test_snap_to_nearest_lattice_cell():
    # origin (10,10), pitch (5,5): (12,18) -> row 10+round(0.4)*5=10, col 10+round(1.6)*5=20
    assert snap((12, 18), (10, 10), (5, 5), (64, 64)) == (10, 20)

def test_snap_identity_on_lattice():
    assert snap((20, 25), (10, 10), (5, 5), (64, 64)) == (20, 25)

def test_snap_clamps_to_bounds():
    # row would snap to 100 -> clamp to 19; col (3-10)/5=-1.4 -> round -1 -> 5
    assert snap((100, 3), (10, 10), (5, 5), (20, 20)) == (19, 5)

def test_snap_zero_pitch_keeps_origin_on_that_axis():
    # col pitch 0 -> column stays at origin's column (10)
    assert snap((12, 18), (10, 10), (5, 0), (64, 64)) == (10, 10)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_motion_lattice.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'arcagi3.motion_lattice'`)

- [ ] **Step 3: Implement**

```python
# src/arcagi3/motion_lattice.py
"""The agent's motion lattice: the discrete grid its moves actually reach.

Frames are 64x64 PIXELS but the agent moves in fixed pixel jumps (e.g. ls20 = 5 px/step), so the
cells it can occupy form a lattice anchored at its start. Planning targets must lie on this lattice
to be reachable; `snap` projects an arbitrary pixel position to the nearest lattice cell.
"""
from __future__ import annotations


def lattice_pitch(deltas: dict) -> tuple[int, int]:
    """Per-axis pixel pitch = the smallest nonzero |displacement| on each axis. 0 = no motion on
    that axis (caller must treat pitch 0 as 'do not snap that axis')."""
    prs = [abs(dr) for dr, dc in deltas.values() if dr != 0]
    pcs = [abs(dc) for dr, dc in deltas.values() if dc != 0]
    return (min(prs) if prs else 0, min(pcs) if pcs else 0)


def snap(pos, origin, pitch, bounds) -> tuple[int, int]:
    """Project `pos` (r,c) to the nearest lattice cell anchored at `origin` with `pitch`, clamped
    to `bounds` (H,W). A 0 pitch on an axis keeps that axis at the origin's coordinate."""
    r, c = pos
    orr, oc = origin
    pr, pc = pitch
    H, W = bounds
    nr = orr if pr == 0 else orr + round((r - orr) / pr) * pr
    nc = oc if pc == 0 else oc + round((c - oc) / pc) * pc
    nr = max(0, min(H - 1, int(nr)))
    nc = max(0, min(W - 1, int(nc)))
    return (nr, nc)
```

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_motion_lattice.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/motion_lattice.py tests/test_motion_lattice.py
git commit -q -m "feat(discovery): motion-lattice helper (pitch + snap)"
```

---

## Task 2: Snap target slots to the lattice in `_build_plan`

**Files:** Modify `src/arcagi3/discovery_explorer.py`; Test `tests/test_discovery_explorer.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_discovery_explorer.py
def test_build_plan_snaps_offlattice_target_to_reachable_cell():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal")
    eng.reset_all(); eng._bg = 0; eng._agent_color = 9
    eng._deltas = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}  # 5px lattice
    eng._tiles = {}; eng._cycles = []; eng._terminal = None
    # agent single cell at (10,10); an OFF-LATTICE target centroid at (12,18)
    grid = np.zeros((64, 64), dtype=np.int8); grid[10, 10] = 9
    import arcagi3.discovery_explorer as DE
    orig = DE.SG.extract
    DE.SG.extract = lambda g, bg: {"target_candidates": [{"centroid": (12.0, 18.0)}]}
    try:
        eng._build_plan(grid)
    finally:
        DE.SG.extract = orig
    # (12,18) snaps to (10,20) on the 5px lattice from origin (10,10); reachable by 2x action-4.
    assert eng._plan == [4, 4]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k snaps_offlattice -v`
Expected: FAIL (current `_build_plan` uses the raw off-lattice centroid `(12,18)`, which is unreachable on the 5px lattice → `plan()` returns None → `_plan` stays `[]`).

- [ ] **Step 3: Implement**

In `src/arcagi3/discovery_explorer.py`, add the import near the other `arcagi3` imports at the top:

```python
from arcagi3.motion_lattice import lattice_pitch, snap
```

In `_build_plan`, find the block that builds `slot_positions` from candidates, immediately followed by `if not slot_positions: return`:

```python
        slot_positions = []
        for t in cands:
            cy, cx = t["centroid"]
            slot_positions.append((int(round(cy)), int(round(cx))))
        if not slot_positions:
            return
```

Replace it with (snap each slot to the agent's motion lattice, then dedup preserving order):

```python
        slot_positions = []
        for t in cands:
            cy, cx = t["centroid"]
            slot_positions.append((int(round(cy)), int(round(cx))))
        if not slot_positions:
            return
        # Snap targets to the agent's motion lattice (pitch from _deltas, origin = start_pos) so
        # they are reachable by the planner's pitch-sized steps (Phase-3 fix for Exp-44).
        pitch = lattice_pitch(self._deltas)
        slot_positions = list(dict.fromkeys(
            snap(p, start_pos, pitch, (H, W)) for p in slot_positions))
```

(`H, W = grid.shape` and `start_pos` are already computed earlier in `_build_plan`.)

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k snaps_offlattice -v`
Expected: PASS (`_plan == [4, 4]`). Then run the whole file:
`PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -v` — all pass (existing tests still green; the existing `test_a1_treats_paint_cells_as_passable` uses an already-on-lattice target so snapping is identity there).

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/discovery_explorer.py tests/test_discovery_explorer.py
git commit -q -m "fix(discovery): snap target slots to the agent motion lattice (Exp-44 fix)"
```

---

## Task 3: Broadened target extraction (collect fallback)

**Files:** Modify `src/arcagi3/discovery_explorer.py`; Test `tests/test_discovery_explorer.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_discovery_explorer.py
def test_build_plan_falls_back_to_small_objects_when_no_scene_targets():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal")
    eng.reset_all(); eng._bg = 0; eng._agent_color = 14
    eng._deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # 1px lattice (collect-like)
    eng._tiles = {}; eng._cycles = []; eng._terminal = None
    # agent (color 14) at (0,0); one item (color 6) at (0,2). No "framed" targets.
    grid = np.zeros((6, 6), dtype=np.int8); grid[0, 0] = 14; grid[0, 2] = 6
    import arcagi3.discovery_explorer as DE
    orig = DE.SG.extract
    DE.SG.extract = lambda g, bg: {"target_candidates": []}   # scene graph finds nothing
    try:
        eng._build_plan(grid)
    finally:
        DE.SG.extract = orig
    # fallback surfaces the item at (0,2); reach-all plans to it: right, right.
    assert eng._plan == [4, 4]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k falls_back -v`
Expected: FAIL (current `_build_plan` returns early at `if not slot_positions: return` because scene-graph gave no candidates → `_plan` stays `[]`).

- [ ] **Step 3: Implement**

In `_build_plan`, locate the slot-sourcing block (after Task 2 it looks like):

```python
        slot_positions = []
        for t in cands:
            cy, cx = t["centroid"]
            slot_positions.append((int(round(cy)), int(round(cx))))
        if not slot_positions:
            return
        # Snap targets ... (Task 2)
        pitch = lattice_pitch(self._deltas)
        slot_positions = list(dict.fromkeys(
            snap(p, start_pos, pitch, (H, W)) for p in slot_positions))
```

Insert the broadened fallback BEFORE the `if not slot_positions: return` line (so the fallback's positions also get snapped by the Task-2 code that follows). The block becomes:

```python
        slot_positions = []
        for t in cands:
            cy, cx = t["centroid"]
            slot_positions.append((int(round(cy)), int(round(cx))))
        # Broadened fallback (Phase-3): when the scene graph surfaces no framed targets, treat
        # small non-agent, non-background objects as targets (e.g. collectible items). reach-all
        # over them == collect-all. Only fires when scene-graph found nothing, so games with framed
        # targets (ls20) are unaffected.
        if not slot_positions:
            for o in P.connected_components(grid, background=self._bg):
                if int(o.color) != int(self._agent_color) and o.size <= _MAX_TARGET_OBJ_SIZE:
                    cy, cx = o.centroid
                    slot_positions.append((int(round(cy)), int(round(cx))))
        if not slot_positions:
            return
        # Snap targets ... (Task 2)
        pitch = lattice_pitch(self._deltas)
        slot_positions = list(dict.fromkeys(
            snap(p, start_pos, pitch, (H, W)) for p in slot_positions))
```

Add the module-level constant near the top of `discovery_explorer.py` (by `_MAX_TRANSFORM_STEPS`):

```python
_MAX_TARGET_OBJ_SIZE = 64   # objects this small (non-agent, non-bg) can be collect targets
```

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k falls_back -v`
Expected: PASS (`_plan == [4, 4]`). Then the whole file + the fast suites:
`PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py tests/test_motion_lattice.py -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/discovery_explorer.py tests/test_discovery_explorer.py
git commit -q -m "feat(discovery): broadened target extraction for collect (small non-agent objects)"
```

---

## Task 4: Re-run the bake-off + update the gated guards

**Files:** Modify `tests/test_discovery_explorer.py` (guards)

- [ ] **Step 1: Full bake-off run; record the VERBATIM table.**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py 10000`
(Slow — several minutes. `salience(ls20)` may be pathologically slow; if the full run stalls, run the four discovery/collect variants + `salience(collect)` individually via a tiny throwaway that imports `discovery_bakeoff` and calls `run_engine(name, eng, budget, game=...)`, as in Phase-2 Task 9.) Capture per (engine,game): levels cleared, L1 actions, L1 efficiency, transfer slope.

- [ ] **Step 2: DIAGNOSE + record the A1-vs-A2 verdict.**

For ls20: did discovery-A1 and/or discovery-A2 now clear L1 (snapped targets reachable)? Record actions + efficiency. **Report the A1-vs-A2 monotone-paint verdict** (did A1 solve and A2 also/again? did A2 hit the node cap?). For collect: did discovery-collect clear L1 via the broadened item targets? If any still fail, localize the phase (throwaway diagnostic in /tmp, removed before commit). Include findings in the report.

- [ ] **Step 3: Update the gated guards to passing thresholds.**

In `tests/test_discovery_explorer.py`, the Phase-2 guards `test_phase2_ls20_paint_clears_l1` and `test_phase2_collect_clears_l1` currently assert `<2000` / `<1100` and are expected-fail. Set `planner_backend` in the ls20 guard to whichever backend won, and set each `l1.actions < <threshold>` to a value just above the OBSERVED winning action count (e.g. observed+25%). If a guard now PASSES, keep it as a real passing test. If a guard STILL fails, do NOT weaken it — keep it as the documented target and say so in the report.

- [ ] **Step 4: Confirm gating + run guards.**

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -q` (guards skipped without RUN_BAKEOFF).
`RUN_BAKEOFF=1 ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k phase2 -v` (report pass/fail honestly — slow).

- [ ] **Step 5: Commit**

```bash
git add tests/test_discovery_explorer.py
git commit -q -m "test(bakeoff): Phase-3 update ls20/collect L1 guards to observed results"
```

---

## Task 5: Experiment 45 write-up

**Files:** Modify `docs/experiment-overview.html`

- [ ] **Step 1: Append an Experiment-45 panel** matching the existing markup/CSS (read recent panels — `panel`, `accent`/`kill`/`win`, `t-win`/`t-kill`, table rows `<tr><td class="m">N</td>`). Report HONESTLY the bake-off table from Task 4, whether the motion-lattice snap let discovery clear ls20 L1 (and via A1/A2 — the monotone-paint verdict), whether broadened extraction let discovery-collect clear collect L1, transfer slopes, and per-level `(A_h/A_m)²`. State plainly whether ONE discovery loop now clears L1 black-box on BOTH mechanic classes (the cross-mechanic generality milestone) — or, if partial, what remains. Add a summary row to the main experiment table (Exp 45, `t-win` or `t-kill` per outcome). If results are positive, note whether v13=0.33 should be revisited (but do NOT submit).

- [ ] **Step 2: Validate HTML well-formed** (HTMLParser) and **commit.**

```bash
git add docs/experiment-overview.html
git commit -q -m "docs: Experiment 45 — motion-lattice snap + collect targets (bake-off re-run)"
```

---

## Self-Review

**Spec coverage:** motion_lattice helper (Task 1, spec §4 C1) ✓; `_build_plan` snapping (Task 2, C2) ✓; broadened collect target extraction (Task 3, C3) ✓; bake-off re-run + guard update (Task 4, C4) ✓; Exp-45 write-up (Task 5, C4) ✓; success bar — ls20 L1 + A1/A2 verdict + collect L1 measured in Task 4/5 ✓; non-goals respected (no downscaling, no planner-logic change — only slot coords) ✓.

**Placeholder scan:** Task 4 thresholds are intentionally set from OBSERVED results at implementation time (not a hidden TODO — the observed numbers don't exist until the run); the guard-passing-vs-failing branch is explicit and honest.

**Type consistency:** `lattice_pitch(deltas) -> (pr,pc)`, `snap(pos, origin, pitch, bounds) -> (r,c)` used identically in Tasks 1–3; `_MAX_TARGET_OBJ_SIZE` defined in Task 3; slot schema `{"pos","attr_req","done"}` unchanged from Phase 2; `_build_plan`'s `start_pos`/`H,W`/`slot_positions` names match the current code.
