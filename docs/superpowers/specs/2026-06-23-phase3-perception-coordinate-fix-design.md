# Phase 3 — Perception/Coordinate Fix (motion-lattice snapping + collect target extraction)

**Date:** 2026-06-23
**Branch:** phase3-perception-coordinate-fix
**Status:** design — awaiting user review
**Builds on:** Phase 2 (`docs/superpowers/specs/2026-06-22-phase2-world-transform-bakeoff-design.md`)

## 1. Context & motivation

Experiment 44 (Phase 2) showed the discovery engine's *induction* works end-to-end (ls20: 4582 paint
deltas captured, 6 recolor rules, 1038 paintable cells) but it clears **0 levels** because the bottleneck
moved to the **perception→planner coordinate contract**, in two distinct ways:

- **ls20 — coordinate-lattice mismatch.** `perception.to_grid` returns the raw 64×64 PIXEL frame; the agent
  moves in 5-pixel steps on a lattice anchored at its start pixel; but `scene_graph` slot centroids are
  off-lattice pixels → unreachable by construction. A1's `plan()` returns None; A2's `plan_painted_set`
  returns "intractable". The A1-vs-A2 / monotone-paint question was never exercised (both blocked upstream).
- **collect — target perception.** `scene_graph.extract` surfaces 0 `target_candidates` (items aren't
  "framed" objects), so `_build_plan` has no slots and the agent never targets the items.

Phase 3 fixes both so the induced plans become executable, then re-runs the bake-off.

## 2. Goal & success bar

Make engine #1's induced plans **executable on the agent's real motion lattice** and give it **targets on
both games**, then validate by re-running the Phase-2 bake-off black-box (source = grader only).

**Success bar:**
1. `discovery` clears **ls20 L1** via A1 and/or A2 (and this finally yields the **A1-vs-A2 monotone-paint
   verdict** Phase 2 never reached), with transfer toward L2+.
2. `discovery-collect` clears **collect L1** via reach-all over snapped item targets.
3. The bake-off table shows engine #1 clearing levels at a meaningful `(A_h / A_m)²` (per-level, via the
   Phase-2 true per-level grader).
4. Negative/partial results are reported honestly and localized.

## 3. Scope & non-goals

**In scope:** a motion-lattice helper (pitch + origin + snap); `_build_plan` snapping; broadened target
extraction for collect; bake-off re-run + Experiment-45 write-up; updating the Phase-2 gated guards to
passing thresholds if engine #1 now clears L1.

**Non-goals (YAGNI):**
- No logical-cell downscaling of the whole pipeline (Approach 2) — snapping is the surgical fix; keep the
  validated pixel-based induction (walls/tiles/paint) intact.
- No new mechanic primitives (paint/collect already exist).
- No change to the planners' search logic (A1 `plan`, A2 `plan_painted_set`) — only the *coordinates of the
  slots* handed to them change.
- Carry forward (do not fix here) the two Phase-2 tech-debt notes: the A1 `if actions:` vs A2
  `if status=="solved"` divergence, and the `_world_obs`-kept-on-level-change doc comment — UNLESS they
  block a Phase-3 test (then fix minimally and note it).

## 4. Architecture / components

Builds on the flat `src/arcagi3/` layout. Each unit has one responsibility:

### Component 1 — `src/arcagi3/motion_lattice.py` (new, pure)
- `lattice_pitch(deltas) -> (pr, pc)`: from `{action: (dr,dc)}`, the per-axis pitch = the min nonzero
  absolute displacement on each axis (e.g. ls20 → (5,5)). Returns `(0,0)`-safe defaults if an axis has no
  motion (treat a 0 pitch as "no snapping on that axis").
- `snap(pos, origin, pitch, bounds) -> (r,c)`: `origin + round((pos-origin)/pitch)*pitch` per axis (pitch 0
  → leave that axis = origin's), clamped to `0..bounds-1`. Returns a pixel coordinate ON the lattice.
- No game constants; pure and unit-testable.

### Component 2 — `_build_plan` snapping (modify `discovery_explorer.py`)
- Compute `origin = start_pos` and `pitch = lattice_pitch(self._deltas)` once.
- Snap every target centroid to the lattice via `snap(...)` before building slots; dedup snapped positions.
- The planner deltas already move by pitch, so snapped slots are reachable. Walls/tiles/paintable cells
  remain pixel-based and unchanged (consistent with the pixel lattice).
- Applies to BOTH backends (A1 and A2) and to collect targets.

### Component 3 — broadened target extraction (modify `discovery_explorer.py`)
- In `_build_plan`, if `scene_graph.extract(...)["target_candidates"]` is empty, fall back to sourcing
  candidate target positions from `P.connected_components(grid, background=self._bg)`: small objects whose
  color is neither the background nor `self._agent_color` (the collectible items). Use their centroids
  (then snapped). This yields reach-all-items = collect-all for the collect game.
- Keep it general (size threshold + not-agent/not-bg), no game constants.

### Component 4 — bake-off re-run + Experiment 45 (modify `discovery_bakeoff.py` usage + `docs/...`)
- Re-run `scripts/discovery_bakeoff.py` (harness unchanged). Record the table.
- Update the Phase-2 gated guards (`test_phase2_ls20_paint_clears_l1`, `test_phase2_collect_clears_l1`) to
  passing thresholds based on observed results (do NOT weaken if still failing — report honestly).
- Append Experiment 45 to `docs/experiment-overview.html` with the table + the A1-vs-A2 verdict.

## 5. Data flow

Unchanged except: in `_build_plan`, target centroids now route through (broadened extraction →) `snap()`
before becoming slots. The action ids the engine emits are unchanged (snapping affects only target
coordinates, not the action vocabulary).

## 6. Risks & mitigations

- **Non-uniform / non-integer lattice** (e.g. a game scaled 14→64 ≈ 4.57 px/cell). Mitigation: pitch is
  derived from the *measured* agent displacement, which is the true step size; snapping to that pitch is
  correct even if it isn't a clean grid divisor. If a game has a genuinely non-uniform lattice, snapping
  degrades gracefully (nearest measured-pitch cell); flagged as a limitation, not silently wrong.
- **Snap picks the wrong cell** (target between two lattice cells). Mitigation: `round` picks the nearest;
  the bake-off + guards validate end-to-end. If a target is consistently mis-snapped, the live run localizes it.
- **Broadened extraction surfaces non-target small objects** (e.g. ls20 transformer tiles as "targets").
  Mitigation: the reach-all planner only succeeds when ALL surfaced slots are satisfiable; spurious targets
  would make plans fail, which the bake-off detects. Prefer the scene-graph candidates first; use the
  broadened fallback only when scene-graph returns none (so ls20, which has framed targets, is unaffected).
- **collect items vanish as collected** — the reach-all plan is computed once on the initial frame; as items
  vanish the remaining plan still targets the original cells (reaching them collects them). Acceptable;
  re-planning on misprediction (REFINE) covers drift.

## 7. File structure

- Create `src/arcagi3/motion_lattice.py` — `lattice_pitch`, `snap`.
- Modify `src/arcagi3/discovery_explorer.py` — `_build_plan`: snap targets; broadened target fallback.
- Modify `tests/test_discovery_explorer.py` — snapping + broadened-extraction unit tests; update gated guards.
- Create `tests/test_motion_lattice.py` — pitch + snap unit tests.
- Modify `docs/experiment-overview.html` — Experiment 45.

## 8. Testing

- **Unit (motion_lattice):** `lattice_pitch` derives (5,5) from ls20-style deltas and handles a zero-motion
  axis; `snap` maps an off-lattice point to the nearest on-lattice cell and clamps to bounds; snapping an
  already-on-lattice point is identity.
- **Unit (discovery):** `_build_plan` with an off-lattice target (and known deltas) now produces a non-empty,
  reachable plan whose slots are on the lattice; broadened extraction surfaces small non-agent/non-bg objects
  as candidates when scene-graph returns none, and does NOT override scene-graph candidates when present.
- **Integration (gated, live, `RUN_BAKEOFF=1`):** the Phase-2 guards re-run — `test_phase2_ls20_paint_clears_l1`
  and `test_phase2_collect_clears_l1` now expected to PASS (thresholds set from observed results; not weakened
  if still failing). The bake-off run is the top-level integration test; ls20/collect true per-level `A_h`
  are the oracles.
