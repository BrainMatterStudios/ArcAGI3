import os
import numpy as np
import pytest
from arcagi3.discovery_explorer import DiscoveryExplorer


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live ls20 bake-off; set RUN_BAKEOFF=1")
def test_discovery_beats_salience_on_l1():
    # TARGET, NOT YET MET: as of Exp-43, discovery(#1) does NOT clear ls20 L1 within budget.
    # Its model-discovery probes for AGENT attribute-change on tile entry (on_enter_cycles),
    # but ls20's mechanic is spatial (the agent paints a trail; its own color/shape never
    # changes), so PROBE_TRANSFORMS records zero triples -> empty model -> empty plan. This
    # test documents the intended bar and is expected to FAIL until the discovery engine
    # learns ls20's actual (non-agent-attribute) mechanic. Do NOT weaken the threshold to pass.
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("bo", "scripts/discovery_bakeoff.py")
    bo = importlib.util.module_from_spec(spec)
    sys.modules["bo"] = bo  # so @dataclass annotations resolve against the module's namespace
    spec.loader.exec_module(bo)
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0); eng.reset_all()
    res = bo.run_engine("discovery", eng, budget=5000, max_level=2)
    l1 = next((x for x in res.levels if x.level == 0 and x.cleared), None)
    assert l1 is not None and l1.actions < 2000   # vastly under salience's 7901


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live bake-off; set RUN_BAKEOFF=1")
def test_phase2_ls20_paint_clears_l1():
    # TARGET, NOT YET MET (Task 9 bake-off, 2026-06-23). NEITHER ls20 backend cleared L1:
    # discovery-A1(traversal) and discovery-A2(painted_set) both cleared 0 levels. Phase-2 paint
    # induction itself WORKS end-to-end (deltas fit; agent_color=12; 4582 world-delta obs;
    # induce_recolor_on_move produced 6 RecolorOnMove rules; paint_colors={11,9,3,8}). The wall is
    # PLAN, caused by a perception/planner COORDINATE-CONTRACT mismatch: to_grid yields the raw
    # 64x64 PIXEL frame, so the agent moves in 5-pixel steps (deltas (+-5,0)/(0,+-5)) on a lattice
    # anchored at the start pixel, but scene_graph slot centroids (e.g. (12,36) from start (15,34))
    # do NOT lie on that 5-step lattice. So A1 plan() returns None for every slot and A2
    # plan_painted_set returns "intractable" (200k node cap). No plan -> no clear. Fix is upstream
    # of this guard (logical-cell downscaling / lattice-snapped targets), NOT a threshold change.
    # The `traversal` backend is used here as the general default; do NOT weaken the threshold.
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("bo", "scripts/discovery_bakeoff.py")
    bo = importlib.util.module_from_spec(spec); sys.modules["bo"] = bo; spec.loader.exec_module(bo)
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal"); eng.reset_all()
    res = bo.run_engine("ls20-paint", eng, budget=8000, game="ls20", max_level=2)
    l1 = next((x for x in res.levels if x.level == 0 and x.cleared), None)
    assert l1 is not None and l1.actions < 2000   # vastly under salience's ~7901


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live bake-off; set RUN_BAKEOFF=1")
def test_phase2_collect_clears_l1():
    # TARGET, NOT YET MET (Task 9 bake-off, 2026-06-23). discovery-collect(traversal) cleared 0
    # levels. Two phase-level walls: (1) scene_graph.extract surfaces 0 target_candidates on the
    # collect frame (the items are not 'framed' objects), so _build_plan returns with no slots;
    # (2) the collect mechanic was never induced (_world_obs==0, _collects==[]) because the agent
    # never contacted an item during probing. salience(collect) DOES clear L0 in 1123 actions
    # (a_h=39), so the game is solvable; the discovery loop just lacks target perception + a
    # collect-reach plan. Keep this as a documented failing target; do NOT weaken to pass.
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("bo", "scripts/discovery_bakeoff.py")
    bo = importlib.util.module_from_spec(spec); sys.modules["bo"] = bo; spec.loader.exec_module(bo)
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal"); eng.reset_all()
    res = bo.run_engine("collect", eng, budget=8000, game="collect", max_level=2)
    l1 = next((x for x in res.levels if x.level == 0 and x.cleared), None)
    assert l1 is not None and l1.actions < 1100   # collect L0 a_h=39; under salience's 1123


def _grid_with_agent(pos, color=9):
    g = np.zeros((8, 8), dtype=np.int8)
    g[pos] = color
    return g


def test_first_decisions_are_movement_probes():
    eng = DiscoveryExplorer(seed=0)
    eng.reset_all()
    g = _grid_with_agent((4, 4))
    seen = set()
    for _ in range(4):
        tok = eng.decide(grid=g, gstate_terminal=False, gstate_notplayed=False,
                         levels=0, available=[1, 2, 3, 4])
        assert tok[0] in ("S", "reset")
        if tok[0] == "S":
            seen.add(tok[1])
    assert len(seen) >= 2  # tries distinct simple actions during PROBE_MOVEMENT


def test_carries_grammar_across_level_increment():
    eng = DiscoveryExplorer(seed=0)
    eng.reset_all()
    eng._deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    eng._phase = "EXECUTE"
    eng.on_level_change(new_level=1)
    assert eng._deltas == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # grammar kept
    assert eng._plan == [] and eng._walls == set()                        # layout flushed


def test_color_triple_records_tile_color_not_agent_color():
    """Regression: entered_color must be the tile the agent stepped onto (from prev frame),
    NOT the agent's own previous color.

    Setup:
      prev frame: agent (color 9) at (2,2); transformer tile (color 4) at (2,3)
      cur  frame: agent moved right onto (2,3) and got recolored to 10; tile pixel overwritten

    The transform triple's entered_color should be 4 (the TILE), not 9 (the old agent color).
    """
    eng = DiscoveryExplorer(seed=0)
    eng.reset_all()

    # Teach the engine that the agent has color 9 and background is 0.
    eng._bg = 0
    eng._agent_color = 9

    # prev frame: agent at (2,2) with color 9; transformer tile at (2,3) with color 4.
    prev = np.zeros((6, 6), dtype=np.int8)
    prev[2, 2] = 9
    prev[2, 3] = 4

    # current frame: agent moved right onto (2,3) and got recolored to 10;
    # tile pixel is now overwritten by the agent.
    cur = np.zeros((6, 6), dtype=np.int8)
    cur[2, 3] = 10
    # Agent is now color 10 at (2,3).  Set _agent_color to 10 so _agent_cells finds it.
    eng._agent_color = 10

    # Wire up the state that _ingest_transform reads:
    # - _prev_grid is the frame before this step (prev)
    # - _prev_attr is the agent's attributes as measured from prev
    from arcagi3.attribute_state import agent_attributes
    eng._prev_grid = prev
    eng._prev_attr = agent_attributes(prev, [(2, 2)])  # agent was at (2,2), color 9

    # Call _ingest_transform with the current grid.
    eng._ingest_transform(cur)

    # A color triple must have been recorded with entered_color == 4 (the TILE color),
    # not 9 (the agent's old color — that was the bug).
    color_triples = [t for t in eng._triples if t["attr"] == "color"]
    assert color_triples, "No color triple recorded — check _ingest_transform wiring."
    assert color_triples[0]["entered_color"] == 4, (
        f"entered_color should be 4 (tile color) but got {color_triples[0]['entered_color']}"
    )


def test_on_level_change_resets_coverage_budget():
    """Fix C: on_level_change must reset the transform-probe budget (_transform_steps) and
    _probe_queue so deeper levels get a full coverage budget, while KEEPING the induced grammar
    (deltas) for cross-level transfer."""
    eng = DiscoveryExplorer(seed=0)
    eng.reset_all()
    eng._deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    eng._transform_steps = 399
    eng._probe_queue = [1, 2, 3]

    eng.on_level_change(1)

    assert eng._transform_steps == 0
    assert eng._probe_queue == []
    assert eng._deltas == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # grammar kept


def test_build_plan_enumerates_shape_goals():
    """Fix A: _build_plan must enumerate SHAPE goal-specs, not only color. A rotation/orientation
    goal manifests as a target shape_sig value (not a color), so the planner must be able to
    express a slot whose attr_req carries a "shape" key.

    We construct an engine whose ONLY attainable transform is a shape change, place the agent and
    a transformer tile such that the only way to satisfy the slot is to acquire the target shape,
    and assert the produced plan's slot carried a "shape" requirement (by checking the model can
    only be solved via a shape goal — i.e. a plan exists and was found under a shape spec)."""
    import arcagi3.factored_model as FM
    import arcagi3.scene_graph as SG

    eng = DiscoveryExplorer(seed=0)
    eng.reset_all()
    eng._bg = 0
    eng._agent_color = 9
    eng._deltas = {4: (0, 1)}  # move right
    # A shape-change cycle on tile color 5: shape () -> ((0,0),(0,1)).
    shape_a = ()
    shape_b = ((0, 0), (0, 1))
    eng._triples = [
        {"entered_color": 5, "attr": "shape", "before": shape_a, "after": shape_b},
    ]

    # Grid: agent at (0,0) color 9; transformer tile color 5 at (0,1); slot target at (0,2).
    grid = np.zeros((1, 3), dtype=np.int8)
    grid[0, 0] = 9
    grid[0, 1] = 5
    # Build the cached cycles/tiles the way the phase machine would.
    eng._build_model(grid)

    # Capture which attr_req specs the planner is asked to solve.
    captured: list[dict] = []
    real_plan = FM.plan

    def spy_plan(model, start, slots, **kw):
        req = dict(slots[0]["attr_req"])
        captured.append(req)
        # Make non-shape specs (reach-only, color) "unsatisfiable" so enumeration is forced to
        # proceed to the shape spec — this isolates the question "is shape ever enumerated?".
        if "shape" not in req:
            return None
        return real_plan(model, start, slots, **kw)

    # Force a single slot target at (0,2) regardless of scene-graph heuristics.
    orig_extract = SG.extract
    monkey = {"target_candidates": [{"centroid": (0.0, 2.0)}]}

    import arcagi3.discovery_explorer as DE
    DE.plan = spy_plan
    DE.SG.extract = lambda g, bg: monkey
    try:
        eng._build_plan(grid)
    finally:
        DE.plan = real_plan
        DE.SG.extract = orig_extract

    # A shape spec must have been among the enumerated attr_req specs.
    shape_specs = [r for r in captured if "shape" in r]
    assert shape_specs, f"_build_plan never tried a shape goal-spec; tried: {captured}"


def test_world_delta_excludes_agent_and_records_recolor():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0); eng.reset_all()
    eng._bg = 0; eng._agent_color = 9
    prev = np.zeros((3, 3), dtype=np.int8); prev[0, 0] = 9; prev[1, 1] = 11
    cur = np.zeros((3, 3), dtype=np.int8); cur[0, 1] = 9; cur[1, 1] = 3
    eng._prev_grid = prev
    eng._ingest_world_delta(cur)
    assert {"from_color": 11, "to_color": 3, "vanished": False} in eng._world_obs
    assert all(o["from_color"] != 9 and o["to_color"] != 9 for o in eng._world_obs)


def test_planner_backend_flag_default_and_set():
    from arcagi3.discovery_explorer import DiscoveryExplorer
    assert DiscoveryExplorer(seed=0)._backend == "traversal"
    assert DiscoveryExplorer(seed=0, planner_backend="painted_set")._backend == "painted_set"


def test_a1_treats_paint_cells_as_passable():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    from arcagi3.transform_induction import RecolorOnMove
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal")
    eng.reset_all(); eng._bg = 0; eng._agent_color = 9
    eng._deltas = {3: (0, -1), 4: (0, 1)}
    eng._recolors = [RecolorOnMove(11, 3)]; eng._paint_colors = {11}
    grid = np.zeros((1, 3), dtype=np.int8); grid[0, 0] = 9; grid[0, 1] = 11
    eng._walls = {(0, 1)}                 # naive probe marked the 11-cell blocked
    eng._tiles = {}; eng._cycles = []; eng._terminal = None
    import arcagi3.discovery_explorer as DE
    orig = DE.SG.extract
    DE.SG.extract = lambda g, bg: {"target_candidates": [{"centroid": (0.0, 2.0)}]}
    try:
        eng._build_plan(grid)
    finally:
        DE.SG.extract = orig
    assert eng._plan and eng._plan[0] == 4   # A1 plans THROUGH the paint cell, not blocked by it


def test_build_plan_snaps_offlattice_target_to_reachable_cell():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal")
    eng.reset_all(); eng._bg = 0; eng._agent_color = 9
    eng._deltas = {1: (-5, 0), 2: (5, 0), 3: (0, -5), 4: (0, 5)}  # 5px lattice
    eng._tiles = {}; eng._cycles = []; eng._terminal = None
    grid = np.zeros((64, 64), dtype=np.int8); grid[10, 10] = 9   # agent single cell at (10,10)
    import arcagi3.discovery_explorer as DE
    orig = DE.SG.extract
    DE.SG.extract = lambda g, bg: {"target_candidates": [{"centroid": (12.0, 18.0)}]}  # OFF-lattice
    try:
        eng._build_plan(grid)
    finally:
        DE.SG.extract = orig
    # (12,18) snaps to (10,20) on the 5px lattice from origin (10,10); reachable by 2x action-4.
    assert eng._plan == [4, 4]


def test_build_plan_falls_back_to_small_objects_when_no_scene_targets():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal")
    eng.reset_all(); eng._bg = 0; eng._agent_color = 14
    eng._deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # 1px lattice (collect-like)
    eng._tiles = {}; eng._cycles = []; eng._terminal = None
    grid = np.zeros((6, 6), dtype=np.int8); grid[0, 0] = 14; grid[0, 2] = 6  # agent + one item
    import arcagi3.discovery_explorer as DE
    orig = DE.SG.extract
    DE.SG.extract = lambda g, bg: {"target_candidates": []}   # scene graph finds nothing
    try:
        eng._build_plan(grid)
    finally:
        DE.SG.extract = orig
    # fallback surfaces the item at (0,2); reach-all plans to it: right, right.
    assert eng._plan == [4, 4]
