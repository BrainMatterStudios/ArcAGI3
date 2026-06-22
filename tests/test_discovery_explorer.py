import numpy as np
from arcagi3.discovery_explorer import DiscoveryExplorer


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
