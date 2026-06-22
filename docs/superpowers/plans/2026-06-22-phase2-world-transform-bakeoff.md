# Phase 2 — World-Transform Primitives + Path-Opening Planner Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend engine #1 to discover *world* transforms (paint/path-opening for ls20, collect for sk48) and bake off two path-opening planner backends, validating one discovery loop across two mechanic classes black-box.

**Architecture:** Add two general DSL primitives (`recolor_on_move`, `collect_on_contact`) fit from world-cell deltas; give the planner two interchangeable path-opening backends (A1 paint-as-traversal, A2 exact painted-set with a paint/step budget) selected by a flag; fix the `A_h` grader to score each level from its true mid-game start; register A1/A2/collect engine variants in the bake-off across ls20 and sk48.

**Tech Stack:** Python 3.12, numpy, pytest (`pythonpath=["src"]`), `arc_agi`/`arcengine` (offline). No GPU/LLM.

**Two hypotheses being tested (be a researcher — neither is assumed):**
- **A1:** ls20 painting is unlimited/permanent/order-free → paintable cells are simply passable; planner state stays Phase-1-small.
- **A2:** painting is budget-limited / order-sensitive → track the painted-set + a budget; exact but node-capped.
The bake-off decides empirically; if both fail, that localizes the path-opening structure as the next unknown.

**Reuse (current signatures, verified):**
- `transform_induction.py`: `OnEnterCycle(tile_color, attribute, order)`, `induce_on_enter_cycles(triples, min_support=1)`, `TerminalPredicate.holds(agent_pos, agent_attrs, slots)`, `induce_terminal(prewin)`.
- `factored_model.py`: `FactoredState(pos, attrs, completed)` (frozen, attrs accepts dict), `InducedModel(deltas, walls, tiles, width, height, terminal)` with `.step`, `_satisfied(state, slots)`, `plan(model, start, slots, max_nodes=200_000)` (slots use `"pos"`,`"attr_req"`,`"done"`).
- `discovery_explorer.py`: `DiscoveryExplorer(seed=0)`; phase machine in `_decide_inner`; `_build_model(grid)` (sets `self._cycles/_tiles/_terminal`), `_build_plan(grid)`, `_ingest_movement`, `_ingest_transform`, `_agent_cells/_agent_pos`, `reset_all`, `on_level_change`.
- `bakeoff_metrics.py`: `efficiency(a_h, a_m)`, `transfer_slope(list)`, `human_baseline_actions(level=0, max_nodes=500_000)`.
- `scripts/truemodel_planner.py`: `load_ls20_class()`, `fresh_game(Ls20)`, `state_key(g)`, `apply(g, action)`, `bfs_solve(Ls20, start_level=0, max_nodes=200_000) -> (sol, expanded, unique, dt, why)`.
- `scripts/discovery_bakeoff.py`: `make_ls20()`, `run_engine(name, engine, budget, max_level=4)`, `LevelResult`, `EngineResult`, `print_table`, `main`.

**Guardrails:** zero game-specific constants in engine primitives; source = grader only; minimal additions (no third mechanic).

---

## File Structure

- Modify `src/arcagi3/transform_induction.py` — add `RecolorOnMove`/`induce_recolor_on_move`, `CollectOnContact`/`induce_collect_on_contact`.
- Modify `src/arcagi3/factored_model.py` — add `plan_painted_set(...)` (A2 backend) returning `(actions, status)`.
- Modify `scripts/truemodel_planner.py` — add `bfs_solve_current(game, max_nodes)` and `optimal_actions_per_level(GameClass, up_to_level, max_nodes)`.
- Create `scripts/truemodel_sk48.py` — sk48 game loader reusing the chain (thin: load sk48 class + reuse generic chain).
- Modify `src/arcagi3/bakeoff_metrics.py` — `human_baseline_actions(game, level)` dispatch + per-game per-level cache.
- Modify `src/arcagi3/discovery_explorer.py` — `_ingest_world_delta`; world-transform induction in `_build_model`; `planner_backend` flag; A1/A2 paint planning + collect goal.
- Modify `scripts/discovery_bakeoff.py` — `make_game(prefix)`; sk48 run; register A1/A2/collect variants.
- Modify `docs/experiment-overview.html` — Experiment 44.
- Tests: extend `tests/test_transform_induction.py`, `tests/test_factored_model.py`, `tests/test_discovery_explorer.py`, `tests/test_bakeoff_metrics.py`; gated live guards.

---

## Task 1: `recolor_on_move` primitive induction

**Files:** Modify `src/arcagi3/transform_induction.py`; Test `tests/test_transform_induction.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_transform_induction.py
from arcagi3.transform_induction import induce_recolor_on_move, RecolorOnMove

def test_induces_recolor_on_move():
    # world-delta observations: cells of color 11 recolor to 3 as the agent moves; color 5 never does.
    obs = [
        {"from_color": 11, "to_color": 3, "vanished": False},
        {"from_color": 11, "to_color": 3, "vanished": False},
    ]
    rules = induce_recolor_on_move(obs)
    assert RecolorOnMove(from_color=11, to_color=3) in rules
    assert all(r.from_color != 5 for r in rules)

def test_recolor_requires_min_support():
    obs = [{"from_color": 11, "to_color": 3, "vanished": False}]
    assert induce_recolor_on_move(obs, min_support=2) == []
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -k recolor -v`
Expected: FAIL (`ImportError: cannot import name 'induce_recolor_on_move'`)

- [ ] **Step 3: Implement (append to `src/arcagi3/transform_induction.py`)**

```python
@dataclass(frozen=True)
class RecolorOnMove:
    from_color: int
    to_color: int


def induce_recolor_on_move(observations: list[dict], min_support: int = 1) -> list[RecolorOnMove]:
    """Induce 'moving recolors cells from_color->to_color' (paint). A world-delta observation is
    {"from_color", "to_color", "vanished": bool}; vanished deltas are collect, not recolor (skip)."""
    counts: dict[tuple, int] = defaultdict(int)
    for o in observations:
        if not o.get("vanished"):
            counts[(o["from_color"], o["to_color"])] += 1
    return [RecolorOnMove(from_color=f, to_color=t)
            for (f, t), n in counts.items() if n >= min_support]
```

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -k recolor -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/transform_induction.py tests/test_transform_induction.py
git commit -q -m "feat(discovery): induce recolor_on_move (paint) primitive"
```

---

## Task 2: `collect_on_contact` primitive induction

**Files:** Modify `src/arcagi3/transform_induction.py`; Test `tests/test_transform_induction.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_transform_induction.py
from arcagi3.transform_induction import induce_collect_on_contact, CollectOnContact

def test_induces_collect_on_contact():
    # cells of color 8 vanish (-> background) on contact; color 11 recolors (not collect).
    obs = [
        {"from_color": 8, "to_color": 0, "vanished": True},
        {"from_color": 8, "to_color": 0, "vanished": True},
        {"from_color": 11, "to_color": 3, "vanished": False},
    ]
    rules = induce_collect_on_contact(obs)
    assert CollectOnContact(color=8) in rules
    assert all(r.color != 11 for r in rules)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -k collect -v`
Expected: FAIL (`ImportError`)

- [ ] **Step 3: Implement (append to `src/arcagi3/transform_induction.py`)**

```python
@dataclass(frozen=True)
class CollectOnContact:
    color: int


def induce_collect_on_contact(observations: list[dict], min_support: int = 1) -> list[CollectOnContact]:
    """Induce 'agent contact removes cells of color C' (collect): world-delta observations whose
    `vanished` is True (cell became background)."""
    counts: dict[int, int] = defaultdict(int)
    for o in observations:
        if o.get("vanished"):
            counts[o["from_color"]] += 1
    return [CollectOnContact(color=c) for c, n in counts.items() if n >= min_support]
```

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -k collect -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/transform_induction.py tests/test_transform_induction.py
git commit -q -m "feat(discovery): induce collect_on_contact primitive"
```

---

## Task 3: A2 painted-set planner backend

**Files:** Modify `src/arcagi3/factored_model.py`; Test `tests/test_factored_model.py`

A2 tracks the painted-set and (optionally) a paint budget. State = `(pos, attrs-tuple, completed, painted)`. A cell is a hard wall iff in `true_walls`. Entering a **paintable** cell (in `paintable_cells`, not yet painted) adds it to `painted`; if `max_paints` is set and that would exceed it, the move is **blocked**. Goal = all slots satisfied (same `attr_req` schema). Returns `(actions|None, status)` where status ∈ `{"solved","no_solution","intractable"}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_factored_model.py
from arcagi3.factored_model import plan_painted_set

def _corridor_painted(max_paints):
    # 1x5 corridor, all cells open EXCEPT cols 1,2,3 are PAINTABLE (must be painted to pass).
    # No transformer tiles, no attr requirement; slot at col 4. Agent starts col 0.
    return dict(
        deltas={3: (0, -1), 4: (0, 1)},
        true_walls=set(),
        paintable_cells={(0, 1), (0, 2), (0, 3)},
        tiles={},
        width=5, height=1,
        start_pos=(0, 0), start_attrs={},
        slots=[{"pos": (0, 4), "attr_req": {}, "done": False}],
        max_paints=max_paints, max_nodes=5000,
    )

def test_a2_solves_when_budget_allows():
    actions, status = plan_painted_set(**_corridor_painted(max_paints=3))
    assert status == "solved" and actions[-1] == 4 and len(actions) == 4

def test_a2_no_solution_when_budget_too_small():
    actions, status = plan_painted_set(**_corridor_painted(max_paints=2))
    assert status == "no_solution" and actions is None

def test_a2_reports_intractable_on_node_cap():
    cfg = _corridor_painted(max_paints=3); cfg["max_nodes"] = 1
    actions, status = plan_painted_set(**cfg)
    assert status == "intractable"
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_factored_model.py -k a2 -v`
Expected: FAIL (`ImportError: cannot import name 'plan_painted_set'`)

- [ ] **Step 3: Implement (append to `src/arcagi3/factored_model.py`)**

```python
def plan_painted_set(*, deltas, true_walls, paintable_cells, tiles, width, height,
                     start_pos, start_attrs, slots, max_paints=None, max_nodes=200_000):
    """A2 backend: BFS over (pos, attrs, completed, painted). Paintable cells must be painted
    (by entering, subject to max_paints) to be traversed; true_walls always block. Tiles cycle
    attributes exactly as InducedModel.step does. Returns (actions|None, status)."""
    def attrs_tuple(d):
        return tuple(sorted(d.items()))

    def satisfied(pos, ad, completed):
        done = set(completed)
        for i, s in enumerate(slots):
            if i in done:
                continue
            if pos == s["pos"] and all(ad.get(k) == v for k, v in s["attr_req"].items()):
                done.add(i)
        return frozenset(done)

    start_ad = dict(start_attrs)
    start_completed = satisfied(start_pos, start_ad, frozenset())
    if len(start_completed) == len(slots):
        return [], "solved"
    start = (start_pos, attrs_tuple(start_ad), start_completed, frozenset())
    seen = {start}
    q = deque([(start_pos, start_ad, start_completed, frozenset(), [])])
    expanded = 0
    while q:
        if expanded >= max_nodes:
            return None, "intractable"
        pos, ad, completed, painted, path = q.popleft()
        expanded += 1
        for a, (dr, dc) in deltas.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < height and 0 <= nc < width) or (nr, nc) in true_walls:
                continue
            npainted = painted
            if (nr, nc) in paintable_cells and (nr, nc) not in painted:
                if max_paints is not None and len(painted) >= max_paints:
                    continue  # out of paint -> cannot enter
                npainted = painted | {(nr, nc)}
            nad = dict(ad)
            tile = tiles.get((nr, nc))
            if tile is not None and nad.get(tile.attribute) in tile.order:
                i = tile.order.index(nad[tile.attribute])
                nad[tile.attribute] = tile.order[(i + 1) % len(tile.order)]
            ncompleted = satisfied((nr, nc), nad, completed)
            if len(ncompleted) == len(slots):
                return path + [a], "solved"
            key = ((nr, nc), attrs_tuple(nad), ncompleted, npainted)
            if key not in seen:
                seen.add(key)
                q.append(((nr, nc), nad, ncompleted, npainted, path + [a]))
    return None, "no_solution"
```

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_factored_model.py -k a2 -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/factored_model.py tests/test_factored_model.py
git commit -q -m "feat(discovery): A2 painted-set planner backend (budget-aware path-opening)"
```

---

## Task 4: Per-level `A_h` grader (fix level>0)

**Files:** Modify `scripts/truemodel_planner.py`; Test `tests/test_bakeoff_metrics.py`

Add `bfs_solve_current(game, max_nodes)` (BFS-snapshots from the game's CURRENT state, not a fresh level-0 game) and `optimal_actions_per_level(GameClass, up_to_level, max_nodes)` (solve→advance chain). READ the existing `bfs_solve`/`fresh_game`/`state_key`/`apply` first — reuse them; `bfs_solve_current` is `bfs_solve` generalized to start from a passed instance instead of `fresh_game(Ls20)`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bakeoff_metrics.py
@pytest.mark.integration
def test_optimal_actions_per_level_advances():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tmp", "scripts/truemodel_planner.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    per_level = m.optimal_actions_per_level(m.load_ls20_class(), up_to_level=1)
    assert per_level[0] == 13                 # Exp-42 L1 optimal
    assert len(per_level) >= 2 and per_level[1] is not None  # true mid-game L2 baseline
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py::test_optimal_actions_per_level_advances -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'optimal_actions_per_level'`)

- [ ] **Step 3: Implement (append to `scripts/truemodel_planner.py`)**

```python
def bfs_solve_current(game, max_nodes=500_000):
    """BFS from the game's CURRENT level/state (deepcopy snapshots from `game`). Returns the
    action list that clears the current level, or None. Mirrors bfs_solve but does not reset."""
    import copy
    start_idx = game.level_index
    seen = {state_key(game)}
    q = deque([(game, [])])
    expanded = 0
    while q:
        g, path = q.popleft()
        expanded += 1
        if expanded > max_nodes:
            return None
        for a in MOVES:
            child = copy.deepcopy(g)
            apply(child, a)
            if child.level_index > start_idx or child._score > game._score:
                return path + [a.value]
            if child._state == GameState.GAME_OVER:
                continue
            k = state_key(child)
            if k not in seen:
                seen.add(k)
                q.append((child, path + [a]))
    return None


def optimal_actions_per_level(GameClass, up_to_level=0, max_nodes=500_000):
    """Per-level A_h via solve->advance: solve the current level, apply that optimal solution to
    reach the next, repeat. Returns [A_h[0], A_h[1], ...] (None for any level the BFS can't solve)."""
    import copy
    g = GameClass()
    g.perform_action(ActionInput(id=GameAction.RESET))
    out = []
    for _ in range(up_to_level + 1):
        sol = bfs_solve_current(g, max_nodes=max_nodes)
        out.append(None if sol is None else len(sol))
        if sol is None:
            break
        for aid in sol:
            g.perform_action(ActionInput(id=GameAction.from_id(aid)))
    return out
```

(Ensure `from collections import deque` and `from arcengine import ActionInput, GameAction, GameState` are imported at module top — add any missing.)

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py::test_optimal_actions_per_level_advances -v`
Expected: PASS (`per_level[0] == 13`, `per_level[1]` non-None)

- [ ] **Step 5: Commit**

```bash
git add scripts/truemodel_planner.py tests/test_bakeoff_metrics.py
git commit -q -m "feat(grader): true per-level A_h via solve->advance chain (fix level>0)"
```

---

## Task 5: sk48 grader + `human_baseline_actions(game, level)` dispatch

**Files:** Create `scripts/truemodel_sk48.py`; Modify `src/arcagi3/bakeoff_metrics.py`; Test `tests/test_bakeoff_metrics.py`

FIRST read `environment_files/sk48/d8078629/sk48.py` to confirm its game class name, action set, and win predicate (do NOT assume it mirrors ls20). `truemodel_sk48.py` exposes `load_sk48_class()` and reuses `truemodel_planner.bfs_solve_current`/`optimal_actions_per_level` (which are game-agnostic — they only use `level_index`, `_score`, `_state`, `state_key`, `apply`/`MOVES`). If sk48's action set differs from `[1,2,3,4]`, parameterize `MOVES` (add a `moves` arg to the generic functions) — adapt as the source requires.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bakeoff_metrics.py
def test_efficiency_dispatch_signature():
    from arcagi3.bakeoff_metrics import human_baseline_actions
    import inspect
    sig = inspect.signature(human_baseline_actions)
    assert "game" in sig.parameters and "level" in sig.parameters

@pytest.mark.integration
def test_sk48_baseline_level1():
    from arcagi3.bakeoff_metrics import human_baseline_actions
    a_h = human_baseline_actions(game="sk48", level=0)
    assert a_h is None or a_h >= 1   # tolerant: confirms the grader runs without error
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py -k "dispatch or sk48" -v`
Expected: FAIL (signature lacks `game`)

- [ ] **Step 3: Implement**

Create `scripts/truemodel_sk48.py`:
```python
"""sk48 A_h grader: load the sk48 env class and reuse the generic per-level solver."""
from __future__ import annotations
import importlib.util
from pathlib import Path

SK48_PATH = Path(__file__).resolve().parent.parent / "environment_files/sk48/d8078629/sk48.py"


def load_sk48_class():
    spec = importlib.util.spec_from_file_location("sk48_env", SK48_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # confirm the class name from the source; ARCBaseGame subclass is the game class.
    return mod.Sk48  # ADAPT if the source names it differently (read sk48.py to confirm)
```

Rewrite `human_baseline_actions` in `src/arcagi3/bakeoff_metrics.py`:
```python
_GRADERS = {
    "ls20": ("scripts/truemodel_planner.py", "load_ls20_class"),
    "sk48": ("scripts/truemodel_sk48.py", "load_sk48_class"),
}
_PER_LEVEL_CACHE: dict[str, list] = {}


def _load(path, fn):
    p = Path(__file__).resolve().parents[2] / path
    spec = importlib.util.spec_from_file_location(Path(path).stem, p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod, getattr(mod, fn)


def human_baseline_actions(game: str = "ls20", level: int = 0, max_nodes: int = 500_000) -> int | None:
    """True per-level A_h via the solve->advance chain (grader-only source access). Cached per game."""
    if game not in _PER_LEVEL_CACHE:
        tm, _ = _load("scripts/truemodel_planner.py", "load_ls20_class")  # for the chain fns
        _, loader = _load(*_GRADERS[game])
        _PER_LEVEL_CACHE[game] = tm.optimal_actions_per_level(loader(), up_to_level=max(level, 4),
                                                              max_nodes=max_nodes)
    per = _PER_LEVEL_CACHE[game]
    return per[level] if level < len(per) else None
```

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py -k "dispatch or sk48 or per_level or baseline" -v`
Expected: PASS (signature test; integration tests pass or tolerantly handle a None/intractable grader). If `mod.Sk48` is the wrong class name, fix per the source and re-run.

- [ ] **Step 5: Commit**

```bash
git add scripts/truemodel_sk48.py src/arcagi3/bakeoff_metrics.py tests/test_bakeoff_metrics.py
git commit -q -m "feat(grader): sk48 per-level A_h + human_baseline_actions(game, level) dispatch"
```

---

## Task 6: World-delta observation in PROBE_TRANSFORMS

**Files:** Modify `src/arcagi3/discovery_explorer.py`; Test `tests/test_discovery_explorer.py`

Add `_ingest_world_delta(grid)`: compute non-agent cells whose color changed `prev_grid -> grid`, EXCLUDING the agent's current+previous footprint, and append observations `{"from_color", "to_color", "vanished"}` to a new `self._world_obs` list. `vanished` = `to_color == self._bg`. Call it from PROBE_TRANSFORMS alongside `_ingest_transform`. Initialize `self._world_obs = []` in `reset_all` (KEEP across `on_level_change` — world grammar transfers, like attribute grammar).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_discovery_explorer.py
def test_world_delta_excludes_agent_and_records_recolor():
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0); eng.reset_all()
    eng._bg = 0; eng._agent_color = 9
    # prev: agent (9) at (0,0); a paint cell (11) at (1,1)
    prev = np.zeros((3, 3), dtype=np.int8); prev[0, 0] = 9; prev[1, 1] = 11
    # cur: agent moved to (0,1); the (1,1) cell got painted 11->3 (a world delta, not the agent)
    cur = np.zeros((3, 3), dtype=np.int8); cur[0, 1] = 9; cur[1, 1] = 3
    eng._prev_grid = prev
    eng._ingest_world_delta(cur)
    assert {"from_color": 11, "to_color": 3, "vanished": False} in eng._world_obs
    # the agent's own move (9 leaving (0,0), arriving (0,1)) must NOT be recorded as a world delta
    assert all(o["from_color"] != 9 and o["to_color"] != 9 for o in eng._world_obs)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k world_delta -v`
Expected: FAIL (`AttributeError: ... has no attribute '_ingest_world_delta'`)

- [ ] **Step 3: Implement**

In `reset_all`, add `self._world_obs: list[dict] = []`. In `on_level_change`, do NOT clear `_world_obs` (grammar transfers). Add the method:
```python
    def _ingest_world_delta(self, grid):
        """Record non-agent cells that changed color this step (paint / collect signal).

        Excludes the agent's current and previous footprint so the agent's own movement isn't
        mistaken for a world transform. vanished == cell became background.
        """
        if self._bg is None or self._prev_grid is None or grid.shape != self._prev_grid.shape:
            return
        footprint = set(self._agent_cells(grid)) | set(self._agent_cells(self._prev_grid))
        changed = np.argwhere(grid != self._prev_grid)
        for r, c in changed:
            if (int(r), int(c)) in footprint:
                continue
            f = int(self._prev_grid[r, c]); t = int(grid[r, c])
            if f == int(self._agent_color) or t == int(self._agent_color):
                continue
            self._world_obs.append({"from_color": f, "to_color": t, "vanished": t == self._bg})
```
Call `self._ingest_world_delta(grid)` inside the `PROBE_TRANSFORMS` block right after `self._ingest_transform(grid)`.

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k world_delta -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/discovery_explorer.py tests/test_discovery_explorer.py
git commit -q -m "feat(discovery): observe world-cell deltas (paint/collect signal)"
```

---

## Task 7: Wire world-transforms + A1/A2 backend + collect into the discovery loop

**Files:** Modify `src/arcagi3/discovery_explorer.py`; Test `tests/test_discovery_explorer.py`

This is the integration task. Add a `planner_backend` flag and use the new primitives:
1. `__init__(self, seed=0, planner_backend="traversal")` → store `self._backend` (`"traversal"`=A1 | `"painted_set"`=A2). Add to `reset_all` nothing new beyond `_world_obs` (Task 6).
2. In `_build_model`, after the existing cycle induction, also compute `self._recolors = induce_recolor_on_move(self._world_obs)` and `self._collects = induce_collect_on_contact(self._world_obs)`, and the set of paint source colors `self._paint_colors = {r.from_color for r in self._recolors}`.
3. In `_build_plan`, compute `paintable_cells` = cells of any `paint_colors` color (from `P.connected_components`). Then branch on `self._backend`:
   - **A1 (`"traversal"`):** build the `InducedModel` with `walls = self._walls - paintable_cells` (paintable cells are passable), and call the existing `plan(model, start, slots)` over the enumerated reqs (reuse the current enumeration code).
   - **A2 (`"painted_set"`):** for each enumerated req, call `plan_painted_set(deltas=self._deltas, true_walls=self._walls - paintable_cells, paintable_cells=paintable_cells, tiles=self._tiles, width=W, height=H, start_pos=start_pos, start_attrs=start_attrs, slots=slots, max_paints=None, max_nodes=200_000)`; take the first `status=="solved"`. (max_paints=None for now — a step/paint budget is a later refinement; A2 still differs from A1 by tracking painted cells, which matters once the live env imposes limits.)
4. Collect: if `self._collects` and no paint rules, set goal as "all collectible cells gone" — represent by adding, for each collectible color, slots that are satisfied when those cells vanish. SIMPLEST consistent approach: treat collect as reach-all — for the collect case, if `scene_graph` target_candidates are collectibles, the existing reach-only enumeration (`req={}`) over multiple slot_positions already drives the agent to each; keep collect handling minimal here and rely on the live bake-off (Task 9) to reveal whether sk48 needs more.

- [ ] **Step 1: Write the failing test (backend flag + A1 paint-passable)**

```python
# add to tests/test_discovery_explorer.py
def test_planner_backend_flag_default_and_set():
    from arcagi3.discovery_explorer import DiscoveryExplorer
    assert DiscoveryExplorer(seed=0)._backend == "traversal"
    assert DiscoveryExplorer(seed=0, planner_backend="painted_set")._backend == "painted_set"

def test_a1_treats_paint_cells_as_passable():
    # With a recolor rule for color 11 and a wall set that includes an 11-cell, A1 must NOT treat
    # that 11-cell as a wall. We check _build_plan's wall handling via a tiny constructed state.
    import numpy as np
    from arcagi3.discovery_explorer import DiscoveryExplorer
    from arcagi3.transform_induction import RecolorOnMove
    eng = DiscoveryExplorer(seed=0, planner_backend="traversal")
    eng.reset_all(); eng._bg = 0; eng._agent_color = 9
    eng._deltas = {3: (0, -1), 4: (0, 1)}
    eng._recolors = [RecolorOnMove(11, 3)]; eng._paint_colors = {11}
    # grid: agent at (0,0); paint cell 11 at (0,1); slot target at (0,2)
    grid = np.zeros((1, 3), dtype=np.int8); grid[0, 0] = 9; grid[0, 1] = 11
    eng._walls = {(0, 1)}              # naive probe marked the 11-cell blocked
    eng._tiles = {}; eng._cycles = []; eng._terminal = None
    # monkeypatch scene graph to give a slot at (0,2)
    import arcagi3.discovery_explorer as DE
    orig = DE.SG.extract
    DE.SG.extract = lambda g, bg: {"target_candidates": [{"centroid": (0.0, 2.0)}]}
    try:
        eng._build_plan(grid)
    finally:
        DE.SG.extract = orig
    assert eng._plan and eng._plan[0] == 4   # A1 plans through the paint cell, not blocked by it
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k "backend or a1_treats" -v`
Expected: FAIL (`_backend`/`_recolors` not set; A1 wall handling not implemented)

- [ ] **Step 3: Implement** the `__init__` flag, the `_build_model` additions (`self._recolors/_collects/_paint_colors`), and the `_build_plan` backend branch described above. Reuse the existing req-enumeration. Ensure `reset_all` initializes `self._recolors = []`, `self._collects = []`, `self._paint_colors = set()`.

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -v`
Expected: PASS (all existing + the 2 new). Also run full suite `PYTHONPATH=src .venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/discovery_explorer.py tests/test_discovery_explorer.py
git commit -q -m "feat(discovery): wire world-transforms + A1/A2 backend + collect into the loop"
```

---

## Task 8: Harness — `make_game`, sk48 run, register A1/A2/collect

**Files:** Modify `scripts/discovery_bakeoff.py`

- [ ] **Step 1: Generalize the env factory and `human_baseline_actions` calls**

Replace `make_ls20()` with `make_game(prefix)` (same body, `startswith(prefix)`), and update `run_engine` to accept a `game` param so it grades with `human_baseline_actions(game=game, level=cur_level)`. Keep `make_ls20 = lambda: make_game("ls20")` if other code references it.

```python
def make_game(prefix: str):
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("bo"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    return client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["bakeoff"]))
```
In `run_engine(name, engine, budget, game="ls20", max_level=4)`: `env = make_game(game)`; and `a_h = a_h_cache.setdefault(cur_level, human_baseline_actions(game=game, level=cur_level))`.

- [ ] **Step 2: Rewrite `main()` to register variants per game**

```python
def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    from arcagi3.discovery_explorer import DiscoveryExplorer
    runs = [
        ("discovery-A1(ls20)", DiscoveryExplorer(seed=0, planner_backend="traversal"), "ls20"),
        ("discovery-A2(ls20)", DiscoveryExplorer(seed=0, planner_backend="painted_set"), "ls20"),
        ("salience(ls20)", SalienceExplorer(seed=0), "ls20"),
        ("discovery-collect(sk48)", DiscoveryExplorer(seed=0, planner_backend="traversal"), "sk48"),
        ("salience(sk48)", SalienceExplorer(seed=0), "sk48"),
    ]
    results = []
    for name, eng, game in runs:
        if hasattr(eng, "reset_all"):
            eng.reset_all()
        print(f"running {name}...", flush=True)
        results.append(run_engine(name, eng, budget, game=game))
    print_table(results)
```

- [ ] **Step 3: Smoke-run**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py 50`
Expected: runs without error; prints a table with all five rows (0 levels at budget 50 is fine). If sk48 env construction errors, diagnose (confirm sk48 game_id prefix); report BLOCKED with the exact error if unreachable.

- [ ] **Step 4: Commit**

```bash
git add scripts/discovery_bakeoff.py
git commit -q -m "feat(bakeoff): make_game + sk48 run + register A1/A2/collect variants"
```

---

## Task 9: Run the bake-off (live) + diagnose + regression guards

**Files:** Modify `tests/test_discovery_explorer.py` (gated guards)

- [ ] **Step 1: Full run, record the table.**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py 10000`
Record verbatim: per (engine, game) levels cleared, L1 actions, L1 efficiency, transfer slope. For A2 also note nodes / intractable.

- [ ] **Step 2: Diagnose each engine #1 result** (as in Phase-1 Task 9): if a variant fails to clear L1, localize the phase — did `_world_obs` capture recolor/vanish deltas? did `induce_recolor_on_move`/`induce_collect_on_contact` produce rules? did A1 build a paintable-aware wall set and a non-empty plan? did A2 solve or hit the node cap (intractable)? Use a temporary throwaway diagnostic; do NOT commit debug code. Report the per-phase findings and the A1-vs-A2 verdict (did the monotone-paint assumption hold?).

- [ ] **Step 3: Add gated regression guards.**

```python
# add to tests/test_discovery_explorer.py
@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live bake-off; set RUN_BAKEOFF=1")
def test_phase2_ls20_paint_clears_l1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bo", "scripts/discovery_bakeoff.py")
    bo = importlib.util.module_from_spec(spec); import sys; sys.modules["bo"] = bo
    spec.loader.exec_module(bo)
    from arcagi3.discovery_explorer import DiscoveryExplorer
    # whichever backend won in Step 1; if neither, this documents the still-unmet target.
    best = DiscoveryExplorer(seed=0, planner_backend="traversal"); best.reset_all()
    res = bo.run_engine("ls20-paint", best, budget=8000, game="ls20", max_level=2)
    l1 = next((x for x in res.levels if x.level == 0 and x.cleared), None)
    assert l1 is not None and l1.actions < 2000   # TARGET; set from Step-1 result, do not weaken
```
Add an analogous `test_phase2_sk48_collect_clears_l1` (game="sk48"). Set the backend in the ls20 guard to whichever variant won; if neither cleared, keep the guard as a documented (failing) target and say so explicitly — do NOT weaken it to pass.

- [ ] **Step 4: Confirm gating + run guards once.**

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -q` (guards SKIPPED).
`RUN_BAKEOFF=1 ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -k phase2 -v` (report pass/fail honestly).

- [ ] **Step 5: Commit**

```bash
git add tests/test_discovery_explorer.py
git commit -q -m "test(bakeoff): Phase-2 ls20-paint + sk48-collect L1 regression guards"
```

---

## Task 10: Experiment 44 write-up

**Files:** Modify `docs/experiment-overview.html`

- [ ] **Step 1: Append an Experiment-44 panel** matching the existing markup/CSS (read recent panels first — `panel`, `accent`/`kill`, `t-win`/`t-kill`). Report HONESTLY: the bake-off table (5 rows across ls20+sk48); whether the paint primitive let engine #1 clear ls20 L1 and via which backend (A1 vs A2 verdict — did monotone-paint hold?); whether collect cleared sk48 L1; transfer slopes; per-level `(A_h/A_m)²`. State whether **one discovery loop now spans paint + collect**. If a variant failed, report the localized phase. Add a summary row to the main experiment table (Exp 44, `t-win` or `t-kill` per outcome).

- [ ] **Step 2: Validate HTML well-formed** (tag balance via HTMLParser) and **commit**.

```bash
git add docs/experiment-overview.html
git commit -q -m "docs: Experiment 44 — world-transform bake-off (paint A1/A2 + collect across ls20/sk48)"
```

---

## Self-Review

**Spec coverage:** world-delta observation (Task 6) ✓; recolor + collect induction (Tasks 1–2) ✓; A1 backend (Task 7, model construction) + A2 backend (Task 3 planner, Task 7 wiring) ✓; per-level `A_h` grader fix (Task 4) + sk48 grader/dispatch (Task 5) ✓; harness make_game + sk48 + variants (Task 8) ✓; bake-off run + decision rule + guards (Task 9) ✓; Exp-44 write-up (Task 10) ✓; strict-generality (no game constants; primitives fit from observation) ✓; A2 intractable-sentinel + node cap ✓.

**Placeholder scan:** Task 5 (`mod.Sk48` class name) and Task 7 (collect handling) are explicitly flagged "read source / refine in live run" rather than hidden TODOs — the sk48 class name must be confirmed from `sk48.py` during Task 5, and collect's exact completion predicate is intentionally minimal pending the Task-9 live signal. These are the two genuine unknowns the spec (§8/§11) already calls out.

**Type consistency:** slot schema `{"pos","attr_req","done"}` is used identically in `plan`, `plan_painted_set`, and `_build_plan`. `RecolorOnMove(from_color,to_color)`, `CollectOnContact(color)`, `human_baseline_actions(game,level)`, `bfs_solve_current(game,max_nodes)`, `optimal_actions_per_level(GameClass,up_to_level,max_nodes)`, `plan_painted_set(...)->(actions,status)`, and `DiscoveryExplorer(seed, planner_backend)` are consistent across tasks.
