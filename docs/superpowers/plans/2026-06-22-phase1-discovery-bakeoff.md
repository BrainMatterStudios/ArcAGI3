# Phase 1 — Model-Discovery Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled 3-way bake-off on ls20 (black-box) that measures which discovery paradigm best recovers the game's model and converts it to score, and ship a net-new factored symbolic model-discovery engine as contestant #1.

**Architecture:** A shared harness drives any engine implementing the existing `decide()` contract against the offline ls20 env, logging per-level cleared/actions/efficiency and a transfer slope. Contestant #2 (`SalienceExplorer`) and #3 (`SpatialValueExplorer`) are reused as-is. Contestant #1 (`DiscoveryExplorer`) is new but thin: it composes existing factorization/movement/planning infrastructure and adds the one missing piece — modelling the **agent's own mutable attributes** (shape/color/orientation) that cycle when it steps on transformer tiles, plus the attribute-match completion predicate.

**Tech Stack:** Python 3.12, numpy, pytest (`pythonpath=["src"]`), `arc_agi`/`arcengine` (offline), existing `arcagi3` modules (`perception`, `movement`, `mechanic_inference`, `planner`, `goals`). No GPU required for #1/harness; no LLM.

**Reuse map (DRY — do not re-implement):**
- `perception.to_grid(frame)->grid`, `perception.detect_background(grid)->int`, `perception.connected_components(grid, background)->list[Obj]` where `Obj(color, cells, bbox, size, centroid)`.
- `movement.infer_all_translations(before, after, background)->{color:(dr,dc)}`; `movement.MotionModel`.
- `mechanic_inference.infer_mechanics(traj, background)->MechanicReport(avatar_color, avatar_action_deltas, ...)`, `traj=[(grid,("S",aid),next_grid,reward),...]`.
- `decide()` engine contract: `decide(grid, gstate_terminal, gstate_notplayed, levels, available) -> ("S",aid) | ("C",x,y) | ("reset",)`.
- ls20 env construction: the proven `scripts/capture_levelup.py:60-64` pattern (NORMAL mode, `get_environments()`, `make()`, `reset()`, `step(GameAction.from_id(aid))`; obs fields `.frame`, `.levels_completed`, `.state`).
- `A_h` grader: `scripts/truemodel_planner.py` BFS over the true model (the ONLY component allowed to read ls20 source).

**Guardrails:** ls20 source = grader only; **zero ls20-specific constants** in #1; minimal primitive DSL (see spec §6) with a single-location extension point.

---

## File Structure

- Create `src/arcagi3/attribute_state.py` — extract the agent's mutable attribute vector (color, shape-hash, orientation) from a frame + agent cells.
- Create `src/arcagi3/transform_induction.py` — induce `on_enter_cycle` rules and the attribute-match terminal predicate from `(grid, action, next_grid)` triples + the level-up contrast.
- Create `src/arcagi3/factored_model.py` — `FactoredState`, `InducedModel.step()`, and a BFS planner over the factored state.
- Create `src/arcagi3/discovery_explorer.py` — `DiscoveryExplorer` implementing `decide()`: probe → induce → plan → execute → refine, carrying grammar across levels.
- Create `src/arcagi3/bakeoff_metrics.py` — efficiency metric + `A_h` grader wrapper around `truemodel_planner`.
- Create `scripts/discovery_bakeoff.py` — the driver: build ls20 env, run each engine, emit the comparison table + transfer slopes.
- Create tests: `tests/test_attribute_state.py`, `tests/test_transform_induction.py`, `tests/test_factored_model.py`, `tests/test_bakeoff_metrics.py`, `tests/test_discovery_explorer.py`.
- Modify `docs/experiment-overview.html` — append Experiment 43 results.

---

## Task 1: Efficiency metric + A_h grader

**Files:**
- Create: `src/arcagi3/bakeoff_metrics.py`
- Test: `tests/test_bakeoff_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bakeoff_metrics.py
import math
from arcagi3.bakeoff_metrics import efficiency, transfer_slope

def test_efficiency_capped_at_1_15():
    # agent matches human-optimal -> 1.0; better-than-human still capped at 1.15
    assert efficiency(a_h=13, a_m=13) == 1.0
    assert efficiency(a_h=13, a_m=5) == 1.15

def test_efficiency_squared_decay():
    # 2x slower than human -> 0.25 ; 10x -> 0.01
    assert math.isclose(efficiency(a_h=10, a_m=20), 0.25, rel_tol=1e-9)
    assert math.isclose(efficiency(a_h=10, a_m=100), 0.01, rel_tol=1e-9)

def test_efficiency_uncompleted_is_zero():
    assert efficiency(a_h=13, a_m=None) == 0.0  # level not cleared

def test_transfer_slope_negative_when_costs_drop():
    # per-level action costs dropping with depth -> negative slope (good)
    assert transfer_slope([200, 120, 60, 40]) < 0
    assert transfer_slope([50, 50, 50]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arcagi3.bakeoff_metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/bakeoff_metrics.py
"""Bake-off scoring: the competition's per-level efficiency + a transfer-slope summary.

efficiency mirrors the verified scoring formula (arcagi3-scoring-and-transfer): per level
min(1.15, (A_h / A_m)^2), uncompleted level = 0. A_h is the human-baseline proxy (Exp-42
BFS-optimal length); A_m is the agent's environment-altering action count for that level.
"""
from __future__ import annotations


def efficiency(a_h: int, a_m: int | None) -> float:
    if a_m is None or a_m <= 0:
        return 0.0
    return min(1.15, (a_h / a_m) ** 2)


def transfer_slope(per_level_actions: list[int]) -> float:
    """Least-squares slope of action-cost vs level index. Negative = costs drop with depth."""
    n = len(per_level_actions)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(per_level_actions) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, per_level_actions))
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else num / den
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/bakeoff_metrics.py tests/test_bakeoff_metrics.py
git commit -q -m "feat(bakeoff): efficiency metric + transfer-slope for Phase 1"
```

---

## Task 2: A_h grader from the true model

**Files:**
- Modify: `src/arcagi3/bakeoff_metrics.py`
- Modify: `scripts/truemodel_planner.py` (expose a reusable per-level solver)
- Test: `tests/test_bakeoff_metrics.py`

- [ ] **Step 1: Refactor `truemodel_planner` to expose a function (no behavior change)**

In `scripts/truemodel_planner.py`, ensure `load_ls20_class()` and `bfs_solve(Ls20, start_level=0, max_nodes=...)` are importable (they already are module-level). Add at top of `bakeoff_metrics.py`:

```python
# append to src/arcagi3/bakeoff_metrics.py
import importlib.util
from pathlib import Path

_TRUEMODEL = Path("scripts/truemodel_planner.py")


def human_baseline_actions(level: int = 0, max_nodes: int = 500_000) -> int | None:
    """A_h proxy = BFS-optimal action count over ls20's TRUE model (grader-only source access)."""
    spec = importlib.util.spec_from_file_location("truemodel_planner", _TRUEMODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sol, *_ = mod.bfs_solve(mod.load_ls20_class(), start_level=level, max_nodes=max_nodes)
    return None if sol is None else len(sol)
```

- [ ] **Step 2: Write the test**

```python
# add to tests/test_bakeoff_metrics.py
import pytest

@pytest.mark.integration
def test_human_baseline_level1_is_optimal():
    from arcagi3.bakeoff_metrics import human_baseline_actions
    a_h = human_baseline_actions(level=0)
    assert a_h is not None and 1 <= a_h <= 30   # Exp-42 found 13
```

- [ ] **Step 3: Run test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_bakeoff_metrics.py::test_human_baseline_level1_is_optimal -v`
Expected: PASS (a_h == 13)

- [ ] **Step 4: Commit**

```bash
git add src/arcagi3/bakeoff_metrics.py tests/test_bakeoff_metrics.py scripts/truemodel_planner.py
git commit -q -m "feat(bakeoff): A_h grader from true-model BFS (per level)"
```

---

## Task 3: Bake-off harness skeleton + baselines (#2, #3)

**Files:**
- Create: `scripts/discovery_bakeoff.py`
- Test: covered by the smoke run in Step 3 (live env; no unit test asserts on stochastic counts)

- [ ] **Step 1: Write the harness**

```python
# scripts/discovery_bakeoff.py
"""Phase 1 — 3-way model-discovery bake-off on ls20 (black-box).

Drives each engine via the decide() contract against the offline ls20 env (capture_levelup
pattern), logging per-level cleared/actions and the (A_h/A_m)^2 efficiency. ls20 source is
touched ONLY by the A_h grader (bakeoff_metrics.human_baseline_actions).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py [budget]
"""
from __future__ import annotations
import logging, sys, time
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.bakeoff_metrics import efficiency, transfer_slope, human_baseline_actions
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.spatial_value_explorer import SpatialValueExplorer

logging.basicConfig(level=logging.ERROR)


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


@dataclass
class LevelResult:
    level: int
    cleared: bool
    actions: int
    a_h: int | None
    eff: float


@dataclass
class EngineResult:
    name: str
    levels_cleared: int
    levels: list[LevelResult] = field(default_factory=list)
    @property
    def slope(self) -> float:
        return transfer_slope([lr.actions for lr in self.levels if lr.cleared])


def make_ls20():
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("bo"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    return client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["bakeoff"]))


def run_engine(name: str, engine, budget: int, max_level: int = 4) -> EngineResult:
    """Drive `engine.decide(...)`; charge each environment-altering action to the current level."""
    env = make_ls20()
    obs = _retry(env.reset)
    res = EngineResult(name=name, levels_cleared=0)
    actions_this_level = 0
    cur_level = int(obs.levels_completed or 0)
    a_h_cache: dict[int, int | None] = {}
    n = 0
    while n < budget and cur_level < max_level:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        token = engine.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=cur_level,
            available=list(obs.available_actions or []),
        )
        if token[0] == "reset":
            obs = _retry(env.reset)
        elif token[0] == "S":
            obs = _retry(lambda: env.step(GameAction.from_id(token[1]))); actions_this_level += 1; n += 1
        else:  # "C"
            obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": token[1], "y": token[2]})); actions_this_level += 1; n += 1
        new_level = int(obs.levels_completed or 0)
        if new_level > cur_level:
            a_h = a_h_cache.setdefault(cur_level, human_baseline_actions(level=cur_level))
            res.levels.append(LevelResult(cur_level, True, actions_this_level, a_h,
                                          efficiency(a_h or 0, actions_this_level)))
            res.levels_cleared = new_level
            cur_level = new_level
            actions_this_level = 0
    return res


def print_table(results: list[EngineResult]) -> None:
    print(f"\n{'engine':<22}{'levels':<8}{'L1 act':<8}{'L1 eff':<8}{'slope':<8}", flush=True)
    for r in results:
        l1 = next((x for x in r.levels if x.level == 0), None)
        print(f"{r.name:<22}{r.levels_cleared:<8}"
              f"{(l1.actions if l1 else '-'):<8}{(round(l1.eff,3) if l1 else '-'):<8}"
              f"{round(r.slope,2):<8}", flush=True)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    engines = [
        ("salience(#2)", SalienceExplorer(seed=0)),
        ("spatial_value(#3)", SpatialValueExplorer(seed=0)),
    ]
    results = []
    for name, eng in engines:
        if hasattr(eng, "reset_all"):
            eng.reset_all()
        print(f"running {name}...", flush=True)
        results.append(run_engine(name, eng, budget))
    print_table(results)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run #2 at a tiny budget**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py 50`
Expected: prints a table with `salience(#2)` and `spatial_value(#3)` rows (likely 0 levels at budget 50 — we only assert it runs without error and produces the table).

- [ ] **Step 3: Commit**

```bash
git add scripts/discovery_bakeoff.py
git commit -q -m "feat(bakeoff): harness + #2/#3 baselines driven via decide()"
```

---

## Task 4: Agent attribute-state extraction (#1 — net-new core)

**Files:**
- Create: `src/arcagi3/attribute_state.py`
- Test: `tests/test_attribute_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attribute_state.py
import numpy as np
from arcagi3.attribute_state import agent_attributes, AttrVec

def _shape_grid(color, orient):
    # an L-tromino so rotation is detectable; placed in a 5x5 patch
    g = np.zeros((5, 5), dtype=np.int8)
    cells = {0: [(0,0),(1,0),(2,0),(2,1)], 1: [(0,0),(0,1),(0,2),(1,0)]}[orient]
    for r, c in cells:
        g[r, c] = color
    return g, [(r, c) for r, c in cells]

def test_attr_captures_color():
    g, cells = _shape_grid(9, 0)
    a = agent_attributes(g, cells)
    assert a.color == 9

def test_attr_distinguishes_rotation():
    g0, c0 = _shape_grid(9, 0)
    g1, c1 = _shape_grid(9, 1)
    assert agent_attributes(g0, c0).shape_sig != agent_attributes(g1, c1).shape_sig

def test_attr_equal_for_same_object():
    g, cells = _shape_grid(7, 0)
    assert agent_attributes(g, cells) == agent_attributes(g.copy(), cells)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_attribute_state.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/attribute_state.py
"""The agent's OWN mutable attributes (color, shape, orientation) as a hashable vector.

This is the piece the rigid-object stack lacks: in transform-match games the avatar's color/
shape/orientation change when it steps on transformer tiles. We capture them generically from
the agent's cell-set, normalising position out so only intrinsic attributes remain.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AttrVec:
    color: int
    shape_sig: tuple   # rotation-/translation-sensitive signature of the cell pattern


def agent_attributes(grid, agent_cells) -> AttrVec:
    cells = sorted((int(r), int(c)) for r, c in agent_cells)
    color = int(grid[cells[0][0], cells[0][1]]) if cells else -1
    r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
    # translation-normalised but NOT rotation-normalised -> orientation is encoded in the sig
    shape_sig = tuple((r - r0, c - c0) for r, c in cells)
    return AttrVec(color=color, shape_sig=shape_sig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_attribute_state.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/attribute_state.py tests/test_attribute_state.py
git commit -q -m "feat(discovery): agent attribute-state extraction (color/shape/orientation)"
```

---

## Task 5: `on_enter_cycle` primitive induction (#1)

**Files:**
- Create: `src/arcagi3/transform_induction.py`
- Test: `tests/test_transform_induction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transform_induction.py
import numpy as np
from arcagi3.transform_induction import induce_on_enter_cycles, OnEnterCycle

def test_induces_color_cycle_on_tile_contact():
    # Build 3 triples: each time the agent enters a cell that WAS color 4, its color advances.
    # Represent each observation as (agent_cells, agent_color_before, entered_cell_color, agent_color_after).
    triples = [
        {"entered_color": 4, "attr": "color", "before": 9, "after": 10},
        {"entered_color": 4, "attr": "color", "before": 10, "after": 11},
        {"entered_color": 7, "attr": "color", "before": 9, "after": 9},   # color-7 tile: no change
    ]
    rules = induce_on_enter_cycles(triples)
    assert any(r.tile_color == 4 and r.attribute == "color" for r in rules)
    assert all(r.tile_color != 7 for r in rules)  # non-effecting tiles are not rules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/transform_induction.py
"""Induce the transform-match grammar from interaction: which tile-color cycles which agent
attribute, and the level-completion predicate. General primitives only — no game constants.

A `triple` is a dict produced by the discovery loop on a step where the agent moved ONTO a
cell: {"entered_color": int, "attr": "color"|"shape", "before": <attr value>, "after": <attr value>}.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class OnEnterCycle:
    tile_color: int
    attribute: str          # "color" | "shape"
    order: tuple            # observed cycle order of attribute values


def induce_on_enter_cycles(triples: list[dict], min_support: int = 1) -> list[OnEnterCycle]:
    seq: dict[tuple, list] = defaultdict(list)
    changed: dict[tuple, int] = defaultdict(int)
    for t in triples:
        key = (t["entered_color"], t["attr"])
        if t["after"] != t["before"]:
            changed[key] += 1
            seq[key].extend([t["before"], t["after"]])
    rules = []
    for (tile_color, attr), n in changed.items():
        if n >= min_support:
            order = tuple(dict.fromkeys(seq[(tile_color, attr)]))  # de-dup, keep first-seen order
            rules.append(OnEnterCycle(tile_color=tile_color, attribute=attr, order=order))
    return rules
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/transform_induction.py tests/test_transform_induction.py
git commit -q -m "feat(discovery): induce on_enter_cycle transform primitive"
```

---

## Task 6: Attribute-match terminal predicate induction (#1)

**Files:**
- Modify: `src/arcagi3/transform_induction.py`
- Test: `tests/test_transform_induction.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_transform_induction.py
from arcagi3.transform_induction import induce_terminal, TerminalPredicate

def test_terminal_is_position_and_attr_match():
    # pre-win: agent at a slot position with attrs (color=9, shape_sig=S0); slot spec matches.
    prewin = {"agent_pos": (3, 5), "agent_attr": ("color9", "S0"),
              "slots": [{"pos": (3, 5), "attr": ("color9", "S0")}]}
    pred = induce_terminal(prewin)
    assert pred.kind == "attr_match_at_slot"
    # holds when agent on slot with matching attrs
    assert pred.holds(agent_pos=(3, 5), agent_attr=("color9", "S0"),
                      slots=[{"pos": (3, 5), "attr": ("color9", "S0"), "done": False}])
    # fails when attrs don't match
    assert not pred.holds(agent_pos=(3, 5), agent_attr=("color9", "S1"),
                          slots=[{"pos": (3, 5), "attr": ("color9", "S0"), "done": False}])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py::test_terminal_is_position_and_attr_match -v`
Expected: FAIL (`ImportError: cannot import name 'induce_terminal'`)

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/arcagi3/transform_induction.py
@dataclass(frozen=True)
class TerminalPredicate:
    kind: str = "attr_match_at_slot"

    def holds(self, agent_pos, agent_attr, slots) -> bool:
        return all(
            s["done"] or (agent_pos == s["pos"] and agent_attr == s["attr"])
            for s in slots
        ) and any(not s["done"] for s in slots) is False or all(
            (agent_pos == s["pos"] and agent_attr == s["attr"]) or s["done"] for s in slots
        )


def induce_terminal(prewin: dict) -> TerminalPredicate:
    """From the contrasted pre-win state, the completion predicate is: every slot is satisfied
    by the agent standing on it with matching attributes. (Spatial+attribute, not a global count
    — confirmed by levelup_contrastive.) The predicate FORM is fixed; its slot specs come live."""
    return TerminalPredicate(kind="attr_match_at_slot")
```

> Note: keep `holds` simple — a slot is satisfied iff already `done` or (agent on it ∧ attrs match); the level is terminal when all slots are satisfied. Simplify the boolean above to: `return all(s["done"] or (agent_pos == s["pos"] and agent_attr == s["attr"]) for s in slots)`.

- [ ] **Step 4: Simplify `holds` to the clean form and re-run**

Replace the body of `holds` with:

```python
        return all(s["done"] or (agent_pos == s["pos"] and agent_attr == s["attr"]) for s in slots)
```

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_transform_induction.py -v`
Expected: PASS (all transform_induction tests)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/transform_induction.py tests/test_transform_induction.py
git commit -q -m "feat(discovery): induce attr-match-at-slot terminal predicate"
```

---

## Task 7: Factored model + BFS planner (#1) — Exp-42 parity

**Files:**
- Create: `src/arcagi3/factored_model.py`
- Test: `tests/test_factored_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_factored_model.py
from arcagi3.factored_model import FactoredState, InducedModel, plan
from arcagi3.transform_induction import OnEnterCycle, TerminalPredicate

def _toy_model():
    # 1x5 corridor; cell at col 2 cycles "rot" attr through (0,1,2,3); slot at col 4 needs rot=0.
    # agent starts col 0 rot=3. Deltas: action 4 = +1 col (right), action 3 = -1 col (left).
    deltas = {3: (0, -1), 4: (0, 1)}
    walls = set()                      # open corridor
    tiles = {(0, 2): OnEnterCycle(tile_color=4, attribute="rot", order=(0, 1, 2, 3))}
    return InducedModel(deltas=deltas, walls=walls, tiles=tiles, width=5, height=1,
                        terminal=TerminalPredicate())

def test_plan_reaches_attr_match_slot():
    model = _toy_model()
    start = FactoredState(pos=(0, 0), attrs={"rot": 3}, completed=frozenset())
    slots = [{"pos": (0, 4), "attr_req": {"rot": 0}, "done": False}]
    actions = plan(model, start, slots, max_nodes=5000)
    assert actions is not None
    # must step on the cycler (to make rot 3->0 == one cycle) and arrive at col 4
    assert actions[-1] == 4
    assert 0 < len(actions) <= 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_factored_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/factored_model.py
"""Factored state + induced transition model + BFS planner. Mirrors Exp-42 (truemodel_planner)
but runs over the INDUCED model (deltas + on_enter_cycle tiles + walls + terminal predicate),
not the env source. State = (agent pos, attr dict, completed-slot set)."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FactoredState:
    pos: tuple
    attrs: tuple  # ((name, value), ...) sorted; dict accepted in ctor below
    completed: frozenset

    def __init__(self, pos, attrs, completed):
        object.__setattr__(self, "pos", tuple(pos))
        object.__setattr__(self, "attrs", tuple(sorted(attrs.items())) if isinstance(attrs, dict) else tuple(attrs))
        object.__setattr__(self, "completed", frozenset(completed))

    @property
    def attr_dict(self) -> dict:
        return dict(self.attrs)


@dataclass
class InducedModel:
    deltas: dict          # action_id -> (dr, dc)
    walls: set            # blocked (r, c)
    tiles: dict           # (r, c) -> OnEnterCycle
    width: int
    height: int
    terminal: object      # TerminalPredicate

    def step(self, state: FactoredState, action: int) -> FactoredState:
        dr, dc = self.deltas.get(action, (0, 0))
        nr, nc = state.pos[0] + dr, state.pos[1] + dc
        if not (0 <= nr < self.height and 0 <= nc < self.width) or (nr, nc) in self.walls:
            return state  # blocked -> no move
        attrs = state.attr_dict
        tile = self.tiles.get((nr, nc))
        if tile is not None:
            cur = attrs.get(tile.attribute)
            if cur in tile.order:
                i = tile.order.index(cur)
                attrs[tile.attribute] = tile.order[(i + 1) % len(tile.order)]
        return FactoredState(pos=(nr, nc), attrs=attrs, completed=state.completed)


def _satisfied(state: FactoredState, slots: list[dict]) -> frozenset:
    done = set(state.completed)
    for i, s in enumerate(slots):
        if i in done:
            continue
        if state.pos == s["pos"] and all(state.attr_dict.get(k) == v for k, v in s["attr_req"].items()):
            done.add(i)
    return frozenset(done)


def plan(model: InducedModel, start: FactoredState, slots: list[dict], max_nodes: int = 200_000):
    """BFS over the factored state; goal = all slots satisfied. Returns action list or None."""
    start = FactoredState(start.pos, start.attr_dict, _satisfied(start, slots))
    seen = {(start.pos, start.attrs, start.completed)}
    q = deque([(start, [])])
    expanded = 0
    while q and expanded < max_nodes:
        st, path = q.popleft()
        expanded += 1
        for a in model.deltas:
            nxt = model.step(st, a)
            nxt = FactoredState(nxt.pos, nxt.attr_dict, _satisfied(nxt, slots))
            if len(nxt.completed) == len(slots):
                return path + [a]
            k = (nxt.pos, nxt.attrs, nxt.completed)
            if k not in seen:
                seen.add(k)
                q.append((nxt, path + [a]))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_factored_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/factored_model.py tests/test_factored_model.py
git commit -q -m "feat(discovery): factored model + BFS planner (Exp-42 parity over induced model)"
```

---

## Task 8: `DiscoveryExplorer.decide()` — the integrated loop (#1)

**Files:**
- Create: `src/arcagi3/discovery_explorer.py`
- Test: `tests/test_discovery_explorer.py`

**Design (phase machine; one action per `decide()` call):**
1. `PROBE_MOVEMENT` — try each `available` simple action once from the start; use `movement.infer_all_translations(prev, cur, bg)` to fit `action->delta` and identify the agent (the consistently-moving small object). Mark non-moving directions against neighbours as candidate walls.
2. `PROBE_TRANSFORMS` — graph-scaffolded coverage (reuse `perception.state_hash` to avoid revisiting); on each move, if the agent's `AttrVec` changed, record a transform triple.
3. `INDUCE` — build `InducedModel` from deltas + `induce_on_enter_cycles(triples)` + walls; read candidate slots from `scene_graph.extract` target candidates; set `TerminalPredicate`.
4. `PLAN` — enumerate candidate slot attr-specs over the small attainable attribute set; call `factored_model.plan`; cache the action list.
5. `EXECUTE` — emit cached plan actions one per call; after each, `verify` predicted vs actual factored state.
6. `REFINE` — on misprediction (or plan exhausted without level-up), add the surprising observation to triples and return to `INDUCE`.
- On `levels` increment: keep the induced grammar (deltas, tiles, terminal form), flush only layout state (walls, slot positions, plan), re-enter `PROBE_TRANSFORMS` briefly to re-anchor positions (the transfer lever).

- [ ] **Step 1: Write the failing test (phase progression on a scripted stub env)**

```python
# tests/test_discovery_explorer.py
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
    # during PROBE_MOVEMENT it should try distinct simple actions, not repeat one
    assert len(seen) >= 2

def test_carries_grammar_across_level_increment():
    eng = DiscoveryExplorer(seed=0)
    eng.reset_all()
    eng._deltas = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # pretend movement learned
    eng._phase = "EXECUTE"
    eng.on_level_change(new_level=1)
    assert eng._deltas == {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # grammar kept
    assert eng._plan == [] and eng._walls == set()                        # layout flushed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `DiscoveryExplorer`**

Implement the class with the phase machine above. Key contract: `decide(grid, gstate_terminal, gstate_notplayed, levels, available) -> ("S",aid)|("C",x,y)|("reset",)`. Maintain `self._phase`, `self._deltas`, `self._walls`, `self._tiles`, `self._triples`, `self._plan`, `self._seen_hashes`, `self._last_level`, `self._prev_grid`, `self._prev_token`, `self._agent_color`. Reuse: `perception.connected_components/state_hash/detect_background`, `movement.infer_all_translations`, `attribute_state.agent_attributes`, `transform_induction.induce_on_enter_cycles/induce_terminal`, `factored_model.{FactoredState,InducedModel,plan}`, `scene_graph.extract`. Detect level change inside `decide` (compare `levels` to `self._last_level`) and call `self.on_level_change(levels)`.

Minimum to pass the two tests (full induction logic filled in during this step, but these two assertions pin the contract):

```python
# src/arcagi3/discovery_explorer.py  (skeleton — fill induction bodies per the Design above)
from __future__ import annotations
from arcagi3 import perception as P, movement as Mv, scene_graph as SG
from arcagi3.attribute_state import agent_attributes
from arcagi3.transform_induction import induce_on_enter_cycles, induce_terminal
from arcagi3.factored_model import FactoredState, InducedModel, plan


class DiscoveryExplorer:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset_all()

    def reset_all(self):
        self._phase = "PROBE_MOVEMENT"
        self._deltas, self._walls, self._tiles = {}, set(), {}
        self._triples, self._plan, self._seen_hashes = [], [], set()
        self._last_level = 0
        self._prev_grid = self._prev_token = self._agent_color = None
        self._probe_queue = []

    def on_level_change(self, new_level: int):
        # transfer lever: keep grammar (deltas/tiles/terminal), flush layout
        self._last_level = new_level
        self._walls = set(); self._plan = []; self._seen_hashes = set()
        self._phase = "PROBE_TRANSFORMS" if self._deltas else "PROBE_MOVEMENT"

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if levels != self._last_level:
            self.on_level_change(levels)
        if gstate_terminal:
            self._plan = []
            return ("reset",)
        tok = self._decide_inner(grid, available)
        self._prev_grid, self._prev_token = grid, tok
        return tok

    def _decide_inner(self, grid, available):
        if self._phase == "PROBE_MOVEMENT":
            if not self._probe_queue:
                self._probe_queue = [a for a in available if a in (1, 2, 3, 4)]
            if self._probe_queue:
                return ("S", self._probe_queue.pop(0))
            self._phase = "PROBE_TRANSFORMS"
        # PROBE_TRANSFORMS / INDUCE / PLAN / EXECUTE / REFINE: see Design.
        # ... (induction + planning bodies) ...
        if self._plan:
            return ("S", self._plan.pop(0))
        # fallback while learning: cycle simple actions deterministically
        return ("S", available[0] if available else 1)
```

Fill `PROBE_TRANSFORMS`→`REFINE` per the Design. After PROBE_MOVEMENT, on each call use `self._prev_grid`/`self._prev_token` to update `self._deltas` via `Mv.infer_all_translations` and to append transform triples when `agent_attributes` changed; once enough coverage, run INDUCE→PLAN and populate `self._plan`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/discovery_explorer.py tests/test_discovery_explorer.py
git commit -q -m "feat(discovery): DiscoveryExplorer decide() loop + cross-level transfer"
```

---

## Task 9: Run the bake-off (all 3) + regression guard

**Files:**
- Modify: `scripts/discovery_bakeoff.py` (register #1)

- [ ] **Step 1: Register `DiscoveryExplorer` in `main()`**

```python
# in scripts/discovery_bakeoff.py main(), add to engines list:
    from arcagi3.discovery_explorer import DiscoveryExplorer
    engines.append(("discovery(#1)", DiscoveryExplorer(seed=0)))
```

- [ ] **Step 2: Full run (empirical — this is the integration test)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/discovery_bakeoff.py 30000`
Expected: a table with all three engines. **Success bar:** `discovery(#1)` clears L1 in far fewer actions than `salience(#2)` (≪7961), with a non-trivial `L1 eff`, and a **negative transfer slope** across L1→L4. Record the exact numbers.

- [ ] **Step 3: Add a regression guard (skips if env unavailable)**

```python
# add to tests/test_discovery_explorer.py
import os, pytest

@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live ls20 bake-off; set RUN_BAKEOFF=1")
def test_discovery_beats_salience_on_l1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bo", "scripts/discovery_bakeoff.py")
    bo = importlib.util.module_from_spec(spec); spec.loader.exec_module(bo)
    from arcagi3.discovery_explorer import DiscoveryExplorer
    eng = DiscoveryExplorer(seed=0); eng.reset_all()
    res = bo.run_engine("discovery", eng, budget=5000, max_level=2)
    l1 = next((x for x in res.levels if x.level == 0 and x.cleared), None)
    assert l1 is not None and l1.actions < 2000   # vastly under salience's 7961
```

- [ ] **Step 4: Run the guard**

Run: `RUN_BAKEOFF=1 ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_discovery_explorer.py::test_discovery_beats_salience_on_l1 -v`
Expected: PASS (or, if #1 fails to discover, a clear failure that localizes the wall — record which phase).

- [ ] **Step 5: Commit**

```bash
git add scripts/discovery_bakeoff.py tests/test_discovery_explorer.py
git commit -q -m "feat(bakeoff): register discovery(#1) + L1 regression guard"
```

---

## Task 10: Write up Experiment 43

**Files:**
- Modify: `docs/experiment-overview.html`

- [ ] **Step 1: Append an Experiment-43 panel** with the recorded comparison table (engine × levels-cleared, L1 actions, L1 efficiency, transfer slope) and a one-paragraph verdict: did discover→plan→transfer beat the salience baseline and confirm "model discovery is the wall, planning is cheap"? State the result honestly even if #1 underperforms.

- [ ] **Step 2: Commit**

```bash
git add docs/experiment-overview.html
git commit -q -m "docs: Experiment 43 — model-discovery bake-off results on ls20"
```

---

## Self-Review

**Spec coverage:** harness + metrics (Tasks 1–3) ✓; engine #1 attribute layer (Task 4), `on_enter_cycle` + terminal predicate (Tasks 5–6), factored planner with Exp-42 parity (Task 7), integrated decide loop + transfer (Task 8) ✓; #2/#3 reuse (Task 3) ✓; bake-off run + decision rule + transfer test (Task 9) ✓; write-up (Task 10) ✓; strict-generality (no ls20 constants in #1; DSL minimal with one extension point in `transform_induction`) ✓; `A_h` proxy via true-model grader (Task 2) ✓.

**Placeholder scan:** Task 8 intentionally ships a skeleton whose induction bodies are filled in the same step — the two unit tests pin the public contract; the empirical behaviour is validated in Task 9 (the live run is the integration test). This is flagged, not a hidden TODO.

**Type consistency:** `decide()` returns `("S",aid)|("C",x,y)|("reset",)` everywhere; `OnEnterCycle(tile_color, attribute, order)`, `TerminalPredicate.holds(agent_pos, agent_attr, slots)`, `FactoredState(pos, attrs, completed)`, `InducedModel(deltas, walls, tiles, width, height, terminal)`, and `plan(model, start, slots, max_nodes)` are consistent across Tasks 5–9.
