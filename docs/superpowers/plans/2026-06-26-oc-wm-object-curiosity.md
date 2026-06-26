# OC-WM Object-Interaction Curiosity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a default-off, graph-only explorer whose reward-free object-interaction-novelty signal re-orders real frontier choices, and measure it against two early kill-gates before any A/B.

**Architecture:** Reuse the banked `TransferExplorer`/`SalienceExplorer` graph explorer and the existing object-centric perception. Add (1) a pure interaction-signature extractor, (2) a count-based curiosity scorer that overrides the existing coverage-safe `_pick_from_batch` action-choice hook, default-off. Kill early at Gate 0 (interaction coverage) and Gate 1 (a first reward edge on a wall game) before building the real A/B.

**Tech Stack:** Python 3.12, project `.venv`, `pytest`, NumPy, existing ARC offline runner/harnesses.

## Global Constraints

- Default-off path must remain byte-identical to the current banked explorer.
- The curiosity layer may re-order real untried actions only; it may not invent transitions.
- No learned transition model and no forward-model planning in this phase.
- No Kaggle submission unless the real holdout gate shows a strict improvement with zero regressions.
- Fix root causes, not symptoms; every stage is measured before proceeding. Do not build past a failed gate.

## Reference: existing interfaces (read before starting)

- `SalienceExplorer._pick_from_batch(self, choices, node)` (`src/arcagi3/salience_explorer.py:275`): returns one action from an equal-tier untried batch; default uniform random over `self.rng`. Docstring sanctions subclass reordering as coverage-safe.
- `SalienceExplorer.reset_all` (`:102`) initializes graph state; subclasses extend it.
- `TransferExplorer.decide` (`src/arcagi3/transfer_explorer.py:76`) reads `self._prev_grid` + `self.prev_action`, calls `super().decide(...)`, then sets `self._prev_grid = grid` at the end.
- `TransferExplorer._cell_to_sig(grid)` (`:57`) → `dict[(row, col) -> (color,)]` for every object cell. A click action is `("C", x, y)` with `x=col, y=row`; its cell is `(action[2], action[1])`.
- Action tokens: `("S", aid)` simple; `("C", x, y)` click; `("reset",)`.
- `self.bg` is the detected background color (int or None) after the first `decide`.

---

### Task 1: Interaction-signature extractor

**Files:**
- Create: `src/arcagi3/object_interaction.py`
- Test: `tests/test_object_interaction.py`

**Interfaces:**
- Produces: `interaction_signature(prev_grid, action, cur_grid, bg) -> tuple` (hashable). `("NONE",)` when nothing changed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_object_interaction.py
import numpy as np

from arcagi3.object_interaction import interaction_signature


def _grid(rows):
    return np.array(rows, dtype=np.int64)


def test_no_change_returns_none_signature():
    g = _grid([[0, 0], [0, 0]])
    sig = interaction_signature(g, ("S", 1), g.copy(), bg=0)
    assert sig == ("NONE",)


def test_object_appears_is_a_distinct_signature():
    prev = _grid([[0, 0], [0, 0]])
    cur = _grid([[0, 3], [0, 0]])  # color 3 appears
    sig = interaction_signature(prev, ("S", 2), cur, bg=0)
    assert sig == ("S", 2, "appeared", frozenset({3}))


def test_object_vanishes_under_click_uses_target_color():
    prev = _grid([[0, 5], [0, 0]])
    cur = _grid([[0, 0], [0, 0]])  # color 5 vanishes; click target is that cell (row0,col1)
    sig = interaction_signature(prev, ("C", 1, 0), cur, bg=0)  # x=col1, y=row0
    assert sig == ("C", 5, "vanished", frozenset({5}))


def test_recolor_is_distinct_from_appeared():
    prev = _grid([[2, 0]])
    cur = _grid([[4, 0]])  # 2 -> 4 (neither is bg)
    sig = interaction_signature(prev, ("S", 1), cur, bg=0)
    assert sig == ("S", 1, "recolored", frozenset({2, 4}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_object_interaction.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'arcagi3.object_interaction'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/object_interaction.py
"""Pure interaction-signature extractor for the OC-WM object-curiosity explorer.

Given two consecutive grids and the action taken between them, produce a hashable signature
describing the object interaction. Reward-free and deterministic. ("NONE",) means no object
change was attributable to the action.
"""

from __future__ import annotations

import numpy as np

NONE_SIGNATURE = ("NONE",)


def _action_descriptor(action, prev_grid, bg):
    """('S', aid) for simple actions; ('C', color_at_target) for clicks."""
    if action[0] == "S":
        return ("S", int(action[1]))
    # click ("C", x=col, y=row): the target color on the PREVIOUS grid
    x, y = int(action[1]), int(action[2])
    h, w = prev_grid.shape
    color = int(prev_grid[y, x]) if 0 <= y < h and 0 <= x < w else -1
    return ("C", color)


def _outcome(prev_colors, cur_colors, bg):
    """Classify the change at the changed cells (sets of colors before/after, bg excluded later)."""
    had_bg = bg in prev_colors if bg is not None else False
    has_bg = bg in cur_colors if bg is not None else False
    prev_obj = prev_colors - ({bg} if bg is not None else set())
    cur_obj = cur_colors - ({bg} if bg is not None else set())
    if not prev_obj and cur_obj:
        return "appeared"
    if prev_obj and not cur_obj:
        return "vanished"
    return "recolored"


def interaction_signature(prev_grid, action, cur_grid, bg):
    prev_grid = np.asarray(prev_grid)
    cur_grid = np.asarray(cur_grid)
    if prev_grid.shape != cur_grid.shape:
        return NONE_SIGNATURE
    changed = prev_grid != cur_grid
    if not changed.any():
        return NONE_SIGNATURE
    prev_colors = set(int(c) for c in np.unique(prev_grid[changed]))
    cur_colors = set(int(c) for c in np.unique(cur_grid[changed]))
    outcome = _outcome(prev_colors, cur_colors, bg)
    involved = frozenset(
        (prev_colors | cur_colors) - ({bg} if bg is not None else set())
    )
    desc = _action_descriptor(action, prev_grid, bg)
    return (desc[0], desc[1], outcome, involved)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_object_interaction.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/object_interaction.py tests/test_object_interaction.py
git commit -m "feat: add object-interaction signature extractor (oc-wm)"
```

---

### Task 2: The curiosity explorer (default-off) + firewall (Gate 2)

**Files:**
- Create: `src/arcagi3/object_curiosity_explorer.py`
- Modify: `src/arcagi3/runner.py` (after line 68, the `value` agent block)
- Test: `tests/test_curiosity_firewall.py`

**Interfaces:**
- Produces: `ObjectCuriosityExplorer(seed=0, enable_object_curiosity=True)`. With `enable_object_curiosity=False`, action trace is byte-identical to `TransferExplorer`.

**Scoring rule (define once, used in the test):**
- `descriptor_apps[d]` = times descriptor `d` was the action taken.
- `descriptor_sigs[d]` = set of non-NONE signatures `d` has produced.
- `sig_counts[s]` = global count of signature `s`.
- `score(d) = UNSEEN_BONUS` if `descriptor_apps[d] == 0`, else `max(1/(1+sig_counts[s]) for s in descriptor_sigs[d])`, or `0.0` if it only ever produced NONE.
- Unexplored descriptors are tried first; among explored, prefer the one that produced the rarest interaction.

- [ ] **Step 1: Write the failing firewall test**

```python
# tests/test_curiosity_firewall.py
import numpy as np

from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer


def _trace(policy, n=400):
    rng = np.random.default_rng(123)
    out = []
    levels = 0
    for i in range(n):
        grid = rng.integers(0, 4, size=(16, 16)).astype(np.int64)
        tok = policy.decide(
            grid=grid, gstate_terminal=False, gstate_notplayed=False,
            levels=levels, available=[1, 2, 3, 4, 5, 6],
        )
        out.append(tok)
    return out


def test_disabled_curiosity_is_byte_identical_to_transfer():
    base = _trace(TransferExplorer(seed=0))
    off = _trace(ObjectCuriosityExplorer(seed=0, enable_object_curiosity=False))
    assert base == off


def test_score_prefers_unseen_then_rarest():
    eng = ObjectCuriosityExplorer(seed=0)
    eng.reset_all()
    # descriptor A produced a rare sig (global count 1); B produced a common sig (count 9)
    sig_rare, sig_common = ("S", 1, "appeared", frozenset({3})), ("S", 2, "moved", frozenset({4}))
    eng.descriptor_apps[("S", 1)] = 1
    eng.descriptor_sigs[("S", 1)] = {sig_rare}
    eng.descriptor_apps[("S", 2)] = 1
    eng.descriptor_sigs[("S", 2)] = {sig_common}
    eng.sig_counts[sig_rare] = 1
    eng.sig_counts[sig_common] = 9
    assert eng._curiosity_score(("S", 1)) > eng._curiosity_score(("S", 2))
    assert eng._curiosity_score(("S", 9)) > eng._curiosity_score(("S", 1))  # unseen wins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_curiosity_firewall.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'arcagi3.object_curiosity_explorer'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/object_curiosity_explorer.py
"""ObjectCuriosityExplorer — graph-only, reward-free object-interaction curiosity.

Re-orders the explorer's equal-tier untried-action batch toward descriptors (clicked object-type
or simple-action id) with the rarest object-interaction history. Reward-free, so it is non-trivial
on the wall games (ls20/re86) where reward-based learning (RVH No-T) is mute. Graph-only: it only
re-orders real untried actions via the coverage-safe _pick_from_batch hook; it never invents an
edge. Firewall: enable_object_curiosity=False -> byte-identical to TransferExplorer.
"""

from __future__ import annotations

from collections import Counter

from .object_interaction import NONE_SIGNATURE, interaction_signature
from .transfer_explorer import TransferExplorer

UNSEEN_BONUS = 1e6


class ObjectCuriosityExplorer(TransferExplorer):
    def __init__(self, *args, enable_object_curiosity: bool = True, **kwargs) -> None:
        self.enable_object_curiosity = bool(enable_object_curiosity)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.descriptor_apps: Counter = Counter()
        self.descriptor_sigs: dict = {}
        self.sig_counts: Counter = Counter()
        self._cur_grid = None

    # --- signature accounting -------------------------------------------------
    def _descriptor_for(self, action, grid):
        if action[0] == "S":
            return ("S", int(action[1]))
        color = self._cell_to_sig(grid).get((action[2], action[1]))
        return ("C", color[0] if color is not None else -1)

    def _count_interaction(self, prev_grid, action, cur_grid):
        desc = self._descriptor_for(action, prev_grid)
        self.descriptor_apps[desc] += 1
        sig = interaction_signature(prev_grid, action, cur_grid, self.bg)
        if sig != NONE_SIGNATURE:
            self.sig_counts[sig] += 1
            self.descriptor_sigs.setdefault(desc, set()).add(sig)

    def _curiosity_score(self, desc):
        if self.descriptor_apps.get(desc, 0) == 0:
            return UNSEEN_BONUS
        sigs = self.descriptor_sigs.get(desc)
        if not sigs:
            return 0.0
        return max(1.0 / (1.0 + self.sig_counts.get(s, 0)) for s in sigs)

    # --- hooks ----------------------------------------------------------------
    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        if (self.enable_object_curiosity and self.prev_action is not None
                and not gstate_terminal and not gstate_notplayed
                and self._prev_grid is not None):
            self._count_interaction(self._prev_grid, self.prev_action, grid)
        self._cur_grid = grid
        return super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)

    def _pick_from_batch(self, choices, node):
        if (not self.enable_object_curiosity or self._cur_grid is None
                or not self.descriptor_apps):
            return super()._pick_from_batch(choices, node)
        scored = [(self._curiosity_score(self._descriptor_for(a, self._cur_grid)), a)
                  for a in choices]
        best = max(s for s, _a in scored)
        top = [a for s, a in scored if s == best]
        return top[int(self.rng.integers(0, len(top)))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_curiosity_firewall.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire a runner entry**

In `src/arcagi3/runner.py`, immediately after the `value` agent block (ends at line 68), add:

```python
    if agent_name == "curiosity":
        from .object_curiosity_explorer import ObjectCuriosityExplorer as _OCE
        return run_reactive(env, game_id, budget, seed,
                            policy_cls=lambda seed=seed: _OCE(seed=seed))
```

- [ ] **Step 6: Run full targeted tests + a local no-regression smoke (part of Gate 3)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_object_interaction.py tests/test_curiosity_firewall.py -q`
Expected: PASS (6 passed)

Run: `PYTHONPATH=src .venv/bin/python -m arcagi3.runner --agent curiosity --game push --budget 4000 --quiet`
Expected: prints a result line; `push` clears 3/3 (no sokoban regression). If push regresses, STOP and debug before continuing.

- [ ] **Step 7: Commit**

```bash
git add src/arcagi3/object_curiosity_explorer.py src/arcagi3/runner.py tests/test_curiosity_firewall.py
git commit -m "feat: add default-off object-curiosity explorer (oc-wm)"
```

---

### Task 3: Interaction-coverage probe — Gate 0

**Files:**
- Create: `scripts/interaction_coverage_probe.py`
- Test: `tests/test_interaction_coverage.py`

**Interfaces:**
- Produces: `coverage(game_id, budget, enable_curiosity, seed) -> dict` with `distinct_signatures`, `levels`, `actions`.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_interaction_coverage.py
from scripts.interaction_coverage_probe import compare_rows


def test_compare_rows_flags_more_coverage():
    rows = [("ls20", 5, 9), ("re86", 4, 4)]  # (game, base_distinct, curiosity_distinct)
    summary = compare_rows(rows)
    assert summary["more_coverage"] == 1   # ls20 only
    assert summary["worse_coverage"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_interaction_coverage.py -q`
Expected: FAIL (missing module/function).

- [ ] **Step 3: Implement the probe**

```python
# scripts/interaction_coverage_probe.py
"""Gate 0: does object-curiosity reach MORE distinct interaction signatures than baseline?"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer
from arcagi3.object_interaction import NONE_SIGNATURE, interaction_signature
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game

logging.basicConfig(level=logging.ERROR)


def compare_rows(rows):
    return {
        "more_coverage": sum(1 for _g, base, cur in rows if cur > base),
        "worse_coverage": sum(1 for _g, base, cur in rows if cur < base),
    }


def coverage(game_id: str, budget: int, enable_curiosity: bool, seed: int = 0) -> dict:
    eng = ObjectCuriosityExplorer(seed=seed, enable_object_curiosity=enable_curiosity)
    eng.reset_all()
    env = make_game(game_id)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    distinct = set()
    prev_grid = None
    prev_tok = None
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        # the transition prev_grid -> grid was caused by prev_tok (taken last iteration)
        if prev_grid is not None and prev_tok is not None and prev_tok[0] in ("S", "C"):
            sig = interaction_signature(prev_grid, prev_tok, grid, eng.bg)
            if sig != NONE_SIGNATURE:
                distinct.add(sig)
        tok = eng.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level, available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset(); prev_grid = None; prev_tok = None
        elif tok[0] == "S":
            prev_grid = grid; prev_tok = tok
            obs = env.step(GameAction.from_id(tok[1])); actions += 1
        else:
            prev_grid = grid; prev_tok = tok
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]}); actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    return {"game": game_id, "levels": level, "actions": actions,
            "distinct_signatures": len(distinct)}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["ls20", "re86"]
    rows = []
    for game in games:
        base = coverage(game, budget, enable_curiosity=False)
        cur = coverage(game, budget, enable_curiosity=True)
        rows.append((game, base["distinct_signatures"], cur["distinct_signatures"]))
        print(f"[{game:8}] base_distinct={base['distinct_signatures']} "
              f"curiosity_distinct={cur['distinct_signatures']} "
              f"(levels {base['levels']}->{cur['levels']})", flush=True)
    summary = compare_rows(rows)
    print("-" * 80, flush=True)
    print(f"more_coverage={summary['more_coverage']} worse_coverage={summary['worse_coverage']}",
          flush=True)
    print("GATE 0: " + ("PASS" if summary["more_coverage"] >= 1 else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_interaction_coverage.py -q`
Expected: PASS

- [ ] **Step 5: Run Gate 0 (decisive)**

Run: `PYTHONPATH=src .venv/bin/python scripts/interaction_coverage_probe.py 3000 ls20 re86`
Expected: a `GATE 0: PASS` or `FAIL` line.
Record the exact numbers. **Decision:** if `GATE 0: FAIL` (curiosity does not reach more distinct interactions than baseline on either wall game), stop here — go to Task 5 and record the kill. Do not build the A/B.

- [ ] **Step 6: Commit**

```bash
git add scripts/interaction_coverage_probe.py tests/test_interaction_coverage.py
git commit -m "feat: add interaction-coverage probe (oc-wm gate 0)"
```

---

### Task 4: Reward-edge starvation kill-test — Gate 1

**Files:**
- Create: `scripts/curiosity_reward_edge_probe.py`
- Test: `tests/test_curiosity_reward_edge.py`

**Interfaces:**
- Reuses `summarize_reward_paths` from `scripts/reward_edge_probe.py`.
- Produces: per-game `(positive_edges_base, positive_edges_curiosity)` and a PASS/FAIL.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_curiosity_reward_edge.py
from scripts.curiosity_reward_edge_probe import gate1_summary


def test_gate1_passes_when_curiosity_unlocks_a_wall_game():
    rows = [("ls20", 0, 1), ("re86", 0, 0)]  # (game, base_pos_edges, curiosity_pos_edges)
    summary = gate1_summary(rows)
    assert summary["unlocked"] == 1
    assert summary["pass"] is True


def test_gate1_fails_when_both_stay_zero():
    rows = [("ls20", 0, 0), ("re86", 0, 0)]
    summary = gate1_summary(rows)
    assert summary["unlocked"] == 0
    assert summary["pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_curiosity_reward_edge.py -q`
Expected: FAIL (missing module/function).

- [ ] **Step 3: Implement the probe**

```python
# scripts/curiosity_reward_edge_probe.py
"""Gate 1 (decisive): does reward-free curiosity stumble a first reward edge on a wall game
where the baseline produces ZERO? If both wall games stay at zero, the bet has failed."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game
from scripts.reward_edge_probe import summarize_reward_paths

logging.basicConfig(level=logging.ERROR)


def gate1_summary(rows):
    unlocked = sum(1 for _g, base, cur in rows if base == 0 and cur > 0)
    return {"unlocked": unlocked, "pass": unlocked >= 1}


def positive_edges(game_id: str, budget: int, enable_curiosity: bool, seed: int = 0) -> int:
    eng = ObjectCuriosityExplorer(seed=seed, enable_object_curiosity=enable_curiosity)
    eng.reset_all()
    env = make_game(game_id)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = eng.decide(
            grid=grid,
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level, available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1])); actions += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]}); actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    summary = summarize_reward_paths(eng, eng.root_key)
    return int(summary["positive_edges"])


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["ls20", "re86"]
    rows = []
    for game in games:
        base = positive_edges(game, budget, enable_curiosity=False)
        cur = positive_edges(game, budget, enable_curiosity=True)
        rows.append((game, base, cur))
        print(f"[{game:8}] base_pos_edges={base} curiosity_pos_edges={cur}", flush=True)
    summary = gate1_summary(rows)
    print("-" * 80, flush=True)
    print(f"unlocked={summary['unlocked']}", flush=True)
    print("GATE 1: " + ("PASS" if summary["pass"] else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_curiosity_reward_edge.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Run Gate 1 (the decisive kill-test)**

Run: `PYTHONPATH=src .venv/bin/python scripts/curiosity_reward_edge_probe.py 3000 ls20 re86`
Expected: a `GATE 1: PASS` or `FAIL` line.
Record the exact numbers. **Decision:** if `GATE 1: FAIL` (both wall games stay at zero positive edges), stop here — go to Task 5 and record the kill. Do not build the A/B and do not escalate to the forward-model substrate.

- [ ] **Step 6: Commit**

```bash
git add scripts/curiosity_reward_edge_probe.py tests/test_curiosity_reward_edge.py
git commit -m "feat: add curiosity reward-edge kill-test (oc-wm gate 1)"
```

---

### Task 5: Real A/B gate (Gate 4) and experiment record

**Files:**
- Create: `scripts/curiosity_ab.py`
- Modify: `docs/experiment-overview.html`
- Test: `tests/test_curiosity_ab_smoke.py`

> Only build the A/B harness (Steps 1–4) if Gates 0 and 1 both passed. Steps 5–6 (record + decide) run unconditionally — a kill is recorded too.

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_curiosity_ab_smoke.py
from scripts.curiosity_ab import summarize_rows


def test_summarize_rows_counts_improvements():
    rows = [("g1", 1, 2), ("g2", 3, 3), ("g3", 2, 1)]
    summary = summarize_rows(rows)
    assert summary["improved"] == 1
    assert summary["regressed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_curiosity_ab_smoke.py -q`
Expected: FAIL (missing module/function).

- [ ] **Step 3: Implement the A/B harness**

```python
# scripts/curiosity_ab.py
"""Gate 4: TransferExplorer vs ObjectCuriosityExplorer on real holdout games."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcagi3 import perception as P
from arcagi3.object_curiosity_explorer import ObjectCuriosityExplorer
from arcagi3.transfer_explorer import TransferExplorer
from arcengine import GameAction, GameState

from scripts.discovery_bakeoff import make_game

logging.basicConfig(level=logging.ERROR)


def summarize_rows(rows):
    return {
        "improved": sum(1 for _g, base, test in rows if test > base),
        "regressed": sum(1 for _g, base, test in rows if test < base),
    }


def run_policy(policy, game: str, budget: int):
    env = make_game(game)
    obs = env.reset()
    level = int(obs.levels_completed or 0)
    actions = 0
    while actions < budget:
        if obs is None or obs.state == GameState.WIN:
            break
        tok = policy.decide(
            P.to_grid(obs.frame),
            gstate_terminal=(obs.state == GameState.GAME_OVER),
            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
            levels=level, available=list(obs.available_actions or []),
        )
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1])); actions += 1
        else:
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]}); actions += 1
        if obs is None:
            break
        level = int(obs.levels_completed or 0)
    return level


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2:] if len(sys.argv) > 2 else ["tu93", "vc33", "ls20", "re86", "wa30"]
    rows = []
    for game in games:
        base = run_policy(TransferExplorer(seed=0), game, budget)
        test = run_policy(ObjectCuriosityExplorer(seed=0), game, budget)
        rows.append((game, int(base), int(test)))
        print(f"[{game:8}] transfer={base} curiosity={test}", flush=True)
    summary = summarize_rows(rows)
    print("-" * 80, flush=True)
    print(f"improved={summary['improved']} regressed={summary['regressed']}", flush=True)
    print("GO-RULE: "
          + ("PASS" if summary["improved"] >= 1 and summary["regressed"] == 0 else "FAIL"),
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_curiosity_ab_smoke.py -q`
Expected: PASS

- [ ] **Step 5: Run the real A/B gate (only if Gates 0+1 passed)**

Run: `PYTHONPATH=src .venv/bin/python scripts/curiosity_ab.py 3000 tu93 vc33 ls20 re86`
Expected: a `GO-RULE: PASS` or `FAIL` line. Record exact per-game numbers.

- [ ] **Step 6: Record the outcome in the experiment log and decide**

Add an `Exp 55` panel to `docs/experiment-overview.html` immediately before the existing
`<div class="panel kill">` whose `<h3>` is `Campaign conclusion — consolidate &amp; ship`, mirroring
the Exp 54 panel's structure (a `<div class="panel kill">` with `<h3>`, a setup `<p>`, a results
`<table>`, and a `<b>Finding.</b>` paragraph). State the measured Gate 0 / Gate 1 / Gate 4 numbers
and the verdict.

Then:

```bash
git add docs/experiment-overview.html scripts/curiosity_ab.py tests/test_curiosity_ab_smoke.py
git commit -m "docs: record OC-WM object-curiosity gate outcome (Exp 55)"
```

**Decision rule:**
- If `GO-RULE: PASS` (≥1 strict improvement, zero regressions): the candidate is ready for submission consideration — stop and report to the user; do not change submission defaults without approval.
- If any gate FAILED: OC-WM is killed. Report the negative to the user and return to the standing recommendation (consolidate & ship). Do not escalate to the forward-model substrate.

---

### Task 6: Final verification checkpoint

**Files:**
- Modify: none required

- [ ] **Step 1: Run all OC-WM targeted tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_object_interaction.py tests/test_curiosity_firewall.py tests/test_interaction_coverage.py tests/test_curiosity_reward_edge.py tests/test_curiosity_ab_smoke.py -q`
Expected: PASS (all)

- [ ] **Step 2: Run the existing firewalls (no-regression on the banked path)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_value_firewall.py tests/test_winning_firewall.py -q`
Expected: PASS

- [ ] **Step 3: Confirm submission default is untouched**

Run: `grep -n "TransferExplorer" submission/my_agent.py`
Expected: the default policy is still `TransferExplorer` — OC-WM is NOT wired into submission.

- [ ] **Step 4: Report the evidence bundle to the user** (gate numbers + verdict). No commit needed if nothing changed.
