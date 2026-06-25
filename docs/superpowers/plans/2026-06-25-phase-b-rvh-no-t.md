# Phase B RVH No-T Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure a default-off Reward-Value Hybrid (`RVH No-T`) that ranks real graph frontiers by learned reward potential without inventing transitions.

**Architecture:** Reuse the existing observed state graph in `world_model.py` and the banked `TransferExplorer` as the execution floor. First prove reward-edge density is sufficient, then train a tiny value head on frozen replay, then wire a default-off value-guided ordering wrapper that only reorders among empirically reachable frontier nodes.

**Tech Stack:** Python 3.12, project `.venv`, `pytest`, NumPy, existing ARC offline runner/harnesses.

## Global Constraints

- Default-off path must remain byte-identical to the current banked explorer.
- The learned layer may rank paths over real edges only; it may not invent transitions.
- No learned transition model in this phase.
- No Kaggle submission unless the real holdout gate shows a strict improvement with zero regressions.
- Fix root causes, not symptoms; every stage is measured before proceeding.

---

### Task 1: Build the reward-edge viability probe

**Files:**
- Create: `scripts/reward_edge_probe.py`
- Modify: `src/arcagi3/transfer_explorer.py`
- Test: `tests/test_reward_edge_probe.py`

**Interfaces:**
- Consumes: `TransferExplorer`, `WorldModel`, real game runner semantics
- Produces: `probe_game(game_id: str, budget: int, seed: int) -> dict`

- [ ] **Step 1: Write the failing test**

```python
from scripts.reward_edge_probe import summarize_reward_paths


def test_summarize_reward_paths_counts_positive_edges():
    graph = {
        b"root": {("S", 1): (b"a", 0.0), ("S", 2): (b"b", 1.0)},
        b"a": {("S", 3): (b"c", 0.0)},
        b"b": {},
        b"c": {},
    }
    summary = summarize_reward_paths(graph, root=b"root")
    assert summary["positive_edges"] == 1
    assert summary["reachable_positive_nodes"] == 1
    assert summary["shortest_reward_distance"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reward_edge_probe.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing function.

- [ ] **Step 3: Write minimal implementation**

```python
from collections import deque


def summarize_reward_paths(graph, root):
    positive = []
    for src, edges in graph.items():
        for _action, (dst, reward) in edges.items():
            if reward > 0:
                positive.append((src, dst))
    shortest = None
    seen = {root}
    q = deque([(root, 0)])
    positive_nodes = {dst for _src, dst in positive}
    while q:
        node, dist = q.popleft()
        if node in positive_nodes:
            shortest = dist
            break
        for _action, (dst, _reward) in graph.get(node, {}).items():
            if dst not in seen:
                seen.add(dst)
                q.append((dst, dist + 1))
    return {
        "positive_edges": len(positive),
        "reachable_positive_nodes": len(positive_nodes & seen) if shortest is not None else 0,
        "shortest_reward_distance": shortest,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reward_edge_probe.py -q`
Expected: PASS

- [ ] **Step 5: Implement live probe script**

```python
# scripts/reward_edge_probe.py
# - run TransferExplorer on selected games
# - collect wm.nodes after budget
# - print per-game:
#   states, positive_edges, shortest_reward_distance, rewarding_actions
```

- [ ] **Step 6: Run live probe**

Run: `PYTHONPATH=src .venv/bin/python scripts/reward_edge_probe.py 30000 ls20 tu93 vc33 re86`
Expected: per-game summary plus an aggregate reward-edge count.

- [ ] **Step 7: Commit**

```bash
git add tests/test_reward_edge_probe.py scripts/reward_edge_probe.py src/arcagi3/transfer_explorer.py
git commit -m "feat: add reward-edge viability probe"
```

### Task 2: Build the offline value separability probe

**Files:**
- Create: `src/arcagi3/value_model.py`
- Create: `scripts/value_probe.py`
- Test: `tests/test_value_model.py`

**Interfaces:**
- Consumes: replay rows of `(features, label)`
- Produces: `ValueModel.fit(rows)`, `ValueModel.score(features) -> float`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from arcagi3.value_model import ValueModel


def test_value_model_ranks_positive_above_negative():
    rows = [
        (np.array([1.0, 0.0, 0.0]), 1.0),
        (np.array([0.0, 1.0, 0.0]), 0.0),
        (np.array([1.0, 0.0, 1.0]), 1.0),
        (np.array([0.0, 1.0, 1.0]), 0.0),
    ]
    model = ValueModel(input_dim=3)
    model.fit(rows, epochs=50)
    assert model.score(np.array([1.0, 0.0, 0.0])) > model.score(np.array([0.0, 1.0, 0.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_value_model.py -q`
Expected: FAIL due to missing module/class.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np


class ValueModel:
    def __init__(self, input_dim: int, lr: float = 0.1):
        self.w = np.zeros(input_dim, dtype=float)
        self.b = 0.0
        self.lr = lr

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def score(self, features):
        return float(self._sigmoid(np.dot(self.w, features) + self.b))

    def fit(self, rows, epochs=20):
        for _ in range(epochs):
            for x, y in rows:
                p = self.score(x)
                err = p - y
                self.w -= self.lr * err * x
                self.b -= self.lr * err
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_value_model.py -q`
Expected: PASS

- [ ] **Step 5: Add a deterministic feature encoder**

```python
def encode_graph_state(node, depth, reward_seen):
    return np.array([
        float(node.visits),
        float(len(node.edges)),
        float(len(node.untried())),
        float(depth),
        float(reward_seen),
    ], dtype=float)
```

- [ ] **Step 6: Add the offline probe script**

Run: `PYTHONPATH=src .venv/bin/python scripts/value_probe.py replay.json`
Expected: prints separability metrics and whether positives rank above negatives.

- [ ] **Step 7: Commit**

```bash
git add src/arcagi3/value_model.py scripts/value_probe.py tests/test_value_model.py
git commit -m "feat: add offline value separability probe"
```

### Task 3: Add the default-off value-guided wrapper

**Files:**
- Create: `src/arcagi3/value_guided_explorer.py`
- Modify: `src/arcagi3/runner.py`
- Test: `tests/test_value_firewall.py`

**Interfaces:**
- Consumes: `TransferExplorer` behavior and `ValueModel`
- Produces: `ValueGuidedExplorer(seed=0, enable_value_guidance=False)`

- [ ] **Step 1: Write the failing firewall test**

```python
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.value_guided_explorer import ValueGuidedExplorer


def test_value_guided_is_byte_identical_when_disabled(sample_trace):
    base = TransferExplorer(seed=0)
    guided = ValueGuidedExplorer(seed=0, enable_value_guidance=False)
    assert sample_trace(base) == sample_trace(guided)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_value_firewall.py -q`
Expected: FAIL due to missing class.

- [ ] **Step 3: Implement the wrapper**

```python
class ValueGuidedExplorer(TransferExplorer):
    def __init__(self, *args, enable_value_guidance=False, **kwargs):
        self.enable_value_guidance = bool(enable_value_guidance)
        super().__init__(*args, **kwargs)

    def _reorder_frontier(self, choices):
        if not self.enable_value_guidance:
            return choices
        return sorted(choices, key=self._value_score, reverse=True)
```

- [ ] **Step 4: Wire a runner entry**

Run: add `value` agent mode in `/Users/ahmed/Documents/ArcAGI3/src/arcagi3/runner.py`
Expected: `--agent value` executes the wrapper without changing default behavior.

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_value_firewall.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/arcagi3/value_guided_explorer.py src/arcagi3/runner.py tests/test_value_firewall.py
git commit -m "feat: add default-off value-guided explorer"
```

### Task 4: Measure the real gate

**Files:**
- Create: `scripts/value_ab.py`
- Modify: `docs/experiment-overview.html`
- Test: `tests/test_value_ab_smoke.py`

**Interfaces:**
- Consumes: `TransferExplorer`, `ValueGuidedExplorer`, real game harness
- Produces: per-game comparison table and gate verdict

- [ ] **Step 1: Write the smoke test**

```python
from scripts.value_ab import summarize_rows


def test_summarize_rows_counts_improvements():
    rows = [
        ("g1", 1, 2),
        ("g2", 3, 3),
        ("g3", 2, 1),
    ]
    summary = summarize_rows(rows)
    assert summary["improved"] == 1
    assert summary["regressed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_value_ab_smoke.py -q`
Expected: FAIL due to missing script/function.

- [ ] **Step 3: Implement the A/B harness**

```python
def summarize_rows(rows):
    return {
        "improved": sum(1 for _g, base, test in rows if test > base),
        "regressed": sum(1 for _g, base, test in rows if test < base),
    }
```

- [ ] **Step 4: Run decisive real measurement**

Run: `PYTHONPATH=src .venv/bin/python scripts/value_ab.py 30000 tu93 vc33 ls20 re86 wa30`
Expected: explicit PASS/FAIL based on `improved >= 1 and regressed == 0`.

- [ ] **Step 5: Update experiment log**

```html
<!-- Add a new experiment row with:
     idea, setup, exact measured outcome, and kill/proceed verdict -->
```

- [ ] **Step 6: Commit**

```bash
git add scripts/value_ab.py tests/test_value_ab_smoke.py docs/experiment-overview.html
git commit -m "test: add value-guided real holdout gate"
```

### Task 5: Final verification checkpoint

**Files:**
- Modify: none required
- Test: full targeted verification run

**Interfaces:**
- Consumes: all prior tasks
- Produces: evidence bundle for proceed/kill decision

- [ ] **Step 1: Run targeted tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_reward_edge_probe.py tests/test_value_model.py tests/test_value_firewall.py tests/test_value_ab_smoke.py -q`
Expected: PASS

- [ ] **Step 2: Run existing firewalls**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_winning_firewall.py tests/test_mechanic_inference.py tests/test_mechanic_certificates.py -q`
Expected: PASS

- [ ] **Step 3: Run the real A/B gate**

Run: `PYTHONPATH=src .venv/bin/python scripts/value_ab.py 30000 tu93 vc33 ls20 re86 wa30`
Expected: explicit PASS/FAIL line.

- [ ] **Step 4: Decide**

```text
If PASS: candidate branch is ready for submission consideration.
If FAIL: kill RVH No-T immediately and fall back to the next Phase B candidate (OC-WM).
```

- [ ] **Step 5: Commit final measurement/doc updates**

```bash
git add docs/experiment-overview.html
git commit -m "docs: record RVH no-t gate outcome"
```
