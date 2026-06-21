# Phase 0a Prune-Capability Oracle Probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an API-free analysis (`prune_analysis.py`) + a live driver (`prune_oracle.py`) that, from banked-v6 trajectories, decides per game whether exploration prune is (Tier-1) possible and (Tier-2) learnable — the kill-gate before any generator/model.

**Architecture:** Pure numpy analysis in `src/arcagi3/prune_analysis.py` (unit-tested on synthetic traces); a thin live-API capture/orchestration driver in `scripts/prune_oracle.py` that runs `SalienceExplorer(trust_threshold=3, border_mask=2)` (exact v6 config) on real public games, reconstructs the level graph from a per-step trace, and emits a per-game ceiling/AUC table + JSON. No agent code is modified; the 0.33 floor is untouched.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), numpy, pytest, `arc_agi`/`arcengine` (live games via `OperationMode.NORMAL`), `arcagi3.perception`.

**Spec:** `docs/superpowers/specs/2026-06-21-phase0-prune-oracle-probe-design.md`

**Run prefix for all commands:** `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python ...`

---

## File Structure

- Create: `src/arcagi3/prune_analysis.py` — `Step`, `LevelSeg`, `build_edges`, `segment_levels`, `shortest_path`, `ceilings`, `state_features`, `roc_auc`, `logistic_fit`/`logistic_score`, `logo_auc`, `shuffle_auc`. Pure; numpy + `perception` only.
- Create: `tests/test_prune_analysis.py` — unit tests on synthetic traces/grids (no API).
- Create: `scripts/prune_oracle.py` — live capture + high-budget completion + orchestration + table/JSON.
- Modify: none.

---

## Task 1: Trace model + edge reconstruction

**Files:**
- Create: `src/arcagi3/prune_analysis.py`
- Test: `tests/test_prune_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prune_analysis.py
import numpy as np
from arcagi3 import prune_analysis as A


def _steps():
    # level 0: detour a->d->a, then path a->b->c; c is the level-up (reward) state.
    # fields: idx, level, from_key, action, reward, tier
    return [
        A.Step(0, 0, b"a", ("S", 4), 0.0, 0),
        A.Step(1, 0, b"d", ("S", 1), 0.0, 9),
        A.Step(2, 0, b"a", ("S", 1), 0.0, 0),
        A.Step(3, 0, b"b", ("S", 2), 0.0, 0),
        A.Step(4, 0, b"c", ("S", 3), 1.0, 0),   # reward -> level up
        A.Step(5, 1, b"z", ("S", 1), 0.0, 0),   # first state of level 1
    ]


def test_build_edges_excludes_reset_and_levelup():
    edges, first_seen = A.build_edges(_steps())
    assert set(edges[b"a"]) == {(("S", 4), b"d"), (("S", 1), b"b")}
    assert edges[b"b"] == [(("S", 2), b"c")]
    assert b"c" not in edges  # step 4 is a reward edge -> excluded
    assert first_seen[b"a"] == 0 and first_seen[b"d"] == 1
    assert first_seen[b"b"] == 3 and first_seen[b"c"] == 4
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py::test_build_edges_excludes_reset_and_levelup -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arcagi3.prune_analysis'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/arcagi3/prune_analysis.py
"""Pure, API-free analysis for the Phase 0a prune-capability oracle probe.

Consumes a captured banked-v6 trajectory (list[Step]) and answers, per completed level:
  Tier-1 (ceilings): how much exploration was OFF the shortest path to the level-up edge.
  Tier-2 (logo_auc): are on-path vs off-path states separable by decision-time features.
numpy only; no live API, no torch. See docs/superpowers/specs/2026-06-21-phase0-prune-oracle-probe-design.md
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from . import perception as P

RESET = ("reset",)


@dataclass
class Step:
    idx: int          # action index
    level: int        # levels_completed BEFORE this action
    from_key: bytes   # state acted from (pol.prev_key after decide)
    action: tuple     # ("S", id) | ("C", x, y) | ("reset",)
    reward: float     # levels_after - levels_before for this action
    tier: int         # salience tier of `action` at from_key


def build_edges(steps):
    """Global directed edges from consecutive steps, EXCLUDING reset actions and
    level-up (reward > 0) transitions (those cross into the next level/board).

    Returns (edges, first_seen):
      edges: dict[from_key] -> list[(action, to_key)]   (to_key = next step's from_key)
      first_seen: dict[key] -> idx of first step where it appears as from_key.
    """
    edges = defaultdict(list)
    first_seen = {}
    for i, s in enumerate(steps):
        if s.from_key not in first_seen:
            first_seen[s.from_key] = s.idx
        if i + 1 < len(steps) and s.action != RESET and s.reward <= 0:
            edges[s.from_key].append((s.action, steps[i + 1].from_key))
    return dict(edges), first_seen
```

- [ ] **Step 4: Run it to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py::test_build_edges_excludes_reset_and_levelup -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/prune_analysis.py tests/test_prune_analysis.py
git commit -q -m "feat(phase0): trace model + edge reconstruction for prune oracle

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 2: Level segmentation + shortest path + Tier-1 ceilings

**Files:**
- Modify: `src/arcagi3/prune_analysis.py`
- Test: `tests/test_prune_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_prune_analysis.py
def test_segment_and_ceilings():
    steps = _steps()
    edges, first_seen = A.build_edges(steps)
    segs = A.segment_levels(steps, first_seen)
    assert set(segs) == {0}                       # only level 0 ends in a reward here
    seg = segs[0]
    assert seg.start_key == b"a" and seg.target_key == b"c"
    assert seg.member_keys == {b"a", b"b", b"c", b"d"}
    assert seg.actual_actions == 4                # end_idx 4 - start_idx 0

    path = A.shortest_path(edges, seg.start_key, seg.target_key, seg.member_keys)
    assert path == [b"a", b"b", b"c"]

    c = A.ceilings(seg, path)
    assert c["reachable"] is True
    assert c["discovered_states"] == 4 and c["path_states"] == 3
    assert c["ceiling_states"] == 0.25
    assert c["path_actions"] == 2 and c["ceiling_actions"] == 0.5


def test_shortest_path_unreachable_returns_none():
    edges = {b"a": [(("S", 1), b"b")]}
    assert A.shortest_path(edges, b"a", b"x", {b"a", b"b"}) is None
    c = A.ceilings(A.LevelSeg(0, b"a", b"x", {b"a", b"b"}, 0, 3), None)
    assert c["reachable"] is False and c["ceiling_actions"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py -k "segment or unreachable" -v`
Expected: FAIL — `AttributeError: module 'arcagi3.prune_analysis' has no attribute 'segment_levels'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/arcagi3/prune_analysis.py
@dataclass
class LevelSeg:
    level: int
    start_key: bytes
    target_key: bytes      # from_key of the reward (level-up) step
    member_keys: set       # keys first-seen within [start_idx, end_idx]
    start_idx: int
    end_idx: int           # the reward step idx

    @property
    def actual_actions(self):
        return self.end_idx - self.start_idx


def segment_levels(steps, first_seen):
    """One LevelSeg per level that ENDS in a reward step (a completed level)."""
    level_first_idx = {}
    reward_step = {}
    for s in steps:
        if s.level not in level_first_idx:
            level_first_idx[s.level] = s.idx
        if s.reward > 0 and s.level not in reward_step:
            reward_step[s.level] = s
    by_idx = {s.idx: s for s in steps}
    segs = {}
    for lvl, rs in reward_step.items():
        if lvl not in level_first_idx:
            continue
        start_idx, end_idx = level_first_idx[lvl], rs.idx
        members = {k for k, fi in first_seen.items() if start_idx <= fi <= end_idx}
        segs[lvl] = LevelSeg(lvl, by_idx[start_idx].from_key, rs.from_key,
                             members, start_idx, end_idx)
    return segs


def shortest_path(edges, start, target, allowed):
    """BFS over `edges` restricted to nodes in `allowed`. Returns [start..target] or None."""
    if start == target:
        return [start]
    if start not in allowed or target not in allowed:
        return None
    seen = {start}
    q = deque([(start, [start])])
    while q:
        k, path = q.popleft()
        for _a, nk in edges.get(k, []):
            if nk in seen or nk not in allowed:
                continue
            seen.add(nk)
            if nk == target:
                return path + [nk]
            q.append((nk, path + [nk]))
    return None


def ceilings(seg, path):
    """Tier-1 numbers for one level. path is None when target is unreachable in-level."""
    disc = len(seg.member_keys)
    if path is None:
        return {"discovered_states": disc, "path_states": None, "ceiling_states": None,
                "actual_actions": seg.actual_actions, "path_actions": None,
                "ceiling_actions": None, "reachable": False}
    ps, pa = len(path), len(path) - 1
    return {"discovered_states": disc, "path_states": ps,
            "ceiling_states": round(1 - ps / max(disc, 1), 3),
            "actual_actions": seg.actual_actions, "path_actions": pa,
            "ceiling_actions": round(1 - pa / max(seg.actual_actions, 1), 3),
            "reachable": True}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py -k "segment or unreachable" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/prune_analysis.py tests/test_prune_analysis.py
git commit -q -m "feat(phase0): level segmentation + shortest-path + Tier-1 ceilings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 3: Decision-time state features

**Files:**
- Modify: `src/arcagi3/prune_analysis.py`
- Test: `tests/test_prune_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_prune_analysis.py
def test_state_features():
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[0:2, 0:2] = 3          # one 4-cell object, color 3
    grid[10, 10] = 5            # one 1-cell object, color 5
    f = A.state_features(grid, background=0, discovery_tier=7, parent_colors={3})
    assert f["n_objects"] == 2.0
    assert f["n_small"] == 2.0           # both objects size <= 4
    assert f["max_obj_size"] == 4.0
    assert f["n_distinct_colors"] == 2.0  # {3,5}
    assert f["discovery_tier"] == 7.0
    assert f["n_new_colors"] == 1.0       # 5 is new vs parent {3}
    assert list(A.FEATURE_ORDER)  # non-empty, stable order
    assert len(A.feature_vector(f)) == len(A.FEATURE_ORDER)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py::test_state_features -v`
Expected: FAIL — `AttributeError: ... 'state_features'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/arcagi3/prune_analysis.py
FEATURE_ORDER = ["n_objects", "n_small", "median_obj_size", "max_obj_size",
                 "n_distinct_colors", "board_fill", "discovery_tier", "n_new_colors"]


def state_features(grid, background, discovery_tier, parent_colors):
    """Features computable ONLY from what is observable at the state's discovery time."""
    objs = P.connected_components(grid, background=background)
    sizes = np.array([o.size for o in objs]) if objs else np.array([0])
    colors = {int(c) for c in np.unique(grid)} - {int(background)}
    n_new = len(colors - set(parent_colors)) if parent_colors is not None else 0
    return {
        "n_objects": float(len(objs)),
        "n_small": float(sum(1 for o in objs if o.size <= 4)),
        "median_obj_size": float(np.median(sizes)),
        "max_obj_size": float(sizes.max()),
        "n_distinct_colors": float(len(colors)),
        "board_fill": float((grid != background).mean()),
        "discovery_tier": float(discovery_tier),
        "n_new_colors": float(n_new),
    }


def feature_vector(f):
    return np.array([f[k] for k in FEATURE_ORDER], dtype=float)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py::test_state_features -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/prune_analysis.py tests/test_prune_analysis.py
git commit -q -m "feat(phase0): decision-time state features for Tier-2

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 4: AUC + logistic + leave-one-game-out + shuffle control

**Files:**
- Modify: `src/arcagi3/prune_analysis.py`
- Test: `tests/test_prune_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_prune_analysis.py
def test_roc_auc_basic():
    assert A.roc_auc([0.1, 0.9], [0, 1]) == 1.0
    assert A.roc_auc([0.9, 0.1], [0, 1]) == 0.0
    assert A.roc_auc([0.5, 0.5], [0, 1]) == 0.5      # ties -> 0.5


def test_logo_auc_separable_vs_shuffle():
    rng = np.random.default_rng(0)
    per_game = {}
    for g in ("g1", "g2", "g3"):
        # feature 0 separates: on-path high, off-path low; other features noise.
        Xpos = np.column_stack([rng.normal(3, 0.3, 40), rng.normal(0, 1, 40)])
        Xneg = np.column_stack([rng.normal(0, 0.3, 60), rng.normal(0, 1, 60)])
        X = np.vstack([Xpos, Xneg])
        y = np.array([1] * 40 + [0] * 60)
        per_game[g] = (X, y)
    auc = A.logo_auc(per_game)
    sh = A.shuffle_auc(per_game, seed=0)
    assert auc >= 0.9                 # held-out separability is real
    assert abs(sh - 0.5) < 0.15       # shuffle control collapses to chance
```

- [ ] **Step 2: Run it to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py -k "roc_auc or logo" -v`
Expected: FAIL — `AttributeError: ... 'roc_auc'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/arcagi3/prune_analysis.py
def roc_auc(scores, labels):
    """AUC via the rank (Mann-Whitney U) statistic, with average ranks for ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank for the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def logistic_fit(X, y, iters=800, lr=0.2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        w -= lr * Xs.T @ (p - y) / len(y)
    return w, mu, sd


def logistic_score(X, w, mu, sd):
    Xs = np.hstack([(np.asarray(X, dtype=float) - mu) / sd, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Xs @ w))


def logo_auc(per_game):
    """Leave-one-GAME-out: fit on all-but-one game, score held-out, pool, AUC.

    per_game: dict[game] -> (X [n,F] float array, y [n] {0,1} array). Measures TRANSFER
    (does the on-path signal generalize to an unseen game), not per-game memorization.
    """
    games = list(per_game)
    pooled_s, pooled_y = [], []
    for held in games:
        tr = [g for g in games if g != held]
        if not tr:
            continue
        Xtr = np.vstack([per_game[g][0] for g in tr])
        ytr = np.concatenate([per_game[g][1] for g in tr])
        if ytr.sum() == 0 or (ytr == 0).sum() == 0:
            continue
        w, mu, sd = logistic_fit(Xtr, ytr)
        pooled_s.append(logistic_score(per_game[held][0], w, mu, sd))
        pooled_y.append(per_game[held][1])
    if not pooled_s:
        return float("nan")
    return roc_auc(np.concatenate(pooled_s), np.concatenate(pooled_y))


def shuffle_auc(per_game, seed=0):
    """Label-permutation control: same features, labels shuffled WITHIN each game."""
    rng = np.random.default_rng(seed)
    shuffled = {g: (X, rng.permutation(y)) for g, (X, y) in per_game.items()}
    return logo_auc(shuffled)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py -k "roc_auc or logo" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full new test module + the existing suite (no regressions)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_prune_analysis.py -v && ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: new module all PASS; full suite still green (no agent code changed).

- [ ] **Step 6: Commit**

```bash
git add src/arcagi3/prune_analysis.py tests/test_prune_analysis.py
git commit -q -m "feat(phase0): AUC + logistic + leave-one-game-out + shuffle control

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 5: Live capture + orchestration driver

**Files:**
- Create: `scripts/prune_oracle.py`

**Note:** This is the integration layer (live API). It is exercised by the real run in Task 6, not by unit tests. Keep all analysis logic in `prune_analysis.py`; this file only captures and reports.

- [ ] **Step 1: Write the driver**

```python
# scripts/prune_oracle.py
"""Phase 0a prune-capability oracle probe (live driver).

Runs banked v6 (SalienceExplorer trust=3, border_mask=2) on real public games, captures a
per-step trace, reconstructs the level graph, and reports per completed level:
  Tier-1 prune ceiling (states + actions) and Tier-2 leave-one-game-out AUC vs shuffle.

Usage:
  ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/prune_oracle.py \
      [budget] [highbudget] [game_prefixes...]
Defaults: budget=40000 highbudget=150000 games=tu93,vc33,m0r0,ls20,lp85,cd82
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3 import prune_analysis as A  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("oracle"))


def run_and_capture(prefix, budget):
    """Run v6 on one real game; return (steps, grid_by_key, colors_by_key, levels, tier_by_key)."""
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    card = client.open_scorecard(tags=["prune-oracle"])
    env = client.make(game_id=gid, scorecard_id=card)
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset()
    steps, grid_by_key, colors_by_key, tier_by_key = [], {}, {}, {}
    n, best = 0, 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        lv_before = int(obs.levels_completed or 0)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         lv_before, list(obs.available_actions or []))
        fk = pol.prev_key  # state acted from (set inside decide)
        tier = 0
        if fk is not None and fk in pol.nodes and tok != A.RESET:
            tier = int(pol.nodes[fk].tier.get(tok, 0))
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        lv_after = int(obs.levels_completed or 0)
        if fk is not None:
            steps.append(A.Step(n, lv_before, fk, tok, float(lv_after - lv_before), tier))
            if fk not in grid_by_key:
                grid_by_key[fk] = grid
                colors_by_key[fk] = {int(c) for c in np.unique(grid)} - {pol.bg or 0}
        best = max(best, lv_after)
        n += 1
    return steps, grid_by_key, colors_by_key, best, gid, (pol.bg or 0)


def discovery_meta(steps, colors_by_key):
    """discovery_tier[to_key] and parent_colors[to_key] from the discovering edge."""
    disc_tier, parent_colors = {}, {}
    for i in range(len(steps) - 1):
        s = steps[i]
        if s.action == A.RESET or s.reward > 0:
            continue
        to_key = steps[i + 1].from_key
        if to_key not in disc_tier:
            disc_tier[to_key] = s.tier
            parent_colors[to_key] = colors_by_key.get(s.from_key, set())
    return disc_tier, parent_colors


def analyze_game(prefix, budget, highbudget):
    steps, grids, colors, best, gid, bg = run_and_capture(prefix, budget)
    edges, first_seen = A.build_edges(steps)
    segs = A.segment_levels(steps, first_seen)
    # If the deepest target level wasn't completed, retry once at high budget.
    if not segs and highbudget > budget:
        print(f"  {gid}: no completed level @ {budget}; retry @ {highbudget}", flush=True)
        steps, grids, colors, best, gid, bg = run_and_capture(prefix, highbudget)
        edges, first_seen = A.build_edges(steps)
        segs = A.segment_levels(steps, first_seen)
    disc_tier, parent_colors = discovery_meta(steps, colors)

    per_level, game_X, game_y = {}, [], []
    for lvl, seg in sorted(segs.items()):
        path = A.shortest_path(edges, seg.start_key, seg.target_key, seg.member_keys)
        cl = A.ceilings(seg, path)
        per_level[lvl] = cl
        if path is None:
            continue
        on = set(path)
        for k in seg.member_keys:
            if k not in grids:
                continue
            f = A.state_features(grids[k], background=bg,
                                 discovery_tier=disc_tier.get(k, 0),
                                 parent_colors=parent_colors.get(k))
            game_X.append(A.feature_vector(f))
            game_y.append(1 if k in on else 0)
    return {"gid": gid, "best_level": best, "per_level": per_level,
            "X": np.array(game_X) if game_X else None,
            "y": np.array(game_y) if game_y else None}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    highbudget = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
    games = sys.argv[3:] if len(sys.argv) > 3 else ["tu93", "vc33", "m0r0", "ls20", "lp85", "cd82"]
    t0 = time.time()
    results, per_game = {}, {}
    for g in games:
        try:
            r = analyze_game(g, budget, highbudget)
        except StopIteration:
            print(f"  {g}: NOT IN DEV SET", flush=True)
            continue
        results[g] = {"gid": r["gid"], "best_level": r["best_level"], "per_level": r["per_level"]}
        for lvl, cl in r["per_level"].items():
            print(f"  [{g} L{lvl}] ceiling_states={cl['ceiling_states']} "
                  f"ceiling_actions={cl['ceiling_actions']} disc={cl['discovered_states']} "
                  f"acts={cl['actual_actions']} reachable={cl['reachable']}", flush=True)
        if r["X"] is not None and r["y"] is not None and r["y"].sum() > 0:
            per_game[g] = (r["X"], r["y"])
    auc = A.logo_auc(per_game) if len(per_game) >= 2 else float("nan")
    sh = A.shuffle_auc(per_game) if len(per_game) >= 2 else float("nan")
    print(f"\n== Tier-2 leave-one-game-out AUC={auc:.3f}  shuffle={sh:.3f}  "
          f"(games={list(per_game)})", flush=True)
    print("== VERDICT thresholds: ceiling_actions>=0.5 AND AUC>=0.65 AND AUC>=shuffle+0.10 "
          "=> BUILD; high ceiling + AUC~shuffle => KILL", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)
    out = "/tmp/prune_oracle.json"
    with open(out, "w") as f:
        json.dump({"auc": auc, "shuffle": sh, "results": results}, f, indent=2, default=str)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test it imports and the analysis wiring is sound (no live call)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('scripts/prune_oracle.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/prune_oracle.py
git commit -q -m "feat(phase0): live capture + orchestration driver for prune oracle

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 6: Run the probe and record results

**Files:** none (execution + results capture).

- [ ] **Step 1: Quick single-game sanity run (fast, completes a level)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/prune_oracle.py 8000 8000 vc33`
Expected: prints at least `[vc33 L0]`/`[vc33 L1]` ceiling lines with `reachable=True`; non-empty. If a level shows `reachable=False`, inspect whether the productive path crossed a reset (spec §7 caveat) before trusting that level's numbers.

- [ ] **Step 2: Full probe run (the 6 target games, high-budget completion enabled)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/prune_oracle.py 40000 150000 tu93 vc33 m0r0 ls20 lp85 cd82 2>&1 | tee /tmp/prune_oracle.log`
Expected: a per-level ceiling table for every game v6 completes a level on; a final `Tier-2 ... AUC=... shuffle=...` line; `/tmp/prune_oracle.json` saved. Games with no completed level even at 150k are reported (and absent from the table) — record them as `no ground truth — excluded`.

- [ ] **Step 3: Apply the pre-registered verdict**

Read `/tmp/prune_oracle.log`. For each game classify per the spec table:
- `ceiling_actions >= 0.5` AND `AUC >= 0.65` AND `AUC >= shuffle + 0.10` → **BUILD** (that mechanic).
- high ceiling but `AUC` not clearly above shuffle → **KILL** the learned-prior bet for that mechanic.
- `ceiling_actions < 0.5` → true coverage wall.
Write the per-game map + the single BUILD/KILL/NARROW call into the design doc's results section (append a `## Results (YYYY-MM-DD)` section to the spec file) and `git add` + commit it.

```bash
git add docs/superpowers/specs/2026-06-21-phase0-prune-oracle-probe-design.md
git commit -q -m "docs(phase0): prune oracle results + build/kill verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

- [ ] **Step 4: STOP for user review**

Do NOT start the generator / Phase 0b / any model code. Present the results table + verdict to the user and wait for direction (per the working agreement: validate-or-kill, then stop before the next build).

---

## Task 7: Record the finding to memory

**Files:** `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-research-findings.md` (+ MEMORY.md pointer if a new file).

- [ ] **Step 1: Append a dated finding**

Append one paragraph to `arcagi3-research-findings.md` recording: the Tier-1 ceilings per game, the Tier-2 LOGO AUC vs shuffle, and the verdict (which mechanics pass the prune gate, which are confirmed coverage walls). Note the optimistic-oracle caveat (ceiling assumes goal location known; AUC is the learnability check). Cross-link `[[arcagi3-progress]]`. This ensures the verdict is never re-derived.

- [ ] **Step 2: No commit** (memory files live outside the repo).

---

## Self-Review

**Spec coverage:**
- §1 capture → Task 5 `run_and_capture` (exact v6 config, per-step trace via `pol.prev_key`). ✓
- §2 Tier-1 ceiling (states + actions) → Tasks 1–2 (`build_edges`, `segment_levels`, `shortest_path`, `ceilings`). ✓
- §3 Tier-2 features + LOGO AUC + base-rate + shuffle controls → Tasks 3–4 (`state_features`, `logo_auc`, `shuffle_auc`; base-rate = AUC 0.5 reference). ✓
- §4 pre-registered thresholds + per-game map → Task 6 Step 3 (verdict applied from the printed line). ✓
- §1 high-budget completion for uncompleted walls → Task 5 `analyze_game` retry @ highbudget; Task 6 Step 2. ✓
- §5 deliverables (script, table, memory write) → Tasks 5, 6, 7. ✓
- §7 caveats (reset-free path validation, small positive class) → Task 6 Step 1 note; LOGO skips games with no positives. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `Step(idx, level, from_key, action, reward, tier)` used identically in tests and driver. `LevelSeg` fields match `segment_levels`/`ceilings`. `RESET = ("reset",)` referenced as `A.RESET` in the driver. `feature_vector`/`FEATURE_ORDER` consistent across Tasks 3–4 and the driver. `logo_auc`/`shuffle_auc` take `dict[game]->(X,y)` in both tests and driver. ✓

**Background handling:** `run_and_capture` returns `pol.bg` (the explorer's detected background) and `analyze_game` threads it into `state_features`, so non-zero-background games are handled correctly. The Task 6 sanity run still prints/eyeballs one game to confirm `pol.bg` is sane.
