# Phase E Walk-Reduction (Increment 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `TourExplorer` (two frontier-ordering modes that cut re-traversal walk distance) and A/B it vs the banked explorer on the holdout efficiency harness at two budgets — keeping the banked `SalienceExplorer` structurally untouched.

**Architecture:** `TourExplorer(SalienceExplorer)` overrides `_observe` (discovery-order bookkeeping) and `_path_to_frontier` (mode-dispatched frontier selection: `v6` passthrough / `yield` / `dfs`). The banked class is unmodified, so 0.33 cannot regress; a `v6`-passthrough byte-identical test guards the plumbing. `scripts/eval_efficiency.py` gains additive `tour-yield`/`tour-dfs` policies for the A/B.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), numpy, pytest, `arc_agi`/`arcengine` (OFFLINE local games for tests; NORMAL live games for the A/B).

**Spec:** `docs/superpowers/specs/2026-06-21-phase-e-walk-reduction-design.md`

**Run prefix:** `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python ...`

---

## File Structure
- Create: `src/arcagi3/tour_explorer.py` — `TourExplorer` (modes v6/yield/dfs).
- Create: `tests/test_tour_explorer.py` — firewall + selection + local-coverage tests.
- Modify: `scripts/eval_efficiency.py` — additive `make_policy` modes (the `"salience"` branch stays byte-identical).

---

## Task 1: TourExplorer (subclass with frontier-ordering modes)

**Files:**
- Create: `src/arcagi3/tour_explorer.py`
- Test: `tests/test_tour_explorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tour_explorer.py
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer, _Node
from arcagi3.tour_explorer import TourExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    """Drive a reactive policy on a local OFFLINE game; return (tokens, final_levels)."""
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset()
    toks, levels = [], 0
    for _ in range(steps):
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame),
                         gstate_terminal=(obs.state == GameState.GAME_OVER),
                         gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        toks.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        levels = max(levels, int(obs.levels_completed or 0))
    return toks, levels


def test_v6_mode_is_byte_identical_to_banked():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base, _ = _drive(SalienceExplorer(**cfg), "push", 400)
    v6, _ = _drive(TourExplorer(frontier_mode="v6", **cfg), "push", 400)
    assert v6 == base and len(base) > 50


def _mk(pol, key, cands_with_tiers, edges):
    n = _Node(key, cands_with_tiers)
    n.edges = dict(edges)
    pol.nodes[key] = n
    return n


def _toy_graph(pol):
    # R(no untried) -> A(1 untried, depth1, disc1), R -> B(2 untried, depth1, disc2) -> C(1 untried, depth2, disc3)
    _mk(pol, b"R", [(("S", 1), 0), (("S", 2), 0)],
        {("S", 1): (b"A", 0.0), ("S", 2): (b"B", 0.0)})
    _mk(pol, b"A", [(("S", 1), 0), (("S", 3), 0)], {("S", 1): (b"R", 0.0)})
    _mk(pol, b"B", [(("S", 1), 0), (("S", 3), 0), (("S", 4), 0), (("S", 5), 0)],
        {("S", 1): (b"R", 0.0), ("S", 5): (b"C", 0.0)})
    _mk(pol, b"C", [(("S", 1), 0), (("S", 6), 0)], {("S", 1): (b"B", 0.0)})
    pol._disc_order = {b"R": 0, b"A": 1, b"B": 2, b"C": 3}


def test_yield_picks_max_untried_nearest_frontier():
    pol = TourExplorer(frontier_mode="yield")
    _toy_graph(pol)
    # depth-1 frontiers A(1 untried) and B(2 untried) -> pick B; path R->B = [("S",2)]
    assert pol._path_to_frontier(b"R", 9) == [("S", 2)]


def test_dfs_picks_most_recently_discovered_frontier():
    pol = TourExplorer(frontier_mode="dfs")
    _toy_graph(pol)
    # frontiers A(disc1) and C(disc3) -> pick C (most recent); path R->B->C = [("S",2),("S",5)]
    assert pol._path_to_frontier(b"R", 9) == [("S", 2), ("S", 5)]


def test_modes_preserve_coverage_on_local_game():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    _, lv_v6 = _drive(SalienceExplorer(**cfg), "navg", 8000)
    _, lv_y = _drive(TourExplorer(frontier_mode="yield", **cfg), "navg", 8000)
    _, lv_d = _drive(TourExplorer(frontier_mode="dfs", **cfg), "navg", 8000)
    assert lv_v6 > 0 and lv_y >= lv_v6 and lv_d >= lv_v6   # no coverage loss on the toy


def test_unknown_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        TourExplorer(frontier_mode="bogus")
```

- [ ] **Step 2: Run to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_tour_explorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arcagi3.tour_explorer'`.

- [ ] **Step 3: Write the implementation**

```python
# src/arcagi3/tour_explorer.py
"""TourExplorer — frontier-ordering A/B variants of the banked SalienceExplorer (Phase E
Increment 2). Tests whether reordering WHICH frontier the explorer walks to cuts re-traversal
walk distance without losing coverage. SalienceExplorer is UNTOUCHED — variants live here in a
subclass selected by frontier_mode, so the banked 0.33 agent cannot regress. frontier_mode="v6"
is a byte-identical passthrough (firewall-tested). See
docs/superpowers/specs/2026-06-21-phase-e-walk-reduction-design.md.
"""
from __future__ import annotations

from collections import deque

from .salience_explorer import SalienceExplorer


class TourExplorer(SalienceExplorer):
    def __init__(self, *args, frontier_mode: str = "v6", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if frontier_mode not in ("v6", "yield", "dfs"):
            raise ValueError(f"unknown frontier_mode: {frontier_mode}")
        self.frontier_mode = frontier_mode
        self._disc_order: dict = {}      # key -> first-seen counter (side-channel, no behaviour effect)
        self._disc_counter = 0

    def _observe(self, key, cands, terminal=False):
        n = super()._observe(key, cands, terminal=terminal)
        if key not in self._disc_order:
            self._disc_order[key] = self._disc_counter
            self._disc_counter += 1
        return n

    def _path_to_frontier(self, start, p):
        if self.frontier_mode == "v6":
            return super()._path_to_frontier(start, p)
        if start not in self.nodes:
            return None
        if self.nodes[start].has_untried_le(p):
            return []
        # Full BFS: shortest action-path (+depth) to every reachable node; collect frontiers.
        frontiers = []  # (key, path, depth)
        seen = {start}
        q = deque([(start, [])])
        while q:
            k, path = q.popleft()
            node = self.nodes.get(k)
            if not node:
                continue
            for a, (nk, _r) in node.edges.items():
                if nk in seen:
                    continue
                seen.add(nk)
                np_ = path + [a]
                nn = self.nodes.get(nk)
                if nn is not None and nn.has_untried_le(p):
                    frontiers.append((nk, np_, len(np_)))
                q.append((nk, np_))
        if not frontiers:
            return None
        if self.frontier_mode == "yield":
            d_min = min(f[2] for f in frontiers)
            near = [f for f in frontiers if f[2] == d_min]
            # most untried at the nearest depth; deterministic tie-break by earliest discovery
            return max(near, key=lambda f: (len(self.nodes[f[0]].untried_le(p)),
                                            -self._disc_order[f[0]]))[1]
        # dfs: most-recently-discovered frontier (disc_order is unique per node)
        return max(frontiers, key=lambda f: self._disc_order[f[0]])[1]
```

- [ ] **Step 4: Run to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_tour_explorer.py -v`
Expected: PASS (5 tests). The `v6` byte-identical test is the firewall; the toy-graph tests verify the selection logic.

- [ ] **Step 5: Run the full suite (no regression — SalienceExplorer untouched)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: all PASS (190 + 5 new = 195).

- [ ] **Step 6: Commit**

```bash
git add src/arcagi3/tour_explorer.py tests/test_tour_explorer.py
git commit -q -m "feat(phase-e): TourExplorer frontier-ordering modes (yield/dfs/v6)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 2: Wire tour modes into the A/B harness

**Files:**
- Modify: `scripts/eval_efficiency.py` (the `make_policy` function only).

- [ ] **Step 1: Add the additive policies**

In `scripts/eval_efficiency.py`, the current `make_policy` is:

```python
def make_policy(name: str):
    if name == "salience":
        return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    raise ValueError(name)
```

Replace it with (the `"salience"` branch is unchanged — byte-identical baseline):

```python
def make_policy(name: str):
    if name == "salience":
        return SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    if name in ("tour-yield", "tour-dfs"):
        from arcagi3.tour_explorer import TourExplorer
        mode = "yield" if name == "tour-yield" else "dfs"
        return TourExplorer(seed=0, trust_threshold=3, border_mask=2, frontier_mode=mode)
    raise ValueError(name)
```

- [ ] **Step 2: Smoke-test (parse + one fast game per mode)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('scripts/eval_efficiency.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/eval_efficiency.py 4000 tour-yield 2>&1 | tail -5`
Expected: prints per-game `[TUNE]/[HOLDOUT]` lines and the `== TUNE/HOLDOUT mean_levels=... sum_eff=...` summary, then `saved /tmp/eval_tour-yield_4000.json`. (Confirms the policy wires up and runs end-to-end; this is not the gate.)

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_efficiency.py
git commit -q -m "feat(phase-e): tour-yield/tour-dfs policies in eval_efficiency (additive)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 3: Run the A/B and apply the pre-registered verdict

**Files:** none (execution + results capture).

- [ ] **Step 1: Run all three policies at both budgets**

Run (six runs; ~minutes each at ~510 act/s):
```
for pol in salience tour-yield tour-dfs; do
  for b in 6000 30000; do
    ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/eval_efficiency.py $b $pol \
      2>&1 | tee /tmp/ab_${pol}_${b}.log
  done
done
```
Expected: each run prints TUNE + HOLDOUT `mean_levels` and `sum_eff`, and saves `/tmp/eval_${pol}_${b}.json`.

- [ ] **Step 2: Apply the pre-registered ship rule**

From the six logs, build the comparison table: for each budget {6000, 30000} and each mode
{tour-yield, tour-dfs} vs salience, record HOLDOUT `mean_levels`, HOLDOUT `sum_eff`, TUNE `mean_levels`.
A mode **wins** iff, at BOTH budgets: HOLDOUT `sum_eff` improves vs salience AND HOLDOUT `mean_levels`
does not drop AND TUNE `mean_levels` does not drop. If neither mode wins at both budgets → **KILL**.
Append a `## Results (YYYY-MM-DD)` section to the spec with the table and the verdict (winner mode, or
kill); commit.

```bash
git add docs/superpowers/specs/2026-06-21-phase-e-walk-reduction-design.md
git commit -q -m "docs(phase-e): walk-reduction A/B results + verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

- [ ] **Step 3: STOP for user review**

Do NOT promote any mode to the submission default and do NOT submit to Kaggle. Per spec §2, even a
winner needs a real Kaggle submission (user-gated) to confirm before promotion. Present the A/B table
+ verdict and wait for direction.

---

## Task 4: Record the finding to memory

**Files:** `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-research-findings.md`

- [ ] **Step 1: Append a dated finding**

Append one paragraph recording: the A/B outcome (per-budget HOLDOUT efficiency + levels for
tour-yield/tour-dfs vs salience), the pre-registered verdict (winner or kill), and — if a winner —
that promotion still needs a live submission (dev≠Kaggle). Cross-link `[[arcagi3-progress]]` and the
Phase E audit finding. Note `SalienceExplorer` was never modified (0.33 firewalled).

- [ ] **Step 2: No commit** (memory files live outside the repo).

---

## Self-Review

**Spec coverage:**
- §1 TourExplorer subclass, `_observe` discovery order, `_path_to_frontier` v6/yield/dfs → Task 1. ✓
- §1 v6 byte-identical firewall + coverage preserved → Task 1 `test_v6_mode_is_byte_identical_to_banked` + `test_modes_preserve_coverage_on_local_game`. ✓
- §2 additive `make_policy` modes, salience untouched → Task 2. ✓
- §2 A/B at both budgets + pre-registered ship rule (HOLDOUT eff up, no HOLDOUT/TUNE level drop, both budgets) → Task 3 Steps 1–2. ✓
- §2 winner needs live submission; not auto-promoted → Task 3 Step 3 STOP. ✓
- §3 deliverables (class, tests, modes, A/B table) → Tasks 1–3. ✓
- §6 subclass plumbing firewall test in the suite → Task 1 Step 5. ✓

**Placeholder scan:** No TBD/TODO; all code shown; commands have expected output. ✓

**Type consistency:** `TourExplorer(seed, trust_threshold, border_mask, frontier_mode)` ctor used
identically in tests, `make_policy`, and the A/B. `frontier_mode` values `"v6"/"yield"/"dfs"` consistent
throughout. `_disc_order`/`_disc_counter` defined in Task 1 and set in the toy-graph test helper with the
same names. `_path_to_frontier(start, p)` signature matches the parent. The toy graph's expected paths
(`[("S",2)]` for yield, `[("S",2),("S",5)]` for dfs) are consistent with the `_mk` edges defined in
`_toy_graph`. ✓

**Determinism note (verified in design):** frontier selection uses no RNG; `_disc_order` is unique per
node, so `max(...)` tie-breaks are unique and reproducible. The only RNG (within-tier action choice) is
inherited unchanged, so seeded A/B runs are deterministic and comparable.
