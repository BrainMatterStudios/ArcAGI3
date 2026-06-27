# History-Augmented State (crack ls20) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. NOTE: a pre-existing test `tests/test_bakeoff_metrics.py::test_human_baseline_level1_is_optimal` HANGS — never run a bare full `pytest`; run the new test module + targeted suites, or deselect that test.

**Goal:** Build a firewalled `HistoryAugmentedExplorer` that folds occlusion-aware, `mod N` visit-counters of the static objects the avatar steps on into the node key, cracking ls20's invisible-rotation wall without regressing the dev set.

**Architecture:** `HistoryAugmentedExplorer(SalienceExplorer)` overrides `_key` (append a counter tuple) and `decide` (update avatar position + object-location memory + edge-triggered visit counts before the key is computed). `augment=False` ⇒ byte-identical to the banked agent. Validated against the offline ls20 engine's `cklxociuu` as a ground-truth oracle.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), numpy, pytest, `arc_agi`/`arcengine` (OFFLINE local games + ls20 engine; NORMAL for the dev A/B), `arcagi3.{perception,movement,salience_explorer}`.

**Spec:** `docs/superpowers/specs/2026-06-21-history-augmented-state-design.md`

**Run prefix:** `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python ...`

---

## File Structure
- Create: `src/arcagi3/history_augmented_explorer.py` — the subclass.
- Create: `tests/test_history_augmented_explorer.py` — firewall + key-augmentation + visit-counter unit tests.
- Modify: `scripts/eval_efficiency.py` — additive `histaug` policy for the dev A/B.
- Create: `scripts/ls20_crack.py` — offline ls20 validation (crack + counter-vs-cklxociuu oracle).

---

## Task 1: Key augmentation + firewall

**Files:**
- Create: `src/arcagi3/history_augmented_explorer.py`
- Test: `tests/test_history_augmented_explorer.py`

Build the subclass with the augmented `_key` and the `augment` flag, but drive the counter dict
directly in tests (the live history-update is Task 2). This isolates the firewall + key logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_history_augmented_explorer.py
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset()
    toks = []
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
    return toks


def test_augment_off_is_byte_identical():
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base = _drive(SalienceExplorer(**cfg), "push", 400)
    off = _drive(HistoryAugmentedExplorer(augment=False, **cfg), "push", 400)
    assert off == base and len(base) > 50


def test_key_unaugmented_when_no_counts():
    import numpy as np
    pol = HistoryAugmentedExplorer(augment=True, seed=0)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    base = SalienceExplorer(seed=0)
    base.bg = 0
    base.vt.update(grid)
    assert pol._key(grid) == base._key(grid)   # no counts -> identical to banked key


def test_key_changes_with_counter():
    import numpy as np
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10, 10] = 5
    pol.bg = 0
    pol.vt.update(grid)
    k0 = pol._key(grid)
    pol._counts = {(5, 10, 10, 10, 10): 1}    # a visited-object counter
    k1 = pol._key(grid)
    pol._counts = {(5, 10, 10, 10, 10): 2}
    k2 = pol._key(grid)
    pol._counts = {(5, 10, 10, 10, 10): 5}    # 5 % 4 == 1 == same residue as count 1
    k5 = pol._key(grid)
    assert k0 != k1 and k1 != k2 and k1 == k5   # mod 4: counts 1 and 5 collapse
```

- [ ] **Step 2: Run to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arcagi3.history_augmented_explorer'`.

- [ ] **Step 3: Write the skeleton + augmented key**

```python
# src/arcagi3/history_augmented_explorer.py
"""HistoryAugmentedExplorer — folds mod-N visit-counters of the static objects the avatar steps on
into the node key, so render-invisible cyclic hidden state (e.g. ls20's rotation) becomes part of the
node identity. Self-gating: no walkable special tiles -> no counters -> byte-identical to the banked
SalienceExplorer (==TransferExplorer key). augment=False is the byte-identical firewall. See
docs/superpowers/specs/2026-06-21-history-augmented-state-design.md
"""
from __future__ import annotations

import numpy as np

from . import perception as P
from .movement import infer_translation
from .salience_explorer import SalienceExplorer

OBJ_MAX_SIZE = 16   # only track small glyphs/tiles as visit targets (not big regions)


class HistoryAugmentedExplorer(SalienceExplorer):
    def __init__(self, *args, augment: bool = False, counter_mod: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.augment = bool(augment)
        self.counter_mod = int(counter_mod)
        self._counts: dict = {}          # object signature -> cumulative visit count
        self._obj_locations: dict = {}   # signature -> frozenset of (r,c) cells (last seen visible)
        self._avatar_colors: set = set()
        self._prev_grid = None
        self._prev_overlaps: set = set()

    def _key(self, grid):
        base = super()._key(grid)
        if not self.augment or not self._counts:
            return base
        aug = tuple(sorted((sig, c % self.counter_mod) for sig, c in self._counts.items()))
        return base + b"|H|" + repr(aug).encode()
```

- [ ] **Step 4: Run to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py -v`
Expected: PASS (3 tests). `test_augment_off_is_byte_identical` is the firewall.

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/history_augmented_explorer.py tests/test_history_augmented_explorer.py
git commit -q -m "feat(phase-q): HistoryAugmentedExplorer key augmentation + firewall

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 2: Occlusion-aware visit counting

**Files:**
- Modify: `src/arcagi3/history_augmented_explorer.py`
- Test: `tests/test_history_augmented_explorer.py`

A *visit* to object O = the avatar's tracked cells enter O's remembered location (edge-triggered: only
on arrival, so standing still doesn't multi-count), AND it works when O is occluded under the avatar
(use remembered locations, not just current-frame objects).

- [ ] **Step 1: Write the failing test (synthetic occlusion sequence)**

```python
# add to tests/test_history_augmented_explorer.py
import numpy as np


def _g(avatar_rc, tile_rc=None):
    g = np.zeros((64, 64), dtype=np.int8)
    if tile_rc is not None:
        g[tile_rc] = 7                 # a static glyph, color 7
    ar, ac = avatar_rc
    g[ar, ac] = 9                      # avatar, color 9 (drawn last -> occludes tile if same cell)
    return g


def test_occlusion_aware_visit_counter():
    pol = HistoryAugmentedExplorer(augment=True, seed=0, counter_mod=4)
    sig = (7, 20, 20, 20, 20)          # (color, r0, c0, r1, c1) of the tile at (20,20)
    # frame 0: avatar at (20,21), tile visible at (20,20) -> remember tile, no visit yet
    pol._update_history(_g((20, 21), tile_rc=(20, 20)))
    # frame 1: avatar moves onto (20,20) -> tile occluded; visit++ (edge-triggered arrival)
    pol._update_history(_g((20, 20), tile_rc=None))
    assert pol._counts.get(sig) == 1
    # frame 2: avatar stays on (20,20) -> NO additional count (not a new arrival)
    pol._update_history(_g((20, 20), tile_rc=None))
    assert pol._counts.get(sig) == 1
    # frame 3: avatar leaves to (20,21); tile reappears
    pol._update_history(_g((20, 21), tile_rc=(20, 20)))
    # frame 4: avatar steps back onto the tile -> visit++ again
    pol._update_history(_g((20, 20), tile_rc=None))
    assert pol._counts.get(sig) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py::test_occlusion_aware_visit_counter -v`
Expected: FAIL — `AttributeError: ... '_update_history'`.

- [ ] **Step 3: Implement `_update_history`**

```python
# add to HistoryAugmentedExplorer in src/arcagi3/history_augmented_explorer.py
    def _avatar_cells(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        if self._prev_grid is not None and self._prev_grid.shape == grid.shape:
            tr = infer_translation(self._prev_grid, grid, self.bg)
            if tr is not None:
                self._avatar_colors.add(int(tr[0]))
        if not self._avatar_colors:
            return set()
        return {(int(r), int(c))
                for r, c in np.argwhere(np.isin(grid, list(self._avatar_colors)))}

    def _update_history(self, grid):
        if self.bg is None:
            self.bg = P.detect_background(grid)
        avatar = self._avatar_cells(grid)
        # refresh remembered locations of small static glyphs that are CURRENTLY visible and
        # NOT the avatar (avatar colors excluded so the avatar isn't a visit target)
        for o in P.connected_components(grid, background=self.bg):
            if o.size <= OBJ_MAX_SIZE and int(o.color) not in self._avatar_colors:
                self._obj_locations[(int(o.color),) + tuple(o.bbox)] = \
                    frozenset((int(r), int(c)) for r, c in o.cells)
        # edge-triggered visit: avatar enters a remembered object's cells now but did not last step
        overlaps_now = set()
        for sig, cells in self._obj_locations.items():
            if avatar & cells:
                overlaps_now.add(sig)
                if sig not in self._prev_overlaps:
                    self._counts[sig] = self._counts.get(sig, 0) + 1
        self._prev_overlaps = overlaps_now
        self._prev_grid = grid
```

- [ ] **Step 4: Run to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/history_augmented_explorer.py tests/test_history_augmented_explorer.py
git commit -q -m "feat(phase-q): occlusion-aware edge-triggered visit counting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 3: Wire history-update into decide + the dev A/B policy

**Files:**
- Modify: `src/arcagi3/history_augmented_explorer.py`
- Modify: `scripts/eval_efficiency.py`
- Test: `tests/test_history_augmented_explorer.py`

- [ ] **Step 1: Write the failing test (counts accrue during a driven game)**

```python
# add to tests/test_history_augmented_explorer.py
def test_decide_updates_history_when_augmenting():
    # navg is a clean arrow-maze with a colored goal glyph the avatar can reach; counts may be 0 if
    # the avatar never overlaps a glyph, but the history machinery must RUN without error and the
    # augment-off run must still equal banked.
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    pol = HistoryAugmentedExplorer(augment=True, **cfg)
    _drive(pol, "navg", 300)
    assert isinstance(pol._counts, dict)          # ran without error
    assert pol._prev_grid is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py::test_decide_updates_history_when_augmenting -v`
Expected: FAIL — `_prev_grid is None` (decide doesn't call `_update_history` yet).

- [ ] **Step 3: Override `decide` to update history before the key is computed**

```python
# add to HistoryAugmentedExplorer
    def decide(self, grid, *args, **kwargs):
        if self.augment:
            self._update_history(grid)   # updates self._counts BEFORE super().decide -> self._key
        return super().decide(grid, *args, **kwargs)
```

- [ ] **Step 4: Run to verify it passes + re-run the firewall**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py -v`
Expected: PASS (5 tests) — incl. `test_augment_off_is_byte_identical` still green (decide early-skips when augment=False).

- [ ] **Step 5: Add the additive `histaug` policy to eval_efficiency**

In `scripts/eval_efficiency.py`, `make_policy`, before `raise ValueError(name)` add:

```python
    if name == "histaug":
        from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer
        return HistoryAugmentedExplorer(seed=0, trust_threshold=3, border_mask=2,
                                        augment=True, counter_mod=4)
```

- [ ] **Step 6: Smoke + commit**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('scripts/eval_efficiency.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

```bash
git add src/arcagi3/history_augmented_explorer.py tests/test_history_augmented_explorer.py scripts/eval_efficiency.py
git commit -q -m "feat(phase-q): wire history-update into decide + histaug A/B policy

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 4: ls20 crack validation (offline engine + cklxociuu oracle)

**Files:**
- Create: `scripts/ls20_crack.py`

- [ ] **Step 1: Write the validation script**

```python
# scripts/ls20_crack.py
"""Phase Q ls20 validation: does HistoryAugmentedExplorer crack ls20, and does its rot-tile counter
track the engine's hidden rotation (cklxociuu)? Offline deterministic engine. See the Phase Q spec.
"""
from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402
from arcagi3.history_augmented_explorer import HistoryAugmentedExplorer  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)


def run(pol, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("ls20"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    env = client.make(game_id=gid, scorecard_id="ls20crack")
    obs = env.reset()
    n, best = 0, 0
    rot_mismatch = 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        # ORACLE: agent's total visit-count mod 4 should equal (engine cklxociuu - start) mod 4 shape.
        if isinstance(pol, HistoryAugmentedExplorer) and pol._counts:
            agent_rot = sum(pol._counts.values()) % 4
            engine_rot_delta = (env._game.cklxociuu - env._game.dhksvilbb.index(
                env._game.current_level.get_data("StartRotation"))) % 4
            if agent_rot != engine_rot_delta:
                rot_mismatch += 1
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        best = max(best, int(obs.levels_completed or 0))
        n += 1
    return best, n, rot_mismatch


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    base = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    b_lv, b_n, _ = run(base, budget)
    print(f"banked (augment off): ls20 best_level={b_lv} actions={b_n}", flush=True)
    aug = HistoryAugmentedExplorer(seed=0, trust_threshold=3, border_mask=2, augment=True, counter_mod=4)
    a_lv, a_n, mism = run(aug, budget)
    print(f"history-augmented:     ls20 best_level={a_lv} actions={a_n}  rot_oracle_mismatches={mism}", flush=True)
    print("RESULT: " + ("ls20 CRACKED by augmentation" if a_lv > b_lv or (a_lv >= 1 and a_n < 2000)
                        else "no crack — investigate visit detection / exploration"), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the validation**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/ls20_crack.py 8000 2>&1 | tail -5`
Expected: prints banked vs augmented ls20 levels+actions and the rot-oracle mismatch count. **Interpretation gates:** (a) `rot_oracle_mismatches` should be ~0 once the rot tile is being counted (the counter tracks the hidden rotation) — if it's high, the visit detector isn't tracking the rot tile (occlusion/identity bug) and must be fixed before judging the crack; (b) augmented `best_level` ≥ 1 at far fewer actions than banked = crack. If the counter tracks but the level doesn't crack, the explorer isn't reaching `(goal-adjacent, matching-rotation)` within budget — raise budget / inspect.

- [ ] **Step 3: Commit (script + a short result note inline in the commit body)**

```bash
git add scripts/ls20_crack.py
git commit -q -m "feat(phase-q): ls20 crack validation (offline engine + cklxociuu oracle)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 5: Dev no-regression A/B + pre-registered verdict

**Files:** none (execution + results capture).

- [ ] **Step 1: Run the A/B (banked vs augmented) on the dev split**

Run both at budget 6000 and 30000:
```
for pol in salience histaug; do for b in 6000 30000; do
  ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/eval_efficiency.py $b $pol \
    2>&1 | tee /tmp/q_${pol}_${b}.log ; done ; done
```
Expected: per-policy TUNE/HOLDOUT `mean_levels` + `sum_eff` at both budgets.

- [ ] **Step 2: Apply the pre-registered gate (spec §4)**

WIN iff: ls20 cracked efficiently (Task 4) AND `histaug` drops **no** TUNE/HOLDOUT levels vs `salience`
at either budget. If ls20 cracks but a game regresses, the augmentation is exploding it → record which,
and (next iteration) tighten the object filter to repeatedly-visitable occluding glyphs. Append a
`## Results (YYYY-MM-DD)` section to the spec with the ls20 numbers + the A/B table + verdict; commit.

```bash
git add docs/superpowers/specs/2026-06-21-history-augmented-state-design.md
git commit -q -m "docs(phase-q): history-augmented results + verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

- [ ] **Step 3: STOP for user review.** Do NOT promote to the submission default or submit. Present the
ls20 crack result + the no-regression A/B + verdict and wait for direction (promotion needs a real
Kaggle submission; dev ≠ Kaggle).

---

## Task 6: Record to memory

**Files:** `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-research-findings.md`

- [ ] **Step 1: Append a dated finding** — whether ls20 was cracked (the first invisible-state wall
crack) + the actions, the rot-oracle result, the dev no-regression outcome, and the verdict/next step
(promotion, object-filter tightening, or kill). Cross-link `[[arcagi3-ls20-mechanic-ground-truth]]`. No
commit (memory is outside the repo).

---

## Self-Review

**Spec coverage:**
- §2 mechanism (mod-N augmented key; self-gating) → Task 1 `_key`. ✓
- §2 occlusion-aware visit detection (object-location memory, edge-triggered, avatar via infer_translation) → Task 2 `_update_history`. ✓
- §3 architecture (subclass off SalienceExplorer; decide hook) → Tasks 1+3. ✓
- §4 firewall (augment=False byte-identical) → Task 1 `test_augment_off_is_byte_identical`. ✓
- §4 gate (ls20 crack + zero dev regression) → Tasks 4 (crack + oracle) + 5 (A/B + verdict). ✓
- §5 risks (visit detection load-bearing → cklxociuu oracle in Task 4; state explosion → A/B in Task 5) → covered. ✓
- §6 deliverables (explorer, tests, ls20 run, A/B, memory) → Tasks 1-6. ✓

**Placeholder scan:** Tasks 1-3 are complete code + tests. Task 4 is a complete script with explicit
interpretation gates. No vague TODOs. ✓

**Type consistency:** `HistoryAugmentedExplorer(augment, counter_mod, seed, trust_threshold, border_mask)`
ctor used identically across tests, eval_efficiency, and ls20_crack. `_counts` (dict sig→int),
`_obj_locations` (dict sig→frozenset), `_update_history(grid)`, `_avatar_cells(grid)` consistent
between Task 2 def and Task 3 use. Object signature `(color,)+bbox` (5-tuple) consistent between
`_update_history` and the Task 1/2 test fixtures. `_key` appends `b"|H|"+repr(aug)` — bytes throughout
(matches `object_state_key` returning bytes). ✓

**One integration risk (flagged, not a gap):** the cklxociuu oracle in Task 4 reads `env._game` internals
(`cklxociuu`, `dhksvilbb`, `current_level.get_data("StartRotation")`) — all verified to exist this
session (the active probe used `env._game.cklxociuu`). If `StartRotation` indexing differs per level,
the oracle's delta math may need a tweak; the crack/no-crack signal (best_level, actions) does not
depend on the oracle.
```
