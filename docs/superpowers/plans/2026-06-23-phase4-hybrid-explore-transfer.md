# Phase 4 — Hybrid Explore-to-Clear, Model-Guide-to-Deepen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `HybridTransferExplorer` = `TransferExplorer` + best-effort model-based action promotion, and run a three-arm lp85 bake-off (salience / signature-transfer / +model) to test whether the discovery model beats the shallow signature heuristic.

**Architecture:** A `ModelGuide` component maintains an induced model from the observed transition stream (reusing Phase-1–3 induction + planner in advisory mode) and exposes `suggest(grid, available) → action_id | None`. `HybridTransferExplorer` subclasses `TransferExplorer`, feeds it each transition, and promotes the suggested action to tier 0 in `_candidates` (the same mechanism `TransferExplorer` uses for the reward signature). Firewall: `enable_model_guidance=False` → byte-identical to `TransferExplorer`; the salience explorer always remains the floor that clears levels.

**Tech Stack:** Python 3.12, numpy, pytest (`pythonpath=["src"]`), offline `arc_agi`/`arcengine`. No GPU/LLM.

**Reuse (verified current signatures):**
- `TransferExplorer(SalienceExplorer)` (`src/arcagi3/transfer_explorer.py`): `__init__(*args, enable_transfer=True, transfer_demote=3, sig_mode="color", **kwargs)`; `reset_all()` (adds `reward_simple`, `reward_click_sig`, `_prev_grid`); `decide(grid, gstate_terminal, gstate_notplayed, levels, available)` (learns on level-up, stores `self._prev_grid`); `_candidates(grid, available) -> [(act, tier)]` (promotes matches to tier 0). Inherits `self.bg`, `self.prev_action`, `self.prev_levels` from `SalienceExplorer`. `decide` returns `("S", aid) | ("C", x, y) | ("reset",)`.
- `DiscoveryExplorer` (`src/arcagi3/discovery_explorer.py`): `_bg`, `_agent_color`, `_deltas`, `_prev_grid`, `_prev_token`, `_last_level`; `_ingest_movement(grid)`, `_ingest_transform(grid)`, `_ingest_world_delta(grid)`, `_build_model(grid)`, `_build_plan(grid)` (sets `self._plan`), `on_level_change(level)`. (These ingests read `self._prev_grid`/`self._prev_token`/`self._bg`/`self._agent_color`.)
- `perception as P`: `P.detect_background(grid)`.

**Guardrails:** firewall keeps the banked baselines byte-identical; guidance is best-effort (salience is the floor); single game (lp85).

---

## File Structure

- Create `src/arcagi3/model_guide.py` — `ModelGuide` (advisory induction/planning → suggested action).
- Create `src/arcagi3/hybrid_transfer_explorer.py` — `HybridTransferExplorer(TransferExplorer)`.
- Create `scripts/hybrid_bakeoff.py` — three-arm lp85 driver.
- Create `tests/test_model_guide.py`, `tests/test_hybrid_transfer_explorer.py`.
- Modify `docs/experiment-overview.html` — Experiment 46.

---

## Task 1: `ModelGuide` — advisory induction → suggested action

**Files:** Create `src/arcagi3/model_guide.py`; Test `tests/test_model_guide.py`

`ModelGuide` wraps a `DiscoveryExplorer` used purely as a model/plan source. It mirrors the bookkeeping `DiscoveryExplorer.decide` does internally (set `_prev_grid`/`_prev_token`, run the ingests), then builds the model+plan and returns the first planned simple action.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_guide.py
import numpy as np
from arcagi3.model_guide import ModelGuide

def test_suggest_none_before_any_observation():
    g = ModelGuide()
    grid = np.zeros((8, 8), dtype=np.int8); grid[3, 3] = 5
    assert g.suggest(grid, [1, 2, 3, 4]) is None   # no model yet

def test_observe_then_suggest_returns_simple_action():
    # Feed a couple of transitions where a color-5 agent moves under actions, so a movement
    # model forms; with a target present, suggest should return one of the simple actions.
    g = ModelGuide()
    a = np.zeros((8, 8), dtype=np.int8); a[3, 3] = 5; a[3, 6] = 9   # agent + a distinct target
    b = np.zeros((8, 8), dtype=np.int8); b[3, 4] = 5; b[3, 6] = 9   # agent moved right (action 4)
    g.observe(prev_grid=a, prev_action=("S", 4), grid=b, level=0)
    c = np.zeros((8, 8), dtype=np.int8); c[3, 5] = 5; c[3, 6] = 9   # moved right again
    g.observe(prev_grid=b, prev_action=("S", 4), grid=c, level=0)
    s = g.suggest(c, [1, 2, 3, 4])
    assert s is None or s in (1, 2, 3, 4)   # a valid simple action id, or None if no plan yet

def test_level_change_does_not_crash():
    g = ModelGuide()
    a = np.zeros((6, 6), dtype=np.int8); a[0, 0] = 5
    b = np.zeros((6, 6), dtype=np.int8); b[0, 1] = 5
    g.observe(prev_grid=a, prev_action=("S", 4), grid=b, level=0)
    g.observe(prev_grid=b, prev_action=("S", 4), grid=b, level=1)  # level increment
    assert g.suggest(b, [1, 2, 3, 4]) in (None, 1, 2, 3, 4)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_guide.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# src/arcagi3/model_guide.py
"""ModelGuide — advisory wrapper over the discovery induction/planner.

Maintains an induced model from the observed (prev_grid, action, grid) stream and suggests a
goal-directed action, WITHOUT emitting actions itself. Used by HybridTransferExplorer to bias the
salience search. Reuses DiscoveryExplorer's induction internals (no duplication); if a level's
model can't yield a plan, suggest() returns None and the explorer falls back to salience.
"""
from __future__ import annotations

from arcagi3 import perception as P
from arcagi3.discovery_explorer import DiscoveryExplorer


class ModelGuide:
    def __init__(self):
        self._d = DiscoveryExplorer(seed=0)
        self._d.reset_all()
        self._obs_count = 0
        self._built_at = -1
        self._plan_cache: list = []

    def observe(self, prev_grid, prev_action, grid, level) -> None:
        d = self._d
        if d._bg is None:
            d._bg = P.detect_background(grid)
        if level != d._last_level:
            d.on_level_change(level)
        # mirror DiscoveryExplorer.decide's bookkeeping so the ingests can read the transition
        d._prev_grid, d._prev_token = prev_grid, prev_action
        d._ingest_movement(grid)
        d._ingest_transform(grid)
        d._ingest_world_delta(grid)
        self._obs_count += 1

    def suggest(self, grid, available):
        d = self._d
        if d._bg is None or not d._deltas:
            return None
        # rebuild the model/plan only when new evidence has arrived (bounded per-step cost)
        if self._obs_count != self._built_at:
            d._build_model(grid)
            d._plan = []
            d._build_plan(grid)
            self._plan_cache = list(d._plan)
            self._built_at = self._obs_count
        if self._plan_cache:
            a = self._plan_cache[0]
            if isinstance(a, int) and a in available:
                return a
        return None
```

NOTE: `DiscoveryExplorer._plan` holds simple action ids (ints). If after wiring the test
`test_observe_then_suggest_returns_simple_action` the build path errors on the discovery internals
(e.g. an ingest guard), adapt the `observe` bookkeeping to satisfy those guards — the goal is a working
`observe`→`suggest`. If the `DiscoveryExplorer`-internals coupling proves too brittle, the documented
fallback (spec §6) is to compute the suggestion directly from the free functions
(`movement.infer_all_translations` for deltas + `factored_model.plan` toward a target). Prefer the reuse;
report `DONE_WITH_CONCERNS` if you take the fallback, explaining why.

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_guide.py -v`
Expected: PASS (3 tests). The asserts are tolerant (`None` or a valid action id) because the point is a
working, non-crashing advisory API, not a specific plan on a toy grid.

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/model_guide.py tests/test_model_guide.py
git commit -q -m "feat(hybrid): ModelGuide — advisory induction/plan -> suggested action"
```

---

## Task 2: `HybridTransferExplorer`

**Files:** Create `src/arcagi3/hybrid_transfer_explorer.py`; Test `tests/test_hybrid_transfer_explorer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hybrid_transfer_explorer.py
import numpy as np
from arcagi3.hybrid_transfer_explorer import HybridTransferExplorer
from arcagi3.transfer_explorer import TransferExplorer

def _seq():
    # a short scripted sequence of (grid, levels) the explorer will decide on
    frames = []
    for i in range(6):
        g = np.zeros((10, 10), dtype=np.int8); g[2, 2 + (i % 3)] = 5; g[2, 8] = 9
        frames.append(g)
    return frames

def _run(explorer, frames):
    explorer.reset_all()
    toks = []
    for g in frames:
        toks.append(explorer.decide(g, gstate_terminal=False, gstate_notplayed=False,
                                    levels=0, available=[1, 2, 3, 4]))
    return toks

def test_firewall_guidance_off_matches_transfer_explorer():
    frames = _seq()
    hyb = HybridTransferExplorer(seed=0, enable_model_guidance=False)
    base = TransferExplorer(seed=0)
    assert _run(hyb, frames) == _run(base, frames)   # byte-identical trace with guidance off

def test_promotes_guide_suggestion(monkeypatch):
    hyb = HybridTransferExplorer(seed=0, enable_model_guidance=True)
    hyb.reset_all()
    # stub the guide to always suggest action 3
    monkeypatch.setattr(hyb._guide, "suggest", lambda grid, available: 3)
    g = np.zeros((10, 10), dtype=np.int8); g[2, 2] = 5
    tok = hyb.decide(g, gstate_terminal=False, gstate_notplayed=False, levels=0, available=[1, 2, 3, 4])
    assert tok == ("S", 3)   # the suggested simple action is promoted and chosen
```

- [ ] **Step 2: Run to verify FAIL**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_hybrid_transfer_explorer.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# src/arcagi3/hybrid_transfer_explorer.py
"""HybridTransferExplorer — TransferExplorer + best-effort model-based action promotion.

The salience explorer stays the floor that clears levels; the ModelGuide's suggested goal-directed
action is promoted to tier 0 in the candidate ordering (same mechanism TransferExplorer uses for the
reward signature). Firewall: enable_model_guidance=False -> byte-identical to TransferExplorer.
"""
from __future__ import annotations

from .model_guide import ModelGuide
from .transfer_explorer import TransferExplorer


class HybridTransferExplorer(TransferExplorer):
    def __init__(self, *args, enable_model_guidance: bool = True, **kwargs) -> None:
        self.enable_model_guidance = bool(enable_model_guidance)
        super().__init__(*args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self._guide = ModelGuide()
        self._guide_prev_grid = None
        self._guide_prev_token = None
        self._guide_suggestion = None   # cached suggestion for the current decide()

    def decide(self, grid, gstate_terminal, gstate_notplayed, levels, available):
        # feed the just-completed transition to the guide (best-effort; never blocks the floor)
        if (self.enable_model_guidance and self._guide_prev_grid is not None
                and self._guide_prev_token is not None and self._guide_prev_token[0] == "S"):
            try:
                self._guide.observe(self._guide_prev_grid, self._guide_prev_token, grid, levels)
            except Exception:  # noqa: BLE001  guidance is best-effort; never crash the explorer
                pass
        self._guide_suggestion = None
        if self.enable_model_guidance and not gstate_terminal and not gstate_notplayed:
            try:
                self._guide_suggestion = self._guide.suggest(grid, available)
            except Exception:  # noqa: BLE001
                self._guide_suggestion = None
        token = super().decide(grid, gstate_terminal, gstate_notplayed, levels, available)
        self._guide_prev_grid, self._guide_prev_token = grid, token
        return token

    def _candidates(self, grid, available):
        cands = super()._candidates(grid, available)   # salience + signature-transfer ordering
        if not self.enable_model_guidance or self._guide_suggestion is None:
            return cands
        sug = ("S", self._guide_suggestion)
        return [(act, 0 if act == sug else tier) for (act, tier) in cands]
```

NOTE on the promotion test: `super().decide` calls `self._candidates`, which promotes `("S", 3)` to tier 0;
the salience selection then picks the tier-0 simple action. If the base explorer's tie-breaking among
tier-0 candidates doesn't deterministically pick the promoted one in the toy test, adjust the test to assert
the promoted action is AT tier 0 in `hyb._candidates(g, [1,2,3,4])` instead of asserting the final token —
i.e. test the promotion mechanism directly. Keep the firewall test as the hard gate.

- [ ] **Step 4: Run to verify PASS**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_hybrid_transfer_explorer.py -v`
Expected: PASS (2 tests — firewall identity + promotion). Then a quick regression:
`PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_guide.py tests/test_hybrid_transfer_explorer.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/arcagi3/hybrid_transfer_explorer.py tests/test_hybrid_transfer_explorer.py
git commit -q -m "feat(hybrid): HybridTransferExplorer (signature + model-guided promotion, firewalled)"
```

---

## Task 3: Three-arm lp85 bake-off driver

**Files:** Create `scripts/hybrid_bakeoff.py`

- [ ] **Step 1: Write the driver**

```python
# scripts/hybrid_bakeoff.py
"""Phase-4 three-arm lp85 bake-off: salience vs signature-transfer vs +model guidance.

Measures actions-to-clear per level for each arm — does discovery-model guidance beat the shallow
signature heuristic? Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/hybrid_bakeoff.py [budget]
"""
from __future__ import annotations
import logging, sys, time
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.hybrid_transfer_explorer import HybridTransferExplorer

logging.basicConfig(level=logging.ERROR)


def _retry(fn, tries=5, delay=1.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))


def make_lp85():
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("hb"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("lp85"))
    return client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["hybrid"]))


def run_arm(name, engine, budget):
    env = make_lp85()
    obs = _retry(env.reset)
    if hasattr(engine, "reset_all"):
        engine.reset_all()
    per_level = []          # actions spent to clear each level
    cur_level = int(obs.levels_completed or 0)
    actions_this = 0
    n = 0
    while n < budget:
        if obs.state == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        tok = engine.decide(grid=grid, gstate_terminal=(obs.state == GameState.GAME_OVER),
                            gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                            levels=cur_level, available=list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = _retry(env.reset)
        elif tok[0] == "S":
            aid = tok[1]; obs = _retry(lambda: env.step(GameAction.from_id(aid))); actions_this += 1; n += 1
        else:
            x, y = tok[1], tok[2]; obs = _retry(lambda: env.step(GameAction.ACTION6, data={"x": x, "y": y})); actions_this += 1; n += 1
        nl = int(obs.levels_completed or 0)
        if nl > cur_level:
            per_level.append((cur_level, actions_this)); cur_level = nl; actions_this = 0
    return name, cur_level, per_level


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    arms = [
        ("salience", SalienceExplorer(seed=0)),
        ("transfer(signature)", TransferExplorer(seed=0)),
        ("hybrid(+model)", HybridTransferExplorer(seed=0)),
    ]
    print(f"{'arm':<24}{'levels':<8}{'per-level (lvl,actions)':<40}{'total'}", flush=True)
    for name, eng in arms:
        print(f"running {name}...", flush=True)
        nm, levels, per = run_arm(name, eng, budget)
        total = sum(a for _, a in per)
        print(f"{nm:<24}{levels:<8}{str(per):<40}{total}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run at a tiny budget**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/hybrid_bakeoff.py 60`
Expected: runs without error; prints three arm rows (likely 0 levels at budget 60 — only need it to run and print). If lp85 env construction errors, confirm the `lp85` prefix via `get_environments` and report BLOCKED with the exact error.

- [ ] **Step 3: Commit**

```bash
git add scripts/hybrid_bakeoff.py
git commit -q -m "feat(hybrid): three-arm lp85 bake-off driver (salience/signature/model)"
```

---

## Task 4: Run the bake-off + gated guard

**Files:** Modify `tests/test_hybrid_transfer_explorer.py` (gated guard)

- [ ] **Step 1: Full run; record the VERBATIM table.**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/hybrid_bakeoff.py 20000`
(Slow — three arms, up to 20k actions each.) Record per arm: levels cleared, per-level actions, total actions.

- [ ] **Step 2: DIAGNOSE the arm-2 vs arm-3 delta.**

Did `hybrid(+model)` clear ≥ the levels `transfer(signature)` clears, in FEWER total actions? Does the gain grow with depth (compare per-level (lvl,actions) between arms 2 and 3)? If the guide rarely fires (suggestion mostly None) or doesn't help, say so and localize why (throwaway diagnostic in /tmp, removed before commit): is `_deltas` fit? does `suggest` return actions? are promotions changing the chosen action?

- [ ] **Step 3: Add a gated regression guard.**

```python
# add to tests/test_hybrid_transfer_explorer.py
import os, pytest

@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_BAKEOFF") != "1", reason="live lp85 bake-off; set RUN_BAKEOFF=1")
def test_hybrid_not_worse_than_transfer_on_lp85():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("hb", "scripts/hybrid_bakeoff.py")
    hb = importlib.util.module_from_spec(spec); sys.modules["hb"] = hb; spec.loader.exec_module(hb)
    from arcagi3.transfer_explorer import TransferExplorer
    from arcagi3.hybrid_transfer_explorer import HybridTransferExplorer
    _, tlev, tper = hb.run_arm("transfer", TransferExplorer(seed=0), 20000)
    _, hlev, hper = hb.run_arm("hybrid", HybridTransferExplorer(seed=0), 20000)
    # hybrid must clear at least as many levels, and not use more total actions to get there.
    assert hlev >= tlev and sum(a for _, a in hper) <= sum(a for _, a in tper)
```
- This is the success bar. If the run shows hybrid does NOT beat (or match) signature-transfer, KEEP the guard as written and report it as a FAILING documented target — do NOT weaken it. The honest negative ("model guidance does not beat the signature heuristic on lp85") is the decisive result.

- [ ] **Step 4: Confirm gating + run the guard.**

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_hybrid_transfer_explorer.py -q` (guard skipped without RUN_BAKEOFF).
`RUN_BAKEOFF=1 ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_hybrid_transfer_explorer.py -k not_worse -v` (report pass/fail honestly — slow).

- [ ] **Step 5: Commit**

```bash
git add tests/test_hybrid_transfer_explorer.py
git commit -q -m "test(hybrid): lp85 hybrid-vs-signature guard (Phase-4 success bar)"
```

---

## Task 5: Experiment 46 write-up

**Files:** Modify `docs/experiment-overview.html`

- [ ] **Step 1: Append an Experiment-46 panel** matching the existing markup/CSS (read recent panels; `panel`, `accent`/`kill`/`win`, `t-win`/`t-kill`, table rows `<tr><td class="m">N</td>`). Report HONESTLY the three-arm table from Task 4, the arm-2→arm-3 delta (does model guidance beat signature-transfer? by how much? does the gain grow with depth?), and the verdict on whether the discovery machinery earns its keep as an efficiency layer. Note the strategic framing (Option C: explorer-as-floor + model-as-accelerator) and whether this is the project's first net-positive use of the discovery code. Add a main-table summary row (Exp 46, `t-win` or `t-kill` per outcome). v13 = 0.33 remains banked; no submission from Phase 4.

- [ ] **Step 2: Validate HTML well-formed** (HTMLParser) and **commit.**

```bash
git add docs/experiment-overview.html
git commit -q -m "docs: Experiment 46 — hybrid explore+model-guide three-arm lp85 bake-off"
```

---

## Self-Review

**Spec coverage:** `ModelGuide` (Task 1, spec §4 C1) ✓; `HybridTransferExplorer` + firewall + `_candidates` promotion (Task 2, C2) ✓; three-arm lp85 bake-off (Task 3, C3) ✓; run + success-bar guard (Task 4) ✓; Exp-46 write-up (Task 5, C4) ✓; firewall non-negotiable — the byte-identical test is Task 2's hard gate ✓; best-effort guidance (try/except around observe/suggest so the floor never crashes) ✓.

**Placeholder scan:** Task 4 thresholds are the success-bar comparison (relative: hybrid ≤ transfer), not magic numbers; the pass/fail branch is explicit and honest. The Task-1/Task-2 NOTEs give the engineer a concrete fallback (free functions; assert promotion directly) rather than vague "handle edge cases."

**Type consistency:** `ModelGuide.observe(prev_grid, prev_action, grid, level)` / `suggest(grid, available) -> int|None` used identically in Tasks 1–2; `HybridTransferExplorer(enable_model_guidance=...)` + `_candidates` promotion consistent; `run_arm(name, engine, budget) -> (name, levels, per_level)` used identically in Tasks 3–4; decide token `("S",aid)|("C",x,y)|("reset",)` matches the explorer contract.
