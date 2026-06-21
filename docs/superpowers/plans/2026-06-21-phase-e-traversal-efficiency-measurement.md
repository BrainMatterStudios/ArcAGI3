# Phase E Traversal-Efficiency Measurement (Increment 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a behavior-identical `InstrumentedExplorer` + a 16-game audit that quantifies where the banked explorer's re-traversal goes, naming one Increment-2 target (or a kill) — without changing the agent.

**Architecture:** `InstrumentedExplorer` subclasses the banked `SalienceExplorer` and overrides `_choose` with a VERBATIM copy of the parent logic, adding only branch-tag counters (no control-flow or RNG change). A byte-identical-trace test is the firewall guaranteeing the instrument reflects the banked agent. `scripts/retraversal_audit.py` runs it across the 16 tune+holdout games and reports the re-traversal anatomy + the pre-registered proceed/kill check.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), numpy, pytest, `arc_agi`/`arcengine` (OFFLINE local games for the test; NORMAL live games for the audit).

**Spec:** `docs/superpowers/specs/2026-06-21-phase-e-traversal-efficiency-measurement-design.md`

**Run prefix:** `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python ...`

---

## File Structure
- Create: `src/arcagi3/instrumented_explorer.py` — `InstrumentedExplorer` (verbatim `_choose` + tags).
- Create: `tests/test_instrumented_explorer.py` — byte-identical-trace firewall + tag-populate tests.
- Create: `scripts/retraversal_audit.py` — 16-game audit driver + pre-registered threshold check.
- Modify: none.

---

## Task 1: InstrumentedExplorer (behavior-identical, tagged)

**Files:**
- Create: `src/arcagi3/instrumented_explorer.py`
- Test: `tests/test_instrumented_explorer.py`

**CRITICAL:** `_choose` must be a VERBATIM copy of `SalienceExplorer._choose`
(src/arcagi3/salience_explorer.py:235-270) with ONLY counter increments / a list append added.
Do not change control flow, the `self.plan = []` invalidation, RNG calls (`self.rng.integers`), or
return values. The byte-identical test in Step 1 is what proves you got this right.

- [ ] **Step 1: Write the failing firewall + populate tests**

```python
# tests/test_instrumented_explorer.py
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.instrumented_explorer import InstrumentedExplorer

GAMES_DIR = "src/arcagi3/games"


def _drive(pol, game_id, steps):
    """Drive a reactive policy on a local OFFLINE game; return the list of returned tokens."""
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


def test_instrumented_is_byte_identical_to_banked():
    # FIREWALL: same seed + config -> identical action sequence as the banked explorer.
    cfg = dict(seed=0, trust_threshold=3, border_mask=2)
    base = _drive(SalienceExplorer(**cfg), "push", 400)
    inst = _drive(InstrumentedExplorer(**cfg), "push", 400)
    assert inst == base and len(base) > 50


def test_tags_populate_sensibly():
    pol = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2)
    _drive(pol, "push", 400)
    assert pol.tags["fresh_local_test"] > 0
    assert pol.tags["plan_replay_walk"] + pol.tags["new_plan_to_frontier"] > 0  # push has walks
    assert isinstance(pol.plan_invalidations, int) and pol.plan_invalidations >= 0
    assert all(isinstance(n, int) and n > 0 for n in pol.new_plan_len)


def test_tags_are_deterministic():
    a = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2); _drive(a, "push", 300)
    b = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2); _drive(b, "push", 300)
    assert dict(a.tags) == dict(b.tags) and a.plan_invalidations == b.plan_invalidations
```

- [ ] **Step 2: Run to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_instrumented_explorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arcagi3.instrumented_explorer'`.

- [ ] **Step 3: Write the implementation (verbatim `_choose` + tags)**

```python
# src/arcagi3/instrumented_explorer.py
"""InstrumentedExplorer — read-only subclass of the banked SalienceExplorer that tags which
_choose branch fired, for the Phase E re-traversal audit. _choose is a VERBATIM copy of the
parent's logic with ONLY counters added: same decisions, same RNG, byte-identical action trace
(enforced by tests/test_instrumented_explorer.py). See
docs/superpowers/specs/2026-06-21-phase-e-traversal-efficiency-measurement-design.md.
"""
from __future__ import annotations

from collections import Counter

from .salience_explorer import MAX_TIER, SalienceExplorer


class InstrumentedExplorer(SalienceExplorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set AFTER super().__init__ (which calls reset_all); these persist across the run.
        self.tags: Counter = Counter()
        self.plan_invalidations: int = 0
        self.new_plan_len: list = []

    def _choose(self, cur):
        node = self.nodes.get(cur)
        if node is None:
            self.tags["random_no_node"] += 1
            return self._random(cur)
        # 1) exploit reward
        ra = node.reward_action()
        if ra is not None:
            self.tags["exploit"] += 1
            return ra
        # 2) active plan replay
        if self.plan:
            if self._expect is not None and cur != self._expect:
                self.plan_invalidations += 1
                self.plan = []
            else:
                self.tags["plan_replay_walk"] += 1
                return self.plan.pop(0)
        # 3) hierarchical tier exploration
        g = self.active_group
        while g <= MAX_TIER:
            local = node.untried_le(g)
            if local:
                self.tags["fresh_local_test"] += 1
                self.active_group = g
                mp = min(node.tier.get(a, 0) for a in local)
                choices = [a for a in local if node.tier.get(a, 0) == mp]
                return choices[int(self.rng.integers(0, len(choices)))]
            path = self._path_to_frontier(cur, g)
            if path:
                self.tags["new_plan_to_frontier"] += 1
                self.new_plan_len.append(len(path))
                self.active_group = g
                self.plan = path
                return self.plan.pop(0)
            g += 1
        # 4) exhausted from here -> bounce off root
        if (self.root_key is not None and cur != self.root_key
                and self.stuck_resets < self.max_stuck_resets):
            self.tags["reset_bounce"] += 1
            self.stuck_resets += 1
            self.expect_reset = True
            self.plan = []
            return ("reset",)
        self.tags["random_exhausted"] += 1
        return self._random(cur)
```

- [ ] **Step 4: Run to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_instrumented_explorer.py -v`
Expected: PASS (3 tests). The byte-identical test confirms zero behavior drift from the banked agent.

- [ ] **Step 5: Run the full suite (no regression — no agent code changed)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: all PASS (187 + 3 new = 190).

- [ ] **Step 6: Commit**

```bash
git add src/arcagi3/instrumented_explorer.py tests/test_instrumented_explorer.py
git commit -q -m "feat(phase-e): behavior-identical InstrumentedExplorer with branch tags

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 2: 16-game re-traversal audit driver

**Files:**
- Create: `scripts/retraversal_audit.py`

**Note:** Integration layer (live NORMAL API). Exercised by the real run in Task 3, not unit tests.

- [ ] **Step 1: Write the driver**

```python
# scripts/retraversal_audit.py
"""Phase E Increment 1 — re-traversal audit across the 16 tune+holdout games.

Runs the behavior-identical InstrumentedExplorer (banked v6 config) on each game and reports the
action anatomy: frontier-walk %, redundant-probe %, discovery %, exploit %, reset %, plus the
pre-registered proceed/kill check (a pool >=30% of actions on >=10/16 games + a structurally-safe
cut => Increment 2; else KILL).

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/retraversal_audit.py [budget]
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
from arcagi3.instrumented_explorer import InstrumentedExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("audit"))

TUNE = ["vc33", "cd82", "sc25", "lp85", "lf52", "tu93", "ar25", "sp80"]
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]


def audit_game(prefix, budget):
    try:
        gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    except StopIteration:
        return None
    card = client.open_scorecard(tags=["retraversal-audit"])
    env = client.make(game_id=gid, scorecard_id=card)
    pol = InstrumentedExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset()
    n, best = 0, 0
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame), st == GameState.GAME_OVER,
                         st == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        best = max(best, int(obs.levels_completed or 0))
        n += 1
    t = pol.tags
    walk = t["plan_replay_walk"] + t["new_plan_to_frontier"]
    local = t["fresh_local_test"]
    distinct = len(pol.nodes)                      # discoveries (only local tests discover)
    redundant = max(0, local - distinct)           # local probes that hit an already-known state
    exploit = t["exploit"]
    resets = t["reset_bounce"]
    tot = max(n, 1)
    wl = np.array(pol.new_plan_len) if pol.new_plan_len else np.array([0])
    return {"gid": gid, "best_level": best, "total_actions": n, "distinct_states": distinct,
            "pct_walk": round(walk / tot, 3), "pct_redundant_probe": round(redundant / tot, 3),
            "pct_discovery": round(distinct / tot, 3), "pct_exploit": round(exploit / tot, 3),
            "pct_reset": round(resets / tot, 3), "plan_invalidations": pol.plan_invalidations,
            "walk_count": len(pol.new_plan_len), "walk_mean_len": round(float(wl.mean()), 1),
            "walk_max_len": int(wl.max())}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    games = sys.argv[2:] if len(sys.argv) > 2 else (TUNE + HOLDOUT)
    t0 = time.time()
    rows = {}
    for g in games:
        r = audit_game(g, budget)
        if r is None:
            print(f"  {g}: NOT IN DEV SET", flush=True)
            continue
        rows[g] = r
        print(f"  {g} L{r['best_level']} acts={r['total_actions']}: "
              f"walk={r['pct_walk']} redundant={r['pct_redundant_probe']} "
              f"discovery={r['pct_discovery']} exploit={r['pct_exploit']} reset={r['pct_reset']} "
              f"invalid={r['plan_invalidations']} (walks={r['walk_count']} mean={r['walk_mean_len']} "
              f"max={r['walk_max_len']})", flush=True)
    if rows:
        n = len(rows)
        for pool in ("pct_walk", "pct_redundant_probe"):
            vals = [r[pool] for r in rows.values()]
            n_ge30 = sum(1 for v in vals if v >= 0.30)
            print(f"== {pool}: action-weighted mean={np.mean(vals):.3f}  games>=30%: {n_ge30}/{n}",
                  flush=True)
        print("== PRE-REGISTERED: a pool >=30% on >=10/16 games + a structurally-safe cut "
              "=> Increment 2; else KILL the efficiency lever.", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)
    out = "/tmp/retraversal_audit.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test (parse + a single fast game)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('scripts/retraversal_audit.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/retraversal_audit.py 8000 vc33`
Expected: one line `vc33 L... acts=...: walk=... redundant=... discovery=...` and `saved /tmp/retraversal_audit.json`. Sanity-check the percentages sum roughly to 1 (walk + redundant + discovery + exploit + reset ≈ 1.0, minus small random_* slack).

- [ ] **Step 3: Commit**

```bash
git add scripts/retraversal_audit.py
git commit -q -m "feat(phase-e): 16-game re-traversal audit driver

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 3: Run the audit + apply the pre-registered decision

**Files:** none (execution + results capture).

- [ ] **Step 1: Full 16-game audit**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/retraversal_audit.py 40000 2>&1 | tee /tmp/retraversal_audit.log`
Expected: a per-game anatomy line for all 16 games (~10-15 min), the two `== pool:` summary lines, and `/tmp/retraversal_audit.json`.

- [ ] **Step 2: Apply the pre-registered proceed/kill rule**

From the log: a pool (`pct_walk` or `pct_redundant_probe`) qualifies for Increment 2 only if it is
`>= 0.30` on `>= 10` of the games AND has a structurally-safe cut (frontier-walks → ordering /
plan-invalidation reduction is coverage-preserving; redundant-probes → only if a large model-free-
detectable share, e.g. via the `walk_mean_len`/invalidation structure). If neither pool qualifies
→ **clean KILL** of the efficiency lever. Append a `## Results (YYYY-MM-DD)` section to the spec
file with the per-game table, the two pool summaries, and the BUILD(target)/KILL verdict; commit.

```bash
git add docs/superpowers/specs/2026-06-21-phase-e-traversal-efficiency-measurement-design.md
git commit -q -m "docs(phase-e): re-traversal audit results + Increment-2 target/kill verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

- [ ] **Step 3: STOP for user review**

Do NOT start Increment 2 (the cut) or touch `SalienceExplorer`. Present the anatomy table + the
pre-registered verdict (target pool or kill) and wait for direction (validate-or-kill; the cut is
its own spec, and any default change needs the holdout gate + a real submission per spec §4).

---

## Task 4: Record the finding to memory

**Files:** `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-research-findings.md`

- [ ] **Step 1: Append a dated finding**

Append one paragraph recording: the re-traversal anatomy (walk% / redundant-probe% aggregate and
games≥30%), the pre-registered verdict (Increment-2 target pool or kill), and the dev≠Kaggle /
non-strict-superset caveat for any eventual ordering cut. Cross-link `[[arcagi3-progress]]` and the
Phase 0a prune-probe finding. Ensures the verdict is never re-derived.

- [ ] **Step 2: No commit** (memory files live outside the repo).

---

## Self-Review

**Spec coverage:**
- §1 InstrumentedExplorer behavior-identical + tagged → Task 1 (verbatim `_choose` + byte-identical firewall test). ✓
- §2 retraversal_audit over 16 games, branch split + walk stats + redundant-probe + plan-invalidation → Task 2. ✓
- §3 pre-registered proceed/kill (≥30% on ≥10/16 + safe cut) → Task 2 summary lines + Task 3 Step 2. ✓
- §4 firewall/gate principles for Increment 2 (out of scope to build; stated) → Task 3 Step 3 STOP + spec. ✓
- §5 deliverables (class, tests, driver, results, memory) → Tasks 1–4. ✓
- §6 success = behavior-identical-verified anatomy naming one target or a kill → Task 1 firewall + Task 3. ✓
- §7 risks (instrumentation drift) → Task 1 byte-identical test in the suite (Step 5). ✓

**Placeholder scan:** No TBD/TODO; every code step is complete; commands have expected output. ✓

**Type consistency:** `InstrumentedExplorer(seed, trust_threshold, border_mask)` matches the
`SalienceExplorer` ctor and is used identically in the test and driver. `self.tags` (Counter),
`self.plan_invalidations` (int), `self.new_plan_len` (list) are defined in Task 1 and read with the
same names in Task 2. Tag keys (`plan_replay_walk`, `new_plan_to_frontier`, `fresh_local_test`,
`exploit`, `reset_bounce`) are identical between the class and the audit's pool math. ✓

**Known approximation (acceptable, noted in driver):** `redundant_probe = max(0, fresh_local_test −
len(pol.nodes))` uses total distinct nodes as the discovery count (only local tests discover, so
this is tight; reward/terminal `_observe` adds a handful of non-discovery nodes, slightly
*under*-counting redundant probes — conservative for the ≥30% threshold).
