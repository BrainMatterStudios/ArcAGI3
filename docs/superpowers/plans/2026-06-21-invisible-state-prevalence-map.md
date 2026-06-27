# Invisible-State Prevalence Map — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: subagents are currently blocked by a macOS Documents-folder TCC sandbox in this repo; execute inline (executing-plans) unless access is re-granted or the repo is moved.

**Goal:** Measure how widespread invisible-state walls are across the dev games — determinism violations (same masked-key+action → different next-key) that are resolvable by a history feature, separated from animation noise and over-merge — anchored on ls20 ground truth, with a minimal scripted ls20 active probe.

**Architecture:** A pure numpy analysis module classifies logged transitions into OVER_MERGE / INVISIBLE_STATE / UNEXPLAINED using the masked key, the unmasked key, and per-step history-feature values. A passive capture driver logs those fields while driving the banked explorer over the dev games. A scripted offline ls20 probe proves (or refutes) that the invisible rotation is observable + history-resolvable when the gate is actively reached. No agent behavior changes; 0.33 untouched.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), numpy, pytest, `arc_agi`/`arcengine` (NORMAL live games for the map; OFFLINE local engine for the ls20 probe), `arcagi3.{perception,movement,tracking,salience_explorer}`.

**Spec:** `docs/superpowers/specs/2026-06-21-invisible-state-prevalence-map-design.md`

**Run prefix:** `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python ...`

---

## File Structure
- Create: `src/arcagi3/invisible_state.py` — pure transition-classification analysis.
- Create: `tests/test_invisible_state.py` — unit tests on synthetic transitions.
- Create: `scripts/invisible_state_map.py` — passive capture driver over dev games + ls20 anchor.
- Create: `scripts/invisible_state_probe.py` — scripted offline ls20 active probe.
- Modify: none.

---

## Task 1: Pure transition-classification analysis

**Files:**
- Create: `src/arcagi3/invisible_state.py`
- Test: `tests/test_invisible_state.py`

**Concept:** A transition is `(mkey, ukey, action, next_mkey, feats)` where `mkey`/`ukey` are the
masked/unmasked state keys of the FROM state, `action` is the token, `next_mkey` is the masked key of
the resulting state, and `feats` is a dict of candidate history-feature values present at that step
(e.g. `{"obj7_mod4": 3}`). A `(mkey, action)` pair with ≥2 distinct `next_mkey` is a VIOLATION. Classify
each violation:
- **OVER_MERGE** — the instances have ≥2 distinct `ukey` (masking merged visibly-different states).
- **INVISIBLE_STATE** — all instances share one `ukey` (truly identical pixels) AND some history feature
  makes `next_mkey` a function of `(mkey, feat, action)`.
- **UNEXPLAINED** — one `ukey`, but no history feature resolves it (noise, or a feature we didn't generate).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_invisible_state.py
from arcagi3 import invisible_state as IS


def _t(mkey, ukey, action, next_mkey, feats):
    return IS.Transition(mkey=mkey, ukey=ukey, action=action, next_mkey=next_mkey, feats=feats)


def test_find_violations():
    ts = [
        _t(b"A", b"A", ("S", 1), b"X", {}),
        _t(b"A", b"A", ("S", 1), b"Y", {}),   # same (mkey,action) -> different next => violation
        _t(b"A", b"A", ("S", 2), b"Z", {}),    # different action, single next => not a violation
    ]
    v = IS.find_violations(ts)
    assert set(v) == {(b"A", ("S", 1))}
    assert v[(b"A", ("S", 1))] == {b"X", b"Y"}


def test_invisible_state_resolved_by_history():
    # same pixels (ukey constant), next depends on a hidden counter exposed via feats["rot_mod4"]
    ts = [
        _t(b"G", b"G", ("S", 1), b"WIN", {"rot_mod4": 0, "noise_mod2": 0}),
        _t(b"G", b"G", ("S", 1), b"BLOCK", {"rot_mod4": 1, "noise_mod2": 1}),
        _t(b"G", b"G", ("S", 1), b"WIN", {"rot_mod4": 0, "noise_mod2": 1}),
        _t(b"G", b"G", ("S", 1), b"BLOCK", {"rot_mod4": 1, "noise_mod2": 0}),
    ]
    report = IS.classify(ts)
    pair = (b"G", ("S", 1))
    assert report[pair]["category"] == "INVISIBLE_STATE"
    assert report[pair]["resolver"] == "rot_mod4"      # noise_mod2 does NOT separate win/block


def test_over_merge_detected():
    # masking merged two visibly-different from-states (distinct ukey)
    ts = [
        _t(b"M", b"U1", ("S", 1), b"X", {"f_mod2": 0}),
        _t(b"M", b"U2", ("S", 1), b"Y", {"f_mod2": 1}),
    ]
    report = IS.classify(ts)
    assert report[(b"M", ("S", 1))]["category"] == "OVER_MERGE"


def test_unexplained_when_no_feature_resolves():
    ts = [
        _t(b"N", b"N", ("S", 1), b"X", {"f_mod2": 0}),
        _t(b"N", b"N", ("S", 1), b"Y", {"f_mod2": 0}),   # identical feats, different next => unresolved
    ]
    report = IS.classify(ts)
    assert report[(b"N", ("S", 1))]["category"] == "UNEXPLAINED"


def test_summary_rates():
    ts = [
        _t(b"G", b"G", ("S", 1), b"WIN", {"rot_mod4": 0}),
        _t(b"G", b"G", ("S", 1), b"BLOCK", {"rot_mod4": 1}),
        _t(b"P", b"P", ("S", 1), b"Q", {"rot_mod4": 0}),   # non-violation
    ]
    s = IS.summary(ts)
    assert s["n_pairs"] == 2 and s["n_violations"] == 1
    assert s["invisible_state"] == 1 and s["over_merge"] == 0 and s["unexplained"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_invisible_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arcagi3.invisible_state'`.

- [ ] **Step 3: Write the implementation**

```python
# src/arcagi3/invisible_state.py
"""Pure transition-classification for the Phase P invisible-state prevalence map.

Classifies determinism violations (same masked from-key + action -> different next masked key) into
OVER_MERGE (masking merged visibly-different states), INVISIBLE_STATE (identical pixels, but a history
feature makes the transition deterministic) and UNEXPLAINED (no feature resolves it). numpy/stdlib
only; no live API. See docs/superpowers/specs/2026-06-21-invisible-state-prevalence-map-design.md
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Transition:
    mkey: bytes          # masked state key of the FROM state
    ukey: bytes          # unmasked state key of the FROM state
    action: tuple        # ("S", id) | ("C", x, y) | ("reset",)
    next_mkey: bytes     # masked state key of the resulting state
    feats: dict = field(default_factory=dict)   # candidate history-feature name -> int value


def find_violations(transitions):
    """dict[(mkey, action)] -> set of next_mkey, only for pairs with >1 distinct next_mkey."""
    nexts = defaultdict(set)
    for t in transitions:
        nexts[(t.mkey, t.action)].add(t.next_mkey)
    return {k: v for k, v in nexts.items() if len(v) > 1}


def _instances(transitions, pair):
    mkey, action = pair
    return [t for t in transitions if t.mkey == mkey and t.action == action]


def _resolves(instances, feat):
    """True iff grouping instances by feats[feat] yields a constant next_mkey per group."""
    groups = defaultdict(set)
    for t in instances:
        if feat not in t.feats:
            return False
        groups[t.feats[feat]].add(t.next_mkey)
    return all(len(s) == 1 for s in groups.values())


def classify(transitions):
    """Per violating (mkey, action): {category, resolver, n_next, n_instances}."""
    violations = find_violations(transitions)
    report = {}
    for pair in violations:
        inst = _instances(transitions, pair)
        ukeys = {t.ukey for t in inst}
        if len(ukeys) > 1:
            report[pair] = {"category": "OVER_MERGE", "resolver": None,
                            "n_next": len(violations[pair]), "n_instances": len(inst)}
            continue
        feat_names = sorted({f for t in inst for f in t.feats})
        resolver = next((f for f in feat_names if _resolves(inst, f)), None)
        cat = "INVISIBLE_STATE" if resolver is not None else "UNEXPLAINED"
        report[pair] = {"category": cat, "resolver": resolver,
                        "n_next": len(violations[pair]), "n_instances": len(inst)}
    return report


def summary(transitions):
    rep = classify(transitions)
    cats = [v["category"] for v in rep.values()]
    return {
        "n_pairs": len({(t.mkey, t.action) for t in transitions}),
        "n_violations": len(rep),
        "over_merge": cats.count("OVER_MERGE"),
        "invisible_state": cats.count("INVISIBLE_STATE"),
        "unexplained": cats.count("UNEXPLAINED"),
        "resolvers": sorted({v["resolver"] for v in rep.values() if v["resolver"]}),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_invisible_state.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Full suite + commit**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
Expected: all PASS (195 + 5 = 200).

```bash
git add src/arcagi3/invisible_state.py tests/test_invisible_state.py
git commit -q -m "feat(phase-p): pure transition-classification for invisible-state map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 2: Passive capture driver + ls20 anchor

**Files:**
- Create: `scripts/invisible_state_map.py`

**Concept:** Drive `SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)` over the dev games. Per
step capture: `mkey = pol.prev_key` (masked key acted from); `ukey = P.object_state_key(grid, bg)` (NO
masking); `action`; and per-step interaction counts for stable salient objects, so the analysis can
test `feat = f"obj{oid}_mod{N}"`. `next_mkey` = next step's `mkey`. Interaction = avatar overlaps the
object (move games, via `MotionModel`) OR a click lands in it (click games). Feed Transitions to
`invisible_state.classify/summary`. The ls20 anchor checks whether any INVISIBLE_STATE violation is
found on ls20 and whether its resolver corresponds to the rot-tile object.

- [ ] **Step 1: Write the driver**

```python
# scripts/invisible_state_map.py
"""Phase P passive invisible-state prevalence map (live capture driver).

For each dev game, drive the banked explorer, log per-step (masked key, unmasked key, action,
per-object interaction-count features), build Transitions, and classify determinism violations into
OVER_MERGE / INVISIBLE_STATE / UNEXPLAINED. Anchor: does ls20 surface an INVISIBLE_STATE violation?

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/invisible_state_map.py [budget] [games...]
"""
from __future__ import annotations

import json, logging, sys, time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from arcagi3 import perception as P  # noqa: E402
from arcagi3 import invisible_state as IS  # noqa: E402
from arcagi3.movement import MotionModel  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("ismap"))

TUNE = ["vc33", "cd82", "sc25", "lp85", "lf52", "tu93", "ar25", "sp80"]
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]
MODS = (2, 3, 4)


def _salient_objects(grid, bg):
    """Stable-ish object ids -> set of (r,c) cells. id = (color, bbox) so it persists across frames for
    static objects (walls, tiles); good enough for interaction-count features."""
    objs = {}
    for o in P.connected_components(grid, background=bg):
        if o.size <= 16:  # salient = small/rare; the dense lattice handles large regions elsewhere
            objs[(int(o.color),) + tuple(o.bbox)] = set(map(tuple, o.cells))
    return objs


def _features(counts):
    """counts: dict[oid]->int interaction count -> {f"obj{i}_mod{N}": count%N}."""
    feats = {}
    for i, (oid, c) in enumerate(sorted(counts.items())):
        for N in MODS:
            feats[f"obj{i}_mod{N}"] = c % N
    return feats


def capture_game(prefix, budget):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    card = client.open_scorecard(tags=["is-map"]); env = client.make(game_id=gid, scorecard_id=card)
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    mm = MotionModel()
    obs = env.reset(); n = 0
    counts = {}                  # oid -> cumulative interaction count
    pending = None               # (mkey, ukey, action, feats) awaiting its next_mkey
    transitions = []
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        bg = P.detect_background(grid)
        # close the previous transition now that we observe the resulting state
        cur_mkey_probe = None
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        mkey = pol.prev_key
        if pending is not None and mkey is not None:
            pm, pu, pa, pf = pending
            transitions.append(IS.Transition(mkey=pm, ukey=pu, action=pa, next_mkey=mkey, feats=pf))
            pending = None
        # update interaction counts from THIS state (avatar overlap or click)
        objs = _salient_objects(grid, bg)
        try:
            mm.update(grid) if hasattr(mm, "update") else None
        except Exception:
            pass
        avatar = None
        try:
            ac = mm.avatar_cells(grid)
            avatar = set(map(tuple, ac)) if ac is not None and len(ac) else None
        except Exception:
            avatar = None
        clicked = (int(tok[2]), int(tok[1])) if tok[0] == "C" else None  # (row,col) = (y,x)
        for oid, cells in objs.items():
            hit = (avatar is not None and avatar & cells) or (clicked is not None and clicked in cells)
            if hit:
                counts[oid] = counts.get(oid, 0) + 1
        if tok != ("reset",) and mkey is not None:
            ukey = P.object_state_key(grid, background=bg)
            pending = (mkey, ukey, tok, _features(counts))
        else:
            pending = None
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
    return gid, transitions, int(obs.levels_completed or 0)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2:] if len(sys.argv) > 2 else (TUNE + HOLDOUT)
    t0 = time.time(); rows = {}
    for g in games:
        try:
            gid, ts, lv = capture_game(g, budget)
        except StopIteration:
            print(f"  {g}: NOT IN DEV SET", flush=True); continue
        s = IS.summary(ts)
        rows[g] = {**s, "gid": gid, "levels": lv, "n_transitions": len(ts)}
        print(f"  {g} L{lv}: pairs={s['n_pairs']} viol={s['n_violations']} "
              f"invisible={s['invisible_state']} over_merge={s['over_merge']} "
              f"unexplained={s['unexplained']} resolvers={s['resolvers'][:3]}", flush=True)
    walled = [g for g, r in rows.items() if r["levels"] == 0]
    inv_walled = [g for g in walled if rows[g]["invisible_state"] > 0]
    print(f"\n== anchor ls20: invisible_state={rows.get('ls20', {}).get('invisible_state', 'n/a')}", flush=True)
    print(f"== walled games={walled}; with INVISIBLE_STATE={inv_walled}", flush=True)
    print("== VERDICT bar: anchor fires AND >=2 walled games beyond ls20 with INVISIBLE_STATE concentrated at gates", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)
    with open("/tmp/invisible_state_map.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("saved /tmp/invisible_state_map.json", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Parse + smoke one game**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('scripts/invisible_state_map.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/invisible_state_map.py 6000 vc33`
Expected: a `vc33 L..: pairs=.. viol=.. invisible=.. over_merge=.. unexplained=..` line and the saved JSON.
**Integration-iteration note (inline execution):** if `MotionModel.avatar_cells`/`update` signatures differ from the guarded calls, adjust to the real API (check `src/arcagi3/movement.py:98-118`); the guards degrade to "no avatar" so the script still runs (click-interaction features still populate). Confirm `viol`/`invisible` counts are plausible (not all-zero, not everything).

- [ ] **Step 3: Commit**

```bash
git add scripts/invisible_state_map.py
git commit -q -m "feat(phase-p): passive invisible-state capture driver + ls20 anchor

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 3: Scripted ls20 active probe (offline engine)

**Files:**
- Create: `scripts/invisible_state_probe.py`

**Concept:** On the LOCAL deterministic ls20 engine, reach the goal-adjacent cell at two different
rotations (by visiting the rotation tile a different number of times) and show: the masked from-key is
IDENTICAL across rotations (rotation invisible) while the goal-entry OUTCOME differs (win vs blocked) —
i.e. the invisible state is real and observable-by-active-probing, and a `rot-tile-visit mod 4` feature
explains it. Navigation uses BFS over visible non-wall cells (general, not a hardcoded path); the
rotation is set by stepping the known rot-tile cell and read back from the engine object for validation.

Ground truth (memory `arcagi3-ls20-mechanic-ground-truth`): rot tile grid (32,19); goal grid (12,34);
agent start grid (47,34); deltas pitch 5 (act1 up(-5,0), act2 down(+5,0), act3 left(0,-5), act4 right(0,+5));
known winning solution `[3,3,3,1,1,1,1,4,4,4,1,1,1]` (rotation 0 at goal => WIN).

- [ ] **Step 1: Write the probe**

```python
# scripts/invisible_state_probe.py
"""Phase P active probe: prove ls20's invisible rotation is observable+history-resolvable when the
goal gate is actively reached. Offline deterministic engine. See the Phase P spec.
"""
from __future__ import annotations

import logging, sys
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402
from arcagi3 import perception as P  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
ED = "environment_files"
ROT_TILE = (32, 19)   # (row, col)
GOAL = (12, 34)
PITCH = 5
# action id -> (drow, dcol)
DELTA = {1: (-PITCH, 0), 2: (PITCH, 0), 3: (0, -PITCH), 4: (0, PITCH)}


def masked_key(pol, grid):
    pol.vt.update(grid)
    if pol.bg is None:
        pol.bg = P.detect_background(grid)
    return pol._key(grid)


def run():
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ED,
                    logger=logging.getLogger("probe"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("ls20"))
    results = []
    for target_visits in (1, 2):   # 1 visit => rotation idx0 (win), 2 => idx1 (blocked)
        env = client.make(game_id=gid, scorecard_id="probe")
        obs = env.reset()
        pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
        # NAVIGATION: drive avatar to ROT_TILE, step on it `target_visits` times, then to GOAL-adjacent.
        # Implemented via a small BFS over visible free cells using the avatar centroid each step.
        # (Inline-execution iteration point: identify wall color + avatar cell from the first frame;
        #  the helpers below are the scaffold to finalize against the live engine.)
        outcome = drive_to_goal(env, pol, obs, target_visits)
        results.append((target_visits, outcome))
    # ASSERTIONS the probe must demonstrate:
    fk = [o["from_mkey"] for _, o in results]
    wins = [o["win"] for _, o in results]
    print(f"from_keys_equal={fk[0] == fk[1]}  outcomes={wins}  "
          f"(expect from_keys_equal=True, outcomes=[True, False])", flush=True)
    print("RESULT: invisible-state OBSERVABLE+RESOLVABLE by active probing"
          if (fk[0] == fk[1] and wins == [True, False]) else
          "RESULT: probe did NOT confirm (investigate nav / rotation indexing)", flush=True)


def drive_to_goal(env, pol, obs, target_visits):
    """Navigate avatar -> rot tile (visit target_visits times) -> goal-adjacent -> attempt entry.
    Returns {from_mkey, win}. BFS over non-wall cells; wall color = the dominant structural color."""
    # ... concrete BFS navigation implemented at execution against the live engine frame ...
    raise NotImplementedError("finalize nav against engine in Step 2")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Implement `drive_to_goal` against the live engine and run**

This is the one task with genuine integration discovery (maze nav + rotation indexing). Implement
`drive_to_goal` concretely while iterating against the offline engine:
1. From the first frame, find the avatar (the object near grid (47,34) that translates when you issue a
   move; verify with one test move) and the wall color (the dominant non-background structural color
   forming the maze).
2. Write a grid BFS over non-wall, in-bounds cells (step = PITCH) returning an action sequence from the
   avatar cell to a target cell; execute it, re-querying the avatar centroid each step to stay robust.
3. Sequence: BFS→ROT_TILE; step onto it `target_visits` times (each entry = +1 rotation; re-step by
   leaving and re-entering); BFS→a cell orthogonally adjacent to GOAL; record `from_mkey = masked_key(pol, grid)`;
   issue the action toward GOAL; `win = (next obs.levels_completed increased) or obs.state==WIN`.
4. Validate rotation indexing by reading the engine's rotation attribute if reachable from `env` (the
   underlying ARCBaseGame; attribute `cklxociuu` per the ground-truth memo) — optional cross-check.

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/invisible_state_probe.py`
Expected: `from_keys_equal=True  outcomes=[True, False]` and `RESULT: invisible-state OBSERVABLE+RESOLVABLE by active probing`.
If `from_keys_equal=False`, the masking is NOT actually merging the rotations (re-examine `_key`); if
outcomes are not `[True, False]`, fix rotation indexing (start idx is 270°=idx3; visits add +1 mod 4,
so idx0 needs the start-state's existing offset accounted for — adjust `target_visits` to hit idx0 vs idx1).

- [ ] **Step 3: Commit**

```bash
git add scripts/invisible_state_probe.py
git commit -q -m "feat(phase-p): scripted offline ls20 active probe (invisible rotation observability)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

---

## Task 4: Run map + probe, apply pre-registered verdict

**Files:** none (execution + results capture).

- [ ] **Step 1: Full passive map**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/invisible_state_map.py 30000 2>&1 | tee /tmp/invisible_state_map.log`
Expected: a per-game classification line for all 16 games (~10-15 min), the ls20 anchor line, the
walled-games-with-INVISIBLE_STATE line, and `/tmp/invisible_state_map.json`.

- [ ] **Step 2: Active probe (always run; it is the make-or-break ls20 evidence)**

Run: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/invisible_state_probe.py 2>&1 | tee /tmp/invisible_state_probe.log`
Expected: the `from_keys_equal=... outcomes=...` line and the RESULT verdict.

- [ ] **Step 3: Apply the pre-registered verdict (spec §4.5) and record**

From the two logs, determine which verdict holds:
- **(i)** anchor fires (ls20 INVISIBLE_STATE > 0) AND ≥2 other walled games show INVISIBLE_STATE at gates → BUILD the history-augmentation fix next.
- **(ii)** anchor fires but only ls20 → narrow lever.
- **(iii)** passive anchor does NOT fire BUT the active probe confirms (`from_keys_equal=True, outcomes=[True,False]`) → passive detection dead; next design = ACTIVE PROBING.
- **(iv)** neither passive nor active confirms on ls20 → the framing is wrong; back to the drawing board.
Append a `## Results (YYYY-MM-DD)` section to the spec with the per-game table, the anchor + probe
outcomes, and the chosen verdict; commit.

```bash
git add docs/superpowers/specs/2026-06-21-invisible-state-prevalence-map-design.md
git commit -q -m "docs(phase-p): invisible-state map results + verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CWT3ggknxkRDBFuQDuw41B"
```

- [ ] **Step 4: STOP for user review**

Do NOT start the fix (history-augmentation or active probing). Present the map table + anchor + probe +
verdict and wait for direction.

---

## Task 5: Record the finding to memory

**Files:** `~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-research-findings.md`

- [ ] **Step 1: Append a dated finding** recording: the per-game INVISIBLE_STATE / OVER_MERGE /
UNEXPLAINED counts, the ls20 anchor + active-probe outcomes, the chosen verdict (i-iv), and the
direction it sets for the next session. Cross-link `[[arcagi3-ls20-mechanic-ground-truth]]`. No commit
(memory lives outside the repo).

---

## Self-Review

**Spec coverage:**
- §4.1 violation multimap → Task 1 `find_violations`; driver logging → Task 2. ✓
- §4.2 history-resolvability + over-merge tag → Task 1 `classify` (OVER_MERGE via ukey, INVISIBLE_STATE via feature resolve); feature family (obj×mod{2,3,4}) → Task 2 `_features`. ✓
- §4.3 ls20 anchor → Task 2 main anchor line + Task 4. ✓
- §4.4 minimal active probe → Task 3. ✓
- §4.5 output + pre-registered verdicts i-iv → Task 4. ✓
- §5 deliverables (analysis module, map driver, probe, results, memory) → Tasks 1-5. ✓
- §6/§7 self-validation + confounds separated (over_merge vs invisible vs unexplained) → Task 1 categories. ✓

**Placeholder scan:** Task 1 is fully concrete (complete code + tests). Tasks 2-3 contain explicit,
labelled INTEGRATION-ITERATION points (avatar API finalization; ls20 nav + rotation indexing) rather
than vague TODOs — these are genuine live-engine discovery steps, scoped with exact verification
criteria and fallbacks, and are executed inline. The one `NotImplementedError` in Task 3 Step 1 is the
declared seam that Step 2 fills against the engine (with exact expected output and failure-mode
guidance). This is the honest shape of an integration probe, not a hidden gap.

**Type consistency:** `IS.Transition(mkey, ukey, action, next_mkey, feats)` used identically in tests
(Task 1) and the driver (Task 2). `classify`/`summary`/`find_violations` signatures match across tasks.
Feature naming `obj{i}_mod{N}` produced in Task 2 `_features` and consumed generically by Task 1
`_resolves` (which is feature-name-agnostic). The probe (Task 3) uses the same `SalienceExplorer._key`
masking as the map for an apples-to-apples from-key.
```
