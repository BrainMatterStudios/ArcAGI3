"""Frontier-explorer floor graft — depth lever #1 for the duck38-v12 lane.

Provenance: port of our own tested patch 11 ("frontier-graph substrate +
stall grinder", submission/_duck_patched/duck_patches.py:1618-2545, with its
978-line test file) with three deliberate deltas for this lane:

  1. MASK: patch 8's HudMaskTracker is replaced by a vendored wavefront
     volatility mask (src/arcagi3/hud_mask.py) learned per session — hidden
     games are anonymized, so the static per-game HUD table cannot key them.
  2. TRIGGER: the watchdog-stall dependency is dropped (it was verified
     dormant in every A/B — grinder_engagements=0); the LEVEL-AGE trigger is
     the sole trigger and is self-contained here.
  3. VETO: the no-op edge veto is NOT ported in v1 (smaller risk surface;
     the grinder is the depth lever, the veto is an efficiency nicety).

Mechanism: every executed engine action records an edge (node, plan) -> node'
where node = (level, crc32 of the volatility-masked grid). When a level has
soaked >= EXPLORER_AGE_ACTIONS scored actions or >= EXPLORER_AGE_TURNS LLM
turns with zero completions this run, a scripted frontier walk takes over at
engine speed: execute untested plans (per-node component-centroid clicks +
basic actions) or BFS through known safe edges to the nearest node with
untested plans, until level-up / budget / danger. Burned actions on a
never-completed level cost exactly 0 score; the grinder permanently
disengages for any level completed this run. On an unlock, the next user
prompt carries the action tail that crossed the boundary so the model can
extract the mechanic (the lever's actual value).

Economics (measured 2026-08-18, depth study wf_9809fe85): 6 of 10 zero-score
dev games have verified mechanical L1 wins in 8-26 actions
(submission/_explorer_floor/fixtures/); 67% of LLM calls are inspection-only
and the box funds ~3.7 levels/game — the floor attacks both.

Fail-open: every hook wraps its body in try/except; EXPLORER=0 disables
everything at call time; install() presence-gates every bundle symbol and
declines cleanly on drift.
"""

from __future__ import annotations

import threading as _threading
import zlib
from collections import deque
from typing import Any

_TLS = _threading.local()

# v3: at most N grinds run concurrently across ALL game sessions — the scored
# run plays ~110 games at concurrency 28 against ONE shared gateway server,
# and a single grind saturates it (~130 act/s). Unlimited concurrent grinds
# starve every game's normal play (the suspected mechanism behind the 1.33
# live draw). Non-blocking: a session that cannot acquire simply skips this
# opportunity and retries on a later trigger poll.
_GRIND_GATE = _threading.BoundedSemaphore(1)


# --------------------------------------------------------------------------
# env knobs
# --------------------------------------------------------------------------

def _enabled() -> bool:
    import os

    return os.environ.get("EXPLORER", "1").strip() not in {"0", "false", "False"}


def _env_int(name: str, default: int) -> int:
    import os

    try:
        return int(float(os.environ.get(name, "") or default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# dynamic HUD mask (vendored wavefront volatility learner)
# --------------------------------------------------------------------------

class VolatilityMask:
    # NOTE: freeze() locks the mask before any BFS uses its cells for state
    # signatures — a mask that keeps learning mid-search makes signatures
    # drift, aliasing states and corrupting the dedup (measured 2026-08-18:
    # dc22 frontier "exhausted" at 136 states vs the fixture's 123-state win).
    """Cells that change on >= threshold of observed transitions are HUD.

    Numpy-free port of src/arcagi3/hud_mask.py WavefrontHUDMask (threshold
    0.85, min_steps 5, plus edge-band promotion at mean freq >= 0.5)."""

    def __init__(self, threshold: float = 0.85, min_steps: int = 5, edge_band: int = 4):
        self.threshold = threshold
        self.min_steps = min_steps
        self.edge_band = edge_band
        self.counts: dict[tuple[int, int], int] = {}
        self.tick_rows: dict[int, int] = {}
        self.tick_cols: dict[int, int] = {}
        self.steps = 0
        self._prev: list[list[int]] | None = None
        self._frozen: list[tuple[int, int]] | None = None

    def freeze(self) -> None:
        self._frozen = self._compute_cells()

    def unfreeze(self) -> None:
        self._frozen = None

    def update(self, grid: Any) -> None:
        rows = [list(r) for r in grid]
        prev = self._prev
        if prev is not None and len(prev) == len(rows) and prev and len(prev[0]) == len(rows[0]):
            self.steps += 1
            h, w = len(rows), len(rows[0])
            band = self.edge_band
            interior_changed = False
            band_rows_changed: set[int] = set()
            band_cols_changed: set[int] = set()
            for y, (pr, nr) in enumerate(zip(prev, rows)):
                for x, (a, b) in enumerate(zip(pr, nr)):
                    if a != b:
                        self.counts[(y, x)] = self.counts.get((y, x), 0) + 1
                        edge_row = y < band or y >= h - band
                        edge_col = x < band or x >= w - band
                        if edge_row:
                            band_rows_changed.add(y)
                        if edge_col:
                            band_cols_changed.add(x)
                        if not edge_row and not edge_col:
                            interior_changed = True
            # SLOW-TICK RULE (July "HUD breaks frame identity" + the community
            # band measurement): a border row/col that changes while the
            # interior changes NOTHING is a status band, however slow its
            # tick. Two such events mask it permanently for this session.
            if not interior_changed:
                for y in band_rows_changed:
                    self.tick_rows[y] = self.tick_rows.get(y, 0) + 1
                for x in band_cols_changed:
                    self.tick_cols[x] = self.tick_cols.get(x, 0) + 1
        self._prev = rows

    def mask_cells(self) -> list[tuple[int, int]]:
        if self._frozen is not None:
            return self._frozen
        return self._compute_cells()

    def _compute_cells(self) -> list[tuple[int, int]]:
        if self.steps < self.min_steps or self._prev is None:
            return []
        h = len(self._prev)
        w = len(self._prev[0]) if h else 0
        out = {cell for cell, n in self.counts.items() if n / self.steps >= self.threshold}
        for y, n in self.tick_rows.items():
            if n >= 2:
                out.update((y, x) for x in range(w))
        for x, n in self.tick_cols.items():
            if n >= 2:
                out.update((y, x) for y in range(h))
        # edge-band promotion: a border row/col whose mean change freq >= 0.5
        band = self.edge_band
        for y in list(range(min(band, h))) + list(range(max(0, h - band), h)):
            total = sum(self.counts.get((y, x), 0) for x in range(w))
            if w and total / (self.steps * w) >= 0.5:
                out.update((y, x) for x in range(w))
        for x in list(range(min(band, w))) + list(range(max(0, w - band), w)):
            total = sum(self.counts.get((y, x), 0) for y in range(h))
            if h and total / (self.steps * h) >= 0.5:
                out.update((y, x) for y in range(h))
        return sorted(out)


# --------------------------------------------------------------------------
# frontier graph (verbatim logic from patch 11)
# --------------------------------------------------------------------------

def _background_color(rows: list) -> int:
    counts: dict[int, int] = {}
    for row in rows:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else 0


def _components(rows: list) -> list[dict[str, Any]]:
    h = len(rows)
    w = len(rows[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps: list[dict[str, Any]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc]:
                continue
            color = rows[sr][sc]
            queue = deque([(sr, sc)])
            seen[sr][sc] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and rows[nr][nc] == color:
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            rs = [cell[0] for cell in cells]
            cs = [cell[1] for cell in cells]
            comps.append({
                "color": color,
                "size": len(cells),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "centroid": (sum(cs) // len(cells), sum(rs) // len(cells)),
            })
    return comps


def compact_click_candidates(rows: list, limit: int = 20) -> list[tuple[int, int]]:
    """BFS-phase click targets: compact non-background components sized 2-80,
    smallest first, spatially deduped — the generic form of the July per-game
    configs (dc22 panel buttons, ka59 units, m0r0 pieces all sit in this band;
    1-px noise and room-sized regions are excluded)."""
    if not rows:
        return []
    background = _background_color(rows)
    comps = [c for c in _components(rows)
             if c["color"] != background and 6 <= c["size"] <= 80]
    comps.sort(key=lambda c: c["size"])
    out: list[tuple[int, int]] = []
    for comp in comps:
        x, y = comp["centroid"]
        # Click must land ON the component's own color (July panel_targets
        # rule) — but ka59-class units carry a different-colored 1px marker at
        # their centroid, so SNAP to the nearest own-color cell instead of
        # skipping (the July dynamic_targets lesson).
        if not (0 <= y < len(rows) and 0 <= x < len(rows[0])):
            continue
        if rows[y][x] != comp["color"]:
            r0, c0, r1, c1 = comp["bbox"]
            best = None
            for yy in range(r0, r1 + 1):
                for xx in range(c0, c1 + 1):
                    if rows[yy][xx] == comp["color"]:
                        d = abs(yy - y) + abs(xx - x)
                        if best is None or d < best[0]:
                            best = (d, xx, yy)
            if best is None:
                continue
            x, y = best[1], best[2]
        if all(abs(x - u[0]) + abs(y - u[1]) > 2 for u in out):
            out.append((x, y))
        if len(out) >= limit:
            break
    return out


def click_candidates(rows: list, limit: int = 64, step: int = 8) -> list[tuple[int, int]]:
    """(x, y) points for ACTION6: component centroids by button-likeness, then
    a coarse sweep (poby's verified FLAT ordering; hard tiers regressed)."""
    if not rows:
        return []
    h = len(rows)
    w = len(rows[0]) if h else 0
    background = _background_color(rows)
    scored = []
    for comp in _components(rows):
        if comp["color"] == background:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        bbox_area = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        fill = comp["size"] / bbox_area
        scored.append((fill / (1.0 + comp["size"]), comp["centroid"]))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, point in scored:
        if point not in seen:
            seen.add(point)
            out.append(point)
    half = max(1, step // 2)
    for y in range(half, h, step):
        for x in range(half, w, step):
            if (x, y) not in seen:
                seen.add((x, y))
                out.append((x, y))
    return out[:limit]


class FrontierGraph:
    """Transition graph over (level, masked-grid crc32) nodes. Pure data."""

    NOOP_MIN_OBS = 2

    def __init__(self, click_limit: int = 64, click_step: int = 8, max_nodes: int = 60000):
        self.click_limit = int(click_limit)
        self.click_step = int(click_step)
        self.max_nodes = int(max_nodes)
        self.nodes: dict[tuple, dict[str, Any]] = {}
        self.edges: dict[tuple, dict[tuple, dict[str, Any]]] = {}

    @staticmethod
    def masked_rows(grid: Any, mask_cells: Any) -> list[list[int]]:
        rows = [list(row) for row in grid]
        for cell in mask_cells or []:
            y, x = int(cell[0]), int(cell[1])
            if 0 <= y < len(rows) and 0 <= x < len(rows[y]):
                rows[y][x] = 0
        return rows

    def node_key(self, level: int, grid: Any, mask_cells: Any) -> tuple:
        rows = self.masked_rows(grid, mask_cells)
        h = len(rows)
        w = len(rows[0]) if h else 0
        payload = bytearray()
        for row in rows:
            for value in row:
                payload.append(int(value) & 0xFF)
        return (int(level), h, w, zlib.crc32(bytes(payload)))

    def ensure_node(self, key: tuple, action_names: list[str], masked_rows: list) -> None:
        if key in self.nodes or len(self.nodes) >= self.max_nodes:
            return
        plans: list[tuple] = [(n,) for n in action_names if n not in ("RESET", "ACTION6")]
        if "ACTION6" in action_names:
            plans.extend(("ACTION6", x, y) for x, y in click_candidates(
                masked_rows, limit=self.click_limit, step=self.click_step))
        self.nodes[key] = {"plans": plans, "dead": set()}
        self.edges.setdefault(key, {})

    def untested_plans(self, key: tuple) -> list[tuple]:
        node = self.nodes.get(key)
        if not node:
            return []
        tested = self.edges.get(key, {})
        dead = node["dead"]
        return [p for p in node["plans"] if p not in tested and p not in dead]

    def pop_untested(self, key: tuple) -> tuple | None:
        plans = self.untested_plans(key)
        return plans[0] if plans else None

    def mark_dead(self, key: tuple, plan: tuple) -> None:
        node = self.nodes.get(key)
        if node is not None:
            node["dead"].add(plan)

    def record(self, prev_key: tuple, plan: tuple, next_key: tuple, *,
               changed: bool, level_up: bool, game_over: bool) -> None:
        edges = self.edges.setdefault(prev_key, {})
        edge = edges.get(plan)
        if edge is None:
            edge = {"dest": next_key, "count": 0, "noop": 0,
                    "level_up": False, "danger": False, "consistent": True}
            edges[plan] = edge
        edge["count"] += 1
        if edge["dest"] != next_key:
            edge["consistent"] = False
            edge["dest"] = next_key
        if level_up:
            edge["level_up"] = True
        if game_over:
            edge["danger"] = True
        if next_key == prev_key and not changed and not level_up and not game_over:
            edge["noop"] += 1

    def edge_dest(self, key: tuple, plan: tuple) -> tuple | None:
        edge = self.edges.get(key, {}).get(plan)
        if edge is None or not edge["consistent"]:
            return None
        return edge["dest"]

    def bfs_to_frontier(self, start: tuple) -> list[tuple] | None:
        queue: deque = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node != start and self.untested_plans(node):
                return path
            for plan, edge in self.edges.get(node, {}).items():
                if not edge["consistent"] or edge["danger"] or edge["level_up"]:
                    continue
                if edge["noop"] == edge["count"] and edge["count"] > 0:
                    continue
                dest = edge["dest"]
                if dest is None or dest in visited or dest[0] != start[0]:
                    continue
                visited.add(dest)
                queue.append((dest, path + [plan]))
        return None

    def diagnostics(self) -> dict[str, int]:
        return {"nodes": len(self.nodes),
                "edges": sum(len(e) for e in self.edges.values())}


def _plan_key(action_name: str, action_data: dict[str, Any] | None) -> tuple:
    if action_name == "ACTION6":
        data = action_data or {}
        return ("ACTION6", int(data.get("x", 0)), int(data.get("y", 0)))
    return (action_name,)


# --------------------------------------------------------------------------
# per-session state
# --------------------------------------------------------------------------

def _session_state(session: Any) -> dict[str, Any]:
    xs = getattr(session, "_xpl_state", None)
    if xs is None:
        xs = {
            "graph": FrontierGraph(
                click_limit=_env_int("EXPLORER_CLICK_LIMIT", 64),
                click_step=_env_int("EXPLORER_CLICK_STEP", 8),
                max_nodes=_env_int("EXPLORER_MAX_NODES", 60000),
            ),
            "mask": VolatilityMask(),
            "level_actions": {},
            "completed_levels": set(),
            "grinds_per_level": {},
            "grind_exhausted": set(),
            "grind_unlocked_levels": set(),
            "grinding": False,
            "age_marks": {},
            "worker_thread": None,
            "diag": {"grinder_engagements": 0, "grinder_actions": 0,
                     "levels_unlocked_by_grinder": 0, "narrations_injected": 0},
        }
        session._xpl_state = xs
    return xs


def explorer_diagnostics(session: Any) -> dict[str, Any]:
    xs = getattr(session, "_xpl_state", None)
    if xs is None:
        return {}
    out = dict(xs["diag"])
    out.update(xs["graph"].diagnostics())
    return out


def _level_age(session: Any, xs: dict[str, Any], level: int) -> tuple[int, int]:
    step = int(getattr(session, "analysis_step", 0) or 0)
    actions = int(xs["level_actions"].get(level, 0))
    mark = xs["age_marks"].get(level)
    if mark is None:
        mark = {"actions": actions, "step": step}
        xs["age_marks"][level] = mark
    return actions - mark["actions"], step - mark["step"]


def _reset_age_mark(session: Any, xs: dict[str, Any], level: int) -> None:
    xs["age_marks"][level] = {
        "actions": int(xs["level_actions"].get(level, 0)),
        "step": int(getattr(session, "analysis_step", 0) or 0),
    }


# --------------------------------------------------------------------------
# grinder
# --------------------------------------------------------------------------

def _maybe_grind(session: Any) -> None:
    from inference.framework import solver

    xs = _session_state(session)
    if xs["grinding"]:
        return
    # engine actions only ever from the session's own worker thread (recorded
    # by the _execute_action wrapper on its first executed action)
    if xs["worker_thread"] is None or xs["worker_thread"] != _threading.get_ident():
        return
    game = session.game
    run = getattr(game, "game_run", None)
    if run is None or run.state != "playing":
        return
    if session.stop_event.is_set():
        return
    if solver._is_run_complete(game) or solver._is_engine_game_over(game):
        return

    # v5 GRIND-OWNED GATE (supersedes v4's zero-progress gate): grind actions
    # only destroy value on levels the LLM completes (measured: 100 -> 1.2 on
    # the official scorer, no subsidy pool on partial runs). A game may grind
    # iff NO level was ever completed by the LLM — either untouched (cached
    # levels_completed == 0 pre-grind) or every completion is grinder-owned.
    llm_completed = xs["completed_levels"] - xs["grind_unlocked_levels"]
    if llm_completed:
        return
    if int(game.current_state.levels_completed) > 0 and not xs["grind_unlocked_levels"]:
        return
    level = solver._level_number(game)
    if level in xs["completed_levels"] or level in xs["grind_exhausted"]:
        return
    actions_since, turns_since = _level_age(session, xs, level)
    age_actions = _env_int("EXPLORER_AGE_ACTIONS", 120)
    age_turns = _env_int("EXPLORER_AGE_TURNS", 10)
    if not ((age_actions > 0 and actions_since >= age_actions)
            or (age_turns > 0 and turns_since >= age_turns)):
        return

    if session.runtime_limit_reached():
        return
    if (session.solver.max_actions_per_game is not None
            and session.action_count >= session.solver.max_actions_per_game):
        return
    soft_remaining = session.solver.soft_time_remaining_seconds()
    if soft_remaining is not None and soft_remaining < 120.0:
        return
    if xs["grinds_per_level"].get(level, 0) >= _env_int("EXPLORER_GRIND_MAX_PER_LEVEL", 1):
        return
    if not _GRIND_GATE.acquire(blocking=False):
        return  # another game is grinding — retry on a later poll
    try:
        xs["grinds_per_level"][level] = xs["grinds_per_level"].get(level, 0) + 1
        xs["diag"]["grinder_engagements"] += 1
        _grind(session, xs, level)
        _reset_age_mark(session, xs, level)
    finally:
        _GRIND_GATE.release()


def _grind(session: Any, xs: dict[str, Any], level: int) -> None:
    """v5 GRIND-TO-WIN-AND-BANK, direct on the engine wrapper.

    Per level: reset-replay BFS (warmup mask learning, phased candidates) — the
    algorithm validated offline (4/5 zero-games) and on the scored GPU (4/4).
    v5 additions, on GRIND-OWNED games only (no LLM-completed level ever):
      1. CONTINUATION — after an unlock, if budget remains, keep grinding the
         next level in the same engagement (BFS finds MINIMAL paths, so a
         fully-searchable game can be cleared end-to-end, e.g. tu93 in 185
         replay actions).
      2. SELF-BANKING — on a full WIN, immediately RESET (post-WIN reset opens
         a fresh play; engine-verified) and replay the concatenated minimal
         per-level sequences: play #2 scores near-baseline efficiency and the
         card takes the max. A zero game becomes a near-perfect game.
    Self-harm theorem (measured 2026-08-19): grind actions only destroy value
    on levels the LLM completes; grind-owned completions have no such term.
    """
    import time as _time

    import arcengine

    graph: FrontierGraph = xs["graph"]
    game = session.game
    game_id = getattr(game.game_run, "game_id", "?")
    env = getattr(game, "env", None)
    if env is None:
        xs["grind_exhausted"].add(level)
        print(f"[explorer] {game_id}: no engine wrapper — grind unavailable", flush=True)
        return

    budget = max(1, _env_int("EXPLORER_GRIND_BUDGET", 500000))
    p0_budget = max(1, _env_int("EXPLORER_PHASE0_BUDGET", 60000))
    time_cap_s = max(30, _env_int("EXPLORER_GRIND_TIME_S", 900))
    # once a grind-owned game starts unlocking, the alternative use of its box
    # is zero — extend the cap (tu93's measured full win took ~3300s)
    owned_time_cap_s = max(time_cap_s, _env_int("EXPLORER_OWNED_TIME_S", 5400))
    p0_time_s = max(10, _env_int("EXPLORER_PHASE0_TIME_S", 240))
    max_depth = max(1, _env_int("EXPLORER_MAX_DEPTH", 30))
    mask: VolatilityMask = xs["mask"]

    print(f"[explorer] {game_id}: engaging on level {level} — grind-to-win, "
          f"budget {budget}/{time_cap_s}s", flush=True)
    xs["grinding"] = True
    executed = 0
    stop_reason = "budget"
    grind_t0 = _time.monotonic()
    resp = None
    val2name = {a.value: a.name for a in arcengine.GameAction}

    def out_of_budget() -> str | None:
        if session.stop_event.is_set():
            return "cancelled"
        if executed >= budget:
            return "budget"
        cap = owned_time_cap_s if xs["grind_unlocked_levels"] else time_cap_s
        if _time.monotonic() - grind_t0 >= cap:
            return "time_cap"
        if session.runtime_limit_reached():
            return "runtime_cap"
        soft_remaining = session.solver.soft_time_remaining_seconds()
        if soft_remaining is not None and soft_remaining < 60.0:
            return "soft_time"
        return None

    def grid_of(r) -> list[list[int]]:
        data = r.frame[-1]
        rows = data.tolist() if hasattr(data, "tolist") else data
        return [[int(c) for c in row] for row in rows]

    def step(plan: tuple):
        nonlocal executed, resp
        try:
            action_id = arcengine.GameAction.from_name(plan[0])
        except Exception:  # noqa: BLE001
            return None
        data = ({"x": int(plan[1]), "y": int(plan[2])}
                if plan[0] == "ACTION6" and len(plan) == 3 else {})
        try:
            r = env.step(action_id, data=data)
        except Exception:  # noqa: BLE001
            return None
        if r is None or not r.frame:
            return None
        executed += 1
        xs["diag"]["grinder_actions"] += 1
        mask.update(grid_of(r))
        resp = r
        return r

    def plan_text(plan: tuple) -> str:
        if plan[0] == "ACTION6" and len(plan) == 3:
            return f"CLICK(x={plan[1]},y={plan[2]})"
        return str(plan[0])

    def narrate(text: str) -> None:
        try:
            _TLS.narration = {"text": text, "diag": xs["diag"]}
        except Exception:  # noqa: BLE001
            pass

    def cands(phase: int) -> list[tuple]:
        names = [val2name[v] for v in (resp.available_actions or []) if v in val2name]
        plans = [(n,) for n in names if n not in ("RESET", "ACTION6")]
        if phase >= 1 and "ACTION6" in names:
            plans.extend(("ACTION6", x, y) for x, y in compact_click_candidates(
                graph.masked_rows(grid_of(resp), mask.mask_cells()),
                limit=_env_int("EXPLORER_BFS_CLICKS", 20)))
        return plans

    def bfs_one_level(base_levels: int) -> tuple[str, list[tuple] | None]:
        """BFS from the current level's start; returns (reason, winning seq)."""

        def sig() -> tuple:
            return graph.node_key(base_levels, grid_of(resp), mask.mask_cells())

        def unlocked() -> bool:
            return resp is not None and int(resp.levels_completed) != base_levels

        for phase in (0, 1):
            phase_exec0 = executed
            phase_t0 = _time.monotonic()
            seen: set[tuple] = set()
            queue: deque = deque([[]])
            if step(("RESET",)) is None:
                return "reset_failed", None
            if unlocked():
                return "level_unlocked", [("RESET",)]
            seen.add(sig())
            while queue:
                reason = out_of_budget()
                if reason:
                    return reason, None
                if phase == 0 and (executed - phase_exec0 >= p0_budget
                                   or _time.monotonic() - phase_t0 >= p0_time_s):
                    break  # phase-0 share spent: move to click phase
                seq = queue.popleft()
                if len(seq) >= max_depth:
                    continue
                if step(("RESET",)) is None:
                    return "reset_failed", None
                dead = False
                for plan in seq:
                    if step(plan) is None or resp.state == arcengine.GameState.GAME_OVER:
                        dead = True
                        break
                    if unlocked():
                        return "level_unlocked", seq
                if dead:
                    continue
                for plan in cands(phase):
                    reason = out_of_budget()
                    if reason:
                        return reason, None
                    if step(("RESET",)) is None:
                        return "reset_failed", None
                    dead = False
                    for prev in seq:
                        if step(prev) is None or resp.state == arcengine.GameState.GAME_OVER:
                            dead = True
                            break
                    if dead:
                        continue
                    if step(plan) is None:
                        continue
                    if unlocked():
                        return "level_unlocked", seq + [plan]
                    if resp.state == arcengine.GameState.GAME_OVER:
                        continue
                    k = sig()
                    if k not in seen:
                        seen.add(k)
                        queue.append(seq + [plan])
        return "frontier_exhausted", None

    def bank_replay(level_seqs: list[list[tuple]]) -> bool:
        """Post-WIN fresh play, replay the minimal per-level sequences."""
        r = step(("RESET",))
        if r is None or int(r.levels_completed) != 0 or r.state == arcengine.GameState.WIN:
            print(f"[explorer] {game_id}: bank ABORT — post-WIN RESET did not "
                  "open a fresh play", flush=True)
            return False
        total = 0
        for seq in level_seqs:
            for plan in seq:
                if step(plan) is None:
                    print(f"[explorer] {game_id}: bank ABORT — engine refused a "
                          "replay step", flush=True)
                    return False
                total += 1
                if resp.state == arcengine.GameState.GAME_OVER:
                    print(f"[explorer] {game_id}: bank ABORT — replay died", flush=True)
                    return False
        won = resp.state == arcengine.GameState.WIN
        print(f"[explorer] {game_id}: bank replay {'WON' if won else 'ended short'} "
              f"in {total} actions", flush=True)
        return won

    try:
        if step(("RESET",)) is None:
            stop_reason = "reset_failed"
            return

        # WARMUP: learn volatility (incl. the slow-tick band rule) — no freeze.
        warm = [val2name[v] for v in (resp.available_actions or [])
                if v in val2name and val2name[v] not in ("RESET", "ACTION6")]
        for _ in range(max(1, _env_int("EXPLORER_WARMUP_ROUNDS", 6))):
            if out_of_budget():
                break
            for wn in warm:
                if step((wn,)) is None:
                    break
                if resp.state == arcengine.GameState.GAME_OVER:
                    step(("RESET",))
        print(f"[explorer] {game_id}: warmup done, mask "
              f"{len(mask.mask_cells())} cells (live-learning)", flush=True)

        level_seqs: list[list[tuple]] = []
        while True:
            base_levels = int(resp.levels_completed)
            reason, seq = bfs_one_level(base_levels)
            if reason != "level_unlocked" or seq is None:
                stop_reason = reason
                if reason == "frontier_exhausted" and not level_seqs:
                    xs["grind_exhausted"].add(level)
                break
            level_seqs.append(seq)
            unlocked_level = base_levels + 1
            xs["diag"]["levels_unlocked_by_grinder"] += 1
            xs["grind_unlocked_levels"].add(unlocked_level)
            xs["completed_levels"].add(unlocked_level)
            print(f"[explorer] {game_id}: level {unlocked_level} UNLOCKED "
                  f"({len(seq)} actions minimal, {executed} spent)", flush=True)
            if resp.state == arcengine.GameState.WIN:
                banked = bank_replay(level_seqs)
                xs["diag"]["games_won_by_grinder"] = xs["diag"].get("games_won_by_grinder", 0) + 1
                narrate(
                    f"[EXPLORER] This game was fully SOLVED by automated search"
                    f"{' and re-played optimally on a fresh attempt (score banked)' if banked else ''}. "
                    "No further actions are needed on this game; if prompted, do "
                    "not reset or replay — the result is already recorded.")
                stop_reason = "game_won" + ("_banked" if banked else "")
                return
            narrate(
                f"[EXPLORER UNLOCK] Level {unlocked_level} was just unlocked by "
                "an automated exhaustive search, NOT by your plan. The minimal "
                f"winning sequence from the level start was: "
                f"{', '.join(plan_text(pl) for pl in seq)}. The LAST action "
                "crossed the boundary. Infer this game's mechanic from that "
                "sequence and apply it deliberately on the current level.")
            if out_of_budget():
                stop_reason = out_of_budget() or "budget"
                break
    finally:
        # leave at the current level's start for the LLM (not after a win/bank)
        if stop_reason not in ("cancelled", "reset_failed") and not stop_reason.startswith("game_won"):
            step(("RESET",))
        xs["grinding"] = False
        try:
            session.write_runtime_state()
        except Exception:  # noqa: BLE001
            pass
        print(f"[explorer] {game_id}: grind ended ({stop_reason}) after {executed} "
              f"actions, {len(xs['grind_unlocked_levels'])} grinder unlocks total", flush=True)


def install() -> str:
    """Wrap the session class + prompt seam. Presence-gated; declines on drift."""
    import arcengine

    from inference.framework import solver

    try:
        from inference.agent import tool_agent as tool_agent_mod
    except Exception:  # noqa: BLE001
        tool_agent_mod = None

    session_cls = getattr(solver, "_HarnessGameSession", None)
    if session_cls is None:
        return "explorer: FAIL (_HarnessGameSession not found)"
    missing = [n for n in ("_execute_action", "should_stop", "timing_payload")
               if not hasattr(session_cls, n)]
    if missing:
        return f"explorer: SKIP (session lacks {missing})"
    missing = [n for n in ("_grid_from_state", "_engine_action_names", "_level_number",
                           "_is_run_complete", "_is_engine_game_over",
                           "_format_action_display")
               if not hasattr(solver, n)]
    if missing:
        return f"explorer: SKIP (solver lacks {missing})"
    if not hasattr(arcengine, "ActionInput") or not hasattr(arcengine, "GameAction"):
        return "explorer: SKIP (arcengine drift)"
    if getattr(session_cls._execute_action, "_xpl_patched", False):
        return "explorer: SKIP (already applied)"

    original_execute = session_cls._execute_action
    original_should_stop = session_cls.should_stop

    def _execute_action(self: Any, action: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        pre = None
        if _enabled():
            try:
                xs = _session_state(self)
                if xs["worker_thread"] is None:
                    xs["worker_thread"] = _threading.get_ident()
                state = self.game.current_state
                grid = solver._grid_from_state(state)
                mask = xs["mask"].mask_cells()
                level = solver._level_number(self.game)
                key = xs["graph"].node_key(level, grid, mask)
                xs["graph"].ensure_node(key, solver._engine_action_names(self.game),
                                        xs["graph"].masked_rows(grid, mask))
                pre = (xs, key, level, int(state.levels_completed))
            except Exception:  # noqa: BLE001
                pre = None
        payload = original_execute(self, action, *args, **kwargs)
        if pre is not None and isinstance(payload, dict) and payload.get("executed"):
            try:
                xs, prev_key, level, levels_before = pre
                state = self.game.current_state
                levels_after = int(state.levels_completed)
                xs["level_actions"][level] = xs["level_actions"].get(level, 0) + 1
                for done in range(levels_before + 1, levels_after + 1):
                    xs["completed_levels"].add(done)
                grid = solver._grid_from_state(state)
                xs["mask"].update(grid)
                name = getattr(getattr(action, "id", None), "name", "")
                if name and name != "RESET":
                    post_key = xs["graph"].node_key(
                        solver._level_number(self.game), grid, xs["mask"].mask_cells())
                    xs["graph"].ensure_node(
                        post_key, solver._engine_action_names(self.game),
                        xs["graph"].masked_rows(grid, xs["mask"].mask_cells()))
                    xs["graph"].record(
                        prev_key,
                        _plan_key(name, dict(getattr(action, "data", None) or {})),
                        post_key,
                        changed=bool(payload.get("board_changed")),
                        level_up=levels_after > levels_before,
                        game_over=bool(payload.get("game_over")),
                    )
            except Exception:  # noqa: BLE001
                pass
        return payload

    def should_stop(self: Any) -> bool:
        if _enabled():
            try:
                _maybe_grind(self)
            except Exception:  # noqa: BLE001
                pass
        return original_should_stop(self)

    _execute_action._xpl_patched = True  # type: ignore[attr-defined]
    should_stop._xpl_patched = True  # type: ignore[attr-defined]
    session_cls._execute_action = _execute_action
    session_cls.should_stop = should_stop

    narration_note = ""
    agent_cls = getattr(tool_agent_mod, "ToolAgent", None) if tool_agent_mod else None
    if agent_cls is None or not hasattr(agent_cls, "_build_user_prompt"):
        narration_note = " (narration unavailable)"
    elif not getattr(agent_cls._build_user_prompt, "_xpl_narration_patched", False):
        original_build = agent_cls._build_user_prompt

        def _build_user_prompt(self: Any, *args: Any, **kwargs: Any) -> str:
            prompt = original_build(self, *args, **kwargs)
            if not _enabled():
                return prompt
            try:
                staged = getattr(_TLS, "narration", None)
                if staged:
                    _TLS.narration = None
                    staged["diag"]["narrations_injected"] += 1
                    print("[explorer] win-path narration injected", flush=True)
                    return prompt + "\n" + staged["text"]
            except Exception:  # noqa: BLE001
                pass
            return prompt

        _build_user_prompt._xpl_narration_patched = True  # type: ignore[attr-defined]
        agent_cls._build_user_prompt = _build_user_prompt

    return "explorer: OK" + narration_note
