"""Scripted-teacher episodes: wrong-hypothesis -> contradiction -> revision -> win.

For every game this module produces a per-level list of TURNS. A turn is:
    {"actions": [engine action dicts],
     "phase":   "orient"|"misstep"|"probe"|"revise"|"execute",
     "sem":     structured semantics for the narrator (narrate.py),
     "assert":  {"level_completed": bool}}

The wrong hypotheses come from a per-family confusion library modeled on the
ACTUAL wrong hypotheses the duck exhibited in the 2026-08-03 unlock diagnosis:
reading removal/teleport as failure or destruction (wa30, lf52), treating the
A5 reset as a pure setback (g50t), ignoring the legend as a goal spec (sk48,
tr87, lf52), single-avatar assumption (m0r0), copy-the-target (tr87), and
walk-straight-to-the-goal (ls20, dc22). Misstep actions are REAL: their
outcomes are computed by the family simulator here and revalidated against the
real engine at corpus-build time (level boundaries + no deaths + final WIN).

~25% of games are CLEAN solves (no scripted misstep) so the revision arc is
not itself a memorized template; within revision games the arc plays out on
the first level where the mechanic can bite, and later levels apply the
learned model.

The turn `sem` records the state AT TURN START (captured before the turn's
actions execute) so the narration is always grounded in the frame the model
actually sees.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from families import DIR_ACTIONS, _named_mask
from families import _nav_bfs, _GATE_TOKENS, _push_bfs
from families import _click_display_point
from families_v2 import (
    ACTION_OF_DELTA,
    DELTA,
    DIR_NAME,
    CarrySim,
    MirrorSim,
    ReplaySim,
    RulesSim,
    _bfs_path,
    _cycle_actions,
    _find,
    _path_actions,
    solve_carry_from,
    solve_mirror_from,
    solve_rules_from,
)

CLEAN_FRACTION = 0.25


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _act(aid: int) -> dict[str, Any]:
    return {"id": f"ACTION{aid}"}


def _dir_act(d: tuple[int, int]) -> dict[str, Any]:
    return {"id": DIR_ACTIONS[d]}


def _aid(action: dict[str, Any]) -> int:
    return int(action["id"][-1])


def _chunks(seq: list[Any], rng: random.Random, lo: int, hi: int) -> list[list[Any]]:
    out, i = [], 0
    while i < len(seq):
        k = rng.randint(lo, hi)
        out.append(seq[i : i + k])
        i += k
    return out


def _scale_offset(width_px: int, height_px: int) -> tuple[int, int, int]:
    scale = min(64 // width_px, 64 // height_px)
    return scale, (64 - width_px * scale) // 2, (64 - height_px * scale) // 2


def cell_box(spec: dict[str, Any], r: int, c: int) -> dict[str, int]:
    """Display-pixel bounding box of a logical cell (for grounded narration)."""
    if spec["family"] == "rules":
        return {"r0": r, "r1": r, "c0": c, "c1": c}
    if spec["family"] == "click":
        g = spec["grid"]
        scale, ox, oy = _scale_offset(g, g)
        return {"r0": r * scale + oy, "r1": (r + 1) * scale + oy - 1,
                "c0": c * scale + ox, "c1": (c + 1) * scale + ox - 1}
    cell = spec["cell_px"]
    w, h = spec["cols"] * cell, spec["rows"] * cell
    scale, ox, oy = _scale_offset(w, h)
    return {"r0": (r * cell) * scale + oy, "r1": ((r + 1) * cell) * scale + oy - 1,
            "c0": (c * cell) * scale + ox, "c1": ((c + 1) * cell) * scale + ox - 1}


def _steps_from_path(path: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {"disp": DIR_NAME[ACTION_OF_DELTA[(r1 - r0, c1 - c0)]],
         "frm": [r0, c0], "to": [r1, c1], "kind": "move"}
        for (r0, c0), (r1, c1) in zip(path, path[1:])
    ]


def _auto_steps(spec: dict[str, Any], sim: Any, actions: list[dict]) -> list[dict]:
    """Per-action ground-truth transitions for a chunk, derived by simulating
    the chunk on a copy of the sim. Family-specific detail (both mirror dots,
    ghost advance, grab/release, cursor/glyph changes, key pickups, pushes)
    so the narrator can enumerate genuinely contentful predictions."""
    fam = spec["family"]
    s = copy.deepcopy(sim)
    steps: list[dict] = []
    for a in actions:
        if a["id"] == "ACTION6":
            g = spec["grid"]
            scale, ox, oy = _scale_offset(g, g)
            steps.append({"disp": "MOUSE", "kind": "click", "frm": None,
                          "to": [(a["y"] - oy) // scale, (a["x"] - ox) // scale]})
            _step_click_sim(s, a)
            continue
        aid = _aid(a)
        disp = DIR_NAME.get(aid, "SPACE")
        if fam == "replay":
            before, gb = list(s.pos), list(s.ghost_pos) if s.ghost_pos else None
            ev = s.step(aid)
            st: dict[str, Any] = {"disp": disp, "frm": before, "to": list(s.pos)}
            if ev.get("banked"):
                st.update(kind="bank")
            elif aid == 5:
                st.update(kind="probe")
            elif ev["blocked_by"]:
                st.update(kind="bump", note=ev["blocked_by"])
            else:
                st.update(kind="move")
            if ev.get("ghost_moved"):
                st["ghost_to"] = list(s.ghost_pos)
            st["door_open"] = s.door_open()
            steps.append(st)
        elif fam == "carry":
            before = list(s.pos)
            ev = s.step(aid)
            if ev.get("grabbed") is not None:
                steps.append({"disp": "SPACE", "kind": "grab", "frm": before,
                              "to": before})
            elif ev.get("released") is not None:
                drop = [s.blocks[ev["released"]][0], s.blocks[ev["released"]][1]]
                steps.append({"disp": "SPACE", "kind": "release", "frm": before,
                              "to": drop})
            elif aid == 5:
                steps.append({"disp": "SPACE", "kind": "probe", "frm": before,
                              "to": before})
            elif ev["blocked_by"]:
                steps.append({"disp": disp, "kind": "bump", "frm": before,
                              "to": before, "note": ev["blocked_by"]})
            else:
                steps.append({"disp": disp, "kind": "move", "frm": before,
                              "to": list(s.pos)})
        elif fam == "mirror":
            a0, b0 = list(s.a), list(s.b)
            s.step(aid)
            steps.append({"disp": disp, "kind": "pair", "frm": a0, "to": list(s.a),
                          "b_frm": b0, "b_to": list(s.b),
                          "gap": [s.a[0] - s.b[0], s.a[1] - s.b[1]]})
        elif fam == "rules":
            c0 = s.cursor
            g0 = s.working[c0]
            s.step(aid)
            if aid in (3, 4):
                steps.append({"disp": disp, "kind": "cursor", "frm": [0, c0],
                              "to": [0, s.cursor]})
            else:
                steps.append({"disp": disp, "kind": "cycle", "frm": None, "to": None,
                              "col": c0, "g0": g0, "g1": s.working[c0],
                              "mark": s.correct(c0)})
        elif fam == "nav":
            before = list(s.pos)
            ev = s.step(aid)
            if ev["blocked_by"]:
                steps.append({"disp": disp, "kind": "bump", "frm": before,
                              "to": before, "note": ev["blocked_by"]})
            elif ev.get("opened"):
                steps.append({"disp": disp, "kind": "pickup", "frm": before,
                              "to": list(s.pos), "token": ev["opened"]})
            else:
                steps.append({"disp": disp, "kind": "move", "frm": before,
                              "to": list(s.pos)})
        elif fam == "push":
            before = list(s.pos)
            ev = s.step(aid)
            if ev.get("pushed") is not None:
                bi = ev["pushed"]
                steps.append({"disp": disp, "kind": "push", "frm": before,
                              "to": list(s.pos), "box_to": list(s.boxes[bi])})
            elif ev["blocked"]:
                steps.append({"disp": disp, "kind": "bump", "frm": before,
                              "to": before})
            else:
                steps.append({"disp": disp, "kind": "move", "frm": before,
                              "to": list(s.pos)})
        else:
            steps.append({"disp": disp, "kind": "move", "frm": None, "to": None})
    return steps


class _LevelScript:
    """Collects turns; captures sim state BEFORE each turn's actions run."""

    def __init__(self, spec, li, sim, state_fn, step_fn, evidence) -> None:
        self.spec = spec
        self.li = li
        self.sim = sim
        self.state_fn = state_fn
        self.step_fn = step_fn
        self.evidence = evidence
        self.turns: list[dict[str, Any]] = []

    def emit(self, actions, phase, hyp, *, steps=None, react=None, wrong_pred=None,
             considered=None, revision=None, plan_note=None, closes_level=False,
             plan_ahead=None):
        exec_now = len(actions)
        if steps is None:
            # derive ground-truth step predictions; when the full remaining
            # level plan is supplied, narrate the WHOLE route and execute a
            # prefix (plan-then-execute, like real deliberation)
            steps = _auto_steps(self.spec, self.sim,
                                list(plan_ahead) if plan_ahead else list(actions))
        sem = {
            "family": self.spec["family"], "level": self.li + 1, "phase": phase,
            "hyp": hyp, "state": self.state_fn(self.sim),
            "steps": steps or [],
            "react": react, "wrong_pred": wrong_pred,
            "considered": considered or [], "revision": revision,
            "plan_note": plan_note, "evidence": list(self.evidence),
            "closes_level": closes_level, "clean": False,
            "exec_now": exec_now,
        }
        self.turns.append({"actions": list(actions), "phase": phase, "sem": sem,
                           "assert": {"level_completed": closes_level}})
        for a in actions:
            self.step_fn(self.sim, a)


# ===========================================================================
# v1-family simulators (mirror the v1 render_game templates exactly)
# ===========================================================================


class NavSim:
    def __init__(self, grid: list[str]) -> None:
        self.grid = grid
        self.rows, self.cols = len(grid), len(grid[0])
        self.pos = _find(grid, "P")[0]
        self.goal = _find(grid, "G")[0]
        self.opened: set[str] = set()
        self.won = False

    def step(self, action_id: int) -> dict[str, Any]:
        ev = {"moved": False, "blocked_by": None, "opened": None}
        d = DELTA.get(action_id)
        if d is None or self.won:
            return ev
        nr, nc = self.pos[0] + d[0], self.pos[1] + d[1]
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            ev["blocked_by"] = "wall"
            return ev
        ch = self.grid[nr][nc]
        if ch == "#":
            ev["blocked_by"] = "wall"
            return ev
        if ch in ("A", "B", "S") and ch.lower() not in self.opened:
            ev["blocked_by"] = "door"
            return ev
        self.pos = (nr, nc)
        ev["moved"] = True
        if ch in ("a", "b", "s") and ch not in self.opened:
            self.opened.add(ch)
            ev["opened"] = ch
        elif ch == "G":
            self.won = True
        return ev


class ClickSim:
    def __init__(self, spec: dict[str, Any], level: dict[str, Any]) -> None:
        self.spec = spec
        self.objects = level["objects"]
        self.size = spec["obj_size"]
        self.masks = {n: _named_mask(n, self.size) for n in
                      ("full", "ring", "plus", "diamond", "notch", "tee", "u")}
        self.alive = set(range(len(self.objects)))
        self.remaining = {i for i, o in enumerate(self.objects) if o["target"]}
        self.armed = not spec["gate"]
        self.lives = spec["lives"]
        self.won = False

    def hit(self, gx: int, gy: int) -> int | None:
        for i in self.alive:
            o = self.objects[i]
            dx, dy = gx - o["x"], gy - o["y"]
            if 0 <= dx < self.size and 0 <= dy < self.size and self.masks[o["shape"]][dy][dx]:
                return i
        return None

    def click(self, gx: int, gy: int) -> dict[str, Any]:
        ev = {"effect": "none"}
        if self.won:
            return ev
        if not self.armed:
            ax, ay = self.spec["gate_xy"]
            gp = self.spec["gate_px"]
            if ax <= gx < ax + gp and ay <= gy < ay + gp:
                self.armed = True
                ev["effect"] = "armed"
            return ev
        i = self.hit(gx, gy)
        if i is None:
            return ev
        if i in self.remaining:
            self.remaining.discard(i)
            self.alive.discard(i)
            ev["effect"] = "removed"
            if not self.remaining:
                self.won = True
        else:
            self.lives -= 1
            ev["effect"] = "life_lost"
        return ev


class PushSim:
    def __init__(self, grid: list[str]) -> None:
        self.rows, self.cols = len(grid), len(grid[0])
        self.walls, self.pads, self.boxes = set(), set(), []
        self.pos = (0, 0)
        for r in range(self.rows):
            for c in range(self.cols):
                ch = grid[r][c]
                if ch == "#":
                    self.walls.add((r, c))
                elif ch == "T":
                    self.pads.add((r, c))
                elif ch == "B":
                    self.boxes.append((r, c))
                elif ch == "O":
                    self.pads.add((r, c))
                    self.boxes.append((r, c))
                elif ch == "P":
                    self.pos = (r, c)
        self.won = False

    def _free(self, cell) -> bool:
        r, c = cell
        return (0 <= r < self.rows and 0 <= c < self.cols
                and cell not in self.walls and cell not in self.boxes)

    def step(self, action_id: int) -> dict[str, Any]:
        ev = {"moved": False, "pushed": None, "blocked": False}
        d = DELTA.get(action_id)
        if d is None or self.won:
            return ev
        t = (self.pos[0] + d[0], self.pos[1] + d[1])
        if t in self.walls or not (0 <= t[0] < self.rows and 0 <= t[1] < self.cols):
            ev["blocked"] = True
            return ev
        if t in self.boxes:
            bt = (t[0] + d[0], t[1] + d[1])
            if not self._free(bt):
                ev["blocked"] = True
                return ev
            bi = self.boxes.index(t)
            self.boxes[bi] = bt
            ev["pushed"] = bi
        self.pos = t
        ev["moved"] = True
        if self.pads and self.pads.issubset(set(self.boxes)):
            self.won = True
        return ev


# per-family step adapters for _LevelScript
def _step_grid_sim(sim, action):
    sim.step(_aid(action))


def _step_click_sim(sim, action):
    # translate the display click back to grid coords via the shared transform
    g = sim.spec["grid"]
    scale, ox, oy = _scale_offset(g, g)
    gx = (action["x"] - ox) // scale
    gy = (action["y"] - oy) // scale
    sim.click(gx, gy)


# ===========================================================================
# family: replay
# ===========================================================================


def _replay_state(sim: ReplaySim) -> dict[str, Any]:
    return {
        "pos": list(sim.pos), "start": list(sim.start), "plate": list(sim.plate),
        "goal": list(sim.goal), "door": [list(d) for d in sorted(sim.doors)],
        "door_open": sim.door_open(), "banked": sim.banked,
        "ghost": list(sim.ghost_pos) if sim.ghost_pos else None,
        "recorded": len(sim.recorded),
        "rows": sim.rows, "cols": sim.cols,
    }


def _ep_replay_level(spec, li, rng, clean, evidence) -> list[dict[str, Any]]:
    grid = spec["levels"][li]["map"]
    sim = ReplaySim(grid)
    sc = _LevelScript(spec, li, sim, _replay_state, _step_grid_sim, evidence)
    door = next(iter(sim.doors))
    dn = (door[0], door[1] - 1)
    bump = _dir_act((0, 1))
    hyp_true = {"id": "ghost_replay", "kind": "true"}

    def passable_closed(cell):
        return grid[cell[0]][cell[1]] not in ("#", "D")

    def passable_open(cell):
        return grid[cell[0]][cell[1]] != "#"

    if not clean:
        # misstep 1: head for the "gap" as if it were open floor
        p_appr = _bfs_path(passable_closed, sim.pos, dn, sim.rows, sim.cols)
        appr_acts = _path_actions(p_appr)
        cut = max(1, len(appr_acts) // 2)
        sc.emit(appr_acts[:cut], "orient", {"id": "direct_walk", "kind": "wrong"},
                steps=_steps_from_path(p_appr[: cut + 1]),
                plan_note={"id": "head_for_gap"})
        sc.emit(appr_acts[cut:] + [bump], "misstep",
                {"id": "direct_walk", "kind": "wrong"},
                steps=_steps_from_path(p_appr[cut:]) + [
                    {"disp": "RIGHT", "frm": list(dn), "to": list(door),
                     "kind": "move", "note": "through_gap"}],
                wrong_pred={"id": "pass_gap"})
        evidence.append(("door_solid", {"door": list(door)}))
        # probe: the lone tinted cell — walk onto it (door visibly opens)
        p_plate = _bfs_path(passable_closed, sim.pos, sim.plate, sim.rows, sim.cols)
        sc.emit(_path_actions(p_plate), "probe", {"id": "plate_latch", "kind": "wrong"},
                react={"ok": False, "id": "blocked_door"},
                steps=_steps_from_path(p_plate),
                considered=["direct_walk"],
                plan_note={"id": "try_plate"})
        evidence.append(("door_opened_on_plate", {"plate": list(sim.plate)}))
        # misstep 2: assume the plate latched the door open; walk to the gap
        p_back = _bfs_path(passable_closed, sim.pos, dn, sim.rows, sim.cols)
        sc.emit(_path_actions(p_back) + [bump], "misstep",
                {"id": "plate_latch", "kind": "wrong"},
                react={"ok": True, "id": "door_open_on_plate"},
                steps=_steps_from_path(p_back) + [
                    {"disp": "RIGHT", "frm": list(dn), "to": list(door),
                     "kind": "move", "note": "through_gap"}],
                wrong_pred={"id": "latch_stays"})
        evidence.append(("door_closed_off_plate", {}))
        # revise: the plate must be HELD; return to it and try SPACE
        p_plate2 = _bfs_path(passable_closed, sim.pos, sim.plate, sim.rows, sim.cols)
        sc.emit(_path_actions(p_plate2) + [_act(5)], "revise",
                {"id": "a5_probe", "kind": "tentative"},
                react={"ok": False, "id": "door_needs_holder"},
                steps=_steps_from_path(p_plate2) + [
                    {"disp": "SPACE", "frm": list(sim.plate), "to": list(sim.plate),
                     "kind": "bank"}],
                revision={"frm": "plate_latch", "to": "a5_probe"},
                considered=["plate_latch", "direct_walk"],
                plan_note={"id": "press_space_on_plate"})
        evidence.append(("a5_ghost_spawn", {"start": list(sim.start)}))
    else:
        p1 = _bfs_path(passable_closed, sim.pos, sim.plate, sim.rows, sim.cols)
        acts = _path_actions(p1) + [_act(5)]
        chunked = _chunks(acts, rng, 4, 7)
        consumed = 0
        for i, ch in enumerate(chunked):
            phase = "orient" if (i == 0 and li == 0) else "execute"
            sc.emit(ch, phase, hyp_true, plan_note={"id": "record_then_bank"},
                    plan_ahead=acts[consumed:])
            consumed += len(ch)

    assert sim.banked, "replay episode: bank did not happen"
    ghost_len = len(sim.ghost_path)
    p2 = _bfs_path(passable_open, sim.start, sim.goal, sim.rows, sim.cols)
    k = p2.index(door)
    fillers = max(0, ghost_len - k)
    if fillers % 2:
        fillers += 1
    filler_dir = next(
        (dr, dc) for dr, dc in DELTA.values()
        if grid[sim.start[0] + dr][sim.start[1] + dc] in ".L")
    back_d = (-filler_dir[0], -filler_dir[1])
    filler_acts = [
        _dir_act(filler_dir if i % 2 == 0 else back_d) for i in range(fillers)
    ]
    goal_acts = _path_actions(p2)
    tail = filler_acts + goal_acts
    plan_note = {"id": "wait_then_cross",
                 "params": {"fillers": fillers, "ghost_len": ghost_len}}
    react = None if clean else {"ok": True, "id": "ghost_marching"}
    if len(tail) > 14:
        cut = len(filler_acts) if fillers else len(tail) // 2
        cut = max(1, min(cut, len(tail) - 1))
        sc.emit(tail[:cut], "execute", hyp_true, react=react,
                plan_note=plan_note, plan_ahead=tail)
        sc.emit(tail[cut:], "execute", hyp_true,
                plan_note=plan_note, plan_ahead=tail[cut:], closes_level=True)
    else:
        sc.emit(tail, "execute", hyp_true, react=react,
                plan_note=plan_note, plan_ahead=tail, closes_level=True)
    assert sim.won, "replay episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns


# ===========================================================================
# family: carry
# ===========================================================================


def _carry_state(sim: CarrySim) -> dict[str, Any]:
    return {
        "pos": list(sim.pos),
        "blocks": [list(b) for i, b in enumerate(sim.blocks) if i != sim.carrying],
        "zone": [list(z) for z in sorted(sim.zone)],
        "carrying": sim.carrying is not None,
        "facing": list(sim.facing) if sim.facing else None,
        "rows": sim.rows, "cols": sim.cols,
    }


def _ep_carry_level(spec, li, rng, clean, evidence) -> list[dict[str, Any]]:
    grid = spec["levels"][li]["map"]
    sim = CarrySim(grid)
    sc = _LevelScript(spec, li, sim, _carry_state, _step_grid_sim, evidence)
    hyp_true = {"id": "carry", "kind": "true"}

    if not clean:
        obstacles = set(sim.blocks)

        def passable(cell):
            return not sim._wall(cell) and cell not in obstacles

        best = None
        for i, b in enumerate(sim.blocks):
            for dr, dc in DELTA.values():
                y = (b[0] - dr, b[1] - dc)
                if sim._wall(y) or y in obstacles:
                    continue
                path = _bfs_path(passable, sim.pos, y, sim.rows, sim.cols)
                if path and (best is None or len(path) < len(best[0])):
                    best = (path, (dr, dc), i)
        assert best is not None
        path, d, bi = best
        block_cell = list(sim.blocks[bi])
        sc.emit(_path_actions(path) + [_dir_act(d), _act(5)], "misstep",
                {"id": "space_probe", "kind": "tentative"},
                steps=_steps_from_path(path) + [
                    {"disp": DIR_NAME[ACTION_OF_DELTA[d]], "frm": list(path[-1]),
                     "to": block_cell, "kind": "bump", "note": "into_block"},
                    {"disp": "SPACE", "frm": list(path[-1]), "to": block_cell,
                     "kind": "probe"}],
                plan_note={"id": "probe_space_on_block"})
        evidence.append(("block_vanished", {"cell": block_cell}))
        # move away, SPACE on empty floor — the "destroyed" block REAPPEARS
        move_a = None
        scratch = copy.deepcopy(sim)
        for dd in DELTA.values():
            t = (scratch.pos[0] + dd[0], scratch.pos[1] + dd[1])
            f = (t[0] + dd[0], t[1] + dd[1])
            if (not scratch._wall(t) and scratch._block_at(t) is None
                    and not scratch._wall(f) and scratch._block_at(f) is None):
                move_a = dd
                break
        assert move_a is not None, "carry misstep: no room to test the release"
        sc.emit([_dir_act(move_a), _act(5)], "probe",
                {"id": "space_destroy", "kind": "wrong"},
                react={"ok": False, "id": "block_vanished"},
                steps=[{"disp": DIR_NAME[ACTION_OF_DELTA[move_a]], "frm": list(sim.pos),
                        "to": [sim.pos[0] + move_a[0], sim.pos[1] + move_a[1]],
                        "kind": "move"},
                       {"disp": "SPACE", "frm": None, "to": None,
                        "kind": "probe", "note": "empty_space"}],
                wrong_pred={"id": "space_noop_empty"},
                considered=["space_destroy"])
        evidence.append(("block_reappeared", {}))
        # revise to the carry model; deliver everything from the current state
        rest = solve_carry_from(copy.deepcopy(sim))
        chunked = _chunks(rest, rng, 4, 8)
        consumed = 0
        for i, ch in enumerate(chunked):
            last = i == len(chunked) - 1
            phase = "revise" if i == 0 else "execute"
            sc.emit(ch, phase, hyp_true,
                    react={"ok": False, "id": "block_reappeared"} if i == 0 else None,
                    revision={"frm": "space_destroy", "to": "carry"} if i == 0 else None,
                    considered=["space_destroy"] if i == 0 else [],
                    plan_note={"id": "deliver_blocks"}, closes_level=last,
                    plan_ahead=rest[consumed:])
            consumed += len(ch)
    else:
        sol = solve_carry_from(copy.deepcopy(sim))
        chunked = _chunks(sol, rng, 4, 8)
        consumed = 0
        for i, ch in enumerate(chunked):
            last = i == len(chunked) - 1
            phase = "orient" if (i == 0 and li == 0) else "execute"
            sc.emit(ch, phase, hyp_true,
                    plan_note={"id": "deliver_blocks"}, closes_level=last,
                    plan_ahead=sol[consumed:])
            consumed += len(ch)
    assert sim.won, "carry episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns


# ===========================================================================
# family: mirror
# ===========================================================================


def _mirror_state(sim: MirrorSim) -> dict[str, Any]:
    return {"a": list(sim.a), "b": list(sim.b), "axis": sim.axis,
            "gap": [sim.a[0] - sim.b[0], sim.a[1] - sim.b[1]],
            "rows": sim.rows, "cols": sim.cols}


def _ep_mirror_level(spec, li, rng, clean, evidence) -> list[dict[str, Any]]:
    grid = spec["levels"][li]["map"]
    sim = MirrorSim(grid, spec["axis"])
    sc = _LevelScript(spec, li, sim, _mirror_state, _step_grid_sim, evidence)
    hyp_true = {"id": "coupled_mirror", "kind": "true"}

    did_misstep = False
    if not clean:
        greedy: list[dict[str, Any]] = []
        scratch = copy.deepcopy(sim)
        for _ in range(3):
            cands = sorted(
                DELTA.items(),
                key=lambda kv: abs(scratch.a[0] + kv[1][0] - scratch.b[0])
                + abs(scratch.a[1] + kv[1][1] - scratch.b[1]))
            step_done = False
            for aid, d in cands:
                probe = copy.deepcopy(scratch)
                ev = probe.step(aid)
                if probe.won or not (ev["a_moved"] or ev["b_moved"]):
                    continue
                scratch.step(aid)
                greedy.append(_act(aid))
                step_done = True
                break
            if not step_done:
                break
        if greedy:
            did_misstep = True
            sc.emit(greedy, "misstep", {"id": "single_dot", "kind": "wrong"},
                    steps=[{"disp": DIR_NAME[_aid(a)], "frm": None, "to": None,
                            "kind": "move"} for a in greedy],
                    wrong_pred={"id": "close_distance"},
                    plan_note={"id": "approach_twin"})
            evidence.append(("twin_mirrors", {"axis": sim.axis}))

    sol = solve_mirror_from(copy.deepcopy(sim))
    assert sol is not None
    chunked = _chunks(sol, rng, 3, 7)
    consumed = 0
    for i, ch in enumerate(chunked):
        last = i == len(chunked) - 1
        if did_misstep and i == 0:
            phase = "revise"
        elif i == 0 and li == 0 and not did_misstep:
            phase = "orient"
        else:
            phase = "execute"
        sc.emit(ch, phase, hyp_true,
                react={"ok": False, "id": "twin_mirrored"} if (did_misstep and i == 0) else None,
                revision={"frm": "single_dot", "to": "coupled_mirror"} if (did_misstep and i == 0) else None,
                considered=["single_dot"] if (did_misstep and i == 0) else [],
                plan_note={"id": "jam_then_merge"}, closes_level=last,
                plan_ahead=sol[consumed:])
        consumed += len(ch)
    assert sim.won, "mirror episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns


# ===========================================================================
# family: rules
# ===========================================================================


def _rules_state_fn(spec):
    def fn(sim: RulesSim) -> dict[str, Any]:
        return {"cursor": sim.cursor, "target": list(sim.target),
                "working": list(sim.working), "marks": sim.marks(),
                "rules": dict(spec["rules"]), "n": sim.n}
    return fn


def _ep_rules_level(spec, li, rng, clean, evidence) -> list[dict[str, Any]]:
    lvl = spec["levels"][li]
    sim = RulesSim(spec["rules"], lvl["target"], lvl["working"])
    sc = _LevelScript(spec, li, sim, _rules_state_fn(spec), _step_grid_sim, evidence)
    hyp_true = {"id": "legend_map", "kind": "true"}

    if not clean:
        copy_acts = _cycle_actions(sim.working[0], sim.target[0])
        sc.emit(copy_acts, "misstep", {"id": "copy_target", "kind": "wrong"},
                steps=[{"disp": DIR_NAME[_aid(a)], "frm": None, "to": None,
                        "kind": "cycle"} for a in copy_acts],
                wrong_pred={"id": "copy_confirm"},
                plan_note={"id": "match_row_above"})
        evidence.append(("no_mark_on_copy", {"cell": 0}))
        fix_acts = _cycle_actions(sim.working[0], spec["rules"][sim.target[0]])
        sc.emit(fix_acts, "revise", hyp_true,
                react={"ok": False, "id": "no_mark"},
                steps=[{"disp": DIR_NAME[_aid(a)], "frm": None, "to": None,
                        "kind": "cycle"} for a in fix_acts],
                revision={"frm": "copy_target", "to": "legend_map"},
                considered=["copy_target"],
                plan_note={"id": "apply_rule_cell0"})
        evidence.append(("mark_on_rule", {"cell": 0}))

    rest = solve_rules_from(copy.deepcopy(sim))
    chunked = _chunks(rest, rng, 4, 9)
    consumed = 0
    for i, ch in enumerate(chunked):
        last = i == len(chunked) - 1
        phase = "orient" if (clean and i == 0 and li == 0) else "execute"
        sc.emit(ch, phase, hyp_true,
                react={"ok": True, "id": "mark_appeared"} if (not clean and i == 0) else None,
                plan_note={"id": "apply_rules_all"}, closes_level=last,
                plan_ahead=rest[consumed:])
        consumed += len(ch)
    assert sim.won, "rules episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns


# ===========================================================================
# family: nav (v1)
# ===========================================================================


def _find_multi(grid, chars):
    return [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch in chars]


def _nav_state_fn(grid):
    def fn(sim: NavSim) -> dict[str, Any]:
        return {
            "pos": list(sim.pos), "goal": list(sim.goal),
            "keys": [list(k) for k in _find_multi(grid, "ab")],
            "switches": [list(s) for s in _find_multi(grid, "s")],
            "doors": [list(d) for d in _find_multi(grid, "ABS")],
            "hazards": [list(h) for h in _find_multi(grid, "x")],
            "opened": sorted(sim.opened),
            "rows": sim.rows, "cols": sim.cols,
        }
    return fn


def _nav_solve_from(grid, pos):
    g = [list(row) for row in grid]
    for r, c in _find_multi(grid, "P"):
        g[r][c] = "."
    g[pos[0]][pos[1]] = "P"
    return _nav_bfs(["".join(row) for row in g], _GATE_TOKENS)


def _ep_nav_level(spec, li, rng, clean, evidence):
    grid = spec["levels"][li]["map"]
    sim = NavSim(grid)
    sc = _LevelScript(spec, li, sim, _nav_state_fn(grid), _step_grid_sim, evidence)
    doors = _find_multi(grid, "ABS")
    hyp_true = {"id": "keys_doors", "kind": "true"}
    did_misstep = False

    if not clean and doors:
        def passable(cell):
            return grid[cell[0]][cell[1]] in ".PGABS"

        p = _bfs_path(passable, sim.pos, sim.goal, sim.rows, sim.cols)
        if p is not None:
            di = next((i for i, cell in enumerate(p)
                       if grid[cell[0]][cell[1]] in "ABS"), None)
            if di is not None and di >= 2:
                did_misstep = True
                appr = p[:di]
                appr_acts = _path_actions(appr)
                dcell = p[di]
                bump_d = (dcell[0] - appr[-1][0], dcell[1] - appr[-1][1])
                cut = max(1, len(appr_acts) // 2)
                sc.emit(appr_acts[:cut], "orient",
                        {"id": "direct_goal", "kind": "wrong"},
                        steps=_steps_from_path(appr[: cut + 1]),
                        plan_note={"id": "straight_to_goal"})
                sc.emit(appr_acts[cut:] + [_dir_act(bump_d)], "misstep",
                        {"id": "direct_goal", "kind": "wrong"},
                        steps=_steps_from_path(appr[cut:]) + [
                            {"disp": DIR_NAME[ACTION_OF_DELTA[bump_d]],
                             "frm": list(appr[-1]), "to": list(dcell),
                             "kind": "move", "note": "through_gap"}],
                        wrong_pred={"id": "pass_gap"})
                evidence.append(("door_solid", {"door": list(dcell)}))

    path = _nav_solve_from(grid, sim.pos)
    assert path is not None
    acts = _path_actions(path)
    chunked = _chunks(acts, rng, 3, 7)
    consumed = 0
    for i, ch in enumerate(chunked):
        last = i == len(chunked) - 1
        if did_misstep and i == 0:
            phase = "revise"
        elif i == 0 and li == 0 and not did_misstep:
            phase = "orient"
        else:
            phase = "execute"
        sc.emit(ch, phase, hyp_true,
                react={"ok": False, "id": "blocked_door"} if (did_misstep and i == 0) else None,
                revision={"frm": "direct_goal", "to": "keys_doors"} if (did_misstep and i == 0) else None,
                considered=["direct_goal"] if (did_misstep and i == 0) else [],
                plan_note={"id": "key_then_goal"}, closes_level=last,
                plan_ahead=acts[consumed:])
        consumed += len(ch)
    assert sim.won, "nav episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns, did_misstep


# ===========================================================================
# family: click (v1)
# ===========================================================================


def _click_state_fn(spec):
    def fn(sim: ClickSim) -> dict[str, Any]:
        objs = [
            {"cell": [o["y"], o["x"]], "shape": o["shape"], "color": o["color"],
             "target": o["target"]}
            for i, o in enumerate(sim.objects) if i in sim.alive
        ]
        return {"objects": objs, "rule": spec["rule"], "gate": spec["gate"],
                "armed": sim.armed, "lives": sim.lives,
                "remaining": len(sim.remaining), "grid": spec["grid"]}
    return fn


def _click_action(spec, o) -> dict[str, Any]:
    s = spec["obj_size"]
    mask = _named_mask(o["shape"], s)
    best = min(((r, c) for r in range(s) for c in range(s) if mask[r][c]),
               key=lambda rc: abs(rc[0] - s // 2) + abs(rc[1] - s // 2))
    dx, dy = _click_display_point(spec, o["x"] + best[1], o["y"] + best[0])
    return {"id": "ACTION6", "x": dx, "y": dy}


def _ep_click_level(spec, li, rng, clean, evidence) -> list[dict[str, Any]]:
    lvl = spec["levels"][li]
    sim = ClickSim(spec, lvl)
    sc = _LevelScript(spec, li, sim, _click_state_fn(spec), _step_click_sim, evidence)
    targets = sorted((o for o in lvl["objects"] if o["target"]),
                     key=lambda o: (o["y"], o["x"]))
    hyp_true = {"id": "legend_target", "kind": "true"}

    if not clean and spec["gate"]:
        first = targets[0]
        sc.emit([_click_action(spec, first)], "misstep",
                {"id": "click_match", "kind": "wrong"},
                steps=[{"disp": "MOUSE", "frm": None,
                        "to": [first["y"], first["x"]], "kind": "click"}],
                wrong_pred={"id": "target_vanish"},
                plan_note={"id": "click_matching"})
        evidence.append(("click_ignored", {"cell": [first["y"], first["x"]]}))
        gp = spec["gate_px"]
        gx, gy = spec["gate_xy"]
        dx, dy = _click_display_point(spec, gx + gp // 2, gy + gp // 2)
        sc.emit([{"id": "ACTION6", "x": dx, "y": dy}], "probe",
                {"id": "gate_arm", "kind": "tentative"},
                react={"ok": False, "id": "click_no_effect"},
                steps=[{"disp": "MOUSE", "frm": None, "to": [gy, gx],
                        "kind": "click_arm"}],
                considered=["click_match"],
                plan_note={"id": "probe_button"})
        evidence.append(("gate_armed", {"cell": [gy, gx]}))
    elif not clean:
        distractors = [o for o in lvl["objects"] if not o["target"]]
        d0 = rng.choice(distractors)
        sc.emit([_click_action(spec, d0)], "misstep",
                {"id": "misread_legend", "kind": "wrong"},
                steps=[{"disp": "MOUSE", "frm": None,
                        "to": [d0["y"], d0["x"]], "kind": "click"}],
                wrong_pred={"id": "target_vanish"},
                plan_note={"id": "click_matching"})
        evidence.append(("life_lost", {"cell": [d0["y"], d0["x"]]}))

    if spec["gate"] and not sim.armed:
        # the gate re-arms every level: known mechanic now — arm it first
        gp = spec["gate_px"]
        gx, gy = spec["gate_xy"]
        dx, dy = _click_display_point(spec, gx + gp // 2, gy + gp // 2)
        sc.emit([{"id": "ACTION6", "x": dx, "y": dy}], "execute",
                {"id": "gate_arm", "kind": "true"},
                steps=[{"disp": "MOUSE", "frm": None, "to": [gy, gx],
                        "kind": "click_arm"}],
                plan_note={"id": "arm_first"})
    click_seq = [(o, _click_action(spec, o)) for o in targets]
    chunked = _chunks(click_seq, rng, 1, 3)
    consumed = 0
    for i, ch in enumerate(chunked):
        last = i == len(chunked) - 1
        if not clean and i == 0:
            phase = "revise"
            if spec["gate"]:
                react = {"ok": True, "id": "gate_armed"}
                revision = {"frm": "click_match", "to": "gate_arm"}
            else:
                react = {"ok": False, "id": "life_lost"}
                revision = {"frm": "misread_legend", "to": "legend_target"}
        else:
            phase = "orient" if (clean and i == 0 and li == 0) else "execute"
            react = revision = None
        sc.emit([a for _o, a in ch], phase, hyp_true,
                react=react, revision=revision,
                plan_note={"id": "click_targets_order"}, closes_level=last,
                plan_ahead=[a for _o, a in click_seq[consumed:]])
        consumed += len(ch)
    assert sim.won, "click episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns


# ===========================================================================
# family: push (v1)
# ===========================================================================


def _push_state(sim: PushSim) -> dict[str, Any]:
    return {"pos": list(sim.pos), "boxes": [list(b) for b in sim.boxes],
            "pads": [list(p) for p in sorted(sim.pads)],
            "rows": sim.rows, "cols": sim.cols}


def _ep_push_level(spec, li, rng, clean, evidence) -> list[dict[str, Any]]:
    grid = spec["levels"][li]["map"]
    sim = PushSim(grid)
    sc = _LevelScript(spec, li, sim, _push_state, _step_grid_sim, evidence)
    hyp_true = {"id": "push_plan", "kind": "true"}
    did_misstep = False

    if not clean:
        cand = None
        for bi, b in enumerate(sim.boxes):
            for d in DELTA.values():
                stand = (b[0] - d[0], b[1] - d[1])
                dest = (b[0] + d[0], b[1] + d[1])
                if not sim._free(stand) or not sim._free(dest):
                    continue
                blocked = set(sim.boxes)

                def passable(cell):
                    return (cell not in sim.walls and cell not in blocked
                            and 0 <= cell[0] < sim.rows and 0 <= cell[1] < sim.cols)

                path = _bfs_path(passable, sim.pos, stand, sim.rows, sim.cols)
                if path is None:
                    continue
                nb = [dest if i == bi else bb for i, bb in enumerate(sim.boxes)]
                # solvable from the post-push-and-step-back position?
                moves = _push_bfs(sim.walls, sim.rows, sim.cols, stand,
                                  tuple(sorted(nb)), frozenset(sim.pads))
                if moves is None:
                    continue
                cand = (path, d, bi)
                break
            if cand:
                break
        if cand:
            did_misstep = True
            path, d, bi = cand
            box_from = list(sim.boxes[bi])
            back_d = (-d[0], -d[1])
            sc.emit(_path_actions(path) + [_dir_act(d), _dir_act(back_d)], "misstep",
                    {"id": "pushpull", "kind": "wrong"},
                    steps=_steps_from_path(path) + [
                        {"disp": DIR_NAME[ACTION_OF_DELTA[d]], "frm": box_from,
                         "to": [box_from[0] + d[0], box_from[1] + d[1]],
                         "kind": "push"},
                        {"disp": DIR_NAME[ACTION_OF_DELTA[back_d]], "frm": None,
                         "to": None, "kind": "move", "note": "pull_test"}],
                    wrong_pred={"id": "box_follows"},
                    plan_note={"id": "shove_and_drag"})
            evidence.append(("box_no_follow", {"box": list(sim.boxes[bi])}))

    moves = _push_bfs(sim.walls, sim.rows, sim.cols, sim.pos,
                      tuple(sorted(sim.boxes)), frozenset(sim.pads))
    assert moves is not None, "push episode: post-misstep state unsolvable"
    acts = [_dir_act(m) for m in moves]
    chunked = _chunks(acts, rng, 3, 7)
    consumed = 0
    for i, ch in enumerate(chunked):
        last = i == len(chunked) - 1
        if did_misstep and i == 0:
            phase = "revise"
        elif i == 0 and li == 0 and not did_misstep:
            phase = "orient"
        else:
            phase = "execute"
        sc.emit(ch, phase, hyp_true,
                react={"ok": False, "id": "box_no_follow"} if (did_misstep and i == 0) else None,
                revision={"frm": "pushpull", "to": "push_plan"} if (did_misstep and i == 0) else None,
                considered=["pushpull"] if (did_misstep and i == 0) else [],
                plan_note={"id": "push_order"}, closes_level=last,
                plan_ahead=acts[consumed:])
        consumed += len(ch)
    assert sim.won, "push episode failed to win in sim"
    evidence.append(("level_done", {"level": li + 1}))
    return sc.turns


# ===========================================================================
# entry point
# ===========================================================================


def build_episode(spec: dict[str, Any], ep_seed: int = 0) -> dict[str, Any]:
    """-> {"clean": bool, "levels": [[turn, ...], ...]}. Deterministic in
    (game_id, ep_seed)."""
    rng = random.Random(f"ep-{spec['game_id']}-{ep_seed}")
    clean_game = rng.random() < CLEAN_FRACTION
    family = spec["family"]
    evidence: list[Any] = []
    levels = []
    revision_done = False
    for li in range(len(spec["levels"])):
        clean_level = clean_game or revision_done
        if family == "nav":
            turns, did = _ep_nav_level(spec, li, rng, clean_level, evidence)
            revision_done = revision_done or did
        elif family == "click":
            turns = _ep_click_level(spec, li, rng, clean_level, evidence)
            revision_done = revision_done or not clean_level
        elif family == "push":
            turns = _ep_push_level(spec, li, rng, clean_level, evidence)
            revision_done = True if not clean_level else revision_done
        elif family == "replay":
            turns = _ep_replay_level(spec, li, rng, clean_level, evidence)
            revision_done = revision_done or not clean_level
        elif family == "carry":
            turns = _ep_carry_level(spec, li, rng, clean_level, evidence)
            revision_done = revision_done or not clean_level
        elif family == "mirror":
            turns = _ep_mirror_level(spec, li, rng, clean_level, evidence)
            revision_done = revision_done or not clean_level
        elif family == "rules":
            turns = _ep_rules_level(spec, li, rng, clean_level, evidence)
            revision_done = revision_done or not clean_level
        else:
            raise ValueError(family)
        if clean_game:
            for t in turns:
                t["sem"]["clean"] = True
        n_actions = sum(len(t["actions"]) for t in turns)
        assert n_actions <= spec["budget"][li], (
            f"{spec['game_id']} L{li + 1}: episode ({n_actions}) exceeds budget "
            f"({spec['budget'][li]})")
        levels.append(turns)
    return {"clean": clean_game, "levels": levels}


def episode_actions(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat action trace of the whole episode (for engine replay checks)."""
    return [a for lvl in episode["levels"] for t in lvl for a in t["actions"]]
