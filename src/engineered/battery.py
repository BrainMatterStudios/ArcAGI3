"""Wiggle probe battery — first-class, deterministic, budget-capped, resumable.

Design doc §2 Layer 2, inheriting Stage 0's verified battery
(scratchpad/engineered_stage0/p3_wiggle_battery.py) including both Stage-0
target-selection traps:

  Trap 1 (hollow-component centroid): the click target is the component pixel
  NEAREST its centroid — a raw centroid of a hollow/non-convex component lands
  off-component (this exact failure nulled all sb26 clicks on the first
  Stage-0 run).

  Trap 2 (duplicate-color buttons): distinct colors first (largest instance
  each), then FURTHER instances of already-seen colors, largest first —
  sb26's live answer buttons are the second, smaller instances of the display
  colors; a dedup-by-color-only policy misses every one of them.

Phases: each available directional (ACTION1-4) twice, then clicks on ranked
component targets. Diffs are recorded RAW (no mask assumed — unlike Stage 0
this battery must work on hidden games); the HUD mask is learned in
`finalize` and SELF/REACTIVE/DEAD are then computed on the non-HUD board.

Budget: every env.step issued after the initial reset counts, including
mid-battery RESETs after GAME_OVER (a RESET costs 1 scored action —
memory `arcagi3-action-accounting-measured`). Default cap 16.

Resumable: `run()` returns (profile, state); state serializes to/from JSON and
a later `run(env, state=state)` continues on the same live env session.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

if __package__ in (None, ""):  # running as a bare script: put src/ on the path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engineered.envs import game_stem
from engineered.perception import (
    FRAME_SHAPE,
    HudLine,
    Perception,
    learn_hud_lines,
    lines_to_mask,
    settled_frame,
)

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # 4-connectivity


# ---------------------------------------------------------------------------
# Component analysis + click-target ranking (both Stage-0 traps live here)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    """One connected same-color component with an on-component click point."""

    click_y: int
    click_x: int
    color: int
    size: int


def components(frame: np.ndarray, bg: int) -> list[Component]:
    """Connected non-background components; click point = component pixel
    nearest the centroid (Trap 1 fix)."""
    out: list[Component] = []
    for col in np.unique(frame):
        if col == bg:
            continue
        lab, n = ndimage.label(frame == col, structure=_S4)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            cy, cx = ys.mean(), xs.mean()
            k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            out.append(Component(int(ys[k]), int(xs[k]), int(col), len(ys)))
    return out


def select_click_targets(
    comps: Sequence[Component], budget: int, dupe_reserve: int = 2
) -> list[Component]:
    """Rank click targets: pass 1 = one component per DISTINCT color, largest
    first; pass 2 = remaining duplicate-color instances, largest first
    (Trap 2 fix). When the budget cannot hold both passes, up to
    `dupe_reserve` duplicate instances are guaranteed slots — sb26's live
    answer buttons are duplicates and a distinct-colors-only cut misses them.
    Duplicates are ordered by spatial diversity (greedy farthest-point from
    everything already selected), not size: duplicate instances NEAR already
    probed components are usually more of the same display, while far-away
    duplicates (sb26's bottom answer strip vs its top display) are where the
    live controls hide. Deterministic: ties broken by (-size, y, x)."""
    ordered = sorted(comps, key=lambda c: (-c.size, c.click_y, c.click_x))
    seen: set[int] = set()
    first: list[Component] = []
    dupes: list[Component] = []
    for c in ordered:
        if c.color not in seen:
            seen.add(c.color)
            first.append(c)
        else:
            dupes.append(c)

    picked = list(first)
    dupes_ranked: list[Component] = []
    pool = list(dupes)
    while pool:
        def min_dist(c: Component) -> int:
            if not picked:
                return 0
            return min(
                (c.click_y - p.click_y) ** 2 + (c.click_x - p.click_x) ** 2
                for p in picked
            )
        best = max(pool, key=lambda c: (min_dist(c), c.size, -c.click_y, -c.click_x))
        pool.remove(best)
        picked.append(best)
        dupes_ranked.append(best)

    if len(first) + len(dupes_ranked) <= budget:
        return first + dupes_ranked
    reserve = min(dupe_reserve, len(dupes_ranked), budget)
    n_first = min(len(first), budget - reserve)
    return first[:n_first] + dupes_ranked[: budget - n_first]


# ---------------------------------------------------------------------------
# Battery state / config / profile
# ---------------------------------------------------------------------------

@dataclass
class BatteryConfig:
    """Deterministic battery parameters. `budget` is the hard action cap."""

    budget: int = 16
    directional_reps: int = 2
    click_budget_with_dirs: int = 8
    # No directionals -> the whole budget belongs to clicks (sb26/lp85 class).
    click_budget_no_dirs: int = 14
    # Aux probes (ACTION5/ACTION7) run only when the game offers NO
    # directionals: those games' housekeeping may tick exclusively on aux
    # actions (sb26's row-53 bar ticks ONLY on ACTION5 — measured 2026-08-14),
    # and dir games already supply action-diverse HUD evidence.
    aux_action_ids: tuple[int, ...] = (5, 7)
    aux_reps_base: int = 2
    aux_reps_bonus: int = 1  # extra rep iff the action produced any diff
    dupe_reserve: int = 2
    hud_min_rate: float = 0.25
    hud_min_count: int = 3
    hud_big_diff_frac: float = 0.30
    hud_max_lines: int = 8
    self_blob_min_px: int = 2


@dataclass
class StepRecord:
    """One executed probe action and its raw frame diff."""

    kind: str  # "dir" | "aux" | "click" | "reset"
    action_id: int
    x: int | None
    y: int | None
    color: int | None
    diff: list[list[int]]  # raw (unmasked) [y, x, old_value, new_value] rows

    def diff_array(self) -> np.ndarray:
        return np.asarray(self.diff, dtype=int).reshape(-1, 4)


@dataclass
class BatteryState:
    """Serializable mid-battery state; pass back to `run` to continue."""

    game_id: str
    stem: str
    phase: str = "dir"  # "dir" | "aux" | "click" | "done"
    avail: list[int] = field(default_factory=list)
    dir_queue: list[int] = field(default_factory=list)
    aux_queue: list[int] = field(default_factory=list)
    aux_diffed: dict[str, bool] = field(default_factory=dict)
    aux_bonused: dict[str, bool] = field(default_factory=dict)
    click_targets: list[list[int]] | None = None  # [y, x, color] rows
    click_index: int = 0
    actions_spent: int = 0
    steps: list[StepRecord] = field(default_factory=list)
    prev_frame: list[list[int]] | None = None
    ended_state: str | None = None  # "WIN" if the battery tripped a win

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "BatteryState":
        d = json.loads(s)
        d["steps"] = [StepRecord(**r) for r in d["steps"]]
        return cls(**d)


@dataclass
class PerceptionProfile:
    """Per-game output of the battery: learned HUD + SELF/REACTIVE/DEAD."""

    game_id: str
    stem: str
    shape: tuple[int, int]
    hud_lines: list[HudLine]
    self_mask: np.ndarray
    reactive_mask: np.ndarray
    dead_mask: np.ndarray
    actions_spent: int
    per_directional_diffs: dict[str, int]
    clicks: list[dict[str, Any]]
    archetype: str  # "avatar" | "click" | "mixed" | "unknown"

    @property
    def hud_mask(self) -> np.ndarray:
        return lines_to_mask(self.hud_lines, self.shape)

    def perception(self) -> Perception:
        return Perception(self.stem, self.hud_lines, self.shape)

    def summary(self) -> str:
        reactive_clicks = sum(1 for c in self.clicks if c["changed"])
        return (
            f"{self.stem}: {self.actions_spent} actions  hud={self.hud_lines}  "
            f"SELF={int(self.self_mask.sum())}px  "
            f"REACTIVE={int(self.reactive_mask.sum())}px  "
            f"DEAD={int(self.dead_mask.sum())}px  "
            f"clicks {reactive_clicks}/{len(self.clicks)} reactive  "
            f"archetype={self.archetype}"
        )

    def to_json(self) -> str:
        d = {
            "game_id": self.game_id,
            "stem": self.stem,
            "shape": list(self.shape),
            "hud_lines": [list(l) for l in self.hud_lines],
            "self_mask": np.argwhere(self.self_mask).tolist(),
            "reactive_mask": np.argwhere(self.reactive_mask).tolist(),
            "dead_mask": np.argwhere(self.dead_mask).tolist(),
            "actions_spent": self.actions_spent,
            "per_directional_diffs": self.per_directional_diffs,
            "clicks": self.clicks,
            "archetype": self.archetype,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "PerceptionProfile":
        d = json.loads(s)
        shape = tuple(d["shape"])

        def unpack(coords: list[list[int]]) -> np.ndarray:
            m = np.zeros(shape, dtype=bool)
            for y, x in coords:
                m[y, x] = True
            return m

        return cls(
            game_id=d["game_id"],
            stem=d["stem"],
            shape=shape,  # type: ignore[arg-type]
            hud_lines=[(k, i) for k, i in d["hud_lines"]],
            self_mask=unpack(d["self_mask"]),
            reactive_mask=unpack(d["reactive_mask"]),
            dead_mask=unpack(d["dead_mask"]),
            actions_spent=d["actions_spent"],
            per_directional_diffs=d["per_directional_diffs"],
            clicks=d["clicks"],
            archetype=d["archetype"],
        )


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

class ProbeBattery:
    """Runs the probe plan on a live env and emits a PerceptionProfile."""

    def __init__(self, config: BatteryConfig | None = None) -> None:
        self.config = config or BatteryConfig()

    # -- execution ---------------------------------------------------------

    def run(
        self,
        env: Any,
        state: BatteryState | None = None,
        max_actions: int | None = None,
    ) -> tuple[PerceptionProfile, BatteryState]:
        """Execute (or continue) the battery on `env`.

        `max_actions` caps THIS call (for interleaved use inside an agent
        loop); the config budget always caps the battery overall. The profile
        is finalized from whatever evidence exists so far — call again with
        the returned state to refine.
        """
        from arcengine import GameAction, GameState

        cfg = self.config
        if state is None:
            obs = env.reset()
            gid = obs.game_id or "unknown"
            avail = sorted(a for a in (obs.available_actions or []) if a)
            state = BatteryState(game_id=gid, stem=game_stem(gid), avail=avail)
            dir_ids = [aid for aid in (1, 2, 3, 4) if aid in avail]
            # No click phase ahead -> the directionals own the whole budget;
            # more reps = more HUD evidence (ls20's rows 61-62 tick too
            # rarely to clear min_count in 8 steps).
            reps = cfg.directional_reps
            if dir_ids and 6 not in avail:
                reps = max(reps, cfg.budget // len(dir_ids))
            state.dir_queue = [aid for aid in dir_ids for _ in range(reps)]
            if not state.dir_queue:
                state.aux_queue = [
                    aid for aid in cfg.aux_action_ids if aid in avail
                    for _ in range(cfg.aux_reps_base)
                ]
            state.prev_frame = settled_frame(obs).tolist()

        spent_this_call = 0

        def budget_left() -> bool:
            if state.actions_spent >= cfg.budget:
                return False
            if max_actions is not None and spent_this_call >= max_actions:
                return False
            return True

        def step(action: Any, data: dict[str, int] | None = None) -> Any:
            nonlocal spent_this_call
            o = env.step(action, data=data) if data else env.step(action)
            state.actions_spent += 1
            spent_this_call += 1
            return o

        def record(kind: str, action_id: int, o: Any,
                   x: int | None = None, y: int | None = None,
                   color: int | None = None) -> np.ndarray:
            cur = settled_frame(o)
            prev = np.asarray(state.prev_frame, dtype=cur.dtype)
            coords = np.argwhere(cur != prev)
            d = np.column_stack([
                coords,
                prev[coords[:, 0], coords[:, 1]],
                cur[coords[:, 0], coords[:, 1]],
            ]) if len(coords) else np.zeros((0, 4), dtype=int)
            state.steps.append(StepRecord(kind, action_id, x, y, color, d.tolist()))
            state.prev_frame = cur.tolist()
            return d

        def handle_terminal(o: Any) -> Any | None:
            """WIN ends the battery; GAME_OVER costs one RESET."""
            if o.state == GameState.WIN:
                state.ended_state = "WIN"
                state.phase = "done"
                return None
            if o.state == GameState.GAME_OVER and budget_left():
                o2 = step(GameAction.RESET)
                record("reset", 0, o2)
                return o2
            return o

        # phase 1: directionals
        while state.phase == "dir":
            if not state.dir_queue or not budget_left():
                if not state.dir_queue:
                    state.phase = "aux"
                break
            aid = state.dir_queue[0]
            o = step(GameAction.from_id(aid))
            state.dir_queue.pop(0)
            record("dir", aid, o)
            o = handle_terminal(o)
            if o is None:
                break

        # phase 1b: aux actions (no-directional games only; queue planned at
        # init). A rep that diffs earns one bonus rep — enough evidence for
        # the HUD learner's min_count without eating the click budget.
        while state.phase == "aux":
            if not state.aux_queue or not budget_left():
                if not state.aux_queue:
                    state.phase = "click"
                break
            aid = state.aux_queue[0]
            o = step(GameAction.from_id(aid))
            state.aux_queue.pop(0)
            d = record("aux", aid, o)
            key = str(aid)
            if len(d):
                state.aux_diffed[key] = True
            if aid not in state.aux_queue and state.aux_diffed.get(key) \
                    and not state.aux_bonused.get(key):
                state.aux_bonused[key] = True
                state.aux_queue.extend([aid] * cfg.aux_reps_bonus)
            o = handle_terminal(o)
            if o is None:
                break

        # phase 2: clicks (only if ACTION6 was offered at battery start; on
        # resume we trust the planned targets / earlier decision)
        if state.phase == "click" and state.click_targets is None:
            if 6 not in state.avail:
                state.phase = "done"
            else:
                frame = np.asarray(state.prev_frame, dtype=np.int16)
                state.click_targets = self._plan_clicks(frame, state)

        while state.phase == "click" and budget_left():
            assert state.click_targets is not None
            if state.click_index >= len(state.click_targets):
                state.phase = "done"
                break
            y, x, color = state.click_targets[state.click_index]
            o = step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            state.click_index += 1
            record("click", 6, o, x=int(x), y=int(y), color=int(color))
            o = handle_terminal(o)
            if o is None:
                break

        if state.phase == "click" and state.click_targets is not None \
                and state.click_index >= len(state.click_targets):
            state.phase = "done"

        return self.finalize(state), state

    def _plan_clicks(self, frame: np.ndarray, state: BatteryState) -> list[list[int]]:
        """Rank click targets once, at click-phase entry.

        A provisional HUD (from the directional-phase diffs, when any exist)
        keeps the battery from spending clicks on the ticking budget bar."""
        cfg = self.config
        had_dirs = any(s.kind == "dir" for s in state.steps)
        probe_steps = [s for s in state.steps if s.kind != "reset"]
        provisional = learn_hud_lines(
            [s.diff_array() for s in probe_steps],
            shape=frame.shape,  # type: ignore[arg-type]
            min_rate=cfg.hud_min_rate,
            min_count=cfg.hud_min_count,
            big_diff_frac=cfg.hud_big_diff_frac,
            max_lines=cfg.hud_max_lines,
            action_ids=[s.action_id for s in probe_steps],
        )
        board = ~lines_to_mask(provisional, frame.shape)  # type: ignore[arg-type]
        bg = int(np.bincount(frame.ravel()).argmax())
        comps = components(np.where(board, frame, bg), bg)
        budget = cfg.click_budget_with_dirs if had_dirs else cfg.click_budget_no_dirs
        budget = min(budget, cfg.budget - state.actions_spent)
        targets = select_click_targets(comps, max(budget, 0), cfg.dupe_reserve)
        return [[c.click_y, c.click_x, c.color] for c in targets]

    # -- analysis ----------------------------------------------------------

    def finalize(self, state: BatteryState) -> PerceptionProfile:
        """Learn the HUD from recorded diffs, then attribute SELF/REACTIVE/
        DEAD on the non-HUD board."""
        cfg = self.config
        shape = FRAME_SHAPE if state.prev_frame is None else (
            len(state.prev_frame), len(state.prev_frame[0]))
        probe_steps = [s for s in state.steps if s.kind != "reset"]
        hud_lines = learn_hud_lines(
            [s.diff_array() for s in probe_steps],
            shape=shape,
            min_rate=cfg.hud_min_rate,
            min_count=cfg.hud_min_count,
            big_diff_frac=cfg.hud_big_diff_frac,
            max_lines=cfg.hud_max_lines,
            action_ids=[s.action_id for s in probe_steps],
        )
        board = ~lines_to_mask(hud_lines, shape)

        self_mask = np.zeros(shape, dtype=bool)
        reactive_mask = np.zeros(shape, dtype=bool)
        touched = np.zeros(shape, dtype=bool)
        per_dir: dict[str, int] = {}
        clicks: list[dict[str, Any]] = []
        for s in probe_steps:
            d = s.diff_array()
            m = np.zeros(shape, dtype=bool)
            if len(d):
                m[d[:, 0], d[:, 1]] = True
            m &= board
            touched |= m
            if s.kind == "dir":
                key = f"A{s.action_id}"
                per_dir[key] = per_dir.get(key, 0) + int(m.any())
                self_mask |= m
            elif s.kind == "aux":  # counted, but attributed to neither SELF
                key = f"A{s.action_id}"  # nor REACTIVE (semantics unknown)
                per_dir[key] = per_dir.get(key, 0) + int(m.any())
            elif s.kind == "click":
                if m.any():
                    reactive_mask |= m
                clicks.append({
                    "y": s.y, "x": s.x, "color": s.color,
                    "changed": bool(m.any()), "cells": int(m.sum()),
                })
        dead_mask = board & ~touched

        has_self = int(self_mask.sum()) >= cfg.self_blob_min_px
        has_reactive = any(c["changed"] for c in clicks)
        if has_self and has_reactive:
            archetype = "mixed"
        elif has_self:
            archetype = "avatar"
        elif has_reactive:
            archetype = "click"
        else:
            archetype = "unknown"

        return PerceptionProfile(
            game_id=state.game_id,
            stem=state.stem,
            shape=shape,  # type: ignore[arg-type]
            hud_lines=hud_lines,
            self_mask=self_mask,
            reactive_mask=reactive_mask,
            dead_mask=dead_mask,
            actions_spent=state.actions_spent,
            per_directional_diffs=per_dir,
            clicks=clicks,
            archetype=archetype,
        )


if __name__ == "__main__":  # standalone: run the battery on a few games
    import sys

    from engineered.envs import open_arcade, resolve_game_ids

    stems = sys.argv[1:] or ["tu93", "sb26", "lp85", "lf52"]
    arcade = open_arcade()
    gid_of = resolve_game_ids(arcade)
    battery = ProbeBattery()
    for stem in stems:
        env = arcade.make(game_id=gid_of[stem], scorecard_id=f"eng1-bat-{stem}")
        profile, st = battery.run(env)
        print(profile.summary())
        assert st.actions_spent <= battery.config.budget
