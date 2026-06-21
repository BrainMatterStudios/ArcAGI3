"""Mechanic inference (Increment 1 of the puzzle-solving research program).

Given a captured trajectory of (grid, action, next_grid, reward) steps, infer the game's
mechanics from a FIXED core-knowledge ontology — no learning, no frame-change steering, a
PORTFOLIO of empirically-tested detectors (not one fragile motion model). This is the cheap
kill-gate: if we cannot reliably INFER mechanics, the planning program is moot.

Detectors:
  - agency / avatar: the color that translates with ACTION-DEPENDENT deltas (different simple
    actions -> different motion), as opposed to action-INDEPENDENT movers (counters/animations).
  - movement type: 'step' (consistent per-action stride) vs 'slide' (variable stride to a wall).
  - contact interactions: push (a non-avatar object translates in the same step the avatar
    moves), block (same action gives zero motion on some steps, motion on others -> walls),
    collect (a non-background object color disappears during the trajectory).
  - goal: the object-level changes coinciding with a reward (level-up) edge.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .movement import infer_all_translations
from . import perception as P


@dataclass
class MechanicReport:
    avatar_color: int | None = None
    avatar_action_deltas: dict = field(default_factory=dict)  # action_id -> dominant (dr,dc)
    movement_type: str = "none"                                # step | slide | none
    interactions: list = field(default_factory=list)           # subset of push/block/collect
    goal_changes: list = field(default_factory=list)           # human-readable level-up deltas
    n_steps: int = 0

    def summary(self) -> str:
        return (f"avatar={self.avatar_color} move={self.movement_type} "
                f"deltas={self.avatar_action_deltas} interactions={self.interactions} "
                f"goal={self.goal_changes} (n={self.n_steps})")


def _dominant(deltas):
    """Most common (dr,dc) in a list."""
    c = defaultdict(int)
    for d in deltas:
        c[d] += 1
    return max(c, key=c.get) if c else None


def infer_mechanics(traj, background: int) -> MechanicReport:
    """traj: list of (grid, action_token, next_grid, reward). action_token = ("S",id)|("C",x,y)."""
    rep = MechanicReport(n_steps=len(traj))
    # 1) collect per-(color) movement keyed by simple action
    move_by_color: dict[int, list] = defaultdict(list)        # color -> [(action_id,(dr,dc))]
    push_colors: set[int] = set()
    seen_colors: set[int] = set()
    last_colors: set[int] = set()
    for (g, act, ng, r) in traj:
        seen_colors |= {int(c) for c in np.unique(g) if int(c) != background}
        last_colors = {int(c) for c in np.unique(ng) if int(c) != background}
        if act[0] != "S":
            continue
        movers = infer_all_translations(g, ng, background)
        for col, d in movers.items():
            move_by_color[int(col)].append((act[1], d))

    # 2) avatar = an action-DEPENDENT mover (different actions -> different deltas); counters
    #    move the same delta regardless of action and are excluded.
    def action_dependent(entries):
        by_a = defaultdict(set)
        for a, d in entries:
            by_a[a].add(d)
        return len(by_a) >= 2 and len({d for _, d in entries}) >= 2

    cands = {c: e for c, e in move_by_color.items() if len(e) >= 3}
    ad = {c: e for c, e in cands.items() if action_dependent(e)}
    pool = ad or cands
    if pool:
        rep.avatar_color = max(pool, key=lambda c: len(pool[c]))
        entries = pool[rep.avatar_color]
        by_a: dict[int, list] = defaultdict(list)
        for a, d in entries:
            by_a[a].append(d)
        rep.avatar_action_deltas = {a: _dominant(ds) for a, ds in by_a.items()}
        # 3) movement type: variable stride for the SAME action -> slide
        variable = False
        for a, ds in by_a.items():
            mags = [abs(dr) + abs(dc) for (dr, dc) in ds]
            if mags and max(mags) - min(mags) >= 2:
                variable = True
        rep.movement_type = "slide" if variable else "step"

    # 4) interactions
    av = rep.avatar_color
    # block: same action -> zero motion some steps (avatar present but didn't move)
    if av is not None:
        zero, nonzero = defaultdict(int), defaultdict(int)
        for (g, act, ng, r) in traj:
            if act[0] != "S":
                continue
            movers = infer_all_translations(g, ng, background)
            (nonzero if av in movers else zero)[act[1]] += 1
        if any(zero[a] > 0 and nonzero[a] > 0 for a in set(zero) | set(nonzero)):
            rep.interactions.append("block")
        # push: a non-avatar, non-counter color translated in the same step the avatar moved
        for (g, act, ng, r) in traj:
            if act[0] != "S":
                continue
            movers = infer_all_translations(g, ng, background)
            if av in movers:
                others = [c for c in movers if c != av]
                if others:
                    push_colors.update(others)
        if push_colors:
            rep.interactions.append("push")
    # collect: an object color present early disappeared by the end (consumed)
    if seen_colors - last_colors:
        rep.interactions.append("collect")

    # 5) goal: object-level diff on reward edges
    for (g, act, ng, r) in traj:
        if r and r > 0:
            bk = set(P._object_tuples(P.connected_components(g, background=background)))
            ak = set(P._object_tuples(P.connected_components(ng, background=background)))
            gained = ak - bk
            lost = bk - ak
            rep.goal_changes.append({"action": act, "gained": len(gained), "lost": len(lost)})
    return rep
