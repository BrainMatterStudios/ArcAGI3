"""Induce the transform-match grammar from interaction: which tile-color cycles which agent
attribute, and the level-completion predicate. General primitives only — no game constants.

A `triple` is a dict produced by the discovery loop on a step where the agent moved ONTO a
cell: {"entered_color": int, "attr": "color"|"shape", "before": <attr value>, "after": <attr value>}.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class OnEnterCycle:
    tile_color: int
    attribute: str          # "color" | "shape"
    order: tuple            # observed cycle order of attribute values


def induce_on_enter_cycles(triples: list[dict], min_support: int = 1) -> list[OnEnterCycle]:
    seq: dict[tuple, list] = defaultdict(list)
    changed: dict[tuple, int] = defaultdict(int)
    for t in triples:
        key = (t["entered_color"], t["attr"])
        if t["after"] != t["before"]:
            changed[key] += 1
            seq[key].extend([t["before"], t["after"]])
    rules = []
    for (tile_color, attr), n in changed.items():
        if n >= min_support:
            order = tuple(dict.fromkeys(seq[(tile_color, attr)]))  # de-dup, keep first-seen order
            rules.append(OnEnterCycle(tile_color=tile_color, attribute=attr, order=order))
    return rules


@dataclass(frozen=True)
class TerminalPredicate:
    kind: str = "attr_match_at_slot"

    def holds(self, agent_pos, agent_attrs, slots) -> bool:
        """Terminal iff every slot is satisfied: already done, or the agent stands on it with all
        required attributes matching. `agent_attrs` is a dict {name: value}; slots use "attr_req"
        (a dict), matching the live planner schema (factored_model._satisfied)."""
        return all(
            s["done"] or (agent_pos == s["pos"]
                          and all(agent_attrs.get(k) == v for k, v in s["attr_req"].items()))
            for s in slots
        )


def induce_terminal(prewin: dict) -> TerminalPredicate:
    """From the contrasted pre-win state, the completion predicate is: every slot is satisfied by
    the agent standing on it with matching attributes. (Spatial+attribute, not a global count —
    confirmed by levelup_contrastive.) The predicate FORM is fixed; its slot specs come live."""
    return TerminalPredicate(kind="attr_match_at_slot")


@dataclass(frozen=True)
class RecolorOnMove:
    from_color: int
    to_color: int


def induce_recolor_on_move(observations: list[dict], min_support: int = 1) -> list[RecolorOnMove]:
    """Induce 'moving recolors cells from_color->to_color' (paint). A world-delta observation is
    {"from_color", "to_color", "vanished": bool}; vanished deltas are collect, not recolor (skip)."""
    counts: dict[tuple, int] = defaultdict(int)
    for o in observations:
        if not o.get("vanished"):
            counts[(o["from_color"], o["to_color"])] += 1
    return [RecolorOnMove(from_color=f, to_color=t)
            for (f, t), n in counts.items() if n >= min_support]
