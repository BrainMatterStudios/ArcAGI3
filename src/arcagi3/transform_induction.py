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
