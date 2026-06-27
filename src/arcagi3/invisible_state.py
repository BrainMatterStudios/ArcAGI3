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
