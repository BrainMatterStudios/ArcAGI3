"""Regression tests for geometry["games"] — the FOCUS SUBSET filter.

2026-08-09: the first version of this filter compared bare stems ("ft09") against
official ids, which are "<stem>-<hash>" ("ft09-7fbdac44"). It matched nothing, and
BOTH ft09-ablation arms died at the guard after ~8.6 min of GPU each. The local
test that "passed" beforehand used invented bare-stem ids, so it could not fail.

These tests use REAL-SHAPED ids for exactly that reason.
"""
from __future__ import annotations

import pytest

# Real-shaped: every official id carries a hash suffix.
OFFICIAL = sorted([
    "ar25-1a2b3c4d", "bp35-2b3c4d5e", "cd82-3c4d5e6f", "cn04-4d5e6f70", "dc22-5e6f7081",
    "ft09-7fbdac44", "g50t-6f708192", "hd91-708192a3", "ig73-8192a3b4", "lf52-92a3b4c5",
    "ls20-a3b4c5d6", "lp85-b4c5d6e7", "mp23-c5d6e7f8", "ne27-d6e7f809", "r11l-e7f8091a",
    "re86-f8091a2b", "sb26-091a2b3c", "sc25-1a2b3c4e", "sq01-2b3c4d5f", "sk01-3c4d5e70",
    "tn36-4d5e6f81", "tu93-5e6f7092", "ul01-6f7081a3", "vc33-708192b4", "wm88-8192a3c5",
])


def apply_focus(official: list[str], focus: tuple[str, ...]) -> list[str]:
    """Mirror of the filter in pc_driver.pc_main. Keep the two in sync."""
    if not focus:
        return official
    stems = {g.split("-")[0]: g for g in official}
    missing = [g for g in focus if g not in stems]
    if missing:
        raise RuntimeError(
            f"geometry.games not in the official set: {missing}; "
            f"available stems: {sorted(stems)}")
    return [stems[g] for g in focus]


def clone_sources(official: list[str], clones: int = 28) -> list[str]:
    """The clone->source reconstruction pc_driver asserts against."""
    return [official[i % len(official)] for i in range(clones)]


def test_single_game_focus_covers_every_clone():
    """The whole point: n=1 per game becomes n=28 on one game."""
    focused = apply_focus(list(OFFICIAL), ("ft09",))
    assert focused == ["ft09-7fbdac44"]
    sources = clone_sources(focused)
    assert len(sources) == 28
    assert set(s.split("-")[0] for s in sources) == {"ft09"}


def test_focus_matches_on_stem_not_full_id():
    """The exact bug: a bare stem must match a hashed id."""
    assert apply_focus(list(OFFICIAL), ("ft09",)) == ["ft09-7fbdac44"]
    # and the naive membership test that caused the outage finds nothing
    assert [g for g in OFFICIAL if g in ("ft09",)] == []


def test_two_game_focus_splits_clones_evenly():
    focused = apply_focus(list(OFFICIAL), ("ft09", "r11l"))
    counts: dict[str, int] = {}
    for s in clone_sources(focused):
        counts[s.split("-")[0]] = counts.get(s.split("-")[0], 0) + 1
    assert counts == {"ft09": 14, "r11l": 14}


def test_empty_focus_is_the_unchanged_full_wave():
    assert apply_focus(list(OFFICIAL), ()) == OFFICIAL
    counts: dict[str, int] = {}
    for s in clone_sources(OFFICIAL):
        counts[s.split("-")[0]] = counts.get(s.split("-")[0], 0) + 1
    assert len(counts) == 25
    assert sum(counts.values()) == 28


def test_unknown_game_raises_and_names_the_alternatives():
    with pytest.raises(RuntimeError) as exc:
        apply_focus(list(OFFICIAL), ("nope",))
    assert "nope" in str(exc.value)
    assert "available stems" in str(exc.value)


def test_focus_order_is_honoured():
    """Clone assignment is round-robin over the returned order."""
    assert apply_focus(list(OFFICIAL), ("r11l", "ft09")) == ["r11l-e7f8091a", "ft09-7fbdac44"]
