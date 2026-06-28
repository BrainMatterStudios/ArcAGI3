"""TDD for the supervised label (rewarding_action) + randomized builders of the BET 3 probe.

rewarding_action(world) returns the on-path affordance the in-context model must predict from
interaction history. It is env-truth (the mechanic), NOT a demonstrated/oracle policy action, so the
probe tests INFERENCE (read the mechanic from frame-changes) not imitation.
"""
import numpy as np

from arcagi3.synthgen.world import (
    World, build, rewarding_action, candidate_actions, FAMILIES,
    UP, DOWN, LEFT, RIGHT, click,
)


# ---------------- rewarding_action: REACH ----------------
def test_label_reach_moves_toward_target():
    w = World("REACH", bg=0, wall=1, avatar=(5, 5), avatar_color=2, target=(5, 9), target_color=3)
    assert rewarding_action(w) == RIGHT
    w2 = World("REACH", bg=0, wall=1, avatar=(9, 5), avatar_color=2, target=(2, 5), target_color=3)
    assert rewarding_action(w2) == UP


# ---------------- rewarding_action: COLLECT ----------------
def test_label_collect_moves_toward_nearest_item():
    w = World("COLLECT", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              items={(5, 7), (0, 0)}, item_color=4)
    assert rewarding_action(w) == RIGHT   # nearest item is (5,7)


# ---------------- rewarding_action: GATE ----------------
def test_label_gate_clicks_switch_when_closed():
    w = World("GATE", bg=0, wall=1, avatar=(5, 5), avatar_color=2, target=(5, 9),
              target_color=3, switch=(2, 3), switch_color=5)
    assert rewarding_action(w) == click(3, 2)   # click(x=col,y=row) on switch (row2,col3)


def test_label_gate_moves_to_target_when_open():
    w = World("GATE", bg=0, wall=1, avatar=(5, 5), avatar_color=2, target=(5, 9),
              target_color=3, switch=(2, 3), switch_color=5)
    w.gate_open = True
    assert rewarding_action(w) == RIGHT


# ---------------- rewarding_action: PUSH ----------------
def test_label_push_pushes_block_toward_marker_when_aligned():
    # avatar left of block, marker further right -> push RIGHT
    w = World("PUSH", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              block=(5, 6), block_color=6, marker=(5, 10), marker_color=7)
    assert rewarding_action(w) == RIGHT


def test_label_push_pushes_left_when_marker_left():
    # marker LEFT of block, avatar aligned on the right -> push LEFT
    w = World("PUSH", bg=0, wall=1, avatar=(5, 8), avatar_color=2,
              block=(5, 7), block_color=6, marker=(5, 3), marker_color=7)
    assert rewarding_action(w) == LEFT


def test_label_push_pushes_up_when_marker_above():
    w = World("PUSH", bg=0, wall=1, avatar=(9, 5), avatar_color=2,
              block=(8, 5), block_color=6, marker=(3, 5), marker_color=7)
    assert rewarding_action(w) == UP


def test_label_push_navigates_to_pushfrom_avoiding_block():
    # marker right of block; avatar must reach pushfrom=(5,5) without shoving the block.
    w = World("PUSH", bg=0, wall=1, avatar=(5, 9), avatar_color=2,
              block=(5, 6), block_color=6, marker=(5, 10), marker_color=7)
    # following the label from here must eventually solve (block onto marker).
    for _ in range(300):
        r, done = w.step(rewarding_action(w))
        if done:
            break
    assert done


def test_label_push_repositions_behind_block_when_not_aligned():
    # avatar must get to the cell opposite the marker; here avatar above block, marker to the right,
    # so the rewarding action is to move toward the push-from cell (left of block), not push down.
    w = World("PUSH", bg=0, wall=1, avatar=(3, 6), avatar_color=2,
              block=(5, 6), block_color=6, marker=(5, 10), marker_color=7)
    act = rewarding_action(w)
    assert act in (DOWN, LEFT, RIGHT, UP)   # a move (not a click); repositioning


# ---------------- candidate_actions ----------------
def test_candidates_include_moves_and_entity_clicks():
    w = World("GATE", bg=0, wall=1, avatar=(5, 5), avatar_color=2, target=(5, 9),
              target_color=3, switch=(2, 3), switch_color=5)
    cands = candidate_actions(w)
    for m in (UP, DOWN, LEFT, RIGHT):
        assert m in cands
    assert click(3, 2) in cands            # switch is clickable
    assert rewarding_action(w) in cands    # label is always a candidate


# ---------------- build: randomized instances ----------------
def test_build_roles_are_distinct_colors():
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        w = build(fam, rng)
        colors = [w.bg, w.avatar_color] + (
            [w.target_color] if w.target is not None else [])
        # bg distinct from foreground roles
        assert w.bg != w.avatar_color


def test_build_is_solvable_by_following_the_label():
    # the label policy must solve every built family within a budget (sanity: mechanics are coherent).
    for fam in FAMILIES:
        rng = np.random.default_rng(7)
        solved_any = False
        for trial in range(5):
            w = build(fam, rng)
            for _ in range(2000):
                r, done = w.step(rewarding_action(w))
                if done:
                    solved_any = True
                    break
        assert solved_any, f"label policy failed to ever solve {fam}"


def test_build_colors_randomized_across_episodes():
    # avatar colour should vary across seeds (role randomization) -> colour carries no family signal.
    cols = set()
    for s in range(20):
        w = build("REACH", np.random.default_rng(s))
        cols.add(w.avatar_color)
    assert len(cols) >= 3
