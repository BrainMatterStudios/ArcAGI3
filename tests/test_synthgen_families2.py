"""TDD for the convex-hull scaling probe's NEW mechanic families (affordance-diverse).

AVOID  = move AWAY from a hazard (inverse of REACH).
TOGGLE = click every switch (broad click affordance, no movement goal).
CHASE  = catch a MOVING target (non-stationary goal).
KEYDOOR= move onto the key, THEN move to the door (ordered movement subgoal, no click).
"""
from arcagi3.synthgen.world import (
    World, build, rewarding_action, FAMILIES, UP, DOWN, LEFT, RIGHT, click,
)


# ---------------- AVOID ----------------
def test_avoid_label_increases_distance_from_hazard():
    from arcagi3.synthgen.world import _manhattan
    w = World("AVOID", bg=0, wall=1, avatar=(8, 8), avatar_color=2,
              hazard=(8, 5), hazard_color=3, flee_dist=12)
    a = rewarding_action(w)                       # a move that flees the hazard
    nxt = (8 + a[1], 8 + a[2])
    assert _manhattan(nxt, (8, 5)) > _manhattan((8, 8), (8, 5))


def test_avoid_win_when_far_enough():
    w = World("AVOID", bg=0, wall=1, avatar=(8, 10), avatar_color=2,
              hazard=(8, 5), hazard_color=3, flee_dist=6)
    r, done = w.step(RIGHT)                       # dist 5 -> 6 >= flee_dist
    assert r == 1.0 and done is True


# ---------------- TOGGLE ----------------
def test_toggle_click_removes_switch():
    w = World("TOGGLE", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              switches={(2, 2), (4, 4)}, switch_color=5)
    r, done = w.step(click(2, 2))
    assert (2, 2) not in w.switches and done is False


def test_toggle_win_when_all_clicked():
    w = World("TOGGLE", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              switches={(2, 2)}, switch_color=5)
    r, done = w.step(click(2, 2))
    assert r == 1.0 and done is True


def test_toggle_label_clicks_nearest_switch():
    w = World("TOGGLE", bg=0, wall=1, avatar=(2, 2), avatar_color=2,
              switches={(2, 3), (9, 9)}, switch_color=5)
    assert rewarding_action(w) == click(3, 2)     # (2,3) is nearest -> click(x=col3,y=row2)


# ---------------- CHASE ----------------
def test_chase_label_tracks_target():
    w = World("CHASE", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              target=(5, 9), target_color=3, chase_drift=(0, 1), chase_slow=99)
    assert rewarding_action(w) == RIGHT


def test_chase_target_moves_on_step():
    w = World("CHASE", bg=0, wall=1, avatar=(0, 0), avatar_color=2,
              target=(5, 5), target_color=3, chase_drift=(0, 1), chase_slow=1)
    w.step(DOWN)                                  # target drifts +1 col each step
    assert w.target == (5, 6)


def test_chase_win_on_overlap():
    w = World("CHASE", bg=0, wall=1, avatar=(5, 7), avatar_color=2,
              target=(5, 8), target_color=3, chase_drift=(0, 1), chase_slow=99)
    r, done = w.step(RIGHT)
    assert r == 1.0 and done is True


# ---------------- KEYDOOR ----------------
def test_keydoor_pickup_key_on_contact():
    w = World("KEYDOOR", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              key=(5, 6), key_color=4, door=(5, 9), door_color=3)
    w.step(RIGHT)
    assert w.has_key is True


def test_keydoor_door_without_key_does_nothing():
    w = World("KEYDOOR", bg=0, wall=1, avatar=(5, 8), avatar_color=2,
              key=(1, 1), key_color=4, door=(5, 9), door_color=3)
    r, done = w.step(RIGHT)
    assert r == 0.0 and done is False


def test_keydoor_win_key_then_door():
    w = World("KEYDOOR", bg=0, wall=1, avatar=(5, 8), avatar_color=2,
              key=(1, 1), key_color=4, door=(5, 9), door_color=3)
    w.has_key = True
    r, done = w.step(RIGHT)
    assert r == 1.0 and done is True


def test_keydoor_label_key_first_then_door():
    w = World("KEYDOOR", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              key=(5, 7), key_color=4, door=(5, 9), door_color=3)
    assert rewarding_action(w) == RIGHT           # toward key
    w.has_key = True
    assert rewarding_action(w) == RIGHT           # then toward door


# ---------------- registry + build solvability ----------------
def test_new_families_registered():
    for fam in ("AVOID", "TOGGLE", "CHASE", "KEYDOOR"):
        assert fam in FAMILIES


def test_new_families_build_and_solve():
    import numpy as np
    for fam in ("AVOID", "TOGGLE", "CHASE", "KEYDOOR"):
        solved = False
        for trial in range(6):
            w = build(fam, np.random.default_rng(trial))
            for _ in range(3000):
                a = rewarding_action(w)
                if a is None:
                    break
                _r, done = w.step(a)
                if done:
                    solved = True
                    break
            if solved:
                break
        assert solved, f"label policy failed to solve {fam}"
