"""TDD for the BET 3 interactive synthetic engine (Stage 0).

Worlds are constructed with EXPLICIT entity positions so mechanics are tested deterministically;
the randomized build_<family> generators are tested separately via invariants.
"""
import numpy as np
import pytest

from arcagi3.synthgen.world import World, UP, DOWN, LEFT, RIGHT, NOOP, click

PAL = dict(bg=0, wall=1, avatar=2, target=3, item=4, switch=5, block=6, marker=7)


# ---------------- REACH ----------------
def test_reach_avatar_moves_in_open_space():
    w = World("REACH", bg=0, wall=1, avatar=(10, 10), avatar_color=2,
              target=(10, 20), target_color=3)
    w.step(RIGHT)
    assert w.avatar == (10, 11)
    w.step(DOWN)
    assert w.avatar == (11, 11)


def test_reach_blocked_by_wall():
    w = World("REACH", bg=0, wall=1, avatar=(10, 10), avatar_color=2,
              target=(10, 20), target_color=3, walls={(10, 11)})
    reward, done = w.step(RIGHT)
    assert w.avatar == (10, 10)  # wall blocks, no move
    assert reward == 0.0 and done is False


def test_reach_win_on_target_contact():
    w = World("REACH", bg=0, wall=1, avatar=(10, 19), avatar_color=2,
              target=(10, 20), target_color=3)
    reward, done = w.step(RIGHT)
    assert w.avatar == (10, 20)
    assert reward == 1.0 and done is True


def test_reach_clicks_are_inert():
    w = World("REACH", bg=0, wall=1, avatar=(10, 10), avatar_color=2,
              target=(10, 20), target_color=3)
    reward, done = w.step(click(20, 20))
    assert w.avatar == (10, 10) and reward == 0.0 and done is False


# ---------------- COLLECT ----------------
def test_collect_picks_up_item_on_contact():
    w = World("COLLECT", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              items={(5, 6), (8, 8)}, item_color=4)
    reward, done = w.step(RIGHT)        # step onto (5,6)
    assert (5, 6) not in w.items
    assert reward == 0.0 and done is False   # one item left -> not solved


def test_collect_win_when_all_items_gone():
    w = World("COLLECT", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              items={(5, 6)}, item_color=4)
    reward, done = w.step(RIGHT)
    assert w.items == set()
    assert reward == 1.0 and done is True


def test_collect_item_disappears_from_frame():
    w = World("COLLECT", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              items={(5, 6), (8, 8)}, item_color=4)
    w.step(RIGHT)
    assert w.frame()[5, 6] != 4   # collected cell no longer shows item colour


# ---------------- GATE (ordered subgoal: click switch -> reach target) ----------------
def test_gate_reaching_target_before_switch_does_nothing():
    w = World("GATE", bg=0, wall=1, avatar=(10, 19), avatar_color=2,
              target=(10, 20), target_color=3, switch=(2, 2), switch_color=5)
    reward, done = w.step(RIGHT)       # onto target, but gate still closed
    assert w.avatar == (10, 20)
    assert reward == 0.0 and done is False
    assert w.gate_open is False


def test_gate_click_on_switch_opens_gate():
    w = World("GATE", bg=0, wall=1, avatar=(10, 10), avatar_color=2,
              target=(10, 20), target_color=3, switch=(2, 2), switch_color=5)
    reward, done = w.step(click(2, 2))   # click(x=col,y=row) -> switch at (row=2,col=2)
    assert w.gate_open is True
    assert reward == 0.0 and done is False   # opening is not itself a win


def test_gate_win_requires_switch_then_target():
    w = World("GATE", bg=0, wall=1, avatar=(10, 19), avatar_color=2,
              target=(10, 20), target_color=3, switch=(2, 2), switch_color=5)
    w.step(click(2, 2))                 # open gate
    reward, done = w.step(RIGHT)        # now reach target
    assert reward == 1.0 and done is True


def test_gate_click_elsewhere_does_not_open():
    w = World("GATE", bg=0, wall=1, avatar=(10, 10), avatar_color=2,
              target=(10, 20), target_color=3, switch=(2, 2), switch_color=5)
    w.step(click(40, 40))
    assert w.gate_open is False


def test_gate_open_is_visible_switch_disappears():
    # gate state must be observable (else GATE tests perception W, not inference W1).
    w = World("GATE", bg=0, wall=1, avatar=(10, 10), avatar_color=2,
              target=(10, 20), target_color=3, switch=(2, 2), switch_color=5)
    assert w.frame()[2, 2] == 5
    w.step(click(2, 2))
    assert w.frame()[2, 2] != 5   # visible change confirms the click worked


# ---------------- PUSH (sokoban: push block onto marker) ----------------
def test_push_block_shifts_when_avatar_moves_into_it():
    w = World("PUSH", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              block=(5, 6), block_color=6, marker=(5, 10), marker_color=7)
    w.step(RIGHT)
    assert w.avatar == (5, 6) and w.block == (5, 7)


def test_push_blocked_when_wall_behind_block():
    w = World("PUSH", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              block=(5, 6), block_color=6, marker=(5, 10), marker_color=7,
              walls={(5, 7)})
    w.step(RIGHT)
    assert w.avatar == (5, 5) and w.block == (5, 6)   # neither moves


def test_push_win_when_block_on_marker():
    w = World("PUSH", bg=0, wall=1, avatar=(5, 8), avatar_color=2,
              block=(5, 9), block_color=6, marker=(5, 10), marker_color=7)
    reward, done = w.step(RIGHT)
    assert w.block == (5, 10) and reward == 1.0 and done is True


def test_push_walking_without_block_is_plain_move():
    w = World("PUSH", bg=0, wall=1, avatar=(5, 5), avatar_color=2,
              block=(20, 20), block_color=6, marker=(5, 10), marker_color=7)
    w.step(RIGHT)
    assert w.avatar == (5, 6) and w.block == (20, 20)
