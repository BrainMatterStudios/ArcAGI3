"""Unit tests for arc3kit.py spatial and graph primitives."""

import numpy as np
import pytest
from arcagi3.arc3kit import (
    connected_components,
    detect_avatar_motion,
    detect_background,
    find_path_bfs,
    frame_diff_summary,
)


def test_detect_background():
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[10:20, 10:20] = 3
    assert detect_background(grid) == 0


def test_connected_components_multi_same_color():
    grid = np.zeros((64, 64), dtype=np.int8)
    grid[5:10, 5:10] = 2  # 5x5 block 1 of color 2
    grid[20:25, 20:25] = 2  # 5x5 block 2 of color 2

    objs = connected_components(grid)
    assert len(objs) == 2
    for o in objs:
        assert o.color == 2
        assert o.size == 25


def test_find_path_bfs_obstacle_colors():
    grid = np.zeros((64, 64), dtype=np.int8)
    # Create wall of color 1 at row 10 (cols 0 to 60)
    grid[10, 0:60] = 1

    path = find_path_bfs(grid, start=(5, 5), target=(15, 5), obstacle_colors={1})
    assert path is not None
    assert path[0] == (5, 5)
    assert path[-1] == (15, 5)
    # Path must route around col 60 wall opening
    assert any(c >= 60 for r, c in path)


def test_detect_avatar_motion_same_color_multi():
    g1 = np.zeros((64, 64), dtype=np.int8)
    g2 = np.zeros((64, 64), dtype=np.int8)

    # Static block of color 3
    g1[5:7, 5:7] = 3
    g2[5:7, 5:7] = 3

    # Moving avatar block of color 3
    g1[10:12, 10:12] = 3
    g2[10:12, 11:13] = 3  # moved right 1 col

    motion = detect_avatar_motion(g1, g2)
    assert motion is not None
    assert motion["moved"] == True
    assert motion["color"] == 3
    assert motion["delta"] == (0, 1)


def test_frame_diff_summary():
    g1 = np.zeros((64, 64), dtype=np.int8)
    g2 = np.zeros((64, 64), dtype=np.int8)

    g2[5, 5] = 7
    diff = frame_diff_summary(g1, g2)

    assert diff["changed"] == True
    assert diff["num_changed_cells"] == 1
    assert diff["changed_colors"] == [7]
