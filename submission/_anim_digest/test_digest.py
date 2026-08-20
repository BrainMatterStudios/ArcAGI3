"""Synthetic unit tests for digest_animation (no engine needed).

Run: .venv/bin/python submission/_anim_digest/test_digest.py  (or pytest)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digest import digest_animation  # noqa: E402

BG = 4  # charcoal background


def _grid(painter=None, size=16):
    g = [[BG] * size for _ in range(size)]
    if painter:
        painter(g)
    return [tuple(row) for row in g]


def test_single_frame_is_empty():
    g = _grid()
    assert digest_animation([g]) == ""
    assert digest_animation([]) == ""


def test_identical_frames_no_events():
    g = _grid()
    out = digest_animation([g, g, g])
    assert "no non-fade changes" in out


def test_blink_detected():
    base = _grid()
    def lit(g):
        for c in range(4, 8):
            g[5][c] = 8  # red bar
    on = _grid(lit)
    out = digest_animation([base, on, base, on, base], before=base)
    assert "flash Rx2" in out, out
    assert "@r5-5c4-7" in out, out
    assert "no net change" in out, out


def test_moving_object_trajectory():
    frames = []
    for x in (2, 5, 8, 11):
        def put(g, x=x):
            for r in (7, 8):
                for c in (x, x + 1):
                    g[r][c] = 9
        frames.append(_grid(put))
    out = digest_animation(frames, before=_grid())
    import re
    obj = re.search(r"obj r[78]c(\d+)", out)
    arrows = re.findall(r"->r[78]c(\d+)", out)
    assert obj and int(obj.group(1)) <= 5, out          # starts on the left
    assert arrows and int(arrows[-1]) >= 9, out         # ends on the right


def test_reveal_paint_in_place():
    # a 3x3 region painted row-by-row into yellow (11), like sb26's check pointer
    frames = []
    state = [[BG] * 16 for _ in range(16)]
    frames.append([tuple(r) for r in state])
    for r in (5, 6, 7):
        for c in (5, 6, 7):
            state[r][c] = 11
        frames.append([tuple(row) for row in state])
    frames.append(frames[-1])
    out = digest_animation(frames, before=frames[0])
    assert "Y@r6c6" in out, out


def test_fade_suppressed():
    # pure grayscale ramp 4->3->2->1->0 over one region: no narrated events
    frames = []
    for v in (4, 3, 2, 1, 0):
        def put(g, v=v):
            for c in range(4, 8):
                g[3][c] = v
        frames.append(_grid(put))
    out = digest_animation(frames, before=_grid())
    assert "no non-fade changes" in out, out


def test_hud_mask_removes_ticks():
    g0 = _grid()
    def tick(g):
        g[15][0] = 3
    g1 = _grid(tick)
    out = digest_animation([g0, g1], before=g0, hud_mask_cells={(15, 0)})
    assert "no net change" in out, out
    assert "chg" not in out, out


def test_deterministic():
    frames = []
    for x in (2, 6, 10):
        def put(g, x=x):
            g[4][x] = 9
            g[4][x + 1] = 9
        frames.append(_grid(put))
    a = digest_animation(frames, before=_grid())
    b = digest_animation(frames, before=_grid())
    assert a == b


def test_budget_cap():
    # many separate change sites -> event list must stay bounded
    state = [[BG] * 64 for _ in range(64)]
    frames = [[tuple(r) for r in state]]
    for k in range(40):
        r, c = 2 + (k * 3) % 60, 2 + (k * 7) % 60
        for rr in (r, r + 1):
            for cc in (c, c + 1):
                state[rr][cc] = 6 + (k % 8)
        frames.append([tuple(row) for row in state])
    out = digest_animation(frames, before=frames[0])
    assert len(out) < 600, (len(out), out)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
