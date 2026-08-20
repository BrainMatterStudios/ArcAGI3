"""FALSIFIABLE TEST: reproduce sb26's check animation offline and assert the
digest carries the slot-visit ORDER information that the v12smoke model had to
derive manually (801s, 10 LLM calls, 8 tracebacks through a 1024-cell crop API).

Setup (offline arc_agi engine, environment_files/sb26):
  - replay the first 23 actions of docs/test-artifacts-2026-08-02/sb26_win_trace.json.
    That fills the two-tray L2 board the "natural but wrong" way (red tray gets
    strip colors 1-3 skipping the pipe-stub slot, green tray gets 4-7).
  - press SPACE (ACTION5): the engine returns the 111-frame FAILED check
    animation -- the same one v12smoke decoded manually (its final frame equals
    the pre-SPACE board; every verdict lives in transient frames).
  - also replay 41 actions (the corrected arrangement) + SPACE: the 227-frame
    PASSING check that walks all seven positions.

The v12smoke manual derivation this must match (transcript sb26-7fbdac44_p0,
analysis steps 5-7 and onward):
  - top strip shows the target sequence O,p,R,b,N,Y,M; during the check each
    strip frame is filled in, in CHECK ORDER;
  - the slot highlight visits red1, red2, then follows the pipe DOWN into the
    green tray (green1..green4), returning to red4 last
    -- i.e. check order red1,red2,red3(stub),green1..green4,red4;
  - a mismatch stops the scan with a red flash at the failing position.

PASS criteria (asserted below):
  FAILED check digest must show: reveal O then reveal p on the strip (ascending
  cols), a mover reaching the green tray (row>=30) BEFORE the mismatch flash,
  the flash colored R (the expected color at the failing 3rd position), and no
  net board change (type-1: final frame hides everything).
  PASSING check digest must show all seven reveals in strip order O,p,R,b,N,Y,M,
  a green-tray mover between reveal 2 and reveal 3, and a return to the red
  tray (row<=27) between reveal 6 and reveal 7.
"""
import json
import os
import re
import sys
import time

REPO = "/Users/ahmed/Documents/ArcAGI3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

from digest import digest_animation  # noqa: E402

REVEAL = re.compile(r"^([A-Za-z])@r(\d+)c(\d+)$")
MOTION = re.compile(r"^(?:obj |->)r(\d+)c(\d+)$")
FLASH = re.compile(r"^flash ([A-Za-z])x(\d+) @r(\d+)-(\d+)c(\d+)-(\d+)$")

HUD_MASK = {(53, c) for c in range(64)}  # sb26 ticks an action counter on row 53


def _events(digest_text):
    body = digest_text.split(": ", 1)[1]
    return [e.strip() for e in body.split(";") if e.strip()]


def _replay_and_space(n_actions):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    trace = json.load(open("docs/test-artifacts-2026-08-02/sb26_win_trace.json"))
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith("sb26"))
    env = client.make(game_id=gid, scorecard_id=f"anim-digest-test-{n_actions}")
    obs = env.reset()
    for t in trace[:n_actions]:
        if t["name"] == "ACTION6":
            obs = env.step(GameAction.ACTION6, data={"x": t["x"], "y": t["y"]})
        else:
            obs = env.step(getattr(GameAction, t["name"]))
    before = obs.frame[-1]
    levels_before = int(obs.levels_completed or 0)
    obs = env.step(GameAction.ACTION5)  # SPACE triggers the check scan
    return before, obs, levels_before


def test_failed_check_digest_carries_visit_order():
    before, obs, levels = _replay_and_space(23)
    assert levels == 1, f"expected mid-L2 state, got levels_completed={levels}"
    frames = obs.frame
    assert len(frames) > 50, f"expected a long check animation, got {len(frames)} frames"

    t0 = time.time()
    digest = digest_animation(frames, hud_mask_cells=HUD_MASK, before=before)
    dt = time.time() - t0

    print("\n--- FAILED L2 check (SPACE after the natural-but-wrong fill) ---")
    print(f"engine frames: {len(frames)}   digest computed in {dt*1000:.0f} ms")
    print("DIGEST:", digest)
    print(f"(~{len(digest)//4} tokens)")
    print(
        "v12smoke manual derivation (801s, 10 LLM calls): strip=O,p,R,b,N,Y,M; "
        "check filled O then p, then FAILED at position 3 with a red flash; the "
        "slot highlight had moved down into the green tray -- check order "
        "red1,red2,red3(stub),green1..green4,red4."
    )

    events = _events(digest)
    reveals = [(m.group(1), int(m.group(2)), int(m.group(3)), k)
               for k, e in enumerate(events) if (m := REVEAL.match(e))]
    motions = [(int(m.group(1)), int(m.group(2)), k)
               for k, e in enumerate(events) if (m := MOTION.match(e))]
    flashes = [(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                int(m.group(5)), int(m.group(6)), k)
               for k, e in enumerate(events) if (m := FLASH.match(e))]

    # 1. The check matched exactly two positions, revealing O then p on the strip.
    strip_reveals = [r for r in reveals if r[1] <= 7]
    assert [r[0] for r in strip_reveals] == ["O", "p"], f"strip reveals: {strip_reveals}"
    assert strip_reveals[0][2] < strip_reveals[1][2], "reveal cols must ascend (left-to-right check)"

    # 2. A mover reached the green tray (row>=30) BEFORE the mismatch flash:
    #    the third checked slot is NOT red3 -- the scan dove through the pipe.
    assert flashes, f"no mismatch flash found: {events}"
    flash = flashes[0]
    dive = [m for m in motions if m[0] >= 30 and m[2] < flash[6]]
    assert dive, f"no green-tray motion before the flash: {events}"

    # 3. The flash is R -- the expected color at the failing 3rd position -- and
    #    its extent touches both the strip and the green tray rows.
    assert flash[0] == "R", f"flash color: {flash}"
    assert flash[2] <= 7 and flash[3] >= 30, f"flash bbox misses strip or tray: {flash}"
    assert flash[4] <= 26 and flash[5] >= 23, f"flash bbox misses strip frame 3 (c23-26): {flash}"

    # 4. Type-1: the animation reverts; nothing of this is visible afterwards.
    assert "no net change" in digest, digest

    # 5. Budget: compact enough to inline into every action result.
    assert len(digest) <= 400, f"{len(digest)} chars is too long"
    print("PASS: digest reveals matched-prefix O,p + pipe-dive + R mismatch, in order.\n")


def test_passing_check_digest_shows_full_visit_order():
    before, obs, levels = _replay_and_space(41)
    assert levels == 1, f"expected pre-check L2 state, got {levels}"
    frames = obs.frame

    digest = digest_animation(frames, hud_mask_cells=HUD_MASK, before=before)
    print("--- PASSING L2 check (SPACE after the corrected fill) ---")
    print(f"engine frames: {len(frames)}")
    print("DIGEST:", digest)
    print(f"(~{len(digest)//4} tokens)")

    events = _events(digest)
    reveals = [(m.group(1), int(m.group(2)), int(m.group(3)), k)
               for k, e in enumerate(events) if (m := REVEAL.match(e))]
    motions = [(int(m.group(1)), int(m.group(2)), k)
               for k, e in enumerate(events) if (m := MOTION.match(e))]

    strip = [r for r in reveals if r[1] <= 7]
    # 1. All seven positions checked, in strip order, left to right.
    assert [r[0] for r in strip] == ["O", "p", "R", "b", "N", "Y", "M"], f"reveals: {strip}"
    cols = [r[2] for r in strip]
    assert cols == sorted(cols), f"reveal cols not ascending: {cols}"

    # 2. Slot highlight in the green tray between reveal 2 (p) and reveal 3 (R):
    #    the visit order detours through the pipe after red2.
    k_p, k_R = strip[1][3], strip[2][3]
    assert any(m[0] >= 30 and k_p < m[2] < k_R for m in motions), \
        f"no green-tray motion between p and R: {events}"

    # 3. Return to the red tray (row<=27) between reveal 6 (Y) and reveal 7 (M):
    #    the last checked slot is red4.
    k_Y, k_M = strip[5][3], strip[6][3]
    assert any(m[0] <= 27 and k_Y < m[2] < k_M for m in motions), \
        f"no red-tray return between Y and M: {events}"

    print(
        "PASS: full visit order O,p | green-tray dive | R,b,N,Y | red-tray return | M\n"
        "      == check order red1,red2,(pipe),green1..green4,red4.\n"
    )


if __name__ == "__main__":
    fails = 0
    for fn in (test_failed_check_digest_carries_visit_order,
               test_passing_check_digest_shows_full_visit_order):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if fails else 0)
