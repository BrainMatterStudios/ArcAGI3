"""witness_probe.py — constructive solvability witnesses for click games.

The BFS verifier abstains on click games (4096-wide action space). For games
whose solution is computable from their own open source or from the rendered
frame, playing that solution through the REAL engine and reaching WIN is a
strictly stronger proof than BFS: an explicit witness trajectory. Games proved
here satisfy holdout rubric R3 exactly like verifier-'proved' stems.

Solvers (each returns clicks for the CURRENT level, called after each level
transition):
    ff01  Flood fill — click the interior center of every closed enclosure
          (enclosure geometry from the game module's own get_level_shapes).
    sy01  Mirror Maker — read the left-half pattern off the live frame and
          click each colored cell's mirrored right-half position.

Run:  .venv/bin/python scratchpad/rl_gate/witness_probe.py [--games ff01,sy01]
Exit 0 iff every probed game reaches WIN with all levels completed.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_DIR = REPO / "scratchpad/arc_interactive_upstream/environment_files"


def load_game_module(stem: str):
    src = sorted(ENV_DIR.glob(f"{stem}/*/{stem}.py"))[-1]
    spec = importlib.util.spec_from_file_location(f"witness_{stem}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame_grid(frame):
    import numpy as np
    g = np.asarray(frame.frame)
    if g.ndim == 3:  # (n_frames, 64, 64) — take the final rendered frame
        g = g[-1]
    return g.tolist()  # 64x64 rows of plain color ints


class CamMap:
    """Numeric inverse of camera.display_to_grid — clicks are sent in DISPLAY
    coords but game geometry lives in GRID coords (the level-0 GAME_OVERs were
    exactly this mismatch: grid coords clicked as display coords miss)."""

    def __init__(self, env):
        cam = env._game.camera
        self.disp_of = {}   # (gx, gy) -> (dx, dy) first display pixel seen
        self.grid_of = {}   # (dx, dy) -> (gx, gy)
        for dy in range(64):
            for dx in range(64):
                g = cam.display_to_grid(dx, dy)
                if g is None:
                    continue
                self.grid_of[(dx, dy)] = tuple(g)
                self.disp_of.setdefault(tuple(g), (dx, dy))

    def click_for(self, gx, gy):
        return self.disp_of.get((int(gx), int(gy)))


def solve_ff01(mod, env, frame, level_idx):
    """Click one guaranteed-interior point of each UNFILLED enclosure, read
    from the LIVE game instance. Two earlier failure modes: re-importing the
    module rebuilds different geometry than the live level, and .center is the
    interior CENTROID — which falls in the hole of donut/ring shapes."""
    game = env._game
    shapes = game._shapes
    filled = game._filled_shapes
    cmap = CamMap(env)
    clicks = []
    for i, shape in enumerate(shapes):
        if i in filled or not shape.interior:
            continue
        gx, gy = shape.interior[len(shape.interior) // 2]
        if not shape.contains(gx, gy):
            gx, gy = shape.interior[0]
        d = cmap.click_for(gx, gy)
        if d is None:
            raise RuntimeError(f"ff01: interior point ({gx},{gy}) not on camera")
        clicks.append(d)
    return clicks


def solve_sy01(mod, env, frame, level_idx):
    """Left panel = reference; right panel = editable mirror around the grid's
    vertical center column. Read cell colors through the camera map, mirror in
    GRID space, click display pixels of wrong right-half cells."""
    from collections import Counter
    g = frame_grid(frame)
    cmap = CamMap(env)
    cell_color = {}
    for (dx, dy), (gx, gy) in cmap.grid_of.items():
        cell_color.setdefault((gx, gy), Counter())[g[dy][dx]] += 1
    color = {c: cnt.most_common(1)[0][0] for c, cnt in cell_color.items()}
    xs = sorted({c[0] for c in color})
    background = Counter(color.values()).most_common(1)[0][0]
    mid = xs[len(xs) // 2]  # center divider column
    clicks = []
    for (gx, gy), v in sorted(color.items()):
        if gx >= mid or v == background:
            continue
        mgx = 2 * mid - gx  # mirror across the divider
        if color.get((mgx, gy)) != v:
            d = cmap.click_for(mgx, gy)
            if d is not None:
                clicks.append(d)
    return clicks


def solve_sq01(mod, env, frame, level_idx):
    """Click the NEXT expected color's 2x2 block (live instance holds the
    sequence and the color->sprite map; sprites vanish when clicked, so emit
    exactly one click per derivation)."""
    game = env._game
    if game._progress >= len(game._sequence):
        return []  # level advance pending; pacing handles the end_frames
    expected = game._sequence[game._progress]
    sprite = game._color_to_sprite.get(expected)
    if sprite is None:
        return []
    cmap = CamMap(env)
    d = cmap.click_for(sprite.x, sprite.y)
    return [d] if d is not None else []


SOLVERS = {"ff01": solve_ff01, "sy01": solve_sy01, "sq01": solve_sq01}


def probe(stem: str) -> dict:
    import arc_agi
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    arcade = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                            environments_dir=str(ENV_DIR))
    arcade.get_environments()
    env = arcade.make(stem)
    frame = env.reset()
    mod = load_game_module(stem)
    solver = SOLVERS[stem]

    total_clicks = 0
    for safety in range(2000):
        level = int(frame.levels_completed)
        if frame.state.name == "WIN":
            break
        if frame.state.name == "GAME_OVER":
            return {"stem": stem, "verdict": "FAILED", "reason": "GAME_OVER",
                    "levels": level, "clicks": total_clicks}
        clicks = solver(mod, env, frame, level)
        if not clicks:
            return {"stem": stem, "verdict": "FAILED",
                    "reason": f"solver produced no clicks at level {level}",
                    "levels": level, "clicks": total_clicks}
        progressed = False
        for (x, y) in clicks:
            frame = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)},
                             reasoning={"thought": "witness", "step": total_clicks})
            total_clicks += 1
            if frame.state.name in ("WIN", "GAME_OVER"):
                progressed = True
                break
            if int(frame.levels_completed) > level:
                progressed = True
                break  # recompute clicks against the new level's frame
        if not progressed and int(frame.levels_completed) == level \
                and frame.state.name == "NOT_FINISHED":
            # full click batch spent without a level transition. Games resolve
            # fills/advances via animation holds over subsequent steps — but
            # blind no-ops can burn step budgets (sq01 counts them!). Pace
            # ONLY while the live instance is inside a hold.
            def _in_hold():
                game = env._game
                return any(int(getattr(game, attr, 0) or 0) > 0 for attr in
                           ("_win_hold", "_end_frames", "_ripple_tail"))
            for _ in range(40):
                if not _in_hold():
                    break
                frame = env.step(GameAction.ACTION1,
                                 reasoning={"thought": "pacing", "step": total_clicks})
                if frame.state.name != "NOT_FINISHED" \
                        or int(frame.levels_completed) > level:
                    break
            if int(frame.levels_completed) > level or frame.state.name == "WIN":
                continue
            clicks2 = solver(mod, env, frame, level)
            if not clicks2:
                return {"stem": stem, "verdict": "FAILED",
                        "reason": f"stuck at level {level} after full batch",
                        "levels": level, "clicks": total_clicks}
    return {"stem": stem,
            "verdict": "WITNESS-PROVED" if frame.state.name == "WIN" else "FAILED",
            "state": frame.state.name, "levels": int(frame.levels_completed),
            "win_levels": int(frame.win_levels), "clicks": total_clicks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="ff01,sy01")
    args = ap.parse_args()
    results = [probe(s.strip()) for s in args.games.split(",") if s.strip()]
    ok = True
    for r in results:
        print(r)
        ok = ok and r["verdict"] == "WITNESS-PROVED" \
            and r["levels"] == r.get("win_levels", -1)
    print("ALL WITNESS-PROVED" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
