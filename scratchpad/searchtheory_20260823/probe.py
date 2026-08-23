"""Search-theory probe (2026-08-23, scratch only).

Snapshot-based search over ALL 25 dev games on the offline engine.
Two algorithms per game, chained level-by-level:
  bfs : plain BFS, HUD-masked frame dedup (baseline, but with O(1) deepcopy
        expansion instead of reset-replay)
  iw1 : IW(1) novelty pruning — a state is kept only if it makes some
        (y, x, color) atom true for the first time on this level's search.

Records per level: solved?, depth, states, nodes, wall, exhausted?.
Writes JSON per game to results/.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from collections import deque

import numpy as np

logging.disable(logging.CRITICAL)
ROOT = "/Users/ahmed/Documents/ArcAGI3"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scratchpad/ideas"))

from arc_agi import Arcade, OperationMode          # noqa: E402
from arcengine import GameAction, GameState        # noqa: E402
from hud_mask import frame_key, mask_frame         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

GAMES = {}
for stem in os.listdir(os.path.join(ROOT, "environment_files")):
    sub = os.listdir(os.path.join(ROOT, "environment_files", stem))
    ver = [v for v in sub if not v.startswith("_") and not v.startswith(".")]
    GAMES[stem] = f"{stem}-{ver[0]}"


def settled(obs):
    a = np.asarray(obs.frame)
    return a[-1] if a.ndim == 3 else a


def lv(obs):
    return int(obs.levels_completed or 0)


def components(grid, bg):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or grid[sr, sc] == bg:
                continue
            color = grid[sr, sc]
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                            and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            comps.append(dict(color=int(color), size=len(cells),
                              bbox=(min(rs), min(cs), max(rs), max(cs)),
                              cy=sum(rs) // len(cells), cx=sum(cs) // len(cells)))
    return comps


def click_targets(grid, stem, max_n=int(os.environ.get("PROBE_CLICKS", "16"))):
    """Frame-only dynamic click targets: compact non-bg component centers."""
    g = mask_frame(grid, stem)
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    comps = [c for c in components(g, bg) if 2 <= c["size"] <= 100]
    comps.sort(key=lambda c: c["size"])
    out = []
    for c in comps:
        t = (c["cx"], c["cy"])
        if all(abs(t[0] - u[0]) + abs(t[1] - u[1]) > 2 for u in out):
            out.append(t)
        if len(out) >= max_n:
            break
    return out


def actions_for(obs, stem):
    avail = set(int(a) for a in (obs.available_actions or []))
    if not avail:
        avail = {1, 2, 3, 4, 5, 6}
    acts = [("S", a) for a in sorted(avail) if a not in (0, 6)]
    if 6 in avail:
        acts += [("C", x, y) for x, y in click_targets(settled(obs), stem)]
    return acts


def apply_tok(env, tok):
    if tok[0] == "C":
        return env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
    return env.step(GameAction.from_id(int(tok[1])))


def search_level2(root_env, stem, target, algo, budget_s, max_states=20000):
    t0 = time.time()
    obs = root_env.reset()
    if lv(obs) >= target:
        return root_env, obs, 0, dict(depth=0, states=1, nodes=0, wall=0.0,
                                      exhausted=False)
    seen = {frame_key(settled(obs), stem)}
    atoms = None
    if algo in ("iw1", "nbfs", "mnbfs"):
        atoms = np.zeros((64, 64, 16), dtype=bool)
        g = mask_frame(settled(obs), stem).clip(0, 15)
        atoms[np.arange(64)[:, None], np.arange(64)[None, :], g] = True
    queue = deque([(root_env, obs, 0)])
    slow = deque()  # nbfs: non-novel states, expanded only when queue empties
    nodes = 0
    exhausted = False
    max_b = 0
    max_depth_seen = 0
    while queue or slow:
        if time.time() - t0 > budget_s or len(seen) > max_states:
            break
        env, obs, depth = queue.popleft() if queue else slow.popleft()
        acts = actions_for(obs, stem)
        if algo == "mnbfs":
            # macro children: repeat each simple action while the masked frame
            # keeps changing (run-length macro, max 12 reps)
            acts = acts + [("M", tok[1]) for tok in acts if tok[0] == "S"]
        max_b = max(max_b, len(acts))
        for i, tok in enumerate(acts):
            if time.time() - t0 > budget_s:
                break
            child = copy.deepcopy(env) if i < len(acts) - 1 else env
            if tok[0] == "M":
                cobs = apply_tok(child, ("S", tok[1]))
                nodes += 1
                reps = 1
                while (cobs is not None and reps < 12
                       and cobs.state != GameState.GAME_OVER
                       and lv(cobs) < target):
                    prev = frame_key(settled(cobs), stem)
                    nxt = apply_tok(child, ("S", tok[1]))
                    nodes += 1
                    reps += 1
                    if nxt is None:
                        break
                    cobs = nxt
                    if frame_key(settled(cobs), stem) == prev:
                        break
            else:
                cobs = apply_tok(child, tok)
                nodes += 1
            if cobs is None:
                continue
            if lv(cobs) >= target:
                return child, cobs, depth + 1, dict(
                    depth=depth + 1, states=len(seen), nodes=nodes,
                    wall=round(time.time() - t0, 1), branching=max_b,
                    exhausted=False)
            if cobs.state == GameState.GAME_OVER:
                continue
            g = settled(cobs)
            k = frame_key(g, stem)
            if k in seen:
                continue
            seen.add(k)
            if algo in ("iw1", "nbfs", "mnbfs"):
                gm = mask_frame(g, stem).clip(0, 15)
                novel_mask = ~atoms[np.arange(64)[:, None],
                                    np.arange(64)[None, :], gm]
                if not novel_mask.any():
                    if algo == "iw1":
                        continue  # width > 1: prune
                    slow.append((child, cobs, depth + 1))
                    max_depth_seen = max(max_depth_seen, depth + 1)
                    continue
                atoms[np.arange(64)[:, None], np.arange(64)[None, :], gm] = True
            queue.append((child, cobs, depth + 1))
            max_depth_seen = max(max_depth_seen, depth + 1)
    else:
        exhausted = True
    return None, None, None, dict(states=len(seen), nodes=nodes,
                                  wall=round(time.time() - t0, 1),
                                  branching=max_b,
                                  max_depth=max_depth_seen, exhausted=exhausted)


def run_game(stem, algo, budget_s):
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(ROOT, "environment_files"))
    env = client.make(GAMES[stem])
    env.reset()
    t_all = time.time()
    levels = []
    won = 0
    while time.time() - t_all < budget_s:
        remain = budget_s - (time.time() - t_all)
        win_env, win_obs, depth, st = search_level2(env, stem, won + 1, algo,
                                                    min(remain, budget_s))
        if win_env is None:
            levels.append(dict(level=won + 1, solved=False, **(st or {})))
            break
        levels.append(dict(level=won + 1, solved=True, **st))
        won += 1
        env = win_env
        if win_obs is not None and win_obs.state == GameState.WIN:
            levels.append(dict(game_won=True))
            break
    return dict(game=stem, algo=algo, levels_won=won, budget_s=budget_s,
                wall=round(time.time() - t_all, 1), levels=levels)


if __name__ == "__main__":
    algo = sys.argv[1]
    budget = int(sys.argv[2])
    stems = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(GAMES)
    for stem in stems:
        try:
            res = run_game(stem, algo, budget)
        except Exception as e:  # noqa: BLE001
            res = dict(game=stem, algo=algo, error=repr(e))
        with open(os.path.join(OUT, f"{stem}_{algo}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "levels"}),
              flush=True)
        for L in res.get("levels", []):
            print("   ", L, flush=True)
