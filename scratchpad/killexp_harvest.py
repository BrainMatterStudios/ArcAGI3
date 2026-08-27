"""Kill-experiment harvester: exhaustive ACTION6 sweep of the level-0 start state.

For every game in environment_files/ we reset to the level-0 start state, click each of
the 4096 cells once (reset before each click -> deterministic, independent probes) and
record the effect class of the clicked cell.

Output: scratchpad/killexp_data/<prefix>.npz  + <prefix>.json (meta)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/ahmed/Documents/ArcAGI3/src")
os.chdir("/Users/ahmed/Documents/ArcAGI3")

import logging  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import perception as P  # noqa: E402

logging.basicConfig(level=logging.ERROR)
OUT = "scratchpad/killexp_data"
os.makedirs(OUT, exist_ok=True)
GRID = 64


def comp_labels(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """4-connectivity same-colour components over ALL cells (background included)."""
    h, w = grid.shape
    lab = -np.ones((h, w), dtype=np.int32)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if lab[r, c] >= 0:
                continue
            color = int(grid[r, c])
            idx = len(comps)
            q = deque([(r, c)])
            lab[r, c] = idx
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and lab[nr, nc] < 0 and grid[nr, nc] == color:
                        lab[nr, nc] = idx
                        q.append((nr, nc))
            arr = np.array(cells)
            comps.append({
                "color": color,
                "size": len(cells),
                "cr": float(arr[:, 0].mean()),
                "cc": float(arr[:, 1].mean()),
                "r0": int(arr[:, 0].min()), "r1": int(arr[:, 0].max()),
                "c0": int(arr[:, 1].min()), "c1": int(arr[:, 1].max()),
            })
    return lab, comps


def find_avatar(env, base_grid, avail):
    """Presumed avatar = centroid of cells that change under simple move actions."""
    moves = [a for a in (1, 2, 3, 4, 5, 7) if a in avail]
    if not moves:
        return None
    changed = np.zeros(base_grid.shape, dtype=np.int32)
    env.reset()
    prev = base_grid
    for i in range(24):
        a = moves[i % len(moves)]
        obs = env.step(GameAction.from_id(a))
        g = P.to_grid(obs.frame)
        if g.shape == prev.shape:
            changed += (g != prev).astype(np.int32)
        prev = g
        if obs.state in (GameState.GAME_OVER, GameState.WIN):
            env.reset()
            prev = base_grid
    m = changed > 0
    if m.sum() == 0 or m.sum() > 400:
        return None
    rs, cs = np.nonzero(m)
    return (float(rs.mean()), float(cs.mean()))


def harvest(prefix: str, cap_s: float = 600.0):
    client = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir="environment_files", logger=logging.getLogger("h"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid)
    obs = env.reset()
    avail = sorted(int(a) for a in (obs.available_actions or []))
    base = P.to_grid(obs.frame)
    win_levels = int(obs.win_levels or 0)

    avatar = find_avatar(env, base, set(avail))
    obs = env.reset()
    base = P.to_grid(obs.frame)

    lab, comps = comp_labels(base)
    colcount = {int(c): int((base == c).sum()) for c in range(16) if (base == c).sum()}
    rank = {c: i for i, (c, _) in enumerate(sorted(colcount.items(), key=lambda kv: kv[1]))}

    t0 = time.time()
    diff_count = np.zeros(base.shape, dtype=np.int32)  # for HUD/volatility detection
    recs = []
    capped = False
    for idx in range(GRID * GRID):
        if time.time() - t0 > cap_s:
            capped = True
            break
        y, x = divmod(idx, GRID)  # y=row, x=col
        env.reset()
        o = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        g = P.to_grid(o.frame)
        lv = int(o.levels_completed or 0)
        st = o.state
        d = (g != base) if g.shape == base.shape else np.ones(base.shape, dtype=bool)
        diff_count += d.astype(np.int32)
        recs.append((y, x, d, lv, int(st == GameState.GAME_OVER), int(st == GameState.WIN)))

    # volatile / HUD cells: change under (almost) every click -> step counters
    n = len(recs)
    volatile = (diff_count / max(n, 1)) >= 0.9
    rows = []
    for (y, x, d, lv, over, win) in recs:
        dm = d & ~volatile
        nch = int(dm.sum())
        if nch:
            rs, cs = np.nonzero(dm)
            far = int(max(np.abs(rs - y).max(), np.abs(cs - x).max()))
            spread = int(max(rs.max() - rs.min(), cs.max() - cs.min()))
        else:
            far, spread = -1, -1
        rows.append((y, x, nch, far, spread, lv, over, win))
    rows = np.array(rows, dtype=np.int32)

    np.savez_compressed(f"{OUT}/{prefix}.npz", base=base, lab=lab,
                        volatile=volatile, rows=rows)
    meta = {"gid": gid, "avail": avail, "win_levels": win_levels, "avatar": avatar,
            "capped": capped, "n_probed": n, "secs": round(time.time() - t0, 1),
            "comps": comps, "rank": {str(k): v for k, v in rank.items()},
            "colcount": {str(k): v for k, v in colcount.items()}}
    with open(f"{OUT}/{prefix}.json", "w") as f:
        json.dump(meta, f)
    eff = int(((rows[:, 2] > 0) | (rows[:, 5] > 0) | (rows[:, 6] > 0) | (rows[:, 7] > 0)).sum())
    print(f"{prefix} {gid} avail={avail} probed={n} capped={capped} "
          f"effective={eff} ({100*eff/max(n,1):.1f}%) volatile={int(volatile.sum())} "
          f"avatar={avatar} t={time.time()-t0:.1f}s", flush=True)
    return eff, n


if __name__ == "__main__":
    prefixes = sys.argv[1:] or sorted(os.listdir("environment_files"))
    for p in prefixes:
        try:
            harvest(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p} FAILED {type(e).__name__}: {e}", flush=True)
