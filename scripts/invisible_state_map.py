"""Phase P passive invisible-state prevalence map (live capture driver).

For each dev game, drive the banked explorer, log per-step (masked key, unmasked key, action,
per-object interaction-count features), build Transitions, and classify determinism violations into
OVER_MERGE / INVISIBLE_STATE / UNEXPLAINED. Anchor: does ls20 surface an INVISIBLE_STATE violation?

The state key is computed by SalienceExplorer (key-IDENTICAL to the banked TransferExplorer v13, which
inherits _key verbatim), so the violation map is a property of the games, not the explorer variant.
Avatar interaction features are learned online via movement.infer_translation (the avatar = the color
whose region translates with moves); the ls20 rot-tile resolver depends on this, and the scripted
active probe (scripts/invisible_state_probe.py) is the ground-truth backstop for the ls20 anchor.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/invisible_state_map.py [budget] [games...]
"""
from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arcagi3 import invisible_state as IS  # noqa: E402
from arcagi3 import perception as P  # noqa: E402
from arcagi3.movement import infer_translation  # noqa: E402
from arcagi3.salience_explorer import SalienceExplorer  # noqa: E402

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("ismap"))

TUNE = ["vc33", "cd82", "sc25", "lp85", "lf52", "tu93", "ar25", "sp80"]
HOLDOUT = ["su15", "sk48", "re86", "wa30", "m0r0", "ls20", "tn36", "tr87"]
MODS = (2, 3, 4)


def _salient_objects(grid, bg):
    """Stable-id static objects -> set of (r,c) cells. id=(color,bbox) persists across frames for
    STATIC objects (walls, tiles, the rot glyph); moving objects (avatar) get a fresh id each frame
    (bbox changes), so they are never counted as an interaction target O."""
    objs = {}
    for o in P.connected_components(grid, background=bg):
        if o.size <= 16:   # salient = small/rare (the dense lattice handles large regions elsewhere)
            objs[(int(o.color),) + tuple(o.bbox)] = {(int(r), int(c)) for r, c in o.cells}
    return objs


def _features(counts):
    """counts: dict[oid]->int -> {f"obj{i}_mod{N}": count % N} (sorted oids for stable naming)."""
    feats = {}
    for i, (_oid, c) in enumerate(sorted(counts.items())):
        for N in MODS:
            feats[f"obj{i}_mod{N}"] = c % N
    return feats


def capture_game(prefix, budget):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    card = client.open_scorecard(tags=["is-map"])
    env = client.make(game_id=gid, scorecard_id=card)
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs = env.reset()
    n = 0
    counts = {}               # oid -> cumulative interaction count
    avatar_colors = set()     # learned online from move-translations
    prev_grid = None
    pending = None            # (mkey, ukey, action, feats) awaiting its next_mkey
    transitions = []
    while n < budget:
        st = obs.state
        if st == GameState.WIN:
            break
        grid = P.to_grid(obs.frame)
        bg = P.detect_background(grid)
        tok = pol.decide(grid, st == GameState.GAME_OVER, st == GameState.NOT_PLAYED,
                         int(obs.levels_completed or 0), list(obs.available_actions or []))
        mkey = pol.prev_key
        # close the previous transition now that we observe the resulting (current) state
        if pending is not None and mkey is not None:
            pm, pu, pa, pf = pending
            transitions.append(IS.Transition(mkey=pm, ukey=pu, action=pa, next_mkey=mkey, feats=pf))
            pending = None
        # learn the avatar color(s) from the transition INTO this state
        if prev_grid is not None and prev_grid.shape == grid.shape:
            tr = infer_translation(prev_grid, grid, bg)
            if tr is not None:
                avatar_colors.add(int(tr[0]))
        avatar_cells = ({(int(r), int(c)) for r, c in np.argwhere(np.isin(grid, list(avatar_colors)))}
                        if avatar_colors else set())
        clicked = (int(tok[2]), int(tok[1])) if tok[0] == "C" else None  # (row,col) = (y,x)
        for oid, cells in _salient_objects(grid, bg).items():
            if (avatar_cells & cells) or (clicked is not None and clicked in cells):
                counts[oid] = counts.get(oid, 0) + 1
        if tok != ("reset",) and mkey is not None:
            ukey = P.object_state_key(grid, background=bg)
            pending = (mkey, ukey, tok, _features(counts))
        else:
            pending = None
        prev_grid = grid
        if tok == ("reset",):
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
    return gid, transitions, int(obs.levels_completed or 0), len(avatar_colors)


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2:] if len(sys.argv) > 2 else (TUNE + HOLDOUT)
    t0 = time.time()
    rows = {}
    for g in games:
        try:
            gid, ts, lv, nav = capture_game(g, budget)
        except StopIteration:
            print(f"  {g}: NOT IN DEV SET", flush=True)
            continue
        s = IS.summary(ts)
        rows[g] = {**s, "gid": gid, "levels": lv, "n_transitions": len(ts), "avatar_colors": nav}
        print(f"  {g} L{lv}: pairs={s['n_pairs']} viol={s['n_violations']} "
              f"invisible={s['invisible_state']} over_merge={s['over_merge']} "
              f"unexplained={s['unexplained']} (avatar_colors={nav})", flush=True)
    walled = [g for g, r in rows.items() if r["levels"] == 0]
    inv_walled = [g for g in walled if rows[g]["invisible_state"] > 0]
    anchor = rows.get("ls20", {}).get("invisible_state", "n/a")
    print(f"\n== anchor ls20: invisible_state={anchor}", flush=True)
    print(f"== walled games={walled}; with INVISIBLE_STATE={inv_walled}", flush=True)
    print("== VERDICT bar: anchor fires AND >=2 walled games beyond ls20 with INVISIBLE_STATE", flush=True)
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)
    with open("/tmp/invisible_state_map.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("saved /tmp/invisible_state_map.json", flush=True)


if __name__ == "__main__":
    main()
