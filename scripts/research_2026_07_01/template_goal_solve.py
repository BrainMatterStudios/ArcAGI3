"""FRONTIER: general GOAL INFERENCE for the "reproduce a reference sequence via a palette" class (sb26-style).

The goal state is DISPLAYED, not searched. Structure, all affordance-grounded (no per-game code):
  - PALETTE  = selector cells (clicking one conditions a later apply); each has a swatch COLOR.
  - SLOTS    = applier cells (their color changes when a selected color is applied).
  - TARGET   = a group of colored cells, same count as the slots, that are NEITHER selectors nor appliers
               (a static reference). Its colors, in spatial order, are the desired slot colors.
Plan: for slot i, click the palette selector whose color == target[i], click slot i; then submit (A5). This is
the human read: "make the row match the target, picking colors from the palette."
"""
from __future__ import annotations
import sys
from collections import Counter
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from click_affordance_probe import salient_targets


def _color_at(grid, cx, cy):
    return int(grid[cy, cx])


def solve(game, verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"tg-{game}")
    obs = env.reset()
    g0 = P.to_grid(obs.frame)
    avail = list(obs.available_actions or [])
    targets = salient_targets(g0, max_n=20)

    def click(cx, cy):
        nonlocal obs; obs = env.step(GameAction.ACTION6, data={"x": cx, "y": cy})
    def won(): return obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1
    def dead(): return obs.state == GameState.GAME_OVER
    def dl(a, b): return int(np.sum(a[:56, :56] != b[:56, :56]))

    # single-click baseline effect
    base = {}
    for (cx, cy) in targets:
        obs = env.reset(); click(cx, cy)
        base[(cx, cy)] = (0 if dead() else dl(P.to_grid(obs.frame), g0), won())

    # two-phase: find (selector S -> applier T) compounds; record which cells are selectors vs appliers
    selectors, appliers = set(), set()
    cand = [t for t in targets if not base[t][1]]
    for s in cand:
        for t in cand:
            if t == s: continue
            obs = env.reset(); click(*s)
            if dead() or won(): continue
            b_before = P.to_grid(obs.frame); click(*t)
            if dead(): continue
            d = dl(P.to_grid(obs.frame), b_before)
            d_alone = base[t][0]
            if won() or (d >= 2 and abs(d - d_alone) >= 2):
                selectors.add(s); appliers.add(t)

    if not selectors or not appliers:
        if verbose: print(f"{game:>6}: no select->apply structure -> n/a"); return dict(game=game, solved=False)

    sel_color = {s: _color_at(g0, s[0], s[1]) for s in selectors}
    palette_colors = set(sel_color.values())
    slots = sorted(appliers, key=lambda t: (t[0], t[1]))     # spatial order left-to-right (col, then row)

    # TARGET = the reference boxes: colored objects whose color is a PALETTE color (you reproduce the target
    # using the palette), that are NOT the palette swatches and NOT the slots. Ordered left-to-right.
    bg = P.detect_background(g0)
    objs = []
    for o in P.connected_components(g0, background=bg):
        if o.color not in palette_colors:
            continue
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        near_sel = any(abs(cx - s[0]) + abs(cy - s[1]) <= 3 for s in selectors)
        near_slot = any(abs(cx - t[0]) + abs(cy - t[1]) <= 3 for t in appliers)
        if not near_sel and not near_slot:
            objs.append((o.color, cy, cx))
    objs_sorted = sorted(objs, key=lambda o: (o[2], o[1]))    # left-to-right
    target_seq = [o[0] for o in objs_sorted]
    if verbose:
        print(f"{game:>6}: selectors={len(selectors)} slots={len(slots)} target_objs={len(target_seq)} "
              f"target_colors={target_seq[:len(slots)]} palette={sorted(set(sel_color.values()))}")
    if len(slots) < 2 or len(target_seq) < len(slots) or any(c in (0,) for c in target_seq[:len(slots)]):
        if verbose: print(f"        degenerate/insufficient target (slots={len(slots)}, "
                          f"target={len(target_seq)}) -> abstain")
        return dict(game=game, solved=False)
    target_seq = target_seq[:len(slots)]

    # PLAN: for slot i, pick palette selector of color target_seq[i], apply to slot; then submit
    obs = env.reset(); n = 0
    for i, slot in enumerate(slots):
        want = target_seq[i]
        sel = next((s for s in selectors if sel_color[s] == want), None)
        if sel is None:
            continue
        click(*sel); n += 1
        if won(): break
        click(slot[0], slot[1]); n += 1
        if won(): break
    if not won() and 5 in avail:
        obs = env.step(GameAction.from_id(5)); n += 1
    solved = won()
    if verbose:
        print(f"        plan {n} actions -> {'*** SOLVED ***' if solved else 'not solved (levels %s)'%obs.levels_completed}")
    return dict(game=game, solved=solved, actions=n)


if __name__ == "__main__":
    games = sys.argv[1:] or ["sb26", "cn04", "lf52"]
    print("General GOAL-INFERENCE for reproduce-reference-via-palette class (zero per-game code):\n")
    for g in games:
        solve(g)
