"""EXPERIMENT #2: two-phase SELECTION -> APPLICATION affordance induction (gap #1 from the click experiment).

Many click games have hidden selection state: clicking a "selector" (palette color, spell, tool) changes little
in the frame but changes the EFFECT of a subsequent "application" click (paint a slot, place a piece). The
single-click probe can't see this (the selector looks like a no-op), so sb26/sc25-class games stalled.

Detection is game-agnostic and interventional: for candidate selector S and applier T,
    effect(T alone)  vs  effect(S then T)
If they DIFFER, S conditions T -> (S,T) is a compound affordance. This recovers the select->apply mechanic from
interaction with no per-game code. We then add the detected compounds as macro-actions to the search.

Reports: whether select->apply was DETECTED, how many compound affordances, and whether compounds unlock a solve.
"""
from __future__ import annotations
import sys
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from click_affordance_probe import salient_targets, HUMAN_PROXY

GAMES = ["sb26", "sc25", "cd82", "cn04", "lf52", "sk48", "tn36"]


def _delta(a, b):
    return int(np.sum(a[:56, :56] != b[:56, :56]))


def run(game, verbose=True, max_nodes=6000):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"sa-{game}")
    obs = env.reset()
    g0 = P.to_grid(obs.frame)
    avail0 = list(obs.available_actions or [])
    targets = salient_targets(g0, max_n=16)

    def click(cx, cy):
        nonlocal obs
        obs = env.step(GameAction.ACTION6, data={"x": cx, "y": cy})
    def won():
        return obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1
    def terminal():
        return obs.state == GameState.GAME_OVER

    # baseline single-click effect for every target (from a fresh reset each time)
    base = {}
    for (cx, cy) in targets:
        obs = env.reset(); click(cx, cy)
        base[(cx, cy)] = (0 if terminal() else _delta(P.to_grid(obs.frame), g0), won())
    single_win = next((t for t in targets if base[t][1]), None)

    # candidate selectors = low-visible-effect clicks; appliers = all targets
    selectors = [t for t in targets if base[t][0] <= 3 and not base[t][1]]
    compounds = []  # (S, T): clicking S first changes what T does
    for s in selectors[:8]:
        for t in targets[:12]:
            if t == s:
                continue
            obs = env.reset(); click(*s)
            if terminal() or won():
                if won(): single_win = single_win or s
                continue
            click(*t)
            d_st = 0 if terminal() else _delta(P.to_grid(obs.frame), g0)
            if won():
                compounds.append((s, t)); continue
            # S conditions T iff (S then T) differs materially from (T alone)
            if abs(d_st - base[t][0]) >= 2:
                compounds.append((s, t))
    detected = len(compounds) > 0

    if verbose:
        print(f"{game:>6}: targets={len(targets)} single-meaningful={sum(1 for t in targets if base[t][0]>=2)} "
              f"selectors={len(selectors)} -> SELECT->APPLY {'DETECTED' if detected else 'not detected'} "
              f"({len(compounds)} compound affordances){'  [single-click win!]' if single_win else ''}")

    # search over: single meaningful clicks + detected compound macros + A5(submit) + moves
    actions = []
    for t in targets:
        if base[t][0] >= 2 and not terminal():
            actions.append([("C", *t)])
    for (s, t) in compounds:
        actions.append([("C", *s), ("C", *t)])           # macro = 2 clicks
    for a in avail0:
        if a in (1, 2, 3, 4, 5):
            actions.append([("S", a)])

    def apply(seq_of_toks):
        nonlocal obs
        for tok in seq_of_toks:
            if tok[0] == "C":
                obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
            else:
                obs = env.step(GameAction.from_id(int(tok[1])))
            if won():
                return
    def replay(seq):
        nonlocal obs
        obs = env.reset()
        for macro in seq:
            apply(macro)
            if won():
                return True
        return False

    replay([]); seen = {P.to_grid(obs.frame)[:56, :56].tobytes()}; q = deque([[]]); sol = None; nodes = 0
    while q and sol is None and nodes < max_nodes:
        seq = q.popleft()
        for macro in actions:
            replay(seq); apply(macro); nodes += 1
            if won():
                sol = seq + [macro]; break
            k = P.to_grid(obs.frame)[:56, :56].tobytes()
            if k not in seen:
                seen.add(k); q.append(seq + [macro])
    plan_len = sum(len(m) for m in sol) if sol else None
    if verbose:
        hp = HUMAN_PROXY.get(game)
        r = f" ({plan_len/hp:.1f}x human)" if plan_len and hp else ""
        print(f"        search over {len(actions)} macro-actions -> "
              f"{'SOLVED plan='+str(plan_len)+r if sol else 'unsolved'} (nodes={nodes})")
    return dict(game=game, detected=detected, n_compounds=len(compounds), solved=sol is not None, plan_len=plan_len)


if __name__ == "__main__":
    games = sys.argv[1:] or GAMES
    print("Two-phase SELECT->APPLY affordance induction (game-agnostic):\n")
    res = [run(g) for g in games]
    print(f"\nSUMMARY: select->apply DETECTED in {sum(r['detected'] for r in res)}/{len(res)}; "
          f"solved {sum(r['solved'] for r in res)}/{len(res)} via compound macros + search.")
