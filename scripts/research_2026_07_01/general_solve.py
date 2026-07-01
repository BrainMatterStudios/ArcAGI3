"""UNIFIED general solver — the human-like loop, ONE pipeline, ZERO per-game code, frames+feedback only.
Measures the honest aggregate across all 25 dev games: how many does a game-agnostic agent solve, at what
efficiency, by (1) parsing objects, (2) inducing single-click + two-phase select->apply affordances, (3)
searching over MEANINGFUL macro-actions (pruned by affordance) toward first reward.

This is the bottom-line the whole phase-V research points at: not per-game solvers, but one agent that induces
each game's mechanic from interaction and exploits it. Honest about what it does and does not cover.
"""
from __future__ import annotations
import sys
from collections import deque, Counter
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from click_affordance_probe import salient_targets, HUMAN_PROXY, CLICK_GAMES

ALL = ["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0",
       "r11l","re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]


def solve(game, max_nodes=3000, probe_cap=18, verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"gen-{game}")
    obs = env.reset()
    g0 = P.to_grid(obs.frame)
    avail0 = list(obs.available_actions or [])
    has_click = 6 in avail0
    targets = salient_targets(g0, max_n=probe_cap) if has_click else []

    def step(tok):
        nonlocal obs
        if tok[0] == "C":
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        else:
            obs = env.step(GameAction.from_id(int(tok[1])))
    def won():
        return obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1
    def dead():
        return obs.state == GameState.GAME_OVER
    def dl(g):
        return int(np.sum(g[:56, :56] != g0[:56, :56]))

    # ---- affordance induction: single-click + two-phase select->apply ----
    macros = []                         # each macro = list of tokens
    base = {}
    single_win = None
    for t in targets:
        obs = env.reset(); step(("C", *t))
        if won():
            single_win = [("C", *t)]; break
        base[t] = (0 if dead() else dl(P.to_grid(obs.frame)))
        if not dead() and base[t] >= 2:
            macros.append([("C", *t)])
    if single_win:
        if verbose: print(f"{game:>6}: single-click win")
        return dict(game=game, solved=True, plan_len=1, has_click=has_click)
    selectors = [t for t in targets if base.get(t, 99) <= 3][:7]
    for s in selectors:
        for t in targets[:10]:
            if t == s: continue
            obs = env.reset(); step(("C", *s))
            if dead() or won(): continue
            step(("C", *t)); d = 0 if dead() else dl(P.to_grid(obs.frame))
            if won() or abs(d - base.get(t, 0)) >= 2:
                macros.append([("C", *s), ("C", *t)])
    # movement + A5 as atomic macros
    macros += [[("S", a)] for a in avail0 if a in (1, 2, 3, 4, 5)]

    # ---- affordance-pruned joint search to first reward ----
    def replay(seq):
        nonlocal obs
        obs = env.reset()
        for m in seq:
            for tok in m:
                step(tok)
                if won(): return True
        return False
    replay([]); seen = {P.to_grid(obs.frame)[:56, :56].tobytes()}; q = deque([[]]); sol = None; nodes = 0
    while q and sol is None and nodes < max_nodes:
        seq = q.popleft()
        for m in macros:
            replay(seq); [step(t) or None for t in m if not won()]
            nodes += 1
            if won(): sol = seq + [m]; break
            k = P.to_grid(obs.frame)[:56, :56].tobytes()
            if k not in seen: seen.add(k); q.append(seq + [m])
    plan_len = sum(len(m) for m in sol) if sol else None
    if verbose:
        hp = HUMAN_PROXY.get(game)
        r = f" {plan_len/hp:.1f}x-human" if plan_len and hp else ""
        print(f"{game:>6}: click={int(has_click)} macros={len(macros)} -> "
              f"{'SOLVED plan='+str(plan_len)+r if sol else 'unsolved'} (nodes={nodes})")
    return dict(game=game, solved=sol is not None, plan_len=plan_len, has_click=has_click)


if __name__ == "__main__":
    games = sys.argv[1:] or ALL
    print("UNIFIED game-agnostic solver across dev games (zero per-game code, frames only):\n")
    res = [solve(g) for g in games]
    s = [r for r in res if r["solved"]]
    print(f"\n=== AGGREGATE: solved {len(s)}/{len(res)} games with ZERO per-game code ===")
    print("  solved:", ", ".join(r["game"] for r in s))
    prox = [r for r in s if HUMAN_PROXY.get(r["game"]) and r["plan_len"]]
    if prox:
        import numpy as _n
        print(f"  efficiency vs human proxy (solved w/ proxy): "
              + ", ".join(f"{r['game']}={r['plan_len']/HUMAN_PROXY[r['game']]:.1f}x" for r in prox))
