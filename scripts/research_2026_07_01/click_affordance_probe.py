"""EXPERIMENT: general click-affordance induction on the click-dominated game distribution.

Motivation (measured): 19/25 dev games need ACTION6 clicks; the movement-model stack is blind to them. A human
plays a click game by clicking a salient thing, seeing what it does, forming a rule, and exploiting it. This
experiment tests, GAME-AGNOSTICALLY (frames+feedback only, no per-game code, no introspection), whether that
approach reaches first-reward efficiently.

Method per game:
  Phase 1 PROBE  — enumerate salient click targets (one solid cell per distinct object). Click each ONCE from the
                   start state (reset before each so effects are attributable), record: masked frame-delta size,
                   reward/level-up, terminal. This yields an AFFORDANCE MAP: which clicks are MEANINGFUL (cause
                   change) vs no-ops, and whether any single click already wins.
  Phase 2 SEARCH — affordance-PRUNED joint BFS over {meaningful clicks} + {moves if available} + {A5 if avail},
                   via reset+replay with HUD-masked frame-state dedup, until first level-up. Pruning to meaningful
                   actions is the human-like move that keeps branching tractable.

Reports per game: n_targets, n_meaningful, single_click_win, solved, scored_plan_len (near-human, the replayable
solution), exploration_actions (probe+search cost). Compares scored_plan_len to a hand-set human/optimal proxy.
"""
from __future__ import annotations
import sys
from collections import deque
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

# near-optimal solve lengths from this session's dev solvers = human proxy (actions to L0)
HUMAN_PROXY = {"sb26": 9, "sc25": 13, "dc22": 20, "re86": 20, "wa30": 25}
CLICK_GAMES = ["ft09","lp85","r11l","s5i5","tn36","vc33","su15",  # click-only
               "sb26","sc25","dc22","ka59","bp35","lf52","sk48"]  # click+move


def salient_targets(grid, max_n=28):
    """one representative SOLID cell per distinct non-bg object, ranked by salience (rarer colors first)."""
    bg = P.detect_background(grid)
    comps = [o for o in P.connected_components(grid, background=bg) if o.color != bg and o.size >= 2]
    from collections import Counter
    colcount = Counter(o.color for o in comps)
    targets = []
    for o in sorted(comps, key=lambda o: (colcount[o.color], -o.size)):
        cy, cx = int(round(o.centroid[0])), int(round(o.centroid[1]))
        if 0 <= cy < 64 and 0 <= cx < 64 and grid[cy, cx] == o.color:
            targets.append((cx, cy))
        if len(targets) >= max_n:
            break
    # dedup close-together
    out = []
    for t in targets:
        if all(abs(t[0]-u[0]) + abs(t[1]-u[1]) > 2 for u in out):
            out.append(t)
    return out


def _mask(grid):
    return grid[:56, :56].tobytes()


def run(game, probe_budget=60, max_nodes=4000, verbose=True):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"clk-{game}")
    obs = env.reset()
    grid0 = P.to_grid(obs.frame)
    base_mask = _mask(grid0)
    avail0 = list(obs.available_actions or [])
    targets = salient_targets(grid0)

    def apply(tok):
        nonlocal obs
        if tok[0] == "C":
            obs = env.step(GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        else:
            obs = env.step(GameAction.from_id(int(tok[1])))

    def won():
        return obs.state == GameState.WIN or int(obs.levels_completed or 0) >= 1

    # ---- Phase 1: single-click affordance probe (reset before each for attribution) ----
    meaningful = []; single_win = None
    for (cx, cy) in targets:
        obs = env.reset()
        obs = env.step(GameAction.ACTION6, data={"x": cx, "y": cy})
        if won():
            single_win = (cx, cy); meaningful.append((cx, cy)); break
        if obs.state == GameState.GAME_OVER:
            continue  # deadly click: exclude from search
        delta = int(np.sum(P.to_grid(obs.frame)[:56, :56].tobytes() != base_mask)) if False else \
            int(np.sum(P.to_grid(obs.frame)[:56, :56] != grid0[:56, :56]))
        if delta >= 2:
            meaningful.append((cx, cy))

    n_targets, n_meaning = len(targets), len(meaningful)
    if single_win:
        if verbose:
            print(f"{game:>6}: targets={n_targets} meaningful={n_meaning} -> SINGLE-CLICK WIN in 1 action")
        return dict(game=game, n_targets=n_targets, n_meaningful=n_meaning, single_win=True,
                    solved=True, plan_len=1, explore=n_targets + 1)

    # ---- Phase 2: affordance-pruned joint search (meaningful clicks + moves + A5) ----
    actions = [("C", cx, cy) for (cx, cy) in meaningful]
    actions += [("S", a) for a in avail0 if a in (1, 2, 3, 4, 5)]

    def replay(seq):
        nonlocal obs
        obs = env.reset()
        for tok in seq:
            apply(tok)
            if won():
                return True
        return False

    replay([]); seen = {_mask(P.to_grid(obs.frame))}; q = deque([[]]); sol = None; nodes = 0
    while q and sol is None and nodes < max_nodes:
        seq = q.popleft()
        for a in actions:
            replay(seq); apply(a); nodes += 1
            if won():
                sol = seq + [a]; break
            k = _mask(P.to_grid(obs.frame))
            if k not in seen:
                seen.add(k); q.append(seq + [a])
    solved = sol is not None
    plan_len = len(sol) if sol else None
    explore = n_targets + nodes
    if verbose:
        hp = HUMAN_PROXY.get(game)
        ratio = f"{plan_len/hp:.1f}x human" if (plan_len and hp) else ""
        print(f"{game:>6}: targets={n_targets} meaningful={n_meaning} "
              f"-> {'SOLVED plan='+str(plan_len)+' '+ratio if solved else 'unsolved'} "
              f"(explore~{explore} actions, search nodes={nodes})")
    return dict(game=game, n_targets=n_targets, n_meaningful=n_meaning, single_win=False,
                solved=solved, plan_len=plan_len, explore=explore)


if __name__ == "__main__":
    games = sys.argv[1:] or CLICK_GAMES
    print("General click-affordance probe+search on click games (game-agnostic, frames-only):\n")
    results = [run(g) for g in games]
    solved = [r for r in results if r["solved"]]
    print(f"\nSUMMARY: solved {len(solved)}/{len(results)} click games with ZERO per-game code.")
    near = [r for r in solved if r["plan_len"] and HUMAN_PROXY.get(r["game"]) and r["plan_len"] <= 2*HUMAN_PROXY[r["game"]]]
    print(f"  of those with a human proxy, {len(near)} solved within 2x human plan length (RHAE-viable if replayed).")
