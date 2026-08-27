"""Task 4: cell-granularity sweep for GoExploreExplorer, OFFLINE.

Same loop as scripts/eval_efficiency.run_game but against environment_files (OperationMode.OFFLINE)
and sweeping GoExploreExplorer(block=...) over {2,4,8,16}.
Usage: PYTHONPATH=src python scratchpad/killexp_blocksweep.py [budget] [game ...]
"""
from __future__ import annotations
import logging, sys, json, time
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.go_explore_explorer import GoExploreExplorer

logging.basicConfig(level=logging.ERROR)
client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
BLOCKS = [2, 4, 8, 16]


def run_game(prefix, block, budget, seed=0):
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id="blocksweep")
    pol = GoExploreExplorer(seed=seed, trust_threshold=3, border_mask=2, block=block)
    obs = env.reset()
    n = best = last = 0
    marks = []
    while n < budget:
        st = obs.state
        tok = pol.decide(P.to_grid(obs.frame),
                         gstate_terminal=(st == GameState.GAME_OVER),
                         gstate_notplayed=(st == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        if st == GameState.WIN:
            break
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        n += 1
        lv = int(obs.levels_completed or 0)
        if lv > best:
            marks.append(n - last); last = n; best = lv
    return {"levels": best, "marks": marks, "actions": n, "archive": len(pol.archive)}


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    games = sys.argv[2:] or ["cd82", "ls20", "tu93", "sk48", "lf52"]
    seeds = [0, 1, 2]
    res = {}
    for g in games:
        for b in BLOCKS:
            for sd in seeds:
                t0 = time.time()
                r = run_game(g, b, budget, seed=sd)
                r["wall"] = round(time.time() - t0, 1)
                res[f"{g}|{b}|{sd}"] = r
                print(f"[{g} block={b:>2} seed={sd}] levels={r['levels']} archive={r['archive']} "
                      f"marks={r['marks']} wall={r['wall']}s", flush=True)
    print("\n  game |" + "".join(f"  b={b:<2} mean_lv (max) arch |" for b in BLOCKS))
    for g in games:
        line = f"{g:>6} |"
        for b in BLOCKS:
            lvs = [res[f"{g}|{b}|{sd}"]["levels"] for sd in seeds]
            ar = sum(res[f"{g}|{b}|{sd}"]["archive"] for sd in seeds) / len(seeds)
            line += f" {sum(lvs)/len(lvs):>13.2f} ({max(lvs)}) {ar:>6.0f} |"
        print(line)
    print("\n  block | mean levels over all games/seeds")
    for b in BLOCKS:
        lvs = [res[f"{g}|{b}|{sd}"]["levels"] for g in games for sd in seeds]
        print(f"  {b:>5} | {sum(lvs)/len(lvs):.3f}   (per-game max sum={sum(max(res[f'{g}|{b}|{sd}']['levels'] for sd in seeds) for g in games)})")
    print("BLOCKSWEEP_JSON " + json.dumps(res))


if __name__ == "__main__":
    main()
