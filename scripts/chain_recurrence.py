"""Chain-recurrence kill-test for the geodesic-replay / CausalChainMacro bet (greenlit by the RHAE
headroom oracle). Question: do consecutive level-ups within a game share a REPLAYABLE proximate causal
chain? Drive SalienceExplorer with EventExtractor; at each level-up capture the role-stripped
(event-type) sequence of the last K real-event steps ending in LEVEL_COMPLETED; measure the longest
common SUFFIX across consecutive levels vs a shuffled baseline. Recurring suffix >=2 => the mechanic is
consistent and replayable (build the executor); ~1 (only LEVEL_COMPLETED shared) => not replayable.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/chain_recurrence.py [budget] [games]
"""
from __future__ import annotations
import logging, sys, random
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.events import EventExtractor
logging.basicConfig(level=logging.ERROR)
random.seed(0)
K = 6
DEFAULT = ["tu93","vc33","lf52","ar25","m0r0","cd82"]   # games that reach >=2 levels


def common_suffix(a, b):
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x == y:
            n += 1
        else:
            break
    return n


def run(game, budget):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files",
                    logger=logging.getLogger("ch"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env = client.make(game_id=gid, scorecard_id=f"ch-{game}")
    pol = SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    ext = EventExtractor()
    obs = env.reset(); before = P.to_grid(obs.frame); bg = P.detect_background(before)
    n = 0; prev_levels = 0; real_log = []; chains = []
    while n < budget:
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(before, obs.state == GameState.GAME_OVER,
                         obs.state == GameState.NOT_PLAYED, int(obs.levels_completed or 0),
                         list(obs.available_actions or []))
        click_xy = (int(tok[1]), int(tok[2])) if tok[0] == "C" else None
        if tok == ("reset",):
            obs = env.reset(); after = P.to_grid(obs.frame)
            real_log = []                     # new episode -> reset proximate window
            before = after; n += 1; continue
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        after = P.to_grid(obs.frame)
        lv = int(obs.levels_completed or 0)
        reward = max(0, lv - prev_levels)
        se = ext.extract(before, after, tok, float(reward), bg=bg, click_xy=click_xy)
        if se.salient:
            real_log.append(tuple(e.type.name for e in se.salient))
        if reward > 0:
            flat = [t for step in real_log[-K:] for t in step]   # flatten the proximate window
            chains.append(tuple(flat[-K:]))
            real_log = []
            prev_levels = lv
        before = after; n += 1
    return chains


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    games = sys.argv[2].split(",") if len(sys.argv) > 2 else DEFAULT
    print(f"chain-recurrence kill-test (budget {budget}); proximate window K={K}\n")
    real_suffixes, shuf_suffixes = [], []
    for g in games:
        chains = run(g, budget)
        print(f"== {g}: {len(chains)} level chains")
        for i, c in enumerate(chains):
            print(f"   L{i+1}: {list(c)}")
        for i in range(1, len(chains)):
            cs = common_suffix(chains[i-1], chains[i])
            real_suffixes.append(cs)
            sh = list(chains[i]); random.shuffle(sh)
            shuf_suffixes.append(common_suffix(chains[i-1], tuple(sh)))
        if len(chains) >= 2:
            css = [common_suffix(chains[i-1], chains[i]) for i in range(1, len(chains))]
            print(f"   -> consecutive common-suffix depths: {css}")
    import statistics as st
    if real_suffixes:
        print(f"\n== ACROSS GAMES: median common-suffix = {st.median(real_suffixes):.1f} "
              f"(shuffled baseline {st.median(shuf_suffixes):.1f}); "
              f">=2 fraction = {sum(1 for x in real_suffixes if x>=2)/len(real_suffixes):.0%} "
              f"(n={len(real_suffixes)} consecutive pairs)")
        print("Interpretation: median suffix >=2 AND >> shuffled => proximate causal chain RECURS across "
              "levels = REPLAYABLE (build the executor). ~1 => only LEVEL_COMPLETED shared = not replayable.")


if __name__ == "__main__":
    main()
