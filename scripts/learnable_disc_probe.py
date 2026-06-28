"""Decisive learnable-discriminator probe: do CLEAN per-transition features separate help from derail
across the dev set? For each game reaching L2+: (banked) per-level effective-sig SETS -> per-transition
jaccard + n_sigs(L_k); (hard) max-level + L2-cost vs banked -> help/derail/neutral label. Tabulate; if
[high jaccard + n_sigs>=2]=help and [low jaccard OR n_sigs=1]=derail separates cleanly -> the discriminator
is LEARNABLE IN PRINCIPLE (the gate failures were noisy ONLINE capture). If classes overlap -> unlearnable."""
from __future__ import annotations
import logging
from dotenv import load_dotenv
load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.salience_explorer import SalienceExplorer
from arcagi3.chain_macro_explorer import ChainMacroExplorer
logging.basicConfig(level=logging.ERROR)
DENSE=dict(seed=0, trust_threshold=3, border_mask=2, coarse_grid_step=4, max_click_targets=256)

def sigsets(game, budget=14000):
    client=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("d"))
    gid=next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env=client.make(game_id=gid, scorecard_id=f"d-{game}")
    pol=SalienceExplorer(seed=0, trust_threshold=3, border_mask=2)
    obs=env.reset(); before=P.to_grid(obs.frame); bg=P.detect_background(before); n=0; prev=0; per=[set()]
    while n<budget:
        if obs.state==GameState.WIN: break
        tok=pol.decide(before, obs.state==GameState.GAME_OVER, obs.state==GameState.NOT_PLAYED, int(obs.levels_completed or 0), list(obs.available_actions or []))
        src=pol.prev_key
        if tok==("reset",): obs=env.reset(); before=P.to_grid(obs.frame); n+=1; continue
        elif tok[0]=="S": obs=env.step(GameAction.from_id(tok[1]))
        else: obs=env.step(GameAction.ACTION6, data={"x":int(tok[1]),"y":int(tok[2])})
        after=P.to_grid(obs.frame)
        if tok[0]=="C" and src is not None and pol._key(after)!=src: per[-1].add(int(before[tok[2],tok[1]]))
        lv=int(obs.levels_completed or 0)
        if lv>prev: per.append(set()); prev=lv
        before=after; n+=1
    return [s for s in per if s]

def maxlevel(make, game, budget=12000):
    client=Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files", logger=logging.getLogger("d"))
    gid=next(e.game_id for e in client.get_environments() if e.game_id.startswith(game))
    env=client.make(game_id=gid, scorecard_id=f"m-{game}")
    pol=make(); obs=env.reset(); n=0; best=0
    while n<budget:
        if obs.state==GameState.WIN: break
        tok=pol.decide(P.to_grid(obs.frame), obs.state==GameState.GAME_OVER, obs.state==GameState.NOT_PLAYED, int(obs.levels_completed or 0), list(obs.available_actions or []))
        if tok==("reset",): obs=env.reset()
        elif tok[0]=="S": obs=env.step(GameAction.from_id(tok[1]))
        else: obs=env.step(GameAction.ACTION6, data={"x":int(tok[1]),"y":int(tok[2])})
        n+=1; best=max(best,int(obs.levels_completed or 0))
    return best

GAMES=["lp85","cd82","vc33","ar25","tu93","lf52","m0r0","su15","sp80"]
print(f"{'game':>6} {'bank_L':>6} {'hard_L':>6} {'label':>8} | per-transition (jaccard, n_sigs(Lk)):")
for g in GAMES:
    bs=sigsets(g)
    bl=maxlevel(lambda:SalienceExplorer(**DENSE),g)
    hl=maxlevel(lambda:ChainMacroExplorer(enable_macro=True,macro_mode="hard",**DENSE),g)
    label="DERAIL" if hl<bl else ("?" if len(bs)<2 else "ok")
    trans=[]
    for i in range(1,len(bs)):
        a,b=bs[i-1],bs[i]; jac=len(a&b)/len(a|b) if (a|b) else 0; trans.append(f"(j={jac:.2f},ns={len(a)})")
    print(f"{g:>6} {bl:>6} {hl:>6} {label:>8} | "+" ".join(trans))
