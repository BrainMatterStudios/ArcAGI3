"""Generalization guard: the GENERAL capabilities (generic novelty search, peg-solitaire class) must solve a
dev game even when the 16-color palette is PERMUTED (a proxy for a re-skinned HIDDEN game). This locks the
color-agnostic property the leave-palette-out experiment established (general = robust; only the niche color-
keyed classes centroid-drag/pull-drag are brittle, which is acceptable — they are dev-specific)."""
import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P
from arcagi3.coroutine_strategy import CoroutineStrategy, general_agent_gen


def _solve_permuted(game, budget=45000):
    rng = np.random.default_rng(7); perm = np.arange(16); rng.shuffle(perm); perm = perm.astype(np.int8)
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith(game))
    env = c.make(game_id=gid, scorecard_id=f"pg-{game}"); obs = env.reset()
    last = np.zeros((64, 64), np.int8); pol = CoroutineStrategy(general_agent_gen); mx = 0
    for _ in range(budget):
        try:
            g = P.to_grid(obs.frame) if (obs.frame is not None and len(obs.frame)) else last
            g = perm[np.clip(g, 0, 15)].astype(np.int8)     # recolor what the agent perceives
        except Exception:
            g = last
        last = g; mx = max(mx, int(obs.levels_completed or 0))
        if mx >= 1:
            return True
        tok = pol.decide(g, gstate_terminal=(obs.state == GameState.GAME_OVER),
                         levels=int(obs.levels_completed or 0), available=list(obs.available_actions or []))
        try:
            if tok[0] == "reset": obs = env.reset()
            elif tok[0] == "S": obs = env.step(GameAction.from_id(tok[1]))
            else: obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
        except Exception:
            obs = env.reset()
    return False


def test_generic_search_palette_invariant():
    assert _solve_permuted("vc33"), "generic novelty search must solve under a permuted palette"


def test_peg_class_palette_invariant():
    assert _solve_permuted("lf52"), "peg-solitaire class must solve under a permuted palette"
