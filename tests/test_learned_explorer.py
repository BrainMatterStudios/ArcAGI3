"""Firewall + T4 fail-safe tests for LearnedExplorer."""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcagi3 import perception as P
from arcagi3.transfer_explorer import TransferExplorer
from arcagi3.learned_explorer import LearnedExplorer, Learner

GAMES_DIR = "src/arcagi3/games"
CFG = dict(seed=0, trust_threshold=3, border_mask=2)


def _drive(pol, game_id, steps):
    client = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAMES_DIR)
    env = client.make(game_id=game_id, scorecard_id=f"sc-{game_id}")
    obs = env.reset(); toks = []
    for _ in range(steps):
        if obs.state == GameState.WIN:
            break
        tok = pol.decide(P.to_grid(obs.frame),
                         gstate_terminal=(obs.state == GameState.GAME_OVER),
                         gstate_notplayed=(obs.state == GameState.NOT_PLAYED),
                         levels=int(obs.levels_completed or 0),
                         available=list(obs.available_actions or []))
        toks.append(tok)
        if tok[0] == "reset":
            obs = env.reset()
        elif tok[0] == "S":
            obs = env.step(GameAction.from_id(tok[1]))
        else:
            obs = env.step(GameAction.ACTION6, data={"x": int(tok[1]), "y": int(tok[2])})
    return toks


class _Abstain(Learner):
    pass   # always abstains (act -> None)


class _Forcer(Learner):
    def act(self, grid, node, key):
        return ("S", 1)   # always proposes "up"


def test_learn_off_byte_identical():
    base = _drive(TransferExplorer(**CFG), "push", 400)
    off = _drive(LearnedExplorer(enable_learn=False, learner=_Forcer(), **CFG), "push", 400)
    assert off == base and len(base) > 50


def test_no_gpu_falls_back_byte_identical():
    """Fail-safe: enable_learn + a learner but NO usable CUDA (this host) -> byte-identical to banked."""
    base = _drive(TransferExplorer(**CFG), "push", 400)
    nogpu = _drive(LearnedExplorer(enable_learn=True, learner=_Forcer(), require_gpu=True, **CFG), "push", 400)
    assert nogpu == base


def test_gpu_check_is_false_without_working_cuda():
    pol = LearnedExplorer(enable_learn=True, learner=_Abstain(), require_gpu=True, **CFG)
    assert pol._gpu_ok() is False        # no CUDA op succeeds on this host
    assert pol._active() is False


def test_abstaining_learner_byte_identical_even_when_active():
    """With the GPU gate bypassed, an abstaining learner still yields the banked trace (coverage safe)."""
    base = _drive(TransferExplorer(**CFG), "push", 400)
    act = _drive(LearnedExplorer(enable_learn=True, learner=_Abstain(), require_gpu=False, **CFG), "push", 400)
    assert act == base


def test_active_learner_action_is_used():
    pol = LearnedExplorer(enable_learn=True, learner=_Forcer(), require_gpu=False, **CFG)
    toks = _drive(pol, "navg", 60)
    assert ("S", 1) in toks              # the learner's forced action appears in the trace
