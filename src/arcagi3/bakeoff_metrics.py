"""Bake-off scoring: the competition's per-level efficiency + a transfer-slope summary.

efficiency mirrors the verified scoring formula: per level min(1.15, (A_h / A_m)^2),
uncompleted level = 0. A_h is the human-baseline proxy (Exp-42 BFS-optimal length); A_m is
the agent's environment-altering action count for that level.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUEMODEL = _REPO_ROOT / "scripts" / "truemodel_planner.py"

# Registry: game_id -> (script_path, loader_fn_name, key_fn_name|None, moves_name|None)
# key_fn_name=None  => use truemodel_planner.state_key (ls20 default)
# moves_name=None   => use truemodel_planner.MOVES (directional 1-4, the ls20 default)
_GRADERS: dict[str, tuple[str, str, str | None, str | None]] = {
    "ls20": ("scripts/truemodel_planner.py", "load_ls20_class", None, None),
    "sk48": ("scripts/truemodel_sk48.py", "load_sk48_class", "sk48_key", "SK48_MOVES"),
    "collect": ("scripts/truemodel_collect.py", "load_collect_class", "collect_key", None),
    "tr87": ("scripts/truemodel_tr87.py", "load_tr87_class", "tr87_key", "TR87_MOVES"),
}

# Cache: game_id -> list of per-level A_h values
_PER_LEVEL_CACHE: dict[str, list] = {}


def _load(rel_path: str, fn_name: str):
    """importlib-load a script relative to repo root; return (module, callable)."""
    full_path = _REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(full_path.stem, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, getattr(mod, fn_name)


def efficiency(a_h: int, a_m: int | None) -> float:
    if a_m is None or a_m <= 0:
        return 0.0
    return min(1.15, (a_h / a_m) ** 2)


def transfer_slope(per_level_actions: list[int]) -> float:
    """Least-squares slope of action-cost vs level index. Negative = costs drop with depth."""
    n = len(per_level_actions)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(per_level_actions) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, per_level_actions))
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else num / den


def human_baseline_actions(game: str = "ls20", level: int = 0, max_nodes: int = 500_000) -> int | None:
    """A_h proxy = BFS-optimal action count over a game's TRUE model (grader-only source access).

    Dispatches by `game` name; results are cached per-game.  Supported games:
      "ls20"    — the original (Exp-42); L0=13, L1=45.
      "sk48"    — snake/rope puzzle; directional moves only, step-budget state.
      "collect" — collect-all-items navigation; directional moves, 14×14 grid.

    Returns None when BFS cannot solve within max_nodes, or on any unexpected error.
    """
    if game not in _GRADERS:
        raise ValueError(f"Unknown game {game!r}. Registered: {list(_GRADERS)}")

    if game not in _PER_LEVEL_CACHE:
        tm_mod, _ = _load("scripts/truemodel_planner.py", "load_ls20_class")

        path, loader_name, key_name, moves_name = _GRADERS[game]
        g_mod, loader = _load(path, loader_name)

        key_fn = getattr(g_mod, key_name) if key_name else tm_mod.state_key
        moves  = getattr(g_mod, moves_name) if moves_name else tm_mod.MOVES

        _PER_LEVEL_CACHE[game] = tm_mod.optimal_actions_per_level(
            loader(),
            up_to_level=max(level, 4),
            max_nodes=max_nodes,
            key_fn=key_fn,
            moves=moves,
        )

    per = _PER_LEVEL_CACHE[game]
    return per[level] if level < len(per) else None
