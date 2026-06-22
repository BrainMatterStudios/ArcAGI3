"""collect-game A_h grader: load the Collect class and a state key over (agent pos, remaining items)."""
from __future__ import annotations
import importlib.util
from pathlib import Path

COLLECT_PATH = Path(__file__).resolve().parent.parent / "src/arcagi3/games/collect/collect.py"


def load_collect_class():
    spec = importlib.util.spec_from_file_location("collect_env", COLLECT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Collect


def collect_key(g):
    """State = agent position + the set of remaining items (frozenset is hashable)."""
    return (getattr(g, "_ax", None), getattr(g, "_ay", None), frozenset(getattr(g, "_items", set())))
