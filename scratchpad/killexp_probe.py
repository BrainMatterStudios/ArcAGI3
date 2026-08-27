"""Task 1: re-verify reset semantics per game, with and without ONLY_RESET_LEVELS.

Black-box where possible (env.reset()/env.step()); white-box only to *position* the game
at a deep level (set_level + _score) so we can ask "does a mid-level reset preserve
levels_completed?" without having to solve the game first.
"""
import os, sys, json
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

GAMES = sys.argv[1:] or ["cd82", "ft09", "ls20", "ka59", "sb26", "sk48", "tu93", "lf52"]


def mk(game):
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    e = next(x for x in c.get_environments() if x.game_id.startswith(game))
    return c.make(game_id=e.game_id, scorecard_id="probe")


def lv(o):
    return int(getattr(o, "levels_completed", 0) or 0)


def first_action(o):
    av = list(o.available_actions or [])
    ids = [a if isinstance(a, int) else getattr(a, "value", a) for a in av]
    ids = [i for i in ids if isinstance(i, int) and 1 <= i <= 5]
    return ids[0] if ids else 1


out = []
for game in GAMES:
    env = mk(game)
    g = env._game
    nlev = len(g._levels)
    o = env.reset()
    row = {"game": game, "n_levels": nlev}

    # --- baseline: fresh reset ---
    row["fresh_levels"] = lv(o)
    row["fresh_full_reset"] = bool(getattr(o, "full_reset", None))

    # --- position deep: pretend levels 0..k-1 were completed ---
    k = min(3, nlev - 1)
    g._score = k
    g.set_level(k)          # action_count := 0
    o = env.step(GameAction.from_id(first_action(o)))   # action_count -> 1
    row["deep_before_levels"] = lv(o)
    row["deep_before_index"] = g._current_level_index

    # --- single reset with action_count>0 : LEVEL reset expected ---
    o1 = env.reset()
    row["r1_levels"] = lv(o1)
    row["r1_index"] = g._current_level_index
    row["r1_full_reset"] = bool(getattr(o1, "full_reset", None))
    row["r1_action_count"] = g._action_count

    # --- second reset with action_count==0 : FULL reset expected ---
    o2 = env.reset()
    row["r2_levels"] = lv(o2)
    row["r2_index"] = g._current_level_index
    row["r2_full_reset"] = bool(getattr(o2, "full_reset", None))

    row["preserves"] = (row["r1_levels"] == k and row["r1_index"] == k)
    row["double_wipes"] = (row["r2_levels"] == 0 and row["r2_index"] == 0)
    out.append(row)
    print(json.dumps(row), flush=True)

print("\nONLY_RESET_LEVELS=" + repr(os.getenv("ONLY_RESET_LEVELS")))
print(f"{'game':>6} {'nlv':>3} | mid-reset preserves | double-reset wipes to L0")
for r in out:
    print(f"{r['game']:>6} {r['n_levels']:>3} | {str(r['preserves']):>19} | {str(r['double_wipes'])}"
          f"   (r1 lv={r['r1_levels']} idx={r['r1_index']} full={r['r1_full_reset']};"
          f" r2 lv={r['r2_levels']} idx={r['r2_index']} full={r['r2_full_reset']})")
