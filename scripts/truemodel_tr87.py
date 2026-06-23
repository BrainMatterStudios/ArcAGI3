"""True-model helpers for tr87 — used by bakeoff_metrics to compute A_h.

tr87 mechanics (reverse-engineered from environment_files/tr87/cd924810/tr87.py):
  Class       : Tr87 (subclass of ARCBaseGame, game_id="tr87")
  Actions     : [1, 2, 3, 4]  — directional only, NO clicks.
                  ACTION3 / ACTION4 = move the edit cursor `qvtymdcqear_index` by -1 / +1
                                      (mod the number of editable groups).
                  ACTION1 / ACTION2 = cycle the VALUE of the currently-selected sprite(s)
                                      by -1 / +1 via `wpbnovjwkv`, which rewrites the
                                      trailing digit of the sprite's `.name` (mod kjgicbtgrt=7).
  Effective BFS moves: all four (TR87_MOVES). Cursor moves (3/4) are on the optimal
    path because an edit (1/2) only affects the group under the cursor.

  This is a cursor-edit / rule-induction puzzle.  Each step decrements a step budget
  `upmkivwyrxz` (starts at vfpimnmtnta = 128 for levels <=4, else 256); hitting 0 -> lose().
  After an ACTION1/2 edit the win predicate `bsqsshqpox()` is checked; on a win it sets
  `yfetxjexviz = 0`, kicking off a multi-step color-remap animation that — within the SAME
  perform_action() call (the engine loops step() until the action completes) — eventually
  calls `next_level()`, which does `_score += 1`.  So the winning move advances `_score`
  inside one perform_action, and the generic BFS harness detects it via `_score`/`level_index`.

  ── What actually determines the win ───────────────────────────────────────────────
  `bsqsshqpox()` (and its helper `iwbhnvdaao`) compare sprites ONLY by `sprite.name`.
  A sprite's editable "value" is exactly the trailing digit of its `.name` (1..7); position
  and rotation never change under editing.  The win compares the editable sequence against
  fixed target sequences (`self.zvojhrjxxm`, which is never edited) per the level's rules
  (`cifzvbcuwqe`), optionally transformed by double_translation / tree_translation.

  Two level variants:
    * non-alter levels edit `self.ztgmtnnufb`  (cursor indexes this list directly).
    * alter_rules levels edit the flattened groups
        [grp for rule in self.cifzvbcuwqe for grp in rule]
      (cursor indexes this flattened list of lists-of-sprites).

  So the MINIMAL latent state that (a) determines all future reachable states and
  (b) the win predicate reads is:

        (qvtymdcqear_index, <names of every editable sprite>)

  Everything else the win reads (zvojhrjxxm targets, rule structure cifzvbcuwqe wiring,
  positions, rotations) is immutable after on_set_level, and on_set_level uses fixed
  per-level seeds (random.Random(ofysoutulp[idx]) / ripmydnety[idx]), so each level is
  deterministic and reproduced identically by every fresh Tr87()+RESET (deepcopy in BFS
  preserves it).

  ── State key: (cursor_index, tuple-of-editable-sprite-names) ──────────────────────
  Collision-free w.r.t. the win predicate: the only mutable inputs to bsqsshqpox are the
  editable sprites' names (their trailing digits), and edits/win always act through the
  cursor index, so two states with identical (index, editable-names) are genuinely
  identical for all future play.  The step budget is deliberately EXCLUDED — BFS finds the
  shortest winning path before the budget matters, and including it would bloat the space;
  lose()/GAME_OVER still prunes any branch that exhausts the budget.

  Verified by replay: after BFS returns a solution we replay it on a fresh game via
  perform_action and assert `_score`/`level_index` actually advances.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from arcengine import GameAction

TR87_PATH = Path(__file__).resolve().parent.parent / "environment_files/tr87/cd924810/tr87.py"

# Directional moves only; tr87 exposes no clicks (available_actions=[1,2,3,4]).
TR87_MOVES = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
]


def load_tr87_class():
    """Load and return the Tr87 game class from the environment source."""
    spec = importlib.util.spec_from_file_location("tr87_env", TR87_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Tr87


def _editable_groups(g):
    """The list of editable sprite-groups the cursor indexes, per level variant.

    Mirrors tr87.step() exactly:
      * alter_rules : flattened [grp for rule in cifzvbcuwqe for grp in rule]
                      (each grp is a list[Sprite]; an edit cycles ALL sprites in the grp).
      * otherwise   : each editable group is a single sprite in self.ztgmtnnufb.
    Returns a list of lists-of-sprites so the key treats both variants uniformly.
    """
    if g.current_level.get_data("alter_rules"):
        return [grp for rule in g.cifzvbcuwqe for grp in rule]
    return [[s] for s in g.ztgmtnnufb]


def tr87_key(g) -> tuple:
    """Compact, collision-free BFS dedup key for tr87.

    Reads exactly the mutable inputs to the win predicate `bsqsshqpox`:
      - g.qvtymdcqear_index : the edit cursor (selects which group ACTION1/2 mutates).
      - the trailing-digit value of every editable sprite's `.name`, captured as the
        full `.name` strings (bsqsshqpox/iwbhnvdaao compare by name only).

    The step budget is intentionally NOT part of the key (see module docstring).
    """
    groups = _editable_groups(g)
    names = tuple(tuple(s.name for s in grp) for grp in groups)
    return (g.qvtymdcqear_index, names)


def main():
    import sys

    import truemodel_planner as tm

    max_nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
    Tr87 = load_tr87_class()

    from arcengine import ActionInput

    # --- determinism check: fresh instances reproduce the same initial state ---
    g1 = Tr87(); g1.perform_action(ActionInput(id=GameAction.RESET))
    g2 = Tr87(); g2.perform_action(ActionInput(id=GameAction.RESET))
    same_init = tr87_key(g1) == tr87_key(g2)
    print(f"=== tr87 determinism: fresh+RESET reproduce initial key: {same_init} ===", flush=True)
    print(f"  L0 alter_rules : {bool(g1.current_level.get_data('alter_rules'))}", flush=True)
    print(f"  L0 editable groups: {len(_editable_groups(g1))}", flush=True)
    print(f"  L0 initial key : {tr87_key(g1)}", flush=True)

    print(f"\n=== BFS over the true model (max_nodes={max_nodes:,}) ===", flush=True)
    per_level = tm.optimal_actions_per_level(
        Tr87, up_to_level=5, max_nodes=max_nodes, key_fn=tr87_key, moves=TR87_MOVES,
    )
    print(f"  per-level A_h : {per_level}", flush=True)

    # --- replay verification: re-derive each level's solution and assert it advances ---
    print("\n=== replay verification (most important check) ===", flush=True)
    g = Tr87(); g.perform_action(ActionInput(id=GameAction.RESET))
    all_pass = True
    for lvl, a_h in enumerate(per_level):
        if a_h is None:
            print(f"  L{lvl}: BFS could not solve within {max_nodes:,} nodes", flush=True)
            all_pass = False
            break
        sol = tm.bfs_solve_current(g, max_nodes=max_nodes, key_fn=tr87_key, moves=TR87_MOVES)
        if sol is None:
            print(f"  L{lvl}: replay re-solve FAILED (None)", flush=True)
            all_pass = False
            break
        score_before, idx_before = g._score, g.level_index
        for aid in sol:
            g.perform_action(ActionInput(id=GameAction.from_id(aid)))
        advanced = g._score > score_before or g.level_index > idx_before
        print(f"  L{lvl}: A_h={len(sol):>3}  advanced={advanced}  "
              f"(score {score_before}->{g._score}, idx {idx_before}->{g.level_index})", flush=True)
        all_pass = all_pass and advanced
    print(f"\n  REPLAY VERIFICATION: {'PASS' if all_pass else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
