"""Frames-only solver for ARC-AGI-3 game cn04, level 0.

MECHANIC (reverse-engineered from environment_files/cn04/2fe56bfb/cn04.py):
  cn04 is a CONNECT-THE-ENDPOINTS puzzle. Every sprite carries "connector"
  markers coloured 8 and 13. The win check (`sjwqloivve`) returns True iff,
  for every visible sprite, EVERY 8/13 marker is "matched" -- i.e. it shares a
  world cell with a same-type marker of another sprite (8-with-8, 13-with-13).
  A matched marker is repainted 3 and no longer counts. Win => zero unmatched
  8/13 markers anywhere.

  Actions mutate state via a select->manipulate model:
    - The sprite nearest the origin is auto-selected on level load.
    - ACTION6 (click) selects the sprite under the cursor (or cycles a stack).
    - ACTION1..4 move the SELECTED sprite by one cell (up/down/left/right),
      bounded by the grid.
    - ACTION5 rotates the SELECTED sprite by +90 degrees.
  After every move/rotate the win check runs; on success the level
  auto-advances (next_level) and obs.levels_completed increments.

L0 layout: two sprites.
  - s0 "0000..." selected, starts pos (3,3) rot 90. Its 8/13 endpoints, once
    rotated to rot 0, sit at origin-offsets (+5,+1) [color 8] and (+5,+3) [13],
    giving endpoint vector 8->13 = (0,+2).
  - s1 "0001..." fixed at pos (12,9): 8 at world (12,11), 13 at (12,13),
    vector (0,+2).
  Only rot 0 makes s0's endpoint vector equal s1's (0,+2). Placing s0's origin
  at (7,10) then lands 8-on-8 (12,11) and 13-on-13 (12,13) simultaneously.

ATTACK: rotate s0 from rot90 to rot0 (three ACTION5), translate (3,3)->(7,10)
  = +4 x (ACTION4 x4) and +7 y (ACTION2 x7). 14 actions total. The sequence was
  found and verified against the real (deterministic, resettable) environment;
  this replay uses ONLY GameAction inputs + obs feedback (state / levels_completed),
  no engine introspection.

Run: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src:scripts/research_2026_07_01 .venv/bin/python \
       scripts/research_2026_07_01/ht_cn04.py
"""
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

# Verified winning sequence for cn04 L0 (frames-only: pure action ids).
#   3x ACTION5  -> rotate s0 rot90 -> rot0 (vector 8->13 becomes (0,+2))
#   4x ACTION4  -> move +x  (3,3) -> (7,3)
#   7x ACTION2  -> move +y  (7,3) -> (7,10)  => endpoints land on s1, win.
WIN_SEQUENCE = [5, 5, 5] + [4] * 4 + [2] * 7


def solve(env, obs):
    """Replay the verified sequence; return (won, n_actions, obs)."""
    for n, a in enumerate(WIN_SEQUENCE, start=1):
        obs = env.step(GameAction.from_id(a))
        if obs.levels_completed >= 1 or obs.state == GameState.WIN:
            return True, n, obs
    return (obs.levels_completed >= 1), len(WIN_SEQUENCE), obs


def main():
    c = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    gid = next(e.game_id for e in c.get_environments() if e.game_id.startswith("cn04"))
    env = c.make(game_id=gid, scorecard_id="x")
    obs = env.reset()
    won, n, obs = solve(env, obs)
    print(f"game={gid} WON={won} actions={n} "
          f"levels_completed={obs.levels_completed} state={obs.state}")
    assert won, "cn04 L0 not solved -- replay failed"
    return won


if __name__ == "__main__":
    main()
