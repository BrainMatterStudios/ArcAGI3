"""ORACLE-MODEL PROBE — can the LLM USE a correct world model it did not build?

THE QUESTION THIS SETTLES (and why it is worth more than any other experiment
on the board right now).

Measured, from 252 recorded plays and 36 full Qwen3.8 transcripts:

  * comprehension, not budget, is the constraint (x1.40 actions -> 25% FEWER
    levels; zero-level games already spend 0.93x the human level-1 baseline);
  * the agent PERCEIVES correctly (exact coordinates every turn) and forms
    EXCELLENT hypotheses (16 sharp, mostly-correct mechanistic claims on sk48
    alone) — then loses them, re-derives them, retracts true ones, and ends in
    a byte-identical period-3 limit cycle;
  * every channel we have ever advertised for persisting those findings is
    ignored: cross-level notes 0/1482 turns, run_probe 1/28 games, plan_queue
    0/20 runs, compaction 0/303, world_model.py edits 0/209 and 0/140.
    LAW: advertised = performative; only structural enforcement is adopted.

So the general-agent programme rests on one unexamined assumption: that the
model would play well IF it had a correct, persistent model of the game. That
assumption has never been tested. Every lever we have built — notes stores,
latch repairs, digests, plan channels, compaction — tries to help the model
BUILD and KEEP a world model. None of them asked whether it can USE one.

This probe removes construction and persistence entirely. It hands the model a
hand-written, verified-correct world model, the goal predicate, and the exact
current state, every single turn, and asks only for the next action.

  * >= 50% of clones clear the level  -> the model can act on a model it did
    not have to build. The missing function is CONSTRUCTION/PERSISTENCE, which
    trained weights (TTT) or structural memory can supply. The general-agent
    programme is alive and has a direction.
  * < 50%                             -> the model cannot execute a correct
    model even when handed one, with nothing to discover and nothing to
    remember. Then notes, digests, matchers, compaction and behaviour
    fine-tunes are ALL dead by construction, and the only remaining route is
    code choosing the actions. This must never be proposed again.

WHY wa30 LEVEL 3 IS THE RIGHT TEST BED. A negative result is only meaningful if
the task is genuinely doable, the oracle is genuinely correct, and the budget is
genuinely sufficient. All three are established here and nowhere else:

  * the mechanics were derived and corrected three times on 2026-08-28 and the
    final account is engine-verified (a sprite-by-sprite trace);
  * `wa30_carriersim.py` reproduces the carrier drive EXACTLY in lockstep with
    the engine on L1-L4 (every carrier position, block position, carry
    relation, every tick);
  * an 82-move solution EXISTS and is engine-verified on a clean replay,
    against the level's own 100-move budget. So "no plan fits" is excluded.

TWO ARMS, because a bare failure would be uninterpretable:
  ORACLE  — board + verified mechanics + goal, every turn
  CONTROL — board + legal actions only (what the shipped agent effectively has)
The contrast is the measurement; the ORACLE number alone is not.

Run:  ONLY_RESET_LEVELS=true .venv/bin/python \
          submission/_oracle_probe/oracle_probe.py [clones] [arm] [model]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

os.environ.setdefault("ONLY_RESET_LEVELS", "true")

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
for _d in (os.path.join(ROOT, "submission/_search_core"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("ORACLE_PROBE_MODEL", "qwen3:8b")

# The verified mechanics. Every claim here is engine-checked; see
# docs/RESEARCH-2026-08-28-the-screen-false-negative.md 4b/4c.
ORACLE_WA30 = """\
GAME MECHANICS (these are COMPLETE and VERIFIED — treat them as ground truth):

1. The board is a 16x16 grid of cells. Cell coordinates below are given as
   (col,row) in cell units, 0..15.
2. You control ONE avatar. Actions 1/2/3/4 move it UP/DOWN/LEFT/RIGHT by one
   cell. Action 5 grabs the block you are FACING, or releases the one you hold.
   You face the direction you last moved.
3. While holding a block, moving drags the block with you.
4. A vertical WALL COLUMN divides the board. You can NEVER walk into or across
   that column. Your reachable region is strictly to its left.
5. Every goal pad is on the FAR side of that column. THEREFORE YOU CAN NEVER
   PUT A BLOCK ON A PAD YOURSELF. Do not try. Any plan that walks you to a pad
   is impossible.
6. BLOCKS, unlike you, CAN be pushed INTO the wall column. That is the entire
   mechanic: you push a block into a free cell of the divider and let go.
7. Autonomous CARRIER agents live on the far side. They move one cell per
   action you take. Each one walks to the nearest unclaimed block it can
   reach, picks it up, carries it to a free pad, and drops it. They can reach
   blocks parked in the divider column. They cannot reach blocks on your side.
8. THE LEVEL IS WON when every block sits on a pad.

THEREFORE THE ONLY WINNING STRATEGY IS: for each block on your side, walk to
it, grab it, drag it to a free cell of the divider column, release it, and let
a carrier ferry it to a pad. Waiting (moving into a wall) is not wasted — the
carriers still advance one cell per action you take.
"""

CONTROL_WA30 = """\
You are playing an unfamiliar grid game. Actions 1/2/3/4 move your avatar
UP/DOWN/LEFT/RIGHT by one cell; action 5 interacts with whatever you face.
Work out the rules yourself from what you observe.
"""

# THIRD ARM — added after the first pilot, to decompose an ambiguous negative.
#
# oracle 0/5 and control 0/5 with IDENTICAL end states told us the mechanics
# text bought nothing, but not WHY. Three very different failures produce that
# same result and they have opposite implications:
#   comprehension — it cannot turn the rules into a goal
#   planning      — it has the goal but cannot sequence a route to it
#   execution     — it has the route and still cannot emit the right keys
# This arm removes comprehension AND planning: every turn it is told, in plain
# language, which block to take and exactly where to put it. What remains is
# pure execution. If it fails HERE, the deficit is not knowledge of the game,
# and that conclusion is far less sensitive to model size than the oracle arm's.
GUIDED_WA30 = ORACLE_WA30 + """
YOU DO NOT NEED TO PLAN. Each turn you are told exactly which block to move and
exactly which cell to push it into. Just walk to the block, face it, press 5 to
grab, drag it to the named cell, and press 5 to release.
"""


def next_goal_text(env, M) -> str:
    """A plain-language instruction naming the block and the divider cell.

    Derived from the same board model the planner uses, so it is correct by the
    same construction the 82-move solution was verified under."""
    g = M.game_of(env)
    per = M.engine_percept(g, env.observation_space)
    if per is None:
        return ""
    avatar, blocks, free_pads, walls, occupied = per
    gaps = M.handoff_cells(walls, occupied, blocks)
    if not blocks or not gaps:
        return ""
    # the block that still needs the avatar = the one not already in the divider
    from collections import Counter
    cols = Counter(x for (x, _y) in walls if 0 <= x < 64)
    col = cols.most_common(1)[0][0] if cols else None
    todo = [b for b in blocks if b[0] != col] or list(blocks)
    b = min(todo, key=lambda p: abs(p[0] - avatar[0]) + abs(p[1] - avatar[1]))
    gap = min(gaps, key=lambda p: abs(p[1] - b[1]))
    c = lambda p: (p[0] // 4, p[1] // 4)  # noqa: E731
    return (f"\n\nYOUR TASK RIGHT NOW: take the block at {c(b)} and push it "
            f"into the divider cell at {c(gap)}. You are at {c(avatar)}.")


def render(per, carriers) -> str:
    """The board as a text grid, in the same cell units the mechanics use."""
    avatar, blocks, free_pads, walls, occupied = per
    bl, fp, oc = set(blocks), set(free_pads), set(occupied)
    car = set(carriers)
    out = ["    " + "".join(f"{x // 4 % 10}" for x in range(0, 64, 4))]
    for y in range(0, 64, 4):
        row = ""
        for x in range(0, 64, 4):
            c = (x, y)
            row += ("A" if c == avatar else "B" if c in bl else
                    "C" if c in car else "o" if c in oc else
                    "." if c in fp else "#" if c in walls else " ")
        out.append(f"{y // 4:3} {row}")
    legend = ("A=you  B=block(needs a pad)  C=carrier  o=pad WITH a block on it"
              "  .=free pad  #=wall  (blank)=floor")
    return "\n".join(out) + "\n" + legend


def state_text(env, M) -> str:
    g = M.game_of(env)
    per = M.engine_percept(g, env.observation_space)
    if per is None:
        return "board unavailable"
    carriers = [(s.x, s.y) for s in
                g.current_level.get_sprites_by_tag("kdweefinfi")]
    avatar, blocks, free_pads, walls, occupied = per
    cell = lambda p: (p[0] // 4, p[1] // 4)  # noqa: E731
    return (render(per, carriers) + "\n\n"
            + f"You are at {cell(avatar)}.\n"
            + f"Blocks not yet on a pad: {[cell(b) for b in blocks]}\n"
            + f"Free pads: {[cell(p) for p in free_pads]}\n"
            + f"Carriers: {[cell(c) for c in carriers]}\n"
            + f"Blocks already delivered: {len(occupied)}")


def ask(messages, model, temperature) -> str:
    # think=False matters: qwen3 defaults to emitting a long private reasoning
    # block, which on a local 8B costs minutes per turn and blew the first
    # smoke's timeout. The probe measures whether the model can ACT on a given
    # model, not how long it can deliberate.
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "think": False,
                       "options": {"temperature": temperature,
                                   "num_ctx": 8192,
                                   "num_predict": 400}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return (d.get("message") or {}).get("content", "") or ""


ACT_RE = re.compile(r"\bACTIONS?\s*[:=]?\s*([1-5](?:\s*[, ]\s*[1-5])*)",
                    re.IGNORECASE)


def parse_actions(text: str, cap: int = 8) -> list[int]:
    """Take the LAST explicit ACTIONS: line — models often think aloud first."""
    hits = ACT_RE.findall(text)
    if not hits:
        return []
    nums = [int(n) for n in re.findall(r"[1-5]", hits[-1])]
    return nums[:cap]


def one_clone(clone: int, arm: str, model: str, max_moves: int,
              verbose: bool) -> dict:
    import logging

    logging.disable(logging.CRITICAL)
    import search_core as sc
    import specialists as sp  # noqa: F401
    import wa30_macro as M
    from arc_agi import Arcade, OperationMode
    from step_budgets import budgets_for

    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(ROOT, "environment_files"))
    probe = arc.make("wa30")
    probe.reset()
    buds = budgets_for(M.game_of(probe))
    env = arc.make("wa30")
    env.reset()
    core = sc.SearchCore(env, backend="snapshot", max_states=20000)
    core.warmup_and_freeze()
    M.chain(core, buds, 2, 300.0)          # reach level 3 exactly as we do live
    live = core.backend.env
    lc0 = live.observation_space.levels_completed
    budget = buds[2]

    system = {"oracle": ORACLE_WA30, "control": CONTROL_WA30,
              "guided": GUIDED_WA30}.get(arm, CONTROL_WA30) + (
        "\n\nEach turn, reply with a short justification and then a final line "
        "of the exact form:\nACTIONS: n, n, n\nusing 1-8 action numbers from "
        "1..5. Nothing after that line.")
    moves, turns = 0, 0
    history: list[str] = []
    t0 = time.time()
    while moves < min(budget, max_moves) and turns < 40:
        turns += 1
        msg = [{"role": "system", "content": system},
               {"role": "user", "content":
                state_text(live, M)
                + (next_goal_text(live, M) if arm == "guided" else "")
                + (f"\n\nRecent moves you made: {history[-6:]}" if history else "")
                + f"\n\nMoves used: {moves}/{budget}. Give your next actions."}]
        try:
            reply = ask(msg, model, 0.2 + 0.05 * (clone % 8))
        except Exception as exc:  # noqa: BLE001
            return {"clone": clone, "arm": arm, "error": str(exc)[:120],
                    "moves": moves, "won": False}
        acts = parse_actions(reply)
        if not acts:
            history.append("(no parsable action)")
            continue
        verdict, done = M.run_actions(live, acts, lc0)
        moves += done
        history.append(",".join(map(str, acts[:done])))
        if verbose:
            print(f"    c{clone} turn{turns}: {acts[:done]} -> {verdict} "
                  f"({moves}/{budget})", flush=True)
        if verdict == "win":
            return {"clone": clone, "arm": arm, "won": True, "moves": moves,
                    "turns": turns, "wall": round(time.time() - t0, 1)}
        if verdict == "dead":
            break
    g = M.game_of(live)
    un, oc = M.status(g)
    return {"clone": clone, "arm": arm, "won": False, "moves": moves,
            "turns": turns, "blocks_left": len(un), "delivered": len(oc),
            "wall": round(time.time() - t0, 1)}


def main() -> int:
    clones = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    arms = ([sys.argv[2]] if len(sys.argv) > 2
            else ["oracle", "control"])
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL
    max_moves = int(os.environ.get("ORACLE_PROBE_MAX_MOVES", "100"))
    print(f"ORACLE-MODEL PROBE — wa30 L3 (82-move solution exists, budget 100)")
    print(f"model={model}  clones={clones}  arms={arms}\n")
    results = []
    for arm in arms:
        print(f"--- arm: {arm} ---", flush=True)
        for c in range(clones):
            r = one_clone(c, arm, model, max_moves, verbose=True)
            results.append(r)
            print(f"  clone {c}: won={r.get('won')} moves={r.get('moves')} "
                  f"turns={r.get('turns')} left={r.get('blocks_left')}"
                  + (f" ERR {r['error']}" if r.get("error") else ""), flush=True)
    print("\n" + "=" * 66)
    # ERRORED CLONES ARE NOT FAILURES. The first smoke timed out on the LLM
    # call and the probe printed "FAIL — it cannot execute a correct model",
    # which would have been a fabricated verdict from a broken instrument.
    # A clone only counts if the model actually got to play.
    for arm in arms:
        a = [r for r in results if r["arm"] == arm]
        ok = [r for r in a if not r.get("error")]
        w = sum(1 for r in ok if r.get("won"))
        print(f"{arm:8} cleared {w}/{len(ok)} valid "
              f"({len(a) - len(ok)} errored, excluded)"
              + (f" = {100.0 * w / len(ok):.0f}%" if ok else ""))
    o = [r for r in results if r["arm"] == "oracle" and not r.get("error")]
    if not o:
        print("\nNO VALID ORACLE CLONES — instrument failure, NO VERDICT. "
              "Fix the harness before reading anything into this run.")
    else:
        rate = sum(1 for r in o if r.get("won")) / len(o)
        print(f"\nPRE-REGISTERED BAR: oracle >= 50% of VALID clones "
              f"(n={len(o)}).  RESULT: "
              + ("PASS — the model CAN act on a model it did not build"
                 if rate >= 0.5 else
                 "FAIL — it cannot execute a correct handed-to-it model"))
        if len(o) < 5:
            print(f"  CAVEAT: n={len(o)} is under-powered; treat as a pilot, "
                  "not the pre-registered result.")
    # Name the file by the arms it holds: a bare results.json meant the
    # `guided` run silently overwrote the oracle/control run's data.
    out = os.path.join(_HERE, f"results_{'_'.join(arms)}.json")
    with open(out, "w") as fh:
        json.dump({"model": model, "clones": clones, "results": results}, fh,
                  indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
