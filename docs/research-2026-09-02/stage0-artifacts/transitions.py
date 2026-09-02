"""Extract LEVEL-0 transition sequences from frontier events.jsonl traces.

Output per game: <out>/<game>_transitions.json with
  {game, game_id, model, trace_dir, entry_grid, transitions:[{index, action, x, y, grid, level_up, dead, win, state}]}
The level-0 sequence = every action_taken event from step 0 up to and including the
first level_up. Records after that belong to level 1.
"""
import json, os, sys

TRACES = "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/search/protocol/hf_traces/claude_fable_opus"
GAMES = {
    "sk48": "claude-fable-5_max_sk48_100.0",
    "tn36": "claude-fable-5_max_tn36_94.74",
    "cn04": "claude-opus-4-8_max_cn04_100.0",
}
OUT = os.path.dirname(os.path.abspath(__file__))


def extract(game):
    d = os.path.join(TRACES, GAMES[game])
    run = json.load(open(os.path.join(d, "run.json")))
    entry = None
    trans = []
    with open(os.path.join(d, "events.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            k = e["kind"]
            if k == "turn_started" and entry is None:
                assert e["level"] == 0 and e["env_step"] == 0, e["seq"]
                entry = e["grid"]
            elif k == "action_taken":
                assert entry is not None
                assert e["step_index"] == len(trans), (e["step_index"], len(trans))
                trans.append({
                    "index": e["step_index"], "action": e["action"], "x": e["x"], "y": e["y"],
                    "grid": e["grid"], "level_up": bool(e["level_up"]), "dead": bool(e["dead"]),
                    "win": bool(e["win"]), "state": e["state"], "level_after": e["level"],
                })
                if e["level_up"]:
                    break
    # sanity: every grid is 64x64 ints 0..15
    for t in trans:
        g = t["grid"]
        assert len(g) == 64 and all(len(r) == 64 for r in g)
        assert all(0 <= v <= 15 for r in g for v in r)
    assert len(entry) == 64
    assert trans[-1]["level_up"], "level 0 never cleared in trace"
    return {"game": game, "game_id": run["game_id"], "model": run["model"], "trace_dir": d,
            "entry_grid": entry, "transitions": trans}


if __name__ == "__main__":
    for game in (sys.argv[1:] or list(GAMES)):
        rec = extract(game)
        p = os.path.join(OUT, f"{game}_transitions.json")
        json.dump(rec, open(p, "w"))
        n = len(rec["transitions"])
        resets = sum(1 for t in rec["transitions"] if t["action"] == 0)
        clicks = sum(1 for t in rec["transitions"] if t["action"] == 6)
        print(f"{game}: {n} level-0 transitions (resets={resets}, clicks={clicks}, "
              f"level_up at index {rec['transitions'][-1]['index']}) -> {p}")
