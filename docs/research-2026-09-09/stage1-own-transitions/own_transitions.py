"""Stage-1: build Stage-0-format transitions from OUR OWN agent's wave events.

Input: a rig wave's artifacts/<stem>_events.jsonl (rows carry the 64x64 `board`,
`action_name`, `action_display` = MOUSE(row=R, col=C) for clicks, and the flags).
Output: <name>_transitions.json in the exact schema stage0.py/backtest.py expect
(entry_grid + [{index, action, x, y, grid, level_up, dead, win, state, level_after}]),
for ONE level segment of one run, capped at --n transitions.

Convention matched to the HF extractor: action = int (ACTION1..6 -> 1..6, RESET -> 0),
x = column, y = row; only clicks carry x/y.
"""
import argparse, json, os, re, sys

CLICK = re.compile(r"MOUSE\(row=(\d+),\s*col=(\d+)\)")


def build(events_path, level, n, out_path, label):
    rows = [json.loads(l) for l in open(events_path)]
    seq = [r for r in rows if r.get("type") in ("initial", "action")]
    idx = [i for i, r in enumerate(seq) if r.get("type") == "action" and r.get("level") == level]
    if not idx:
        sys.exit(f"no action rows at level {level} in {events_path}")
    # `level` on a row is the level AFTER the action, so the FIRST row at level L is the
    # action that CLEARED level L-1: its board is the entry board of level L and it is not
    # part of level L's own play. Drop it (except at level 1, whose entry is the initial board).
    first = idx[0]
    if level > 1 and seq[first].get("level_completed"):
        entry = seq[first]["board"]
        idx = idx[1:]
    else:
        entry = seq[first - 1]["board"]
    if not idx:
        sys.exit(f"level {level} has no play actions after its entry in {events_path}")
    trans = []
    for k, i in enumerate(idx[:n]):
        r = seq[i]
        name = str(r.get("action_name") or "")
        act = 0 if name == "RESET" else int(name.replace("ACTION", "") or 0)
        x = y = None
        m = CLICK.search(str(r.get("action_display") or ""))
        if m:
            y, x = int(m.group(1)), int(m.group(2))    # row, col -> y, x
        trans.append({"index": k, "action": act, "x": x, "y": y, "grid": r["board"],
                      "level_up": bool(r.get("level_completed")), "dead": bool(r.get("game_over")),
                      "win": bool(r.get("run_complete")), "state": r.get("state"),
                      "level_after": r.get("level")})
    rec = {"game": label, "game_id": os.path.basename(events_path).split("_events")[0],
           "model": "Qwen/Qwen3.8-Flash-Next-NVFP4 (our own agent)", "trace_dir": events_path,
           "entry_grid": entry, "transitions": trans}
    json.dump(rec, open(out_path, "w"))
    nclick = sum(1 for t in trans if t["x"] is not None)
    print(f"{label}: {len(trans)} transitions from level {level} (of {len(idx)} available), "
          f"clicks={nclick}, level_up={sum(t['level_up'] for t in trans)}, deaths={sum(t['dead'] for t in trans)} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("events"); ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.events, a.level, a.n, a.out, a.label)
