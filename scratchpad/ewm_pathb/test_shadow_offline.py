"""Offline validation of shadow_replay on REAL recorded duck history (no GPU, no model):
1. determinism — replay the duck's full recorded tu93 history into a fresh OFFLINE game, reproduce the win
2. extract the winning-attempt segment
3. replay ONLY the segment into a fresh OFFLINE game, re-win with far fewer actions
4. confirm the segment play's RHAE >> the full/original play's (what max-over-plays would bank)
"""
import json, sys
from pathlib import Path

TAAF = "/Users/ahmed/Documents/ArcAGI3/submission/_adopt/taaf-src/src/tufa-arc-agi-framework/src"
HERE = "/Users/ahmed/Documents/ArcAGI3/scratchpad/ewm_pathb"
for p in (TAAF, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import arc_agi
from taaf.game_api import GameAPI, ArcadeSpec
from shadow_replay import extract_winning_segments, replay_segments

ENV = "/Users/ahmed/Documents/ArcAGI3/environment_files"
BJ = "/Users/ahmed/Documents/ArcAGI3/submission/_adopt/taaf-src/src/ARC3-Inference/runs/20260703_090517/benchmark.json"
BASE_L1 = 19  # tu93 metadata baseline_actions[0]

bm = json.load(open(BJ))
run = next(r for r in bm["game_runs"] if r["game_id"].startswith("tu93") and r["levels_completed"] >= 1)
hist = run["history"]; apl = run["actions_per_level"]; lc = run["levels_completed"]
print(f"recorded duck run: tu93 levels={lc} total_actions={len(hist)} apl={apl}")


def fresh():
    spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=ENV)
    g = GameAPI(env_name="tu93", arcade_spec=spec)
    g.start_game()
    return g


def rhae(actions):
    return min(115.0, (BASE_L1 / max(actions, 1)) ** 2 * 100)

# 1. determinism: replay full recorded history
g1 = fresh()
import arcengine
for rec in hist:
    a = rec["action"]
    if g1.game_run.state != "playing":
        break
    try:
        g1.execute_action(arcengine.ActionInput(id=arcengine.GameAction[a["id"]], data=dict(a.get("data") or {})))
    except Exception as e:
        print("  (full replay stopped:", e, ")"); break
full_lc = g1.current_state.levels_completed
print(f"1. FULL-history replay -> levels_completed={full_lc}  (determinism {'OK' if full_lc>=1 else 'FAILED'}); orig RHAE(L1 @ {apl[0]} actions) = {rhae(apl[0]):.2f}")

# 2. extract winning segment
segs = extract_winning_segments(hist, apl, lc)
seg_len = sum(len(w) for _, w in segs)
print(f"2. extracted {len(segs)} winning segment(s), total {seg_len} actions (vs {len(hist)} recorded)")

# 3. replay only the segment into a fresh game
g2 = fresh()
res = replay_segments(g2, segs)
print(f"3. SEGMENT replay -> levels={res['levels']} actions={res['actions']} desynced_at={res['desynced_at']}")

# 4. compare
if res["levels"] >= 1 and res["desynced_at"] is None:
    print(f"4. banked RHAE(L1 @ {res['actions']} actions) = {rhae(res['actions']):.2f}  vs orig {rhae(apl[0]):.2f}  ->  {rhae(res['actions'])/max(rhae(apl[0]),1e-9):.1f}x")
    print("SHADOW-REPLAY VALIDATED OFFLINE ON REAL DUCK DATA" if rhae(res["actions"]) > rhae(apl[0]) else "no uplift")
else:
    print("SEGMENT replay did not re-win (desync) — the per-level guard would abandon this game; duck score stands.")
