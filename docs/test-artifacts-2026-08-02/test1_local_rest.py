"""E1: Local competition-mode REST end-to-end test of the post-WIN clean-replay lever.

Boots the shipped arc_agi Flask app (create_app, competition_mode=True) exactly as the
eval gateway shape, with ONLY_RESET_LEVELS=true (the eval setting). Then over HTTP:

  play 1 (sloppy): K3's recorded 142-action sb26 win with 200 wasted background
                   clicks injected during level 8 (weight 8/36 -> drags the
                   weighted score below the 100 cap, so the two plays differ)
  RESET while state==WIN                       <- the carve-out under test
  assert second play opened with actions=0     (in-process scorecard + closed JSON)
  play 2 (clean): the 142-action trace only
  close scorecard, assert game score == max(play scores) == play-2 score
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import urllib.request

os.environ["ONLY_RESET_LEVELS"] = "true"  # must be set before arc_agi/arcengine import
logging.disable(logging.CRITICAL)

from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server
import socketserver

from arc_agi import Arcade, OperationMode
from arc_agi.server import create_app

REPO = "/Users/ahmed/Documents/ArcAGI3"
HERE = os.path.dirname(os.path.abspath(__file__))
API_KEY = "e1-test-key"
GAME = "sb26-7fbdac44"
N_WASTE = 200
L8_START = 125  # cumulative level_actions [9,33,15,15,17,19,17,17] -> level 8 begins at index 125

results: list[dict] = []


def check(name: str, ok: bool, detail):
    results.append({"assert": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


class _ThreadingWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _Quiet(WSGIRequestHandler):
    def log_message(self, *a):  # noqa: D102
        return


def http(base, method, path, body=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def act(base, a, guid, card_id):
    body = {"game_id": GAME, "guid": guid, "card_id": card_id}
    if a["name"] == "ACTION6":
        body["x"] = a["x"]
        body["y"] = a["y"]
    return http(base, "POST", f"/api/cmd/{a['name']}", body)


def snapshot_card(api, card_id):
    sc = api.arcade.scorecard_manager.get_scorecard(card_id, API_KEY)
    card = sc.cards.get(GAME)
    if card is None:
        return None
    return {
        "total_plays": card.total_plays,
        "actions": list(card.actions),
        "resets": list(card.resets),
        "states": [s.name for s in card.states],
        "levels_completed": list(card.levels_completed),
    }


def main():
    trace = json.load(open(os.path.join(HERE, "sb26_win_trace.json")))
    assert len(trace) == 142

    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir=os.path.join(REPO, "environment_files"))
    arcade.available_environments = [
        e for e in arcade.available_environments if e.game_id == GAME
    ]
    assert len(arcade.available_environments) == 1, arcade.available_environments
    app, api = create_app(arcade, competition_mode=True)
    server = make_server("127.0.0.1", 0, app,
                         server_class=_ThreadingWSGIServer, handler_class=_Quiet)
    base = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"server up at {base} (competition_mode=True, ONLY_RESET_LEVELS=true)")

    # 1. open competition scorecard
    st, out = http(base, "POST", "/api/scorecard/open", {"tags": ["agent"]})
    card_id = out.get("card_id")
    check("open scorecard", st == 200 and bool(card_id), {"status": st, "card_id": card_id})

    # 2. initial RESET to start play 1
    st, out = http(base, "POST", "/api/cmd/RESET", {"game_id": GAME, "card_id": card_id})
    guid = out.get("guid")
    check("initial RESET starts game", st == 200 and out.get("state") == "NOT_FINISHED",
          {"status": st, "state": out.get("state"), "levels": out.get("levels_completed")})

    # 3+4. sloppy play: trace with 200 wasted background clicks injected inside level 8
    for i, a in enumerate(trace[:L8_START]):
        st, out = act(base, a, guid, card_id)
        if st != 200:
            check("play-1 trace step ok", False, {"i": i, "status": st, "out": out})
            return
    lv_before_waste = out.get("levels_completed")
    waste_changed = 0
    for i in range(N_WASTE):
        st, out = act(base, {"name": "ACTION6", "x": 2 + (i % 3), "y": 18 + (i % 3)},
                      guid, card_id)
        if st != 200:
            check("wasted click accepted", False, {"i": i, "status": st, "out": out})
            return
        if out.get("levels_completed", 0) != lv_before_waste:
            waste_changed += 1
    check("200 wasted clicks in level 8, no level progress", waste_changed == 0,
          {"n_waste": N_WASTE, "level_advances": waste_changed,
           "level_during_waste": lv_before_waste})
    for i, a in enumerate(trace[L8_START:]):
        st, out = act(base, a, guid, card_id)
        if st != 200:
            check("play-1 tail step ok", False, {"i": i, "status": st, "out": out})
            return
    check("play 1 reaches WIN (sloppy, 142+200 actions)",
          out.get("state") == "WIN" and out.get("levels_completed") == 8,
          {"state": out.get("state"), "levels": out.get("levels_completed")})
    snap1 = snapshot_card(api, card_id)
    print("scorecard after play 1:", json.dumps(snap1))

    # 5. mid-run GET scorecard should be 403 in competition mode (eval parity surface)
    st, out = http(base, "GET", f"/api/scorecard/{card_id}")
    check("mid-run GET scorecard blocked (competition mode)", st == 403,
          {"status": st, "out": out})

    # 6. RESET while state==WIN
    st, out = http(base, "POST", "/api/cmd/RESET",
                   {"game_id": GAME, "guid": guid, "card_id": card_id})
    check("RESET at WIN accepted", st == 200,
          {"status": st, "state": out.get("state"), "levels": out.get("levels_completed")})
    snap2 = snapshot_card(api, card_id)
    print("scorecard after RESET-at-WIN:", json.dumps(snap2))
    check("second play opened with actions=0",
          snap2 is not None and snap2["total_plays"] == 2 and snap2["actions"][1] == 0,
          snap2)
    check("post-reset frame is a fresh game",
          out.get("state") == "NOT_FINISHED" and out.get("levels_completed") == 0,
          {"state": out.get("state"), "levels": out.get("levels_completed")})

    # 7. clean replay (play 2)
    for i, a in enumerate(trace):
        st, out = act(base, a, guid, card_id)
        if st != 200:
            check("play-2 trace step ok", False, {"i": i, "status": st, "out": out})
            return
    check("play 2 reaches WIN (clean, 142 actions)",
          out.get("state") == "WIN" and out.get("levels_completed") == 8,
          {"state": out.get("state"), "levels": out.get("levels_completed")})
    snap3 = snapshot_card(api, card_id)
    print("scorecard after play 2:", json.dumps(snap3))
    check("play accounting after clean replay",
          snap3["total_plays"] == 2 and snap3["actions"][1] == 142
          and snap3["states"] == ["WIN", "WIN"],
          snap3)

    # 8. close scorecard, assert score == max over plays
    st, closed = http(base, "POST", "/api/scorecard/close", {"card_id": card_id})
    check("close scorecard", st == 200, {"status": st})
    env = next((e for e in closed.get("environments", []) if e["id"].startswith("sb26")), None)
    if env is None:
        check("closed scorecard has sb26 entry", False, closed)
        return
    runs = env["runs"]
    run_scores = [r["score"] for r in runs]
    check("closed scorecard shows 2 runs", len(runs) == 2,
          [{"score": r["score"], "actions": r["actions"], "state": r["state"],
            "levels_completed": r["levels_completed"]} for r in runs])
    check("game score == max over plays",
          abs(env["score"] - max(run_scores)) < 1e-9,
          {"game_score": env["score"], "run_scores": run_scores})
    check("clean replay strictly beats sloppy play", run_scores[1] > run_scores[0],
          {"sloppy": run_scores[0], "clean": run_scores[1]})
    print("closed sb26 entry:", json.dumps(env, indent=1))
    print("closed card overall score:", closed.get("score"))

    json.dump({"card_id": card_id, "results": results, "closed_sb26": env,
               "closed_score": closed.get("score"),
               "snapshots": {"after_play1": snap1, "after_reset_at_win": snap2,
                             "after_play2": snap3}},
              open(os.path.join(HERE, "test1_local_results.json"), "w"), indent=1)
    server.shutdown()
    n_fail = sum(1 for r in results if not r["ok"])
    print(f"\nE1 DONE: {len(results)-n_fail}/{len(results)} assertions passed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
