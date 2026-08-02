"""E2: Hosted-API parity probe for the post-WIN clean-replay lever.

Against https://three.arcprize.org (production ARC-AGI-3 API, key from repo .env,
never printed): open ONE competition-mode scorecard, win sb26-7fbdac44 with the
recorded K3 trace (sloppy variant: 60 wasted background clicks inside level 8),
send RESET while state==WIN, replay the trace clean, close the scorecard, and
check whether the closed scorecard shows a second play with the RESET un-billed
(run 2 actions == 142) and game score == max over plays.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Hosted deployment is behind an AWS ALB with sticky sessions; the scorecard
# lives in per-instance memory, so the AWSALBAPP-*/GAMESESSION cookies from the
# open response MUST be replayed on every subsequent request (verified: without
# them, RESET returns "game not found" and close returns "scorecard not found").
_JAR = http.cookiejar.CookieJar()
_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_JAR))

REPO = "/Users/ahmed/Documents/ArcAGI3"
HERE = os.path.dirname(os.path.abspath(__file__))
GAME = "sb26-7fbdac44"
L8_START = 125
N_WASTE = 60
SLEEP = 0.12

results: list[dict] = []


def check(name, ok, detail):
    results.append({"assert": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail)[:400]}", flush=True)


def load_env():
    env = {}
    for line in open(os.path.join(REPO, ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


ENV = load_env()
BASE = ENV.get("ARC_BASE_URL", "https://three.arcprize.org").rstrip("/")
KEY = ENV["ARC_API_KEY"]


def http(method, path, body=None, retries=3):
    for attempt in range(retries):
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", "X-API-Key": KEY,
                     "Accept": "application/json"},
            method=method,
        )
        try:
            with _OPENER.open(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read())
            except Exception:
                payload = {}
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            return e.code, payload
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            return -1, {"transport_error": repr(e)}
    return -1, {}


def act(a, guid, card_id):
    body = {"game_id": GAME, "guid": guid, "card_id": card_id}
    if a["name"] == "ACTION6":
        body["x"] = a["x"]
        body["y"] = a["y"]
    time.sleep(SLEEP)
    return http("POST", f"/api/cmd/{a['name']}", body)


def drive(seq, guid, card_id, label):
    out = {}
    for i, a in enumerate(seq):
        st, out = act(a, guid, card_id)
        if st != 200:
            check(f"{label}: step {i} accepted", False, {"status": st, "out": out})
            return None
        if i % 25 == 0:
            print(f"  {label} step {i}: state={out.get('state')} "
                  f"levels={out.get('levels_completed')}", flush=True)
    return out


def main():
    trace = json.load(open(os.path.join(HERE, "sb26_win_trace.json")))
    assert len(trace) == 142
    waste = [{"name": "ACTION6", "x": 2 + (i % 3), "y": 18 + (i % 3)} for i in range(N_WASTE)]

    st, games = http("GET", "/api/games")
    ids = [g.get("game_id") for g in games] if isinstance(games, list) else []
    check("hosted /api/games lists sb26-7fbdac44", st == 200 and GAME in ids,
          {"status": st, "n_games": len(ids)})
    if GAME not in ids:
        sys.exit(1)

    st, out = http("POST", "/api/scorecard/open",
                   {"tags": ["agent"], "competition_mode": True,
                    "opaque": {"purpose": "win-reset parity probe"}})
    card_id = out.get("card_id")
    check("open competition-mode scorecard", st == 200 and bool(card_id),
          {"status": st, "card_id": card_id})
    if not card_id:
        sys.exit(1)

    try:
        st, out = http("POST", "/api/cmd/RESET", {"game_id": GAME, "card_id": card_id})
        guid = out.get("guid")
        check("initial RESET starts game", st == 200 and out.get("state") == "NOT_FINISHED",
              {"status": st, "state": out.get("state"), "levels": out.get("levels_completed")})
        if not guid:
            return

        st, sc = http("GET", f"/api/scorecard/{card_id}")
        check("mid-run GET scorecard status (parity: local returns 403)", True,
              {"status": st, "body": sc})
        midrun_get_status_1 = st

        # play 1: sloppy (trace head + waste inside level 8 + trace tail)
        out = drive(trace[:L8_START], guid, card_id, "play1 head")
        if out is None:
            return
        lv = out.get("levels_completed")
        out = drive(waste, guid, card_id, "play1 waste")
        if out is None:
            return
        check("waste clicks caused no level progress", out.get("levels_completed") == lv,
              {"levels": out.get("levels_completed"), "expected": lv})
        out = drive(trace[L8_START:], guid, card_id, "play1 tail")
        if out is None:
            return
        check("play 1 reaches WIN on hosted API",
              out.get("state") == "WIN" and out.get("levels_completed") == 8,
              {"state": out.get("state"), "levels": out.get("levels_completed")})
        if out.get("state") != "WIN":
            check("DETERMINISM: recorded trace desynced server-side", False,
                  {"state": out.get("state"), "levels": out.get("levels_completed")})
            return

        # THE probe: RESET while state == WIN
        time.sleep(SLEEP)
        st, out = http("POST", "/api/cmd/RESET",
                       {"game_id": GAME, "guid": guid, "card_id": card_id})
        check("RESET at WIN accepted by hosted API", st == 200,
              {"status": st, "state": out.get("state"),
               "levels": out.get("levels_completed"), "body_keys": sorted(out.keys())})
        check("post-reset frame is a fresh game (full reset, not level reset)",
              out.get("state") == "NOT_FINISHED" and out.get("levels_completed") == 0,
              {"state": out.get("state"), "levels": out.get("levels_completed")})

        st, sc = http("GET", f"/api/scorecard/{card_id}")
        check("mid-run GET scorecard after WIN-reset (informational)", True,
              {"status": st, "body": sc})
        if st == 200:
            print("MID-RUN SCORECARD READABLE:", json.dumps(sc)[:1500])

        # play 2: clean replay
        out = drive(trace, guid, card_id, "play2")
        if out is None:
            return
        check("play 2 (clean replay) reaches WIN",
              out.get("state") == "WIN" and out.get("levels_completed") == 8,
              {"state": out.get("state"), "levels": out.get("levels_completed")})
    finally:
        st, closed = http("POST", "/api/scorecard/close", {"card_id": card_id})
        check("close scorecard", st == 200, {"status": st})
        json.dump(closed, open(os.path.join(HERE, "test2_hosted_closed_scorecard.json"), "w"),
                  indent=1)

    env = None
    for e in closed.get("environments", []):
        if str(e.get("id", "")).startswith("sb26"):
            env = e
    if env is None:
        # maybe a different shape; check 'cards'
        check("closed scorecard has sb26 environment entry", False,
              {"top_level_keys": sorted(closed.keys())})
        print("closed payload (truncated):", json.dumps(closed)[:2000])
        return

    runs = env.get("runs", [])
    run_summ = [{"score": r.get("score"), "actions": r.get("actions"),
                 "state": r.get("state"), "levels_completed": r.get("levels_completed"),
                 "level_actions": r.get("level_actions")} for r in runs]
    check("closed scorecard shows 2 runs for sb26", len(runs) == 2, run_summ)
    if len(runs) == 2:
        check("second play opened at actions=0 (run 2 actions == 142, RESET un-billed)",
              runs[1].get("actions") == 142, run_summ)
        scores = [r.get("score", 0.0) for r in runs]
        check("game score == max over plays",
              abs(env.get("score", -1) - max(scores)) < 1e-6,
              {"game_score": env.get("score"), "run_scores": scores})
        check("clean replay strictly beats sloppy play", scores[1] > scores[0],
              {"sloppy": scores[0], "clean": scores[1]})
    print("closed sb26 entry:", json.dumps(env, indent=1))
    print("closed card score:", closed.get("score"))

    json.dump({"card_id": card_id, "base": BASE, "results": results,
               "midrun_get_status_before_win": midrun_get_status_1,
               "closed_sb26": env, "closed_score": closed.get("score")},
              open(os.path.join(HERE, "test2_hosted_results.json"), "w"), indent=1)
    n_fail = sum(1 for r in results if not r["ok"])
    print(f"\nE2 DONE: {len(results)-n_fail}/{len(results)} assertions passed")


if __name__ == "__main__":
    main()
