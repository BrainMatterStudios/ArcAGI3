#!/usr/bin/env python3
"""INDEPENDENT probe: does competition mode swallow a RESET at a level-up boundary?

Written from scratch to check two agents' claims, using a DIFFERENT mechanism than
either of them: the shipped Flask app driven in-process via werkzeug's test client
(no network, no external state, fully deterministic and inspectable).

This is the eval-faithful path. taaf's CompetitionArcadeServer builds exactly this
app with competition_mode=True (competition_arcade.py:188-193) and the offline rig
runs against it, so the code exercised here is the code eval exercises — unlike a
direct LocalEnvironmentWrapper test, which bypasses the server guard entirely and
therefore REPRODUCES the exploit misleadingly.

Causal isolation: identical script, identical trace, only competition_mode flipped.
Note api.py:169-171 sets competition_mode=True if the client sends the key AT ALL
(`competition_mode_raw is not None`), so the non-competition arm must both build a
non-competition server AND omit the field when opening.

Validity guards (a probe that cannot fail is not a probe):
  * aborts unless the replayed trace actually produces levels_completed == 1
  * asserts the engine's _action_count is really 0 at the moment of the RESET
  * reports action_input.id, which fingerprints whether g.step() ran at all

Usage:  .venv/bin/python scripts/probe_reset_competition_guard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACE = REPO / "docs/test-artifacts-2026-08-02/sb26_win_trace.json"
ENV_DIR = REPO / "environment_files"
L1_ACTIONS = 9  # the recorded trace clears sb26 level 1 in 9 actions


def build(competition: bool):
    import arc_agi
    from arc_agi import server as arc_agi_server

    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    app, api = arc_agi_server.create_app(arcade, competition_mode=competition)
    app.config.update(TESTING=True)
    return arcade, app.test_client()


def post(client, path, body):
    r = client.post(path, json=body)
    try:
        return r.status_code, r.get_json()
    except Exception:  # noqa: BLE001
        return r.status_code, {}


def game_id(arcade) -> str:
    for e in arcade.available_environments:
        if e.game_id.startswith("sb26"):
            return e.game_id
    raise SystemExit("sb26 not found in environment_files")


def inner_game(arcade, guid):
    """Reach the live ARCBaseGame so we can read _action_count directly."""
    for attr in ("environments", "_environments", "games", "_games"):
        d = getattr(arcade, attr, None)
        if isinstance(d, dict):
            for k, v in d.items():
                if guid in str(k):
                    g = getattr(v, "_game", None)
                    if g is not None:
                        return g
    return None


def run(competition: bool) -> dict:
    trace = json.loads(TRACE.read_text())
    arcade, client = build(competition)
    gid = game_id(arcade)

    st, out = post(client, "/api/scorecard/open",
                   {"tags": ["guard-probe"], **({"competition_mode": True} if competition else {})})
    card = out.get("card_id")
    if not card:
        return {"error": f"scorecard open failed {st} {out}"}

    st, out = post(client, "/api/cmd/RESET", {"game_id": gid, "card_id": card})
    guid = out.get("guid")
    if not guid:
        return {"error": f"initial RESET failed {st} {out}"}

    for i, a in enumerate(trace[:L1_ACTIONS]):
        body = {"game_id": gid, "guid": guid, "card_id": card}
        if a["name"] == "ACTION6":
            body["x"], body["y"] = a["x"], a["y"]
        st, out = post(client, f"/api/cmd/{a['name']}", body)
        if st != 200:
            return {"error": f"action {i} failed {st} {out}"}

    levels = out.get("levels_completed")
    if levels != 1:
        return {"error": f"PROBE INVALID — trace did not level up (levels_completed={levels})"}
    frame_before = json.dumps(out.get("frame"))

    g = inner_game(arcade, guid)
    ac_before = getattr(g, "_action_count", None) if g is not None else None

    st, out = post(client, "/api/cmd/RESET", {"game_id": gid, "guid": guid, "card_id": card})
    result = {
        "competition_mode": competition,
        "action_count_at_reset": ac_before,
        "http": st,
        "levels_completed_after": out.get("levels_completed"),
        "state_after": out.get("state"),
        "full_reset_field": out.get("full_reset"),
        "action_input_id": (out.get("action_input") or {}).get("id"),
        "frame_unchanged": json.dumps(out.get("frame")) == frame_before,
    }

    st, closed = post(client, "/api/scorecard/close", {"card_id": card})
    entry = None
    for e in (closed.get("environments") or []):
        if e.get("id") == gid or str(e.get("game_id", "")).startswith("sb26"):
            entry = e
    if entry is None:
        entry = (closed.get("cards") or {}).get(gid)
    if isinstance(entry, dict):
        result["total_plays"] = entry.get("total_plays")
        result["actions"] = entry.get("actions")
        result["resets"] = entry.get("resets")
        result["actions_by_level"] = entry.get("actions_by_level")
    return result


def main() -> int:
    rows = []
    for competition in (True, False):
        r = run(competition)
        rows.append(r)
        print(f"\n=== competition_mode={competition} ===")
        if "error" in r:
            print("  ", r["error"])
            continue
        for k, v in r.items():
            if k != "competition_mode":
                print(f"   {k:24s} {v}")

    print("\n" + "=" * 62)
    ok = [r for r in rows if "error" not in r]
    if len(ok) != 2:
        print("INCONCLUSIVE — an arm failed to produce a valid measurement")
        return 1
    comp, free = ok[0], ok[1]
    swallowed = comp["levels_completed_after"] == 1 and comp["frame_unchanged"]
    fired = free["levels_completed_after"] == 0
    print(f"competition arm swallowed the RESET : {swallowed}")
    print(f"non-competition arm full-reset      : {fired}")
    if swallowed and fired:
        print("VERDICT: the guard is real and competition_mode is the cause. Lever DEAD at eval.")
    elif not swallowed:
        print("VERDICT: the RESET was NOT swallowed in competition mode — LEVER MAY BE ALIVE.")
    else:
        print("VERDICT: ambiguous — both arms behaved the same; probe cannot separate cause.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
