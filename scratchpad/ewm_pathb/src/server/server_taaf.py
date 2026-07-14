"""server_taaf.py — flask server that wraps a TAAF `Game` in-process.

Lets the EWM scaffold (client + ewm_agent, subprocess) drive a TAAF game via the same REST
API my client already speaks, so every action flows through `game.execute_action(...)` and is
SCORED by the TAAF benchmark. Runs in a background thread inside a taaf Solver's _run_games;
the game is pre-started (game.start_game()) by the solver. Reuses the FrameDataRaw serialization
shape from server.py so the client + verifier are unchanged.

Usage (from a Solver):
    srv = serve_game(game, port=8890)   # returns a Stoppable; game must be start_game()'d already
    ... run ewm_agent against http://127.0.0.1:8890 ...
    srv.stop()
"""
from __future__ import annotations
import logging
import threading
from typing import Any

import arcengine
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

LOGGER = logging.getLogger("arc-server-taaf")


def serialize_frame(gs: Any, step_index: int) -> dict[str, Any]:
    """Serialize a taaf GameState (via its .raw FrameDataRaw) to the client payload."""
    raw = gs.raw
    frame_layers = [f.tolist() if hasattr(f, "tolist") else f for f in raw.frame]
    ai = None
    if getattr(raw, "action_input", None) is not None:
        aid = raw.action_input.id
        ai = {"id": aid.name if hasattr(aid, "name") else str(aid), "data": raw.action_input.data}
    return {
        "step_index": step_index,
        "state": raw.state.name if hasattr(raw.state, "name") else str(raw.state),
        "levels_completed": raw.levels_completed,
        "win_levels": getattr(raw, "win_levels", 0),
        "available_actions": list(raw.available_actions),
        "action_input": ai,
        "frame": frame_layers,
    }


class _Server(threading.Thread):
    def __init__(self, app: Flask, host: str, port: int):
        super().__init__(daemon=True)
        self._srv = make_server(host, port, app, threaded=True)
        self._ctx = app.app_context()
        self._ctx.push()

    def run(self) -> None:
        self._srv.serve_forever()

    def stop(self) -> None:
        self._srv.shutdown()


def serve_game(game: Any, *, host: str = "127.0.0.1", port: int = 8890) -> _Server:
    app = Flask(f"taaf_srv_{port}")
    state = {"step_index": 0}
    lock = threading.RLock()

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.post("/game/start")
    def start_game() -> Any:
        with lock:
            gs = game.current_state  # solver already called game.start_game()
            return jsonify({"session": {"session_token": "taaf"},
                            "frame": serialize_frame(gs, state["step_index"])})

    @app.post("/game/action")
    def apply_action() -> Any:
        payload = request.get_json(silent=True) or {}
        name = payload.get("action")
        data = payload.get("data")
        if not name:
            return jsonify({"error": "action is required"}), 400
        try:
            action = arcengine.ActionInput(id=arcengine.GameAction[str(name)], data=dict(data) if data else {})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"bad action {name}: {exc}"}), 400
        with lock:
            try:
                gs = game.execute_action(action)
            except Exception as exc:  # noqa: BLE001 — surface engine/validation rejects to the client
                return jsonify({"error": f"action failed: {exc}"}), 500
            state["step_index"] += 1
            sf = serialize_frame(gs, state["step_index"])
            if sf["action_input"] is None:
                sf["action_input"] = {"id": str(name), "data": data or {}}
            return jsonify({"session": {"session_token": "taaf"}, "frame": sf})

    @app.get("/game/last-step")
    def last_step() -> Any:
        with lock:
            return jsonify({"session": {"session_token": "taaf"},
                            "frame": serialize_frame(game.current_state, state["step_index"])})

    @app.get("/game/current")
    def current() -> Any:
        with lock:
            return jsonify({"session": {"session_token": "taaf"},
                            "frame": serialize_frame(game.current_state, state["step_index"])})

    @app.post("/game/stop")
    def stop() -> Any:
        return jsonify({"stopped": True, "session": {"session_token": "taaf"}})

    srv = _Server(app, host, port)
    srv.start()
    return srv
