"""Extract kill-or-scale metrics from a Tycho run directory (one game)."""
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
game = sys.argv[2]
rec_path = run_dir / f"game_{game}.json"
out = {"game": game, "record_exists": rec_path.exists()}

if rec_path.exists():
    r = json.loads(rec_path.read_text())
    out.update({
        "levels_completed": r.get("levels_completed"),
        "n_levels": r.get("n_levels"),
        "baselines": r.get("baselines"),
        "total_actions": r.get("total_actions_including_unfinished", r.get("total_actions")),
        "resets": r.get("resets"),
        "final_state": r.get("final_state"),
        "env_score_rhae": r.get("env_score"),
        "wall_clock_s": round(r.get("wall_clock_s", 0), 1),
        "stop_reason": r.get("stop_reason"),
        "error": r.get("error"),
        "noop_actions": r.get("noop_actions"),
        "distinct_frames": r.get("distinct_frames"),
        "revisits": r.get("revisits"),
        "actions_before_first_level": r.get("actions_before_first_level"),
        "builder_invocations": r.get("builder_invocations"),
        "truncated_levels": r.get("truncated_levels"),
    })
    tin = tout = ncalls = 0
    call_types = {}
    latencies = []
    for step in r.get("trace", []):
        for c in ((step.get("reasoning") or {}).get("llm_calls") or []):
            tin += c.get("tokens_in", 0); tout += c.get("tokens_out", 0); ncalls += 1
            call_types[c.get("call_type", "?")] = call_types.get(c.get("call_type", "?"), 0) + 1
            if c.get("latency_ms"): latencies.append(c["latency_ms"])
    out["llm_calls_in_trace"] = ncalls
    out["tokens_in"] = tin
    out["tokens_out"] = tout
    out["tokens_total"] = tin + tout
    out["call_types"] = call_types
    if latencies:
        latencies.sort()
        out["latency_ms_median"] = latencies[len(latencies)//2]
        out["latency_ms_p90"] = latencies[int(len(latencies)*0.9)]

status_path = run_dir / "status" / game / "status.json"
if status_path.exists():
    s = json.loads(status_path.read_text())
    out["status"] = {k: s.get(k) for k in (
        "state", "level", "levels_completed", "action_count", "current_rhae", "error")}

ws = run_dir / "ws" / game
if ws.exists():
    wm = ws / "world_model.py"
    out["workspace_files"] = sorted(p.name for p in ws.iterdir())
    if wm.exists():
        text = wm.read_text()
        out["world_model_bytes"] = len(text)
        out["world_model_is_template"] = "your editable model of this game" in text[:200]
print(json.dumps(out, indent=2))
