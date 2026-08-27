"""Tool-discipline stats for the Track-1 KAT probe (and 27B comparisons).

Answers the pre-registered question directly from the trace:
  - did the actor EDIT world_model.py (write_file events targeting it,
    plus final on-disk template check)?
  - did it RUN the verifier (run_python / shell events invoking verify.py)?
Plus general tool discipline: per-tool counts, failed tool results,
unknown-tool events.
"""
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
game = sys.argv[2]
rec = run_dir / f"game_{game}.json"
out = {"run": str(run_dir), "record_exists": rec.exists()}

if rec.exists():
    r = json.loads(rec.read_text())
    tools = {}
    wm_writes = 0
    verify_invocations = 0
    failed_results = 0
    unknown_tools = 0
    verify_reports = 0
    for step in r.get("trace", []):
        rz = step.get("reasoning") or {}
        if rz.get("verify"):
            verify_reports += 1
        for ev in rz.get("tool_trace") or []:
            if not isinstance(ev, dict):
                unknown_tools += 1
                continue
            name = ev.get("tool") or "?"
            tools[name] = tools.get(name, 0) + 1
            if name == "?":
                unknown_tools += 1
            args = json.dumps(ev.get("args") or {})
            result = str(ev.get("result"))[:2000]
            if name == "write_file" and "world_model" in args:
                wm_writes += 1
            if "verify.py" in args or ("verify" in args and name == "run_python"):
                verify_invocations += 1
            low = result.lower()
            if any(k in low for k in ("traceback", "error", "failed", "no such file")):
                failed_results += 1
    out.update({
        "tool_counts": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
        "world_model_write_file_events": wm_writes,
        "verify_invocations_in_args": verify_invocations,
        "verify_reports_in_trace": verify_reports,
        "tool_results_with_error_text": failed_results,
        "unknown_tool_events": unknown_tools,
        "total_tool_events": sum(tools.values()),
    })

ws = run_dir / "ws" / game
wm = ws / "world_model.py"
if wm.exists():
    text = wm.read_text()
    out["world_model_bytes"] = len(text)
    out["world_model_is_template"] = "your editable model of this game" in text[:200]
vf = ws / "verify.py"
if vf.exists():
    out["verify_py_bytes"] = len(vf.read_text())
att = ws / "attempts"
if att.exists():
    out["attempt_dirs"] = sorted(p.name for p in att.iterdir())[:10]

print(json.dumps(out, indent=2))
