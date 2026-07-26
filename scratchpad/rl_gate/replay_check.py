"""replay_check.py — ReplayMockLLM: prove a harness change is a behavioral no-op.

Replays a RECORDED episode's LLM responses (trace.jsonl from capture_proxy)
back at the live duck harness while it replays the same game, and diffs every
request the harness produces against the recorded one. If the harness change
under test (e.g. the two-tier memory ledger) alters ANY prompt byte on a
death-free trace, the first divergence is reported with request seq + a
unified diff excerpt, and the exit code is nonzero.

This is the Stage-1 gate for memory/harness changes:
    record:  .venv/bin/python scratchpad/rl_gate/smoke_test.py --game ul01 \
                 --env-dir scratchpad/holdout_arcint/environment_files
    check:   .venv/bin/python scratchpad/rl_gate/replay_check.py \
                 --episode scratchpad/rl_gate/episodes/<dir> --game ul01 \
                 [--env-dir <same env dir>]

Normalization: UUIDs and float timestamps inside message content are masked
before diffing (game GUIDs differ across runs; frames must not).
"""
from __future__ import annotations

import argparse
import difflib
import http.server
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parents[1] / ".venv" / "bin" / "python"

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
TS_RE = re.compile(r"\b17[0-9]{8}(?:\.[0-9]+)?\b")  # epoch seconds, this decade
# wall-clock fields the harness embeds in tool outputs — volatile by nature
# (values live inside escaped JSON strings, hence the loose \\-and-quote glue)
WALLCLOCK_RE = re.compile(
    r'((?:run_elapsed_seconds|time_remaining_seconds)[\\"]*\s*:\s*)[-+0-9.eE]+')


def normalize(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    s = UUID_RE.sub("<UUID>", s)
    s = TS_RE.sub("<TS>", s)
    s = WALLCLOCK_RE.sub(r"\1<T>", s)
    return s


class ReplayState:
    def __init__(self, records):
        self.records = records
        self.idx = 0
        self.mismatches = []
        self.lock = threading.Lock()


def make_handler(state: ReplayState):
    class ReplayMock(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with state.lock:
                if state.idx >= len(state.records):
                    self.send_response(410)
                    self.end_headers()
                    self.wfile.write(b'{"error":"replay exhausted"}')
                    state.mismatches.append(
                        {"seq": state.idx + 1, "kind": "extra_request"})
                    return
                rec = state.records[state.idx]
                state.idx += 1
            got = normalize(json.loads(body.decode()))
            want = normalize(rec["request"])
            if got != want:
                diff = "\n".join(difflib.unified_diff(
                    want.split(","), got.split(","),
                    "recorded", "live", lineterm="", n=1))
                state.mismatches.append(
                    {"seq": rec["seq"], "kind": "request_diverged",
                     "diff_excerpt": diff[:2000]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(rec["response"]).encode())

        def do_GET(self):  # /v1/models liveness
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"replay-mock"}]}')

    return ReplayMock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", required=True, type=Path,
                    help="recorded episode dir containing trace.jsonl")
    ap.add_argument("--game", required=True)
    ap.add_argument("--env-dir", default="")
    ap.add_argument("--max-actions", type=int, default=0,
                    help="default: recorded request count")
    args = ap.parse_args()

    records = [json.loads(l) for l in
               (args.episode / "trace.jsonl").read_text().splitlines()]
    records.sort(key=lambda r: r["seq"])
    assert records, "empty recorded trace"
    print(f"[replay] {len(records)} recorded requests from {args.episode}")

    state = ReplayState(records)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    upstream = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    workdir = HERE / "episodes" / f"replay_{args.game}_{time.strftime('%H%M%S')}"
    cmd = [str(PY), str(HERE / "run_rollout.py"), "--game", args.game,
           "--upstream", upstream, "--workdir", str(workdir),
           "--max-actions", str(args.max_actions or len(records)),
           "--max-runtime-s", "600", "--multimodal",
           "--rollout-id", "replay1"]
    if args.env_dir:
        cmd += ["--env-dir", args.env_dir]
    print("[replay] running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    srv.shutdown()
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-3000:])
        print(f"[replay] FAIL: run_rollout rc={res.returncode}")
        return 2

    consumed = state.idx
    print(f"[replay] live run consumed {consumed}/{len(records)} recorded turns")
    if consumed < len(records):
        state.mismatches.append({"seq": consumed + 1, "kind": "missing_requests",
                                 "detail": f"live run stopped after {consumed}"})
    if state.mismatches:
        print(f"[replay] VERDICT: DIVERGED ({len(state.mismatches)} mismatches)")
        for m in state.mismatches[:3]:
            print(json.dumps(m, indent=2)[:2500])
        (workdir / "replay_mismatches.json").write_text(
            json.dumps(state.mismatches, indent=2))
        return 1
    print("[replay] VERDICT: BYTE-IDENTICAL (normalized) — harness change is a "
          "no-op on this trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
