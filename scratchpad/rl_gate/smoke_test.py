"""smoke_test.py — GPU-free verification of the whole capture path.

Mock brain (fixed `python` tool call -> action('UP')) <- capture_proxy <- REAL
duck harness (ToolAgent + _HarnessGameSession + TAAF GameAPI) playing ONE real
dev game offline. Then verifies:
  1. trace.jsonl is well-formed, with image_url parts intact (multimodal on),
  2. manifest has levels_completed=0 and sane counters,
  3. traces_to_grpo emits grouped per-turn records.

Run:  .venv/bin/python scratchpad/rl_gate/smoke_test.py [--game tu93]
"""
from __future__ import annotations

import argparse
import http.server
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parents[1] / ".venv" / "bin" / "python"

MOCK_REPLY = {
    "id": "cmpl-mock", "object": "chat.completion", "model": "mock",
    "choices": [{
        "index": 0,
        "finish_reason": "tool_calls",
        "message": {
            "role": "assistant",
            "content": "Plan: probe with UP.",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": "action('UP')"})},
            }],
        },
    }],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 30, "total_tokens": 1230},
}


class MockBrain(http.server.BaseHTTPRequestHandler):
    n_requests = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        _ = self.rfile.read(n)
        MockBrain.n_requests += 1
        body = json.dumps(MOCK_REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="tu93")
    ap.add_argument("--max-actions", type=int, default=3)
    args = ap.parse_args()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockBrain)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    upstream = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    print(f"[smoke] mock brain at {upstream}")

    workdir = HERE / "episodes" / f"smoke_{args.game}_{time.strftime('%H%M%S')}"
    cmd = [str(PY), str(HERE / "run_rollout.py"), "--game", args.game,
           "--upstream", upstream, "--workdir", str(workdir),
           "--max-actions", str(args.max_actions), "--max-runtime-s", "180",
           "--multimodal", "--rollout-id", "smoke1"]
    print("[smoke] running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr[-4000:])
    assert res.returncode == 0, f"run_rollout failed rc={res.returncode}"

    # 1. trace well-formed + images intact
    trace = workdir / "trace.jsonl"
    records = [json.loads(l) for l in trace.read_text().splitlines()]
    assert records, "no captured requests"
    img_parts = 0
    for rec in records:
        assert rec["rollout_id"] == "smoke1"
        assert rec["response"]["choices"][0]["message"]["tool_calls"], "tool_calls lost in capture"
        for msg in rec["request"]["messages"]:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if part.get("type") == "image_url":
                        img_parts += 1
                        url = part["image_url"]["url"]
                        assert url.startswith("data:image/png;base64,") and len(url) > 100
    assert img_parts > 0, "no image parts captured despite --multimodal"
    print(f"[smoke] OK trace: {len(records)} request/response pairs, {img_parts} intact image parts")

    # 2. manifest
    manifest = json.loads((workdir / "manifest.json").read_text())
    assert manifest["levels_completed"] == 0, manifest  # mock brain can't win
    assert manifest["n_model_requests"] == len(records)
    assert manifest["n_turns"] >= 1
    print(f"[smoke] OK manifest: reward={manifest['reward']} turns={manifest['n_turns']} "
          f"actions={manifest['n_actions']} tokens={manifest['tokens_total']}")
    if manifest["n_actions"] == 0:
        print("[smoke] NOTE: 0 engine actions executed (UP may be invalid for this game) — "
              "capture path still fully exercised")

    # 3. grpo conversion
    out_dir = workdir / "grpo"
    res2 = subprocess.run([str(PY), str(HERE / "traces_to_grpo.py"), str(workdir),
                           "--out", str(out_dir)], capture_output=True, text=True)
    sys.stdout.write(res2.stdout)
    assert res2.returncode == 0, res2.stderr
    samples = [json.loads(l) for l in (out_dir / "grpo_samples.jsonl").read_text().splitlines()]
    assert len(samples) == len(records)
    assert all(s["episode"] == "smoke1" and s["reward"] == 0 for s in samples)
    assert all(s["action_message"].get("tool_calls") for s in samples)
    print(f"[smoke] OK grpo: {len(samples)} per-turn samples grouped under episode smoke1")
    print("[smoke] ALL CHECKS PASSED")
    srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
