"""capture_proxy.py — token-faithful capture proxy for the duck's model calls.

Sits between the duck harness and an OpenAI-compatible endpoint. Forwards
requests verbatim and appends one JSONL record per /chat/completions
request/response pair: full messages (incl. image_url base64 parts),
sampling params, usage, timestamps, rollout id.

Verified against the real duck call site
(submission/_adopt/taaf-src/src/ARC3-Inference/inference/agent/tool_agent.py
_chat_completion): a single blocking `requests.post(json=payload)` with
"stream": False (inference/utils/openai_compat.py build_chat_payload). So the
buffered JSON path is the hot path; a chunked passthrough is kept for
stream=True requests just in case (raw bytes are relayed as they arrive and
the concatenated text is logged under "raw_stream").

Usage:
  standalone:  python capture_proxy.py --upstream http://127.0.0.1:8000/v1 \
                 --out trace.jsonl --rollout-id ep1 [--port 9000]
  in-process:  from capture_proxy import start_proxy
               proxy = start_proxy(upstream, out_path, rollout_id)  # .base_url, .stop()
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HOP_BY_HOP = {
    "host", "content-length", "connection", "keep-alive", "transfer-encoding",
    "te", "upgrade", "proxy-authorization", "proxy-authenticate", "accept-encoding",
}


class _CaptureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # set by start_proxy on the server object: upstream, out_path, rollout_id, lock, seq

    def log_message(self, *args):  # keep the duck's stdout clean
        pass

    def do_GET(self):
        self._forward(b"")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        self._forward(self.rfile.read(n) if n else b"")

    def _forward(self, body: bytes) -> None:
        srv = self.server
        url = srv.upstream.rstrip("/") + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        headers["Accept-Encoding"] = "identity"  # keep bodies loggable
        req = urllib.request.Request(url, data=body or None, headers=headers, method=self.command)
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=srv.upstream_timeout)
        except urllib.error.HTTPError as exc:  # forward upstream errors verbatim
            resp = exc
        except Exception as exc:  # upstream unreachable
            self.send_response(502)
            msg = json.dumps({"error": f"capture_proxy: upstream unreachable: {exc}"}).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        status = resp.status if hasattr(resp, "status") else resp.code
        resp_headers = resp.headers
        is_stream = "text/event-stream" in (resp_headers.get("Content-Type") or "")

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() not in _HOP_BY_HOP:
                self.send_header(k, v)

        if is_stream:  # relay chunks live, log concatenation (duck never uses this path)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            chunks = []
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")
            resp_body = b"".join(chunks)
        else:
            resp_body = resp.read()
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        if self.path.rstrip("/").endswith("/chat/completions"):
            self._log(body, resp_body, status, t0, streamed=is_stream)

    def _log(self, body: bytes, resp_body: bytes, status: int, t0: float, *, streamed: bool) -> None:
        srv = self.server

        def _parse(raw: bytes):
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                return None

        request_payload = _parse(body)
        response_payload = None if streamed else _parse(resp_body)
        with srv.lock:
            srv.seq += 1
            record = {
                "schema": "rl_gate.capture.v1",
                "rollout_id": self.headers.get("X-Rollout-Id") or srv.rollout_id,
                "seq": srv.seq,
                "path": self.path,
                "status": status,
                "ts_request": t0,
                "ts_response": time.time(),
                "latency_s": round(time.time() - t0, 3),
                "request": request_payload if request_payload is not None else {"raw": body.decode("utf-8", "replace")},
                "response": response_payload,
            }
            if streamed:
                record["raw_stream"] = resp_body.decode("utf-8", "replace")
            with open(srv.out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Proxy:
    def __init__(self, server: ThreadingHTTPServer, thread: threading.Thread):
        self._server = server
        self._thread = thread
        self.port = server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)


def start_proxy(upstream: str, out_path: str | Path, rollout_id: str = "",
                port: int = 0, upstream_timeout: float = 600.0) -> Proxy:
    """Start the proxy on 127.0.0.1:<port> (0 = ephemeral). `upstream` should
    include the /v1 suffix, e.g. http://127.0.0.1:8000/v1 — the duck appends
    /chat/completions to its base url, and we append the incoming path to
    upstream minus its own /v1 (paths arrive as /v1/...)."""
    upstream = upstream.rstrip("/")
    if upstream.endswith("/v1"):
        upstream = upstream[: -len("/v1")]
    server = ThreadingHTTPServer(("127.0.0.1", port), _CaptureHandler)
    server.upstream = upstream
    server.upstream_timeout = upstream_timeout
    server.out_path = str(out_path)
    server.rollout_id = rollout_id
    server.lock = threading.Lock()
    server.seq = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return Proxy(server, thread)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream", required=True, help="model endpoint incl. /v1")
    ap.add_argument("--out", required=True, help="JSONL trace output path")
    ap.add_argument("--rollout-id", default="")
    ap.add_argument("--port", type=int, default=9009)
    args = ap.parse_args()
    proxy = start_proxy(args.upstream, args.out, args.rollout_id, port=args.port)
    print(f"[capture_proxy] listening on {proxy.base_url} -> {args.upstream}, logging {args.out}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        proxy.stop()


if __name__ == "__main__":
    main()
