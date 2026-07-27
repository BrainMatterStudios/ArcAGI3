"""claude_shim.py — OpenAI-compatible endpoint backed by the Claude Code CLI.

Presents POST /v1/chat/completions + GET /v1/models so the duck harness
(run_rollout / capture_proxy) can use Claude as the teacher over Ahmed's
Claude Code SUBSCRIPTION — no API key. Each request is translated into one
headless `claude -p` invocation (stream-json in/out, base64 images passed
through as image blocks), and the reply is returned as an OpenAI chat
completion whose content carries the duck markup tool-call format, which
tool_agent._recover_tool_calls_from_markup parses natively.

Compliance: outputs land in the capture pipeline for the CONDITIONAL
Claude-teacher corpus (scratchpad/claude_corpus/, DO_NOT_TRAIN marker).
No training on them until written Anthropic approval (standing rule).

Usage:
    .venv/bin/python scratchpad/claude_teacher/claude_shim.py \
        --port 8114 [--model opus] [--claude-bin claude] [--max-wait-s 7200]

Notes:
  * Requests are serialized (one CLI call at a time) — deliberate, to keep
    subscription usage observable and paced.
  * On a rate_limit block the shim sleeps until the reported reset (capped
    by --max-wait-s) and retries once — the harness sees latency, not error.
  * Sampling params from the harness are ignored (CLI controls decoding);
    recorded traces keep whatever the harness *sent*, so downstream tooling
    is unaffected.
"""
from __future__ import annotations

import argparse
import http.server
import json
import re
import subprocess
import threading
import time
import uuid

# mirror tool_agent.py's markup grammar exactly — the shim converts Claude's
# markup reply into real OpenAI tool_calls so captured traces are shape-
# identical to the K3 corpus (harvest_sft keys on message.tool_calls)
_BLOCK_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE)
_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", flags=re.DOTALL | re.IGNORECASE)


def markup_to_tool_calls(text: str) -> tuple[str, list]:
    """(stripped_content, openai_tool_calls) from a markup-format reply."""
    calls = []
    for match in _BLOCK_RE.finditer(text):
        name = match.group(1).strip()
        arguments = {p.strip(): v for p, v in _PARAM_RE.findall(match.group(2) or "")}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    return _BLOCK_RE.sub("", text).strip(), calls

FORMAT_INSTRUCTION = """
You are the acting player in the game transcript above: respond with the assistant's NEXT TURN only.

Output contract (strict):
- Optionally start with short update lines (`World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, `Cross-level notes:`).
- Then emit EXACTLY ONE python tool call in this literal format (no markdown fences, no other tool syntax, nothing after it):
<tool_call><function=python><parameter=code>
# your python code here
</parameter></function></tool_call>
- The code runs in the game sandbox described in the system prompt (globals like current_frame, previous_frame, history, transitions, valid_actions, and action(actions) are available there).
- Do not use any tools of your own; your entire reply is plain text in the format above.
""".strip()


def _render_tool_calls(tool_calls: list) -> str:
    parts = []
    for call in tool_calls or []:
        fn = (call or {}).get("function", {})
        name = fn.get("name", "python")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        body = "".join(
            f"<parameter={key}>\n{value}\n</parameter>" for key, value in args.items()
        )
        parts.append(f"<tool_call><function={name}>{body}</function></tool_call>")
    return "\n".join(parts)


def build_claude_message(openai_messages: list) -> tuple[str, list]:
    """(system_prompt, content_blocks) — one flattened user message."""
    system_prompt = ""
    blocks: list = []

    def text(t: str) -> None:
        if t:
            blocks.append({"type": "text", "text": t})

    for message in openai_messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            system_prompt = content if isinstance(content, str) else json.dumps(content)
            continue
        header = {"user": "== GAME STATE (user) ==", "assistant": "== YOUR PREVIOUS TURN (assistant) ==",
                  "tool": "== TOOL RESULT (python) =="}.get(role, f"== {role} ==")
        text(f"\n{header}")
        if isinstance(content, str):
            text(content)
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    text(part.get("text", ""))
                elif ptype == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:image/"):
                        media, _, data = url.partition(";base64,")
                        blocks.append({"type": "image", "source": {
                            "type": "base64",
                            "media_type": media.split(":", 1)[1],
                            "data": data}})
                    else:
                        text(f"[external image: {url}]")
        if role == "assistant" and message.get("tool_calls"):
            text(_render_tool_calls(message["tool_calls"]))

    text("\n" + FORMAT_INSTRUCTION)
    return system_prompt, blocks


class ShimState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        self.calls = 0
        self.total_cost = 0.0


def call_claude(state: ShimState, system_prompt: str, blocks: list) -> dict:
    payload = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}}) + "\n"
    cmd = [state.args.claude_bin, "-p",
           "--input-format", "stream-json",
           "--output-format", "stream-json",
           "--verbose", "--max-turns", "1",
           "--tools", ""]
    if state.args.model:
        cmd += ["--model", state.args.model]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]

    last_error = ""
    for attempt in range(1, 5):
        proc = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                              timeout=state.args.call_timeout_s)
        result_event = None
        blocked_until = None
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "result":
                result_event = event
            if event.get("type") == "rate_limit_event":
                info = event.get("rate_limit_info") or {}
                if str(info.get("status", "")).startswith(("blocked", "rejected", "queued")):
                    blocked_until = info.get("resetsAt")
        if result_event and not result_event.get("is_error"):
            return result_event
        last_error = (f"rc={proc.returncode} subtype={(result_event or {}).get('subtype')} "
                      f"errors={(result_event or {}).get('errors')}")
        with open("/tmp/claude_shim_failures.log", "a") as debug:
            debug.write(f"--- {time.strftime('%F %T')} attempt {attempt} {last_error}\n"
                        f"stderr: {proc.stderr[-1500:]}\nstdout tail: {proc.stdout[-1500:]}\n")
        if blocked_until:
            wait = min(max(30.0, float(blocked_until) - time.time() + 60.0),
                       float(state.args.max_wait_s))
            print(f"[shim] rate-limited; sleeping {wait:.0f}s until reset", flush=True)
            time.sleep(wait)
        else:
            wait = 10.0 * attempt
            print(f"[shim] transient CLI failure ({last_error}); retry in {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"claude CLI failed after retries: {last_error}")


def make_handler(state: ShimState):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json(200, {"data": [{"id": state.args.served_name}]})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                request = json.loads(raw.decode())
                system_prompt, blocks = build_claude_message(request.get("messages", []))
                with state.lock:
                    started = time.time()
                    event = call_claude(state, system_prompt, blocks)
                    state.calls += 1
                    state.total_cost += float(event.get("total_cost_usd") or 0.0)
                usage = event.get("usage") or {}
                raw = str(event.get("result", ""))
                content, tool_calls = markup_to_tool_calls(raw)
                message: dict = {"role": "assistant", "content": content}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                elapsed = time.time() - started
                print(f"[shim] call {state.calls}: {elapsed:.0f}s, "
                      f"out={usage.get('output_tokens')}, tool_calls={len(tool_calls)}, "
                      f"cum_cost=${state.total_cost:.2f}", flush=True)
                self._json(200, {
                    "id": f"shim-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": state.args.served_name,
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }],
                    "usage": {
                        "prompt_tokens": int(usage.get("input_tokens") or 0)
                        + int(usage.get("cache_read_input_tokens") or 0)
                        + int(usage.get("cache_creation_input_tokens") or 0),
                        "completion_tokens": int(usage.get("output_tokens") or 0),
                    },
                })
            except Exception as exc:  # noqa: BLE001 — surface as OpenAI-style error
                print(f"[shim] ERROR: {exc}", flush=True)
                self._json(500, {"error": {"message": str(exc), "type": "shim_error"}})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8114)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--served-name", default="claude-teacher")
    ap.add_argument("--call-timeout-s", type=float, default=900.0)
    ap.add_argument("--max-wait-s", type=float, default=7200.0)
    args = ap.parse_args()
    state = ShimState(args)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"[shim] listening on http://127.0.0.1:{args.port}/v1 (model={args.model})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
