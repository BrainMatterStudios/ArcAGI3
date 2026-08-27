"""One-call transport smoke: exercise Tycho's openai chat backend against our Modal vLLM.

Verifies: bearer auth, native tool_calls parsing (qwen3_coder parser), enable_thinking,
usage accounting, and the local sampling patch. Text-only (the 27B has no vision tower).
"""
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tycho"))
from tycho.serving.llm_client import LLMConfig, chat_tools

cfg = LLMConfig(
    model="vrfai/Qwen3.6-27B-FP8",
    backend="openai",
    base_url="https://a-m-mobasher--arc3-vllm-serve.modal.run/v1",
    api_key=open(os.path.expanduser("~/.arc3_vllm_token")).read().strip(),
    reasoning_effort="medium",
)

tools = [{
    "name": "report_transport_ok",
    "description": "Report that text and tool calling were received.",
    "schema": {
        "type": "object",
        "properties": {"echo": {"type": "string", "description": "repeat the code word"}},
        "required": ["echo"],
    },
}]
history = [{"role": "user", "content":
            "The code word is 'kepler'. Call report_transport_ok with echo set to the code word."}]

t0 = time.time()
reply = chat_tools(history, tools, cfg, system="You are a precise assistant.",
                   max_tokens=4096, timeout=1500, effort="medium", call_type="smoke")
dt = time.time() - t0
print(json.dumps({
    "latency_s": round(dt, 1),
    "text": (reply.get("text") or "")[:200],
    "reasoning_head": (reply.get("reasoning") or "")[:200],
    "tool_calls": reply.get("tool_calls"),
    "stop": reply.get("stop"),
    "usage": reply.get("usage"),
}, indent=2))
ok = any(c["name"] == "report_transport_ok" and c["input"].get("echo", "").lower() == "kepler"
         for c in reply.get("tool_calls") or [])
print("SMOKE", "PASS" if ok else "FAIL")
