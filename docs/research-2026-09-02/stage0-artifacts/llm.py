"""Minimal OpenAI-compatible chat client for the Modal vLLM endpoint. Never prints the token."""
import json, os, time, requests

BASE = "https://a-m-mobasher--arc3-vllm38-serve.modal.run/v1"
MODEL = "Qwen/Qwen3.8-27B-FP8"


def _token():
    t = os.environ.get("ARC3_VLLM_TOKEN")
    if not t:
        t = open(os.path.expanduser("~/.config/arc3/vllm_token")).read().strip()
        os.environ["ARC3_VLLM_TOKEN"] = t
    return t


def _headers():
    return {"Authorization": "Bearer " + _token(), "Content-Type": "application/json"}


def wait_for_model(cap_s=20 * 60, every_s=20, log=print):
    t0 = time.time()
    last = None
    while time.time() - t0 < cap_s:
        try:
            r = requests.get(BASE + "/models", headers=_headers(), timeout=60)
            if r.status_code == 200:
                ids = [m.get("id") for m in r.json().get("data", [])]
                if MODEL in ids:
                    log(f"model ready after {time.time()-t0:.0f}s: {ids}")
                    return True
                last = f"200 but ids={ids}"
            else:
                last = f"HTTP {r.status_code}: {r.text[:120]}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
        log(f"  waiting ({time.time()-t0:.0f}s): {last}")
        time.sleep(every_s)
    log(f"GAVE UP after {cap_s}s: {last}")
    return False


def tokenize_count(text):
    """Token count via vLLM /tokenize (falls back to chars/3 if unavailable)."""
    try:
        r = requests.post(BASE.rsplit("/v1", 1)[0] + "/tokenize", headers=_headers(),
                          json={"model": MODEL, "prompt": text}, timeout=120)
        if r.status_code == 200:
            return int(r.json()["count"]), "vllm"
    except Exception:
        pass
    return len(text) // 3, "approx"


def messages_token_count(messages):
    """Prompt token count for a chat request (chat template applied server-side); fallback = sum of parts + 200."""
    try:
        r = requests.post(BASE.rsplit("/v1", 1)[0] + "/tokenize", headers=_headers(),
                          json={"model": MODEL, "messages": messages, "add_generation_prompt": True,
                                "chat_template_kwargs": {"enable_thinking": True}}, timeout=120)
        if r.status_code == 200:
            return int(r.json()["count"]), "vllm-chat"
    except Exception:
        pass
    n = sum(tokenize_count(m["content"])[0] for m in messages)
    return n + 200, "sum+200"


def chat(messages, max_tokens=16000, temperature=0.6, top_p=0.95, timeout=3600, stall_s=600):
    """Streaming chat completion (SSE). Modal kills non-streaming HTTP requests at ~300 s, so we stream and
    accumulate; usage comes from the final chunk (stream_options.include_usage)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}}
    t0 = time.time()
    content, reasoning, finish, usage = [], [], None, {}
    with requests.post(BASE + "/chat/completions", headers=_headers(), json=body, stream=True,
                       timeout=(60, stall_s)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        for raw in r.iter_lines(decode_unicode=True):
            if time.time() - t0 > timeout:
                raise RuntimeError(f"client timeout after {timeout}s")
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    reasoning.append(rc)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    dt = time.time() - t0
    return {"content": "".join(content), "reasoning": "".join(reasoning), "finish_reason": finish,
            "usage": usage, "seconds": dt, "streamed": True}


if __name__ == "__main__":
    wait_for_model()


CTX = 65536


def _tok_messages(messages):
    r = requests.post(BASE.rsplit("/v1", 1)[0] + "/tokenize", headers=_headers(),
                      json={"model": MODEL, "messages": messages, "add_generation_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": True}}, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]


def _tok_text(text):
    r = requests.post(BASE.rsplit("/v1", 1)[0] + "/tokenize", headers=_headers(),
                      json={"model": MODEL, "prompt": text, "add_special_tokens": False}, timeout=120)
    r.raise_for_status()
    return r.json()["tokens"]


def chat_chained(messages, max_tokens=40000, seg_tokens=10000, temperature=0.6, top_p=0.95, timeout=290, log=None):
    """One logical generation of up to max_tokens, produced as a chain of /v1/completions segments of
    <= seg_tokens each (the Modal endpoint kills any HTTP request at ~300 s). The chat template is applied
    server-side via /tokenize (generation prompt ends with '<think>\\n'); each segment's output tokens are
    appended to the prompt ids and generation continues until EOS or the budget is spent."""
    t0 = time.time()
    ids = _tok_messages(messages)
    prompt_tokens = len(ids)
    text, total, segments, finish = "", 0, [], None
    while True:
        rem = max_tokens - total
        seg = min(seg_tokens, rem, CTX - len(ids) - 16)
        if seg <= 0:
            finish = "length"
            break
        body = {"model": MODEL, "prompt": ids, "max_tokens": seg, "temperature": temperature, "top_p": top_p,
                "return_token_ids": True}
        ts = time.time()
        r = requests.post(BASE + "/completions", headers=_headers(), json=body, timeout=(60, timeout + 30))
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        ch = j["choices"][0]
        seg_text = ch.get("text") or ""
        u = j.get("usage") or {}
        n = u.get("completion_tokens") or 0
        total += n
        text += seg_text
        segments.append({"tokens": n, "seconds": round(time.time() - ts, 1), "finish": ch.get("finish_reason")})
        if log:
            log(f"    segment {len(segments)}: {n} tokens in {time.time()-ts:.0f}s finish={ch.get('finish_reason')}")
        if ch.get("finish_reason") == "stop":
            finish = "stop"
            break
        new_ids = ch.get("token_ids")
        if not new_ids:
            new_ids = _tok_text(seg_text)
        ids = ids + list(new_ids)
        if total >= max_tokens:
            finish = "length"
            break
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
    else:
        reasoning, content = text, ""
    return {"content": content, "reasoning": reasoning, "finish_reason": finish,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": total},
            "seconds": time.time() - t0, "segments": segments, "chained": True}
