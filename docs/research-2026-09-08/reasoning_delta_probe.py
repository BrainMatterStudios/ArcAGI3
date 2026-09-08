#!/usr/bin/env python3
"""Does prior-turn reasoning reach the model? Offline read from a rig wave.

Within one analyze() turn, consecutive requests differ by exactly one assistant
message (reasoning + content + tool call) and one tool message. The server's
usage.prompt_tokens (shim record) delta must equal the rendered tokens of those
two messages under the model's chat template. We tokenize the transcript's exact
[THINKING] text (R), the assistant content + tool-call markup (C) and the tool
result display (T, approximate: display, not the JSON sent) with the real
tokenizer and regress D = prompt_{i+1} - prompt_i on R.

  slope ~1  -> reasoning IS in the rendered prompt (vLLM maps reasoning ->
               reasoning_content; template keeps all <think> blocks)
  slope ~0  -> reasoning is dropped
"""
import json
import re
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

SCRATCH = Path(__file__).resolve().parent   # needs tokenizer.json beside it: huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4 @ 7b71922 (12.8 MB, not committed)
WAVE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/ahmed/Documents/ArcAGI3/offkaggle/results/20260908T0752-keith_probe/20260908-095256-regime-keith_probe")
tok = Tokenizer.from_file(str(SCRATCH / "tokenizer.json"))


def ntok(s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False).ids)


HEADER_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$", re.M)
SECTION_RE = re.compile(r"^\[([A-Z][A-Z :a-z_-]*)\]\n", re.M)


def parse_turn(body: str):
    """-> list of calls; each call = dict(reasoning, content, tool_calls[markup], tool_results[display])"""
    secs = list(SECTION_RE.finditer(body))
    items = []
    for i, m in enumerate(secs):
        end = secs[i + 1].start() if i + 1 < len(secs) else len(body)
        items.append((m.group(1), body[m.end():end].strip("\n")))
    calls = []
    cur = None
    for label, text in items:
        if label == "MODEL RESPONSE META":
            cur = {"reasoning": "", "content": "", "tool_calls": [], "tool_results": [], "meta": text}
            calls.append(cur)
        elif cur is None:
            continue
        elif label == "THINKING":
            cur["reasoning"] = text
        elif label == "ASSISTANT":
            cur["content"] = text
        elif label.startswith("TOOL CALL: "):
            # probe markers live inside this section; strip them
            text = text.split("\n[HARNESS PROBE]", 1)[0]
            cur["tool_calls"].append(text)
        elif label.startswith("TOOL RESULT: "):
            cur["tool_results"].append(text)
    return calls


def render_assistant(call, with_reasoning: bool) -> str:
    reasoning = call["reasoning"].strip() if with_reasoning else ""
    content = call["content"].strip()
    s = "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content
    for k, tc in enumerate(call["tool_calls"]):
        if k == 0:
            s += ("\n\n" if content else "") + tc
        else:
            s += "\n" + tc
    return s + "<|im_end|>\n"


def render_tool(results) -> str:
    s = "<|im_start|>user"
    for r in results:
        s += "\n<tool_response>\n" + r + "\n</tool_response>"
    return s + "<|im_end|>\n"


shim = {}
for line in (WAVE / "requests_shim.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("status") == 200 and r.get("prompt_tokens") is not None:
        shim.setdefault(r["run_stem"], []).append(r)
for v in shim.values():
    v.sort(key=lambda r: r["t"])

rows = []
skipped = {}
for path in sorted((WAVE / "transcripts").glob("*_p*.txt")):
    stem = path.stem
    text = path.read_text(encoding="utf-8", errors="replace")
    heads = list(HEADER_RE.finditer(text))
    turns = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        turns.append(parse_turn(text[m.end():end]))
    calls_flat = [(ti, c) for ti, t in enumerate(turns) for c in t]
    recs = shim.get(stem, [])
    if len(recs) != len(calls_flat):
        skipped[stem] = (len(recs), len(calls_flat))
        continue
    for k in range(len(calls_flat) - 1):
        ti, c = calls_flat[k]
        tj, _ = calls_flat[k + 1]
        if ti != tj:
            continue                                   # a new turn adds a user message (+image)
        a, b = recs[k], recs[k + 1]
        if b["n_messages"] != a["n_messages"] + 2:
            continue                                   # trimmed or multi-tool: skip
        if not c["tool_calls"] or not c["tool_results"]:
            continue
        D = b["prompt_tokens"] - a["prompt_tokens"]
        R = ntok(c["reasoning"].strip()) if c["reasoning"].strip() else 0
        C = ntok(render_assistant(c, with_reasoning=False))
        T = ntok(render_tool(c["tool_results"]))
        rows.append({"stem": stem, "k": k, "D": D, "R": R, "C": C, "T": T, "resid_with": D - (C + R + T), "resid_without": D - (C + T)})

print(f"wave {WAVE.name}: {len(rows)} within-turn pairs from {len(set(r['stem'] for r in rows))} runs; "
      f"skipped runs (shim!=transcript calls): {skipped}")
big = [r for r in rows if r["R"] >= 300]
for name, sel in (("all pairs", rows), ("pairs with R >= 300 tokens", big)):
    if not sel:
        continue
    rw = [r["resid_with"] for r in sel]
    ro = [r["resid_without"] for r in sel]
    print(f"\n{name}: n={len(sel)}  median R={statistics.median(r['R'] for r in sel):.0f}  median T={statistics.median(r['T'] for r in sel):.0f}")
    print(f"  D - (C+R+T)  [reasoning INCLUDED hypothesis]: median {statistics.median(rw):+.0f}  mean {statistics.mean(rw):+.0f}  "
          f"IQR [{statistics.quantiles(rw, n=4)[0]:+.0f}, {statistics.quantiles(rw, n=4)[2]:+.0f}]")
    print(f"  D - (C+T)    [reasoning DROPPED hypothesis]:  median {statistics.median(ro):+.0f}  mean {statistics.mean(ro):+.0f}  "
          f"IQR [{statistics.quantiles(ro, n=4)[0]:+.0f}, {statistics.quantiles(ro, n=4)[2]:+.0f}]")
    # slope of D-(C+T) on R
    xs = [r["R"] for r in sel]
    ys = [r["D"] - r["C"] - r["T"] for r in sel]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    print(f"  slope of (D-C-T) on R: {slope:.3f}  (1 = included, 0 = dropped)")
print("\nsample rows (D, R, C, T, resid_with, resid_without):")
for r in sorted(rows, key=lambda r: -r["R"])[:8]:
    print(f"  {r['stem']:<22} k={r['k']:<3} D={r['D']:<6} R={r['R']:<5} C={r['C']:<5} T={r['T']:<5} with={r['resid_with']:+6d} without={r['resid_without']:+6d}")

# ---- cross-turn pairs: the assistant message is now BEFORE the new user message ----------------
# D = C + R + T + U (new user prompt text) + image tokens + template overhead
USER_RE = re.compile(r"^\[USER PROMPT\]\n(.*?)(?=\n\n\[|\Z)", re.S | re.M)
cross = []
for path in sorted((WAVE / "transcripts").glob("*_p*.txt")):
    stem = path.stem
    text = path.read_text(encoding="utf-8", errors="replace")
    heads = list(HEADER_RE.finditer(text))
    turns, users = [], []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end]
        turns.append(parse_turn(body))
        um = USER_RE.search(body)
        users.append(um.group(1).strip() if um else "")
    calls_flat = [(ti, c) for ti, t in enumerate(turns) for c in t]
    recs = shim.get(stem, [])
    if len(recs) != len(calls_flat):
        continue
    for k in range(len(calls_flat) - 1):
        ti, c = calls_flat[k]
        tj, _ = calls_flat[k + 1]
        if tj != ti + 1:
            continue
        a, b = recs[k], recs[k + 1]
        if b["n_messages"] != a["n_messages"] + 3 or not c["tool_calls"] or len(c["tool_results"]) != 1:
            continue
        D = b["prompt_tokens"] - a["prompt_tokens"]
        R = ntok(c["reasoning"].strip()) if c["reasoning"].strip() else 0
        C = ntok(render_assistant(c, with_reasoning=False))
        T = ntok(render_tool(c["tool_results"]))
        U = ntok("<|im_start|>user\n" + users[tj] + "\n\nCurrent grid image:<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n")
        cross.append({"stem": stem, "D": D, "R": R, "C": C, "T": T, "U": U, "with": D - (C + R + T + U), "without": D - (C + T + U)})
print(f"\nCROSS-TURN pairs (assistant reasoning now precedes the newest user message): n={len(cross)}")
if cross:
    for lab, key in (("INCLUDED", "with"), ("DROPPED", "without")):
        v = [r[key] for r in cross]
        q = statistics.quantiles(v, n=4)
        print(f"  D - (C+T+U{'+R' if key == 'with' else ''}) [{lab}]: median {statistics.median(v):+.0f} mean {statistics.mean(v):+.0f} IQR [{q[0]:+.0f}, {q[2]:+.0f}]"
              f"   (expected residual if hypothesis holds: ~+63 image-pad tokens + ~30 overhead)")
    xs = [r["R"] for r in cross]; ys = [r["D"] - r["C"] - r["T"] - r["U"] for r in cross]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    print(f"  slope of (D-C-T-U) on R: {sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx:.3f}")
