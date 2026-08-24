#!/usr/bin/env python3
"""Stage 3 — serve-replay of the judge prompt pack (one command once a serve
window opens). NO part of answer_key.jsonl is ever sent to the model; it is
read only after all completions return, to compute the pre-registered metrics.

Pre-registered protocol (docs/RESEARCH-2026-08-23-searchcore-and-multirole.md §B
+ task order 2026-08-24): 3 samples/prompt, temperature 1.0, effort medium.

Pre-registered metrics (majority verdict over the 3 samples per case):
  FLAG               = P(REJECT | deserves_rejection=True)          [want high]
  RESCUE             = among deserving cases with known mechanic keywords: fraction
                       where >=1 sampled REJECT ruling proposes a hypothesis matching
                       the true mechanic class (keyword match, case-insensitive)
  CONTROL-false-flag = P(REJECT | kind=control)                     [want low]
                       (also reported over ALL deserves_rejection=False cases)
  DIVERSITY          = mean pairwise Jaccard similarity of the 3 hypotheses within
                       each REJECT sample (want low; structurally different), plus
                       the fraction of REJECT samples with all pairwise Jaccard < 0.5

Usage:
  .venv/bin/python run_stage3.py --endpoint http://HOST:PORT/v1 --model MODEL \
      [--samples 3] [--temperature 1.0] [--effort medium] \
      [--effort-field reasoning_effort|chat_template_kwargs|none] [--max-tokens 2048] \
      [--parallel 1]
API key: --api-key or OPENAI_API_KEY env (many local serves accept any string).
--parallel N runs N cases concurrently (protocol-neutral: same prompts, same
samples, same params; only wall-clock changes — added for the GPU serve probe).
Outputs: stage3_raw.jsonl (every sample), stage3_metrics.json (+ printed table).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def call_chat(endpoint, api_key, payload, timeout=300):
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(KEEP|REJECT)", re.I | re.M)
HYP_RE = re.compile(r"^\s*H([123])\s*:\s*(.+)$", re.M)


def parse_ruling(text: str):
    m = VERDICT_RE.search(text or "")
    verdict = m.group(1).upper() if m else None
    hyps = [h.strip() for _, h in HYP_RE.findall(text or "")]
    return verdict, hyps


def toks(s: str):
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if A | B else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL (…/v1)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--effort-field", default="reasoning_effort",
                    choices=["reasoning_effort", "chat_template_kwargs", "none"])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N cases")
    ap.add_argument("--parallel", type=int, default=1,
                    help="cases run concurrently (protocol-neutral; wall-clock only)")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(os.path.join(HERE, "judge_prompts.jsonl"))]
    if args.limit:
        prompts = prompts[: args.limit]

    raw_path = os.path.join(HERE, "stage3_raw.jsonl")
    raw_f = open(raw_path, "w")
    raw_lock = threading.Lock()
    done_count = [0]

    def run_case(p):
        samples = []
        for s in range(args.samples):
            payload = {
                "model": args.model,
                "messages": p["messages"],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if args.effort_field == "reasoning_effort":
                payload["reasoning_effort"] = args.effort
            elif args.effort_field == "chat_template_kwargs":
                payload["chat_template_kwargs"] = {"reasoning_effort": args.effort}
            t0 = time.time()
            try:
                resp = call_chat(args.endpoint, args.api_key, payload)
                text = resp["choices"][0]["message"].get("content") or ""
            except Exception as e:  # keep going; a dead sample is data
                resp, text = {"error": str(e)}, ""
            verdict, hyps = parse_ruling(text)
            samples.append({"verdict": verdict, "hypotheses": hyps, "text": text})
            with raw_lock:
                raw_f.write(json.dumps({"id": p["id"], "sample": s, "verdict": verdict,
                                        "hypotheses": hyps, "text": text,
                                        "seconds": round(time.time() - t0, 1)}) + "\n")
                raw_f.flush()
        rej = sum(1 for smp in samples if smp["verdict"] == "REJECT")
        with raw_lock:
            done_count[0] += 1
            print(f"[{done_count[0]}/{len(prompts)}] {p['id']}: {rej}/{len(samples)} REJECT",
                  file=sys.stderr)
        return p["id"], samples

    workers = max(1, args.parallel)
    if workers == 1:
        results = dict(run_case(p) for p in prompts)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = dict(pool.map(run_case, prompts))
    raw_f.close()

    # ---- metrics: the answer key is opened ONLY NOW, after all completions ----
    key_rows = [json.loads(l) for l in open(os.path.join(HERE, "answer_key.jsonl"))]
    key = {k["id"]: k for k in key_rows if "id" in k}

    def majority_reject(cid):
        ss = results.get(cid, [])
        v = [s["verdict"] for s in ss if s["verdict"]]
        return v.count("REJECT") > len(ss) / 2 if ss else None

    deserving = [k for k in key.values() if k["deserves_rejection"] and k["id"] in results]
    controls = [k for k in key.values() if k["kind"] == "control" and k["id"] in results]
    keeps_all = [k for k in key.values() if not k["deserves_rejection"] and k["id"] in results]

    flag_hits = [k["id"] for k in deserving if majority_reject(k["id"])]
    ctrl_flags = [k["id"] for k in controls if majority_reject(k["id"])]
    keep_flags = [k["id"] for k in keeps_all if majority_reject(k["id"])]

    rescuable = [k for k in deserving if k["match_keywords"]]
    rescued = []
    for k in rescuable:
        ok = False
        for s in results[k["id"]]:
            if s["verdict"] == "REJECT":
                blob = " ".join(s["hypotheses"]).lower()
                if any(kw.lower() in blob for kw in k["match_keywords"]):
                    ok = True
        if ok:
            rescued.append(k["id"])

    sims, diverse = [], 0
    n_rej_samples = 0
    for cid, ss in results.items():
        for s in ss:
            if s["verdict"] == "REJECT" and len(s["hypotheses"]) >= 2:
                n_rej_samples += 1
                pairs = [(a, b) for i, a in enumerate(s["hypotheses"])
                         for b in s["hypotheses"][i + 1:]]
                pj = [jaccard(a, b) for a, b in pairs]
                sims.extend(pj)
                if all(x < 0.5 for x in pj):
                    diverse += 1

    metrics = {
        "config": {k: v for k, v in vars(args).items() if k != "api_key"},
        "n_cases": len(results),
        "FLAG": {"num": len(flag_hits), "den": len(deserving),
                 "rate": len(flag_hits) / len(deserving) if deserving else None},
        "RESCUE": {"num": len(rescued), "den": len(rescuable),
                   "rate": len(rescued) / len(rescuable) if rescuable else None,
                   "rescued": rescued},
        "CONTROL_false_flag": {"num": len(ctrl_flags), "den": len(controls),
                               "rate": len(ctrl_flags) / len(controls) if controls else None,
                               "flagged": ctrl_flags},
        "all_keep_false_flag": {"num": len(keep_flags), "den": len(keeps_all),
                                "rate": len(keep_flags) / len(keeps_all) if keeps_all else None},
        "DIVERSITY": {"mean_pairwise_jaccard": sum(sims) / len(sims) if sims else None,
                      "reject_samples": n_rej_samples,
                      "all_pairs_below_0.5": diverse,
                      "diverse_rate": diverse / n_rej_samples if n_rej_samples else None},
        "missed_deserving": [k["id"] for k in deserving if k["id"] not in flag_hits],
    }
    with open(os.path.join(HERE, "stage3_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=1))
    print(f"wrote stage3_raw.jsonl + stage3_metrics.json")


if __name__ == "__main__":
    main()
