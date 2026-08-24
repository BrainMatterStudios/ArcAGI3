#!/usr/bin/env python3
"""ROUND 2 serve-replay + scorer for the bankruptcy-judge falsifier.

Runs the four round-2 arms (V1 menu / V2 prediction / V3 rich evidence / R1X
round-1 replay) against an OpenAI-compatible endpoint, then scores against
round2_key.jsonl — which is opened ONLY after every completion has returned;
no part of it ever enters a model context.

PRE-REGISTERED BARS (task order round 2, 2026-08-24):
  V1 MENU_PICK        PASS >= 60%   KILL < 40%     (chance 25%)
  V2 PRED_DIVERGENCE  PASS >= 60%   (else FAIL)
  V3 FLAG_RICH        REVIVED >= 3/6   DEAD <= 1/6   (2/6 = AMBER)
  R1X (diagnostic)    FLAG_R1X >= 50% reclassifies the round-1 kill as an
                      instrument artifact (truncation), not a capability verdict.

Instrument fix vs round 1: max_tokens default 8192 (round 1's 2048 produced
87/114 empty completions — reasoning ate the budget); finish_reason and
reasoning length are recorded per sample.

Usage:
  python run_round2.py --endpoint http://HOST:PORT/v1 --model MODEL \
      [--samples 3] [--temperature 1.0] [--effort medium] \
      [--effort-field chat_template_kwargs] [--max-tokens 8192] [--parallel 8]
Outputs (next to this script): round2_raw.jsonl, round2_results.json.
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


def call_chat(endpoint, api_key, payload, timeout=900):
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------- parsers
VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*(KEEP|REJECT)", re.I | re.M)
HYP_RE = re.compile(r"^\s*H([123])\s*:\s*(.+)$", re.M)
ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*([ABCD])\b", re.I | re.M)
PRED_RE = re.compile(
    r"^\s*P([123])\s*:\s*band\s*=\s*(ZERO|SMALL|MEDIUM|LARGE)\b"
    r".{0,40}?zone\s*=\s*(NW|NE|SW|SE|NONE|N|S|E|W|C)\b",
    re.I | re.M)
CITE_RE = re.compile(r"#(\d+)")


def parse_generic(text: str):
    m = VERDICT_RE.search(text or "")
    verdict = m.group(1).upper() if m else None
    hyps = [h.strip() for _, h in HYP_RE.findall(text or "")]
    cites = [int(x) for x in CITE_RE.findall(text or "")]
    return verdict, hyps, cites


def parse_menu(text: str):
    m = ANSWER_RE.search(text or "")
    return m.group(1).upper() if m else None


def parse_preds(text: str):
    out = {}
    for n, band, zone in PRED_RE.findall(text or ""):
        out[int(n)] = {"band": band.upper(), "zone": zone.upper()}
    return [out.get(i) for i in (1, 2, 3)]


def majority(vals, target):
    """True if > half of the NON-None entries equal target (and any exist)."""
    valid = [v for v in vals if v is not None]
    return bool(valid) and valid.count(target) > len(valid) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--effort-field", default="chat_template_kwargs",
                    choices=["reasoning_effort", "chat_template_kwargs", "none"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in open(os.path.join(HERE, "round2_prompts.jsonl"))]
    if args.limit:
        prompts = prompts[: args.limit]

    raw_path = os.path.join(HERE, "round2_raw.jsonl")
    raw_f = open(raw_path, "w")
    lock = threading.Lock()
    done = [0]

    def run_case(p):
        samples = []
        for s in range(args.samples):
            payload = {"model": args.model, "messages": p["messages"],
                       "temperature": args.temperature, "max_tokens": args.max_tokens}
            if args.effort_field == "reasoning_effort":
                payload["reasoning_effort"] = args.effort
            elif args.effort_field == "chat_template_kwargs":
                payload["chat_template_kwargs"] = {"reasoning_effort": args.effort}
            t0 = time.time()
            finish = reasoning_len = None
            try:
                resp = call_chat(args.endpoint, args.api_key, payload)
                ch = resp["choices"][0]
                text = ch["message"].get("content") or ""
                finish = ch.get("finish_reason")
                reasoning_len = len(ch["message"].get("reasoning_content") or "")
            except Exception as e:
                text = ""
                finish = f"error:{e}"
            samples.append({"text": text, "finish_reason": finish,
                            "reasoning_len": reasoning_len})
            with lock:
                raw_f.write(json.dumps({"id": p["id"], "variant": p["variant"],
                                        "sample": s, "finish_reason": finish,
                                        "reasoning_len": reasoning_len, "text": text,
                                        "seconds": round(time.time() - t0, 1)}) + "\n")
                raw_f.flush()
        with lock:
            done[0] += 1
            ne = sum(1 for x in samples if x["text"].strip())
            print(f"[{done[0]}/{len(prompts)}] {p['id']}: {ne}/{len(samples)} non-empty",
                  file=sys.stderr, flush=True)
        return p["id"], samples

    if args.parallel <= 1:
        results = dict(run_case(p) for p in prompts)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = dict(pool.map(run_case, prompts))
    raw_f.close()

    # ==== the key is opened ONLY NOW, after all completions have returned ====
    key_rows = [json.loads(l) for l in open(os.path.join(HERE, "round2_key.jsonl"))]
    key = {k["id"]: k for k in key_rows if "id" in k}

    # ---------------- instrument health --------------------------------------
    health = {}
    for p in prompts:
        v = p["variant"]
        h = health.setdefault(v, {"samples": 0, "non_empty": 0, "finish_length": 0,
                                  "errors": 0})
        for s in results[p["id"]]:
            h["samples"] += 1
            if s["text"].strip():
                h["non_empty"] += 1
            if s["finish_reason"] == "length":
                h["finish_length"] += 1
            if str(s["finish_reason"] or "").startswith("error"):
                h["errors"] += 1

    # ---------------- V1 MENU_PICK -------------------------------------------
    v1_cases, v1_hits = [], []
    v1_split = {"deserving": [0, 0], "control": [0, 0]}
    for k in key.values():
        if k["variant"] != "V1" or k["id"] not in results:
            continue
        picks = [parse_menu(s["text"]) for s in results[k["id"]]]
        hit = majority(picks, k["correct_letter"])
        v1_cases.append(k["id"])
        if hit:
            v1_hits.append(k["id"])
        bucket = "deserving" if k["deserves_rejection"] else "control"
        v1_split[bucket][1] += 1
        v1_split[bucket][0] += int(hit)
    menu_pick = len(v1_hits) / len(v1_cases) if v1_cases else None

    # ---------------- V2 PRED_DIVERGENCE -------------------------------------
    v2_detail, v2_diverging = {}, []
    for k in key.values():
        if k["variant"] != "V2" or k["id"] not in results:
            continue
        sample_flags = []
        for s in results[k["id"]]:
            preds = parse_preds(s["text"])
            if any(p is None for p in preds):
                sample_flags.append(None)  # invalid sample
                continue
            mism = 0
            for p, t in zip(preds, k["truth"]):
                band_ok = p["band"] == t["band"]
                zone_ok = (p["zone"] == "NONE") if t["band"] == "ZERO" \
                    else (p["zone"] in t["zones"])
                if not (band_ok and zone_ok):
                    mism += 1
            sample_flags.append(mism >= 2)
        div = majority(sample_flags, True)
        v2_detail[k["id"]] = {"sample_diverges": sample_flags, "case_diverges": div}
        if div:
            v2_diverging.append(k["id"])
    v2_n = len(v2_detail)
    pred_divergence = len(v2_diverging) / v2_n if v2_n else None

    # ---------------- V3 FLAG_RICH -------------------------------------------
    v3_detail, v3_flips = {}, []
    for k in key.values():
        if k["variant"] != "V3" or k["id"] not in results:
            continue
        shown = set(k["shown_transitions"])
        verdicts, cited_reject = [], False
        for s in results[k["id"]]:
            verdict, hyps, cites = parse_generic(s["text"])
            verdicts.append(verdict)
            if verdict == "REJECT" and shown & set(cites):
                cited_reject = True
        flip = majority(verdicts, "REJECT") and cited_reject
        v3_detail[k["id"]] = {"verdicts": verdicts, "cited_reject": cited_reject,
                              "flip": flip}
        if flip:
            v3_flips.append(k["id"])
    v3_n = len(v3_detail)

    # ---------------- R1X (round-1 replay at fixed budget) -------------------
    def r1x_majority_reject(kid):
        return majority([parse_generic(s["text"])[0] for s in results[kid]], "REJECT")

    r1x = [k for k in key.values() if k["variant"] == "R1X" and k["id"] in results]
    r1x_deserving = [k for k in r1x if k["deserves_rejection"]]
    r1x_controls = [k for k in r1x if k["kind"] == "control"]
    r1x_flag = [k["id"] for k in r1x_deserving if r1x_majority_reject(k["id"])]
    r1x_ctrl_flag = [k["id"] for k in r1x_controls if r1x_majority_reject(k["id"])]
    rescuable = [k for k in r1x_deserving if k["match_keywords"]]
    rescued = []
    for k in rescuable:
        for s in results[k["id"]]:
            verdict, hyps, _ = parse_generic(s["text"])
            if verdict == "REJECT":
                blob = " ".join(hyps).lower()
                if any(kw.lower() in blob for kw in k["match_keywords"]):
                    rescued.append(k["id"])
                    break
    flag_r1x = len(r1x_flag) / len(r1x_deserving) if r1x_deserving else None
    rescue_r1x = len(rescued) / len(rescuable) if rescuable else None
    ctrl_ff_r1x = len(r1x_ctrl_flag) / len(r1x_controls) if r1x_controls else None

    # ---------------- verdicts vs pre-registered bars ------------------------
    def v1_bar(r):
        return "NO-DATA" if r is None else "PASS" if r >= 0.60 else \
            "KILL" if r < 0.40 else "AMBER"

    v1_verdict = v1_bar(menu_pick)
    v2_verdict = ("NO-DATA" if pred_divergence is None
                  else "PASS" if pred_divergence >= 0.60 else "FAIL")
    v3_verdict = ("NO-DATA" if not v3_n
                  else "REVIVED" if len(v3_flips) >= 3
                  else "DEAD" if len(v3_flips) <= 1 else "AMBER")
    r1x_verdict = ("NO-DATA" if flag_r1x is None
                   else "INSTRUMENT-ARTIFACT" if flag_r1x >= 0.50
                   else "CAPABILITY-KILL-CONFIRMED")

    results_doc = {
        "probe": "arc3-judge-probe2",
        "protocol": {"samples": args.samples, "temperature": args.temperature,
                     "effort": args.effort, "effort_field": args.effort_field,
                     "max_tokens": args.max_tokens, "parallel": args.parallel,
                     "n_cases": len(results)},
        "instrument_health": health,
        "V1_MENU_PICK": {"rate": menu_pick, "num": len(v1_hits), "den": len(v1_cases),
                         "split": {b: {"num": n, "den": d}
                                   for b, (n, d) in v1_split.items()},
                         "hits": v1_hits,
                         "bar": "pass>=0.60 kill<0.40", "verdict": v1_verdict},
        "V2_PRED_DIVERGENCE": {"rate": pred_divergence, "num": len(v2_diverging),
                               "den": v2_n, "detail": v2_detail,
                               "bar": "pass>=0.60", "verdict": v2_verdict},
        "V3_FLAG_RICH": {"flips": len(v3_flips), "den": v3_n, "flipped": v3_flips,
                         "detail": v3_detail,
                         "bar": "revived>=3/6 dead<=1/6", "verdict": v3_verdict},
        "R1X": {"FLAG": {"rate": flag_r1x, "num": len(r1x_flag),
                         "den": len(r1x_deserving)},
                "RESCUE": {"rate": rescue_r1x, "num": len(rescued),
                           "den": len(rescuable)},
                "CONTROL_false_flag": {"rate": ctrl_ff_r1x, "num": len(r1x_ctrl_flag),
                                       "den": len(r1x_controls),
                                       "flagged": r1x_ctrl_flag},
                "flagged": r1x_flag,
                "bar": "diagnostic: FLAG>=0.50 => round-1 kill was truncation artifact",
                "verdict": r1x_verdict},
    }
    with open(os.path.join(HERE, "round2_results.json"), "w") as f:
        json.dump(results_doc, f, indent=1)

    print("=" * 72)
    print("JUDGE ROUND-2 VERDICT TABLE (pre-registered bars)")
    print(f"HEALTH: " + " | ".join(
        f"{v} non-empty {h['non_empty']}/{h['samples']} len-cut {h['finish_length']} "
        f"err {h['errors']}" for v, h in sorted(health.items())))
    print(f"METRIC V1 MENU_PICK        {menu_pick if menu_pick is not None else 'n/a'} "
          f"({len(v1_hits)}/{len(v1_cases)}; deserving {v1_split['deserving'][0]}/"
          f"{v1_split['deserving'][1]}, controls {v1_split['control'][0]}/"
          f"{v1_split['control'][1]})  bar >=0.60 kill <0.40 -> {v1_verdict}")
    print(f"METRIC V2 PRED_DIVERGENCE  "
          f"{pred_divergence if pred_divergence is not None else 'n/a'} "
          f"({len(v2_diverging)}/{v2_n})  bar >=0.60 -> {v2_verdict}")
    print(f"METRIC V3 FLAG_RICH        {len(v3_flips)}/{v3_n}  "
          f"bar revive>=3 dead<=1 -> {v3_verdict}")
    print(f"METRIC R1X FLAG            "
          f"{flag_r1x if flag_r1x is not None else 'n/a'} "
          f"({len(r1x_flag)}/{len(r1x_deserving)}) RESCUE "
          f"{rescue_r1x if rescue_r1x is not None else 'n/a'} ({len(rescued)}/"
          f"{len(rescuable)}) CTRL-FF "
          f"{ctrl_ff_r1x if ctrl_ff_r1x is not None else 'n/a'} "
          f"({len(r1x_ctrl_flag)}/{len(r1x_controls)}) -> {r1x_verdict}")
    print("wrote round2_raw.jsonl + round2_results.json")


if __name__ == "__main__":
    main()
