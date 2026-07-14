#!/usr/bin/env python3
"""ewm_agent.py — portable local-model ReAct loop that drives the EWM coding brain.

Replaces the Claude subagent / Codex CLI for OFFLINE deployment. Talks to any
OpenAI-compatible /v1/chat/completions endpoint (MLX on-host, or vLLM on Kaggle),
using a robust XML-tag tool protocol (no native tool-calling required) so it works
across weak local models. Dependency-light (stdlib only): urllib + json.

Tools the model can use (exactly ONE per reply, as the LAST thing in the message):
  <bash>shell command</bash>                 run in the workspace (timeout-guarded)
  <write path="rel/or/abs">CONTENT</write>   overwrite a file with CONTENT
  <read path="..."/>                         read a file (returned truncated)
  <done>final summary</done>                 stop

Usage:
  ewm_agent.py --workspace DIR --base-url http://127.0.0.1:1234/v1 --model NAME \
               --port 8879 [--max-turns 120] [--max-ctx-chars 90000] [--obs-cap 1800]
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time, urllib.request

def http_chat(base_url, model, messages, temperature, max_tokens, timeout=600):
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    return msg.get("content") or "", usage

ACTION_RE = re.compile(
    r"<(bash|write|read|done)(?:\s+path=\"([^\"]*)\")?\s*(?:/>|>(.*?)</\1>)",
    re.DOTALL)

def parse_action(text):
    """Return the LAST well-formed action tag (models often think, then act)."""
    matches = list(ACTION_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {"tool": m.group(1), "path": m.group(2), "body": (m.group(3) or "").strip()}

def run_bash(cmd, cwd, timeout=180):
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except Exception as e:
        return f"[bash error: {e}]"

def cap(s, n):
    if len(s) <= n:
        return s
    head = s[: int(n * 0.6)]
    tail = s[-int(n * 0.35):]
    return f"{head}\n...[{len(s)-n} chars elided]...\n{tail}"

SYSTEM = """You are the coding brain of a verification-by-execution game solver. You solve an unknown 64x64 grid game by (1) probing it with a few real actions, (2) writing an EXECUTABLE WORLD MODEL in Python that reproduces the observed frames EXACTLY, (3) VERIFYING it against recorded frames, (4) PLANNING through the verified model and executing the plan. Real game actions are costly (scored by (human_actions/your_actions)^2); simulation is free.

You act by emitting EXACTLY ONE tool call as the LAST thing in every reply, using these XML tags (content goes between the tags, no JSON escaping needed):
  <bash>shell command</bash>            -> runs in your workspace; use for ./g, ./verify, ./plan, cat, ls, python
  <write path="world_model_engine.py">FULL FILE CONTENT</write>   -> overwrites the file
  <read path="client/session/level_01_attempt_01/initial_frame.txt"/>  -> prints the file
  <done>summary of levels solved and status</done>  -> stop
Think briefly BEFORE the tag, then emit the single tag. Never emit two tags.

Game interface (run via <bash>):
  ./g move ACTION1            (ACTION1..4 = up/down/left/right; ACTION5 = interact; ACTION6 = click, add --x N --y N in 0..63; ACTION7 = undo)
  ./g move RESET             (fresh deterministic attempt of the current level; only after >=1 action; use after GAME_OVER or a wasted line)
  ./g status                 (current state)
  ./verify                   (replays recorded attempts through your model+renderer; must PASS)
  ./plan --from-current      (runs your planner from the live state, prints actions to execute)
Frames are 64x64 hex-digit grids (0..F = colors 0..15) at client/session/level_XX_attempt_YY/*_final.txt and initial_frame.txt; *_metadata.json has state/levels_completed/available_actions. Read main_prompt.md for the exact world_model function signatures (world_model_engine, initial_state_reconstruction, state_renderer, planner).

YOU DO NOT NEED TO WRITE SEARCH. `world_model_main_planner.py` is already a WORKING planner that calls a correct BFS from `search_lib.py` over YOUR engine. Focus your effort on the three modeling pieces: (1) `world_model_engine(state, action)` dynamics, (2) `world_model_state_io.py` (initial_state_reconstruction + state_renderer that renders the frame EXACTLY), and (3) if the game uses ACTION6 clicks, edit ONLY the small `candidate_actions(state)` function in world_model_main_planner.py to return the few meaningful click targets (object centroids, buttons, palette swatches) — never all 64x64 cells. Then `./plan --from-current` and `./verify` just work.

CRITICAL — context is LIMITED. Do NOT re-read big frames repeatedly. Persist everything you learn to world_model.md and notes in your model files, so knowledge survives even if old outputs scroll away. Work in small verified steps: probe a little, model it, ./verify until PASS, then ./plan and execute. FIREWALL: infer only from frames; never read game source or look outside your workspace.

Get level 1 solved with a verified model + planner first, then extend the SAME model to deeper levels. When you truly cannot progress, emit <done>."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8879)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--max-ctx-chars", type=int, default=90000)
    ap.add_argument("--obs-cap", type=int, default=4600)  # must exceed one 64x64 frame (4160 chars) or perception breaks
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    ws = os.path.abspath(a.workspace)
    os.environ["GAME_SERVER_URL"] = f"http://127.0.0.1:{a.port}"
    log = open(a.log or os.path.join(ws, "agent.log"), "w")
    def emit(x):
        print(x, flush=True); log.write(x + "\n"); log.flush()

    # seed with the initial observation so a weak model starts grounded
    init_txt = ""
    try:
        d = os.path.join(ws, "client/session/level_01_attempt_01")
        with open(os.path.join(d, "initial_frame.txt")) as f:
            init_txt = f.read()
        with open(os.path.join(d, "initial_metadata.json")) as f:
            meta = json.load(f)
    except Exception:
        meta = {}
    seed = (f"The game session is started at level 1. Initial available_actions={meta.get('available_actions')}, "
            f"win_levels={meta.get('win_levels')}. Initial frame (64x64 hex):\n{init_txt}\n"
            "Begin: probe a few actions, then build+verify your world model. Emit ONE tool tag.")

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": seed}]
    tot_in = tot_out = 0
    solved_hi = 0
    fails = 0
    t0 = time.time()

    def _safe(path):  # firewall: keep read/write inside the workspace
        p = os.path.abspath(path if os.path.isabs(path) else os.path.join(ws, path))
        return p if (p == ws or p.startswith(ws + os.sep)) else None

    for turn in range(1, a.max_turns + 1):
        # context budget: elide oldest tool observations if over budget
        while sum(len(m["content"]) for m in messages) > a.max_ctx_chars and len(messages) > 4:
            for i in range(2, len(messages)):
                if messages[i]["role"] == "user" and not messages[i]["content"].startswith("[elided"):
                    messages[i]["content"] = "[elided older observation]"
                    break
            else:
                break
        try:
            content, usage = http_chat(a.base_url, a.model, messages, a.temperature, a.max_tokens)
            fails = 0
        except Exception as e:
            fails += 1
            emit(f"[turn {turn}] MODEL CALL FAILED ({fails}): {e}")
            if fails >= 5:
                emit("[abort] 5 consecutive model-call failures"); break
            time.sleep(3); continue
        tot_in += usage.get("prompt_tokens", 0); tot_out += usage.get("completion_tokens", 0)
        act = parse_action(content)
        emit(f"\n===== turn {turn} (ctx~{sum(len(m['content']) for m in messages)//4} tok, "
             f"cumul out={tot_out}) =====\n{cap(content, 1200)}")
        # store a bounded copy: a full <write> file body is on disk already, so echoing it in
        # history just bloats context (can overflow the model window) — keep the reasoning + tag head.
        messages.append({"role": "assistant", "content": cap(content, 3000)})
        if act is None:
            messages.append({"role": "user", "content":
                "No valid tool tag found. Reply with EXACTLY ONE tag: <bash>...</bash>, "
                "<write path=\"...\">...</write>, <read path=\"...\"/>, or <done>...</done>."})
            continue
        if act["tool"] == "done":
            emit(f"\n[DONE] {act['body']}"); break
        if act["tool"] == "bash":
            obs = run_bash(act["body"], ws)
        elif act["tool"] == "write":
            p = _safe(act["path"])
            if p is None:
                obs = "[write refused: path is outside the workspace]"
            else:
                try:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as f:
                        f.write(act["body"])
                    obs = f"[wrote {len(act['body'])} chars to {act['path']}]"
                except Exception as e:
                    obs = f"[write error: {e}]"
        elif act["tool"] == "read":
            p = _safe(act["path"])
            if p is None:
                obs = "[read refused: path is outside the workspace]"
            else:
                try:
                    with open(p) as f:
                        obs = f.read()
                except Exception as e:
                    obs = f"[read error: {e}]"
        else:
            obs = "[unknown tool]"
        # track progress from the client session (levels_completed)
        try:
            import glob as _g
            metas = sorted(_g.glob(os.path.join(ws, "client/session/level_*_attempt_*/step_*_metadata.json")))
            for mp in metas[-6:]:
                lc = json.load(open(mp)).get("levels_completed", 0)
                solved_hi = max(solved_hi, int(lc or 0))
        except Exception:
            pass
        messages.append({"role": "user", "content": cap(obs, a.obs_cap) +
                         f"\n[levels_completed so far: {solved_hi}]"})

    dt = time.time() - t0
    emit(f"\n===== END turns_used<= {a.max_turns} | levels_completed_max={solved_hi} | "
         f"in={tot_in} out={tot_out} tok | {dt:.0f}s | {tot_out/max(dt,1):.1f} out-tok/s =====")
    print(json.dumps({"levels_completed_max": solved_hi, "prompt_tokens": tot_in,
                      "completion_tokens": tot_out, "seconds": round(dt, 1)}))

if __name__ == "__main__":
    main()
