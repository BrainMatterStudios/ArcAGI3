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
import argparse, json, os, re, subprocess, sys, time, urllib.error, urllib.request

def http_chat(base_url, model, messages, temperature, max_tokens, timeout=600,
              top_p=None, repetition_penalty=None, stop=None):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    # Without these the loop degenerates: at temp 0.3, top_p 1.0 and no repetition
    # penalty the model re-emits a byte-identical <write> hundreds of times (71-84% of
    # the v3 gate token budget). top_p/repetition_penalty are vLLM sampling params; stop
    # lets us end generation the moment a tool tag closes, so a truncated file can't eat
    # the whole completion.
    if top_p is not None:
        payload["top_p"] = top_p
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if stop:
        payload["stop"] = stop
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                          "Authorization": "Bearer local"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # vLLM returns the REAL reason (context overflow, bad param, ...) in the RESPONSE BODY.
        # Without reading it the caller only sees "HTTP Error 400: Bad Request" and cannot tell
        # why — that blind spot cost 2 of 3 games in gate v2 (2026-07-21).
        try:
            detail = e.read().decode("utf-8", "replace")[:600]
        except Exception:
            detail = "<no body>"
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
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

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\n(.*?)\n```\s*$", re.DOTALL)

def strip_code_fence(body):
    """Remove a wrapping ```lang ... ``` fence models emit around file bodies.

    v3 wrote the fence verbatim into every .py file, so `world_model_engine.py` began
    with '```python' and raised SyntaxError on import — ./verify could never pass, for a
    reason unrelated to the model's reasoning.
    """
    m = _FENCE_RE.match(body)
    return m.group(1) if m else body

# Scaffold files the model must not overwrite. world_model_*.py are the DELIVERABLE and
# stay writable; world_model_main_planner.py is writable too (the model edits its small
# candidate_actions function). Everything here is provided, working infrastructure —
# clobbering search_lib.py is part of how cd82 died in v3.
_SCAFFOLD_LOCKED = {
    "search_lib.py", "verify_world_model.py", "verify_main_planner.py",
    "run_main_planner.py", "run_aux_planner.py", "plan_executor.py",
    "session_tools.py", "script_tools.py", "timeout_tools.py", "game_status.py",
    "g", "verify", "plan",
}

def describe_write(path, body):
    """A MEANINGFUL observation for a write — never the uninformative 'wrote N chars'.

    The old observation carried zero information, so (identical write, 'wrote N chars')
    pairs formed a perfect attractor. For Python files we compile and report OK or the
    exact SyntaxError, which gives the model a real gradient to act on.
    """
    n = len(body)
    if path.endswith(".py"):
        try:
            compile(body, path, "exec")
            return f"[wrote {n} chars to {path}; py-compile OK]"
        except SyntaxError as e:
            return (f"[wrote {n} chars to {path}; SyntaxError line {e.lineno}: {e.msg}. "
                    "Fix it before running anything.]")
    return f"[wrote {n} chars to {path}]"

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
Frames are 64x64 hex-digit grids (0..F = colors 0..15) at client/session/level_XX_attempt_YY/*_final.txt and initial_frame.txt; *_metadata.json has state/levels_completed/available_actions.

EXACT SIGNATURES you must implement (this is the contract ./verify checks — do not guess it):
  world_model_engine.py:  world_model_engine(state, action) -> (new_state, game_status)
     * state is a DICT with an obligatory 'level' key plus your internal representation
     * action is a DICT {"name": "ACTION1".."ACTION7", "x": int, "y": int}  (x,y only for ACTION6)
     * game_status is one of the exact strings "RUNNING", "LEVEL_COMPLETED", "GAME_OVER"
     * model ONE attempt's dynamics only; never load a real frame into the engine
  world_model_state_io.py:  initial_state_reconstruction(...) builds the level's start state;
     state_renderer(state) returns the 64x64 grid that must match the recorded frame EXACTLY
  world_model_main_planner.py:  edit ONLY candidate_actions(state) if the game uses ACTION6 clicks
Infer COMPACT general mechanics; do not hardcode level layouts.
DO NOT define apply_render_overrides — the scaffold already provides a working default. Put
only initial_state_reconstruction and state_renderer in world_model_state_io.py. Fix real
mismatches in your engine/renderer, never with a render override.

YOU DO NOT NEED TO WRITE SEARCH. `world_model_main_planner.py` is already a WORKING planner that calls a correct BFS from `search_lib.py` over YOUR engine. Focus your effort on the three modeling pieces: (1) `world_model_engine(state, action)` dynamics, (2) `world_model_state_io.py` (initial_state_reconstruction + state_renderer that renders the frame EXACTLY), and (3) if the game uses ACTION6 clicks, edit ONLY the small `candidate_actions(state)` function in world_model_main_planner.py to return the few meaningful click targets (object centroids, buttons, palette swatches) — never all 64x64 cells. Then `./plan --from-current` and `./verify` just work.

CRITICAL — context is LIMITED. Do NOT re-read big frames repeatedly. Persist everything you learn to world_model.md and notes in your model files, so knowledge survives even if old outputs scroll away. Work in small verified steps: probe a little, model it, ./verify until PASS, then ./plan and execute. FIREWALL: infer only from frames; never read game source or look outside your workspace.

Get level 1 solved with a verified model + planner first, then extend the SAME model to deeper levels. When you truly cannot progress, emit <done>."""

def _load_fewshot(game, root, ws):
    """Format a solved Opus world model as a worked example, or '' if unavailable.

    Leakage guard: if the workspace we are solving is itself this game, skip — an example
    must come from a DIFFERENT game family than the one being scored.
    """
    if not game:
        return ""
    if game in os.path.basename(os.path.dirname(ws)) or game in ws.split(os.sep)[-3:][0:1]:
        return ""
    base = root or os.path.dirname(os.path.abspath(__file__))
    import glob as _g
    hits = sorted(_g.glob(os.path.join(base, "runs", f"{game}_*", "workspace")))
    hits += sorted(_g.glob(os.path.join(base, f"{game}_*", "workspace")))
    md = eng = None
    for h in hits:
        mp, ep = os.path.join(h, "world_model.md"), os.path.join(h, "world_model_engine.py")
        if os.path.exists(mp) and os.path.exists(ep):
            md, eng = open(mp).read(), open(ep).read()
            break
    if not md:
        return ""
    # Trim the engine so the example teaches structure, not length.
    eng = cap(eng, 6000)
    return (
        f"WORKED EXAMPLE — a solved world model for an UNRELATED game ({game}). This is the "
        "SHAPE and LEVEL OF DETAIL of the deliverable, NOT the mechanics of your game "
        "(yours are different — infer them from YOUR frames). Study how it (a) names each "
        "object by color and role, (b) states the turn order exactly, (c) keeps the engine "
        "a compact general rule with level-specific data isolated.\n\n"
        f"--- world_model.md ---\n{md}\n\n"
        f"--- world_model_engine.py (excerpt) ---\n{eng}\n--- end example ---\n"
        "Now solve YOUR game to this standard.")

def _norm(s):
    """Collapse whitespace so trivially-different repeats still count as repeats."""
    return re.sub(r"\s+", " ", s or "").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8879)
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--max-ctx-chars", type=int, default=90000)
    # REAL token cap. chars/4 under-counts 64x64 hex-digit frames ~3x, so a "ctx~17K tok" estimate
    # was really >65536 and every call 400'd (gate3 lost sb26+cd82 that way). Calibrated at runtime
    # from the API's reported prompt_tokens. Default leaves room for --max-tokens inside a 65536 window.
    ap.add_argument("--max-ctx-tokens", type=int, default=45000)
    # obs-cap must exceed main_prompt.md (13,932 chars) so the model can actually read
    # the contract it is scored against, not just a pointer to it.
    ap.add_argument("--obs-cap", type=int, default=16000)
    # Duck-matched sampling (LOCAL_ANALYZER_TEMPERATURE=0.6, TOP_P=0.95) plus a light
    # repetition penalty. Temp 0.3 + no penalty was the v3 attractor.
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    # A 64x64 hex frame is ~4,300 chars; a state-io file that embeds one does not fit in
    # 4096 tokens. v3 burned 23 consecutive turns hitting exactly 4096 (truncated writes
    # with no closing tag -> "no tool tag" -> rewrite the same file). 16384 fits a full file.
    ap.add_argument("--max-tokens", type=int, default=16384)
    # Opus solved tu93 8/9 and left a complete world_model.md + engine in runs/. Feed one
    # (from a DIFFERENT game than the target, to avoid leakage) as a worked example of the
    # deliverable. This is the highest-information asset we own and was never used.
    ap.add_argument("--fewshot-game", default="tu93",
                    help="game whose Opus solution seeds the worked example; '' disables")
    ap.add_argument("--fewshot-root", default=None,
                    help="dir holding <game>_*/workspace Opus artifacts (default: alongside this script)")
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
    fewshot = _load_fewshot(a.fewshot_game, a.fewshot_root, ws)

    seed = (f"The game session is started at level 1. Initial available_actions={meta.get('available_actions')}, "
            f"win_levels={meta.get('win_levels')}. Initial frame (64x64 hex):\n{init_txt}\n"
            "Begin: probe a few actions, then build+verify your world model. Emit ONE tool tag.")

    messages = [{"role": "system", "content": SYSTEM}]
    if fewshot:
        messages.append({"role": "user", "content": fewshot})
    messages.append({"role": "user", "content": seed})
    tot_in = tot_out = 0
    solved_hi = 0
    fails = 0
    t0 = time.time()

    def _safe(path):  # firewall: keep read/write inside the workspace
        p = os.path.abspath(path if os.path.isabs(path) else os.path.join(ws, path))
        return p if (p == ws or p.startswith(ws + os.sep)) else None

    budget_tok = a.max_ctx_tokens
    chars_per_tok = 4.0        # calibrated from REAL usage after the first successful call
    done_rejects = 0
    # Pin the prefix (system [+ few-shot] + seed) so elision never deletes the contract,
    # the worked example, or the initial frame.
    n_pin = len(messages)
    last_asst = None            # normalized previous assistant message, for repeat detection
    repeat_run = 0
    for turn in range(1, a.max_turns + 1):
        temp = min(1.0, a.temperature + 0.2 * repeat_run)   # escalate out of any attractor
        # Elide oldest observations until the REAL token estimate fits. We calibrate
        # chars_per_tok from the API's reported prompt_tokens rather than assuming 4.0: 64x64
        # hex-grid frames tokenize at ~1-2 chars/token, so the naive estimate read "ctx~17K tok"
        # while the server was actually rejecting >65536 (gate3 lost sb26 + cd82 to this).
        # SLIDING WINDOW: drop the OLDEST turn outright (keep system + seed at [0],[1]).
        # The old version only replaced *user* messages with placeholders and never touched
        # ASSISTANT messages; once every observation was elided it hit its `else: break` and sent
        # the oversized prompt anyway -> HTTP 400 x5 -> abort. That killed tu93+cd82 in gate3 v2
        # (ctx~60282 tok against a 45000 budget). Deleting guarantees the loop converges.
        while len(messages) > n_pin + 2:
            sent_chars = sum(len(m["content"]) for m in messages)
            if sent_chars / max(chars_per_tok, 0.5) <= budget_tok and sent_chars <= a.max_ctx_chars:
                break
            del messages[n_pin]     # drop the oldest non-pinned turn
        sent_chars = sum(len(m["content"]) for m in messages)
        try:
            # No `stop` on closing tags: the parser needs the closing tag present, and the
            # API excludes stop strings from the output. Repetition penalty + the loop-
            # breaker below handle the runaway-generation the stop was meant to catch.
            content, usage = http_chat(a.base_url, a.model, messages, temp, a.max_tokens,
                                       top_p=a.top_p, repetition_penalty=a.repetition_penalty)
            fails = 0
            budget_tok = a.max_ctx_tokens     # recovered -> restore the full budget
            pt = int(usage.get("prompt_tokens") or 0)
            if pt > 0:
                chars_per_tok = max(0.5, sent_chars / pt)   # ground truth from the tokenizer
        except Exception as e:
            fails += 1
            emit(f"[turn {turn}] MODEL CALL FAILED ({fails}): {e}")
            if fails >= 5:
                emit("[abort] 5 consecutive model-call failures"); break
            # Retrying the IDENTICAL request just fails identically — in gate v2 that burned all 5
            # attempts on the same HTTP 400 and killed the run. Halve the budget so the next attempt
            # is MATERIALLY smaller (the elision loop above then drops more observations).
            budget_tok = max(4000, budget_tok // 2)
            emit(f"[retry] shrinking context budget -> {budget_tok} tokens")
            time.sleep(3); continue
        tot_in += usage.get("prompt_tokens", 0); tot_out += usage.get("completion_tokens", 0)
        act = parse_action(content)
        emit(f"\n===== turn {turn} (ctx~{int(sent_chars / max(chars_per_tok, 0.5))} tok "
             f"@{chars_per_tok:.2f} ch/tok, "
             f"cumul out={tot_out}) =====\n{cap(content, 1200)}")
        # store a bounded copy: a full <write> file body is on disk already, so echoing it in
        # history just bloats context (can overflow the model window) — keep the reasoning + tag head.
        messages.append({"role": "assistant", "content": cap(content, 3000)})

        # LOOP-BREAKER. v3 emitted a byte-identical message for 272 consecutive turns
        # (71% of the token budget) with nothing to stop it. Detect the repeat, escalate
        # temperature next turn (temp uses repeat_run), and inject a concrete order.
        cur = _norm(content)
        if cur and cur == last_asst:
            repeat_run += 1
            emit(f"[loop-breaker] identical reply x{repeat_run+1}; temp -> "
                 f"{min(1.0, a.temperature + 0.2*(repeat_run)):.2f}")
            messages.append({"role": "user", "content":
                "You just repeated your previous message verbatim. That makes no progress. "
                "Do something DIFFERENT now, and make your next message a single <bash> tag "
                "running exactly one of: `./g move ACTION1` (probe the game), `./verify` "
                "(check your model against recorded frames), or `./plan --from-current` "
                "(search for a winning sequence). Run one NOW."})
            if repeat_run >= 3:
                # Truly stuck: hard-reset the recent context so the attractor state is gone.
                del messages[n_pin:-2]
                repeat_run = 0
            continue
        repeat_run = 0
        last_asst = cur

        if act is None:
            messages.append({"role": "user", "content":
                "No valid tool tag found. Reply with EXACTLY ONE tag: <bash>...</bash>, "
                "<write path=\"...\">...</write>, <read path=\"...\"/>, or <done>...</done>."})
            continue
        if act["tool"] == "done":
            # Models declare victory without solving anything: in gate3 the coder emitted <done>
            # at turn 51 after 11 moves ("I have sufficient understanding to conclude that I can
            # solve this level") and the run ended at 116s of a 3600s box — 97% of the budget
            # wasted. A <done> with ZERO levels completed is not a result; push back and continue.
            # Cap was 6 in gate3 v2 -> sb26 emitted <done> a 7th time and stopped at 308s of a
            # 3600s box (91% wasted). The TIME BOX should decide when we stop, not the model's
            # eagerness to declare success, so allow many more rejections and escalate.
            if solved_hi <= 0 and done_rejects < 20:
                done_rejects += 1
                emit(f"\n[done REJECTED {done_rejects}/20 — levels_completed is still 0]")
                nudge = (
                    "You emitted <done> but levels_completed is STILL 0 — nothing has been solved, "
                    "so you are NOT done. Understanding the mechanics is not the goal; COMPLETING A "
                    "LEVEL is. Do not stop. Follow verification-by-execution: (1) write/extend "
                    "world_model_engine.py, (2) run ./verify — it must predict frames EXACTLY, "
                    "(3) fix every mismatch, (4) run ./plan to SEARCH for a winning action "
                    "sequence, (5) execute that sequence with ./g. Continue now with the single "
                    "next concrete step.")
                if done_rejects >= 3:
                    # Repeating the same nudge invites a <done> loop; give ONE concrete command.
                    nudge += ("\nSTOP repeating <done>. Your very next message must be a <bash> tag "
                              "that runs exactly one of: `./verify` (check your model), `./plan` "
                              "(search for a winning sequence), or `./g move ACTION1` (probe). "
                              "Pick one and run it NOW.")
                messages.append({"role": "user", "content": nudge})
                continue
            emit(f"\n[DONE] {act['body']}"); break
        if act["tool"] == "bash":
            obs = run_bash(act["body"], ws)
        elif act["tool"] == "write":
            p = _safe(act["path"])
            if p is None:
                obs = "[write refused: path is outside the workspace]"
            elif os.path.basename(p) in _SCAFFOLD_LOCKED:
                obs = (f"[write refused: {os.path.basename(p)} is provided, working scaffold. "
                       "Do not overwrite it. Edit only world_model_engine.py, "
                       "world_model_state_io.py, or candidate_actions() in "
                       "world_model_main_planner.py.]")
            else:
                try:
                    body = strip_code_fence(act["body"])
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as f:
                        f.write(body)
                    obs = describe_write(act["path"], body)
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
