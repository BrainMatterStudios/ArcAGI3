"""Stage-0 kill test loop: frontier-recorded level-0 transitions + backtest tool -> Qwen3.8-27B
must produce a backtest-green world model within <= 20 calls / <= 60 min per game.

usage: stage0.py [--mock] [--max-calls 20] [--budget-min 60] [--prompt-cap 24000] game [game ...]
Artifacts per game in <here>/runs/<game>/: prompt0.txt, cand_NN.py, bt_NN.json, resp_NN.json, log.jsonl, summary.json
"""
import argparse, copy, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backtest, llm

FRONTIER_FINAL = {  # used only by --mock
    "sk48": "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/search/protocol/hf_traces/claude_fable_opus/claude-fable-5_max_sk48_100.0/world_model_v5.py",
    "tn36": "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/search/protocol/hf_traces/claude_fable_opus/claude-fable-5_max_tn36_94.74/world_model_v5.py",
    "cn04": "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/4567d58c-c4de-46ec-a756-19dc06bd709d/scratchpad/search/protocol/hf_traces/claude_fable_opus/claude-opus-4-8_max_cn04_100.0/world_model_v5.py",
}

HEX = "0123456789abcdef"
CTX = 32768  # served max-model-len (Flash-Next; the 27B run used 65536)
ENCODING = "hex"  # "hex" = pre-registered arm; "int" = Stage-0b encoding variant (NOT pre-registered)


def hexrow(row):
    return "".join(HEX[v] for v in row)


def encode_grid(g):
    """Hex rows, consecutive identical rows collapsed: 'r00-11 5555...'."""
    out = []
    r = 0
    while r < len(g):
        s = r
        while r + 1 < len(g) and g[r + 1] == g[s]:
            r += 1
        tag = f"r{s:02d}" if s == r else f"r{s:02d}-{r:02d}"
        out.append(f"{tag} {hexrow(g[s])}")
        r += 1
    return "\n".join(out)


def encode_diff(prev, cur):
    """Changed cells as per-row runs: 'r20 c11-16 555555>666666' (old>new)."""
    lines = []
    for r in range(len(cur)):
        pr, cr = prev[r], cur[r]
        c = 0
        runs = []
        while c < len(cr):
            if pr[c] != cr[c]:
                s = c
                while c + 1 < len(cr) and pr[c + 1] != cr[c + 1]:
                    c += 1
                runs.append(f"c{s:02d}-{c:02d} {hexrow(pr[s:c+1])}>{hexrow(cr[s:c+1])}" if c > s
                            else f"c{s:02d} {HEX[pr[s]]}>{HEX[cr[s]]}")
            c += 1
        if runs:
            lines.append(f"  r{r:02d} " + " | ".join(runs))
    if not lines:
        return "  (no cells changed)"
    return "\n".join(lines)


def encode_grid_int(g):
    """Coordinate-explicit: column ruler on top, row label at left, width-2 space-separated ints;
    consecutive identical rows collapsed as 'r00-11'."""
    out = ["      c: " + " ".join(f"{c:2d}" for c in range(len(g[0])))]
    r = 0
    while r < len(g):
        s = r
        while r + 1 < len(g) and g[r + 1] == g[s]:
            r += 1
        tag = f"r{s:02d}   " if s == r else f"r{s:02d}-{r:02d}"
        out.append(f"{tag}: " + " ".join(f"{v:2d}" for v in g[s]))
        r += 1
    return "\n".join(out)


def encode_diff_int(prev, cur, cap=200):
    """Sparse changed cells '(r,c): old->new'; None if more than cap cells changed."""
    cells = [(r, c, prev[r][c], cur[r][c]) for r in range(len(cur)) for c in range(len(cur[r])) if prev[r][c] != cur[r][c]]
    if len(cells) > cap:
        return None, len(cells)
    if not cells:
        return "  (no cells changed)", 0
    by_row = {}
    for r, c, o, n in cells:
        by_row.setdefault(r, []).append((c, o, n))
    lines = []
    for r in sorted(by_row):
        items, buf, i = by_row[r], [], 0
        while i < len(items):
            c0, o, n = items[i]
            j = i
            while j + 1 < len(items) and items[j + 1][0] == items[j][0] + 1 and items[j + 1][1] == o and items[j + 1][2] == n:
                j += 1
            buf.append(f"c{c0}:{o}->{n}" if j == i else f"c{c0}-{items[j][0]}:{o}->{n}")
            i = j + 1
        lines.append(f"  r{r}: " + " ".join(buf))
    return "\n".join(lines), len(cells)


def build_data_block_int(rec):
    tr = rec["transitions"]
    entry = rec["entry_grid"]
    parts = []
    parts.append(f"GAME {rec['game']} — LEVEL 0 — {len(tr)} recorded transitions (indices 0..{len(tr)-1}).")
    parts.append("Grid encoding: 64 rows x 64 cols of integer colors 0-15. Each row is labelled 'rNN' (row index y) at the left; "
                 "the column ruler 'c:' on top gives the column index x of every value; runs of identical consecutive rows are "
                 "collapsed as 'rAA-BB'. Cells are grid[y][x] (row y, column x). Transitions list ONLY the changed cells, grouped by row, as "
                 "'rROW: cCOL:old->new cCOL:old->new ...' (explicit row and column indices; 'cA-B:old->new' means every column A..B "
                 "inclusive changed from old to new); every other cell is unchanged.")
    parts.append("\n=== ENTRY GRID (state before transition #0; also available to your code as the global ENTRY_GRID) ===")
    parts.append(encode_grid_int(entry))
    prev = entry
    for t in tr:
        i = t["index"]
        hdr = f"\n=== TRANSITION #{i}: {act_str(t)} -> flags: {flags_str(t)}"
        if t["action"] == 0:
            parts.append(hdr + " (RESET: the grid goes back to the ENTRY GRID; this step is NOT checked; your state is re-initialised with init_state)")
            prev = t["grid"]; continue
        if t.get("level_up") or t.get("win"):
            parts.append(hdr + " (TERMINAL: your model must return level_up=True here; the resulting grid is the NEXT level's board and is NOT checked)")
            prev = t["grid"]; continue
        d, n = encode_diff_int(prev, t["grid"])
        if d is None:
            parts.append(hdr + f" — {n} cells changed (too many to list) — RESULTING GRID (full):")
            parts.append(encode_grid_int(t["grid"]))
        else:
            parts.append(hdr + f" — CHANGED CELLS ({n}), grouped by row, cCOL:old->new:")
            parts.append(d)
        prev = t["grid"]
    return "\n".join(parts)


def flags_str(t):
    f = [k for k in ("level_up", "dead", "win") if t.get(k)]
    return ",".join(f) if f else "none"


def act_str(t):
    a = t["action"]
    if a == 6:
        return f"action=6 CLICK x={t['x']} y={t['y']}"
    if a == 0:
        return "action=0 RESET"
    return f"action={a}"


def build_data_block(rec, n_full):
    tr = rec["transitions"]
    entry = rec["entry_grid"]
    parts = []
    parts.append(f"GAME {rec['game']} — LEVEL 0 — {len(tr)} recorded transitions (indices 0..{len(tr)-1}).")
    parts.append("Grid encoding: 64 rows x 64 cols, one hex digit per cell (0-9,a-f = colors 0-15); "
                 "'rNN' = row index (y); runs of identical consecutive rows are collapsed as 'rAA-BB'. "
                 "Cells are grid[y][x] (row y, column x).")
    parts.append("\n=== ENTRY GRID (state before transition #0; also available to your code as the global ENTRY_GRID) ===")
    parts.append(encode_grid(entry))
    prev = entry
    for t in tr:
        i = t["index"]
        hdr = f"\n=== TRANSITION #{i}: {act_str(t)} -> flags: {flags_str(t)}"
        if t["action"] == 0:
            parts.append(hdr + " (RESET: the grid goes back to the ENTRY GRID; this step is NOT checked; your state is re-initialised with init_state)")
            prev = t["grid"]
            continue
        if t.get("level_up") or t.get("win"):
            parts.append(hdr + " (TERMINAL: your model must return level_up=True here; the resulting grid is the NEXT level's board and is NOT checked)")
            prev = t["grid"]
            continue
        if i < n_full:
            parts.append(hdr + " — RESULTING GRID (full):")
            parts.append(encode_grid(t["grid"]))
        else:
            parts.append(hdr + " — CHANGED CELLS (old>new), everything else unchanged:")
            parts.append(encode_diff(prev, t["grid"]))
        prev = t["grid"]
    return "\n".join(parts)


SYSTEM = """You are building an executable WORLD MODEL for one level of an unknown ARC-AGI-3 game, from recorded transitions only.
You will be given the level's entry grid and every recorded (action -> resulting grid, flags) transition. A backtest tool then
replays your model over those transitions and reports the first mismatch. You get several rounds to fix it. Goal: BACKTEST-GREEN
(every checked transition reproduced exactly: full 64x64 grid AND the flags).

CONTRACT — output ONE complete, self-contained Python file (standard library and numpy only), in a single ```python fence.
Either style is accepted:
  (A) stateless:  def step(grid, action, x=None, y=None) -> (new_grid, flags)
  (B) stateful:   def init_state(entry_grid) -> state
                  def predict(state, grid, action, x=None, y=None) -> (new_grid, flags, new_state)
      (use B only if hidden state is genuinely needed, e.g. a counter that is not readable from the grid)
  optional:       def is_goal(state, grid) -> bool
- grid: list of 64 lists of 64 ints (0-15). Cell (row y, column x) is grid[y][x]. Return a NEW grid (do not mutate the input).
- flags: dict {"level_up": bool, "dead": bool, "win": bool} (all False unless the step ends the level / kills / wins).
- action ids: 1,2,3,4,5 are keyboard-like actions (semantics unknown: infer them from the data), 6 = CLICK at (x, y) where x is
  the column and y is the row; 0 = RESET (never sent to your model). x, y are None for non-click actions.
- ENTRY_GRID (the level's entry grid, list of lists) is injected as a module-level global before your code runs.

HOW THE BACKTEST SCORES (teacher forcing):
- Start from the entry grid (state = init_state(entry) for style B). For each recorded transition your model gets the RECORDED
  previous grid and the action; the predicted grid must equal the recorded resulting grid cell-for-cell, and flags must match.
- On a TERMINAL transition (recorded level_up) only the flags are checked (you must return level_up=True); the grid is not.
- RESET transitions are skipped (state re-initialised on the post-reset grid, which equals the entry grid).
- A crash counts as a mismatch. A model that hangs is killed.

RULES: implement the game's MECHANICS generally (rules that would also hold for unseen action sequences: object positions,
movement, clamping, counters/budget bars, goal condition). Do NOT hardcode a lookup table of the recorded grids/diffs. Prefer
reading object positions from the input grid each step over trusting hidden state. Keep your reasoning focused: you have a
limited output budget — think efficiently, then emit the file. Your reply must END with the complete file in one ```python fence.

OUTPUT REQUIREMENT (hard): the reply MUST end with the complete Python file inside a single ```python fence. A reply without the
file scores zero. Keep thinking proportionate to the problem — do not re-count characters of long rows by hand: the row strings
are exactly 64 characters, and column indices can be read from the 'cNN' markers in the changed-cell lines."""


def extract_code(text):
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if not fences:
        return None
    return fences[-1]


def render_mismatch(res, rec, k_cells=48):
    mm = res.get("first_mismatch")
    n_mm = len(res.get("mismatches", []))
    head = f"BACKTEST RESULT: {res['matched']}/{res['total']} transitions fully correct ({res.get('skipped',0)} reset skipped); {n_mm} mismatched."
    if res.get("green"):
        return head + " GREEN."
    if mm is None:
        return head
    out = [head]
    idxs = [m["index"] for m in res.get("mismatches", [])][:25]
    out.append(f"Mismatched transition indices: {idxs}{' ...' if n_mm > 25 else ''}")
    t = None
    for tt in rec["transitions"]:
        if tt["index"] == mm.get("index"):
            t = tt
    out.append(f"\nFIRST MISMATCH — transition #{mm.get('index')}: {act_str(t) if t else mm.get('action')}")
    if mm.get("kind") in ("exception", "timeout", "crash") or mm.get("traceback") or mm.get("error"):
        out.append("Your model raised / failed:\n" + (mm.get("traceback") or mm.get("error") or "")[-2500:])
        return "\n".join(out)
    out.append(f"expected flags: {mm.get('expected_flags')}   your flags: {mm.get('got_flags')}")
    if mm.get("shape_error"):
        out.append("Your returned grid does not have shape 64x64.")
        return "\n".join(out)
    nc = mm.get("cells_differ", 0)
    if nc:
        out.append(f"grid: {nc} cell(s) differ (row,col: expected>yours), first {min(nc, k_cells)}:")
        if ENCODING == "int":
            out.append("  " + "  ".join(f"({r},{c}): {e}->{g}" for r, c, e, g in mm.get("cells", [])[:k_cells]))
        else:
            out.append("  " + " ".join(f"({r},{c}: {HEX[e]}>{HEX[g]})" for r, c, e, g in mm.get("cells", [])[:k_cells]))
        rows = sorted({r for r, _, _, _ in mm.get("cells", [])})[:6]
        if t is not None:
            out.append("expected full rows (first differing rows):")
            for r in rows:
                if ENCODING == "int":
                    out.append(f"  r{r:02d}: " + " ".join(f"{v:2d}" for v in t['grid'][r]))
                else:
                    out.append(f"  r{r:02d} {hexrow(t['grid'][r])}")
    return "\n".join(out)


class Mock:
    def __init__(self, game):
        self.src = open(FRONTIER_FINAL[game]).read()

    def chat(self, messages, **kw):
        return {"content": "Here is the model.\n```python\n" + self.src + "\n```", "reasoning": "",
                "finish_reason": "stop", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "seconds": 0.0}


def run_game(game, args, chat_fn, log):
    rec = json.load(open(os.path.join(HERE, f"{game}_transitions.json")))
    rdir = os.path.join(HERE, "runs", game + ("_0b" if ENCODING == "int" else "") + ("_mock" if args.mock else ""))
    os.makedirs(rdir, exist_ok=True)
    trans_path = os.path.join(HERE, f"{game}_transitions.json")
    n_tr = len(rec["transitions"])
    # choose how many transitions get full grids so the data block fits the prompt cap
    n_full, data, ntok, how = 0, None, None, None
    if ENCODING == "int":
        data = build_data_block_int(rec)
        ntok, how = llm.tokenize_count(SYSTEM + data) if not args.mock else (len(SYSTEM + data) // 3, "approx")
        n_full = "sparse(int)"
    for cand in (sorted({0, 1, 2, 3, 4, 6, 8, 12, n_tr}) if ENCODING == "hex" else []):
        d = build_data_block(rec, cand)
        c, h = llm.tokenize_count(SYSTEM + d) if not args.mock else (len(SYSTEM + d) // 3, "approx")
        if c <= args.prompt_cap:
            n_full, data, ntok, how = cand, d, c, h
        else:
            break
    if data is None:  # even 0 full grids too big -> take it anyway
        data = build_data_block(rec, 0)
        ntok, how = llm.tokenize_count(SYSTEM + data) if not args.mock else (len(SYSTEM + data) // 3, "approx")
    user0 = (data + f"\n\nOutput the complete world-model file now (one ```python fence). "
             f"It must reproduce all {n_tr} transitions above exactly.")
    open(os.path.join(rdir, "prompt0.txt"), "w").write("### SYSTEM\n" + SYSTEM + "\n\n### USER\n" + user0)
    log(f"[{game}] {n_tr} transitions; n_full={n_full}; initial prompt ~{ntok} tokens ({how})")

    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user0}]
    history = []  # list of (assistant_content, feedback) pairs, only last K kept in the prompt
    K = 2 if (ntok or 0) <= 25000 else 1  # keep >= ~25k output room on big prompts
    best = {"matched": -1, "total": None, "call": None}
    scoreboard = []
    t_start = time.time()
    tok_p = tok_c = 0
    green = False
    calls = 0
    final_mm = None
    consecutive_length = 0
    failure_mode = None
    for call in range(1, args.max_calls + 1):
        if time.time() - t_start > args.budget_min * 60:
            log(f"[{game}] budget exceeded ({args.budget_min} min) before call {call}; stopping")
            break
        msgs = list(base)
        for a, fb in history[-K:]:
            msgs.append({"role": "assistant", "content": a})
            msgs.append({"role": "user", "content": fb})
        if args.mock:
            ptok, phow = len("".join(m["content"] for m in msgs)) // 3, "approx"
        else:
            ptok, phow = llm.messages_token_count(msgs)
        max_tok = max(1024, min(args.max_tokens, CTX - ptok - 512))
        try:
            resp = chat_fn(msgs, max_tokens=max_tok)
        except Exception as e:
            log(f"[{game}] call {call} FAILED: {e}")
            json.dump({"error": str(e)}, open(os.path.join(rdir, f"resp_{call:02d}.json"), "w"))
            calls += 1
            time.sleep(10)
            continue
        calls += 1
        u = resp.get("usage") or {}
        tok_p += u.get("prompt_tokens", 0) or 0
        tok_c += u.get("completion_tokens", 0) or 0
        log(f"[{game}] call {call} usage: prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} "
            f"(max_tokens={max_tok}, pre-count {ptok} {phow}) finish={resp['finish_reason']} {resp['seconds']:.0f}s "
            f"segments={len(resp.get('segments') or [])}")
        json.dump(resp, open(os.path.join(rdir, f"resp_{call:02d}.json"), "w"))
        code = extract_code(resp["content"])
        if code is None and resp["finish_reason"] == "length":
            consecutive_length += 1
        else:
            consecutive_length = 0
        if code is None:
            fb = (f"BACKTEST RESULT: no ```python fence found in your reply (finish_reason={resp['finish_reason']}, "
                  f"{u.get('completion_tokens')} tokens generated). You must end your reply with the COMPLETE file in one "
                  f"```python fence. Keep your thinking much shorter so the file fits in the output budget.")
            res = {"matched": 0, "total": sum(1 for t in rec["transitions"] if t["action"] != 0), "green": False, "no_code": True}
            asst = resp["content"][-2000:] if resp["content"] else "(empty reply)"
            log(f"[{game}] call {call}: NO CODE (finish={resp['finish_reason']}, {resp['seconds']:.0f}s, out={u.get('completion_tokens')})")
        else:
            cpath = os.path.join(rdir, f"cand_{call:02d}.py")
            open(cpath, "w").write(code)
            res = backtest.run_subprocess(cpath, trans_path, os.path.join(rdir, f"bt_{call:02d}.json"),
                                          timeout_s=args.bt_timeout, python=sys.executable)
            fb = render_mismatch(res, rec)
            asst = "```python\n" + code + "\n```"
            log(f"[{game}] call {call}: {res['matched']}/{res['total']} (finish={resp['finish_reason']}, {resp['seconds']:.0f}s, "
                f"in={u.get('prompt_tokens')} out={u.get('completion_tokens')}) first_mm={(res.get('first_mismatch') or {}).get('index')}")
        scoreboard.append((call, res["matched"], res["total"]))
        if res["matched"] > best["matched"]:
            best = {"matched": res["matched"], "total": res["total"], "call": call}
        final_mm = fb
        with open(os.path.join(rdir, "log.jsonl"), "a") as f:
            f.write(json.dumps({"call": call, "matched": res["matched"], "total": res["total"], "seconds": resp["seconds"],
                                "finish": resp["finish_reason"], "usage": u, "elapsed": time.time() - t_start}) + "\n")
        if res.get("green"):
            green = True
            log(f"[{game}] GREEN at call {call}")
            break
        if consecutive_length >= 2:
            failure_mode = f"cannot emit code within a {args.max_tokens}k-token call".replace("000k", "k")
            log(f"[{game}] STOP: two consecutive calls ended finish=length with no code -> {failure_mode}")
            break
        sb = "; ".join(f"call{c}:{m}/{t}" for c, m, t in scoreboard)
        fb += (f"\n\nAttempt history (matched/total): {sb}. Fix the model and output the COMPLETE corrected file again "
               f"(one ```python fence, whole file, not a patch).")
        history.append((asst, fb))
    summary = {"game": game, "green": green, "best_matched": best["matched"], "total": best["total"], "best_call": best["call"],
               "calls": calls, "wall_s": round(time.time() - t_start, 1), "prompt_tokens": tok_p, "completion_tokens": tok_c,
               "n_full_grids": n_full, "prompt0_tokens": ntok, "prompt0_tokens_how": how, "scoreboard": scoreboard,
               "final_feedback": final_mm, "mock": args.mock, "failure_mode": failure_mode,
               "max_tokens_cap": args.max_tokens, "ctx": CTX}
    json.dump(summary, open(os.path.join(rdir, "summary.json"), "w"), indent=1)
    log(f"[{game}] DONE green={green} best={best['matched']}/{best['total']} calls={calls} wall={summary['wall_s']}s tokens={tok_p}+{tok_c}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="+")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--max-calls", type=int, default=20)
    ap.add_argument("--budget-min", type=float, default=60)
    ap.add_argument("--prompt-cap", type=int, default=24000)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--bt-timeout", type=int, default=180)
    ap.add_argument("--encoding", choices=["hex", "int"], default="hex")
    ap.add_argument("--seg-tokens", type=int, default=10000, help="max output tokens per HTTP request (Modal kills requests at ~300s)")
    args = ap.parse_args()
    global ENCODING
    ENCODING = args.encoding
    if ENCODING == "int":
        global SYSTEM
        SYSTEM = SYSTEM.replace("the row strings\nare exactly 64 characters, and column indices can be read from the 'cNN' markers in the changed-cell lines.",
                                "every value's column index\nis given by the column ruler, and every changed cell is listed with explicit (row,col) coordinates.")
    logf = open(os.path.join(HERE, "stage0.log" if ENCODING == "hex" else "stage0b.log"), "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    if not args.mock:
        if not llm.wait_for_model(log=log):
            log("model never came up; aborting"); sys.exit(2)
    out = []
    for g in args.games:
        chat_fn = Mock(g).chat if args.mock else (lambda msgs, max_tokens: llm.chat_chained(msgs, max_tokens=max_tokens, seg_tokens=args.seg_tokens, log=log))
        out.append(run_game(g, args, chat_fn, log))
    print(json.dumps(out, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
