#!/usr/bin/env python3
"""extract_samples.py — build the INSPECTION-PROMPT REPLAY battery.

Walks the six 08-20/08-22 duck corpora (xpl7, xpl4, packv22, depthdiag,
v12smoke, xd), reconstructs the exact per-request chat context the 27B saw
(system prompt + persistent history + user prompt with regenerated grid
image + in-block assistant/tool messages), rehydrates the sandbox state for
every candidate inspection call from intermediate_states.pkl (our own run
artifact — trusted first-party pickle), replays the 27B's recorded snippet
through the REAL ToolAgent dispatch path, and keeps only samples where the
replayed output matches the recorded transcript output (the fidelity gate).
Emits:

  assets_build/samples.json.gz   — per-sample contexts + recorded refs
  assets_build/gamepacks.json.gz — per-game state packs for the sandbox

Run:  .venv/bin/python submission/_inspect_replay/extract_samples.py
(needs the repo venv for arcengine; taaf src auto-added below.)
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import pickle
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import replay_lib  # noqa: E402
from replay_lib import (  # noqa: E402
    StateRebuilder,
    code_calls_action,
    display_is_error,
    invert_display_to_payload,
    parse_transcript,
    run_snippet,
    split_requests,
)

TAAF_SRC = replay_lib.BUNDLE_SRC / "tufa-arc-agi-framework" / "src"
sys.path.insert(0, str(TAAF_SRC))

CORPUS_BASE = Path(
    os.environ.get(
        "INSPECT_CORPUS_BASE",
        "/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/"
        "f53fb37a-4abf-4bf0-b006-420d5ed2bb2b/scratchpad",
    )
)
CORPORA = ["xpl7", "xpl4", "packv22", "depthdiag", "v12smoke", "xd"]
OUT_DIR = HERE / "assets_build"
MAX_SAMPLES = int(os.environ.get("INSPECT_MAX_SAMPLES", "180"))
PER_GAME_CAP = int(os.environ.get("INSPECT_PER_GAME_CAP", "10"))

TA = replay_lib.bundle_setup()

from inference.agent.runtime_state import Frame  # noqa: E402
from inference.utils.animation import pick_animation, summarize_animation  # noqa: E402

import taaf.game  # noqa: E402,F401 — needed to unpickle GameState


# --------------------------------------------------------------------------
# game state packs

def build_gamepack(run_meta: dict, states: list) -> dict:
    frames, actions, levels, available, won = [], [], [], [], []
    all_frames: dict[str, list] = {}
    for i, st in enumerate(states):
        grid = [[int(c) for c in row] for row in st.frame.data]
        frames.append(grid)
        levels.append(int(st.levels_completed))
        available.append([int(a) for a in st.available_actions])
        won.append(bool(st.won))
        raw_frames = [[[int(c) for c in row] for row in f.data] for f in st.all_frames]
        if len(raw_frames) > 1:
            all_frames[str(i)] = raw_frames
        if i == 0:
            actions.append("")
        else:
            pa = st.previous_action
            if pa is None:
                actions.append("")
            else:
                name = pa.id.name
                data = dict(pa.data or {})
                if name == "ACTION6":
                    actions.append(f"MOUSE(row={int(data.get('y', 0))}, col={int(data.get('x', 0))})")
                else:
                    from inference.agent.action_names import to_model_action  # noqa: PLC0415

                    actions.append(to_model_action(name))
    return {
        "game_id": run_meta["game_id"],
        "number_of_levels": int(run_meta["number_of_levels"]),
        "frames": frames,
        "actions": actions,
        "levels": levels,
        "available": available,
        "won": won,
        "all_frames": all_frames,
        "raw_states": [str(getattr(st.raw, "state", "")) for st in states],
        "just_won": [bool(st.just_won_level) for st in states],
    }


# --------------------------------------------------------------------------
# last_action_result reconstruction (batch heuristic; gate-verified)

def synth_last_action_result(pack: dict, rb: StateRebuilder, start_idx: int, end_idx: int, agent) -> dict | None:
    """Approximate the agent's persisted compact result for the action batch
    that moved the game from state start_idx to end_idx (exclusive of start)."""
    if end_idx <= start_idx:
        return None
    batch = list(range(start_idx + 1, end_idx + 1))
    last = batch[-1]
    prev = last - 1
    board_changed_any = any(pack["frames"][i] != pack["frames"][i - 1] for i in batch)
    n_levels = max(1.0, float(pack["number_of_levels"]))
    reward_total = float(pack["levels"][last] - pack["levels"][start_idx]) / n_levels
    last_reward = float(pack["levels"][last] - pack["levels"][prev]) / n_levels
    raw_state = pack["raw_states"][last]
    state_name = raw_state.rsplit(".", 1)[-1] if raw_state else "NOT_FINISHED"
    is_win = "WIN" in state_name
    is_game_over = state_name == "GAME_OVER"
    level_completed = bool(pack["just_won"][last] and not is_win)
    per_action_anims = []
    for i in batch:
        fr = pack["all_frames"].get(str(i)) or [pack["frames"][i]]
        nf = [tuple(tuple(int(c) for c in row) for row in f) for f in fr]
        bc = pack["frames"][i] != pack["frames"][i - 1]
        per_action_anims.append(summarize_animation(nf, board_changed=bc))
    payload = {
        "executed": True,
        "action_num": last,
        "level": rb.level_number(last),
        "score": int(pack["levels"][last]),
        "reward": reward_total,
        "last_reward": last_reward,
        "state": state_name,
        "valid_actions": rb.valid_action_names(last),
        "board_changed": board_changed_any,
        "done": bool(is_win),
        "level_completed": level_completed,
        "game_over": bool(is_game_over),
        "run_complete": bool(is_win),
        "action_display": pack["actions"][last],
        "action_name": pack["actions"][last],
        "batched": len(batch) > 1,
        "requested_count": len(batch),
        "executed_count": len(batch),
        "requested_actions": [pack["actions"][i] for i in batch],
        "executed_actions": [pack["actions"][i] for i in batch],
        "frame_count": max(
            len(pack["all_frames"].get(str(i), [None])) if str(i) in pack["all_frames"] else 1
            for i in batch
        ),
        "stopped_early": False,
    }
    batch_anim = pick_animation(per_action_anims)
    if batch_anim is not None:
        payload["animation"] = batch_anim
    compact = TA.ToolAgent._compact_action_result(agent, payload)
    return compact


# --------------------------------------------------------------------------
# context reconstruction

def build_tool_calls(req: dict, astep: int, ridx: int) -> list[dict]:
    raw = req["meta"].get("raw_tool_calls")
    if isinstance(raw, list) and raw:
        return copy.deepcopy(raw)
    calls = []
    for i, ev in enumerate(req["tool_events"]):
        code = ev["code"] if ev["code"] is not None else ""
        calls.append(
            {
                "id": f"recon-{astep}-{ridx}-{i}",
                "type": "function",
                "function": {"name": ev["name"], "arguments": json.dumps({"code": code})},
            }
        )
    return calls


NORM_TIMING_RE = re.compile(r'("?(?:run_elapsed_seconds|time_remaining_seconds)"?\s*:\s*)[0-9.eE+-]+')


def norm_display(text: str) -> str:
    t = NORM_TIMING_RE.sub(r"\g<1>X", (text or "").strip())
    return "\n".join(line.rstrip() for line in t.split("\n")).strip()


def process_game(corpus: str, tpath: Path, pack: dict, agent, tools: list[dict], stats: dict) -> list[dict]:
    """Simulate the agent conversation for one game; return candidate samples."""
    blocks = parse_transcript(tpath)
    game_id = pack["game_id"]
    n_states = len(pack["frames"])
    rb = StateRebuilder(pack, TA)
    history_messages: list[dict] = []
    system_prompt = None
    last_action_result: dict | None = None
    prev_block_action: int | None = None
    candidates: list[dict] = []

    for bi, block in enumerate(blocks):
        parts = split_requests(block)
        if parts["system_prompt"]:
            system_prompt = parts["system_prompt"]
        header_action = block["action"]
        state_idx = header_action - 1
        if state_idx >= n_states:
            stats["blocks_beyond_states"] += 1
            break
        # cross-block action jump => the previous block executed actions
        if prev_block_action is not None and state_idx > prev_block_action - 1:
            try:
                last_action_result = synth_last_action_result(
                    pack, rb, prev_block_action - 1, state_idx, agent
                )
            except Exception:  # noqa: BLE001
                stats["lar_synth_errors"] += 1
                last_action_result = None
        prev_block_action = header_action

        if system_prompt is None or parts["user_prompt"] is None:
            stats["blocks_missing_prompts"] += 1
            continue

        grid = tuple(tuple(int(c) for c in row) for row in pack["frames"][state_idx])
        cur_frame = Frame(grid=grid, step=state_idx, level=rb.level_number(state_idx))
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            *copy.deepcopy(history_messages),
            agent._build_user_message(parts["user_prompt"], cur_frame),
        ]
        messages = agent._trim_messages_for_context(messages, tools=tools, preserve_recent=1)
        block_failed = any(
            s.startswith("request_error:") or s.startswith("error:")
            for req in parts["requests"]
            for s in req["statuses"]
        ) or any(s.startswith("request_error:") or s.startswith("error:") for s in parts["pre_statuses"])
        overflow_seen = False
        prior_events_clean = True  # all prior in-block tool events inspection & inverted OK

        for ridx, req in enumerate(parts["requests"]):
            if any("context_overflow_recovered" in s for s in req["statuses"]):
                overflow_seen = True
            meta = req["meta"]
            n_calls = meta["tool_call_count"]
            ev = req["tool_events"][0] if req["tool_events"] else None
            if (
                ridx >= 1
                and not overflow_seen
                and prior_events_clean
                and n_calls == 1
                and ev is not None
                and ev["name"] == "python"
                and ev["code"] is not None
                and not code_calls_action(ev["code"])
                and ev["result_display"] is not None
            ):
                candidates.append(
                    {
                        "corpus": corpus,
                        "game_id": game_id,
                        "transcript": tpath.name,
                        "astep": block["astep"],
                        "action": header_action,
                        "state_idx": state_idx,
                        "block_index": bi,
                        "position": ridx,
                        "messages": copy.deepcopy(messages),
                        "valid_actions": rb.valid_action_names(state_idx),
                        "last_action_result": copy.deepcopy(last_action_result),
                        "recorded_code": ev["code"],
                        "recorded_display": ev["result_display"],
                        "recorded_error": display_is_error(ev["result_display"]),
                        "recorded_reasoning_chars": meta["reasoning_chars"],
                        "recorded_content_chars": meta["content_chars"],
                    }
                )

            # mirror the analyze loop's message appends for this request
            tool_calls = build_tool_calls(req, block["astep"], ridx)
            reasoning = req["thinking"] or ""
            content = req["assistant"] or ""
            assistant_message: dict = {"role": "assistant"}
            if reasoning:
                assistant_message["reasoning"] = reasoning
            if not tool_calls:
                if content:
                    assistant_message["content"] = content
                elif reasoning:
                    assistant_message["content"] = None
                if content or reasoning:
                    messages.append(assistant_message)
                if req["followup_user"] is not None:
                    messages.append({"role": "user", "content": req["followup_user"]})
                messages = agent._trim_messages_for_context(messages, tools=tools, preserve_recent=1)
                continue
            if content:
                assistant_message["content"] = content
            assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)
            for ei, tev in enumerate(req["tool_events"]):
                display = tev["result_display"] or ""
                payload = invert_display_to_payload(display)
                if payload is None:
                    prior_events_clean = False
                    content_str = display  # degraded: display text as content
                    stats["display_invert_fallback"] += 1
                else:
                    content_str = json.dumps(payload, indent=2)
                call_id = tool_calls[ei]["id"] if ei < len(tool_calls) else ""
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content_str})
                if tev["code"] is not None and code_calls_action(tev["code"]):
                    prior_events_clean = False
            if req["followup_user"] is not None:
                messages.append({"role": "user", "content": req["followup_user"]})
            messages = agent._trim_messages_for_context(messages, tools=tools, preserve_recent=1)

        # persistent history update (mirrors analyze()'s finally block)
        if block_failed:
            stats["blocks_failed"] += 1
            continue  # preserve_history=False => history reverts
        history_messages = agent._persistent_history_messages(messages, tools=tools)

    return candidates


# --------------------------------------------------------------------------

def tools_for_game(prompt_log: Path) -> list[dict]:
    desc = None
    if prompt_log.exists():
        text = prompt_log.read_text(errors="replace")
        m = re.search(r"^- python: (.*?)^\[MODEL INPUT\]", text, re.S | re.M)
        if m:
            desc = m.group(1).strip()
    if not desc:
        desc = TA._PYTHON_TOOL_DESCRIPTION
    return [
        {
            "type": "function",
            "function": {
                "name": "python",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code to run. The snippet is ephemeral and is not saved across tool calls."
                            ),
                        },
                    },
                    "required": ["code"],
                },
            },
        }
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agent = TA.ToolAgent(provider="vllm", base_url="http://127.0.0.1:9/v1")
    stats: dict = defaultdict(int)
    all_candidates: list[dict] = []
    gamepacks: dict[str, dict] = {}

    for corpus in CORPORA:
        cdir = CORPUS_BASE / corpus
        bench = json.loads((cdir / "benchmark.json").read_text())
        with open(cdir / "intermediate_states.pkl", "rb") as f:
            states_all = pickle.load(f)  # first-party artifact from our own runs
        runs = bench["game_runs"]
        for gi, run_meta in enumerate(runs):
            gid = run_meta["game_id"]
            tpath = cdir / "transcripts" / f"{gid}_p0.txt"
            if not tpath.exists():
                stats["missing_transcripts"] += 1
                continue
            pack_key = f"{corpus}/{gid}"
            pack = build_gamepack(run_meta, states_all[gi])
            gamepacks[pack_key] = pack
            tools = tools_for_game(cdir / "prompts" / f"{gid}_p0.log")
            t0 = time.time()
            cands = process_game(corpus, tpath, pack, agent, tools, stats)
            for c in cands:
                c["pack_key"] = pack_key
                c["tools"] = tools
            all_candidates.extend(cands)
            print(f"{pack_key}: {len(cands)} candidates ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nTotal candidates (pos>=1 inspection, clean blocks): {len(all_candidates)}")

    # ---------------- stratified pre-selection (cap fidelity-gate cost) -----
    bygame: dict[str, list[dict]] = defaultdict(list)
    for c in all_candidates:
        bygame[c["game_id"]].append(c)
    for group in bygame.values():
        group.sort(key=lambda c: (c["corpus"], c["astep"], c["position"]))
        n = len(group)
        want = min(n, PER_GAME_CAP * 3)
        step = max(1, n // want)
        group[:] = group[::step][: PER_GAME_CAP * 3]

    # ---------------- fidelity gate ----------------------------------------
    selected: list[dict] = []
    gate_fail = 0
    workdir = Path(tempfile.mkdtemp(prefix="inspect_gate_"))
    order = sorted(bygame.keys())
    rebuilders: dict = {}
    round_robin = 0
    per_game_kept: dict[str, int] = defaultdict(int)
    exhausted = False
    while not exhausted and len(selected) < MAX_SAMPLES:
        exhausted = True
        for gid in order:
            group = bygame[gid]
            if not group or per_game_kept[gid] >= PER_GAME_CAP:
                continue
            exhausted = False
            cand = group.pop(0)
            pk = cand["pack_key"]
            if pk not in rebuilders:
                rebuilders[pk] = StateRebuilder(gamepacks[pk], TA)
            rb = rebuilders[pk]
            try:
                res = run_snippet(
                    TA, rb, cand["state_idx"], cand["recorded_code"],
                    cand["last_action_result"], cand["valid_actions"],
                    workdir / f"g{round_robin}",
                )
            except Exception:  # noqa: BLE001
                stats["gate_exec_exception"] += 1
                continue
            round_robin += 1
            if norm_display(res["display"]) == norm_display(cand["recorded_display"]):
                cand["fidelity"] = "exact"
            else:
                gate_fail += 1
                stats["gate_mismatch"] += 1
                continue
            cand["recorded_stdout"] = res["stdout"]
            selected.append(cand)
            per_game_kept[gid] += 1
            if len(selected) >= MAX_SAMPLES:
                break

    print(f"Fidelity gate: kept {len(selected)}, mismatched {gate_fail}, "
          f"exec-exceptions {stats['gate_exec_exception']}")

    # ---------------- summary stats ----------------------------------------
    n_err = sum(1 for c in selected if c["recorded_error"])
    print(f"27B recorded execution-error rate on kept samples: {n_err}/{len(selected)}"
          f" = {n_err/max(1,len(selected)):.3f}")
    per_game = defaultdict(int)
    for c in selected:
        per_game[c["game_id"]] += 1
    print("per-game:", dict(sorted(per_game.items())))
    print("stats:", dict(stats))

    # ---------------- write outputs -----------------------------------------
    used_packs = {c["pack_key"] for c in selected}
    packs_out = {k: v for k, v in gamepacks.items() if k in used_packs}
    for i, c in enumerate(selected):
        c["sample_id"] = f"s{i:03d}"
    with gzip.open(OUT_DIR / "samples.json.gz", "wt", encoding="utf-8") as f:
        json.dump(selected, f)
    with gzip.open(OUT_DIR / "gamepacks.json.gz", "wt", encoding="utf-8") as f:
        json.dump(packs_out, f)
    sizes = {p.name: p.stat().st_size for p in OUT_DIR.glob("*.gz")}
    print("wrote:", sizes)


if __name__ == "__main__":
    main()
