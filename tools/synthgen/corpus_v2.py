"""Corpus builder v2: deliberation-preserving records with revision arcs.

Same fidelity contract as corpus.py (byte-identical system prompt, real engine
frames through the harness image path, corpus_v3 record schema) with the two
v1 defects fixed:

  * TARGETS: long-form programmatic deliberation (narrate.py) whose token
    distribution is gated against corpus_v3 (mean/P10/P90 within ±25%) —
    the v1 corpus's ~265-token templated targets taught the adapter to
    under-deliberate (−47% levels at play, 2026-08-04 behavioral eval).
  * TRAJECTORIES: scripted wrong-hypothesis -> contradiction -> revision -> win
    arcs (episodes.py) executed through the REAL engine, with ~25% clean
    solves mixed in.

History is windowed at build time with the same drop-oldest-turn rule the
kernel uses (sft_common.truncate_front) so records encode under the kernel
front-truncation cap with few or no drops.

Output dir mirrors corpus_v3/corpus_synth_v1: train/val/holdout_transfer
JSONL + tools.json + sft_common.py + tokenizer_bundle + prep_stats.json,
plus gates.json with the v2 acceptance-gate measurements (validate_v2.py).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from corpus import (  # noqa: E402
    CORPUS_V3,
    SFT_K3_DIR,
    _display,
    _knowledge_lines,
    _load_tokenizer,
    _token_counts,
    build_user_text,
    system_prompt,
    teacher_code,
    user_message,
)
from episodes import build_episode  # noqa: E402
from narrate import CHARS_PER_TOKEN, draw_target_budget, render_turn  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
from inference.agent.action_names import ENGINE_TO_MODEL_ACTION  # noqa: E402

IMAGE_TOKENS_EST = 85       # ~(256/28)^2 merged patches + wrappers per image
WINDOW_TOKENS = 23000       # est(messages)+est(target) kept below this
KERNEL_MAX_LEN_DEFAULT = 24576
KERNEL_MAX_LEN_FULL = 32768

TEACHER_TAG = "synthgen/scripted-teacher-v2"


def _est_msg_tokens(m: dict[str, Any]) -> float:
    t = 0.0
    c = m.get("content")
    if isinstance(c, str):
        t += len(c) / CHARS_PER_TOKEN
    elif isinstance(c, list):
        for p in c:
            if p.get("type") == "text":
                t += len(p["text"]) / CHARS_PER_TOKEN
            elif p.get("type") == "image_url":
                t += IMAGE_TOKENS_EST
    if m.get("reasoning_content"):
        t += len(m["reasoning_content"]) / CHARS_PER_TOKEN
    for tc in m.get("tool_calls") or []:
        args = tc.get("function", {}).get("arguments")
        t += len(args["code"] if isinstance(args, dict) else str(args)) / CHARS_PER_TOKEN
    return t + 8


def _truncate_front(messages: list[dict]) -> list[dict] | None:
    """Same rule as sft_common.truncate_front (kept dependency-free here)."""
    if len(messages) <= 3:
        return None
    msgs = messages[:2] + messages[3:]
    while len(msgs) > 2 and msgs[2]["role"] == "tool":
        msgs = msgs[:2] + msgs[3:]
    return msgs if len(msgs) < len(messages) else None


def _window(messages: list[dict], target: dict) -> tuple[list[dict], int]:
    est = sum(_est_msg_tokens(m) for m in messages) + _est_msg_tokens(target)
    dropped = 0
    msgs = messages
    while est > WINDOW_TOKENS:
        nxt = _truncate_front(msgs)
        if nxt is None:
            break
        est -= sum(_est_msg_tokens(m) for m in msgs[2 : 2 + (len(msgs) - len(nxt))])
        msgs = nxt
        dropped += 1
    return msgs, dropped


def build_game_records_v2(
    games_root: Path, game_dir: Path, seed: int, sys_prompt: str
) -> list[dict[str, Any]]:
    spec = json.loads((game_dir / "spec.json").read_text())
    solution = json.loads((game_dir / "solution.json").read_text())
    spec["solution_levels"] = solution["per_level"]
    game_id = spec["game_id"]
    base_id = game_id.split("-")[0]
    total_levels = len(spec["levels"])
    episode = build_episode(spec, seed)

    valid_models = [ENGINE_TO_MODEL_ACTION[f"ACTION{i}"] for i in spec["available_actions"]]
    valid_line = ", ".join(valid_models)
    mouse = "MOUSE" in valid_models

    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(games_root))
    env = arcade.make(game_id, save_recording=False)
    if env is None:
        raise RuntimeError(f"could not load {game_id}")
    frame = env.reset()
    grid = [list(map(int, row)) for row in frame.frame[-1]]

    system_msg = {"role": "system", "content": sys_prompt}
    messages: list[dict[str, Any]] = [
        system_msg,
        user_message(
            build_user_text(summary=None, step=1, level=1, valid_line=valid_line,
                            knowledge=[], first_turn=True, mouse=mouse),
            grid, step=0, level=1),
    ]

    flat_turns = [(li, t) for li, lvl in enumerate(episode["levels"]) for t in lvl]
    records: list[dict[str, Any]] = []
    executed = 0
    score = 0
    for analysis_step, (li, turn) in enumerate(flat_turns):
        chunk = turn["actions"]
        code = teacher_code(chunk)
        nrng = random.Random(f"narr-{game_id}-{analysis_step}-{seed}")
        total_budget = draw_target_budget(nrng)
        overhead = int((len(code) + 60) / CHARS_PER_TOKEN) + 45
        reasoning, content, wm, plan = render_turn(
            spec, turn["sem"], nrng, max(90, total_budget - overhead))
        assistant = {
            "role": "assistant",
            "content": content,
            "tool_calls": [{
                "type": "function", "index": 0, "id": f"python_{analysis_step}",
                "function": {"name": "python", "arguments": {"code": code}},
            }],
            "reasoning_content": reasoning,
        }
        rec_msgs, dropped = _window(messages, assistant)
        records.append({
            "messages": [dict(m) for m in rec_msgs],
            "target": assistant,
            "meta": {
                "game": base_id,
                "level": li + 1,
                "actions_used": len([a for _l, t in flat_turns for a in t["actions"]]),
                "baseline_actions": sum(len(l) for l in spec["solution_levels"]),
                "teacher": TEACHER_TAG,
                "episode": f"synthgen2_{game_id}",
                "analysis_step": analysis_step,
                "first_action_num": executed + 1,
                "n_actions_turn": len(chunk),
                "level_won": True,
                "flags": [],
                "dropped_turns": dropped,
                "n_images": sum(
                    1 for m in rec_msgs
                    if isinstance(m.get("content"), list)
                    and any(p.get("type") == "image_url" for p in m["content"])),
                "family": spec["family"],
                "phase": turn["phase"],
                "clean_episode": episode["clean"],
                "hyp": turn["sem"]["hyp"]["id"],
                "revised": bool(turn["sem"].get("revision")),
                "gen_seed": spec["seed"],
                "gen_index": spec["index"],
                "synthetic": True,
            },
        })

        # execute through the real engine
        board_changed_any = False
        prev_score = score
        for action in chunk:
            game_action = GameAction.from_name(action["id"])
            data = ({"x": int(action["x"]), "y": int(action["y"])}
                    if action["id"] == "ACTION6" else None)
            frame = env.step(game_action, data=data)
            if frame is None:
                raise RuntimeError(f"step failed for {game_id}: {action}")
            new_grid = [list(map(int, row)) for row in frame.frame[-1]]
            board_changed_any = board_changed_any or new_grid != grid
            grid = new_grid
            executed += 1
            score = int(frame.levels_completed)
        state = frame.state.name
        is_win = state == "WIN"
        if state == "GAME_OVER":
            raise RuntimeError(f"{game_id}: scripted episode hit GAME_OVER at turn {analysis_step}")
        completed = bool(score > prev_score)
        if completed != turn["assert"]["level_completed"]:
            raise RuntimeError(
                f"{game_id} turn {analysis_step}: level_completed={completed} "
                f"but script asserted {turn['assert']['level_completed']}")
        level_num = total_levels if is_win else max(1, min(total_levels, score + 1))
        result = {
            "board_changed": board_changed_any,
            "level_completed": bool(completed and not is_win),
            "game_over": False,
            "done": is_win,
            "level": level_num,
            "score": score,
        }
        stdout = f"result {result!r}\n"
        tool_msg = {
            "role": "tool",
            "tool_call_id": f"python_{analysis_step}",
            "content": json.dumps({"tool": "python", "returncode": 0, "stdout": stdout}, indent=2),
        }
        messages.append(assistant)
        messages.append(tool_msg)

        if analysis_step < len(flat_turns) - 1:
            summary = {
                "executed_count": len(chunk),
                "executed_actions": [_display(a) for a in chunk],
                "level_transition": bool(completed and not is_win),
                "run_complete": is_win,
                "game_over": False,
            }
            messages.append(user_message(
                build_user_text(summary=summary, step=executed + 1, level=level_num,
                                valid_line=valid_line,
                                knowledge=_knowledge_lines(wm, plan),
                                first_turn=False, mouse=mouse),
                grid, step=executed, level=level_num))

    if frame.state.name != "WIN":
        raise RuntimeError(f"{game_id}: episode did not end in WIN (state={frame.state.name})")
    return records


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------


def build_corpus_v2(
    games_root: Path,
    out_dir: Path,
    holdout_family: str = "mirror",
    val_frac: float = 0.15,
    seed: int = 0,
    with_tokens: bool = True,
) -> dict[str, Any]:
    manifest_path = games_root / "manifest.jsonl"
    manifest = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    valid_ids = [m["game_id"] for m in manifest if m["ok"]]
    if len(valid_ids) != len(manifest):
        bad = [m["game_id"] for m in manifest if not m["ok"]]
        raise RuntimeError(f"refusing to build corpus with invalid games: {bad}")

    sys_prompt = system_prompt()
    tools = json.loads((CORPUS_V3 / "tools.json").read_text())
    tokenizer = _load_tokenizer() if with_tokens else None

    by_family: dict[str, list[tuple[str, list[dict]]]] = {}
    for game_id in valid_ids:
        base, version = game_id.split("-")
        game_dir = games_root / base / version
        records = build_game_records_v2(games_root, game_dir, seed, sys_prompt)
        for rec in records:
            total, target = _token_counts(tokenizer, tools, rec)
            rec["meta"]["qwen_total_text"] = total
            rec["meta"]["qwen_target_text"] = target
        family = records[0]["meta"]["family"]
        by_family.setdefault(family, []).append((game_id, records))
        print(f"  built {game_id} ({family}): {len(records)} records", flush=True)

    rng = random.Random(f"split2-{seed}")
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for family, games in sorted(by_family.items()):
        if family == holdout_family:
            for _gid, recs in games:
                holdout_rows.extend(recs)
            continue
        games = list(games)
        rng.shuffle(games)
        n_val = max(1, round(len(games) * val_frac))
        for _gid, recs in games[:n_val]:
            val_rows.extend(recs)
        for _gid, recs in games[n_val:]:
            train_rows.extend(recs)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("val.jsonl", val_rows),
                       ("holdout_transfer.jsonl", holdout_rows)):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    shutil.copy2(CORPUS_V3 / "tools.json", out_dir / "tools.json")
    shutil.copy2(SFT_K3_DIR / "sft_common.py", out_dir / "sft_common.py")
    bundle_dst = out_dir / "tokenizer_bundle"
    if not bundle_dst.exists():
        shutil.copytree(CORPUS_V3 / "tokenizer_bundle", bundle_dst)

    all_rows = train_rows + val_rows + holdout_rows

    def _tok_stats(rows: list[dict]) -> dict[str, Any]:
        totals = sorted(r["meta"]["qwen_total_text"] for r in rows
                        if r["meta"].get("qwen_total_text"))
        targets = sorted(r["meta"]["qwen_target_text"] for r in rows
                         if r["meta"].get("qwen_target_text"))
        if not totals:
            return {}

        def pct(xs, p):
            return xs[min(len(xs) - 1, int(p * len(xs)))]

        return {
            "n": len(totals),
            "total_mean": round(sum(totals) / len(totals), 1),
            "total_p50": pct(totals, 0.5),
            "total_max": totals[-1],
            "target_mean": round(sum(targets) / len(targets), 1),
            "target_p10": pct(targets, 0.10),
            "target_p50": pct(targets, 0.50),
            "target_p90": pct(targets, 0.90),
            "target_max": targets[-1],
        }

    stats = {
        "generator": "synthgen_v2",
        "split_unit": "game",
        "holdout_family": holdout_family,
        "n_games": len(valid_ids),
        "games_per_family": {f: len(g) for f, g in sorted(by_family.items())},
        "clean_episodes": sum(
            1 for f, games in by_family.items() for gid, recs in games
            if recs[0]["meta"]["clean_episode"]),
        "rows": {"train": len(train_rows), "val": len(val_rows),
                 "holdout_transfer": len(holdout_rows)},
        "phase_counts": _count_by(all_rows, "phase"),
        "token_stats_text_only": {
            "train": _tok_stats(train_rows),
            "val": _tok_stats(val_rows),
            "holdout_transfer": _tok_stats(holdout_rows),
            "all": _tok_stats(all_rows),
        },
        "note": (
            "qwen_*_text counts exclude image patch tokens (the processor adds "
            "~85/image); corpus_v3 qwen_total included them"
        ),
    }
    (out_dir / "prep_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        v = str(r["meta"].get(key))
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=Path, default=_HERE / "out" / "games_v2")
    ap.add_argument("--out", type=Path, default=_HERE / "out" / "corpus_synth_v2")
    ap.add_argument("--holdout-family",
                    choices=("nav", "click", "push", "replay", "carry", "mirror", "rules"),
                    default="mirror")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tokens", action="store_true")
    args = ap.parse_args()
    stats = build_corpus_v2(args.games, args.out, holdout_family=args.holdout_family,
                            val_frac=args.val_frac, seed=args.seed,
                            with_tokens=not args.no_tokens)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
