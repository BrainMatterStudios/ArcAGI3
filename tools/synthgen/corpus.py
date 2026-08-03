"""Corpus builder: replay reference traces through the REAL harness observation
pipeline and emit training records in the exact corpus_v3 JSONL format the
existing (debugged) LoRA pipeline consumes.

Fidelity contract (verified against submission/_sft_k3/corpus_v3):
  * system prompt: built by the SAME `inference.agent.tool_agent._build_system_prompt`
    (editable install at submission/_adopt/taaf-src — the tree that produced the
    K3 corpus); byte-identical to corpus_v3's (asserted at build time).
  * frame images: real engine frames rendered by the SAME
    `inference.agent.vision_context.frame_to_png_data_url` (upscale=4 -> 256x256,
    matching corpus_v3).
  * user-turn text: same template lines as tool_agent._build_user_prompt
    (previous-sequence recap, state/valid-action lines, fixed tool-contract
    lines, carried world model, closing guidance, MOUSE row/col note).
  * target: one assistant message with a single `python` tool_call whose code
    calls `action([...])` (sandbox-accepted forms: action-name strings, or
    {'action': 'MOUSE', 'row': R, 'col': C}); tool responses carry
    {"tool": "python", "returncode": 0, "stdout": ...} where stdout is exactly
    what the code's print() would produce given the engine replay.
  * record shape: {"messages": [...], "target": {...}, "meta": {...}} with the
    harvest meta fields plus synthgen split axes (family, gen_seed, ...).

Difference from the K3 corpus (deliberate, documented): the synthetic teacher
is a mechanical reference solver, so assistant content/reasoning are templated
ground-truth annotations and turns are act-only (no exploratory analysis
sessions). Structure is identical; content style is simpler.

Output dir mirrors corpus_v3: train.jsonl, val.jsonl, holdout_transfer.jsonl,
tools.json, sft_common.py, tokenizer_bundle/, prep_stats.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

CORPUS_V3 = REPO_ROOT / "submission" / "_sft_k3" / "corpus_v3"
SFT_K3_DIR = REPO_ROOT / "submission" / "_sft_k3"
IMAGE_UPSCALE = 4  # corpus_v3 images are 256x256 for 64x64 grids

# The harness must see multimodal mode before building prompts.
os.environ.setdefault("MULTIMODAL_CONTEXT", "current_grid")

import inference  # noqa: E402  (editable install -> submission/_adopt/taaf-src)
from inference.agent.action_names import ENGINE_TO_MODEL_ACTION  # noqa: E402
from inference.agent.prompts import TOOL_CALL_FORMAT_GUIDANCE  # noqa: E402
from inference.agent.runtime_state import Frame  # noqa: E402
from inference.agent.tool_agent import _build_system_prompt  # noqa: E402
from inference.agent.vision_context import frame_to_png_data_url  # noqa: E402

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

RESULT_KEYS = ("board_changed", "level_completed", "game_over", "done", "level", "score")

# Fixed user-turn lines, verbatim from tool_agent._build_user_prompt.
_FIXED_LINES = [
    "Only tool: `python`. It receives `current_frame`, `previous_frame`, `history`, `transitions`, `last_transition`, `valid_actions`, `last_action_result`, and `action(actions)`.",
    "Only letter-coded board views and lightweight metadata are exposed; raw numeric color IDs are not available.",
    "Keep tool output compact: use `current_frame.segmentation` as the primary view, and `current_frame.ascii` only for a small specific region; never print full boards.",
    "For the most recent change, compare `previous_frame` to `current_frame`, or `last_transition.before_frame` to `last_transition.after_frame`; `history[-1].frame` is the current frame, not the previous one.",
    "Use Python to inspect the evidence, refine that world model from the newest history, and search or score candidate actions or short sequences against the current goal as you currently understand it.",
    "Maintain a compact working world model of what the current level seems to contain, what actions appear to do, what the goal seems to be, what is still uncertain, and what plan currently looks best.",
    "Below you are provided with the current world model from the previous turn. The default behavior is to copy it and add or remove things based on the evidence that you gathered. BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE WORLD MODEL.",
]
_STOP_LINE = (
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it, "
    "but stop immediately if a result reports `game_over`, `run_complete`, `level_completed`, or `done`."
)
_GROUND_LINE = "Ground yourself in `current_frame` before acting, but start with a compact structural summary rather than restating the full frame."
_FOCUS_LINE = "Focus on what changed most recently in `history`, update the target environment change if needed, and separate gameplay-object changes from HUD-only changes."
_CLOSING_LINES = [
    "When ready, call `action(actions)` from inside the `python` tool with the best valid action or ordered batch selected by your code. If your code has found a reliable short sequence, prefer batching it in one call.",
    "You may call `action(actions)` more than once in one Python snippet if your search or control loop needs it.",
    "If you include assistant text before a tool call, keep it short and use it to update the world model. Helpful optional prefixes are `World model:`, `Goal model:`, `Action model:`, `Recent findings:`, `Open questions:`, `Plan:`, and `Cross-level notes:`.",
    TOOL_CALL_FORMAT_GUIDANCE,
]
_MOUSE_LINE = "If you use MOUSE, include integer row and col arguments."


def system_prompt() -> str:
    prompt = _build_system_prompt(tool_output_tokens=1024)
    ref = CORPUS_V3 / "train.jsonl"
    if ref.exists():
        with ref.open() as fh:
            reference = json.loads(fh.readline())["messages"][0]["content"]
        if prompt != reference:
            raise RuntimeError(
                "harness system prompt no longer matches corpus_v3 — harness tree drifted; "
                "refusing to build a mixed-format corpus"
            )
    return prompt


def _knowledge_lines(world_model: str, plan: str) -> list[str]:
    entries = [("World model", world_model), ("Plan", plan)]
    items = [f"- {label}: {value}" for label, value in entries if value]
    if not items:
        return []
    return [
        "Working world model carried from earlier turns:",
        *items,
        "- Revise any item above immediately if `current_frame` or `history` contradicts it.",
    ]


def build_user_text(
    *,
    summary: dict[str, Any] | None,
    step: int,
    level: int,
    valid_line: str,
    knowledge: list[str],
    first_turn: bool,
    mouse: bool,
) -> str:
    lines: list[str] = []
    if summary:
        n = summary["executed_count"]
        label = "action" if n == 1 else "actions"
        lines.append(f"The code executed {n} {label} in the previous sequence.")
        acts = summary["executed_actions"]
        if acts:
            prefix = "Executed actions (first 10):" if len(acts) > 10 else "Executed actions:"
            lines.append(f"{prefix} {', '.join(acts[:10])}.")
        else:
            lines.append("Executed actions: none.")
        if summary.get("run_complete"):
            lines.append("You have completed the run!")
        elif summary.get("level_transition"):
            lines.append("You have progressed to a new level!")
        else:
            lines.append("You are still on the same level.")
        if summary.get("game_over"):
            lines.append("The game is over.")
    else:
        lines.append("No previous sequence has been executed yet.")
    lines.append(f"Current state: step {step}, level {level}.")
    lines.append(f"Valid actions right now: {valid_line}.")
    lines.extend(_FIXED_LINES)
    lines.append(_STOP_LINE)
    lines.extend(knowledge)
    lines.append("end of world model. ")
    lines.append(_GROUND_LINE if first_turn else _FOCUS_LINE)
    lines.extend(_CLOSING_LINES)
    if mouse:
        lines.append(_MOUSE_LINE)
    return "\n".join(lines)


def user_message(text: str, grid: list[list[int]], step: int, level: int) -> dict[str, Any]:
    frame = Frame(grid=tuple(tuple(int(v) for v in row) for row in grid), step=step, level=level)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{text}\n\nCurrent grid image:"},
            {
                "type": "image_url",
                "image_url": {"url": frame_to_png_data_url(frame, upscale=IMAGE_UPSCALE)},
            },
        ],
    }


def _model_action(action: dict[str, Any]) -> str | dict[str, Any]:
    name = ENGINE_TO_MODEL_ACTION[action["id"]]
    if action["id"] == "ACTION6":
        return {"action": "MOUSE", "row": int(action["y"]), "col": int(action["x"])}
    return name


def _display(action: dict[str, Any]) -> str:
    if action["id"] == "ACTION6":
        return f"MOUSE(row={int(action['y'])}, col={int(action['x'])})"
    return ENGINE_TO_MODEL_ACTION[action["id"]]


def teacher_code(chunk: list[dict[str, Any]]) -> str:
    acts = [_model_action(a) for a in chunk]
    keys = ", ".join(repr(k) for k in RESULT_KEYS)
    return (
        f"acts = {acts!r}\n"
        "r = action(acts)\n"
        f"print('result', {{k: r.get(k) for k in ({keys})}})\n"
    )


# ---------------------------------------------------------------------------
# ground-truth annotations (mechanical, from the spec — no LLM)
# ---------------------------------------------------------------------------


def _cells(grid_map: list[str], chars: str) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, row in enumerate(grid_map)
        for c, ch in enumerate(row)
        if ch in chars
    ]


def annotate(
    spec: dict[str, Any],
    level_idx: int,
    chunk: list[dict[str, Any]],
    remaining: int,
    pos: tuple[int, int] | None,
) -> tuple[str, str, str, str]:
    """Returns (content, reasoning, world_model, plan) for the coming turn."""
    family = spec["family"]
    lvl = spec["levels"][level_idx]
    names = ", ".join(_display(a) for a in chunk)
    if family == "nav":
        m = lvl["map"]
        goal = _cells(m, "G")[0]
        keys = _cells(m, "ab")
        doors = _cells(m, "ABS")
        switches = _cells(m, "s")
        hazards = _cells(m, "x")
        wm = (
            f"maze level {level_idx + 1}: player block at cell {pos}, goal tile at {goal}"
            + (f", keys at {keys} opening doors at {doors}" if keys else "")
            + (f", floor switches at {switches} opening doors at {doors}" if switches and not keys else "")
            + (f", {len(hazards)} lethal hazard tiles to avoid" if hazards else "")
            + ". Moves shift the player one cell; walls and closed doors block."
        )
        plan = f"follow the shortest safe path ({remaining} moves left); next: {names}."
        reasoning = (
            f"Path plan for level {level_idx + 1}:\n"
            f"- player cell {pos}, goal cell {goal}.\n"
            + (f"- gates: doors {doors} require keys {keys or switches} first.\n" if doors else "")
            + (f"- hazards at {hazards} are excluded from the path.\n" if hazards else "")
            + f"- executing {len(chunk)} planned move(s): {names}."
        )
    elif family == "click":
        rule = spec["rule"]
        objs = lvl["objects"]
        targets = [o for o in objs if o["target"]]
        wm = (
            f"selection level {level_idx + 1}: {len(objs)} objects; rule={rule}"
            + (
                f"; legend swatch marks the target color"
                if rule == "match_color"
                else f"; legend marks the target shape"
                if rule == "match_shape"
                else "; the single odd-colored object is the target"
            )
            + (
                "; arm button at bottom-right must be clicked before targets respond"
                if spec["gate"]
                else ""
            )
            + f"; {len(targets)} target(s) total, wrong clicks cost a life pip."
        )
        plan = f"click remaining targets in reading order; next: {names}."
        reasoning = (
            f"Click plan for level {level_idx + 1}:\n"
            f"- rule {rule}; targets at grid cells "
            + str([(o['y'], o['x']) for o in targets])
            + ".\n"
            + ("- gate button is clicked first to arm the targets.\n" if spec["gate"] else "")
            + f"- executing {len(chunk)} click(s): {names}."
        )
    else:  # push
        m = lvl["map"]
        boxes = _cells(m, "BO")
        pads = _cells(m, "TO")
        wm = (
            f"push level {level_idx + 1}: player at cell {pos}, boxes at {boxes}, target pads at {pads}. "
            "Walking into a box pushes it one cell if the cell behind is free; "
            "the level completes when every pad is covered."
        )
        plan = f"push boxes onto pads along the planned route ({remaining} moves left); next: {names}."
        reasoning = (
            f"Push plan for level {level_idx + 1}:\n"
            f"- player {pos}, boxes {boxes}, pads {pads}.\n"
            f"- order matters: the search avoids locking a box against walls.\n"
            f"- executing {len(chunk)} planned move(s): {names}."
        )
    content = f"World model: {wm}\nPlan: {plan}"
    return content, reasoning, wm, plan


def _track_pos(spec: dict[str, Any], level_idx: int) -> tuple[int, int] | None:
    if spec["family"] == "click":
        return None
    m = spec["levels"][level_idx]["map"]
    return _cells(m, "P")[0]


def _advance_pos(
    spec: dict[str, Any], pos: tuple[int, int] | None, action: dict[str, Any]
) -> tuple[int, int] | None:
    """Track the player's cell along the reference path (solver moves are never
    blocked, so applying the delta is exact)."""
    if pos is None:
        return None
    delta = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}
    d = delta.get(action["id"])
    return (pos[0] + d[0], pos[1] + d[1]) if d else pos


# ---------------------------------------------------------------------------
# episode -> records
# ---------------------------------------------------------------------------


def _chunk(seq: list[Any], rng: random.Random, lo: int, hi: int) -> list[list[Any]]:
    out, i = [], 0
    while i < len(seq):
        k = rng.randint(lo, hi)
        out.append(seq[i : i + k])
        i += k
    return out


def build_game_records(
    games_root: Path, game_dir: Path, seed: int, sys_prompt: str
) -> list[dict[str, Any]]:
    spec = json.loads((game_dir / "spec.json").read_text())
    solution = json.loads((game_dir / "solution.json").read_text())
    per_level = solution["per_level"]
    game_id = spec["game_id"]
    base_id = game_id.split("-")[0]
    total_levels = len(per_level)
    trace_len = sum(len(lvl) for lvl in per_level)
    rng = random.Random(f"corpus-{game_id}-{seed}")

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
    executed = 0
    score = 0
    analysis_step = 0
    records: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        system_msg,
        user_message(
            build_user_text(
                summary=None,
                step=1,
                level=1,
                valid_line=valid_line,
                knowledge=[],
                first_turn=True,
                mouse=mouse,
            ),
            grid,
            step=0,
            level=1,
        ),
    ]

    chunk_lo, chunk_hi = (1, 3) if spec["family"] == "click" else (2, 6)
    for level_idx, level_actions in enumerate(per_level):
        pos = _track_pos(spec, level_idx)
        chunks = _chunk(list(level_actions), rng, chunk_lo, chunk_hi)
        remaining = len(level_actions)
        for chunk in chunks:
            content, reasoning, wm, plan = annotate(spec, level_idx, chunk, remaining, pos)
            code = teacher_code(chunk)
            assistant = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "type": "function",
                        "index": 0,
                        "id": f"python_{analysis_step}",
                        "function": {"name": "python", "arguments": {"code": code}},
                    }
                ],
                "reasoning_content": reasoning,
            }
            first_action_num = executed + 1
            records.append(
                {
                    "messages": list(messages),
                    "target": assistant,
                    "meta": {
                        "game": base_id,
                        "level": level_idx + 1,
                        "actions_used": trace_len,
                        "baseline_actions": trace_len,
                        "teacher": "synthgen/reference-solver-v1",
                        "episode": f"synthgen_{game_id}",
                        "analysis_step": analysis_step,
                        "first_action_num": first_action_num,
                        "n_actions_turn": len(chunk),
                        "level_won": True,
                        "flags": [],
                        "dropped_turns": 0,
                        "n_images": sum(
                            1
                            for m in messages
                            if isinstance(m.get("content"), list)
                            and any(p.get("type") == "image_url" for p in m["content"])
                        ),
                        # synthgen split axes
                        "family": spec["family"],
                        "gen_seed": spec["seed"],
                        "gen_index": spec["index"],
                        "synthetic": True,
                    },
                }
            )

            # execute the chunk through the real engine
            board_changed_any = False
            prev_score = score
            for action in chunk:
                game_action = GameAction.from_name(action["id"])
                data = (
                    {"x": int(action["x"]), "y": int(action["y"])}
                    if action["id"] == "ACTION6"
                    else None
                )
                frame = env.step(game_action, data=data)
                if frame is None:
                    raise RuntimeError(f"step failed for {game_id}: {action}")
                new_grid = [list(map(int, row)) for row in frame.frame[-1]]
                board_changed_any = board_changed_any or new_grid != grid
                grid = new_grid
                executed += 1
                score = int(frame.levels_completed)
                pos = _advance_pos(spec, pos, action)
            remaining -= len(chunk)
            state = frame.state.name
            is_win = state == "WIN"
            level_num = total_levels if is_win else max(1, min(total_levels, score + 1))
            result = {
                "board_changed": board_changed_any,
                "level_completed": bool(score > prev_score and not is_win),
                "game_over": state == "GAME_OVER",
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
            analysis_step += 1

            is_last = level_idx == total_levels - 1 and remaining == 0
            if not is_last:
                summary = {
                    "executed_count": len(chunk),
                    "executed_actions": [_display(a) for a in chunk],
                    "level_transition": bool(score > prev_score and not is_win),
                    "run_complete": is_win,
                    "game_over": state == "GAME_OVER",
                }
                messages.append(
                    user_message(
                        build_user_text(
                            summary=summary,
                            step=executed + 1,
                            level=level_num,
                            valid_line=valid_line,
                            knowledge=_knowledge_lines(wm, plan),
                            first_turn=False,
                            mouse=mouse,
                        ),
                        grid,
                        step=executed,
                        level=level_num,
                    )
                )

    if frame.state.name != "WIN":
        raise RuntimeError(f"{game_id}: episode did not end in WIN (state={frame.state.name})")
    return records


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(CORPUS_V3 / "tokenizer_bundle"))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tokenizer unavailable ({exc}); skipping token stats", file=sys.stderr)
        return None


def _token_counts(tokenizer, tools, record) -> tuple[int | None, int | None]:
    """Text-token counts via the SAME sft_common.render path the kernel uses.
    (Image patch tokens are added later by the processor and are NOT counted.)"""
    if tokenizer is None:
        return None, None
    sys.path.insert(0, str(SFT_K3_DIR))
    try:
        from sft_common import render
    finally:
        sys.path.remove(str(SFT_K3_DIR))
    full_text, target_text = render(tokenizer, record["messages"], record["target"], tools)
    total = len(tokenizer(full_text, add_special_tokens=False).input_ids)
    target = len(tokenizer(target_text, add_special_tokens=False).input_ids)
    return total, target


def build_corpus(
    games_root: Path,
    out_dir: Path,
    holdout_family: str = "push",
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
        records = build_game_records(games_root, game_dir, seed, sys_prompt)
        for rec in records:
            total, target = _token_counts(tokenizer, tools, rec)
            rec["meta"]["qwen_total_text"] = total
            rec["meta"]["qwen_target_text"] = target
        family = records[0]["meta"]["family"]
        by_family.setdefault(family, []).append((game_id, records))
        print(f"  built {game_id} ({family}): {len(records)} records", flush=True)

    rng = random.Random(f"split-{seed}")
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
        for gid, recs in games[:n_val]:
            val_rows.extend(recs)
        for gid, recs in games[n_val:]:
            train_rows.extend(recs)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("train.jsonl", train_rows),
        ("val.jsonl", val_rows),
        ("holdout_transfer.jsonl", holdout_rows),
    ):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    # kernel-side support files, copied verbatim from the working corpus
    shutil.copy2(CORPUS_V3 / "tools.json", out_dir / "tools.json")
    shutil.copy2(SFT_K3_DIR / "sft_common.py", out_dir / "sft_common.py")
    bundle_dst = out_dir / "tokenizer_bundle"
    if not bundle_dst.exists():
        shutil.copytree(CORPUS_V3 / "tokenizer_bundle", bundle_dst)

    def _tok_stats(rows: list[dict]) -> dict[str, Any]:
        totals = [r["meta"]["qwen_total_text"] for r in rows if r["meta"].get("qwen_total_text")]
        targets = [r["meta"]["qwen_target_text"] for r in rows if r["meta"].get("qwen_target_text")]
        if not totals:
            return {}
        totals_sorted = sorted(totals)
        return {
            "n": len(totals),
            "total_min": totals_sorted[0],
            "total_median": totals_sorted[len(totals_sorted) // 2],
            "total_max": totals_sorted[-1],
            "total_mean": round(sum(totals) / len(totals), 1),
            "target_mean": round(sum(targets) / len(targets), 1),
        }

    stats = {
        "generator": "synthgen_v1",
        "split_unit": "game",
        "holdout_family": holdout_family,
        "n_games": len(valid_ids),
        "games_per_family": {f: len(g) for f, g in sorted(by_family.items())},
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "holdout_transfer": len(holdout_rows),
        },
        "token_stats_text_only": {
            "train": _tok_stats(train_rows),
            "val": _tok_stats(val_rows),
            "holdout_transfer": _tok_stats(holdout_rows),
        },
        "note": (
            "qwen_*_text counts exclude image patch tokens (processor adds them); "
            "corpus_v3 qwen_total included them"
        ),
    }
    (out_dir / "prep_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=Path, default=_HERE / "out" / "games")
    ap.add_argument("--out", type=Path, default=_HERE / "out" / "corpus_synth_v1")
    ap.add_argument("--holdout-family", choices=("nav", "click", "push"), default="push")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-tokens", action="store_true")
    args = ap.parse_args()
    stats = build_corpus(
        args.games,
        args.out,
        holdout_family=args.holdout_family,
        val_frac=args.val_frac,
        seed=args.seed,
        with_tokens=not args.no_tokens,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
