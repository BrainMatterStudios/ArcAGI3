"""traces_to_grpo.py — captured duck traces -> GRPO training samples (SKELETON).

Input: one or more episode dirs (each with manifest.json + trace.jsonl from
run_rollout.py). Each trace record is one /chat/completions pair; the duck
resends its full message history every call, so each record already carries
(state = request.messages prefix, action = response assistant message).

Output: <out>/grpo_samples.jsonl — one record per assistant turn, grouped by
episode (rollout_id), episode-level reward attached — plus a token/image
summary on stdout.

IMPORTANT — this is generic-JSON, NOT yet a trainer's native format. The
trainer-specific conversion is stubbed (see TODOs): verl and friends want
token-level (input_ids, response_ids, loss masks) produced with the SAME chat
template + image processor the rollout server used. Images: any multimodal
part is preserved verbatim here and loudly WARNED about — dropping them would
train on a state the policy never saw.

Usage:
  .venv/bin/python scratchpad/rl_gate/traces_to_grpo.py EPISODE_DIR [EPISODE_DIR ...] \
      [--out scratchpad/rl_gate/grpo_out]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_image_parts(messages: list) -> int:
    n = 0
    for msg in messages or []:
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for part in content
                     if isinstance(part, dict) and part.get("type") == "image_url")
    return n


def convert_episode(episode_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    trace_path = Path(manifest.get("trace_path") or episode_dir / "trace.jsonl")
    samples: list[dict] = []
    stats = {"episode": manifest["rollout_id"], "reward": manifest["reward"],
             "turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "image_parts": 0}

    with open(trace_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("status") != 200 or not rec.get("response"):
                continue  # failed calls are not policy actions
            request, response = rec["request"], rec["response"]
            choice = (response.get("choices") or [{}])[0]
            usage = response.get("usage") or {}
            n_images = _count_image_parts(request.get("messages", []))
            stats["turns"] += 1
            stats["image_parts"] += n_images
            stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            samples.append({
                "schema": "rl_gate.grpo_sample.v1",
                "episode": rec.get("rollout_id") or manifest["rollout_id"],
                "turn": rec.get("seq"),
                "reward": manifest["reward"],  # episode-level; per-turn shaping TODO below
                # state: the exact prompt the policy saw (messages prefix incl. images)
                "state_messages": request.get("messages", []),
                # action: what the policy emitted (text + tool_calls)
                "action_message": choice.get("message", {}),
                "finish_reason": choice.get("finish_reason"),
                "sampling": {k: request.get(k) for k in
                             ("temperature", "top_p", "top_k", "max_tokens", "seed")
                             if k in request},
                "model": request.get("model"),
                "tools": request.get("tools"),
                "usage": usage,
                "n_image_parts": n_images,
            })
    return samples, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes", nargs="+", help="episode dirs from run_rollout.py")
    ap.add_argument("--out", default=str(Path(__file__).parent / "grpo_out"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grpo_samples.jsonl"

    all_stats, total_images, total_samples = [], 0, 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for ep in args.episodes:
            samples, stats = convert_episode(Path(ep))
            all_stats.append(stats)
            total_images += stats["image_parts"]
            total_samples += len(samples)
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[grpo] wrote {total_samples} samples from {len(all_stats)} episode(s) -> {out_path}")
    for st in all_stats:
        print(f"[grpo]   {st['episode']}: reward={st['reward']} turns={st['turns']} "
              f"prompt_tok={st['prompt_tokens']} completion_tok={st['completion_tokens']} "
              f"images={st['image_parts']}")
    if total_images:
        print(f"[grpo] WARNING: {total_images} image_url parts present in states. "
              "The trainer MUST tokenize these through the model's image processor "
              "(multimodal RL). Dropping them = training on states the policy never saw.")

    # ------------------------------------------------------------------ TODOs
    # TODO(trainer-format): verl multi-turn expects token-level tensors, not chat
    #   JSON. Required steps once the trainer is chosen (see STAGE_A_PLAN.md):
    #   1. Re-tokenize state_messages with the EXACT chat template vLLM served
    #      (tokenizer.apply_chat_template + AutoProcessor for image parts) and
    #      verify round-trip token counts against usage.prompt_tokens — any
    #      mismatch means the trainer would compute log-probs on tokens the
    #      policy never emitted (token-faithfulness check, Stage A criterion).
    #   2. Build loss masks: loss only on the assistant completion tokens of
    #      each turn (tool/user/system tokens masked out).
    #   3. GRPO grouping: group = all episodes of the SAME game from the same
    #      prompt state; advantage = (reward - group_mean)/group_std. With one
    #      episode per game this degenerates — Stage A must run >=4 rollouts
    #      per game to form a group.
    #   4. Per-turn credit: episode-level levels_completed is attached to every
    #      turn here (trajectory-level GRPO). Optional shaping (level-delta per
    #      turn) can be recovered from the duck's tool payloads if needed.
    # TODO(images): confirm chosen trainer accepts interleaved image_url content
    #   in multi-turn rollout records; else pre-encode pixel_values per sample.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
