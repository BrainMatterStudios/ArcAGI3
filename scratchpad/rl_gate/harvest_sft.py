"""harvest_sft.py — turn winning teacher (Kimi K3) duck-harness traces into SFT samples.

TRACE / TURN RECONSTRUCTION RULE (derived from tool_agent.py `analyze()` + real traces)
--------------------------------------------------------------------------------------
The duck harness runs one `analyze()` call per "turn" (analysis_step). Inside a turn it
issues SEVERAL chat.completions requests (inner python tool loop: assistant tool_call ->
tool result -> next request). A turn ends the moment a python call executes real env
action(s) (`step_executed`), so the ACTED-ON assistant response is always the response of
the FINAL request of its turn, and that request carries the fullest message prefix.

The raw trace is NOT a clean chain:
  * status-200 bodies may be provider errors ({"error":...} with no "choices") -> dropped.
  * client-side analyzer timeouts abandon in-flight requests; the harness RETRIES the same
    analysis_step with restored history, while the abandoned request still completes at the
    proxy later -> DEAD BRANCH records interleaved by ts (seq is response-arrival order).
Dead responses were never received by the agent, so their code never ran -> they can never
have executed actions.

We therefore anchor reconstruction on the solver's viewer event log
(artifacts/viewer_data_events.jsonl): every executed action is an event carrying
analysis_step + action_num + level (+ level_completed / run_complete). Turn grouping =
non-RESET action events grouped by analysis_step. Each turn's final request is found by:
  * every request's last user message embeds "Current state: step S, level L" where S is
    the next action number -> S must equal the turn's first executed action_num;
  * candidate responses must contain an `action(` python call;
  * candidates must have been RECEIVED: their tool_call argument strings reappear verbatim
    in a later request's stored assistant messages (harness persists accepted responses),
    OR be the final valid record of the episode (nothing after to reference them);
  * among several received candidates with the same S (failed-action attempt then retry),
    the LAST by ts_response is the one that actually executed (execution ends the turn).

LEVEL BOUNDARY RULE
-------------------
Action events report the level AFTER the action: a completing action has
level_completed=True and level = L+1 (its own level is L); the final WIN action has
run_complete=True / state=WIN at level = last level. So:
  level_of(event) = event.level - 1 if event.level_completed else event.level
  completed levels = {e.level-1 : e.level_completed} | {e.level : e.run_complete & WIN}
A turn belongs to level_of(first event). Cross-checked against the L parsed from the
request's "Current state" line.

Usage:  .venv/bin/python scratchpad/rl_gate/harvest_sft.py [--episodes k3_ft09 ...]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EPISODES = HERE / "episodes"
OUT = HERE / "sft_data"
WIN_EPISODES = ["k3_ft09", "k3_batch1_sb26", "k3_batch1_su15",
                "k3_batch1_vc33", "k3_batch1_cd82", "k3_batch1_lp85"]
STATE_RE = re.compile(r"Current state: step (\d+), level (\d+)")
MAX_TOKENS = 32768


# ---------------------------------------------------------------- loading
def load_records(ep_dir: Path) -> list[dict]:
    """Valid model exchanges, seq order. Drops provider-error bodies (no choices)."""
    recs = []
    for line in (ep_dir / "trace.jsonl").open():
        r = json.loads(line)
        if (r.get("response") or {}).get("choices"):
            recs.append(r)
    recs.sort(key=lambda r: r["seq"])
    return recs


def load_events(ep_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (ep_dir / "artifacts" / "viewer_data_events.jsonl").open()]


def baseline_actions(game: str) -> list[int] | None:
    for meta in (REPO / "environment_files" / game).glob("*/metadata.json"):
        return json.loads(meta.read_text()).get("baseline_actions")
    return None


# ---------------------------------------------------------------- trace helpers
def resp_msg(r: dict) -> dict:
    return r["response"]["choices"][0]["message"]


def tc_args(msg: dict) -> tuple[str, ...]:
    return tuple(tc["function"]["arguments"] for tc in (msg.get("tool_calls") or []))


def code_of(msg: dict) -> str:
    out = []
    for a in tc_args(msg):
        try:
            out.append(json.loads(a).get("code", ""))
        except (json.JSONDecodeError, AttributeError):
            out.append(a)
    return "\n".join(out)


def state_line(r: dict) -> tuple[int, int] | None:
    """(step S, level L) from the last user message of the request."""
    for m in reversed(r["request"]["messages"]):
        if m["role"] != "user":
            continue
        c = m.get("content")
        texts = [c] if isinstance(c, str) else [p.get("text", "") for p in c if p.get("type") == "text"]
        hit = STATE_RE.search("\n".join(t or "" for t in texts))
        return (int(hit.group(1)), int(hit.group(2))) if hit else None
    return None


def count_images(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") == "image_url")


# ---------------------------------------------------------------- reconstruction
def level_of(ev: dict) -> int:
    return ev["level"] - 1 if ev.get("level_completed") else ev["level"]


def reconstruct(records: list[dict], events: list[dict]) -> tuple[list[dict], set[int], list[str]]:
    """-> (turns, completed_levels, warnings); turn = {astep, level, events, record, flags}."""
    warnings: list[str] = []
    acts = [e for e in events if e.get("type") == "action" and e.get("action_name") != "RESET"]
    groups: dict[int, list[dict]] = defaultdict(list)
    for e in acts:
        groups[e["analysis_step"]].append(e)

    completed = {e["level"] - 1 for e in acts if e.get("level_completed")}
    completed |= {e["level"] for e in acts if e.get("run_complete") and e.get("state") == "WIN"}

    stored = {a for r in records for m in r["request"]["messages"]
              if m["role"] == "assistant" for a in tc_args(m)}
    last_seq = records[-1]["seq"] if records else -1

    def received(r: dict) -> bool:
        args = tc_args(resp_msg(r))
        return bool(args) and all(a in stored for a in args) or r["seq"] == last_seq

    cands: dict[int, list[dict]] = defaultdict(list)  # first-action-num -> records
    for r in records:
        st = state_line(r)
        if st and "action(" in code_of(resp_msg(r)) and (received(r) or r["seq"] == last_seq):
            cands[st[0]].append(r)

    turns = []
    for astep in sorted(groups):
        evs = sorted(groups[astep], key=lambda e: e["action_num"])
        first_anum, lvl = evs[0]["action_num"], level_of(evs[0])
        flags = []
        match = sorted(cands.get(first_anum, []), key=lambda r: r["ts_response"])
        if not match:
            warnings.append(f"astep {astep} (action {first_anum}): no matching trace record — turn dropped")
            continue
        if len(match) > 1:
            flags.append("multi_attempt_turn")
        rec = match[-1]
        st = state_line(rec)
        if st and st[1] != lvl:
            flags.append(f"level_mismatch_prompt_{st[1]}_events_{lvl}")
            warnings.append(f"astep {astep}: prompt level {st[1]} != event level {lvl}")
        turns.append({"astep": astep, "level": lvl, "events": evs, "record": rec, "flags": flags})
    return turns, completed, warnings


# ---------------------------------------------------------------- sample emission
def harvest_episode(name: str) -> dict:
    ep_dir = EPISODES / name
    manifest = json.loads((ep_dir / "manifest.json").read_text())
    records, events = load_records(ep_dir), load_events(ep_dir)
    turns, completed, warnings = reconstruct(records, events)
    game = manifest["game"]
    base = baseline_actions(game)
    acts = [e for e in events if e.get("type") == "action"]
    actions_by_level: dict[int, int] = defaultdict(int)
    for e in acts:
        actions_by_level[level_of(e)] += 1

    samples = []
    for t in turns:
        rec, msg = t["record"], resp_msg(t["record"])
        flags = list(t["flags"])
        prefix = copy.deepcopy(rec["request"]["messages"])
        if any("You have not acted yet" in str(m.get("content", "")) for m in prefix):
            flags.append("no_tool_call_retry_in_prefix")
        usage = rec["response"].get("usage") or {}
        tokens = usage.get("total_tokens", 0)
        if tokens > MAX_TOKENS:
            flags.append("over_32768_tokens")
        target = {k: msg.get(k) for k in ("role", "content", "tool_calls", "reasoning") if msg.get(k) is not None}
        lvl = t["level"]
        samples.append({
            "messages": prefix,
            "target": target,
            "meta": {
                "game": game, "level": lvl,
                "actions_used": actions_by_level[lvl],
                "baseline_actions": (base[lvl - 1] if base and lvl <= len(base) else None),
                "teacher": rec["request"].get("model", "unknown"),
                "episode": name, "analysis_step": t["astep"],
                "first_action_num": t["events"][0]["action_num"],
                "n_actions_turn": len(t["events"]),
                "tokens": tokens, "images": count_images(prefix),
                "level_won": lvl in completed, "flags": flags,
            },
        })
    return {"name": name, "game": game, "manifest": manifest, "samples": samples,
            "completed_levels": sorted(completed), "warnings": warnings,
            "n_turns_events": len({e["analysis_step"] for e in acts if e.get("action_name") != "RESET"})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", nargs="*", default=WIN_EPISODES)
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    all_wins, stats_rows, total_warn = [], [], []
    for name in args.episodes:
        ep = harvest_episode(name)
        raw_path = OUT / f"{name}.jsonl"
        with raw_path.open("w") as f:
            for s in ep["samples"]:
                f.write(json.dumps(s) + "\n")
        wins = [s for s in ep["samples"] if s["meta"]["level_won"]]
        all_wins.extend(wins)
        total_warn += [f"{name}: {w}" for w in ep["warnings"]]
        per_level: dict[int, list[dict]] = defaultdict(list)
        for s in ep["samples"]:
            per_level[s["meta"]["level"]].append(s)
        for lvl in sorted(per_level):
            ss = per_level[lvl]
            m = ss[0]["meta"]
            stats_rows.append({
                "episode": name, "game": ep["game"], "level": lvl,
                "won": m["level_won"], "samples": len(ss),
                "tokens": sum(s["meta"]["tokens"] for s in ss),
                "images": sum(s["meta"]["images"] for s in ss),
                "actions_used": m["actions_used"], "baseline_actions": m["baseline_actions"],
                "flagged": sum(1 for s in ss if s["meta"]["flags"]),
                "over_32768": sum(1 for s in ss if "over_32768_tokens" in s["meta"]["flags"]),
            })
        print(f"[{name}] {ep['game']}: {len(ep['samples'])} acted turns "
              f"(events expect {ep['n_turns_events']}), win levels {ep['completed_levels']}, "
              f"{len(wins)} win samples -> {raw_path.name}")
        for w in ep["warnings"]:
            print(f"  WARN {w}")

    with (OUT / "all_wins.jsonl").open("w") as f:
        for s in all_wins:
            f.write(json.dumps(s) + "\n")

    stats = {
        "total_samples_raw": sum(r["samples"] for r in stats_rows),
        "total_samples_wins": len(all_wins),
        "total_tokens_wins": sum(s["meta"]["tokens"] for s in all_wins),
        "over_32768_wins": sum(1 for s in all_wins if "over_32768_tokens" in s["meta"]["flags"]),
        "warnings": total_warn, "per_level": stats_rows,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))

    hdr = f"{'episode':<18}{'game':<6}{'lvl':>4}{'won':>5}{'smp':>5}{'tokens':>9}{'imgs':>6}{'acts':>6}{'base':>6}{'>32k':>6}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in stats_rows:
        print(f"{r['episode']:<18}{r['game']:<6}{r['level']:>4}{str(r['won']):>5}{r['samples']:>5}"
              f"{r['tokens']:>9}{r['images']:>6}{r['actions_used']:>6}{str(r['baseline_actions']):>6}{r['over_32768']:>6}")
    print(f"\nTOTAL raw={stats['total_samples_raw']} wins={stats['total_samples_wins']} "
          f"win_tokens={stats['total_tokens_wins']} over32k(wins)={stats['over_32768_wins']}")
    print(f"stats -> {OUT / 'stats.json'}; combined -> {OUT / 'all_wins.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
