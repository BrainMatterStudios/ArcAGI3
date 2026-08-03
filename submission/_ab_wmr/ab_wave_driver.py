# ============================================================================
# A/B wave driver — ROUND 3: the mechanic-archetype PLAYBOOK (patch13) as the
# single variable, on the real bundled Qwen3.6-27B duck.
#
# Arms (env pins only; the patch layer is identical everywhere — apply_all(),
# exactly what the submitted v6 kernel installs, now including patch13/14):
#   A = current pins: WMR trio ON, TAAF_GRAPH=0, TAAF_COMPACT=0,
#       TAAF_PLAYBOOK=0, TAAF_ANTIFREEZE=1 (patch14 point fix, BOTH arms —
#       like the HUD slow-tick rotation fix it is a default now, so the
#       playbook isolates cleanly).
#   B = A + TAAF_PLAYBOOK=1 (patch13: archetype playbook appended to the
#       system prompt — the 2026-08-03 unlock diagnosis's #1 intervention).
#
# Design (mechanism carried wholesale from rounds 1-2, commits 3c21726/ae1286b):
#   * ONE kernel session, ONE vLLM server. duck_patches.py provably reads every
#     switch AT CALL TIME (_playbook_enabled/_antifreeze_enabled/
#     _watchdog_enabled/... all read os.environ per call), so toggling between
#     waves in one process is a genuine arm switch. Analyzers (ToolAgent) are
#     constructed per game INSIDE each wave, so patch13's system-prompt
#     injection follows the wave's env.
#   * Waves A,B,B,A (mirrored counterbalance): each arm's two runs average to
#     the same mean wave index (A:1.5, B:1.5), cancelling linear server drift
#     deterministically. Realized order is logged (law 5).
#   * Per the A/A noise floor (RMS 0.707 levels/game-run), score deltas at
#     2 waves/arm are NOISE — see pre_registered_reading in the result JSON:
#     primary readout is paired per-game L1-UNLOCK EVENTS on the 8 target
#     games, secondary is antifreeze triggers + m0r0 mask confirmation.
#   * "An env toggle is not a shipped arm": every wave logs positive runtime
#     proof of its toggle state, a patch-proof snapshot, AND a playbook
#     presence proof (the playbook text asserted present/absent in real
#     system prompts built during the wave — direct probe BEFORE the wave,
#     sampled in-run prompts after). The serving assert proves the 27B is
#     genuinely served BEFORE any game runs.
#
# Round-3 instrumentation additions (on top of round 2's fixed gen_tokens
# accounting, HUD/watchdog/replay diagnostics, compact/graph counters):
#   * patch14 antifreeze diagnostics: per-wave ANTIFREEZE_DIAGNOSTICS delta +
#     per-game trigger counts (module-global _antifreeze_note is wrapped to
#     attribute triggers to the calling agent; tr87 is the predicted site).
#   * HUD mask stats were already per-row (mask_cells/confirmed_lines); the
#     slow-tick rotation fix should now confirm masks on m0r0.
#   * playbook presence proof per wave (see above), logged like toggle proof.
#
# This file is BOTH inlined into the kernel's run cell by build_ab_wmr.py AND
# imported by dry_run.py for the GPU-free local test. Edit here, then rebuild.
# ============================================================================
import glob as _ab_glob
import hashlib as _ab_hashlib
import json as _ab_json
import os as _ab_os
import sys as _ab_sys
import time as _ab_time
import traceback as _ab_traceback
import urllib.request as _ab_urlreq

_AB_WMR_TRIO = {"TAAF_WATCHDOG": "1", "TAAF_HUD_MASK": "1", "TAAF_WIN_REPLAY": "1",
                "TAAF_GRID_BURNER": "0"}
# A is the current pin set: v6 EXPERIMENT_ENV (GRAPH=0 COMPACT=0 PLAYBOOK=0)
# plus the two point fixes both arms inherit as defaults (HUD rotation fix has
# no switch; patch14 antifreeze pinned ON explicitly for the toggle proof).
_AB_ROUND3_BASE = {**_AB_WMR_TRIO, "TAAF_GRAPH": "0", "TAAF_COMPACT": "0",
                   "TAAF_ANTIFREEZE": "1"}
AB_ARM_ENV = {
    "A": {**_AB_ROUND3_BASE, "TAAF_PLAYBOOK": "0"},
    "B": {**_AB_ROUND3_BASE, "TAAF_PLAYBOOK": "1"},
}

# Panel (10 games) — NEW for round 3, diagnosis-targeted (docs/test-artifacts-
# 2026-08-02/UNLOCK-DIAGNOSIS-9-GAMES-2026-08-03.md): 8 never-unlocked target
# games the playbook PREDICTS unlocks on (sk48 also gets its first
# post-ACTION7-fix measurement) + 2 continuity games from rounds 1-2.
# dc22 deliberately EXCLUDED: capability-blocked (the K3 teacher fails it too).
_AB_TARGET_GAMES = ("cn04", "lf52", "ls20", "m0r0", "wa30", "g50t", "tr87", "sk48")
_AB_CONTINUITY_GAMES = ("ft09", "re86")
AB_DEFAULT_GAMES = ",".join(_AB_TARGET_GAMES + _AB_CONTINUITY_GAMES)


def _ab_cfg(name: str, default: str) -> str:
    return _ab_os.environ.get(name, "").strip() or default


AB_GAMES = [s.strip() for s in _ab_cfg("AB_GAMES", AB_DEFAULT_GAMES).split(",") if s.strip()]
AB_WAVES = [w.strip().upper() for w in _ab_cfg("AB_WAVES", "A,B,B,A").split(",") if w.strip()]
AB_BUDGET = float(_ab_cfg("AB_BUDGET", "3600"))          # per-game wall cap, seconds
AB_DEADLINE_S = float(_ab_cfg("AB_DEADLINE_S", str(int(8 * 3600))))  # from notebook start
AB_WAVE_OVERHEAD_S = 900.0                               # server start + collect + slack
AB_DRY_RUN = _ab_os.environ.get("AB_DRY_RUN", "") == "1"

AB_SESSIONS = []                # (wave_index, session) — filled by the registry wrapper
_AB_SYSPROMPTS = []             # (wave_index, prompt) — filled by the prompt recorder
_AB_WAVE = {"i": -1}

# Pre-registered reading — written verbatim into the result JSON header so the
# analysis cannot quietly move the goalposts after the data lands.
AB_PREREGISTERED_READING = {
    "primary": (
        "Paired per-game L1-unlock events on the 8 target games "
        f"{list(_AB_TARGET_GAMES)}: B unlocking a target game A never does = "
        "signal. The diagnosis predicts lf52/wa30/cn04 reachable and the ls20 "
        "rate up; g50t/tr87/sk48 are stretch; dc22 is excluded as "
        "capability-blocked (the K3 teacher fails it too)."
    ),
    "secondary": (
        "patch14 antifreeze trigger counts and any freeze-broken game (tr87 is "
        "the predicted trigger site); m0r0 HUD mask confirmation "
        "(mask_cells/confirmed_lines) after the slow-tick rotation fix; sk48's "
        "first post-ACTION7-fix measurement."
    ),
    "not_a_readout": (
        "Score deltas at 2 waves/arm are noise (A/A floor RMS 0.707 "
        "levels/game-run) — do not headline them."
    ),
}


def _ab_patch_module():
    """The namespace holding duck_patches' module-level objects.

    In the kernel the patch layer is inlined into the notebook, so
    graph_diagnostics / COMPACT_DIAGNOSTICS live in __main__; in the dry run
    they live in the synthetic 'duck_patches' module."""
    mod = _ab_sys.modules.get("duck_patches")
    if mod is not None and hasattr(mod, "COMPACT_DIAGNOSTICS"):
        return mod
    main = _ab_sys.modules.get("__main__")
    if main is not None and hasattr(main, "COMPACT_DIAGNOSTICS"):
        return main
    return None


# --- toggle semantics: EXACTLY duck_patches.py's readers -----------------------


def _ab_toggles():
    def default_on(name):
        return _ab_os.environ.get(name, "1").strip() not in {"0", "false", "False"}

    def opt_in(name):  # grid burner only: _grid_burner_enabled() is opt-in
        return _ab_os.environ.get(name, "0").strip() in {"1", "true", "True"}

    return {
        "TAAF_WATCHDOG": default_on("TAAF_WATCHDOG"),
        "TAAF_HUD_MASK": default_on("TAAF_HUD_MASK"),
        "TAAF_WIN_REPLAY": default_on("TAAF_WIN_REPLAY"),
        "TAAF_GRAPH": default_on("TAAF_GRAPH"),
        "TAAF_COMPACT": default_on("TAAF_COMPACT"),
        "TAAF_PLAYBOOK": default_on("TAAF_PLAYBOOK"),
        "TAAF_ANTIFREEZE": default_on("TAAF_ANTIFREEZE"),
        "TAAF_GRID_BURNER": opt_in("TAAF_GRID_BURNER"),
    }


def _ab_expected_toggles(arm):
    return {k: v == "1" for k, v in AB_ARM_ENV[arm].items()}


def _ab_patch_proof():
    """Positive proof the full patch layer is installed on the live classes.

    Session-class play/_execute_action markers are read BEFORE the registry
    wrapper re-wraps them (behav_probe already hides inner markers on
    _execute_action, hence the or-chain, as in round 1). ToolAgent and
    step_env are never wrapped by probe/registry, so their markers stay
    readable all run and are re-snapshotted per wave.
    """
    from inference.agent import tool_agent as _ta
    from inference.framework import solver as _sv

    cls = _sv._HarnessGameSession
    agent_cls = _ta.ToolAgent
    return {
        "watchdog_should_stop_patched": bool(getattr(cls.should_stop, "_watchdog_patched", False)),
        "graph_should_stop_patched": bool(getattr(cls.should_stop, "_graph_patched", False)),
        "graph_step_env_patched": bool(getattr(cls.step_env, "_graph_patched", False)),
        "hud_or_outer_execute_patched": bool(getattr(cls._execute_action, "_hud_patched", False)
                                             or getattr(cls._execute_action, "_win_replay_patched", False)
                                             or getattr(cls._execute_action, "_graph_patched", False)
                                             or getattr(cls._execute_action, "_behav", False)),
        "play_patched": bool(getattr(cls.play, "_win_replay_patched", False)
                             or getattr(cls.play, "_watchdog_play_patched", False)),
        "plan_queue_analyze_patched": bool(getattr(agent_cls.analyze, "_plan_queue_patched", False)),
        "compaction_history_patched": bool(
            getattr(agent_cls._persistent_history_messages, "_compact12_patched", False)),
        "compact_prompt_injector_patched": bool(
            getattr(agent_cls._build_user_prompt, "_compact12_patched", False)),
        "playbook_system_prompt_patched": bool(
            getattr(_ta._build_system_prompt, "_playbook_patched", False)),
        "antifreeze_user_prompt_patched": bool(
            getattr(agent_cls._build_user_prompt, "_antifreeze_patched", False)),
        "execute_chain_repr": repr(cls._execute_action),
        "play_chain_repr": repr(cls.play),
    }


def _ab_wave_patch_proof():
    """Per-arm re-proof on the surfaces the registry wrapper does not touch."""
    from inference.agent import tool_agent as _ta
    from inference.framework import solver as _sv

    cls = _sv._HarnessGameSession
    agent_cls = _ta.ToolAgent
    return {
        "graph_step_env_patched": bool(getattr(cls.step_env, "_graph_patched", False)),
        "watchdog_should_stop_patched": bool(getattr(cls.should_stop, "_watchdog_patched", False)),
        "plan_queue_analyze_patched": bool(getattr(agent_cls.analyze, "_plan_queue_patched", False)),
        "compaction_history_patched": bool(
            getattr(agent_cls._persistent_history_messages, "_compact12_patched", False)),
        "playbook_system_prompt_patched": bool(
            getattr(_ta._build_system_prompt, "_playbook_patched", False)),
        "antifreeze_user_prompt_patched": bool(
            getattr(agent_cls._build_user_prompt, "_antifreeze_patched", False)),
    }


def _ab_install_registry():
    """Outermost wrappers: record sessions per wave + count HUD suppressions."""
    from inference.framework import solver as _sv

    cls = _sv._HarnessGameSession
    if getattr(cls.play, "_ab_registered", False):
        return
    orig_play = cls.play
    orig_exec = cls._execute_action

    def play(self):
        AB_SESSIONS.append((_AB_WAVE["i"], self))
        return orig_play(self)

    def _execute_action(self, action, *args, **kwargs):
        payload = orig_exec(self, action, *args, **kwargs)
        try:
            if isinstance(payload, dict) and payload.get("board_changed_hud_only"):
                self._ab_hud_suppressed = getattr(self, "_ab_hud_suppressed", 0) + 1
        except Exception:
            pass
        return payload

    play._ab_registered = True
    _execute_action._ab_registered = True
    cls.play = play
    cls._execute_action = _execute_action


# --- round 3: playbook presence proof -------------------------------------------


def _ab_playbook_marker():
    """The playbook's identifying text, read from the live patch layer when
    possible so the proof cannot drift from what patch13 actually injects."""
    mod = _ab_patch_module()
    text = getattr(mod, "_PLAYBOOK_TEXT", None) if mod else None
    if isinstance(text, str) and text.strip():
        return text.splitlines()[0].strip()
    return "Mechanic playbook"


def _ab_install_prompt_recorder():
    """Outermost wrapper on tool_agent._build_system_prompt: record every real
    system prompt built during play, tagged with the current wave index.
    Installed AFTER apply_all(), so it observes patch13's output. Patch markers
    are forwarded so per-wave patch proofs still see _playbook_patched."""
    from inference.agent import tool_agent as _ta

    builder = _ta._build_system_prompt
    if getattr(builder, "_ab_recorded", False):
        return

    def _build_system_prompt(*args, **kwargs):
        prompt = builder(*args, **kwargs)
        _AB_SYSPROMPTS.append((_AB_WAVE["i"], prompt))
        return prompt

    for k, v in vars(builder).items():
        if k.endswith("_patched"):
            setattr(_build_system_prompt, k, v)
    _build_system_prompt._ab_recorded = True
    _ta._build_system_prompt = _build_system_prompt


def _ab_playbook_probe(arm):
    """Direct pre-wave probe: build one real system prompt through the patched
    builder and hard-assert playbook presence matches the arm BEFORE any GPU
    minute is spent on a mis-toggled wave (toggle-proof law)."""
    from inference.agent import tool_agent as _ta

    prompt = _ta._build_system_prompt(tool_output_tokens=2000)
    marker = _ab_playbook_marker()
    present = marker in prompt
    expected = AB_ARM_ENV[arm].get("TAAF_PLAYBOOK") == "1"
    if present != expected:
        raise RuntimeError(
            f"arm {arm} playbook probe mismatch: marker {marker!r} present={present} "
            f"expected={expected} (prompt tail: ...{prompt[-300:]!r})")
    return {"marker": marker, "present": present, "expected": expected,
            "prompt_sha256": _ab_hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_chars": len(prompt)}


# --- round 3: per-agent antifreeze trigger attribution ---------------------------


def _ab_install_antifreeze_counter():
    """Wrap the patch module's _antifreeze_note (the module-global patch14's
    _build_user_prompt wrapper resolves at call time) so each fired breaker is
    also attributed to the calling agent. Observational only — the note text
    is returned unchanged."""
    mod = _ab_patch_module()
    if mod is None or not callable(getattr(mod, "_antifreeze_note", None)):
        return False
    orig = mod._antifreeze_note
    if getattr(orig, "_ab_counted", False):
        return True

    def _antifreeze_note(agent, current_frame):
        note = orig(agent, current_frame)
        if note:
            try:
                agent.__dict__["_ab_antifreeze_triggers"] = (
                    agent.__dict__.get("_ab_antifreeze_triggers", 0) + 1)
            except Exception:
                pass
        return note

    _antifreeze_note._ab_counted = True
    mod._antifreeze_note = _antifreeze_note
    return True


def _ab_antifreeze_snapshot():
    mod = _ab_patch_module()
    diag = getattr(mod, "ANTIFREEZE_DIAGNOSTICS", None) if mod else None
    if not isinstance(diag, dict):
        return None
    return {"triggers": int(diag.get("triggers", 0) or 0)}


# --- serving assert ------------------------------------------------------------


def ab_serving_assert(working_dir):
    """Prove the 27B is actually served BEFORE any game runs (2026-07-31 law).

    Three checks: (1) /models answers with the expected served id; (2) a real
    chat completion returns non-empty output WITH finite logprobs (the model
    is generating, not just mounted); (3) the vLLM server process's own
    cmdline --model argument points at the FP8 snapshot on disk (identity of
    the loaded weights — the served NAME is a fixed alias and proves nothing).
    Raises unless every check passes (dry run relaxes only the cmdline check,
    which needs /proc and the kernel's pid file).
    """
    from pathlib import Path

    checks = []

    def rec(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), **detail})
        print(f"[ab-serving] {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
        return bool(ok)

    base_url = (_ab_os.environ.get("LOCAL_ANALYZER_BASE_URL")
                or _ab_os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    expected_id = _ab_os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
    rec("analyzer_base_url", bool(base_url), {"base_url": base_url, "expected_id": expected_id})

    ids = []
    try:
        with _ab_urlreq.urlopen(base_url + "/models", timeout=30) as r:
            ids = [m.get("id", "") for m in _ab_json.loads(r.read().decode()).get("data", [])]
        rec("endpoint_models", bool(ids) and (not expected_id or expected_id in ids),
            {"served_ids": ids})
    except Exception as e:
        rec("endpoint_models", False, {"error": f"{type(e).__name__}: {e}"})

    try:
        payload = {
            "model": expected_id or (ids[0] if ids else ""),
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "temperature": 0.0, "max_tokens": 8,
            "logprobs": True, "top_logprobs": 2,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = _ab_urlreq.Request(base_url + "/chat/completions",
                                 data=_ab_json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
        with _ab_urlreq.urlopen(req, timeout=120) as r:
            resp = _ab_json.loads(r.read().decode())
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = (msg.get("content") or "").strip()
        lp = ((choice.get("logprobs") or {}).get("content") or [])
        lp_ok = bool(lp) and all(
            isinstance(t.get("logprob"), (int, float)) and t["logprob"] <= 0.0 for t in lp[:5])
        rec("generation_logprobs",
            bool(text or msg.get("tool_calls")) and lp_ok,
            {"text": text[:80], "n_logprobs": len(lp),
             "first_logprobs": [round(float(t.get("logprob", 1)), 4) for t in lp[:3]]})
    except Exception as e:
        rec("generation_logprobs", False, {"error": f"{type(e).__name__}: {e}"})

    pid_path = Path(working_dir) / "vllm-openai-server.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
            model_arg = None
            for i, tok in enumerate(cmdline):
                if tok == "--model" and i + 1 < len(cmdline):
                    model_arg = cmdline[i + 1]
            ok = (model_arg is not None and Path(model_arg).exists()
                  and "qwen3-6-27b-fp8" in model_arg.lower())
            rec("server_model_arg", ok, {"pid": pid, "model_arg": model_arg})
        except Exception as e:
            rec("server_model_arg", False, {"error": f"{type(e).__name__}: {e}"})
    else:
        rec("server_model_arg", AB_DRY_RUN,
            {"pid_file": "absent", "relaxed_for_dry_run": AB_DRY_RUN})

    if not all(c["ok"] for c in checks) or len(checks) < 4:
        raise RuntimeError("serving assert FAILED — refusing to burn GPU hours on an "
                           "unproven serve: %s" % checks)
    print("=" * 20 + " AB SERVING-ASSERT PASS " + "=" * 20, flush=True)
    return {"checks": checks}


# --- scoring (mirrors scratchpad/ideas/score_rig.py exactly) -------------------


def ab_env_score(levels_completed, actions_per_level, n_levels, baselines):
    if not n_levels or len(baselines) != n_levels:
        return None
    total_w, total_score, max_w = 0, 0.0, 0
    for i in range(n_levels):
        w = i + 1
        total_w += w
        a = actions_per_level[i] if i < len(actions_per_level) else 0
        completed = i < (levels_completed or 0)
        s = min(115.0, (baselines[i] / a) ** 2 * 100.0) if (completed and a > 0) else 0.0
        if s > 0:
            max_w += w
        total_score += s * w
    return round(min(total_score / total_w, max_w / total_w * 100.0), 4)


def ab_load_baselines(env_dir):
    out = {}
    for meta in (_ab_glob.glob(f"{env_dir}/*/metadata.json")
                 + _ab_glob.glob(f"{env_dir}/*/*/metadata.json")):
        try:
            d = _ab_json.loads(open(meta).read())
        except Exception:
            continue
        if d.get("game_id") and d.get("baseline_actions"):
            out[d["game_id"]] = list(d["baseline_actions"])
    if not out:
        raise RuntimeError(f"no baselines found under {env_dir}")
    return out


# --- per-session diagnostics ---------------------------------------------------

_COMPACT_DIAG_KEYS = (
    "evictions_seen", "compactions_done", "compaction_failures", "compaction_tokens",
    "queue_plans", "queue_plans_rejected", "queue_steps_executed", "queue_aborts",
    "llm_calls_saved",
)
_COMPACT_KNOWLEDGE_KEYS = ("facts", "action_effects", "failed_hypotheses", "open_questions")


def _ab_compact_snapshot():
    mod = _ab_patch_module()
    diag = getattr(mod, "COMPACT_DIAGNOSTICS", None) if mod else None
    if not isinstance(diag, dict):
        return None
    return {k: int(diag.get(k, 0) or 0) for k in _COMPACT_DIAG_KEYS}


def _ab_session_diag(session):
    d = {}
    wd = getattr(session, "_watchdog_state", None)
    if isinstance(wd, dict):
        d["watchdog"] = {"stall_s": wd.get("stall_s"),
                         "resets_done": wd.get("resets_done"),
                         "killed": wd.get("killed")}
    tr = getattr(session, "_hud_tracker", None)
    if tr is not None:
        try:
            confirmed = tr._apply_stack_rule(tr._confirmed_lines())
            d["hud"] = {
                "segments": int(getattr(tr, "_segment_id", 0)),
                "mask_cells": len(tr.mask_cells()),
                "confirmed_lines": len(confirmed),
                "guard_dead_lines": len(getattr(tr, "_dead_lines", []) or []),
                "dropped_cells": (int(tr._dropped.sum())
                                  if getattr(tr, "_dropped", None) is not None else 0),
                "suppressed_hud_only": int(getattr(session, "_ab_hud_suppressed", 0)),
            }
        except Exception as e:
            d["hud"] = {"error": repr(e)}
    rp = getattr(session, "_win_replay_result", None)
    if rp is not None:
        d["replay"] = rp

    # -- round 2: graph diagnostics (C waves; state exists only when TAAF_GRAPH=1)
    if getattr(session, "_graph_state", None) is not None:
        mod = _ab_patch_module()
        try:
            if mod is not None and hasattr(mod, "graph_diagnostics"):
                d["graph"] = dict(mod.graph_diagnostics(session))
            else:  # same fields graph_diagnostics returns, read directly
                gs = session._graph_state
                d["graph"] = {**gs["diag"], **gs["graph"].diagnostics()}
        except Exception as e:
            d["graph"] = {"error": repr(e)}

    # -- round 2: analyzer-side token + compaction/queue state
    an = getattr(session, "analyzer", None)
    if an is not None:
        try:
            d["analyzer_tokens"] = {
                "generated": int(getattr(an, "generated_tokens", 0) or 0),
                "total": int(getattr(an, "total_tokens", 0) or 0),
            }
        except Exception as e:
            d["analyzer_tokens"] = {"error": repr(e)}
        # round 3: patch14 antifreeze — triggers attributed by the counter
        # wrapper, plus the guard's final streak state for context.
        try:
            af = an.__dict__.get("_antifreeze14") or {}
            d["antifreeze"] = {
                "triggers": int(an.__dict__.get("_ab_antifreeze_triggers", 0) or 0),
                "last_streak": af.get("streak"),
            }
        except Exception as e:
            d["antifreeze"] = {"error": repr(e)}
        cs = getattr(an, "_compact12_state", None)
        if isinstance(cs, dict):
            try:
                d["compact"] = {
                    "compactions_done": int(cs.get("compactions_done", 0) or 0),
                    "queue_total_last": int(cs.get("queue_total", 0) or 0),
                    "queue_pending": len(cs.get("queue") or []),
                    "knowledge_items": {
                        k: len((cs.get("knowledge") or {}).get(k) or [])
                        for k in _COMPACT_KNOWLEDGE_KEYS},
                }
            except Exception as e:
                d["compact"] = {"error": repr(e)}

    d["trace_len"] = len(getattr(session, "_replay_trace", None) or [])
    d["action_count"] = int(getattr(session, "action_count", 0) or 0)
    return d


def _ab_row_tokens(gr):
    """FIXED token accounting (round-1 defect): per-action generated tokens are
    recorded on run.history entries; final_generated_tokens is never populated
    by this bundle's solver (finish_game() is called with no arguments)."""
    total = 0
    for rec in (getattr(gr, "history", None) or []):
        try:
            total += max(0, int(getattr(rec, "generated_tokens", 0) or 0))
        except (TypeError, ValueError):
            continue
    try:
        total += max(0, int(getattr(gr, "final_generated_tokens", 0) or 0))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    return total


# --- main ----------------------------------------------------------------------


async def ab_main(bm, target, working_dir, notebook_start=None,
                  behav_report=None, behav_raw=None):
    from pathlib import Path

    working_dir = Path(working_dir)
    t0 = notebook_start if notebook_start is not None else _ab_time.time()
    stage = {"s": "init"}
    result = {
        "experiment": "ab_round3_playbook",
        "pre_registered_reading": AB_PREREGISTERED_READING,
        "arms": AB_ARM_ENV,
        "games": AB_GAMES,
        "target_games": list(_AB_TARGET_GAMES),
        "continuity_games": list(_AB_CONTINUITY_GAMES),
        "waves_planned": AB_WAVES,
        "per_game_budget_s": AB_BUDGET,
        "deadline_s_from_start": AB_DEADLINE_S,
        "dry_run": AB_DRY_RUN,
        "stage": "init",
        "error": None,
        "serving_assert": None,
        "patch_proof": None,
        "config": None,
        "waves": [],
    }

    def dump():
        result["stage"] = stage["s"]
        result["elapsed_s"] = round(_ab_time.time() - t0, 1)
        (working_dir / "ab_result.json").write_text(
            _ab_json.dumps(result, indent=1, default=str))

    try:
        # -- config snapshot: proof all arms share sampling/config -------------
        stage["s"] = "config_snapshot"
        try:
            from inference.agent import tool_agent as _ta
            result["config"] = {
                "temperature": float(getattr(_ta, "_LOCAL_ANALYZER_TEMPERATURE", float("nan"))),
                "top_p": float(getattr(_ta, "_LOCAL_ANALYZER_TOP_P", float("nan"))),
                "top_k": int(getattr(_ta, "_LOCAL_ANALYZER_TOP_K", -1)),
                "context_window": int(getattr(_ta, "_LOCAL_ANALYZER_CONTEXT_WINDOW", -1)),
                "concurrency": getattr(bm.solver, "concurrency", None),
                "model": str(getattr(getattr(bm.solver, "analyzer_model", None), "model_id", ""))
                         or _ab_os.environ.get("LOCAL_ANALYZER_MODEL_ID"),
            }
        except Exception as e:
            result["config"] = {"error": repr(e)}
        print(f"[ab] config {result['config']}", flush=True)

        # -- patch proof BEFORE registry re-wrap -------------------------------
        stage["s"] = "patch_proof"
        result["patch_proof"] = _ab_patch_proof()
        print(f"[ab] patch proof {result['patch_proof']}", flush=True)
        _ab_install_registry()
        _ab_install_prompt_recorder()
        result["antifreeze_counter_installed"] = _ab_install_antifreeze_counter()
        if not result["antifreeze_counter_installed"]:
            raise RuntimeError("antifreeze counter could not install — patch module "
                               "namespace lacks _antifreeze_note (patch14 missing?)")

        # -- serving assert (hard gate) ----------------------------------------
        stage["s"] = "serving_assert"
        result["serving_assert"] = ab_serving_assert(working_dir)
        dump()

        # -- resolve games ------------------------------------------------------
        stage["s"] = "resolve_games"
        import arc_agi
        import taaf.benchmark
        import taaf.competition_arcade as _ca
        import taaf.game_api
        env_dir = _ab_os.environ.get(
            "ARC_ENVIRONMENTS_DIR", "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files")
        if not _ab_os.path.isdir(env_dir):
            cands = _ab_glob.glob("/kaggle/input/**/environment_files", recursive=True)
            if not cands:
                raise RuntimeError("no environment_files directory found")
            env_dir = cands[0]
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                                environments_dir=env_dir)
        by_stem = {e.game_id.split("-")[0]: e.game_id for e in arcade.available_environments}
        missing = [g for g in AB_GAMES if g not in by_stem]
        if missing:
            raise RuntimeError(f"panel games not in {env_dir}: {missing} "
                               f"(available: {sorted(by_stem)})")
        chosen = [by_stem[g] for g in AB_GAMES]
        result["env_dir"] = env_dir
        result["chosen_ids"] = chosen
        baselines = ab_load_baselines(env_dir)
        print(f"[ab] env_dir={env_dir} chosen={chosen}", flush=True)

        # -- waves --------------------------------------------------------------
        for wi, arm in enumerate(AB_WAVES):
            elapsed = _ab_time.time() - t0
            if elapsed + AB_BUDGET + AB_WAVE_OVERHEAD_S > AB_DEADLINE_S:
                print(f"[ab] SKIPPING wave {wi} ({arm}): elapsed {elapsed:.0f}s + budget "
                      f"would pass deadline {AB_DEADLINE_S:.0f}s", flush=True)
                result["waves"].append({"wave": wi, "arm": arm, "skipped": "deadline"})
                dump()
                continue

            stage["s"] = f"wave{wi}_{arm}_env"
            _ab_os.environ.update(AB_ARM_ENV[arm])
            toggles = _ab_toggles()
            expected = _ab_expected_toggles(arm)
            if toggles != expected:
                raise RuntimeError(
                    f"arm {arm} toggle mismatch: realized {toggles} != expected {expected}")
            _AB_WAVE["i"] = wi
            probe = _ab_playbook_probe(arm)   # raises on mismatch BEFORE the wave runs
            print(f"[ab] === wave {wi} arm {arm} toggles {toggles} "
                  f"playbook_probe={probe} elapsed {elapsed:.0f}s ===", flush=True)

            stage["s"] = f"wave{wi}_{arm}_server"
            compact_before = _ab_compact_snapshot()
            antifreeze_before = _ab_antifreeze_snapshot()
            n_sp_before = len(_AB_SYSPROMPTS)   # after the probe: in-run samples only
            srv = _ca.CompetitionArcadeServer(
                game_ids=tuple(chosen), total_runs=len(chosen),
                environments_dir=env_dir).start()
            try:
                clones = list(srv.exposed_game_ids)
                cmap = {}
                for ei in srv._arcade.available_environments:
                    src = None
                    for tag in (getattr(ei, "private_tags", None) or []):
                        if str(tag).startswith("taaf_source_game:"):
                            src = str(tag).split(":", 1)[1]
                    cmap[ei.game_id] = src
                if sorted(x for x in cmap.values() if x) != sorted(chosen):
                    raise RuntimeError(f"clone map incomplete/mismatched: {cmap}")

                stage["s"] = f"wave{wi}_{arm}_run"
                games = [taaf.game_api.GameAPI(env_name=c, arcade_spec=srv.arcade_spec)
                         for c in clones]
                wave_bm = taaf.benchmark.Benchmark(
                    label=f"ab_{arm}_{wi}", games=games, solver=bm.solver, n_passes=1)
                for attr in ("max_runtime_s_per_game", "max_runtime_s"):
                    if hasattr(wave_bm.solver, attr):
                        setattr(wave_bm.solver, attr, AB_BUDGET)
                n_before = len(AB_SESSIONS)
                _AB_WAVE["i"] = wi
                tw = _ab_time.time()
                await wave_bm.run(soft_end_time=None, runtime_environment=target,
                                  minimal_diagnostics=False)
                wall = _ab_time.time() - tw

                stage["s"] = f"wave{wi}_{arm}_collect"
                diag_by_run = {}
                for swi, sess in AB_SESSIONS[n_before:]:
                    if swi != wi:
                        continue
                    run = getattr(sess.game, "game_run", None)
                    if run is not None:
                        diag_by_run[id(run)] = _ab_session_diag(sess)
                rows = []
                for gr in (getattr(wave_bm, "game_runs", None) or []):
                    apl = list(getattr(gr, "actions_per_level", []) or [])
                    hist = len(getattr(gr, "history", []) or [])
                    src = cmap.get(getattr(gr, "game_id", None))
                    lv = int(getattr(gr, "levels_completed", 0) or 0)
                    nl = int(getattr(gr, "number_of_levels", 0) or 0)
                    row = {
                        "clone_id": getattr(gr, "game_id", None),
                        "source_game": src,
                        "levels_completed": lv,
                        "levels_total": nl,
                        "actions_per_level": apl,
                        "actions_total": hist,
                        "apl_sum_matches_history": (sum(apl) == hist),
                        "state": str(getattr(gr, "state", None)),
                        "wallclock_s": getattr(gr, "final_wallclock_seconds", None),
                        "gen_tokens": _ab_row_tokens(gr),
                        "solver_note": str(getattr(gr, "solver_note", "") or ""),
                        "score": (ab_env_score(lv, apl, nl, baselines[src])
                                  if src in baselines else None),
                    }
                    row.update(diag_by_run.get(id(gr), {}))
                    rows.append(row)
                    print(f"[ab] w{wi} {arm} {src}: levels={lv}/{nl} actions={hist} "
                          f"tok={row['gen_tokens']} score={row['score']} "
                          f"wd={row.get('watchdog')} hud={row.get('hud')} "
                          f"antifreeze={row.get('antifreeze')} "
                          f"replay={(row.get('replay') or {}).get('status')}",
                          flush=True)

                compact_after = _ab_compact_snapshot()
                antifreeze_after = _ab_antifreeze_snapshot()

                # -- round 3: in-run playbook presence proof (real sampled
                # system prompts built while games played this wave) ----------
                marker = _ab_playbook_marker()
                pb_expected = AB_ARM_ENV[arm].get("TAAF_PLAYBOOK") == "1"
                samples = [p for swi, p in _AB_SYSPROMPTS[n_sp_before:] if swi == wi]
                mismatches = [i for i, p in enumerate(samples)
                              if (marker in p) != pb_expected]
                playbook_proof = {
                    "marker": marker,
                    "expected_present": pb_expected,
                    "n_prompts_sampled": len(samples),
                    "n_mismatches": len(mismatches),
                    "probe_before_wave": probe,
                    "sample_sha256": (_ab_hashlib.sha256(samples[-1].encode()).hexdigest()
                                      if samples else None),
                    "sample_chars": len(samples[-1]) if samples else None,
                }
                if not samples or mismatches:
                    raise RuntimeError(
                        f"arm {arm} in-run playbook proof failed: {playbook_proof}")
                print(f"[ab] w{wi} {arm} playbook proof: {len(samples)} real system "
                      f"prompts, all {'WITH' if pb_expected else 'WITHOUT'} playbook",
                      flush=True)

                wave_rec = {
                    "wave": wi, "arm": arm, "env": dict(AB_ARM_ENV[arm]),
                    "toggles_verified": toggles,
                    "patch_proof_wave": _ab_wave_patch_proof(),
                    "playbook_proof": playbook_proof,
                    "wall_s": round(wall, 1),
                    "started_at_elapsed_s": round(elapsed, 1), "rows": rows,
                }
                if antifreeze_before is not None and antifreeze_after is not None:
                    wave_rec["antifreeze_diag_delta"] = {
                        "triggers": antifreeze_after["triggers"] - antifreeze_before["triggers"]}
                    wave_rec["antifreeze_diag_cumulative"] = antifreeze_after
                if compact_before is not None and compact_after is not None:
                    wave_rec["compact_diag_delta"] = {
                        k: compact_after[k] - compact_before[k] for k in _COMPACT_DIAG_KEYS}
                    wave_rec["compact_diag_cumulative"] = compact_after
                if callable(behav_report):
                    try:
                        wave_rec["behav_cumulative"] = behav_report()
                    except Exception as e:
                        wave_rec["behav_cumulative"] = {"error": repr(e)}
                if callable(behav_raw):
                    try:
                        wave_rec["behav_raw_cumulative"] = behav_raw()
                    except Exception as e:
                        wave_rec["behav_raw_cumulative"] = {"error": repr(e)}
                result["waves"].append(wave_rec)
                dump()
            finally:
                try:
                    srv.stop()
                except Exception:
                    pass
        stage["s"] = "done"
    except Exception:
        result["error"] = _ab_traceback.format_exc()
        print(f"[ab] FAILED at stage={stage['s']}:\n{result['error']}", flush=True)

    # -- summary table (always printed, even on partial failure) ---------------
    try:
        _ab_print_summary(result)
    except Exception as e:
        print(f"[ab] summary failed: {e!r}", flush=True)
    dump()
    print(f"[ab] DONE stage={stage['s']} elapsed={_ab_time.time() - t0:.0f}s", flush=True)
    return result


def _ab_print_summary(result):
    waves = [w for w in result.get("waves", []) if not w.get("skipped")]
    print("\n[ab] ================ SUMMARY ================", flush=True)
    print(f"[ab] wave order realized: "
          f"{[(w['wave'], w['arm']) for w in result.get('waves', [])]}", flush=True)
    header = (f"{'game':6} {'wave':4} {'arm':3} {'lvl':>6} {'acts':>6} {'tok':>8} {'score':>8} "
              f"{'wd_rst':>6} {'hud_supp':>8} {'mask':>5} {'conf':>5} {'antifz':>6} "
              f"{'replay':>10}")
    print("[ab] " + header, flush=True)
    per_arm = {}
    for w in waves:
        for r in w.get("rows", []):
            stem = (r.get("source_game") or "?").split("-")[0]
            wd = r.get("watchdog") or {}
            hud = r.get("hud") or {}
            af = r.get("antifreeze") or {}
            rp = r.get("replay") or {}
            score = r.get("score")
            score_s = f"{score:8.2f}" if isinstance(score, (int, float)) else f"{'-':>8}"
            print(f"[ab] {stem:6} {w['wave']:<4} {w['arm']:3} "
                  f"{r.get('levels_completed', 0):>3}/{r.get('levels_total', 0):<2} "
                  f"{r.get('actions_total', 0):>6} {r.get('gen_tokens', 0):>8} {score_s} "
                  f"{wd.get('resets_done', '-')!s:>6} "
                  f"{hud.get('suppressed_hud_only', '-')!s:>8} "
                  f"{hud.get('mask_cells', '-')!s:>5} "
                  f"{hud.get('confirmed_lines', '-')!s:>5} "
                  f"{af.get('triggers', '-')!s:>6} {rp.get('status', '-')!s:>10}",
                  flush=True)
            slot = per_arm.setdefault(stem, {}).setdefault(w["arm"], {"lvl": [], "score": []})
            slot["lvl"].append(r.get("levels_completed", 0) or 0)
            if isinstance(score, (int, float)):
                slot["score"].append(score)
    for w in waves:
        af_delta = w.get("antifreeze_diag_delta")
        if af_delta and any(af_delta.values()):
            print(f"[ab] wave {w['wave']} ({w['arm']}) antifreeze delta: {af_delta}",
                  flush=True)
        delta = w.get("compact_diag_delta")
        if delta and any(delta.values()):
            print(f"[ab] wave {w['wave']} ({w['arm']}) UNEXPECTED compact delta "
                  f"(COMPACT pinned 0 in both arms!): {delta}", flush=True)

    # -- PRIMARY pre-registered readout: L1-unlock events on the target games --
    print("[ab] ---- PRIMARY: target-game L1 unlocks (B-only unlock = signal) ----",
          flush=True)
    for stem in _AB_TARGET_GAMES:
        arms = per_arm.get(stem, {})
        la = arms.get("A", {}).get("lvl", [])
        lb = arms.get("B", {}).get("lvl", [])
        a_unlocked = any(v > 0 for v in la)
        b_unlocked = any(v > 0 for v in lb)
        flag = ("B-ONLY UNLOCK" if b_unlocked and not a_unlocked else
                "A-only unlock" if a_unlocked and not b_unlocked else
                "both" if a_unlocked else "neither")
        print(f"[ab] {stem:6} A_levels={la} B_levels={lb} -> {flag}", flush=True)

    if any(w["arm"] == "B" for w in waves):
        print("[ab] ---- paired per-game means (B - A) [SECONDARY; scores are noise] ----",
              flush=True)
        dl_sum, ds_sum, n = 0.0, 0.0, 0
        for stem in sorted(per_arm):
            arms = per_arm[stem]
            if "A" not in arms or "B" not in arms:
                continue
            la = sum(arms["A"]["lvl"]) / max(len(arms["A"]["lvl"]), 1)
            lb = sum(arms["B"]["lvl"]) / max(len(arms["B"]["lvl"]), 1)
            sa = sum(arms["A"]["score"]) / max(len(arms["A"]["score"]), 1)
            sb = sum(arms["B"]["score"]) / max(len(arms["B"]["score"]), 1)
            dl_sum += lb - la
            ds_sum += sb - sa
            n += 1
            print(f"[ab] {stem:6} levels A={la:.1f} B={lb:.1f} d={lb - la:+.1f}   "
                  f"score A={sa:.2f} B={sb:.2f} d={sb - sa:+.2f}", flush=True)
        if n:
            print(f"[ab] TOTAL paired delta (B-A) over {n} games: "
                  f"levels {dl_sum:+.1f}, score {ds_sum:+.2f}  (A/A noise floor: RMS 0.707 "
                  f"levels/game-run — read unlock events, not small score deltas)", flush=True)
