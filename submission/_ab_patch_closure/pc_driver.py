# ============================================================================
# Patch-closure single-arm driver.
#
# ONE competition-sim wave at the submission-shaped eval geometry (28 clones of
# the 25 official games, 7920s per game, concurrency 28 — submission/_rig's
# measured regime), for ONE arm whose environment the hook cell pinned before
# apply_all(). Writes /kaggle/working/patch_closure_result.json incrementally
# (rig law: an ERRORed kernel has no retrievable log, but working-dir files
# survive) with the frozen schema patch_closure_config.classify_result reads.
#
# This file is BOTH inlined into each arm's run cell by build_patch_closure.py
# AND imported by dry_run.py for the GPU-free local test. Edit here, rebuild.
#
# The arm delta under test (candidate vs byte-reconstructed v7 base):
#   TAAF_WATCHDOG_STALL_S 900 -> 600   (heartbeat beats the 15-min stale close)
#   TAAF_ANIMATION        0   -> 1    (patch15 last_animation into the sandbox)
#   TAAF_GRAPH            0   -> 1    (patch11 veto + level-age grinder/narration)
# Everything else — including the f0ec605 refill-aware HUD unmask guard — is
# SHARED BASE, byte-identical in both arms.
# ============================================================================
import glob as _pc_glob
import json as _pc_json
import os as _pc_os
import sys as _pc_sys
import time as _pc_time
import traceback as _pc_traceback
import urllib.request as _pc_urlreq

PC_SESSIONS = []                 # (run_token, session) — filled by the registry
_PC_RUN = {"token": None}
PC_ANIMATION_UPTAKE = {"payload_deliveries": 0, "frames_delivered": 0}
PC_LLM_TURNS = {"analyze_calls": 0}   # deliberations — the adoption denominator


def _pc_patch_module():
    """The namespace holding duck_patches' module-level objects: __main__ when
    the patch layer is inlined into the notebook, the synthetic 'duck_patches'
    module in the dry run (same resolution _ab_wmr uses)."""
    mod = _pc_sys.modules.get("duck_patches")
    if mod is not None and hasattr(mod, "ANTIFREEZE_DIAGNOSTICS"):
        return mod
    main = _pc_sys.modules.get("__main__")
    if main is not None and hasattr(main, "ANTIFREEZE_DIAGNOSTICS"):
        return main
    return None


# --- toggle semantics: EXACTLY duck_patches.py's readers ------------------------


def _pc_toggles():
    def default_on(name):
        return _pc_os.environ.get(name, "1").strip() not in {"0", "false", "False"}

    def opt_in(name):  # grid burner only
        return _pc_os.environ.get(name, "0").strip() in {"1", "true", "True"}

    return {
        "TAAF_WATCHDOG": default_on("TAAF_WATCHDOG"),
        "TAAF_HUD_MASK": default_on("TAAF_HUD_MASK"),
        "TAAF_WIN_REPLAY": default_on("TAAF_WIN_REPLAY"),
        "TAAF_ANTIFREEZE": default_on("TAAF_ANTIFREEZE"),
        "TAAF_ANIMATION": default_on("TAAF_ANIMATION"),
        "TAAF_GRAPH": default_on("TAAF_GRAPH"),
        "TAAF_COMPACT": default_on("TAAF_COMPACT"),
        "TAAF_PLAYBOOK": default_on("TAAF_PLAYBOOK"),
        "TAAF_GRID_BURNER": opt_in("TAAF_GRID_BURNER"),
    }


def _pc_verify_arm_env(arm, arm_env):
    """Realized env + realized toggle semantics vs the frozen arm mapping.
    Raises on any mismatch — a mis-pinned arm must abort before play."""
    env_realized = {k: _pc_os.environ.get(k) for k in arm_env}
    env_drift = {k: (arm_env[k], env_realized[k])
                 for k in arm_env if env_realized[k] != arm_env[k]}
    toggles = _pc_toggles()
    expected = {k: arm_env[k] == "1" for k in toggles if k in arm_env}
    toggle_drift = {k: (expected[k], toggles[k])
                    for k in expected if toggles[k] != expected[k]}
    ok = not env_drift and not toggle_drift
    if not ok:
        raise RuntimeError(
            f"[pc] arm {arm} environment drifted: env={env_drift} toggles={toggle_drift}")
    return {"env_realized": env_realized, "toggles_realized": toggles,
            "toggles_expected": expected, "toggles_ok": ok}


# --- patch-layer identity -------------------------------------------------------


def _pc_patch_proof():
    """Positive markers on the live classes for every patched surface this
    experiment relies on (the _ab_wmr proof + the two animation seams)."""
    from inference.agent import python_tool_sandbox as _ptx
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
                                             or getattr(cls._execute_action, "_animation_sandbox_patched", False)
                                             or getattr(cls._execute_action, "_behav", False)),
        "play_patched": bool(getattr(cls.play, "_win_replay_patched", False)
                             or getattr(cls.play, "_watchdog_play_patched", False)
                             or getattr(cls.play, "_animation_sandbox_play_patched", False)),
        "animation_sandbox_patched": bool(
            getattr(_ptx.run_sandboxed_python, "_animation_sandbox_patched", False)),
        "animation_prompt_patched": bool(
            getattr(_ta._build_system_prompt, "_animation_prompt_patched", False)),
        "antifreeze_user_prompt_patched": bool(
            getattr(agent_cls._build_user_prompt, "_antifreeze_patched", False)),
        "playbook_system_prompt_patched": bool(
            getattr(_ta._build_system_prompt, "_playbook_patched", False)),
        "execute_chain_repr": repr(cls._execute_action),
        "play_chain_repr": repr(cls.play),
    }


def _pc_animation_probe(arm, arm_env):
    """Toggle-proof (the _ab_wmr playbook-probe law): build ONE real system
    prompt through the patched builder and hard-assert the patch15 announcing
    line matches the arm BEFORE any GPU minute is spent."""
    import hashlib as _hl

    from inference.agent import tool_agent as _ta

    prompt = _ta._build_system_prompt(tool_output_tokens=2000)
    present = "last_animation" in prompt
    expected = arm_env.get("TAAF_ANIMATION") == "1"
    if present != expected:
        raise RuntimeError(
            f"[pc] arm {arm} animation prompt probe mismatch: present={present} "
            f"expected={expected} (prompt tail: ...{prompt[-300:]!r})")
    return {"marker": "last_animation", "present": present, "expected": expected,
            "prompt_sha256": _hl.sha256(prompt.encode()).hexdigest(),
            "prompt_chars": len(prompt)}


# --- registry + mechanism counters ---------------------------------------------


def _pc_install_registry():
    """Outermost play wrapper: record sessions per run token."""
    from inference.framework import solver as _sv

    cls = _sv._HarnessGameSession
    if getattr(cls.play, "_pc_registered", False):
        return
    orig_play = cls.play

    def play(self):
        PC_SESSIONS.append((_PC_RUN["token"], self))
        return orig_play(self)

    for k, v in vars(orig_play).items():
        if k.endswith("_patched"):
            setattr(play, k, v)
    play._pc_registered = True
    cls.play = play


def _pc_install_turn_counter():
    """Count LLM DELIBERATIONS (ToolAgent.analyze calls) — the denominator of
    the adoption metric. The behavioural probe's `turns` cannot serve here:
    under TAAF_STRUCT every plan step executes as its own single-action batch
    (patch 18's one-action-per-handler-call pattern), so batch_index==1 fires
    per STEP and probe actions_per_turn reads 1.0 by construction. cfeb92a's
    adoption dry run used mock-brain POSTs; analyze calls are the kernel-side
    equivalent (the mock makes exactly one POST per analyze)."""
    from inference.agent import tool_agent as _ta

    agent_cls = _ta.ToolAgent
    if getattr(agent_cls.analyze, "_pc_turns_counted", False):
        return True
    orig = agent_cls.analyze

    def analyze(self, *args, **kwargs):
        PC_LLM_TURNS["analyze_calls"] += 1
        return orig(self, *args, **kwargs)

    for k, v in vars(orig).items():
        if k.endswith("_patched"):
            setattr(analyze, k, v)
    analyze._pc_turns_counted = True
    agent_cls.analyze = analyze
    return True


def _pc_install_animation_counter():
    """Wrap the patch module's _animation_current_payload (which patch15's
    sandbox route resolves at call time) so every non-empty animation payload
    DELIVERY into the sandbox is counted. Observational only."""
    mod = _pc_patch_module()
    if mod is None or not callable(getattr(mod, "_animation_current_payload", None)):
        return False
    orig = mod._animation_current_payload
    if getattr(orig, "_pc_counted", False):
        return True

    def _animation_current_payload():
        frames = orig()
        if frames:
            PC_ANIMATION_UPTAKE["payload_deliveries"] += 1
            PC_ANIMATION_UPTAKE["frames_delivered"] += len(frames)
        return frames

    _animation_current_payload._pc_counted = True
    mod._animation_current_payload = _animation_current_payload
    return True


def _pc_antifreeze_snapshot():
    mod = _pc_patch_module()
    diag = getattr(mod, "ANTIFREEZE_DIAGNOSTICS", None) if mod else None
    return {"triggers": int(diag.get("triggers", 0) or 0)} if isinstance(diag, dict) else None


# Patches 16-22 (probe-infrastructure package + structural channel) module
# counters, snapshotted as deltas like antifreeze/compact. Absent attrs (older
# patch bytes) are simply omitted. Values are ints or one level of int->int
# histogram (STRUCT_DIAGNOSTICS["plan_lengths"]).
_PC_PACKAGE_DIAG_ATTRS = {
    "diff_lines": "DIFF_LINES_DIAGNOSTICS",
    "wiggle": "WIGGLE_DIAGNOSTICS",
    "run_probe": "RUN_PROBE_DIAGNOSTICS",
    "dispatch": "DISPATCH_DIAGNOSTICS",
    "struct": "STRUCT_DIAGNOSTICS",
}


def _pc_package_snapshot():
    mod = _pc_patch_module()
    if mod is None:
        return None
    out = {}
    for key, attr in _PC_PACKAGE_DIAG_ATTRS.items():
        diag = getattr(mod, attr, None)
        if not isinstance(diag, dict):
            continue
        snap = {}
        for k, v in diag.items():
            if isinstance(v, (int, float)):
                snap[k] = int(v or 0)
            elif isinstance(v, dict):  # histogram (e.g. plan_lengths)
                snap[k] = {str(hk): int(hv or 0) for hk, hv in v.items()
                           if isinstance(hv, (int, float))}
        out[key] = snap
    return out or None


def _pc_package_delta(before, after):
    if before is None or after is None:
        return None
    out = {}
    for key in after:
        b, a = before.get(key, {}), after[key]
        d = {}
        for k, v in a.items():
            if isinstance(v, dict):
                bh = b.get(k, {}) if isinstance(b.get(k), dict) else {}
                d[k] = {hk: v.get(hk, 0) - bh.get(hk, 0)
                        for hk in set(v) | set(bh)
                        if v.get(hk, 0) - bh.get(hk, 0)}
            else:
                d[k] = v - (b.get(k, 0) if isinstance(b.get(k), (int, float)) else 0)
        out[key] = d
    return out


def _pc_compact_snapshot():
    mod = _pc_patch_module()
    diag = getattr(mod, "COMPACT_DIAGNOSTICS", None) if mod else None
    if not isinstance(diag, dict):
        return None
    return {k: int(v or 0) for k, v in diag.items() if isinstance(v, (int, float))}


# --- serving probe --------------------------------------------------------------


def _pc_serving_probe():
    """Both arms serve the SAME base weights, so a light identity probe
    suffices: endpoint configured + /models answers with the expected alias.
    Raises on failure — two hours of a dead endpoint measure nothing."""
    base_url = (_pc_os.environ.get("LOCAL_ANALYZER_BASE_URL")
                or _pc_os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    expected_id = _pc_os.environ.get("LOCAL_ANALYZER_MODEL_ID", "")
    if not base_url:
        raise RuntimeError("[pc] no analyzer base URL configured")
    with _pc_urlreq.urlopen(base_url + "/models", timeout=30) as r:
        ids = [m.get("id", "") for m in _pc_json.loads(r.read().decode()).get("data", [])]
    if not ids or (expected_id and expected_id not in ids):
        raise RuntimeError(f"[pc] /models does not serve {expected_id!r}: {ids}")
    return {"base_url": base_url, "served_ids": ids, "expected_id": expected_id, "ok": True}


# --- per-session diagnostics (the _ab_wmr collector, compact section dropped) ---


def _pc_session_diag(session):
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
            }
        except Exception as e:  # noqa: BLE001
            d["hud"] = {"error": repr(e)}
    rp = getattr(session, "_win_replay_result", None)
    if rp is not None:
        d["replay"] = rp
    ws = getattr(session, "_wiggle_state", None)
    if ws is not None:  # patch17 per-game controllability record
        try:
            d["wiggle"] = {
                "mode": str(getattr(ws, "mode", "UNCLEAR")),
                "battery_done": bool(getattr(ws, "battery_done", False)),
                "battery_presses": int(getattr(ws, "battery_presses", 0) or 0),
                "reprobe_presses": int(getattr(ws, "reprobe_presses", 0) or 0),
            }
        except Exception as e:  # noqa: BLE001
            d["wiggle"] = {"error": repr(e)}
    if getattr(session, "_graph_state", None) is not None:
        mod = _pc_patch_module()
        try:
            if mod is not None and hasattr(mod, "graph_diagnostics"):
                d["graph"] = dict(mod.graph_diagnostics(session))
            else:
                gs = session._graph_state
                d["graph"] = {**gs["diag"], **gs["graph"].diagnostics()}
        except Exception as e:  # noqa: BLE001
            d["graph"] = {"error": repr(e)}
    an = getattr(session, "analyzer", None)
    if an is not None:
        try:
            d["analyzer_tokens"] = {
                "generated": int(getattr(an, "generated_tokens", 0) or 0),
                "total": int(getattr(an, "total_tokens", 0) or 0),
            }
        except Exception as e:  # noqa: BLE001
            d["analyzer_tokens"] = {"error": repr(e)}
        try:
            af = an.__dict__.get("_antifreeze14") or {}
            d["antifreeze"] = {"last_streak": af.get("streak")}
        except Exception as e:  # noqa: BLE001
            d["antifreeze"] = {"error": repr(e)}
    d["trace_len"] = len(getattr(session, "_replay_trace", None) or [])
    d["action_count"] = int(getattr(session, "action_count", 0) or 0)
    return d


_PC_GRAPH_DIAG_KEYS = (
    "vetoes_issued", "vetoes_capped", "grinder_engagements", "grinder_age_triggers",
    "grinder_actions", "levels_unlocked_by_grinder", "narrations_injected",
)


def collect_patch_diagnostics(sessions, animation_delta, antifreeze_delta, compact_delta,
                              package_delta=None):
    """The arm's mechanism-level diagnostics block, frozen shape (the closure
    classifier reads animation.payload_deliveries, graph.grinder_*,
    watchdog.stall_kills; the package screen additionally reads diff_lines /
    wiggle / run_probe / dispatch and the per-session mode histogram)."""
    graph = {k: 0 for k in _PC_GRAPH_DIAG_KEYS}
    graph["sessions_with_graph_state"] = 0
    stall_kills = wall_kills = resets = 0
    stall_s_observed = set()
    wiggle_modes = {}
    for sess in sessions:
        gs = getattr(sess, "_graph_state", None)
        if gs is not None:
            graph["sessions_with_graph_state"] += 1
            for k in _PC_GRAPH_DIAG_KEYS:
                graph[k] += int(gs["diag"].get(k, 0) or 0)
        wd = getattr(sess, "_watchdog_state", None)
        if isinstance(wd, dict):
            stall_kills += int(wd.get("killed") == "stall")
            wall_kills += int(wd.get("killed") == "wall_cap")
            resets += int(wd.get("resets_done", 0) or 0)
            if wd.get("stall_s") is not None:
                stall_s_observed.add(float(wd["stall_s"]))
        ws = getattr(sess, "_wiggle_state", None)
        if ws is not None:
            mode = str(getattr(ws, "mode", "UNCLEAR"))
            wiggle_modes[mode] = wiggle_modes.get(mode, 0) + 1
    out = {
        "animation": dict(animation_delta),
        "graph": graph,
        "watchdog": {"stall_kills": stall_kills, "wall_cap_kills": wall_kills,
                     "recovery_resets": resets,
                     "stall_s_observed": sorted(stall_s_observed)},
        "antifreeze": antifreeze_delta,
        "compact": compact_delta,
    }
    if package_delta is not None:
        out.update(package_delta)  # diff_lines / wiggle / run_probe / dispatch
    if wiggle_modes:
        out["wiggle_modes_assigned"] = wiggle_modes
    return out


def _pc_row_tokens(gr):
    """Per-action generated tokens live on run.history entries (the _ab_wmr
    round-1 finding: final_generated_tokens is never populated here)."""
    total = 0
    for rec in (getattr(gr, "history", None) or []):
        try:
            total += max(0, int(getattr(rec, "generated_tokens", 0) or 0))
        except (TypeError, ValueError):
            continue
    try:
        total += max(0, int(getattr(gr, "final_generated_tokens", 0) or 0))
    except (TypeError, ValueError):
        pass
    return total


# --- offline scoring (informative only; the classifier reads levels) ------------


def pc_env_score(levels_completed, actions_per_level, n_levels, baselines):
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


def pc_load_baselines(env_dir):
    out = {}
    for meta in (_pc_glob.glob(f"{env_dir}/*/metadata.json")
                 + _pc_glob.glob(f"{env_dir}/*/*/metadata.json")):
        try:
            d = _pc_json.loads(open(meta).read())
        except Exception:  # noqa: BLE001
            continue
        if d.get("game_id") and d.get("baseline_actions"):
            out[d["game_id"]] = list(d["baseline_actions"])
    return out


# --- main -----------------------------------------------------------------------


async def pc_main(bm, target, working_dir, arm, arm_env, hypothesis, geometry,
                  source_base_sha256, patch_sha256,
                  behav_report=None, behav_assert=None,
                  dry_run=False, dry_exercise=None, reading=None):
    """Run ONE arm at the given geometry and write patch_closure_result.json.

    `dry_exercise(sessions)` is a dry-run-only seam invoked after the wave and
    before diagnostics collection (deterministic mechanism exercises); the GPU
    kernels never pass it.
    """
    from pathlib import Path

    working_dir = Path(working_dir)
    t0 = _pc_time.time()
    stage = {"s": "init"}
    result = {
        "schema_version": 1,
        "hypothesis": hypothesis,
        "arm": arm,
        "arm_env": dict(arm_env),
        "source_base_sha256": source_base_sha256,
        "patch_sha256": patch_sha256,
        "geometry": dict(geometry),
        "identity": {},
        "rows": [],
        "rows_by_source": {},
        "behavior": None,
        "patch_diagnostics": None,
        "dry_run": bool(dry_run),
        "stage": "init",
        "error": None,
    }
    if reading is not None:
        # Pre-registered reading contract, written verbatim into the artifact
        # header so the analysis cannot quietly move the goalposts afterwards
        # (the _ab_wmr precedent).
        result["pre_registered_reading"] = reading

    def dump():
        result["stage"] = stage["s"]
        result["elapsed_s"] = round(_pc_time.time() - t0, 1)
        (working_dir / "patch_closure_result.json").write_text(
            _pc_json.dumps(result, indent=1, default=str))

    try:
        # -- arm identity: env pins, patch markers, animation prompt probe ------
        stage["s"] = "verify_arm_env"
        identity = {"arm": arm}
        identity.update(_pc_verify_arm_env(arm, arm_env))
        identity["patch_proof"] = _pc_patch_proof()
        bad_markers = [k for k, v in identity["patch_proof"].items()
                       if isinstance(v, bool) and not v]
        if bad_markers:
            raise RuntimeError(f"[pc] patch layer incomplete on live classes: {bad_markers}")
        identity["animation_prompt_probe"] = _pc_animation_probe(arm, arm_env)
        result["identity"] = identity
        print(f"[pc] arm={arm} env verified; patch markers all present; "
              f"animation prompt {'PRESENT' if identity['animation_prompt_probe']['present'] else 'absent'}",
              flush=True)

        stage["s"] = "config_snapshot"
        try:
            from inference.agent import tool_agent as _ta
            identity["config"] = {
                "temperature": float(getattr(_ta, "_LOCAL_ANALYZER_TEMPERATURE", float("nan"))),
                "top_p": float(getattr(_ta, "_LOCAL_ANALYZER_TOP_P", float("nan"))),
                "top_k": int(getattr(_ta, "_LOCAL_ANALYZER_TOP_K", -1)),
                "context_window": int(getattr(_ta, "_LOCAL_ANALYZER_CONTEXT_WINDOW", -1)),
                "model": str(getattr(getattr(bm.solver, "analyzer_model", None), "model_id", ""))
                         or _pc_os.environ.get("LOCAL_ANALYZER_MODEL_ID"),
            }
        except Exception as e:  # noqa: BLE001
            identity["config"] = {"error": repr(e)}
        print(f"[pc] config {identity['config']}", flush=True)

        stage["s"] = "serving_probe"
        identity["serving"] = _pc_serving_probe()

        stage["s"] = "install_instruments"
        _pc_install_registry()
        if not _pc_install_turn_counter():
            raise RuntimeError("[pc] LLM-turn counter could not install")
        if not _pc_install_animation_counter():
            raise RuntimeError("[pc] animation uptake counter could not install "
                               "(_animation_current_payload missing from the patch module)")
        turns_before = dict(PC_LLM_TURNS)
        anim_before = dict(PC_ANIMATION_UPTAKE)
        antifreeze_before = _pc_antifreeze_snapshot()
        compact_before = _pc_compact_snapshot()
        package_before = _pc_package_snapshot()
        dump()

        # -- official ids + competition server (the _rig mechanism) -------------
        stage["s"] = "official_ids"
        import arc_agi
        import taaf.benchmark
        import taaf.competition_arcade as _ca
        import taaf.game_api
        env_dir = _pc_os.environ.get(
            "ARC_ENVIRONMENTS_DIR", "/kaggle/input/arc-prize-2026-arc-agi-3/environment_files")
        if not _pc_os.path.isdir(env_dir):
            cands = _pc_glob.glob("/kaggle/input/**/environment_files", recursive=True)
            if not cands:
                raise RuntimeError("no environment_files directory found")
            env_dir = cands[0]
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE,
                                environments_dir=env_dir)
        official = sorted(e.game_id for e in arcade.available_environments)
        if len(official) != 25:
            raise RuntimeError(f"expected 25 official environments, got {len(official)}")
        # Optional FOCUS SUBSET. The clone map is round-robin over `official`
        # (see `recon` below), so a full 25-game wave gives n=1 per game — which
        # is why every per-game verdict in this rig's history rests on a single
        # clone. Naming fewer games here spreads the same 28 clones over them,
        # turning n=1 into n=28 on one game. The 25-environment assert above
        # still runs, so a broken env_dir is caught before any filtering.
        focus = tuple(geometry.get("games") or ())
        if focus:
            missing = [g for g in focus if g not in official]
            if missing:
                raise RuntimeError(f"geometry.games not in the official set: {missing}")
            official = [g for g in official if g in focus]
            print(f"[pc] FOCUS SUBSET: {len(official)} game(s) {official} "
                  f"across {geometry['clones']} clones", flush=True)
        result["env_dir"] = env_dir
        baselines = pc_load_baselines(env_dir)

        stage["s"] = "start_arcade_server"
        clones_n = int(geometry["clones"])
        srv = _ca.CompetitionArcadeServer(
            game_ids=tuple(official), total_runs=clones_n,
            environments_dir=env_dir).start()
        try:
            clones = list(srv.exposed_game_ids)
            if len(clones) != clones_n:
                raise RuntimeError(f"expected {clones_n} clones, got {len(clones)}")
            cmap = {}
            for ei in srv._arcade.available_environments:
                src = None
                for tag in (getattr(ei, "private_tags", None) or []):
                    if str(tag).startswith("taaf_source_game:"):
                        src = str(tag).split(":", 1)[1]
                cmap[ei.game_id] = src
            recon = {f"k{i:03d}": official[i % len(official)] for i in range(len(clones))}
            mismatch = {k: (v, recon.get(k)) for k, v in cmap.items() if v != recon.get(k)}
            if mismatch or any(v is None for v in cmap.values()):
                raise RuntimeError(f"clone->source mapping broken: {mismatch or cmap}")
            result["clone_map"] = cmap

            stage["s"] = "run"
            games = [taaf.game_api.GameAPI(env_name=c, arcade_spec=srv.arcade_spec)
                     for c in clones]
            wave_bm = taaf.benchmark.Benchmark(
                label=f"pc_{arm}", games=games, solver=bm.solver, n_passes=1)
            for attr in ("max_runtime_s_per_game", "max_runtime_s"):
                if hasattr(wave_bm.solver, attr):
                    setattr(wave_bm.solver, attr, float(geometry["per_game_s"]))
            if hasattr(wave_bm.solver, "concurrency"):
                wave_bm.solver.concurrency = int(geometry["concurrency"])
            _PC_RUN["token"] = f"pc_{arm}_{int(t0)}"
            print(f"[pc] arm {arm}: {len(games)} clones @ {geometry['per_game_s']}s "
                  f"concurrency={getattr(wave_bm.solver, 'concurrency', None)} "
                  f"arcade={srv.base_url}", flush=True)
            await wave_bm.run(soft_end_time=None, runtime_environment=target,
                              minimal_diagnostics=False)

            stage["s"] = "collect"
            sessions = [s for tok, s in PC_SESSIONS if tok == _PC_RUN["token"]]
            if callable(dry_exercise):
                dry_exercise(sessions)
            diag_by_run = {}
            for sess in sessions:
                run = getattr(sess.game, "game_run", None)
                if run is not None:
                    diag_by_run[id(run)] = _pc_session_diag(sess)
            rows = []
            rows_by_source = {}
            for gr in (getattr(wave_bm, "game_runs", None) or []):
                apl = list(getattr(gr, "actions_per_level", []) or [])
                hist = len(getattr(gr, "history", []) or [])
                src = cmap.get(getattr(gr, "game_id", None))
                stem = (src or "?").split("-")[0]
                lv = int(getattr(gr, "levels_completed", 0) or 0)
                nl = int(getattr(gr, "number_of_levels", 0) or 0)
                row = {
                    "clone_id": getattr(gr, "game_id", None),
                    "source_game": stem,
                    "levels_completed": lv,
                    "levels_total": nl,
                    "actions_per_level": apl,
                    "actions_total": hist,
                    "apl_sum_matches_history": (sum(apl) == hist),
                    "state": str(getattr(gr, "state", None)),
                    "wallclock_s": getattr(gr, "final_wallclock_seconds", None),
                    "gen_tokens": _pc_row_tokens(gr),
                    "solver_note": str(getattr(gr, "solver_note", "") or ""),
                    "score": (pc_env_score(lv, apl, nl, baselines[src])
                              if src in baselines else None),
                }
                row.update(diag_by_run.get(id(gr), {}))
                rows.append(row)
                slot = rows_by_source.setdefault(
                    stem, {"levels": 0, "levels_total": nl, "n_clones": 0,
                           "actions_total": 0, "clone_ids": []})
                slot["levels"] = max(slot["levels"], lv)
                slot["n_clones"] += 1
                slot["actions_total"] += hist
                slot["clone_ids"].append(row["clone_id"])
                print(f"[pc] {arm} {stem}: levels={lv}/{nl} actions={hist} "
                      f"tok={row['gen_tokens']} score={row['score']} "
                      f"wd={row.get('watchdog')} graph={'yes' if 'graph' in row else 'no'}",
                      flush=True)
            result["rows"] = rows
            result["rows_by_source"] = rows_by_source

            stage["s"] = "diagnostics"
            anim_after = dict(PC_ANIMATION_UPTAKE)
            anim_delta = {k: anim_after[k] - anim_before[k] for k in anim_after}
            antifreeze_after = _pc_antifreeze_snapshot()
            antifreeze_delta = (
                {k: antifreeze_after[k] - antifreeze_before[k] for k in antifreeze_after}
                if antifreeze_before is not None and antifreeze_after is not None else None)
            compact_after = _pc_compact_snapshot()
            compact_delta = (
                {k: compact_after.get(k, 0) - compact_before.get(k, 0) for k in compact_after}
                if compact_before is not None and compact_after is not None else None)
            package_delta = _pc_package_delta(package_before, _pc_package_snapshot())
            result["patch_diagnostics"] = collect_patch_diagnostics(
                sessions, anim_delta, antifreeze_delta, compact_delta,
                package_delta=package_delta)

            # -- ADOPTION: real env actions per LLM deliberation ---------------
            llm_turns = PC_LLM_TURNS["analyze_calls"] - turns_before["analyze_calls"]
            actions_total_run = sum(int(r.get("actions_total") or 0) for r in rows)
            struct_d = (package_delta or {}).get("struct") or {}
            result["adoption"] = {
                "llm_turns": llm_turns,
                "actions_total": actions_total_run,
                "actions_per_llm_turn": (
                    round(actions_total_run / llm_turns, 2) if llm_turns else None),
                "plan_actions": int(struct_d.get("plan_actions", 0) or 0),
                "plan_actions_per_llm_turn": (
                    round(int(struct_d.get("plan_actions", 0) or 0) / llm_turns, 2)
                    if llm_turns else None),
            }
            print(f"[pc] adoption: {result['adoption']}", flush=True)
            stalls = result["patch_diagnostics"]["watchdog"]["stall_s_observed"]
            identity["watchdog_stall_s_observed"] = stalls
            expected_stall = float(arm_env.get("TAAF_WATCHDOG_STALL_S", "600"))
            if stalls and set(stalls) != {expected_stall}:
                raise RuntimeError(
                    f"[pc] arm {arm} sessions ran with watchdog stall {stalls}, "
                    f"expected {expected_stall} — env pin did not reach the sessions")

            stage["s"] = "behavior"
            if callable(behav_report):
                try:
                    result["behavior"] = behav_report()
                    print(f"[pc] behav corpus: {result['behavior']['corpus']}", flush=True)
                except Exception as e:  # noqa: BLE001
                    result["behavior"] = {"error": repr(e)}
            if callable(behav_assert):
                behav_assert()
        finally:
            try:
                srv.stop()
            except Exception:  # noqa: BLE001
                pass
        stage["s"] = "done"
    except Exception:
        result["error"] = _pc_traceback.format_exc()
        print(f"[pc] FAILED at stage={stage['s']}:\n{result['error']}", flush=True)

    dump()
    lvl = [r.get("levels_completed") or 0 for r in result["rows"]]
    pd = result.get("patch_diagnostics") or {}
    grinder = {k: v for k, v in (pd.get("graph") or {}).items() if "grind" in k}
    print(f"[pc] RESULT arm={arm} games={len(lvl)} "
          f"levels_sum={sum(lvl)} games_with_level={sum(1 for x in lvl if x > 0)} "
          f"animation={pd.get('animation')} grinder={grinder} "
          f"stale_closes={(pd.get('watchdog') or {}).get('stall_kills')}", flush=True)
    print(f"[pc] DONE stage={stage['s']} in {round(_pc_time.time() - t0, 1)}s", flush=True)
    return result
