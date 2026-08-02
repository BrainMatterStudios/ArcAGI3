# ============================================================================
# A/B wave driver — behavioral effect of the three shipped duck patches
# (watchdog f591392, HUD mask b28569c, replay-at-WIN a1d0378) on the real
# bundled Qwen3.6-27B duck.
#
# Design:
#   * ONE kernel session, ONE vLLM server. Arms differ ONLY in the three env
#     kill switches (TAAF_WATCHDOG / TAAF_HUD_MASK / TAAF_WIN_REPLAY), which
#     duck_patches.py provably reads AT CALL TIME (_watchdog_enabled(),
#     _hud_mask_enabled(), _win_replay_enabled() all read os.environ per call),
#     so toggling between waves in one process is a genuine arm switch.
#   * Waves A,B,B,A (counterbalanced): with exactly 2 pairs, ABBA cancels
#     linear server drift deterministically, which dominates randomisation at
#     n=2 (an unlucky draw gives AABB and re-introduces drift). The realized
#     order is logged per law 5 (HANDOFF-2026-08-01 §8).
#   * Per the A/A noise floor (RMS 0.707 levels/game-run), a single-session
#     score delta is NOISE. The primary readout is qualitative event counts
#     (watchdog fires, mask engagement, replay events, per-arm patch proof)
#     plus paired per-game level deltas — not the score.
#   * "An env toggle is not a shipped arm": every wave logs positive runtime
#     proof of its toggle state AND the patch layer's applied markers; the
#     serving assert proves the 27B is genuinely served BEFORE any game runs.
#
# This file is BOTH inlined into the kernel's run cell by build_ab_wmr.py AND
# imported by dry_run.py for the GPU-free local test. Edit here, then rebuild.
# ============================================================================
import glob as _ab_glob
import json as _ab_json
import os as _ab_os
import time as _ab_time
import traceback as _ab_traceback
import urllib.request as _ab_urlreq

AB_ARM_ENV = {
    "A": {"TAAF_WATCHDOG": "0", "TAAF_HUD_MASK": "0", "TAAF_WIN_REPLAY": "0"},
    "B": {"TAAF_WATCHDOG": "1", "TAAF_HUD_MASK": "1", "TAAF_WIN_REPLAY": "1"},
}

# Panel (10 games) chosen from the 196-run g0 base distribution
# (docs/test-artifacts-2026-08-02/RESULTS-2026-08-02-trace-audits.md):
#   unlock-sensitive (base sometimes completes 1-2 levels, mean lvl in parens):
#     sb26 (1.29) re86 (1.14) su15 (0.71) tu93 (0.71) vc33 (0.71)
#     bp35 (0.64) ar25 (0.57)
#   stall-prone / watchdog-sensitive (lowest realized throughput in the 3600s
#   box, 0 levels in 7 repeats): g50t (4-62 actions) ls20 (15-65 actions);
#     ft09 (14-34 actions but can hit 2 levels) doubles as both.
#   replay-sensitive: sb26 (lowest baselines of all 25, the only realistic
#     fast-full-win candidate). vc33/ls20/tu93 also exercise the HUD mask's
#     hardest cases (paint fills, two-row bar, edge strip).
AB_DEFAULT_GAMES = "sb26,re86,su15,tu93,vc33,bp35,ar25,ft09,g50t,ls20"


def _ab_cfg(name: str, default: str) -> str:
    return _ab_os.environ.get(name, "").strip() or default


AB_GAMES = [s.strip() for s in _ab_cfg("AB_GAMES", AB_DEFAULT_GAMES).split(",") if s.strip()]
AB_WAVES = [w.strip().upper() for w in _ab_cfg("AB_WAVES", "A,B,B,A").split(",") if w.strip()]
AB_BUDGET = float(_ab_cfg("AB_BUDGET", "3600"))          # per-game wall cap, seconds
AB_DEADLINE_S = float(_ab_cfg("AB_DEADLINE_S", str(int(7.6 * 3600))))  # from notebook start
AB_WAVE_OVERHEAD_S = 900.0                               # server start + collect + slack
AB_DRY_RUN = _ab_os.environ.get("AB_DRY_RUN", "") == "1"

AB_SESSIONS = []                # (wave_index, session) — filled by the registry wrapper
_AB_WAVE = {"i": -1}


# --- toggle semantics: EXACTLY duck_patches.py's readers -----------------------


def _ab_toggles():
    def on(name):
        return _ab_os.environ.get(name, "1").strip() not in {"0", "false", "False"}

    return {
        "TAAF_WATCHDOG": on("TAAF_WATCHDOG"),
        "TAAF_HUD_MASK": on("TAAF_HUD_MASK"),
        "TAAF_WIN_REPLAY": on("TAAF_WIN_REPLAY"),
    }


def _ab_patch_proof():
    """Positive proof the patch layer is installed on the live session class.

    Captured BEFORE the registry wrapper re-wraps play/_execute_action (the
    wrapper would hide the markers on the class attributes).
    """
    from inference.framework import solver as _sv

    cls = _sv._HarnessGameSession
    return {
        "watchdog_should_stop_patched": bool(getattr(cls.should_stop, "_watchdog_patched", False)),
        "hud_or_outer_execute_patched": bool(getattr(cls._execute_action, "_hud_patched", False)
                                             or getattr(cls._execute_action, "_win_replay_patched", False)
                                             or getattr(cls._execute_action, "_behav", False)),
        "play_patched": bool(getattr(cls.play, "_win_replay_patched", False)
                             or getattr(cls.play, "_watchdog_play_patched", False)),
        "execute_chain_repr": repr(cls._execute_action),
        "play_chain_repr": repr(cls.play),
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
    d["trace_len"] = len(getattr(session, "_replay_trace", None) or [])
    d["action_count"] = int(getattr(session, "action_count", 0) or 0)
    return d


# --- main ----------------------------------------------------------------------


async def ab_main(bm, target, working_dir, notebook_start=None,
                  behav_report=None, behav_raw=None):
    from pathlib import Path

    working_dir = Path(working_dir)
    t0 = notebook_start if notebook_start is not None else _ab_time.time()
    stage = {"s": "init"}
    result = {
        "experiment": "ab_wmr",
        "arms": AB_ARM_ENV,
        "games": AB_GAMES,
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
        # -- config snapshot: proof both arms share sampling/config ------------
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
            expected = arm == "B"
            if any(v != expected for v in toggles.values()):
                raise RuntimeError(f"arm {arm} toggle mismatch: {toggles}")
            print(f"[ab] === wave {wi} arm {arm} toggles {toggles} "
                  f"elapsed {elapsed:.0f}s ===", flush=True)

            stage["s"] = f"wave{wi}_{arm}_server"
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
                        "gen_tokens": getattr(gr, "final_generated_tokens", None),
                        "score": (ab_env_score(lv, apl, nl, baselines[src])
                                  if src in baselines else None),
                    }
                    row.update(diag_by_run.get(id(gr), {}))
                    rows.append(row)
                    print(f"[ab] w{wi} {arm} {src}: levels={lv}/{nl} actions={hist} "
                          f"score={row['score']} wd={row.get('watchdog')} "
                          f"hud={row.get('hud')} replay={(row.get('replay') or {}).get('status')}",
                          flush=True)

                wave_rec = {
                    "wave": wi, "arm": arm, "env": dict(AB_ARM_ENV[arm]),
                    "toggles_verified": toggles, "wall_s": round(wall, 1),
                    "started_at_elapsed_s": round(elapsed, 1), "rows": rows,
                }
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
    header = (f"{'game':6} {'wave':4} {'arm':3} {'lvl':>6} {'acts':>6} {'score':>8} "
              f"{'wd_rst':>6} {'wd_kill':>8} {'hud_cells':>9} {'hud_supp':>8} "
              f"{'guard':>5} {'replay':>12}")
    print("[ab] " + header, flush=True)
    per_arm = {}
    for w in waves:
        for r in w.get("rows", []):
            stem = (r.get("source_game") or "?").split("-")[0]
            wd = r.get("watchdog") or {}
            hud = r.get("hud") or {}
            rp = r.get("replay") or {}
            score = r.get("score")
            score_s = f"{score:8.2f}" if isinstance(score, (int, float)) else f"{'-':>8}"
            print(f"[ab] {stem:6} {w['wave']:<4} {w['arm']:3} "
                  f"{r.get('levels_completed', 0):>3}/{r.get('levels_total', 0):<2} "
                  f"{r.get('actions_total', 0):>6} {score_s} "
                  f"{wd.get('resets_done', '-')!s:>6} {wd.get('killed', '-')!s:>8} "
                  f"{hud.get('mask_cells', '-')!s:>9} {hud.get('suppressed_hud_only', '-')!s:>8} "
                  f"{hud.get('guard_dead_lines', '-')!s:>5} {rp.get('status', '-')!s:>12}",
                  flush=True)
            slot = per_arm.setdefault(stem, {}).setdefault(w["arm"], {"lvl": [], "score": []})
            slot["lvl"].append(r.get("levels_completed", 0) or 0)
            if isinstance(score, (int, float)):
                slot["score"].append(score)
    print("[ab] ---- paired per-game means (B - A) ----", flush=True)
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
        print(f"[ab] TOTAL paired delta over {n} games: levels {dl_sum:+.1f}, "
              f"score {ds_sum:+.2f}  (A/A noise floor: RMS 0.707 levels/game-run — "
              f"read event counts, not small score deltas)", flush=True)
