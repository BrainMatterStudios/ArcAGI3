#!/usr/bin/env python3
"""build_carryover_smoke.py — GPU smoke for the yield_carryover graft (ladder lever #2).

ARM (2026-08-24, smoke queue #2 in docs/RESULTS-2026-08-22-testing-campaign.md):
duck38-v12 base (Aug-07 anim bundle + Qwen3.8-27B-FP8 repacked, boot-attested)
+ TWO grafts, stated explicitly:

  1. submission/_effort_medium/graft_effort.py with EFFORT_MEDIUM=1,
     EFFORT_DEAD_RETRY=0 — the VALIDATED BASELINE (effort-smoke 08-23 ran
     clean: vc33 L2/4.73, tn36 L1/1.52, dead completions 0/212, 217/217
     payloads carried reasoning_effort=medium). Effort rides tonight's arm,
     so the ladder read for carryover is measured ON TOP of effort.
  2. submission/_yield_carryover/graft_carryover.py with YIELD_CARRYOVER=1,
     YIELD_SLICE_CAP=0 — THE ladder variable: digest-only first, per its
     smoke plan (the 3-slice cap smokes later). Targets the ~2.2h/9h
     yield-resume context-wipe wall (docs/RESEARCH-2026-08-21-bug-lever-hunt.md
     Tier-1 #1: 51.5% of wall in slices ending without an action; 71
     byte-identical re-issued snippets; worst turn 35 slices; history
     collapse to 2-4 messages on resume).

SMOKE GEOMETRY (identical to effort-smoke): 4 offline dev games —
vc33-5430563c, tn36-ef4dde99, ft09-0d8bbf25, sk48-d8078629; per-game cap
3000 s (~50 min), global soft end 12600 s, concurrency 28.

DIAGNOSTIC (the reads this smoke exists for; printed + persisted JSON):
per-game levels/scores; yield-resume event count; slices per turn (max +
distribution); byte-identical duplicate snippets within turns (falsifier —
corpus baseline 71 duplicates, worst turn 35 slices); history_messages count
on resumed slices (corpus collapse: 2-4); digest injection count (exactly one
alive per resumed slice — multi-digest counters must stay 0); plus effort
baseline proof (reasoning_effort=medium payload counter) and dead completions.

PRE-REGISTERED READING: PASS = duplicate snippets and max-slices-per-turn
collapse vs the corpus baseline AND no level regression vs the effort-smoke
comparator (vc33 >= L2, tn36 >= L1).

Usage:
  .venv/bin/python submission/_carryover_smoke/build_carryover_smoke.py
  cd submission/_carryover_smoke && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_NB = HERE.parent / "_duck38_v12" / "arc3-duck38-v12.ipynb"
EFFORT_PY = HERE.parent / "_effort_medium" / "graft_effort.py"
CARRYOVER_PY = HERE.parent / "_yield_carryover" / "graft_carryover.py"
KERNEL_SLUG = "arc3-carryover-smoke"

MARK_IMPORTS = "NOTEBOOK_START_EPOCH = time.time()"
MARK_SMOKE = "# Smoke/eval hook:"
MARK_RUN = "run_context = contextlib.nullcontext()"
MARK_ATTEST = "attest: OK"

GPU_ASSERT_CELL = r'''# Fail-fast GPU assert: metadata machine_shape + --accelerator alone can still
# bind P100 (3 wasted pushes on serving-lab proved it; the competition source
# attachment is the real RTX Pro 6000 gate). Die here, before any setup cost.
import subprocess as _sp

_gpu = _sp.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("boot gpu:", (_gpu.stdout or "").strip() or (_gpu.stderr or "").strip())
_gpu_name = (_gpu.stdout or "").upper()
assert "RTX" in _gpu_name and "6000" in _gpu_name, (
    f"GPU misbind — expected RTX Pro 6000, got: {_gpu.stdout!r} {_gpu.stderr!r}"
)
'''

SMOKE_CELL = r'''# Smoke/eval hook: a NORMAL COMMIT runs a 4-game offline smoke on the scored
# GPU class — the yield_carryover ladder read (on top of the effort baseline).
# The scored rerun path (KAGGLE_IS_COMPETITION_RERUN) never enters this branch.
# Games identical to effort-smoke: vc33/tn36 carry effort-smoke comparators
# (L2/4.73, L1/1.52); sk48 is the thinking-stall game; ft09 the digest-lever game.
SMOKE_GAMES = ["vc33-5430563c", "tn36-ef4dde99", "ft09-0d8bbf25", "sk48-d8078629"]

if not run_as_submission:
    import arc_agi
    from taaf.game_api import ArcadeSpec, GameAPI

    def _resolve_env_dir():
        candidates = [
            Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"),
            Path("/kaggle/input/arc-prize-2026-arc-agi-3/environment_files"),
        ]
        for cand in candidates:
            if cand.is_dir():
                return str(cand)
        for hit in Path("/kaggle/input").rglob("environment_files"):
            if hit.is_dir():
                return str(hit)
        raise RuntimeError("environment_files dir not found in /kaggle/input")

    _env_dir = _resolve_env_dir()
    _spec = ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=_env_dir)
    bm.games = [GameAPI(env_name=name, arcade_spec=_spec) for name in SMOKE_GAMES]
    bm.n_passes = 1
    bm.game_weights = None
    bm.label = "carryover-smoke"
    # Per-game cap ~50 min (bundled solver config: 7920 s). Games run
    # concurrently (solver concurrency 28), so wall = setup + ~50 min + teardown.
    bm.solver.max_runtime_s_per_game = 3000.0
    # Global backstop 3.5 h from notebook start => total run <= 4 h.
    soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=12600)
    print(f"smoke hook: {len(bm.games)} games, env_dir={_env_dir}, "
          f"per_game_cap={bm.solver.max_runtime_s_per_game}s, soft_end={soft_end}")
else:
    print("scored rerun: smoke hook inert — full competition games")

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
'''

GRAFT_CELL_TEMPLATE = '''# Graft install — TWO grafts, stated explicitly (ladder arm, not single-variable
# vs stock: the variable is carryover, measured ON TOP of the effort baseline).
# 1. EFFORT_MEDIUM=1 (graft_effort; EFFORT_DEAD_RETRY=0) — the validated
#    baseline riding tonight's arm (effort-smoke 08-23: clean, 0 dead, 217/217
#    medium payloads, vc33 L2 / tn36 L1).
# 2. YIELD_CARRYOVER=1 (graft_carryover; YIELD_SLICE_CAP=0) — THE ladder
#    variable of this smoke: slice-digest carry-forward only, cap disabled
#    (digest-only first per the smoke plan; the cap smokes later).
os.environ["EFFORT_MEDIUM"] = "1"
os.environ["EFFORT_DEAD_RETRY"] = "0"
os.environ["YIELD_CARRYOVER"] = "1"
os.environ["YIELD_SLICE_CAP"] = "0"

_EFFORT_SOURCE = {effort_source!r}
_CARRYOVER_SOURCE = {carryover_source!r}

import importlib.util as _ilu


def _load_graft(name, source):
    path = WORKING_DIR / (name + ".py")
    path.write_text(source, encoding="utf-8")
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Install ORDER: effort first (the baseline), carryover second (the ladder
# variable wraps on top). Seams are disjoint except ToolAgent.analyze, where
# carryover deliberately wraps effort's wrapper.
_effort = _load_graft("graft_effort", _EFFORT_SOURCE)
_effort_status = _effort.install()
print("[effort]", _effort_status)
assert _effort_status == "effort_medium: OK", (
    "baseline graft must be live, got: " + repr(_effort_status)
)
_carry = _load_graft("graft_carryover", _CARRYOVER_SOURCE)
_carry_status = _carry.install()
print("[carryover]", _carry_status)
# A smoke that silently measures the baseline alone is worse than one that
# dies: hard gate on the ladder variable.
assert _carry_status == "yield_carryover: OK", (
    "ladder-variable graft must be live, got: " + repr(_carry_status)
)
'''

TELEMETRY_CELL = r'''# Carryover smoke telemetry — the reads this smoke exists for:
# yield-resume events; slices per turn (max + distribution); byte-identical
# duplicate snippets within turns (corpus baseline: 71 duplicates, worst turn
# 35 slices); history_messages count on resumed slices (corpus collapse: 2-4);
# digest injections (exactly one alive per resumed slice — multi-digest
# counters must stay 0); effort baseline proof (reasoning_effort=medium
# payloads) and dead completions (effort-smoke read continuity).
# Wraps the ALREADY-GRAFTED seams — counting sits outside both grafts.
import threading as _tel_threading
from collections import Counter as _tel_Counter

import graft_carryover as _g_carry
import graft_effort as _g_eff
from inference.agent import tool_agent as _tel_ta
from inference.utils import openai_compat as _tel_oc

TELEMETRY = {
    "requests": 0,
    "payloads": 0,
    "payloads_effort_medium": 0,
    "dead_completions": 0,
    "reasoning_chars": 0,
    "tool_call_completions": 0,
    "finish_reasons": {},
    "analyze_calls": 0,
    "turns_completed": 0,
    "yield_noaction_events": 0,
    "resume_slices": 0,
    "slices_per_turn_dist": {},
    "max_slices_per_turn": 0,
    "duplicate_snippets_total": 0,
    "worst_turn_duplicates": 0,
    "turns_with_duplicates": 0,
    "history_len_on_resume": [],
    "history_len_on_resume_stats": {"n": 0, "min": None, "max": None, "sum": 0},
    "digest_prompts": 0,
    "final_prompts": 0,
    "multi_digest_prompts": 0,
    "requests_with_digest": 0,
    "multi_digest_requests": 0,
    "max_digest_blocks_in_request": 0,
}
_tel_lock = _tel_threading.Lock()
_TEL_PATH = WORKING_DIR / "carryover_telemetry.json"
_TEL_AGENTS = {}  # id(agent) -> agent, so open turns can be flushed at report time

# --- effort baseline proof: payloads actually carrying reasoning_effort=medium ---
_tel_inner_build = _tel_oc.build_chat_payload
assert getattr(_tel_inner_build, "_effort_medium_patched", False), "effort graft must be installed first"


def _tel_build(*args, **kwargs):
    payload = _tel_inner_build(*args, **kwargs)
    try:
        effort = (payload.get("chat_template_kwargs") or {}).get("reasoning_effort")
        with _tel_lock:
            TELEMETRY["payloads"] += 1
            if effort == "medium":
                TELEMETRY["payloads_effort_medium"] += 1
    except Exception:  # noqa: BLE001 — telemetry must never break a request
        pass
    return payload


_tel_build._effort_medium_patched = True  # keep graft idempotence marker intact
_tel_oc.build_chat_payload = _tel_build
_tel_ta.build_chat_payload = _tel_build

# --- digest injection read: the slice's initial user prompt, post-graft ---
_tel_inner_prompt = _tel_ta.ToolAgent._build_user_prompt


def _tel_prompt(self, *args, **kwargs):
    prompt = _tel_inner_prompt(self, *args, **kwargs)
    try:
        n_digest = prompt.count(_g_carry.DIGEST_HEADER)
        with _tel_lock:
            if n_digest:
                TELEMETRY["digest_prompts"] += 1
                if n_digest > 1:
                    TELEMETRY["multi_digest_prompts"] += 1
            if prompt.startswith(_g_carry.FINAL_SLICE_INSTRUCTION):
                TELEMETRY["final_prompts"] += 1  # must stay 0: YIELD_SLICE_CAP=0
    except Exception:  # noqa: BLE001
        pass
    return prompt


_tel_ta.ToolAgent._build_user_prompt = _tel_prompt

# --- wire-level: digest blocks per request + dead completions + turn snippets ---
_tel_inner_chat = _tel_ta.ToolAgent._chat_completion


def _tel_record(self, messages, result):
    digest_blocks = 0
    try:
        for message in messages or []:
            if isinstance(message, dict):
                digest_blocks += _g_carry._message_text(message).count(_g_carry.DIGEST_HEADER)
    except Exception:  # noqa: BLE001
        digest_blocks = -1
    message = getattr(result, "message", None)
    message = message if isinstance(message, dict) else {}
    finish = str(getattr(result, "finish_reason", "") or "")
    try:
        reasoning = _tel_ta._extract_reasoning_text(message)
    except Exception:  # noqa: BLE001
        reasoning = ""
    dead = _g_eff._is_dead_completion(result, _tel_ta)
    codes = []
    for tool_call in message.get("tool_calls") or []:
        try:
            codes.append(_g_carry._tool_call_code(tool_call))
        except Exception:  # noqa: BLE001
            pass
    if codes:
        bucket = getattr(self, "_tel_turn_codes", None)
        if bucket is None:
            bucket = []
            self._tel_turn_codes = bucket
        bucket.extend(codes)
    with _tel_lock:
        TELEMETRY["requests"] += 1
        TELEMETRY["finish_reasons"][finish] = TELEMETRY["finish_reasons"].get(finish, 0) + 1
        TELEMETRY["reasoning_chars"] += len(reasoning or "")
        if message.get("tool_calls"):
            TELEMETRY["tool_call_completions"] += 1
        if dead:
            TELEMETRY["dead_completions"] += 1
        if digest_blocks > 0:
            TELEMETRY["requests_with_digest"] += 1
            if digest_blocks > TELEMETRY["max_digest_blocks_in_request"]:
                TELEMETRY["max_digest_blocks_in_request"] = digest_blocks
            if digest_blocks > 1:
                TELEMETRY["multi_digest_requests"] += 1
        n = TELEMETRY["requests"]
        snapshot = json.dumps(TELEMETRY, indent=1)
    if n % 20 == 0:
        try:
            _TEL_PATH.write_text(snapshot, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[carryover-telemetry] req={n} resumes={TELEMETRY['resume_slices']} "
            f"yields={TELEMETRY['yield_noaction_events']} "
            f"dups={TELEMETRY['duplicate_snippets_total']} "
            f"max_slices={TELEMETRY['max_slices_per_turn']} "
            f"digest_prompts={TELEMETRY['digest_prompts']} "
            f"effort_payloads={TELEMETRY['payloads_effort_medium']}/{TELEMETRY['payloads']}",
            flush=True,
        )


def _tel_chat(self, messages, *args, **kwargs):
    result = _tel_inner_chat(self, messages, *args, **kwargs)
    try:
        _tel_record(self, messages, result)
    except Exception:  # noqa: BLE001
        pass
    return result


_tel_ta.ToolAgent._chat_completion = _tel_chat

# --- turn/slice accounting around analyze (outermost wrapper) ---
_tel_inner_analyze = _tel_ta.ToolAgent.analyze
assert getattr(_tel_inner_analyze, "_yield_carryover_patched", False), "carryover graft must be installed first"


def _tel_flush_turn(self):
    codes = getattr(self, "_tel_turn_codes", None) or []
    slices = int(getattr(self, "_tel_turn_slices", 0) or 0)
    if slices <= 0 and not codes:
        return
    counts = _tel_Counter(codes)
    dups = sum(c - 1 for c in counts.values() if c > 1)
    with _tel_lock:
        TELEMETRY["turns_completed"] += 1
        key = str(slices)
        TELEMETRY["slices_per_turn_dist"][key] = TELEMETRY["slices_per_turn_dist"].get(key, 0) + 1
        if slices > TELEMETRY["max_slices_per_turn"]:
            TELEMETRY["max_slices_per_turn"] = slices
        TELEMETRY["duplicate_snippets_total"] += dups
        if dups:
            TELEMETRY["turns_with_duplicates"] += 1
            if dups > TELEMETRY["worst_turn_duplicates"]:
                TELEMETRY["worst_turn_duplicates"] = dups
    self._tel_turn_codes = []
    self._tel_turn_slices = 0


def _tel_analyze(self, state_path, action_num, *args, **kwargs):
    try:
        _TEL_AGENTS[id(self)] = self
        analysis_step = kwargs.get("analysis_step")
        key = (str(state_path), analysis_step)
        # Mirror the graft's resume test against the graft's OWN pre-call state
        # (telemetry is outermost, so _yc_* still reflect the previous slice).
        is_resume = (
            analysis_step is not None
            and key == getattr(self, "_yc_key", None)
            and int(getattr(self, "_yc_slices", 0) or 0) >= 1
        )
        if is_resume:
            hist = getattr(self, "_history_messages", None)
            hist_len = len(hist) if isinstance(hist, list) else -1
            with _tel_lock:
                TELEMETRY["resume_slices"] += 1
                stats = TELEMETRY["history_len_on_resume_stats"]
                stats["n"] += 1
                stats["sum"] += max(hist_len, 0)
                if stats["min"] is None or hist_len < stats["min"]:
                    stats["min"] = hist_len
                if stats["max"] is None or hist_len > stats["max"]:
                    stats["max"] = hist_len
                if len(TELEMETRY["history_len_on_resume"]) < 400:
                    TELEMETRY["history_len_on_resume"].append(hist_len)
        else:
            _tel_flush_turn(self)  # previous turn left open on this agent
        self._tel_turn_slices = int(getattr(self, "_tel_turn_slices", 0) or 0) + 1
    except Exception:  # noqa: BLE001
        pass
    # Never guard the inner call: an inner crash must propagate as stock.
    result = _tel_inner_analyze(self, state_path, action_num, *args, **kwargs)
    try:
        with _tel_lock:
            TELEMETRY["analyze_calls"] += 1
        yielded = result is not None and bool(getattr(result, "yielded_control", False))
        acted = result is not None and bool(getattr(result, "step_executed", False))
        if yielded and not acted:
            with _tel_lock:
                TELEMETRY["yield_noaction_events"] += 1
        else:
            _tel_flush_turn(self)
    except Exception:  # noqa: BLE001
        pass
    return result


_tel_analyze._yield_carryover_patched = True  # keep graft idempotence marker intact
_tel_ta.ToolAgent.analyze = _tel_analyze
print("[carryover-telemetry] counters installed (corpus baseline: 71 dup snippets, worst turn 35 slices, history collapse 2-4)")
'''

REPORT_CELL = r'''# ---- carryover-smoke final report (grep for CARRYOVER SMOKE / GAME / TELEMETRY) ----
for _agent in list(_TEL_AGENTS.values()):
    try:
        _tel_flush_turn(_agent)  # close any turn left open at game end
    except Exception:  # noqa: BLE001
        pass

print("=" * 72)
print("CARRYOVER SMOKE RESULTS (YIELD_CARRYOVER=1, YIELD_SLICE_CAP=0, EFFORT_MEDIUM=1, EFFORT_DEAD_RETRY=0)")
games_out = []
for game_run in bm.game_runs:
    actions = sum(game_run.actions_per_level) if game_run.actions_per_level else len(game_run.history)
    row = {
        "game_id": game_run.game_id,
        "state": game_run.state,
        "levels_completed": game_run.levels_completed,
        "number_of_levels": game_run.number_of_levels,
        "final_score": game_run.final_score,
        "actions": actions,
        "wallclock_s": game_run.final_wallclock_seconds,
    }
    games_out.append(row)
    print(
        f"GAME {row['game_id']}: state={row['state']} "
        f"levels={row['levels_completed']}/{row['number_of_levels']} "
        f"final_score={row['final_score']} actions={row['actions']} "
        f"wallclock_s={row['wallclock_s']}"
    )

with _tel_lock:
    tel = json.loads(json.dumps(TELEMETRY))
hist_stats = tel["history_len_on_resume_stats"]
hist_mean = (hist_stats["sum"] / hist_stats["n"]) if hist_stats["n"] else None
print(
    f"TELEMETRY analyze_calls={tel['analyze_calls']} turns_completed={tel['turns_completed']} "
    f"yield_noaction_events={tel['yield_noaction_events']} resume_slices={tel['resume_slices']}"
)
print(
    f"TELEMETRY slices_per_turn max={tel['max_slices_per_turn']} "
    f"(corpus worst 35) dist={tel['slices_per_turn_dist']}"
)
print(
    f"TELEMETRY duplicate_snippets_total={tel['duplicate_snippets_total']} "
    f"(corpus baseline 71) worst_turn={tel['worst_turn_duplicates']} "
    f"turns_with_duplicates={tel['turns_with_duplicates']}"
)
print(
    f"TELEMETRY history_len_on_resume min={hist_stats['min']} max={hist_stats['max']} "
    f"mean={hist_mean} n={hist_stats['n']} (corpus collapse: 2-4)"
)
print(
    f"TELEMETRY digest_prompts={tel['digest_prompts']} final_prompts={tel['final_prompts']} "
    f"multi_digest_prompts={tel['multi_digest_prompts']} "
    f"requests_with_digest={tel['requests_with_digest']} "
    f"multi_digest_requests={tel['multi_digest_requests']} "
    f"max_digest_blocks_in_request={tel['max_digest_blocks_in_request']}"
)
print(
    f"TELEMETRY requests={tel['requests']} payloads={tel['payloads']} "
    f"effort_medium_payloads={tel['payloads_effort_medium']} "
    f"dead_completions={tel['dead_completions']} "
    f"reasoning_chars_total={tel['reasoning_chars']} "
    f"tool_call_completions={tel['tool_call_completions']}"
)
print("TELEMETRY finish_reasons=", tel["finish_reasons"])

_levels = {row["game_id"].split("-")[0]: row["levels_completed"] for row in games_out}
_no_regression = _levels.get("vc33", 0) >= 2 and _levels.get("tn36", 0) >= 1
print(
    "PASS BARS: dup snippets + max slices/turn collapse vs corpus (71 dups / worst 35); "
    "no level regression vs effort-smoke (vc33>=2, tn36>=1). "
    f"level_bar_met={_no_regression} "
    f"observed dups={tel['duplicate_snippets_total']} max_slices={tel['max_slices_per_turn']}"
)

_results = {
    "arm": "carryover-smoke",
    "flags": {
        "YIELD_CARRYOVER": "1",
        "YIELD_SLICE_CAP": "0",
        "EFFORT_MEDIUM": "1",
        "EFFORT_DEAD_RETRY": "0",
    },
    "games": games_out,
    "telemetry": tel,
    "level_bar_met": _no_regression,
    "comparators": {
        "effort_smoke_0823": {
            "vc33": {"levels": 2, "score": 4.7319},
            "tn36": {"levels": 1, "score": 1.5232},
            "ft09": {"levels": 0, "score": 0.0},
            "sk48": {"levels": 0, "score": 0.0},
        },
        "corpus_duplicate_snippets": 71,
        "corpus_worst_turn_slices": 35,
        "corpus_history_collapse": "2-4",
    },
}
(WORKING_DIR / "carryover_smoke_results.json").write_text(
    json.dumps(_results, indent=1), encoding="utf-8"
)
print("wrote", WORKING_DIR / "carryover_smoke_results.json")
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    effort_source = EFFORT_PY.read_text()
    carryover_source = CARRYOVER_PY.read_text()
    assert "def install() -> str:" in effort_source
    assert "def install() -> str:" in carryover_source

    def idx_of(marker: str) -> int:
        hits = [
            i
            for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and marker in "".join(c["source"])
        ]
        assert len(hits) == 1, (marker, len(hits))
        return hits[0]

    def code_cell(text: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }

    # 1. GPU fail-fast right after the imports cell.
    nb["cells"].insert(idx_of(MARK_IMPORTS) + 1, code_cell(GPU_ASSERT_CELL))

    # 2. Replace the base smoke hook with the 4-game carryover geometry.
    nb["cells"][idx_of(MARK_SMOKE)]["source"] = SMOKE_CELL.splitlines(keepends=True)

    # 3+4. Graft install (effort baseline + carryover variable) + telemetry,
    # immediately before the run cell.
    run_idx = idx_of(MARK_RUN)
    graft_cell_text = GRAFT_CELL_TEMPLATE.format(
        effort_source=effort_source, carryover_source=carryover_source
    )
    # The graft sources must ride the notebook byte-identically: the template
    # embeds them via !r, and Python str repr round-trips by language guarantee.
    assert repr(effort_source) in graft_cell_text
    assert repr(carryover_source) in graft_cell_text
    nb["cells"].insert(run_idx, code_cell(graft_cell_text))
    nb["cells"].insert(run_idx + 1, code_cell(TELEMETRY_CELL))

    # 5. Final report after the run cell.
    nb["cells"].insert(idx_of(MARK_RUN) + 1, code_cell(REPORT_CELL))

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined
    assert joined.index("GPU misbind") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index('SMOKE_GAMES = ["vc33-5430563c"')
    assert joined.index('os.environ["YIELD_CARRYOVER"] = "1"') < joined.index(MARK_RUN)
    assert joined.index('os.environ["YIELD_SLICE_CAP"] = "0"') < joined.index(MARK_RUN)
    assert joined.index('os.environ["EFFORT_MEDIUM"] = "1"') < joined.index(MARK_RUN)
    assert joined.index("counters installed") < joined.index(MARK_RUN)
    assert joined.index(MARK_RUN) < joined.index("CARRYOVER SMOKE RESULTS")
    assert joined.count("sk48-d8078629") == 1

    (HERE / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (HERE / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"ahmedmobasher86/{KERNEL_SLUG}",
                "title": KERNEL_SLUG,
                "code_file": f"{KERNEL_SLUG}.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "machine_shape": "NvidiaRtxPro6000",
                "dataset_sources": [
                    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
                    "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
                    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
                ],
                "kernel_sources": [],
                # LAW: the competition source is what gates the RTX Pro 6000
                # pool — machine_shape + --accelerator alone bind P100.
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "model_sources": [
                    "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"
                ],
            },
            indent=2,
        )
        + "\n"
    )

    import hashlib

    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    print("built", KERNEL_SLUG, "code-cell sha256", hashlib.sha256(code.encode()).hexdigest())


if __name__ == "__main__":
    main()
