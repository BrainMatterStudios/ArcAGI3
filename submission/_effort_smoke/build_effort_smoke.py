#!/usr/bin/env python3
"""build_effort_smoke.py — single-variable GPU smoke for the effort_medium graft.

ARM (2026-08-22, smoke queue #1 in docs/RESULTS-2026-08-22-testing-campaign.md):
duck38-v12 base (Aug-07 anim bundle + Qwen3.8-27B-FP8 repacked, boot-attested)
+ submission/_effort_medium/graft_effort.py installed with

    EFFORT_MEDIUM=1   (reasoning_effort=medium rides chat_template_kwargs on
                       every vllm request — the official template defaults to
                       xhigh; docs/RESEARCH-2026-08-21-bug-lever-hunt.md Tier-1 #2)
    EFFORT_DEAD_RETRY=0  (retry hardening OFF — single-variable purity)

PRE-FLIGHT (this session): the graft's 13-test offline suite run against the
ACTUAL mounted bundle (jakobbrggen/taaf-kaggle-source-anim-20260807-anim,
tool_agent.py md5 6a0e3dfd963e59d14f0a57d121c578eb — differs from June stock)
=> 13/13 PASS; openai_compat.py is byte-identical to June stock.

SMOKE GEOMETRY: 4 offline dev games — vc33-5430563c, tn36-ef4dde99,
ft09-0d8bbf25, sk48-d8078629 (comparators: v12 smoke vc33 10.71 / tn36 3.57;
sk48 = thinking-stall game, ft09 = digest-lever game). Per-game cap 3000 s
(~50 min; the bundled solver config carries 7920 s), global soft end 12600 s
(3.5 h) => total <= 4 h including ~35 min serve setup. Games run concurrently
(bundled solver concurrency = 28).

DIAGNOSTIC (the read this smoke exists for): every chat completion is counted;
DEAD = finish_reason=stop AND zero tool calls AND empty content (corpus
baseline: 122 dead of ~3,480 requests = 3.5%, 2.62M reasoning chars — bug-lever
hunt forensics). The kernel prints per-game scores/levels AND
dead-completion count + total reasoning chars, plus a wire-level proof counter
(payloads carrying reasoning_effort=medium).

PRE-REGISTERED READING: PASS = dead-completion rate collapses vs 3.5% AND no
score regression beyond smoke noise on vc33/tn36. FAIL = regression or no
dead-completion change.

Usage:
  .venv/bin/python submission/_effort_smoke/build_effort_smoke.py
  cd submission/_effort_smoke && python3 -m kaggle kernels push -p . --accelerator NvidiaRtxPro6000
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE_NB = HERE.parent / "_duck38_v12" / "arc3-duck38-v12.ipynb"
GRAFT_PY = HERE.parent / "_effort_medium" / "graft_effort.py"
KERNEL_SLUG = "arc3-effort-smoke"

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
# GPU class — the effort_medium single-variable read. The scored rerun path
# (KAGGLE_IS_COMPETITION_RERUN) never enters this branch.
# Games: vc33/tn36 carry v12-smoke comparators (10.71 / 3.57); sk48 is the
# thinking-stall game; ft09 the digest-lever game.
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
    bm.label = "effort-smoke"
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

GRAFT_CELL_TEMPLATE = '''# EFFORT_MEDIUM graft install (THE single variable of this smoke).
# EFFORT_DEAD_RETRY=0: the paired retry hardening stays OFF for purity — only
# reasoning_effort=medium is injected (setdefault into chat_template_kwargs on
# vllm requests; official template default is xhigh).
# Pre-flight: 13/13 offline tests PASS against this exact mounted bundle
# (tool_agent.py md5 6a0e3dfd963e59d14f0a57d121c578eb).
os.environ["EFFORT_MEDIUM"] = "1"
os.environ["EFFORT_DEAD_RETRY"] = "0"

_GRAFT_SOURCE = {graft_source!r}

import importlib.util as _ilu

_graft_path = WORKING_DIR / "graft_effort.py"
_graft_path.write_text(_GRAFT_SOURCE, encoding="utf-8")
_graft_spec = _ilu.spec_from_file_location("graft_effort", _graft_path)
_graft = _ilu.module_from_spec(_graft_spec)
sys.modules["graft_effort"] = _graft
_graft_spec.loader.exec_module(_graft)
_graft_status = _graft.install()
print("[effort]", _graft_status)
# A smoke that silently measures stock is worse than one that dies: hard gate.
assert _graft_status == "effort_medium: OK", (
    f"single-variable smoke requires the graft live, got: {{_graft_status}}"
)
'''

TELEMETRY_CELL = r'''# Smoke telemetry — the diagnostic this smoke exists to read.
# DEAD completion = finish_reason=stop AND zero tool calls AND empty content
# (corpus baseline: 122 dead / ~3,480 requests = 3.5%; 2.62M reasoning chars —
# docs/RESEARCH-2026-08-21-bug-lever-hunt.md Tier-1 #2 forensics).
# Wraps the ALREADY-GRAFTED seams (counting sits outside the graft), plus a
# wire-level proof counter: payloads that actually carry
# chat_template_kwargs.reasoning_effort == "medium".
import threading as _tel_threading

from inference.agent import tool_agent as _tel_ta
from inference.utils import openai_compat as _tel_oc

TELEMETRY = {
    "requests": 0,
    "payloads": 0,
    "payloads_effort_medium": 0,
    "dead_completions": 0,
    "reasoning_chars": 0,
    "content_chars": 0,
    "tool_call_completions": 0,
    "finish_reasons": {},
}
_tel_lock = _tel_threading.Lock()
_TEL_PATH = WORKING_DIR / "effort_telemetry.json"

_tel_inner_build = _tel_oc.build_chat_payload
assert getattr(_tel_inner_build, "_effort_medium_patched", False), "graft must be installed first"


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

_tel_inner_chat = _tel_ta.ToolAgent._chat_completion


def _tel_record(result):
    import graft_effort as _g

    message = getattr(result, "message", None)
    message = message if isinstance(message, dict) else {}
    finish = str(getattr(result, "finish_reason", "") or "")
    try:
        reasoning = _tel_ta._extract_reasoning_text(message)
    except Exception:  # noqa: BLE001
        reasoning = ""
    try:
        content = _tel_ta._normalize_message_content(message.get("content", ""))
    except Exception:  # noqa: BLE001
        content = ""
    dead = _g._is_dead_completion(result, _tel_ta)
    with _tel_lock:
        TELEMETRY["requests"] += 1
        TELEMETRY["finish_reasons"][finish] = TELEMETRY["finish_reasons"].get(finish, 0) + 1
        TELEMETRY["reasoning_chars"] += len(reasoning or "")
        TELEMETRY["content_chars"] += len(content or "")
        if message.get("tool_calls"):
            TELEMETRY["tool_call_completions"] += 1
        if dead:
            TELEMETRY["dead_completions"] += 1
        n = TELEMETRY["requests"]
        snapshot = json.dumps(TELEMETRY, indent=1)
    if n % 20 == 0:
        try:
            _TEL_PATH.write_text(snapshot, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        print(
            f"[effort-telemetry] req={n} dead={TELEMETRY['dead_completions']} "
            f"reasoning_chars={TELEMETRY['reasoning_chars']} "
            f"effort_payloads={TELEMETRY['payloads_effort_medium']}/{TELEMETRY['payloads']}",
            flush=True,
        )


def _tel_chat(self, messages, *args, **kwargs):
    result = _tel_inner_chat(self, messages, *args, **kwargs)
    try:
        _tel_record(result)
    except Exception:  # noqa: BLE001
        pass
    return result


_tel_ta.ToolAgent._chat_completion = _tel_chat
print("[effort-telemetry] counters installed (baseline: 3.5% dead, 2.62M reasoning chars corpus-wide)")
'''

REPORT_CELL = r'''# ---- effort-smoke final report (grep for EFFORT SMOKE / GAME / TELEMETRY) ----
print("=" * 72)
print("EFFORT SMOKE RESULTS (EFFORT_MEDIUM=1, EFFORT_DEAD_RETRY=0)")
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
req = max(1, tel["requests"])
dead_pct = 100.0 * tel["dead_completions"] / req
print(
    f"TELEMETRY requests={tel['requests']} payloads={tel['payloads']} "
    f"effort_medium_payloads={tel['payloads_effort_medium']}"
)
print(
    f"TELEMETRY dead_completions={tel['dead_completions']} ({dead_pct:.2f}% of requests; "
    f"corpus baseline 3.5% = 122/~3480)"
)
print(
    f"TELEMETRY reasoning_chars_total={tel['reasoning_chars']} "
    f"content_chars_total={tel['content_chars']} "
    f"tool_call_completions={tel['tool_call_completions']}"
)
print("TELEMETRY finish_reasons=", tel["finish_reasons"])

_results = {
    "arm": "effort-smoke",
    "flags": {"EFFORT_MEDIUM": "1", "EFFORT_DEAD_RETRY": "0"},
    "games": games_out,
    "telemetry": tel,
    "dead_pct": dead_pct,
    "comparators": {
        "v12_smoke_scores": {"vc33": 10.71, "tn36": 3.57},
        "corpus_dead_rate_pct": 3.5,
        "corpus_reasoning_chars_dead": 2_620_000,
    },
}
(WORKING_DIR / "effort_smoke_results.json").write_text(
    json.dumps(_results, indent=1), encoding="utf-8"
)
print("wrote", WORKING_DIR / "effort_smoke_results.json")
'''


def main() -> None:
    nb = json.loads(BASE_NB.read_text())
    graft_source = GRAFT_PY.read_text()
    assert "def install() -> str:" in graft_source

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

    # 2. Replace the base smoke hook with the 4-game effort geometry.
    nb["cells"][idx_of(MARK_SMOKE)]["source"] = SMOKE_CELL.splitlines(keepends=True)

    # 3+4. Graft install + telemetry, immediately before the run cell.
    run_idx = idx_of(MARK_RUN)
    graft_cell_text = GRAFT_CELL_TEMPLATE.format(graft_source=graft_source)
    # The graft source must ride the notebook byte-identically: the template
    # embeds it via !r, and Python str repr round-trips by language guarantee.
    assert repr(graft_source) in graft_cell_text
    nb["cells"].insert(run_idx, code_cell(graft_cell_text))
    nb["cells"].insert(run_idx + 1, code_cell(TELEMETRY_CELL))

    # 5. Final report after the run cell.
    nb["cells"].insert(idx_of(MARK_RUN) + 1, code_cell(REPORT_CELL))

    joined = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert MARK_ATTEST in joined
    assert joined.index("GPU misbind") < joined.index(MARK_ATTEST)
    assert joined.index(MARK_ATTEST) < joined.index('SMOKE_GAMES = ["vc33-5430563c"')
    assert joined.index("EFFORT_DEAD_RETRY") < joined.index(MARK_RUN)
    assert joined.index("counters installed") < joined.index(MARK_RUN)
    assert joined.index(MARK_RUN) < joined.index("EFFORT SMOKE RESULTS")
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
