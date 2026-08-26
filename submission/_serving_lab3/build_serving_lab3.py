#!/usr/bin/env python3
"""build_serving_lab3.py — serving-lab-3: TRIMMED throughput RE-MEASURE.

WHY (2026-08-26): live scores declined monotonically across five consecutive
submissions (1.66 08-22 -> 1.50 -> 1.24 -> 1.00 -> 0.99 08-26) spanning
DIFFERENT arms, including two draws of byte-identical code (pack-v22 v1:
1.66 then 0.99). Identical-bytes sigma is ~0.19 (8 duck-base repeats), so a
0.67 same-bytes spread and a 5-long monotone decline are both anomalous
(p~0.01 each). Every run COMPLETED cleanly (parquet 3626-3663 B, no
errorDescription).

HYPOTHESIS UNDER TEST: the scored Kaggle GPU environment degraded
(throughput / contention), starving sessions of tokens — the campaign's
binding currency.

METHOD: boot the EXACT scored serve chain (same anim bundle, same
foysalemonshanto/qwen3-8-27b-fp8-repacked-v1 PyTorch/hf-fp8/1 model, same
wheelhouse, prefix caching ON, KV bf16) and re-run the SAME duck-shaped load
generator that produced the 08-22/08-23 numbers, at conc 8 and conc 28.
Everything else from serving-lab 1/2 (MTP arm, soak, 0.27 upgrade, DFlash2,
quality battery) is STRIPPED — this kernel answers one question.

REFERENCE (arc3-serving-lab v4, 08-22, RTX Pro 6000; reproduced by
arc3-serving-lab2 v1 on 08-22/23):
    conc  8: 1626.0 gen-tok/min/session
    conc 16: 1140.1
    conc 28:  642.6   (lab2 v1 reproduction: 699.1 — 1.09x)

DECISION RULE (pre-registered, printed by the final cell):
    conc28 >= 550 tok/min/session -> THROUGHPUT UNCHANGED (drift is elsewhere:
                                     hidden set, gateway, or our arms)
    conc28 400-550                -> MODERATE degradation
    conc28 <  400                 -> SEVERE degradation explains the decline

ALSO CAPTURED (the environment diff): nvidia-smi GPU name + driver + CUDA +
max/current SM & memory clocks + power cap + throttle reasons, CPU/RAM of the
host, vLLM version, the KV-cache size and max concurrency vLLM reports at
boot, the served argv, and the model config sha256 (boot attestation cell,
kept verbatim).

Budget: boot ~20 min + parser ~2 + conc1 probe ~3 + conc8 ~12 + conc28 ~13
+ final ~1 = ~50 min wall, hard cap 80 min. No games. No submission.

Usage:
  python3 submission/_serving_lab3/build_serving_lab3.py
"""
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
SCAFFOLD = HERE.parent / "_parity_ab" / "scaffold-arc3-duck-v12-with-qwen-3-8-27b.ipynb"
DUCK38_BUILD = HERE.parent / "_duck38_v12" / "build_duck38_v12.py"
LAB1_BUILD = HERE.parent / "_serving_lab" / "build_serving_lab.py"
KERNEL_SLUG = "arc3-serving-lab3"

MARK_IMPORTS = "NOTEBOOK_START_EPOCH = time.time()"
MARK_CONFIG = "# Qwen3.8 / Kaggle input configuration"
MARK_AUDIT = "# Audit the attached inputs that matter for this run."
MARK_SETUP = "TAAF/vLLM setup completed for Qwen3.8"

# 08-22 reference numbers (arc3-serving-lab v4) + 08-22/23 reproduction.
REF_TOKMIN = {"8": 1626.0, "16": 1140.1, "28": 642.6}
REF_REPRO_C28 = 699.1


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD_HEADER = """\
# arc3-serving-lab3 — scored-GPU throughput RE-MEASURE (NO games, NO submission)

**Why:** five consecutive live reads fell monotonically (1.66 → 1.50 → 1.24 →
1.00 → 0.99) across DIFFERENT arms, including two draws of **byte-identical**
pack-v22 code (1.66 then 0.99). Same-bytes sigma is ~0.19, so a 0.67 spread is
~p0.01, and so is a 5-long monotone run. All runs completed cleanly. This
kernel tests the one environmental explanation we can measure directly:
**did the scored GPU pool's throughput degrade?**

**Method:** boot the exact scored serve chain (anim bundle setup, Qwen3.8-27B
FP8 repack, prefix caching ON, KV bf16 — no deviations) and re-run the SAME
duck-shaped load generator (17-25k-token prompts carrying board images,
thinking ON, temp 0.6 / top_p 0.95 / top_k 20, 1-3k gen tokens) at conc 8 and
conc 28. Nothing else runs.

| Phase | What | Budget |
|---|---|---|
| boot | GPU assert → config → audit → bundle setup (boots vLLM) → weight attestation | ~20 min |
| env | nvidia-smi (name/driver/CUDA/clocks/power/throttle), host CPU+RAM, vLLM version, boot-reported KV cache size + max concurrency, served argv | ~1 min |
| 0 | parser round-trip (tool-call + reasoning parse still healthy) | ~2 min |
| 1 | conc-1 single-stream decode probe (isolates raw GPU speed from batching) | ~3 min |
| 2 | conc-8 duck-shaped load, 90 s warmup + 9 min measure | ~12 min |
| 3 | conc-28 duck-shaped load, 90 s warmup + 10 min measure | ~13 min |
| final | comparison table vs 08-22/08-23 + verdict | ~1 min |

**Reference (arc3-serving-lab v4, 08-22, same GPU pool):** conc8 **1626.0**,
conc16 1140.1, conc28 **642.6** gen-tok/min/session; arc3-serving-lab2 v1
reproduced conc28 at **699.1** (1.09x) on 08-22/23.

**DECISION RULE (pre-registered):** conc-28 per-session gen tok/min
**>= 550 → THROUGHPUT UNCHANGED** (the drift is elsewhere: hidden set,
gateway, or our own arms) · **400-550 → MODERATE degradation** ·
**< 400 → SEVERE degradation explains the score decline**.

Results land in `/kaggle/working/serving_lab3_results.json`, rewritten after
every phase. This kernel never plays a game and never touches the competition
rerun path.
"""


CELL_GPU_ASSERT = r'''# ===================== FAIL-FAST GPU ASSERT (before any setup) ==============
# This lab is only meaningful on the RTX Pro 6000 pool (the scored pool). Die
# IMMEDIATELY on a P100/T4 rehoming so the slot costs minutes, not hours.
_gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True)
GPU_NAME = (_gpu_query.stdout or "").strip()
print(f"serving-lab3: GPU = {GPU_NAME!r} (rc={_gpu_query.returncode})")
if _gpu_query.returncode != 0 or not GPU_NAME:
    raise RuntimeError("WRONG-GPU: nvidia-smi failed — no usable GPU. Aborting fast.")
_gpu_upper = GPU_NAME.upper()
if "6000" not in _gpu_upper or "RTX" not in _gpu_upper:
    raise RuntimeError(
        f"WRONG-GPU: expected an RTX Pro 6000, got {GPU_NAME!r}. Aborting fast "
        "so the session dies in minutes (P100/T4 rehoming).")
print("serving-lab3: GPU assert PASS", flush=True)
'''


# @@REFS@@ is substituted by the builder with the 08-22 reference dict.
CELL_ENV = r'''# ================= ENVIRONMENT CAPTURE (the degradation diff) ===============
# Everything here is a candidate explanation for a throughput change that is
# NOT our code: a different silicon SKU, an older/newer driver, a lower power
# cap or clock ceiling, active throttling, a thinner host CPU/RAM allotment,
# a different vLLM build, or less KV cache (=> less real concurrency).
ENVCAP = {}


def _smi_one(field):
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + field,
                        "--format=csv,noheader"], capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        return None
    return out.splitlines()[0].strip()


def _smi_query(fields):
    # Driver renamed clocks_throttle_reasons.* -> clocks_event_reasons.* ; ONE
    # bad field fails the whole batch query, so fall back field-by-field.
    names = [f.strip() for f in fields.split(",") if f.strip()]
    r = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(names),
                        "--format=csv,noheader"], capture_output=True, text=True)
    out = (r.stdout or "").strip()
    if r.returncode == 0 and out:
        vals = [v.strip() for v in out.splitlines()[0].split(",")]
        if len(vals) == len(names):
            return dict(zip(names, vals))
    result = {"_batch_query_failed": (r.stderr or "")[:200]}
    for name in names:
        alt = name.replace("clocks_throttle_reasons", "clocks_event_reasons")
        value = _smi_one(name)
        if value is None and alt != name:
            value = _smi_one(alt)
            if value is not None:
                name = alt
        result[name] = value
    return result


ENVCAP["gpu_static"] = _smi_query(
    "name,driver_version,vbios_version,memory.total,clocks.max.sm,"
    "clocks.max.mem,clocks.max.graphics,power.max_limit,power.limit,"
    "pcie.link.gen.max,pcie.link.width.max,compute_mode,persistence_mode")
ENVCAP["gpu_now"] = _smi_query(
    "clocks.sm,clocks.mem,clocks.graphics,temperature.gpu,power.draw,"
    "utilization.gpu,utilization.memory,memory.used,"
    "clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,"
    "clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown")
_smi_all = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
ENVCAP["nvidia_smi_head"] = [ln for ln in (_smi_all.stdout or "").splitlines()[:12]]

try:
    _cpuinfo = Path("/proc/cpuinfo").read_text(errors="replace")
    _models = [ln.split(":", 1)[1].strip() for ln in _cpuinfo.splitlines()
               if ln.lower().startswith("model name")]
    ENVCAP["cpu_model"] = _models[0] if _models else None
    ENVCAP["cpu_logical_count"] = len(_models) or os.cpu_count()
except Exception as _exc:
    ENVCAP["cpu_error"] = repr(_exc)[:200]
ENVCAP["os_cpu_count"] = os.cpu_count()
try:
    _mem = dict(
        (ln.split(":", 1)[0], ln.split(":", 1)[1].strip())
        for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln)
    ENVCAP["mem_total"] = _mem.get("MemTotal")
    ENVCAP["mem_available"] = _mem.get("MemAvailable")
except Exception as _exc:
    ENVCAP["mem_error"] = repr(_exc)[:200]
try:
    ENVCAP["loadavg"] = Path("/proc/loadavg").read_text().strip()
    ENVCAP["uptime_s"] = float(Path("/proc/uptime").read_text().split()[0])
except Exception:
    pass

# ---- vLLM identity + what it reported at boot -------------------------------
def _find_boot_log():
    cands = []
    for pat in ("*.log", "**/*.log"):
        for p in WORKING_DIR.glob(pat):
            try:
                if p.stat().st_size > 2000:
                    cands.append(p)
            except OSError:
                pass
    best, best_score = None, -1
    for p in cands:
        try:
            head = p.read_text(errors="replace")[:2_000_000]
        except Exception:
            continue
        score = head.count("vLLM API server version") * 10 + head.count("KV cache")
        if score > best_score:
            best, best_score = p, score
    return best if best_score > 0 else None


_boot_log = _find_boot_log()
ENVCAP["boot_log_path"] = str(_boot_log) if _boot_log else None
_boot_txt = ""
if _boot_log is not None:
    try:
        _boot_txt = _boot_log.read_text(errors="replace")
    except Exception as _exc:
        ENVCAP["boot_log_error"] = repr(_exc)[:200]

_PATTERNS = {
    "vllm_api_server_version": "vLLM API server version",
    "kv_cache_size": "GPU KV cache size",
    "max_concurrency": "Maximum concurrency for",
    "kv_cache_memory": "Available KV cache memory",
    "model_weights_mem": "Model loading took",
    "graph_capture": "graph capturing finished",
    "torch_compile": "torch.compile takes",
    "args_namespace": "args: Namespace",
}
ENVCAP["boot_lines"] = {}
for _key, _needle in _PATTERNS.items():
    _hits = [ln.strip()[:600] for ln in _boot_txt.splitlines() if _needle in ln]
    if _hits:
        ENVCAP["boot_lines"][_key] = _hits[:3]

_ver = subprocess.run(
    [sys.executable, "-c",
     "import vllm, torch, sys; "
     "print(vllm.__version__); print(torch.__version__); "
     "print(torch.version.cuda); print(torch.cuda.get_device_name(0)); "
     "print(torch.cuda.get_device_capability(0))"],
    capture_output=True, text=True, env={**os.environ, "VLLM_NO_USAGE_STATS": "1"})
_vlines = (_ver.stdout or "").strip().splitlines()
if len(_vlines) >= 5:
    ENVCAP["vllm_version"] = _vlines[0]
    ENVCAP["torch_version"] = _vlines[1]
    ENVCAP["torch_cuda"] = _vlines[2]
    ENVCAP["torch_device_name"] = _vlines[3]
    ENVCAP["torch_sm"] = _vlines[4]
else:
    ENVCAP["vllm_version_probe_error"] = ((_ver.stderr or "")[-400:] or "no output")

# served argv (proves the flags actually in force, incl. prefix caching)
_ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
ENVCAP["vllm_argv"] = [ln.strip()[:900] for ln in (_ps.stdout or "").splitlines()
                       if "vllm" in ln and ("api_server" in ln or "entrypoints" in ln)][:2]

print(json.dumps(ENVCAP, indent=2, default=str)[:6000], flush=True)
RESULTS["meta"]["environment"] = ENVCAP
RESULTS["meta"]["reference_tokmin_session_0822"] = @@REFS@@
RESULTS["meta"]["reference_conc28_reproduction_0823"] = @@REPRO@@
save_results()
'''


CELL_MEASURE = r'''# ============ THE MEASUREMENT — parser + conc 1 / 8 / 28 duck load ==========
# Identical load generator, identical flags, identical model as the 08-22 run
# that measured 1626.0 @conc8 and 642.6 @conc28 (and 699.1 @conc28 on 08-23).
try:
    parser_roundtrip("scored_stack")
except Exception:
    traceback.print_exc()
    RESULTS["phases"].setdefault("parser_roundtrip_scored_stack", {"error": "see traceback"})
    save_results()

for _name, _conc, _warm, _meas in [("probe_conc1", 1, 30, 150),
                                   ("remeasure_conc8", 8, 90, 540),
                                   ("remeasure_conc28", 28, 90, 600)]:
    try:
        run_load_phase(_name, _conc, _warm, _meas)
    except Exception:
        traceback.print_exc()
        RESULTS["phases"].setdefault(_name, {"error": "see traceback"})
        save_results()
print("serving-lab3: measurement phases done at", round(elapsed_min(), 1), "min", flush=True)
save_results()
'''


CELL_FINAL = r'''# ============== FINAL — comparison table + pre-registered verdict ===========
def _cell(value, width=13):
    return str(value if value is not None else "-").rjust(width)


_ref = RESULTS["meta"]["reference_tokmin_session_0822"]
_repro = RESULTS["meta"]["reference_conc28_reproduction_0823"]
_now = {}
for _key, _phase in (("8", "remeasure_conc8"), ("28", "remeasure_conc28")):
    _now[_key] = per_session_of(RESULTS["phases"].get(_phase) or {})

print("\n" + "=" * 92)
print("serving-lab3 — SCORED-GPU THROUGHPUT RE-MEASURE (gen tok/min/session)")
print("=" * 92)
print("  conc |     08-22 ref |  08-23 repro |    2026-08-26 |   ratio vs 08-22")
for _key in ("8", "28"):
    _r = _ref.get(_key)
    _p = _repro if _key == "28" else None
    _n = _now.get(_key)
    _ratio = f"{_n / _r:.2f}x" if (_n and _r) else "-"
    print(f"  {_key:>4} | {_cell(_r)} | {_cell(_p, 12)} | {_cell(_n)} |   {_ratio}")

for _key, _phase in (("1", "probe_conc1"), ("8", "remeasure_conc8"), ("28", "remeasure_conc28")):
    _ph = RESULTS["phases"].get(_phase) or {}
    if not _ph or _ph.get("skipped") or _ph.get("error"):
        print(f"\n  conc {_key}: NO DATA ({_ph.get('skipped') or _ph.get('error') or 'phase missing'})")
        continue
    print(f"\n  conc {_key} detail: {per_session_of(_ph)} tok/min/session | "
          f"aggregate {_ph.get('gen_tok_s_aggregate_metric') or _ph.get('gen_tok_s_aggregate_usage')} tok/s | "
          f"{_ph.get('requests_in_window')} reqs / {_ph.get('errors')} errs | "
          f"prompt mean {_ph.get('prompt_tokens_mean')} "
          f"({_ph.get('prompt_tokens_min')}-{_ph.get('prompt_tokens_max')}) | "
          f"gen mean {_ph.get('completion_tokens_mean')} | "
          f"p50 lat {_ph.get('p50_latency_s')}s | mean lat {_ph.get('mean_latency_s')}s")

_rate28 = _now.get("28")
if _rate28 is None:
    _verdict = "NO-DATA — conc-28 phase did not produce a rate; hypothesis UNTESTED"
elif _rate28 >= 550:
    _verdict = (f"THROUGHPUT UNCHANGED ({_rate28} >= 550 tok/min/session at conc 28) — "
                "the GPU-degradation hypothesis is REFUTED; the score drift is "
                "elsewhere (hidden set, gateway, or our own arms)")
elif _rate28 >= 400:
    _verdict = (f"MODERATE DEGRADATION ({_rate28} in 400-550 tok/min/session at conc 28) — "
                "partial token starvation; contributes to but does not fully "
                "explain the 5-draw decline")
else:
    _verdict = (f"SEVERE DEGRADATION ({_rate28} < 400 tok/min/session at conc 28) — "
                "the scored GPU pool got slower; token starvation explains the "
                "score decline")
RESULTS["verdicts"]["throughput_2026_08_26"] = _verdict
RESULTS["verdicts"]["conc8_tokmin_session"] = _now.get("8")
RESULTS["verdicts"]["conc28_tokmin_session"] = _rate28
RESULTS["verdicts"]["conc1_tokmin_session"] = per_session_of(
    RESULTS["phases"].get("probe_conc1") or {})
RESULTS["verdicts"]["decision_rule"] = (
    ">=550 UNCHANGED | 400-550 MODERATE | <400 SEVERE (conc 28, gen tok/min/session)")
RESULTS["meta"]["finished_utc"] = datetime.utcnow().isoformat() + "Z"
RESULTS["meta"]["total_elapsed_min"] = round(elapsed_min(), 1)

_env = RESULTS["meta"].get("environment", {})
print("\n" + "-" * 92)
print("ENVIRONMENT (compare against the 08-22 run):")
print("  gpu       :", _env.get("gpu_static", {}))
print("  gpu now   :", _env.get("gpu_now", {}))
print("  host      :", _env.get("cpu_model"), "|", _env.get("cpu_logical_count"),
      "vcpu |", _env.get("mem_total"), "| load", _env.get("loadavg"))
print("  vllm      :", _env.get("vllm_version"), "| torch", _env.get("torch_version"),
      "| cuda", _env.get("torch_cuda"), "| sm", _env.get("torch_sm"))
for _k, _v in (_env.get("boot_lines") or {}).items():
    print(f"  boot.{_k}: {_v[0] if _v else ''}")
print("-" * 92)
print("DECISION RULE:", RESULTS["verdicts"]["decision_rule"])
print("VERDICT      :", _verdict)
print("=" * 92, flush=True)
save_results()
print("results ->", RESULTS_PATH, RESULTS_PATH.stat().st_size, "bytes", flush=True)
'''


def _code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def _lab_library() -> str:
    """Reuse arc3-serving-lab's library VERBATIM (same load generator = the
    only way the numbers are comparable), renamed for lab3 and with the MTP
    machinery left inert."""
    lab1 = _load_module(LAB1_BUILD, "build_serving_lab")
    lib = lab1.CELL_LAB_LIB

    def sub(old, new, count=1):
        nonlocal lib
        assert lib.count(old) >= count, f"missing in lab1 library: {old!r}"
        lib = lib.replace(old, new)

    sub('"kernel": "arc3-serving-lab"', '"kernel": "arc3-serving-lab3"')
    sub('RESULTS_PATH = WORKING_DIR / "serving_lab_results.json"',
        'RESULTS_PATH = WORKING_DIR / "serving_lab3_results.json"')
    sub("LAB_HARD_CAP_MIN = 150.0   # skip any phase starting after this",
        "LAB_HARD_CAP_MIN = 80.0    # trimmed lab: skip any phase starting after this")
    sub('Q1_RULE = (">=400 tok/min/session at conc 28 = throughput hypothesis DEAD; "\n'
        '           "<300 = throughput is the live discount")',
        'Q1_RULE = (">=550 tok/min/session at conc 28 = throughput UNCHANGED; "\n'
        '           "400-550 = MODERATE degradation; <400 = SEVERE degradation")')
    # Strip the (unused) MTP speculative-decode flag definitions entirely: this
    # kernel must contain NO deviation from the scored serve chain, not even a
    # dead constant that could be mistaken for one.
    sub('MTP3_FLAGS = ["--speculative-config",\n'
        '              \'{"method": "mtp", "num_speculative_tokens": 3}\',\n'
        '              "--no-enable-prefix-caching"]\n'
        'MTP2_FLAGS = ["--speculative-config",\n'
        '              \'{"method": "mtp", "num_speculative_tokens": 2}\',\n'
        '              "--no-enable-prefix-caching"]\n', "")
    sub('        "mtp3_flags": BASE_SERVE_FLAGS + MTP3_FLAGS,\n'
        '        "mtp2_flags": BASE_SERVE_FLAGS + MTP2_FLAGS,\n', "")
    # cosmetic: every log line self-identifies as lab3
    lib = lib.replace("serving-lab:", "serving-lab3:")
    lib = lib.replace(
        "#   Q1 live throughput at conc 8/16/28 on the EXACT scored serve config.\n"
        "#   Q2 MTP speculative decoding: nst=3 (+2 if time), acceptance, parser, soak.",
        "#   ONE question: is the scored GPU pool still delivering the 08-22\n"
        "#   throughput (1626 @conc8 / 642.6 @conc28 gen-tok/min/session)?\n"
        "#   The load generator below is byte-identical to the 08-22 run.")
    assert "serving-lab3:" in lib and "serving_lab3_results.json" in lib
    return lib


def main() -> None:
    scaffold = json.loads(SCAFFOLD.read_text())

    def find_cell(marker: str) -> str:
        hits = [c for c in scaffold["cells"]
                if c["cell_type"] == "code" and marker in "".join(c["source"])]
        assert len(hits) == 1, (marker, len(hits))
        return "".join(hits[0]["source"])

    imports_cell = find_cell(MARK_IMPORTS)
    config_cell = find_cell(MARK_CONFIG)
    audit_cell = find_cell(MARK_AUDIT)
    setup_cell = find_cell(MARK_SETUP)
    attest_cell = _load_module(DUCK38_BUILD, "build_duck38_v12").ATTEST_CELL

    env_cell = (CELL_ENV
                .replace("@@REFS@@", repr(REF_TOKMIN))
                .replace("@@REPRO@@", repr(REF_REPRO_C28)))
    assert "@@" not in env_cell

    cells = [
        _md_cell(MD_HEADER),
        _code_cell(imports_cell),
        _code_cell(CELL_GPU_ASSERT),   # fail-fast: dies on P100 in minutes
        _code_cell(config_cell),
        _code_cell(audit_cell),
        _code_cell(setup_cell),        # boots vLLM with the exact scored flags
        _code_cell(attest_cell),       # doctrine v2: weights verified + config sha
        _code_cell(_lab_library()),
        _code_cell(env_cell),
        _code_cell(CELL_MEASURE),
        _code_cell(CELL_FINAL),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:
            raise SystemExit(f"cell {i} failed ast.parse: {exc}") from exc

    joined = "\n".join("".join(c["source"]) for c in notebook["cells"])
    # Ordering + content invariants (hard-won from labs 1 and 2):
    assert joined.index("WRONG-GPU") < joined.index(MARK_CONFIG), "GPU assert must precede config"
    assert joined.index("attest: OK") < joined.index("SERVING LAB LIBRARY")
    assert joined.index("SERVING LAB LIBRARY") < joined.index("ENVIRONMENT CAPTURE")
    assert joined.index("ENVIRONMENT CAPTURE") < joined.index("THE MEASUREMENT")
    # The scored serve chain must be untouched: prefix caching stays ON (the
    # bundle's own flag), KV stays bf16, and NO speculative config is passed.
    assert "--no-enable-prefix-caching" not in joined, "prefix caching must stay ON"
    assert '"--kv-cache-dtype"' not in joined, "KV must stay bf16"
    assert "start_server(MTP" not in joined and "MTP3_FLAGS," not in joined
    assert "run_load_phase(_name, _conc, _warm, _meas)" in joined
    # No games, no submission machinery.
    for banned in ("submission.parquet", "GameAgent", "scorecard"):
        assert banned not in joined, f"banned token in lab kernel: {banned}"

    nb_path = HERE / f"{KERNEL_SLUG}.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1) + "\n")

    (HERE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ahmedmobasher86/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaRtxPro6000",
        "keywords": ["gpu"],
        "dataset_sources": [
            "driessmit1/arc3-vllm-h100-wheelhouse-v3",
            "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
        ],
        "kernel_sources": [],
        # THE RTX Pro 6000 GATE (08-22 push lesson): machine_shape and
        # --accelerator alone still bind P100; the competition source is what
        # actually admits the kernel to the scored GPU pool.
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
    }, indent=2) + "\n")

    import hashlib
    code = "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code")
    print("built", KERNEL_SLUG, "code-cell sha256",
          hashlib.sha256(code.encode()).hexdigest())
    print("cells:", len(notebook["cells"]), "| notebook:", nb_path)


if __name__ == "__main__":
    main()
