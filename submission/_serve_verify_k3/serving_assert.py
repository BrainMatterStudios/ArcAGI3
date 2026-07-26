"""serving_assert.py — in-submission proof that the adapter is actually served.

Born from the 1.26=base retraction (sub 54554985: LoRA never served, nobody
noticed) — this module makes that failure mode LOUD. The scored/debut kernel
inlines it and runs the chain around its in-kernel merge:

    1. base_health_nll(model, rows)         base sane? (<6.0 nats, the run-1-5 gate)
    2. assert_adapter_attached(pmodel)      PEFT actually wrapped the model
    3. sample_merge_checks(pmodel)          cache (w_pre, B@A*alpha/r) BEFORE merge
    4. assert_merge_delta(merged, checks)   post-merge delta == expectation
    5. assert_nll_improves(base, merged)    merged target-NLL beats base by margin
    6. assert_endpoint_serves(url, path)    vLLM up AND --model arg == merged dir

Policy: STRICT (gate kernels) raises on first failure. SAFE (scored kernels)
never raises — it prints an unmissable marker block, writes
``serving_assert_FAILED.json``, and returns False so the caller can fall back
to the base model EXPLICITLY (a farmed base draw beats a dead slot).

Pure-torch; unit-tested on CPU with toy modules (test_serving_assert.py).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import torch

FAIL_MARKER = "!" * 20 + " SERVING-ASSERT FAILED " + "!" * 20
PASS_MARKER = "=" * 20 + " SERVING-ASSERT PASS " + "=" * 20

BASE_NLL_MAX = 6.0          # nats; corrupted base measured 14.7 (> uniform 12.4)
MERGE_REL_ERR_MAX = 0.05    # |delta - B@A*s| / |B@A*s|
MIN_LORA_MODULES = 400      # run-8 adapter attaches 400+ target Linears
MIN_NLL_GAIN = 0.02         # merged must beat base target-NLL by >=2% (run 8: 12.5%)


@dataclass
class AssertReport:
    strict: bool
    out_path: str = "/kaggle/working/serving_assert_FAILED.json"
    checks: list = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: dict) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), **detail})
        line = f"[serving-assert] {name}: {'PASS' if ok else 'FAIL'} {detail}"
        print(line, flush=True)
        if not ok:
            if self.strict:
                raise AssertionError(line)
            for _ in range(3):
                print(FAIL_MARKER, flush=True)
            print(f"[serving-assert] check {name!r} failed — caller must fall "
                  f"back to BASE explicitly. Detail: {detail}", flush=True)
            json.dump(self.checks, open(self.out_path, "w"), indent=2)
        return ok

    @property
    def all_ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def finish(self, min_checks: int = 5) -> bool:
        """PASS requires every check green AND at least min_checks ran.

        all([]) is True — without the count floor, a chain that crashed before
        its first record() would report PASS (audit finding, 2026-07-26).
        """
        if len(self.checks) < min_checks:
            self.record("chain_completeness", False,
                        {"checks_ran": len(self.checks), "min": min_checks})
            return False
        if self.all_ok:
            print(PASS_MARKER, flush=True)
            print(f"[serving-assert] {len(self.checks)} checks passed "
                  f"@ {time.strftime('%H:%M:%S')}", flush=True)
        return self.all_ok


def target_nll(model, rows, device=None) -> float:
    """Mean per-target-token NLL over prepared val rows.

    Each row: {"input_ids": [...], "labels": [...]} with -100 on non-target
    positions (the trainer's val format). Kept tiny (2 rows) — this runs twice
    inside the scored kernel's startup budget.
    """
    losses, total = [], 0
    model.train(False)
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor([row["input_ids"]])
            labels = torch.tensor([row["labels"]])
            if device is not None:
                ids, labels = ids.to(device), labels.to(device)
            out = model(input_ids=ids, labels=labels)
            n = int((labels != -100).sum())
            losses.append(float(out.loss) * n)
            total += n
    return sum(losses) / max(total, 1)


def check_base_health(rep: AssertReport, model, rows, device=None) -> float:
    try:
        nll = target_nll(model, rows, device)
    except Exception as e:  # noqa: BLE001 — SAFE mode must record, not crash
        rep.record("base_health_nll", False,
                   {"error": f"{type(e).__name__}: {e}"})
        return float("nan")
    rep.record("base_health_nll", math.isfinite(nll) and nll < BASE_NLL_MAX,
               {"nll": round(nll, 4), "max": BASE_NLL_MAX})
    return nll


def assert_adapter_attached(rep: AssertReport, pmodel,
                            min_modules: int = MIN_LORA_MODULES):
    mods = [(n, m) for n, m in pmodel.named_modules()
            if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]
    rep.record("adapter_attached", len(mods) >= min_modules,
               {"lora_modules": len(mods), "min": min_modules})
    return mods


def sample_merge_checks(lora_mods, scaling: float, k: int = 3, seed: int = 0):
    """Cache (name, w_pre, expected_delta) for k random LoRA modules PRE-merge."""
    import random
    rng = random.Random(seed)
    picked = rng.sample(lora_mods, min(k, len(lora_mods)))
    checks = []
    for n, m in picked:
        ba = (m.lora_B["default"].weight @ m.lora_A["default"].weight) * scaling
        checks.append((n, m.base_layer.weight.detach().clone(),
                       ba.detach().clone()))
    return checks


PEFT_PREFIX = "base_model.model."


def _resolve_merged_module(named: dict, name: str):
    """peft's merge_and_unload() strips the 'base_model.model.' prefix that
    pre-merge module names carry — a raw lookup KeyErrors on real peft 0.19.x
    (audit finding, 2026-07-26; toy tests missed it because they preserved
    names). Try raw, then stripped, then prefixed."""
    for candidate in (name,
                      name.removeprefix(PEFT_PREFIX),
                      PEFT_PREFIX + name):
        if candidate in named:
            return candidate, named[candidate]
    return None, None


def assert_merge_delta(rep: AssertReport, merged, checks,
                       rel_err_max: float = MERGE_REL_ERR_MAX) -> bool:
    try:
        named = dict(merged.named_modules())
        results, ok = [], True
        for (n, w_pre, expected) in checks:
            resolved, mod = _resolve_merged_module(named, n)
            if mod is None:
                results.append({"module": n, "error": "not found post-merge"})
                ok = False
                continue
            w_post = mod.weight.detach()
            delta = (w_post - w_pre).float()
            exp = expected.float()
            rel = float((delta - exp).norm() / (exp.norm() + 1e-9))
            good = rel < rel_err_max and float(exp.norm()) > 0
            results.append({"module": resolved,
                            "delta_norm": round(float(exp.norm()), 5),
                            "rel_err": round(rel, 5)})
            ok = ok and good
        return rep.record("merge_weight_delta", ok,
                          {"rel_err_max": rel_err_max, "modules": results})
    except Exception as e:  # noqa: BLE001 — SAFE mode must record, not crash
        return rep.record("merge_weight_delta", False,
                          {"error": f"{type(e).__name__}: {e}"})


def assert_nll_improves(rep: AssertReport, base_nll: float, merged, rows,
                        device=None, min_gain: float = MIN_NLL_GAIN) -> bool:
    try:
        m_nll = target_nll(merged, rows, device)
    except Exception as e:  # noqa: BLE001 — SAFE mode must record, not crash
        return rep.record("merged_nll_improves", False,
                          {"error": f"{type(e).__name__}: {e}"})
    if not math.isfinite(base_nll):
        return rep.record("merged_nll_improves", False,
                          {"error": "base_nll not finite",
                           "base_nll": str(base_nll)})
    gain = (base_nll - m_nll) / max(base_nll, 1e-9)
    return rep.record("merged_nll_improves", gain >= min_gain,
                      {"base_nll": round(base_nll, 4),
                       "merged_nll": round(m_nll, 4),
                       "gain": round(gain, 4), "min_gain": min_gain})


def assert_server_model_arg(rep: AssertReport, server_cmd: list,
                            merged_dir: str) -> bool:
    """The vLLM launch command's --model argument must be the merged dir.

    This is the identity check that actually works: the duck's serve line sets
    --served-model-name to a FIXED name (e.g. 'vrfai/Qwen3.6-27B-FP8'), so a
    /models id comparison against the merged dir is vacuous — it would always
    fail and, in SAFE mode, auto-fall back to base at the debut (audit
    finding, 2026-07-26). The --model arg is what vLLM loads; assert THAT.
    """
    import os
    try:
        cmd = [str(c) for c in server_cmd]
        model_arg = None
        for i, tok in enumerate(cmd):
            if tok == "--model" and i + 1 < len(cmd):
                model_arg = cmd[i + 1]
            elif tok.startswith("--model="):
                model_arg = tok.split("=", 1)[1]
        ok = (model_arg is not None
              and os.path.realpath(model_arg) == os.path.realpath(merged_dir))
        return rep.record("server_model_arg", ok,
                          {"model_arg": model_arg, "expected": merged_dir})
    except Exception as e:  # noqa: BLE001 — SAFE mode must record, not crash
        return rep.record("server_model_arg", False,
                          {"error": f"{type(e).__name__}: {e}",
                           "expected": merged_dir})


def assert_endpoint_serves(rep: AssertReport, base_url: str,
                           expected_served_name: str = "",
                           timeout: float = 30.0, _opener=None) -> bool:
    """Liveness: vLLM answers /models; optional exact served-name equality.

    Pass expected_served_name ONLY if the kernel mints a versioned name (e.g.
    'run8/checkpoint-8') and propagates it identically to --served-model-name
    and the analyzer env. Never pass a filesystem path here — the served name
    is whatever --served-model-name says, not the model path.
    """
    import urllib.request
    opener = _opener or urllib.request.urlopen
    try:
        with opener(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            data = json.loads(r.read().decode())
        ids = [m.get("id", "") for m in data.get("data", [])]
        ok = bool(ids) and (not expected_served_name
                            or expected_served_name in ids)
        return rep.record("endpoint_alive", ok,
                          {"served_ids": ids,
                           "expected_name": expected_served_name or "(any)"})
    except Exception as e:  # noqa: BLE001 — any transport failure = check fails
        return rep.record("endpoint_alive", False,
                          {"error": f"{type(e).__name__}: {e}"})
