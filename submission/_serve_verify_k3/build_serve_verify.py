"""build_serve_verify.py — emit serve-verify-k3.ipynb + kernel-metadata.json.

SERVING-VERIFICATION GATE (RUNBOOK §5) for the K3-distillation checkpoint ladder.
One RTX-6000 commit: in-kernel LoRA merge -> weight-level delta assert -> serve the
MERGED model with the duck's EXACT vLLM line -> 3 fixed greedy probes (2 val rows with
images + 1 plain text) -> same probes against the BASE FP8 snapshot -> verdict JSON.

Design constraints this encodes (2026-07-26):
  * Kaggle output cap 20GB + offline RTX6000 => the merged 55GB bf16 model CANNOT be
    published from a kernel. It lives in /tmp scratch; only logs/JSON are output.
    The winning checkpoint gets published later via a local pipeline.
  * Merge inherits the trainer's exact base-weight semantics by replicating its load
    path verbatim (AutoModelForImageTextToText bf16 + strip_quantization_runtime).
  * vLLM provisioning copied from the duck's proven setup_commands (wheelhouse
    driessmit1/arc3-vllm-h100-wheelhouse-v3, pip --no-index --target).
  * base-vs-merged output diff is REPORTED but not hard-asserted (FP8-vs-bf16
    precision confound); the hard asserts are weight-level delta math + merged
    model serves + emits well-formed tool_calls on the val probes.

Run: python3 submission/_serve_verify_k3/build_serve_verify.py
Push: cd submission/_serve_verify_k3 && kaggle kernels push -p . --accelerator NvidiaRtxPro6000
Env knobs: GATE_CKPT (default checkpoint-16), GATE_MAX_TOKENS (default 3072).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MD_HEADER = """\
# ARC-3 serve-verify K3 — merge ladder checkpoint into Qwen3.6-27B and prove it serves

July's fine-tune died because a LoRA never reached the served model. This gate proves, in one
offline RTX-6000 commit, that a ladder checkpoint (default `checkpoint-16`, the OOD-early-peak arm):
1. **merges** into the base with exactly the trainer's load semantics (delta == B@A x alpha/r, checked),
2. **serves** under the duck's exact vLLM line (bf16; FP8 requant deferred to the publish step),
3. **behaves**: greedy val-prompt probes emit well-formed `<tool_call>` python calls,
4. differs from the BASE FP8 snapshot on the same probes (reported; precision-confounded).
Output = logs + `gate_result.json` only; the merged model stays in scratch by design (20GB cap).
"""

C_GUARD = """\
import glob, json, os, shutil, signal, subprocess, sys, time, urllib.request
deps = sorted(glob.glob("/kaggle/input/**/deps", recursive=True))
if deps: sys.path.insert(0, deps[0])
print("deps on path:", deps[:1])
import torch
DEV = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
RUN = "RTX PRO 6000" in DEV.upper() or os.environ.get("GATE_FORCE") == "1"
print(f"device: {DEV} | RUN={RUN}")
import transformers, peft
print("transformers", transformers.__version__, "| peft", peft.__version__)
GATE_CKPT = os.environ.get("GATE_CKPT", "checkpoint-16")
GATE_MAX_TOKENS = int(os.environ.get("GATE_MAX_TOKENS", 6144))  # thinking is served-on; leave room before the tool_call
"""

C_INPUTS = """\
CORPUS = os.path.dirname(sorted(glob.glob("/kaggle/input/**/train.jsonl", recursive=True))[0])
sys.path.insert(0, CORPUS)
from sft_common import strip_quantization_runtime
MODEL = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
             if "tokenizer_bundle" not in p and json.load(open(p)).get("model_type") == "qwen3_5")
CKPT = next(p for p in sorted(glob.glob("/kaggle/input/**/sft_out/checkpoint-*", recursive=True))
            if p.endswith(GATE_CKPT))
WHEELHOUSE = os.path.dirname(sorted(glob.glob("/kaggle/input/**/requirements.lock", recursive=True))[0])
print("model:", MODEL, "\\nckpt:", CKPT, "\\nwheelhouse:", WHEELHOUSE, "\\ncorpus:", CORPUS)
TOOLS = json.load(open(os.path.join(CORPUS, "tools.json")))
val_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "val.jsonl"))]
# fixed probes: first val row WITH an image, first val row WITHOUT, chosen deterministically
def has_image(r):
    return any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"])
               for m in r["messages"])
PROBE_IMG = next(r for r in val_rows if has_image(r))
# corpus is fully multimodal (all 43 val rows carry images) -> probe B is just a distinct second row,
# text-only if one ever exists
PROBE_TXT = next((r for r in val_rows if not has_image(r)),
                 next(r for r in val_rows if r is not PROBE_IMG))
print("probe A (image):", len(PROBE_IMG["messages"]), "msgs | probe B:", len(PROBE_TXT["messages"]), "msgs")
MERGED = "/tmp/merged_" + GATE_CKPT
"""

C_MERGE = """\
if RUN:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
    for a in ("quantization_config", "_pre_quantization_dtype"):
        if hasattr(model.config, a):
            try: setattr(model.config, a, None)
            except Exception: pass
    model.is_quantized = False
    if hasattr(model, "hf_quantizer"): model.hf_quantizer = None
    print("strip:", strip_quantization_runtime(model), f"| load {time.time()-t0:.0f}s")

    # cache 3 targeted base weights for the post-merge delta assert
    acfg = json.load(open(os.path.join(CKPT, "adapter_config.json")))
    scaling = acfg["lora_alpha"] / acfg["r"]
    from peft import PeftModel
    pmodel = PeftModel.from_pretrained(model, CKPT, is_trainable=False)
    lora_mods = [(n, m) for n, m in pmodel.named_modules()
                 if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})]
    assert len(lora_mods) > 400, f"adapter did not attach: {len(lora_mods)} lora modules"
    import random; random.seed(0)
    checks = []
    for n, m in random.sample(lora_mods, 3):
        BA = (m.lora_B["default"].weight @ m.lora_A["default"].weight) * scaling
        checks.append((n, m.base_layer.weight.detach().clone(), BA.detach().clone()))
    merged = pmodel.merge_and_unload()
    ok = []
    for (n, w_pre, BA) in checks:
        w_post = dict(merged.named_modules())[n].weight.detach()
        delta = (w_post - w_pre).float(); exp = BA.float()
        rel = (delta - exp).norm() / (exp.norm() + 1e-9)
        ok.append({"module": n, "delta_norm": float(exp.norm()), "rel_err": float(rel)})
        print(f"[delta] {n}: |BA|={exp.norm():.4f} rel_err={rel:.4f}")
    assert all(c["rel_err"] < 0.05 and c["delta_norm"] > 0 for c in ok), ok
    print("WEIGHT-LEVEL MERGE ASSERT: PASS")

    merged.config.torch_dtype = torch.bfloat16
    merged.config.use_cache = True
    t0 = time.time()
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
    try:
        proc = AutoProcessor.from_pretrained(MODEL)
    except Exception as e:
        print("processor from snapshot failed:", e)
        proc = AutoProcessor.from_pretrained(os.path.join(CORPUS, "tokenizer_bundle"))
    proc.save_pretrained(MERGED)
    for extra in ("chat_template.jinja", "chat_template.json"):
        src = os.path.join(MODEL, extra)
        if os.path.exists(src) and not os.path.exists(os.path.join(MERGED, extra)):
            shutil.copy(src, MERGED)
    cfg = json.load(open(os.path.join(MERGED, "config.json")))
    for k in ("quantization_config", "_pre_quantization_dtype", "compression_config"):
        cfg.pop(k, None)
    json.dump(cfg, open(os.path.join(MERGED, "config.json"), "w"), indent=2)
    print(f"saved {MERGED} in {time.time()-t0:.0f}s:", sorted(os.listdir(MERGED))[:8], "...")
    del merged, pmodel, model, checks
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
                         capture_output=True, text=True).stdout)
"""

C_VLLM = """\
if RUN:
    SITE = "/kaggle/working/vllm-site-packages"
    if not os.path.exists(os.path.join(SITE, "vllm")):
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "--find-links", WHEELHOUSE, "--requirement",
                        os.path.join(WHEELHOUSE, "requirements.lock"), "--target", SITE,
                        "--upgrade", "--ignore-installed", "--only-binary", ":all:",
                        "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts"],
                       check=True, capture_output=True)
    ENV = dict(os.environ, PYTHONPATH=SITE, USE_TF="0", TRANSFORMERS_NO_TF="1",
               TRANSFORMERS_NO_TORCHVISION="1", VLLM_NO_USAGE_STATS="1")
    v = subprocess.run([sys.executable, "-c", "import vllm; print(vllm.__version__)"],
                       env=ENV, capture_output=True, text=True)
    print("vllm:", v.stdout.strip(), v.stderr.strip()[-200:])
    assert v.returncode == 0, "vLLM import failed from wheelhouse site-packages"
"""

C_SERVE_FN = """\
BASE_URL = "http://127.0.0.1:1234/v1"

def req(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def start_server(model_path, log_path):
    # the duck's EXACT serve line (setup_commands.json), model path swapped
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model_path, "--served-model-name", "gate/model",
           "--host", "127.0.0.1", "--port", "1234", "--tensor-parallel-size", "1",
           "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
           "--generation-config", "vllm", "--enable-prefix-caching",
           "--default-chat-template-kwargs", '{"preserve_thinking": true}',
           "--reasoning-parser", "qwen3", "--max-model-len", "65536"]
    lh = open(log_path, "w")
    p = subprocess.Popen(cmd, env=ENV, stdout=lh, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if p.poll() is not None:
            print(open(log_path).read()[-4000:])
            raise RuntimeError(f"vLLM died rc={p.returncode} for {model_path}")
        try:
            req(BASE_URL + "/models", timeout=5); print("server ready:", model_path); return p
        except Exception:
            time.sleep(5)
    print(open(log_path).read()[-4000:])
    raise TimeoutError("vLLM never became ready")

def stop_server(p):
    p.send_signal(signal.SIGTERM)
    try: p.wait(60)
    except subprocess.TimeoutExpired: p.kill(); p.wait(30)
    for _ in range(36):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        if used and int(used.split()[0]) < 8000: break
        time.sleep(5)
    print("gpu after stop:", used, "MiB")

def probe(tag):
    out = {}
    for name, row in (("val_img", PROBE_IMG), ("val_txt", PROBE_TXT)):
        r = req(BASE_URL + "/chat/completions",
                {"model": "gate/model", "messages": row["messages"], "tools": TOOLS,
                 "temperature": 0.0, "max_tokens": GATE_MAX_TOKENS}, timeout=900)
        msg = r["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        wf = bool(tcs) and all(isinstance(json.loads(t["function"]["arguments"])
                                          if isinstance(t["function"]["arguments"], str)
                                          else t["function"]["arguments"], dict) for t in tcs)
        out[name] = {"tool_calls": [t["function"]["name"] for t in tcs], "well_formed": wf,
                     "content": (msg.get("content") or "")[:400],
                     "reasoning": (msg.get("reasoning_content") or "")[:200],
                     "raw_args": [str(t["function"]["arguments"])[:300] for t in tcs]}
        print(f"[{tag}:{name}] tools={out[name]['tool_calls']} well_formed={wf}")
    r = req(BASE_URL + "/chat/completions",
            {"model": "gate/model", "temperature": 0.0, "max_tokens": 128,
             "chat_template_kwargs": {"enable_thinking": False},
             "messages": [{"role": "user", "content": "Answer in one short sentence: what is 17 * 23?"}]},
            timeout=180)
    out["plain"] = {"content": r["choices"][0]["message"].get("content", "").strip()}
    print(f"[{tag}:plain] {out['plain']['content'][:120]}")
    return out
"""

C_RUN = """\
if RUN:
    results = {"ckpt": GATE_CKPT}
    p = start_server(MERGED, "/kaggle/working/vllm-merged.log")
    results["merged"] = probe("merged")
    stop_server(p)
    p = start_server(MODEL, "/kaggle/working/vllm-base.log")
    results["base"] = probe("base")
    stop_server(p)

    differ = {k: results["merged"][k] != results["base"][k] for k in ("val_img", "val_txt", "plain")}
    results["outputs_differ"] = differ
    merged_wf = all(results["merged"][k]["well_formed"] for k in ("val_img", "val_txt"))
    results["verdict"] = {"merged_serves": True, "merged_tool_calls_well_formed": merged_wf,
                          "differs_from_base_anywhere": any(differ.values())}
    json.dump(results, open("/kaggle/working/gate_result.json", "w"), indent=2)
    print("\\n" + "=" * 80)
    print(f"GATE VERDICT ({GATE_CKPT}): serves=True well_formed={merged_wf} differ={differ}")
    print("=" * 80)
    assert merged_wf, "merged model did not emit well-formed tool_calls on the val probes"
    print("SERVING-VERIFICATION GATE: PASS")
"""


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "cells": [cell("markdown", MD_HEADER), cell("code", C_GUARD), cell("code", C_INPUTS),
                cell("code", C_MERGE), cell("code", C_VLLM), cell("code", C_SERVE_FN),
                cell("code", C_RUN)]}
(HERE / "serve-verify-k3.ipynb").write_text(json.dumps(nb, indent=1))

meta = {
    "id": "ahmedmobasher86/arc-agi-3-serve-verify-k3",
    "title": "arc-agi-3-serve-verify-k3",
    "code_file": "serve-verify-k3.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True,
    "enable_gpu": True, "enable_internet": False,
    "dataset_sources": ["ahmedmobasher86/arc3-sft-k3-corpus",
                        "ahmedmobasher86/arc3-sft-k3-ckpts",
                        "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
                        "driessmit1/arc3-vllm-h100-wheelhouse-v3"],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "kernel_sources": ["ahmedmobasher86/arc3-deps-prep"],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("wrote serve-verify-k3.ipynb + kernel-metadata.json")
