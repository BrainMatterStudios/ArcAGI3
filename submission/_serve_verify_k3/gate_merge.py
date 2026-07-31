# gate_merge.py — merge + NLL as a SUBPROCESS.
# In-process, del/gc.collect()/empty_cache() still left 59,475 MiB resident
# (measured, serve-verify v7), so vLLM died at startup:
#   "Free memory on device cuda:0 (36.35/94.97 GiB) ... less than desired GPU
#    memory utilization (0.9, 85.47 GiB)".
# Some reference survives in the peft/transformers graph. Rather than hunt it, do
# what the duck-sft submission already does and let the OS reclaim every byte at
# process exit — which also makes this gate mirror the submission's real path.
import glob, json, os, random, shutil, subprocess, sys, time

# A subprocess does NOT inherit the notebook's sys.path. The vendored deps
# (arc3-deps-prep) carry transformers 5.14.0.dev0, which is the only build that
# knows model_type 'qwen3_5'; the system transformers raises
#   ValueError: checkpoint has model type `qwen3_5` but Transformers does not
#   recognize this architecture
# So replicate the notebook guard cell's deps insert BEFORE importing torch or
# transformers, and fail loudly if the deps tree is absent rather than silently
# falling back to a build that cannot load the model.
_deps = sorted(glob.glob("/kaggle/input/**/deps", recursive=True))
assert _deps, "arc3-deps-prep deps/ not found — refusing to run on system transformers"
sys.path.insert(0, _deps[0])
print(f"[gate_merge] deps on path: {_deps[0]}", flush=True)

import torch
import transformers
print(f"[gate_merge] transformers {transformers.__version__}", flush=True)

_IN = json.load(open("/kaggle/working/gate_inputs.json"))
MODEL, CKPT = _IN["MODEL"], _IN["CKPT"]
CORPUS, MERGED, GATE_CKPT = _IN["CORPUS"], _IN["MERGED"], _IN["GATE_CKPT"]
os.environ.setdefault("GATE_NLL_MAX_LEN", str(_IN["NLL_MAX_LEN"]))
sys.path.insert(0, CORPUS)
from sft_common import dequantize_fp8_inplace, strip_quantization_runtime
TOOLS = json.load(open(os.path.join(CORPUS, "tools.json")))
val_rows = [json.loads(l) for l in open(os.path.join(CORPUS, "val.jsonl"))]

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
# 07-26 fix: apply FP8 scales BEFORE strip deletes them (v1 died in merge on f8 +=;
# same root cause invalidated training runs 1-5)
print("dequant:", dequantize_fp8_inplace(model))
print("strip:", strip_quantization_runtime(model), f"| load {time.time()-t0:.0f}s")
dt = {str(p.dtype) for p in model.parameters()}
assert "torch.float8_e4m3fn" not in dt, dt

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
# ---- NLL: adapter-attached vs base vs merged -------------------------------
# The weight-level rel_err below is a proxy; THIS is the quantity that matters.
# 2026-07-31: the merge assert failed with rel_err 0.64-0.75 across modules
# spanning a 20x range of delta magnitudes. A synthetic reproduction showed
# that is exactly what bf16 STORAGE of a delta ~1.5e-3 the size of the weights
# produces (bf16 0.696 / fp16 0.138 / fp32 0.000) — and that accumulating the
# merge in fp32 does NOT help, because the final cast is what destroys it.
# So measure whether the merged model actually keeps the fine-tune's NLL gain,
# rather than inferring model quality from weight fidelity.
from sft_common import encode_with_mask
_nll_rows = val_rows[:4]
# Memory history, both real bugs found by running this gate:
#  v3: full forward at ctx 32768 OOM'd — HF materializes logits for EVERY position,
#      ~33k x ~152k vocab x 2B ~ 10GB (x2 more when upcast to fp32).
#  v4: capping ctx at 4096 made encode_with_mask drop ALL rows as untruncatable
#      (val rows average ~21k tokens and carry images), so every NLL came back
#      0.0 over 0 tokens and the gate then "diagnosed" a dead checkpoint from an
#      EMPTY measurement. A metric that cannot detect its own invalidity is worse
#      than no metric — hence the hard ntok assert below.
# Correct fix: keep the full context, but only compute logits for the scored
# suffix. The loss mask is the target only (~1.5k tokens), so logits_to_keep
# shrinks the hog by ~14x without touching what is measured.
NLL_MAX_LEN = int(os.environ.get("GATE_NLL_MAX_LEN", 32768))

def _row_nll(mdl, feats):
    ids = feats["input_ids"][0]
    n_tgt = int((feats["labels"][0] != -100).sum())
    fwd = {k: v for k, v in feats.items() if k != "labels"}
    with torch.no_grad():
        try:
            out = mdl(**fwd, logits_to_keep=n_tgt + 1)
        except TypeError:  # older signature
            out = mdl(**fwd, num_logits_to_keep=n_tgt + 1)
        lg = out.logits[0, :-1].float()          # predicts the last n_tgt tokens
        tgt = ids[-n_tgt:].to(lg.device)
        loss = torch.nn.functional.cross_entropy(lg, tgt, reduction="mean")
    return float(loss), n_tgt

def _nll(mdl, tag):
    tot, ntok, used, skipped = 0.0, 0, 0, []
    for r in _nll_rows:
        feats, info = encode_with_mask(proc_for_nll, r["messages"], r["target"],
                                       TOOLS, NLL_MAX_LEN)
        if feats is None:
            skipped.append(info.get("error"))
            continue
        feats = {k: (v.to(0) if hasattr(v, "to") else v) for k, v in feats.items()}
        v, n = _row_nll(mdl, feats)
        tot += v * n
        ntok += n
        used += 1
        del feats
        torch.cuda.empty_cache()
    # NEVER return a number derived from zero rows — that is what made v4 lie.
    assert ntok > 0, (
        f"NLL probe measured NOTHING for {tag}: 0 of {len(_nll_rows)} rows encoded "
        f"(errors={skipped}, ctx<={NLL_MAX_LEN}). Fix the probe before reading any "
        f"verdict — a zero here is a broken instrument, not a dead adapter.")
    v = tot / ntok
    print(f"[nll] {tag}: {v:.4f} over {ntok} target tokens "
          f"({used}/{len(_nll_rows)} rows, ctx<={NLL_MAX_LEN})", flush=True)
    return v

from transformers import AutoProcessor as _AP
try:
    proc_for_nll = _AP.from_pretrained(MODEL)
except Exception:
    proc_for_nll = _AP.from_pretrained(os.path.join(CORPUS, "tokenizer_bundle"))

nll_adapter = _nll(pmodel, "adapter attached (unmerged)")
with pmodel.disable_adapter():
    nll_base = _nll(pmodel, "base (adapter disabled)")

merged = pmodel.merge_and_unload()
nll_merged = _nll(merged, "merged")

gain_attached = nll_base - nll_adapter
gain_merged = nll_base - nll_merged
retained = gain_merged / gain_attached if abs(gain_attached) > 1e-6 else 0.0
print(f"[nll] base={nll_base:.4f} attached={nll_adapter:.4f} merged={nll_merged:.4f}")
print(f"[nll] gain attached={gain_attached:+.4f} merged={gain_merged:+.4f} "
      f"RETAINED={retained:.1%}", flush=True)
results_nll = {"base": nll_base, "attached": nll_adapter, "merged": nll_merged,
               "gain_attached": gain_attached, "gain_merged": gain_merged,
               "retained_fraction": retained}
# ----------------------------------------------------------------------------

ok = []
_named = dict(merged.named_modules())
def _resolve(name):
    # peft's merge_and_unload strips the 'base_model.model.' prefix that
    # pre-merge names carry (KeyError on raw lookup, peft 0.19.x) —
    # audit fix 2026-07-26, mirrored in serving_assert._resolve_merged_module
    for c in (name, name.removeprefix("base_model.model."), "base_model.model." + name):
        if c in _named: return _named[c]
    raise KeyError(f"module {name!r} not found post-merge (tried prefix variants)")
for (n, w_pre, BA) in checks:
    w_post = _resolve(n).weight.detach()
    delta = (w_post - w_pre).float(); exp = BA.float()
    rel = (delta - exp).norm() / (exp.norm() + 1e-9)
    ok.append({"module": n, "delta_norm": float(exp.norm()), "rel_err": float(rel)})
    print(f"[delta] {n}: |BA|={exp.norm():.4f} rel_err={rel:.4f}")
# The weight-level delta is now DIAGNOSTIC, not pass/fail. A bf16 merge of a
# delta this small is provably lossy (see the note above), so a high rel_err
# here is expected and is not by itself evidence the adapter is broken — the
# NLL block already measures what actually matters. Kept because delta_norm==0
# would still mean the adapter never attached, which IS fatal.
json.dump({"nll": results_nll, "deltas": ok},
          open("/kaggle/working/gate_merge_result.json", "w"), indent=2)
print("[gate] wrote gate_merge_result.json", flush=True)
assert all(c["delta_norm"] > 0 for c in ok), f"adapter contributed nothing: {ok}"
if any(c["rel_err"] >= 0.05 for c in ok):
    print(f"[warn] weight deltas degraded by merge rounding (expected for bf16): {ok}",
          flush=True)

# A1-PROTOCOL §1 pre-registers: merged target-NLL gain >= 2% on the val rows.
assert gain_attached > 0, (
    f"adapter gives NO NLL gain even attached — the checkpoint, not the merge, "
    f"is the problem: {results_nll}")
rel_gain_merged = gain_merged / nll_base if nll_base else 0.0
print(f"[nll] merged relative gain = {rel_gain_merged:.2%} (A1 §1 requires >= 2%)")
assert rel_gain_merged >= 0.02, (
    f"MERGE GATE FAIL — merged model does not carry the fine-tune "
    f"({rel_gain_merged:.2%} < 2%; {retained:.1%} of the attached gain survived). "
    f"Serve the adapter unmerged via vLLM --enable-lora instead. {results_nll}")
print("MERGE GATE: PASS")

merged.config.torch_dtype = torch.bfloat16
merged.config.use_cache = True
# merge_and_unload() hands back a model that still carries the compressed-tensors
# plumbing (the "Compressing/Decompressing model" bars in the log). save_pretrained
# then routes through a quantizer whose compressor is None ->
# AttributeError: 'NoneType' object has no attribute 'convert'. The weights are
# already true bf16 at this point (dequant applied, 0 f8 params), so the right move
# is to detach the quantization path entirely before writing.
for _a in ("hf_quantizer", "_hf_peft_config_loaded"):
    if hasattr(merged, _a):
        try: setattr(merged, _a, None)
        except Exception: pass
merged.is_quantized = False
for _a in ("quantization_config", "_pre_quantization_dtype", "compression_config"):
    if hasattr(merged.config, _a):
        try: delattr(merged.config, _a)
        except Exception:
            try: setattr(merged.config, _a, None)
            except Exception: pass
_left = [n for n, p in merged.named_parameters() if p.dtype == torch.float8_e4m3fn]
assert not _left, f"f8 params present at save time: {_left[:5]}"
print(f"[save] quantization detached; dtypes={sorted({str(p.dtype) for p in merged.parameters()})}",
      flush=True)
t0 = time.time()
# save_original_format defaults True in transformers 5.14.0.dev0, which runs
# revert_weight_conversion() -> mapping.convert() where one op is None:
#   AttributeError: 'NoneType' object has no attribute 'convert'
# We are feeding vLLM, which reads HF format, so reverting to the original
# checkpoint layout is both broken here and not what we want.
try:
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB",
                           save_original_format=False)
except TypeError:  # older transformers without the kwarg
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
