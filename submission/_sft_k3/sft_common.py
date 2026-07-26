"""sft_common.py — shared corpus/encoding logic for the K3 teacher-distillation SFT.

Used by prep_dataset.py (local), test_loss_mask.py (local) and sft-k3.ipynb (Kaggle,
imported from the corpus dataset). Single source of truth for:
  * message normalization (raw duck-harness trace -> Qwen3.5 chat-template shape)
  * render (full conversation text + target-suffix split at the LAST
    '<|im_start|>assistant' — a special-token boundary, so the suffix tokenizes
    identically in isolation and in context, and the mask survives the processor's
    image-token expansion which only touches the prefix)
  * encode_with_mask (input_ids/pixel_values/labels; loss on target tokens only)
  * front-truncation of overlong conversations
"""
from __future__ import annotations

import base64
import copy
import io
import json

ASSISTANT_MARK = "<|im_start|>assistant"
IMAGE_PAD_ID = 248056  # <|image_pad|> in the Qwen3.6 tokenizer


def dequantize_fp8_inplace(model) -> dict:
    """Materialize TRUE bf16 weights from compressed-tensors FP8 storage: w_bf16 = w_f8 * weight_scale.

    MUST run BEFORE strip_quantization_runtime, which deletes the scales without
    applying them. 2026-07-26 root cause: on the VL snapshot, transformers keeps the
    256 quantized Linears as f8e4m3 STORAGE (values = w_true/scale, per-tensor scale);
    runs 1-5 stripped the scales unapplied and trained the LoRA against weights
    inflated by 1/scale — base per-token NLL ~14.7 nats (above uniform ln(V)~12.4).
    The July text-model recipe's "upcast on load" claim did not hold for this snapshot.
    """
    import torch
    n, missing = 0, []
    for name, mod in model.named_modules():
        w = mod._parameters.get("weight")
        if w is None or w.dtype != torch.float8_e4m3fn:
            continue
        scale = None
        for store in (mod._parameters, mod._buffers):
            if "weight_scale" in store:
                scale = store["weight_scale"]
        if scale is None:
            missing.append(name)
            continue
        mod._parameters["weight"] = torch.nn.Parameter(
            (w.float() * scale.float()).to(torch.bfloat16), requires_grad=False)
        n += 1
    assert not missing, f"f8 weights without a weight_scale: {missing[:5]}"
    left = [n2 for n2, p in model.named_parameters() if p.dtype == torch.float8_e4m3fn]
    assert not left, f"f8 params remain after dequant: {left[:5]}"
    return {"dequantized": n}


def strip_quantization_runtime(model) -> dict:
    """Remove compressed-tensors' RUNTIME QDQ so forward is pure dense-bf16 matmul.

    transformers decompresses the FP8 checkpoint to dense bf16 params at load
    (run_compressed=False), but every quantized Linear keeps an INSTANCE-level
    `forward` wrapper (compressed_tensors set_forward_quantized:
    `module.forward = quantized_forward.__get__(module)`) that fake-quantizes the
    weight on EVERY forward — large fp32 clamp temporaries caused the 2026-07-23
    CUDA OOM at step 5. Config-level stripping does NOT remove these.

    Inverse (verified against compressed-tensors 0.17.1 source): delete the
    instance attr so the class forward reappears; set the documented off-switch
    `quantization_enabled = False` (belt-and-braces for other versions); drop
    the registered qparams and scheme attrs. Returns counts for logging.
    """
    qsuffix = ("_scale", "_zero_point", "_g_idx", "_global_scale")
    n_mod = n_fwd = n_par = 0
    for mod in model.modules():
        if not hasattr(mod, "quantization_scheme"):
            continue
        n_mod += 1
        if "forward" in mod.__dict__:  # the QDQ wrapper shadows the class method
            del mod.__dict__["forward"]
            n_fwd += 1
        mod.quantization_enabled = False
        for store in (mod._parameters, mod._buffers):
            for name in [n for n in store if n.endswith(qsuffix)]:
                del store[name]
                n_par += 1
        for attr in ("quantization_scheme", "quantization_status"):
            mod.__dict__.pop(attr, None)
    shadowed = [n for n, m in model.named_modules() if "forward" in m.__dict__]
    assert not shadowed, f"forward still shadowed on: {shadowed[:5]}"
    return {"quantized_modules": n_mod, "unwrapped_forwards": n_fwd,
            "removed_qparams": n_par}


def normalize_sample(raw: dict) -> tuple[list[dict], dict]:
    """Harvested sample -> (messages, target) in chat-template shape.

    The Qwen3.5 template needs `reasoning_content` (ours is `reasoning`) and
    dict-form tool_call `arguments` (ours are JSON strings, per the OpenAI wire
    format the capture proxy recorded).
    """
    def fix_msg(m: dict) -> dict:
        m = copy.deepcopy(m)
        if "reasoning" in m:
            m["reasoning_content"] = m.pop("reasoning")
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    tc["function"]["arguments"] = {"code": args}
        return m

    return [fix_msg(m) for m in raw["messages"]], fix_msg(raw["target"])


def extract_images(messages: list[dict]):
    """PIL images from data-url content parts, in document order."""
    from PIL import Image  # deferred: kernel-side torch image stack
    out = []
    for m in messages:
        if isinstance(m.get("content"), list):
            for p in m["content"]:
                if p.get("type") == "image_url":
                    b64 = p["image_url"]["url"].split(",", 1)[1]
                    out.append(Image.open(io.BytesIO(b64.decode() if isinstance(b64, bytes) else base64.b64decode(b64))).convert("RGB"))
    return out


def render(tokenizer, messages: list[dict], target: dict, tools: list[dict]) -> tuple[str, str]:
    """-> (full_text, target_text). preserve_thinking=True matches how the model
    is SERVED (vLLM --default-chat-template-kwargs preserve_thinking) and how the
    teacher data was collected: prior turns keep their <think> blocks."""
    full = tokenizer.apply_chat_template(
        messages + [target], tools=tools, tokenize=False,
        add_generation_prompt=False, preserve_thinking=True)
    cut = full.rfind(ASSISTANT_MARK)
    if cut <= 0:
        raise ValueError("no assistant block found in rendered conversation")
    return full, full[cut:]


def truncate_front(messages: list[dict], max_over: int = 200):
    """Drop the oldest turn after [system, first-user]; then any orphaned tool
    responses. Returns a NEW list one unit shorter, or None if nothing droppable."""
    if len(messages) <= 3:
        return None
    msgs = messages[:2] + messages[3:]
    while len(msgs) > 2 and msgs[2]["role"] == "tool":
        msgs = msgs[:2] + msgs[3:]
    return msgs if len(msgs) < len(messages) else None


def encode_with_mask(processor, messages: list[dict], target: dict,
                     tools: list[dict], max_len: int):
    """-> (features dict, info dict). Loss mask = target suffix only.

    input_ids come from the PROCESSOR (image pads expanded to real image tokens);
    n_tgt comes from tokenizing target_text alone — exact because the suffix
    starts at the special token <|im_start|>, across which BPE cannot merge, and
    contains no images. Front-truncates whole turns until <= max_len.
    """
    tok = processor.tokenizer
    dropped_turns = 0
    while True:
        full_text, target_text = render(tok, messages, target, tools)
        images = extract_images(messages)
        # truncation=False, max_length=None are REQUIRED: the saved tokenizer's
        # init_kwargs carry a stale max_length=1024 which the processor merges
        # into text_kwargs and silently truncates with (found 2026-07-23).
        enc = processor(text=[full_text], images=images or None,
                        return_tensors="pt", truncation=False, max_length=None)
        n_total = enc["input_ids"].shape[1]
        if n_total <= max_len:
            break
        nxt = truncate_front(messages)
        if nxt is None:
            return None, {"error": "untruncatable_overlong", "n_total": int(n_total)}
        messages, dropped_turns = nxt, dropped_turns + 1

    tgt_ids = tok(target_text, add_special_tokens=False)["input_ids"]
    n_tgt = len(tgt_ids)
    ids = enc["input_ids"][0]
    if ids[-n_tgt:].tolist() != tgt_ids:
        raise AssertionError("target suffix tokens do not match in-context tail")
    labels = ids.clone()
    labels[:-n_tgt] = -100
    feats = {k: v for k, v in enc.items()}
    feats["labels"] = labels.unsqueeze(0)
    info = {"n_total": int(n_total), "n_target": n_tgt, "n_images": len(images),
            "n_image_tokens": int((ids == IMAGE_PAD_ID).sum()),
            "dropped_turns": dropped_turns}
    return feats, info
