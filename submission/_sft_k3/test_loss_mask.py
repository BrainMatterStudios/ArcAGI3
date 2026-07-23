"""test_loss_mask.py — unit tests for sft_common loss masking on synthetic samples.

Run: .venv/bin/python submission/_sft_k3/test_loss_mask.py
"""
import base64
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sft_common import IMAGE_PAD_ID, encode_with_mask, normalize_sample  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "python", "description": "run code",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                         "required": ["code"]}}}]


def png_data_url(px=256):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (px, px), (200, 30, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def synthetic(n_filler_turns=1):
    msgs = [{"role": "system", "content": "You solve grid puzzles."},
            {"role": "user", "content": [{"type": "text", "text": "State: step 1, level 1"},
                                         {"type": "image_url", "image_url": {"url": png_data_url()}}]}]
    for i in range(n_filler_turns):
        msgs += [{"role": "assistant", "content": f"probe {i}", "reasoning": f"hmm {i}",
                  "tool_calls": [{"type": "function", "id": f"t{i}", "function": {
                      "name": "python", "arguments": f'{{"code":"print({i})"}}'}}]},
                 {"role": "tool", "tool_call_id": f"t{i}", "content": f"out {i}"}]
    target = {"role": "assistant", "reasoning": "The red cell is the goal.",
              "tool_calls": [{"type": "function", "id": "tz", "function": {
                  "name": "python", "arguments": '{"code":"action([{\'action\':\'MOUSE\',\'row\':3,\'col\':4}])"}'}}]}
    return {"messages": msgs, "target": target}


def main():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(HERE / "tokenizer_bundle")
    tok = proc.tokenizer

    # 1. mask covers exactly the target block, all prefix (incl. image tokens) masked
    msgs, tgt = normalize_sample(synthetic())
    feats, info = encode_with_mask(proc, msgs, tgt, TOOLS, 32768)
    ids, lbl = feats["input_ids"][0], feats["labels"][0]
    trained = lbl[lbl != -100]
    assert (lbl != -100).sum() == info["n_target"], "mask size != target token count"
    assert (lbl[:-info["n_target"]] == -100).all(), "prefix not fully masked"
    dec = tok.decode(trained)
    assert dec.startswith("<|im_start|>assistant\n<think>\nThe red cell is the goal."), dec[:90]
    assert "action([{'action':'MOUSE','row':3,'col':4}])" in dec, "target code missing from trained span"
    assert dec.endswith("</tool_call><|im_end|>\n"), repr(dec[-40:])
    assert "probe 0" not in dec and "out 0" not in dec, "history leaked into loss"
    img_positions = (ids == IMAGE_PAD_ID)
    assert img_positions.sum() == 64, "expected 64 image tokens for one 256x256 image"
    assert (lbl[img_positions] == -100).all(), "image tokens must be masked"
    # trained span sits at the very end of input_ids
    assert (ids[-info["n_target"]:] == trained).all(), "trained tokens are not the suffix"
    print(f"test 1 OK  (total={info['n_total']}, target={info['n_target']}, img_tokens=64)")

    # 2. front-truncation drops oldest turns, keeps sample valid, target intact
    msgs, tgt = normalize_sample(synthetic(n_filler_turns=40))
    full_len = encode_with_mask(proc, msgs, tgt, TOOLS, 32768)[1]["n_total"]
    feats, info = encode_with_mask(proc, msgs, tgt, TOOLS, full_len - 1)
    assert info["dropped_turns"] >= 1 and info["n_total"] <= full_len - 1
    dec = tok.decode(feats["labels"][0][feats["labels"][0] != -100])
    assert "row':3" in dec.replace('"', "'"), "target lost after truncation"
    print(f"test 2 OK  (dropped {info['dropped_turns']} turns, {full_len} -> {info['n_total']})")

    # 3. untruncatable sample reports an error instead of crashing
    tiny = {"messages": synthetic(0)["messages"], "target": synthetic(0)["target"]}
    msgs, tgt = normalize_sample(tiny)
    feats, info = encode_with_mask(proc, msgs, tgt, TOOLS, 10)
    assert feats is None and info["error"] == "untruncatable_overlong"
    print("test 3 OK  (overlong drop path)")
    print("ALL LOSS-MASK TESTS PASSED")


if __name__ == "__main__":
    main()
