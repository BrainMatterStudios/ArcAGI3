"""test_strip_quant.py — verify strip_quantization_runtime against REAL compressed-tensors.

Builds a toy model, applies an FP8 quantization config (same lifecycle the FP8 27B
snapshot goes through in-kernel), asserts the QDQ forward wrapper is live and alters
outputs, strips, and asserts forward is restored bit-identical to plain dense bf16.

Run: PYTHONPATH=<dir containing compressed_tensors> .venv/bin/python submission/_sft_k3/test_strip_quant.py
(skips cleanly if compressed_tensors is not importable)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import strip_quantization_runtime  # noqa: E402

try:
    from compressed_tensors.quantization import (QuantizationConfig,
                                                 QuantizationStatus,
                                                 apply_quantization_config)
except ImportError:
    print("SKIP: compressed_tensors not importable (set PYTHONPATH to the ctdeps dir)")
    sys.exit(0)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(128, 64, dtype=torch.bfloat16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)
    model, ref = Toy(), Toy()
    ref.load_state_dict(model.state_dict())
    x = torch.randn(4, 64, dtype=torch.bfloat16)

    cfg = QuantizationConfig(
        config_groups={"group_0": {"targets": ["Linear"],
                                   "weights": {"num_bits": 8, "type": "float",
                                               "strategy": "channel", "symmetric": True}}},
        quantization_status=QuantizationStatus.FROZEN)  # post-decompress state in-kernel
    apply_quantization_config(model, cfg)
    # scales default to ~1.0 placeholders; that makes QDQ visibly clamp real weights,
    # which is fine — we only need "wrapper active" vs "wrapper gone"
    assert "forward" in model.fc1.__dict__, "expected instance-level QDQ wrapper"
    assert hasattr(model.fc1, "weight_scale"), "expected registered qparams"
    y_qdq = model(x)
    assert not torch.equal(y_qdq, ref(x)), "QDQ wrapper had no effect — test is vacuous"

    stats = strip_quantization_runtime(model)
    print("strip stats:", stats)
    assert stats["quantized_modules"] == 2 and stats["unwrapped_forwards"] == 2
    assert "forward" not in model.fc1.__dict__ and not hasattr(model.fc1, "weight_scale")
    assert not any(n.endswith("_scale") for n, _ in model.named_parameters())
    y = model(x)
    assert torch.equal(y, ref(x)), "stripped forward must be bit-identical to dense bf16"
    # idempotent + no grad-graph surprises
    assert strip_quantization_runtime(model)["quantized_modules"] == 0
    print("STRIP-QUANT TEST PASSED (wrapper removed, dense forward restored bit-identical)")


if __name__ == "__main__":
    main()
