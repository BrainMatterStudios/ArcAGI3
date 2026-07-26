"""test_dequant.py — dequantize_fp8_inplace applies scales correctly and composes with strip.

Simulates compressed-tensors FP8 storage on toy Linears: weight stored as
f8e4m3(value = w_true/scale) + per-tensor weight_scale parameter, plus the
instance-forward wrapper + scheme attrs that strip_quantization_runtime removes.
"""
import torch
import torch.nn as nn

from sft_common import dequantize_fp8_inplace, strip_quantization_runtime


def make_quantized_linear(out_f, in_f, scale_val):
    lin = nn.Linear(in_f, out_f, bias=False)
    w_true = lin.weight.detach().clone()
    scale = torch.tensor(scale_val, dtype=torch.float32)
    w_f8 = (w_true / scale).to(torch.float8_e4m3fn)
    lin._parameters["weight"] = nn.Parameter(w_f8, requires_grad=False)
    lin._parameters["weight_scale"] = nn.Parameter(scale, requires_grad=False)
    lin._parameters["input_scale"] = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    lin.quantization_scheme = object()
    lin.forward = (lambda self, x: x).__get__(lin)  # stand-in QDQ wrapper
    return lin, w_true, scale


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1, self.w1, self.s1 = make_quantized_linear(8, 4, 0.004)
        self.q2, self.w2, self.s2 = make_quantized_linear(6, 8, 0.0007)
        self.plain = nn.Linear(4, 4)  # untouched bf16/fp32 module


def main():
    m = Toy()
    w1_expected = (m.w1 / m.s1).to(torch.float8_e4m3fn).float() * m.s1

    stats = dequantize_fp8_inplace(m)
    assert stats == {"dequantized": 2}, stats
    assert m.q1.weight.dtype == torch.bfloat16
    assert torch.allclose(m.q1.weight.float(), w1_expected, atol=0, rtol=0.02), \
        "dequant must reproduce f8_value * scale exactly (up to bf16 cast)"
    # the OLD bug: scale-less cast is ~1/scale off — prove we are NOT doing that
    inflation = m.q1.weight.float().norm() / m.w1.norm()
    assert 0.8 < inflation < 1.25, f"weights look scale-less-inflated: x{inflation:.1f}"
    print("dequant values PASSED (inflation factor ~1.0, not ~%d)" % round(1 / 0.004))

    strip_stats = strip_quantization_runtime(m)
    assert strip_stats["unwrapped_forwards"] == 2
    assert strip_stats["removed_qparams"] == 4  # weight_scale + input_scale on q1,q2
    assert not any(p.dtype == torch.float8_e4m3fn for p in m.parameters())
    x = torch.randn(2, 4)
    m.q1.float()(x)  # class forward restored, dense matmul runs
    print("strip-after-dequant PASSED")

    # regression guard for the run-1..5 failure mode: strip WITHOUT dequant leaves
    # f8 params that can never be repaired (scales gone)
    m2 = Toy()
    strip_quantization_runtime(m2)
    assert any(p.dtype == torch.float8_e4m3fn for p in m2.parameters())
    print("ALL test_dequant PASSED")


if __name__ == "__main__":
    main()
