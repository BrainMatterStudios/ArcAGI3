"""Unit tests for serving_assert.py — CPU-only, toy modules.

Run:  .venv/bin/python -m pytest submission/_serve_verify_k3/test_serving_assert.py -q
"""
from __future__ import annotations

import io
import json
import math

import pytest
import torch
import torch.nn as nn

import serving_assert as sa


# ---------- toy fixtures ----------

class ToyLoraLinear(nn.Module):
    """Mimics peft's LoRA-wrapped Linear closely enough for the asserts."""

    def __init__(self, din=8, dout=8, r=2, zero_ba=False):
        super().__init__()
        self.base_layer = nn.Linear(din, dout, bias=False)
        a = nn.Linear(din, r, bias=False)
        b = nn.Linear(r, dout, bias=False)
        if zero_ba:
            nn.init.zeros_(a.weight)
            nn.init.zeros_(b.weight)
        self.lora_A = {"default": a}
        self.lora_B = {"default": b}


class ToyPeftModel(nn.Module):
    def __init__(self, n_mods=5, zero_ba=False):
        super().__init__()
        self.mods = nn.ModuleDict(
            {f"m{i}": ToyLoraLinear(zero_ba=zero_ba) for i in range(n_mods)})

    def merge(self, scaling):
        """Return a plain module dict with merged weights (peft merge_and_unload analog)."""
        merged = nn.ModuleDict()
        for name, m in self.mods.items():
            lin = nn.Linear(8, 8, bias=False)
            ba = (m.lora_B["default"].weight @ m.lora_A["default"].weight) * scaling
            with torch.no_grad():
                lin.weight.copy_(m.base_layer.weight + ba)
            merged[name] = lin
        return merged


class ToyLM(nn.Module):
    """Tiny 'language model': vocab 16, returns CE loss over labels != -100."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.emb = nn.Embedding(16, 16)
        self.head = nn.Linear(16, 16)
        with torch.no_grad():
            self.head.bias.fill_(bias)

    def forward(self, input_ids, labels):
        logits = self.head(self.emb(input_ids))
        loss = nn.functional.cross_entropy(
            logits.view(-1, 16), labels.view(-1), ignore_index=-100)
        return type("Out", (), {"loss": loss})()


ROWS = [
    {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
    {"input_ids": [5, 6, 7, 8], "labels": [-100, 7, 8, -100]},
]


def lora_mods(pm):
    return [(f"mods.{k}", m) for k, m in pm.mods.items()]


# ---------- tests ----------

def test_full_chain_passes_strict():
    rep = sa.AssertReport(strict=True)
    pm = ToyPeftModel()
    mods = [(n, m) for n, m in
            [(f"mods.{k}", v) for k, v in pm.mods.items()]]
    sa.assert_adapter_attached(rep, pm, min_modules=5)
    checks = sa.sample_merge_checks(mods, scaling=2.0, k=3)
    merged_container = nn.Module()
    merged_container.mods = pm.merge(scaling=2.0)
    assert sa.assert_merge_delta(rep, merged_container, checks)
    assert rep.all_ok


def test_merge_delta_catches_unmerged_base():
    """The 1.26=base scenario: 'merged' weights identical to base."""
    rep = sa.AssertReport(strict=False, out_path="/tmp/sa_fail1.json")
    pm = ToyPeftModel()
    mods = [(f"mods.{k}", v) for k, v in pm.mods.items()]
    checks = sa.sample_merge_checks(mods, scaling=2.0, k=3)
    unmerged = nn.Module()
    unmerged.mods = nn.ModuleDict()
    for name, m in pm.mods.items():
        lin = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            lin.weight.copy_(m.base_layer.weight)  # no delta applied
        unmerged.mods[name] = lin
    assert not sa.assert_merge_delta(rep, unmerged, checks)
    assert not rep.all_ok


def test_merge_delta_catches_zero_adapter():
    """An all-zero B@A must fail (delta_norm == 0 is not a valid merge proof)."""
    rep = sa.AssertReport(strict=False, out_path="/tmp/sa_fail2.json")
    pm = ToyPeftModel(zero_ba=True)
    mods = [(f"mods.{k}", v) for k, v in pm.mods.items()]
    checks = sa.sample_merge_checks(mods, scaling=2.0, k=3)
    merged_container = nn.Module()
    merged_container.mods = pm.merge(scaling=2.0)
    assert not sa.assert_merge_delta(rep, merged_container, checks)


def test_adapter_attached_threshold():
    rep = sa.AssertReport(strict=False, out_path="/tmp/sa_fail3.json")
    pm = ToyPeftModel(n_mods=3)
    sa.assert_adapter_attached(rep, pm, min_modules=400)
    assert not rep.all_ok


def test_base_health_and_nll_gate():
    rep = sa.AssertReport(strict=True)
    lm = ToyLM()
    nll = sa.check_base_health(rep, lm, ROWS)
    assert math.isfinite(nll) and nll < sa.BASE_NLL_MAX  # ln(16) ~ 2.77 max


def test_nll_improvement_pass_and_fail():
    torch.manual_seed(0)
    base = ToyLM()
    base_nll = sa.target_nll(base, ROWS)

    rep = sa.AssertReport(strict=False, out_path="/tmp/sa_fail4.json")
    assert not sa.assert_nll_improves(rep, base_nll, base, ROWS)  # same model: 0 gain

    # a genuinely better model: nudge head toward the true labels
    better = ToyLM()
    better.load_state_dict(base.state_dict())
    opt = torch.optim.SGD(better.parameters(), lr=0.5)
    for _ in range(30):
        for row in ROWS:
            out = better(torch.tensor([row["input_ids"]]),
                         torch.tensor([row["labels"]]))
            opt.zero_grad(); out.loss.backward(); opt.step()
    rep2 = sa.AssertReport(strict=True)
    assert sa.assert_nll_improves(rep2, base_nll, better, ROWS)


def test_endpoint_identity_pass_fail_and_transport_error():
    payload_ok = {"data": [{"id": "/tmp/merged_checkpoint-8"}]}
    payload_bad = {"data": [{"id": "/kaggle/input/base-snapshot"}]}

    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def opener_for(payload):
        def _open(url, timeout):
            return FakeResp(json.dumps(payload).encode())
        return _open

    rep = sa.AssertReport(strict=True)
    assert sa.assert_endpoint_serves(rep, "http://x/v1", "merged_checkpoint-8",
                                     _opener=opener_for(payload_ok))
    rep2 = sa.AssertReport(strict=False, out_path="/tmp/sa_fail5.json")
    assert not sa.assert_endpoint_serves(rep2, "http://x/v1", "merged_checkpoint-8",
                                         _opener=opener_for(payload_bad))

    def opener_raises(url, timeout):
        raise OSError("connection refused")
    rep3 = sa.AssertReport(strict=False, out_path="/tmp/sa_fail6.json")
    assert not sa.assert_endpoint_serves(rep3, "http://x/v1", "m",
                                         _opener=opener_raises)


def test_strict_mode_raises_and_safe_mode_writes_marker(tmp_path):
    rep = sa.AssertReport(strict=True)
    with pytest.raises(AssertionError):
        rep.record("boom", False, {"why": "test"})

    out = tmp_path / "fail.json"
    rep2 = sa.AssertReport(strict=False, out_path=str(out))
    rep2.record("boom", False, {"why": "test"})
    assert out.exists()
    assert json.loads(out.read_text())[0]["check"] == "boom"
    assert not rep2.finish()
