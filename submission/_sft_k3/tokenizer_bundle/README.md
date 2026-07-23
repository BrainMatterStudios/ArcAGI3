---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.6-27B
tags:
  - fp8
  - w8a8
  - quantized
  - compressed-tensors
  - qwen3.6
  - vlm
  - vllm
quantized_by: vrfai
---

# Qwen3.6-27B-FP8

FP8 (W8A8) quantized version of [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) by [vrfai](https://huggingface.co/vrfai) using [llm-compressor](https://github.com/vllm-project/llm-compressor).

Also available: [vrfai/Qwen3.6-27B-NVFP4](https://huggingface.co/vrfai/Qwen3.6-27B-NVFP4) — more aggressive quantization for Blackwell GPUs only.

## FP8 Quantization Details

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| **Quantization** | W8A8 FP8 — weights FP8 static, activations FP8 static |
| **Strategy** | `tensor` (per-tensor symmetric, memoryless minmax) |
| **Format** | `compressed-tensors` (native vLLM support) |
| **Tool** | [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor) |
| **Requires** | NVIDIA Ampere / Hopper / Blackwell (SM 89+) |

### What's Quantized / What's Not

Same selective strategy as the NVFP4 variant — sensitive components are preserved in BF16:

| Component | Precision | Reason |
|---|---|---|
| FFN / MLP — all 64 transformer layers | **FP8** | High parameter density, stable under quantization |
| Full-attention projections (q/k/v/o) — 16 GQA layers | **FP8** | Standard attention, tolerant to 8-bit |
| DeltaNet / Linear-attention projections — 48 layers | **BF16** | Gated linear recurrence sensitive to numerical errors |
| Vision encoder — all 27 blocks + merger | **BF16** | Vision tower preserved for multimodal quality |
| `lm_head` | **BF16** | Output logits preserved for generation stability |

### Quantization Config (llm-compressor)

```yaml
# recipe.yaml
QuantizationModifier:
  targets: [Linear]
  scheme: FP8
  # static W8A8, per-tensor symmetric
  ignore:
    - lm_head
    - re:model\.visual\.blocks\.\d+\..*
    - model.visual.merger.linear_fc1
    - model.visual.merger.linear_fc2
    - re:model\.language_model\.layers\.\d+\.linear_attn\..*
```

---

## Quick Start (vLLM)

```bash
vllm serve vrfai/Qwen3.6-27B-FP8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --trust-remote-code \
  --tensor-parallel-size 2
```

Single GPU (≥ 24 GB VRAM, SM 89+):

```bash
vllm serve vrfai/Qwen3.6-27B-FP8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --dtype auto \
  --trust-remote-code
```

## Quantization Script

The recipes and scripts used to quantize this model can be found in the following repository:
- [VinRobotics/model-quantization-recipes](https://github.com/VinRobotics/model-quantization-recipes/tree/main/recipes/qwen36-27b)

### Python (Transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "vrfai/Qwen3.6-27B-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Explain quantization in one paragraph."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### OpenAI-compatible API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="vrfai/Qwen3.6-27B-FP8",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

---

## NVFP4 vs FP8 Comparison

| | [NVFP4](https://huggingface.co/vrfai/Qwen3.6-27B-NVFP4) | FP8 (this) |
|---|---|---|
| Weight bits | 4 | 8 |
| Activation bits | 4 (dynamic) | 8 (static) |
| Model size | ~26 GB | ~34 GB |
| Hardware | Blackwell only (SM 120+) | Ampere / Hopper / Blackwell |
| Speed | Faster | Slightly slower |
| Quality | Slightly lower | Higher |

---

## Tested Environment

| Component | Version |
|-----------|---------|
| vLLM | 0.19.1 |
| Transformers | 5.6.2 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 (nvcc 12.8.61) |
| llm-compressor | compressed-tensors 0.14.0.1 |
| GPU | 2× NVIDIA RTX 5090 (tensor-parallel-size 2) |

---

## Best Practices

| Mode | temperature | top_p | top_k | presence_penalty |
|------|-------------|-------|-------|------------------|
| Thinking — general | 1.0 | 0.95 | 20 | 0.0 |
| Thinking — coding | 0.6 | 0.95 | 20 | 0.0 |
| Non-thinking / instruct | 0.7 | 0.80 | 20 | 1.5 |

**Thinking mode:**
```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    chat_template_kwargs={"enable_thinking": True},
)
```

---

## Credits

- **Original model:** [Qwen Team](https://huggingface.co/Qwen) (Alibaba Group)
- **FP8 quantization:** [vrfai](https://huggingface.co/vrfai)
- **Quantization framework:** [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)

---

*Below is the original model card from [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B):*

---

<img width="400px" src="https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.6/logo.png">

[![Qwen Chat](https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5)](https://chat.qwen.ai)

> [!Note]
> This repository contains model weights and configuration files for the post-trained model in the Hugging Face Transformers format.

Following the February release of the Qwen3.5 series, we're pleased to share the first open-weight variant of Qwen3.6. Built on direct feedback from the community, Qwen3.6 prioritizes stability and real-world utility, offering developers a more intuitive, responsive, and genuinely productive coding experience.

## Qwen3.6 Highlights

- **Agentic Coding:** the model now handles frontend workflows and repository-level reasoning with greater fluency and precision.
- **Thinking Preservation:** reasoning context from historical messages is retained, streamlining iterative development.

![Benchmark Results](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3.6/Figures/qwen3.6_27b_score.png)

For more details, please refer to our blog post [Qwen3.6-27B](https://qwen.ai/blog?id=qwen3.6-27b).

## Model Overview

- Type: Causal Language Model with Vision Encoder
- Number of Parameters: 27B
- Context Length: 262,144 natively and extensible up to 1,010,000 tokens

### Citation

```bibtex
@misc{qwen3.6-27b,
    title  = {{Qwen3.6-27B}: Flagship-Level Coding in a {27B} Dense Model},
    author = {{Qwen Team}},
    month  = {April},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.6-27b}
}
```
