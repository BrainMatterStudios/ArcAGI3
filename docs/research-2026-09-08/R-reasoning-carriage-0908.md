# Does prior-turn reasoning reach the model? — verified on the rig's own requests (2026-09-08)

Question (Track A1 prerequisite, docs/HANDOFF-2026-09-08-fresh-session-prompt.md): the plan assumed the
Qwen3.8 chat template drops earlier-turn reasoning ("kept only after the last user message"), so the
model's thoughts would be wiped roughly every second call. Before building "carry", this had to be read
from evidence, not assumed.

## Answer: the stock duck ALREADY carries reasoning for every retained turn

Three independent reads agree.

1. **The template** (`flashnext_chat_template_7b71922.jinja`, fetched from the served HF revision
   `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b71922`; served with `--chat-template <model_dir>/chat_template.jinja`,
   `offkaggle/modal_flashnext_serve.py:176`): for `role == "assistant"` it renders
   `<think>\n{reasoning_content}\n</think>` when `preserve_thinking is undefined or preserve_thinking is true
   or loop.index0 > ns.last_query_index`. The duck sends only `chat_template_kwargs={"enable_thinking": true}`
   (`inference/utils/openai_compat.py:68`), so `preserve_thinking` is undefined and EVERY assistant message
   keeps its think block, not only those after the last user message. The template reads
   `message.reasoning_content`; the duck sends `reasoning` (`tool_agent.py:1892`).
2. **vLLM** (main, `vllm/entrypoints/chat_utils.py::_parse_chat_message_content`): for assistant messages,
   `reasoning = message.get("reasoning")` is copied onto both `result_msg["reasoning"]` and
   `result_msg["reasoning_content"]` ("keep compatibility"). The pinned build `0.1.dev20073+g8e685d198`
   is not on GitHub (custom image `vllm/vllm-openai:qwen38-flash-next`), so (1)+(2) alone are a prior, not proof.
3. **The server's own token counts** (`reasoning_delta_probe.py`, run on two 25-game waves). Within one
   `analyze()` turn, consecutive requests differ by exactly one assistant message (reasoning + note + tool
   call) and one tool message; `usage.prompt_tokens` was recorded per request by the shim. Tokenizing the
   transcript's exact `[THINKING]` text (R), the rendered assistant remainder (C) and the tool result (T)
   with the served tokenizer:

| wave | pairs | median R | D − (C+R+T) median [IQR] | D − (C+T) median [IQR] | slope of (D−C−T) on R |
|---|---|---|---|---|---|
| 20260908T0752-keith_probe, within-turn | 408 | 466 | **+30 [+29, +33]** | +496 [+206, +1058] | **0.999** |
| 20260907T0702-keith_yield900-cb2, within-turn | 539 | 456 | **+30 [+29, +34]** | +487 [+195, +956] | **0.998** |
| keith_probe, **cross-turn** (assistant now BEFORE the newest user message) | 321 | — | **+93 [+92, +94]** (= 64 image-pad tokens + ~30 template overhead) | +502 [+253, +1118] | **1.000** |

   The residual under the "reasoning included" hypothesis is the fixed template overhead (~30 tokens;
   +64 image-pad tokens across a turn boundary) with an IQR of 4 tokens over ~1,300 pairs; the largest
   pairs (R = 7,000–10,000 tokens) match to within 50 tokens. Reasoning is in the rendered prompt for
   same-turn AND earlier-turn assistant messages.

**Consequence for A1:** re-injecting reasoning is unnecessary and would double-count it. The lever is what the
trimmer does when the window fills.

## What IS lost: eviction

`_trim_messages_for_context` (`tool_agent.py:1668-1686`) drops the oldest history block whenever the
estimated request (len(json)/3) exceeds `context_budget_tokens` = 31,744. On the same two waves:

| wave | consecutive request pairs | pairs where n_messages fell (eviction) | max prompt_tokens per run (median / max) | mean prompt_tokens/call |
|---|---|---|---|---|
| keith_probe | 1,273 | **338 (27 %)** | 27,099 / 30,050 | 20,286 |
| yield900-cb2 | 1,330 | **341 (26 %)** | 27,763 / 30,421 | 20,405 |

The window fills after ~10 turns (~25 min of a 132-min game) and then rolls: everything older than the
retained ~10 turns survives only as the one-line "World model:" note (`_summarized_knowledge_lines`).
Roughly 7,000 of the ~27,000 retained tokens are the stock user-prompt boilerplate repeated in every
retained turn (not changed by A1; noted as a follow-up rider).

## Design change to A1 (built as `submission/_throughput_v1/graft_carry.py`)

* carry = **measured**, not re-injected: `[CARRY-CALL]` per request records assistant messages carrying
  `reasoning` and their chars (expected ≈ retained turns × 1, i.e. several thousand chars per request).
* compaction = the lever: when the trimmer must drop, drop a chunk (down to 50 % of the budget), ask the
  model in one extra no-tools call (thinking off, max_tokens 1500) to fold the dropped turns into a
  compacted-knowledge block, keep the block in the system message (single copy, rebuilt per request,
  never persisted), then enforce the hard budget with the stock trimmer.

Files: `reasoning_delta_probe.py` (needs `tokenizer.json` from the HF revision beside it), the template copy,
`chat_utils` read from vLLM main (not committed). Nothing in the repo's stock bundles was modified.
