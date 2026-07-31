# CLAUDE-SOURCED DATA — provenance and authorization record

*(Filename retained: `sweep_driver.py` asserts on it as a segregation marker, and
segregation still applies — this data must stay identifiable as Claude-sourced and
must never be silently merged into a K3 corpus. What changed is the training gate.)*

Everything under `scratchpad/claude_corpus/` is **Claude-generated teacher data**,
produced through Ahmed's Claude Code subscription via `claude_shim.py`.

## Authorization

**Training on this corpus is AUTHORIZED by Ahmed as of 2026-07-31.**

Approval reference: **Ahmed's own decision — Anthropic did not respond to 3 requests.**

Rationale of record, in his words: Anthropic's terms prohibit using outputs to train
**competing** models; this is narrow ARC-AGI-3 research on a task-specific
game-playing agent fine-tuned from Qwen, not a competing general model. Three
requests for written clarification went unanswered, so the judgment call and the
responsibility for it are his.

This supersedes the prior standing rule (HANDOFF-2026-07-25 onward) that required
written Anthropic approval before any training on Claude outputs. The earlier gate
was Ahmed's own; so is lifting it.

## Segregation rules that still apply

- Keep `meta.teacher` accurate on every sample so Claude-sourced rows stay
  identifiable and separable after the fact.
- Never blend into `rl_gate/sft_data/all_wins.jsonl` or a `arc3-sft-k3-*` dataset
  without a distinct teacher tag — corpus v3 exists precisely because silent
  teacher mixing went unnoticed once already (22 kimi-k2.7-code rows in v2).
- Any Kaggle dataset carrying these rows should name the teacher in its
  description, not just its title.
