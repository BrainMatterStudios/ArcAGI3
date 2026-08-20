"""Animation-digest graft — harness-side decoding of transient animation
frames into compact prompt text.

Motivation (measured, wf_9809fe85 + prototype test 2026-08-20): on sb26 the
level-deciding information lives ONLY in transient animation frames; the run
that decoded them manually (801s, 10 LLM calls) cleared the level, its twin
misread them and got permanently stuck. digest_animation reproduces the
rule-deciding content deterministically in ~24ms / ~30 tokens (falsifiable
test in submission/_anim_digest/test_sb26_check.py: PASS on both the failing
and passing check scenarios).

Seams (verified against the Aug-07 bundle):
- inference/framework/solver.py:52 imports summarize_animation BY NAME and
  calls it at :822 with the raw frames — we wrap the function and rebind it
  in BOTH namespaces (the animation module AND solver's import-time binding;
  the duck-mem silent-no-op lesson).
- inference/agent/tool_agent.py:39 imports describe_animation BY NAME, calls
  it at :1441 — same double rebind; the wrapper appends summary["digest"].

Fail-open: any exception inside either wrapper returns the stock result.
DIGEST=0 disables at call time.
"""

from __future__ import annotations

from typing import Any

from digest import digest_animation  # inlined alongside in the notebook


def _enabled() -> bool:
    import os

    return os.environ.get("DIGEST", "1").strip() not in {"0", "false", "False"}


def install() -> str:
    try:
        from inference.utils import animation as anim_mod
    except Exception as exc:  # noqa: BLE001
        return f"digest: SKIP (animation module missing: {exc!r})"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"digest: SKIP (solver module missing: {exc!r})"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"digest: SKIP (tool_agent module missing: {exc!r})"

    for name, mod in (("summarize_animation", anim_mod), ("describe_animation", anim_mod)):
        if not hasattr(mod, name):
            return f"digest: SKIP (missing {name})"
    if not hasattr(solver_mod, "summarize_animation"):
        return "digest: SKIP (solver lacks summarize_animation binding)"
    if not hasattr(agent_mod, "describe_animation"):
        return "digest: SKIP (tool_agent lacks describe_animation binding)"
    if getattr(anim_mod.summarize_animation, "_digest_patched", False):
        return "digest: SKIP (already applied)"

    original_summarize = anim_mod.summarize_animation
    original_describe = anim_mod.describe_animation

    def summarize_with_digest(frames: Any, *, board_changed: bool) -> dict[str, Any] | None:
        summary = original_summarize(frames, board_changed=board_changed)
        if summary is None or not _enabled():
            return summary
        try:
            text = digest_animation(frames)
            if text:
                summary["digest"] = text
        except Exception:  # noqa: BLE001 — a broken digest must never break an action
            pass
        return summary

    def describe_with_digest(summary: dict[str, Any] | None) -> str:
        line = original_describe(summary)
        if not _enabled() or not summary:
            return line
        try:
            text = summary.get("digest")
            if text:
                return f"{line} Decoded animation: {text}" if line else f"Decoded animation: {text}"
        except Exception:  # noqa: BLE001
            pass
        return line

    summarize_with_digest._digest_patched = True  # type: ignore[attr-defined]
    describe_with_digest._digest_patched = True  # type: ignore[attr-defined]
    # BOTH namespaces, per the silent-no-op lesson
    anim_mod.summarize_animation = summarize_with_digest
    solver_mod.summarize_animation = summarize_with_digest
    anim_mod.describe_animation = describe_with_digest
    agent_mod.describe_animation = describe_with_digest
    return "digest: OK"
