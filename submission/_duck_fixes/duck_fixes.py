"""Layer-2 defect fixes for the duck harness, as in-memory patches (read-only dataset).

Both fixes are MODEL-FACING in effect (they change what context the model sees), so per
the campaign's hard-won lesson they ship only after a true-stack dev A/B — never straight
to a scored submission. This module is the patch + test artifact for that pipeline.

FIX A — don't erase the working world model on death.
    `ToolAgent._update_summarized_knowledge_from_step_summary` (tool_agent.py:1139-1151)
    wipes world_model/goal_model/action_model/recent_findings/open_questions/current_plan
    on `level_transition` OR `run_complete` OR `game_over`. The level_transition and
    run_complete wipes are deliberate design (fresh grounding per level, with
    `cross_level_notes` as the intended carry-over channel — it survives the wipe).
    But on `game_over` the auto-reset (`framework/solver.py:278-283`) resumes the SAME
    level — and the agent faces it with its memory of that exact level erased. Deaths
    concentrate on deep levels, which the scorer weights 9x, so re-discovery after death
    is a direct depth tax. Fix: stop treating `game_over` as a wipe trigger; add one
    line of death context to recent_findings instead.

FIX B — count images as vision tokens, not as base64 length.
    `_estimate_tokens` (tool_agent.py:484-489) is len(json)/3. A 256x256 grid PNG
    data-url is ~1.1KB base64 -> estimated ~380 tokens vs ~102 real Qwen-VL vision
    tokens (measured 2026-07-22: overcount 3.7-4.0x/image). With images retained in up
    to 30 history turns (_PERSISTENT_HISTORY_ASSISTANT_TURNS=30, images NOT stripped
    from old user messages), the estimator wastes up to ~8.3K phantom tokens of a
    ~28.4K budget (~29%) at full history — evicting real, useful history early. Fix:
    estimate image_url parts at their true vision-token cost; leave text estimation
    untouched.
"""
from __future__ import annotations

import math
import re
from typing import Any

_DATA_URL_RE = re.compile(r"^data:image/[a-z0-9.+-]+;base64,", re.IGNORECASE)

# Qwen-VL charges ~one token per 28x28 patch (+2 wrapper tokens). We can't cheaply
# decode PNG dimensions here without PIL, so use the harness's own configuration:
# frames are 64x64 upscaled by MULTIMODAL_UPSCALE (default 16 in code, 4 in the
# shipped env), giving a fixed, known image size per run.
def _configured_image_tokens() -> int:
    import os

    raw = os.environ.get("MULTIMODAL_UPSCALE", "").strip()
    try:
        scale = max(1, int(raw)) if raw else 16
    except ValueError:
        scale = 16
    side = 64 * scale
    return math.ceil(side / 28) ** 2 + 2


def patch_estimator() -> str:
    """FIX B: make _estimate_tokens image-aware."""
    from inference.agent import tool_agent as ta

    if getattr(ta._estimate_tokens, "_image_aware", False):
        return "fixB estimator: SKIP (already applied)"

    original = ta._estimate_tokens
    image_tokens = _configured_image_tokens()

    def _walk_strip_images(value: Any) -> tuple[Any, int]:
        """Replace image data-urls with a placeholder; count how many were removed."""
        if isinstance(value, dict):
            if (
                value.get("type") == "image_url"
                and isinstance(value.get("image_url"), dict)
                and _DATA_URL_RE.match(str(value["image_url"].get("url", "")))
            ):
                return {"type": "image_url", "image_url": {"url": "<image>"}}, 1
            out = {}
            n = 0
            for k, v in value.items():
                out[k], dn = _walk_strip_images(v)
                n += dn
            return out, n
        if isinstance(value, list):
            out_list = []
            n = 0
            for v in value:
                item, dn = _walk_strip_images(v)
                out_list.append(item)
                n += dn
            return out_list, n
        return value, 0

    def _estimate_tokens(value: Any) -> int:
        stripped, n_images = _walk_strip_images(value)
        return original(stripped) + n_images * image_tokens

    _estimate_tokens._image_aware = True  # type: ignore[attr-defined]
    ta._estimate_tokens = _estimate_tokens
    return f"fixB estimator: OK (images now {image_tokens} tok each, was len/3 of base64)"


def patch_gameover_wipe() -> str:
    """FIX A: keep the working world model across a death on the same level."""
    from inference.agent import tool_agent as ta

    cls = ta.ToolAgent
    if getattr(cls._update_summarized_knowledge_from_step_summary, "_death_safe", False):
        return "fixA wipe: SKIP (already applied)"

    def _update(self: Any) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("level_transition") or summary.get("run_complete"):
            # Deliberate design: fresh grounding on a NEW level; cross_level_notes
            # carries forward. Unchanged.
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""
            return
        if summary.get("game_over"):
            # The defect path: auto-reset resumes the SAME level. Keep the model of
            # that level; discard only the now-invalid plan, and record the death so
            # the agent treats the fatal transition as evidence.
            self._summarized_knowledge["current_plan"] = ""
            findings = self._summarized_knowledge.get("recent_findings", "")
            death_note = "GAME_OVER just occurred; the last action(s) were fatal — avoid repeating them."
            if death_note not in findings:
                self._summarized_knowledge["recent_findings"] = (
                    f"{findings} {death_note}".strip()
                )

    _update._death_safe = True  # type: ignore[attr-defined]
    cls._update_summarized_knowledge_from_step_summary = _update
    return "fixA wipe: OK (game_over keeps world model, drops plan, notes the death)"


def apply_all(verbose: bool = True) -> list[str]:
    results = []
    for fn in (patch_gameover_wipe, patch_estimator):
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001 — a broken patch must not kill a run
            results.append(f"{fn.__name__}: FAIL ({exc})")
    if verbose:
        for line in results:
            print(f"[duck-fix] {line}", flush=True)
    return results
