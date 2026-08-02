"""Depth pack — every change targets LEVELS COMPLETED, the only quantity that scores.

Evidence base (2026-08-02, two independent judges over disjoint samples):
- 8/10 rig-completed and 67% of 66 episode-completed levels sit AT or UNDER the human
  median baseline; the completed-share cap binds. Marginal efficiency is worth zero.
- 0.36 levels/game x 3.52% mean cap = 1.27 = our exact leaderboard score.
- One game taken 0->6 is +0.673 on the 110-game mean; the whole gap to the leader is 0.59.

THE CHANGES:

D1a  Death-safe memory (re-ship of duck-l2's FIX A, evidence unchanged). game_over
     auto-resets the SAME level; wiping the world model forces re-discovery of a level
     the agent already understood. 36 game-over events across 44 episodes.

D1b  Mechanical knowledge carry. On level_transition, the harness deletes world_model /
     action_model / etc and preserves only cross_level_notes -- which the model writes
     0.00% of the time (0/3,541 responses). At the exact moment depth becomes possible,
     carried state is emptied into a channel that is always empty. The model will not
     volunteer the field, so the harness now PROMOTES the last populated world_model and
     action_model into cross_level_notes mechanically. Levels of one game share
     mechanics; the summary of "how this game works" is exactly what the next level
     needs. (Distinct from the REFUTED transfer experiment: that tested colour/action
     effectiveness priors in a search agent, not semantic knowledge in the LLM loop.)

D1c  RESET advertised + explained. solver._engine_action_names strips RESET
     unconditionally (solver.py:116-117), so the model cannot reset even when stuck in
     an unwinnable position. Humans use reset as rollback-and-replay 35.8% of the time;
     our agent's 43 resets are 100% harness-injected death recovery. Under
     ONLY_RESET_LEVELS a reset restarts the CURRENT level (never the game) at a cost of
     1 action. Guidance states cost and semantics so the model does not spam it.

D1d  ACTION7 guidance. The bare mapping fix was measured inert (offered 229 times,
     chosen 0). Where ACTION7 exists it is UNDO -- a one-step rollback that makes
     probing near-free. The model is told that, once, in the system prompt.

Deliberately NOT included: anything touching sampling, serving, concurrency, budgets,
or board_changed (measured directionally negative per-turn).
"""
from __future__ import annotations

_MARK = "_depth_pack"

RESET_GUIDE = (
    "RESET restarts the CURRENT level only (never the whole game) and costs 1 action. "
    "Use it deliberately when the level has reached an unwinnable or badly damaged "
    "state - a fresh attempt with your current knowledge usually beats grinding a "
    "broken position. Do not use it casually: your action count carries across resets."
)
UNDO_GUIDE = (
    "Where ACTION7 is listed it is UNDO: it rolls back exactly one action. That makes "
    "cautious probing cheap - try an action, observe, undo if it hurt. Prefer one "
    "probe+undo over repeating an action whose effect you do not understand."
)


def patch_knowledge_lifecycle() -> bool:
    """D1a + D1b: keep memory through deaths; carry it mechanically across levels."""
    from inference.agent import tool_agent as ta

    original = ta.ToolAgent._update_summarized_knowledge_from_step_summary
    if getattr(original, _MARK, False):
        return True

    def _update(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("game_over") and not (
            summary.get("level_transition") or summary.get("run_complete")
        ):
            # D1a: death auto-resets the SAME level; the model is still valid.
            return
        if summary.get("level_transition") or summary.get("run_complete"):
            # D1b: promote before the wipe. Append, never overwrite - the model may
            # one day write the field itself.
            k = self._summarized_knowledge
            carry = []
            wm, am = k.get("world_model", ""), k.get("action_model", "")
            if wm:
                carry.append(f"[carried world model] {wm}")
            if am:
                carry.append(f"[carried action model] {am}")
            if carry:
                prior = k.get("cross_level_notes", "")
                joined = " | ".join(carry)
                k["cross_level_notes"] = f"{prior} | {joined}".strip(" |") if prior else joined
            for key in ("world_model", "goal_model", "action_model",
                        "recent_findings", "open_questions", "current_plan"):
                k[key] = ""

    _update._depth_pack = True
    ta.ToolAgent._update_summarized_knowledge_from_step_summary = _update
    return True


def patch_reset_advertised() -> bool:
    """D1c: stop stripping RESET from the advertised action list."""
    from inference.framework import solver as sv

    original = sv._engine_action_names
    if getattr(original, _MARK, False):
        return True

    def _engine_action_names(game):
        names = []
        for action_id in game.current_state.available_actions:
            try:
                name = sv.arcengine.GameAction.from_id(int(action_id)).name
            except Exception:
                continue
            if name not in names:
                names.append(name)
        return names

    _engine_action_names._depth_pack = True
    sv._engine_action_names = _engine_action_names
    return True


def patch_guidance() -> bool:
    """D1c + D1d: one-time system-prompt guidance for RESET and UNDO."""
    from inference.agent import tool_agent as ta

    original = ta._build_system_prompt
    if getattr(original, _MARK, False):
        return True

    def _build_system_prompt(*args, **kwargs):
        base = original(*args, **kwargs)
        return f"{base}\n- {RESET_GUIDE}\n- {UNDO_GUIDE}\n"

    _build_system_prompt._depth_pack = True
    ta._build_system_prompt = _build_system_prompt
    return True


def apply_all() -> dict:
    return {
        "knowledge_lifecycle": patch_knowledge_lifecycle(),
        "reset_advertised": patch_reset_advertised(),
        "guidance": patch_guidance(),
    }


def verify() -> None:
    """Effect checks, not install checks. Raises on any failure."""
    from inference.agent import tool_agent as ta
    from inference.framework import solver as sv

    # D1a/D1b behave correctly on a synthetic agent state.
    class _A:
        _summarized_knowledge = {"world_model": "walls block", "action_model": "A1=up",
                                 "cross_level_notes": "", "goal_model": "",
                                 "recent_findings": "", "open_questions": "", "current_plan": ""}
        _last_step_summary = {"game_over": True}
    a = _A()
    ta.ToolAgent._update_summarized_knowledge_from_step_summary(a)
    assert a._summarized_knowledge["world_model"] == "walls block", "death wiped the model"
    a._last_step_summary = {"level_transition": True}
    ta.ToolAgent._update_summarized_knowledge_from_step_summary(a)
    assert a._summarized_knowledge["world_model"] == "", "transition did not wipe"
    assert "walls block" in a._summarized_knowledge["cross_level_notes"], "carry did not happen"
    assert "A1=up" in a._summarized_knowledge["cross_level_notes"], "action model not carried"

    # D1c: RESET must survive the name filter.
    class _S:
        class current_state:
            available_actions = [0, 1, 6]
    names = sv._engine_action_names(_S())
    assert "RESET" in names, f"RESET still stripped: {names}"

    # D1d: guidance present exactly once.
    p = ta._build_system_prompt(tool_output_tokens=1024)
    assert p.count(RESET_GUIDE) == 1 and p.count(UNDO_GUIDE) == 1, "guidance missing or duplicated"
