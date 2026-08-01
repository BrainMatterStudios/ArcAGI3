"""Measured harness levers — monkey-patch pack for the duck.

Every lever here was measured against 41 real recorded episodes (5,198 scored
actions, 3,556 model requests) by the 2026-08-01 audit. Nothing in this pack is
speculative: each patch cites the number that justifies it, and each one fails
loudly rather than degrading quietly, because scored-rerun logs are never
retrievable (`kaggle kernels output` returns the commit run, not the scored one).

LEVERS
  L1 token estimator.  `_estimate_tokens` is `len(json)//3`. Measured against real
     `usage.prompt_tokens` over 3,556 requests the ratio is 1.44x (p50 1.39, p90
     1.66). The trimmer drops oldest history until the *estimate* fits a 31,744
     budget and sits pinned at exactly that ceiling on 11.8% of requests, so the
     agent discards roughly 30% of the context window it has already paid for.
     JSON escaping plus a 3-chars-per-token constant (real is ~4.3 for this
     content) compound. Fix: divide by 4. Still errs high (~1.08x), which is the
     safe direction -- underestimating would overrun the served window.

  L2 context budget.  vLLM serves `--max-model-len 65536`; the analyzer is capped
     at 32768 (`_LOCAL_ANALYZER_CONTEXT_WINDOW`). Combined with L1 the real
     utilisation is ~28%. Raised to 49152 rather than the served maximum: with
     `_reply_reserve_tokens` hardcoded to 512, a 65536 window would leave only
     ~5.8k tokens for generation against long thinking traces. At 49152 the real
     prompt caps near 44.6k and generation keeps ~21k.

     NOTE this constant is read at module import and consumed in ToolAgent.__init__,
     so setting the environment variable from the notebook is a no-op. The module
     attribute must be rebound, and it must happen before any analyzer is built.

  L4 board lattice.  The system prompt asserts "boards are presented as 64 x 64
     color grids". False for 9 of 25 games, which render a small grid upscaled
     (bp35/lf52 8x8@8, m0r0 11x11@5, sp80 16x16@4, cn04 20x20@3, ar25 21x21@3,
     ft09/vc33 32x32@2, lp85 32x19@2). Consequence measured over 2,511 real
     clicks: 279 of them (11.1%) land in the same *game cell* as the immediately
     preceding click -- the model believes it stepped to a neighbouring cell and
     did not. The scale is recoverable from the frame alone, so this works in
     competition mode with no engine access.

Model, sampling, concurrency, per-game budget, the game list and the submission
path are untouched.
"""
from __future__ import annotations

from typing import Any

# Candidate upscale factors, descending. 1 means "already a true 64x64 board".
_SCALES = (8, 7, 6, 5, 4, 3, 2)

# A lattice claim needs at least this many cells on a side, otherwise a nearly
# uniform frame (a title screen, a fade) would "prove" any scale you like.
_MIN_CELLS = 3


def _rows_cols(grid: Any) -> tuple[int, int]:
    try:
        return len(grid), len(grid[0])
    except (TypeError, IndexError):
        return 0, 0


def detect_lattice(grid: Any) -> tuple[int, int, int] | None:
    """Return (scale, cell_rows, cell_cols), or None when the board is truly 1:1.

    A frame is upscaled by `s` when every s x s block in the active region is a
    single colour and whatever remains on the right/bottom is uniform letterbox.
    Checked largest-scale-first so 8 wins over its divisors 4 and 2.
    """
    nrows, ncols = _rows_cols(grid)
    if nrows == 0 or ncols == 0:
        return None

    for s in _SCALES:
        cell_rows, cell_cols = nrows // s, ncols // s
        if cell_rows < _MIN_CELLS or cell_cols < _MIN_CELLS:
            continue

        active_r, active_c = cell_rows * s, cell_cols * s
        if not _blocks_uniform(grid, s, active_r, active_c):
            continue
        # Trailing strip must be letterbox: one flat colour, not real content.
        if not _strip_uniform(grid, nrows, ncols, active_r, active_c):
            continue
        # Reject the degenerate case where the whole active region is one colour.
        if _cell_colours(grid, s, active_r, active_c) < 2:
            continue
        return s, cell_rows, cell_cols
    return None


def _blocks_uniform(grid: Any, s: int, active_r: int, active_c: int) -> bool:
    for r in range(active_r):
        anchor_r = (r // s) * s
        row, anchor_row = grid[r], grid[anchor_r]
        for c in range(active_c):
            if row[c] != anchor_row[(c // s) * s]:
                return False
    return True


def _strip_uniform(grid: Any, nrows: int, ncols: int, active_r: int, active_c: int) -> bool:
    seen = set()
    for r in range(nrows):
        for c in range(ncols):
            if r >= active_r or c >= active_c:
                seen.add(grid[r][c])
                if len(seen) > 1:
                    return False
    return True


def _cell_colours(grid: Any, s: int, active_r: int, active_c: int) -> int:
    return len({grid[r][c] for r in range(0, active_r, s) for c in range(0, active_c, s)})


def lattice_line(grid: Any) -> str | None:
    """The one prompt line describing the true board lattice, or None if 1:1."""
    found = detect_lattice(grid)
    if found is None:
        return None
    s, cell_rows, cell_cols = found
    return (
        f"BOARD LATTICE: the displayed {len(grid)}x{len(grid[0])} frame is an upscale of a true "
        f"{cell_rows}x{cell_cols} game grid. Each game cell is an {s}x{s} block of identical "
        f"pixels; game cell (i,j) covers display rows {s}*i..{s}*i+{s - 1} and cols "
        f"{s}*j..{s}*j+{s - 1}. Adjacent cells are {s} apart, not 1 -- a MOUSE click at "
        f"(row, col) and one at (row+1, col) usually hit the SAME game cell. Click cell "
        f"centres, and reason about distances and object sizes in game cells, not display pixels."
    )


# --------------------------------------------------------------------------
# patches
# --------------------------------------------------------------------------

def patch_token_estimator() -> bool:
    from inference.agent import tool_agent as ta

    if getattr(ta._estimate_tokens, "_levers", False):
        return True
    import json as _json

    def _estimate_tokens(value: Any) -> int:
        try:
            rendered = _json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        except TypeError:
            rendered = str(value)
        return max(1, (len(rendered) + 3) // 4)

    _estimate_tokens._levers = True
    ta._estimate_tokens = _estimate_tokens
    return True


def patch_context_window(window: int = 49152) -> bool:
    from inference.agent import tool_agent as ta

    ta._LOCAL_ANALYZER_CONTEXT_WINDOW = int(window)
    return ta._LOCAL_ANALYZER_CONTEXT_WINDOW == int(window)


def patch_lattice_disclosure() -> bool:
    """NOT SHIPPED -- kept with its evidence so the idea is not re-derived.

    The premise is refuted. The 9 upscaled games do NOT render block-constant
    frames: bp35 (nominally 8x8 at scale 8) has top-row colour runs of 31/23/10,
    none a multiple of 8, and only 10 of 63 adjacent row pairs are identical;
    cn04 (scale 3) shows runs of 16/1/31. The games draw fine detail *inside*
    each logical cell, so `detect_lattice` correctly finds nothing on all 25
    games -- zero false positives and zero true positives.

    The underlying phenomenon is real: 279 of 2,511 recorded clicks (11.1%) land
    in the same game cell as the previous click. But that was measured with
    arcengine's Camera.display_to_grid, and in competition mode the environment
    is an HTTP wrapper that returns frames only -- there is no camera object and
    no way to recover the scale without spending actions to probe for it. Any
    revival of this lever needs an action-priced probe, not frame inference.
    """
    from inference.agent import tool_agent as ta

    original = ta.ToolAgent._build_user_prompt
    if getattr(original, "_levers", False):
        return True

    def _build_user_prompt(self, action_num, **kwargs):
        text = original(self, action_num, **kwargs)
        frame = kwargs.get("current_frame")
        grid = getattr(frame, "grid", None)
        if grid is None:
            return text
        try:
            line = lattice_line(grid)
        except Exception:
            # Never let a perception nicety break the turn; the arm-active check
            # below already proved the patch itself is installed.
            return text
        if not line:
            return text
        # Place it directly after the state line so it sits with the other
        # per-turn facts rather than at the end of a long instruction block.
        marker = "\nValid actions right now:"
        if marker in text:
            return text.replace(marker, f"\n{line}{marker}", 1)
        return f"{text}\n{line}"

    _build_user_prompt._levers = True
    ta.ToolAgent._build_user_prompt = _build_user_prompt
    return True


def install_budget_assert(expected_window: int = 49152) -> bool:
    """Prove L2 actually reached the analyzer, once, at the first construction.

    `_context_budget_tokens` is computed in ToolAgent.__init__ from the module
    global, and analyzers are built inside bm.run() -- after this hook cell. So a
    hook-cell check of the module attribute proves the rebind but not the effect.
    This one-shot wrapper closes that gap.

    It RAISES on mismatch rather than warning. Scored-rerun logs are never
    retrievable, so a printed warning would be invisible forever; the only signal
    that reaches us is ERROR-versus-score, and an ERROR costs no submission slot.
    The first analyzer is built within minutes of bm.run(), so this fails fast.
    """
    from inference.agent import tool_agent as ta

    original = ta.ToolAgent.__init__
    if getattr(original, "_levers", False):
        return True
    state = {"checked": False}
    expected_budget = max(1024, int(expected_window) - 512 - 512)

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not state["checked"]:
            state["checked"] = True
            actual = getattr(self, "_context_budget_tokens", None)
            reserve = getattr(self, "_reply_reserve_tokens", None)
            print(
                f"[levers] first analyzer: context_budget_tokens={actual} "
                f"reply_reserve={reserve} (expected budget {expected_budget})",
                flush=True,
            )
            if actual != expected_budget:
                raise RuntimeError(
                    f"[levers] L2 did not reach the analyzer: context_budget_tokens="
                    f"{actual}, expected {expected_budget}. Refusing to score a run "
                    f"that is silently base duck."
                )

    __init__._levers = True
    ta.ToolAgent.__init__ = __init__
    return True


def apply_all(context_window: int = 49152) -> dict[str, bool]:
    """Install the shipped levers.

    L4 (board lattice) is deliberately NOT installed -- see the note on
    `patch_lattice_disclosure`. L3 (hard no-op guard) is not in this file at all;
    see the note below.

    L3 hard no-op guard -- REJECTED 2026-08-01 on measurement, do not re-derive.
        Proposal was to memoise (grid_hash, action) -> no board change and skip
        the engine call, saving a scored action. Measured 6.79% of actions are
        suppressible. It is unsafe: a byte-identical frame does NOT imply
        unchanged state. Direct engine instrumentation found silent internal
        state changes in 13 of 25 games (hidden per-level step budgets, off-frame
        sprite motion, hidden flags). The decisive case is wa30 holding UP into a
        wall: two frame-identical actions decrement a hidden counter to zero and
        flip _state to GAME_OVER. On recorded traffic the naive guard is wrong on
        12.8% of the actions it suppresses, and 5 of 42 replayed episodes
        diverged. A hardened form (N>=4 confirmations + one-strike kill switch)
        does replay clean, but 83.8% of its savings come from ft09 alone -- the
        single dev game with no hidden state -- while scoring happens on 55
        unseen private games. Not worth a silent, unbounded failure mode for ~4%
        fewer actions when score is capped by completed-level share.
    """
    return {
        "L1_token_estimator": patch_token_estimator(),
        "L2_context_window": patch_context_window(context_window),
        "L2_budget_assert": install_budget_assert(context_window),
    }
