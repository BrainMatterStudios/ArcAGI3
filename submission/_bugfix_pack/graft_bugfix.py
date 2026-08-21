"""Bugfix-pack graft — eight mechanical fixes from the 08-21 bug-lever hunts.

Sources: docs/RESEARCH-2026-08-21-bug-lever-hunt.md (Tier-1 #3 tool-API
friction, Tier-2 #5 analyzer timeout, Tier-2 #7 stale-image history, Tier-2 #8
runtime-state O(n^2) + stderr constant) and the wave-2 corrections doc.

Target tree: the June stock harness (jeroencottaar/taaf-kaggle-source-share ==
thtennant/taaf-kaggle-source-share-fork, the dataset pack-v22 mounts; the
original f53fb37a bundles/june reference copy was evicted by tmp cleanup —
the srcpull fork copy is byte-identical to the original share and is the tree
all file:line cites below refer to). One patch (animation_doc) has its seam
only in the Aug-07 anim-lineage bundle (v12-lane arms); on June stock it
reports SKIP (seam absent) — the pack is safe to install on either tree.

Patches (one flag, one function each; env BUGFIX_<NAME>=0 disables one,
BUGFIX_PACK=0 disables all):

1. animation_doc — the anim-bundle prompt bullet promises `animation()`
   timeline entries "with `changed`, `bbox`, and either `changes` ..." in a
   shape models read as nested dicts (30x `'str' object has no attribute
   'get'` across >=8 runs). Lower-risk side taken: correct the PROMPT text to
   the true format (entries are dicts of scalars/preformatted STRINGS; view
   key is `steps`) with a one-line example; the return type is untouched.
   Ground truth recovered from the shipping anim bundle's animation.py
   (build_animation_view returns {**header, 'unique_frames', 'board_unchanged',
   'steps': [...], 'note', ['omitted_steps']}; each step entry is
   {'step': int, 'changed': int-cell-count, 'bbox': 'rows a-b, cols c-d',
   ['held_for_frames': int], and 'changes': [str,...] OR 'transitions':
   {str:int}}).
2. sandbox_stderr — python_tool_sandbox.py:401-404 `_sanitize_host_error_text`
   returns the same constant in BOTH branches; the real crash stderr (read at
   :516, used at :519) is discarded. Fix: append the first 300 chars of
   stderr when present.
3. sandbox_imports — whitelist `difflib` in the sandbox import policy
   (python_tool_sandbox.py:42-57 SAFE_MODULES inside the _SANDBOX_BOOTSTRAP
   string) and in the prompt line advertising the module list
   (prompts.py:81, PYTHON_ADDENDUM). `sys` is NOT whitelisted: the sandbox
   JSON protocol runs over the child's real stdin/stdout
   (python_tool_sandbox.py:115-124) and user code touching `sys.stdin` /
   `sys.__stdout__` would consume or corrupt protocol frames — not
   trivially safe.
4. runtime_state_cap — runtime_state.py:122-135 write_runtime_state rewrites
   the FULL unbounded history to disk with indent=2 EVERY action, and
   tool_agent.py:442-449 _ascii_history_view_payload re-ships the full
   history into the sandbox per action() call (65/129MB files at
   n=1000/2000 per the hunt). Fix: cap both to the last 50 entries and drop
   indent=2. Dual-namespace: solver.py:31-35 imports write_runtime_state BY
   NAME, so both bindings are patched.
5. analyzer_timeout_cap — solver.py:227-244 request_timeout_seconds() feeds
   the decaying remaining-run budget straight into requests.post(timeout=...)
   at tool_agent.py:1302-1308; against a hung server one attempt blocks up to
   the full decayed deadline (observed 289s; ~2,800s over 3 turns). A blanket
   10s cap would kill healthy long decodes (a non-streaming chat completion
   sends nothing until decode finishes), so the cap is HEALTH-GATED: after a
   Timeout/ConnectionError, each retry first probes {base_url}/models (2s);
   probe dead -> the retry's timeout is capped at 10s (fast-fail); probe
   alive -> full budget (transient hiccup, decode allowed). Any success
   clears the state. Healthy runs never enter the gated path.
6. estimator_images — tool_agent.py:462-467 _estimate_tokens prices
   everything at len(json)/3, so one upscale-4 data: PNG (~10-40k chars)
   is billed as thousands of tokens when its true vision cost is ~64
   (June-fork line; the hunt doc's anim-tree cite is tool_agent.py:484).
   Fix: price each data: URL image part at a fixed 64-token placeholder plus
   the placeholder's own JSON overhead; text pricing unchanged.
7. history_image_strip — tool_agent.py:1147-1158 attaches the current-grid
   PNG to every user turn and tool_agent.py:2017 persists those turns
   verbatim, so ~20-30 STALE board images captioned "Current grid image:"
   ride along in every request (hunt Tier-2 #7). Fix: rewrite PRIOR user
   messages to text-only inside _persistent_history_messages
   (tool_agent.py:1653-1670), replacing each image part with
   '[frame image for step N omitted]' (N parsed from the turn's own
   "Current state: step N" line); only the newest turn — rebuilt fresh with
   its image by _build_user_message each analyze() — keeps an image.
   INTERACTION (hunt doc): the len/3 overcharge accidentally evicts old
   images early, so 6 and 7 MUST ship together — both default ON here.
8. click_range_reject — solver.py:530-531 silently clamps MOUSE row/col into
   0..63 (`max(0, min(63, int(...)))`), so a hallucinated (200, 340) click
   lands on the border and reads as a real probe result. Fix: pre-validate in
   _HarnessGameSession._normalize_actions (solver.py:491-542) and return an
   error payload instead (rejected actions go out via _error_payload,
   solver.py:544-550, and cost ZERO engine actions); plus one sentence in the
   multimodal prompt addendum (prompts.py:69-74) stating the image is an
   Nx upscale (N read from vision_context.current_grid_image_upscale(), 4 in
   the shipping arms) and coords are 0-63 grid units.

House conventions (per submission/_retry_guard/graft_retry.py): monkey-patch
at import, presence gates on every seam symbol, each patch fail-open behind
its own flag, install() returns a status string (one line per patch), blanket
try/except guards, FLAGS dict defaulting every patch ON individually.

Fail-open invariants:
- install() never raises; every patch reports OK / SKIP(reason) / FAIL(exc)
  and a failed patch leaves stock behavior in place.
- Wrappers never wrap the inner/original call in a swallowing except: an
  inner crash propagates exactly as stock. Guard logic around it sits in
  blanket try/except that falls back to the stock result/path.
- Behavior with a flag off at install time == stock, byte-for-byte, for the
  string-constant patches (1, 3-prompt-half, 8-prompt-half); the callable
  patches (2, 4, 5, 6, 7, 8-reject) additionally honor the env flag at CALL
  time, so a shipped-but-disabled wrapper is a pure pass-through.
- Envelope: no patch extends run duration; 4 and 5 strictly shorten
  (smaller serialization, bounded dead blocking).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

# ---------------------------------------------------------------------------
# Flags — every patch defaults ON; env BUGFIX_<NAME>=0 turns one off,
# BUGFIX_PACK=0 turns the whole pack off.
# ---------------------------------------------------------------------------

FLAGS: dict[str, bool] = {
    "animation_doc": True,
    "sandbox_stderr": True,
    "sandbox_imports": True,
    "runtime_state_cap": True,
    "analyzer_timeout_cap": True,
    "estimator_images": True,
    "history_image_strip": True,
    "click_range_reject": True,
}

_FALSY = {"0", "false", "False", "no", "off"}

RUNTIME_STATE_HISTORY_CAP = 50
ANALYZER_FAST_FAIL_TIMEOUT_S = 10.0
ANALYZER_PROBE_TIMEOUT_S = 2.0
IMAGE_TOKEN_ESTIMATE = 64
SANDBOX_STDERR_CHARS = 300


def _pack_enabled() -> bool:
    return os.environ.get("BUGFIX_PACK", "1").strip() not in _FALSY


def _enabled(name: str) -> bool:
    if not _pack_enabled():
        return False
    raw = os.environ.get(f"BUGFIX_{name.upper()}", "").strip()
    if raw:
        return raw not in _FALSY
    return bool(FLAGS.get(name, True))


# ---------------------------------------------------------------------------
# Patch 1 — animation_doc (anim-lineage seam; SKIP on June stock)
# ---------------------------------------------------------------------------

# Exact bullet shipped in the anim bundle's prompts.py (recovered verbatim
# from the shipping pyc; byte-exact match is the presence gate).
_ANIM_DOC_OLD = (
    "- `animation()` returns a compact diff timeline of the frames the last "
    "animated action produced: one entry per distinct frame with `changed`, "
    "`bbox`, and either `changes` (cells as `'W>R @ (row,col) (row,col)'`) "
    "or a `transitions` census when too many cells changed. Identical "
    "consecutive frames are collapsed into `held_for_frames`.\n"
)

_ANIM_DOC_NEW = (
    "- `animation()` returns a dict whose `steps` list holds one entry per "
    "distinct frame. Every field is a scalar or a preformatted STRING: "
    "`changed` is an integer cell count, `bbox` is a string like "
    "`'rows 2-4, cols 1-3'`, and `changes` is a list of plain strings like "
    "`'W>R @ (2,3) (2,4)'` (when too many cells changed you instead get a "
    "`transitions` dict counting `'W>R'`-style keys). One-line example "
    "entry: `{'step': 1, 'changed': 2, 'bbox': 'rows 2-4, cols 1-3', "
    "'changes': ['W>R @ (2,3) (2,4)']}` -- the `bbox`/`changes` values are "
    "strings, not objects, so never call `.get(...)` on them. Identical "
    "consecutive frames are collapsed into `held_for_frames`.\n"
)


def _patch_module_strings(modules: list[Any], old: str, new: str) -> int:
    """Replace `old` with `new` in every module-level str attribute that
    contains it, across all given namespaces (prompt constants are imported
    BY NAME into tool_agent, so both bindings must move together)."""
    replaced = 0
    for mod in modules:
        for attr in dir(mod):
            if attr.startswith("__"):
                continue
            value = getattr(mod, attr, None)
            if isinstance(value, str) and old in value:
                setattr(mod, attr, value.replace(old, new))
                replaced += 1
    return replaced


def patch_animation_doc() -> str:
    if not _enabled("animation_doc"):
        return "animation_doc: SKIP (flag off)"
    try:
        from inference.agent import prompts as prompts_mod
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"animation_doc: SKIP (modules missing: {exc!r})"
    try:
        if any(
            isinstance(getattr(m, a, None), str)
            and _ANIM_DOC_NEW in getattr(m, a)
            for m in (prompts_mod, agent_mod)
            for a in dir(m)
            if not a.startswith("__")
        ):
            return "animation_doc: SKIP (already applied)"
        replaced = _patch_module_strings(
            [prompts_mod, agent_mod], _ANIM_DOC_OLD, _ANIM_DOC_NEW
        )
        if replaced == 0:
            return "animation_doc: SKIP (seam absent — June stock has no animation() doc)"
        return (
            f"animation_doc: OK — prompt now shows the true string format "
            f"with an example ({replaced} binding(s) rewritten)"
        )
    except Exception as exc:  # noqa: BLE001
        return f"animation_doc: FAIL-OPEN ({exc!r})"


# ---------------------------------------------------------------------------
# Patch 2 — sandbox_stderr
# ---------------------------------------------------------------------------

def patch_sandbox_stderr() -> str:
    if not _enabled("sandbox_stderr"):
        return "sandbox_stderr: SKIP (flag off)"
    try:
        from inference.agent import python_tool_sandbox as sandbox_mod
    except Exception as exc:  # noqa: BLE001
        return f"sandbox_stderr: SKIP (module missing: {exc!r})"
    original = getattr(sandbox_mod, "_sanitize_host_error_text", None)
    if original is None:
        return "sandbox_stderr: SKIP (missing _sanitize_host_error_text)"
    if getattr(original, "_bugfix_patched", False):
        return "sandbox_stderr: SKIP (already applied)"

    def sanitize_with_stderr(text: str) -> str:
        base = "Sandbox process exited unexpectedly."
        try:
            if not _enabled("sandbox_stderr"):
                return original(text)
            snippet = str(text or "").strip()
            if not snippet:
                return base
            return (
                f"{base} stderr (first {SANDBOX_STDERR_CHARS} chars): "
                f"{snippet[:SANDBOX_STDERR_CHARS]}"
            )
        except Exception:  # noqa: BLE001 — fall back to the stock constant
            return base

    sanitize_with_stderr._bugfix_patched = True  # type: ignore[attr-defined]
    # Single-namespace: only run_sandboxed_python (same module, resolved from
    # module globals at call time) references it.
    sandbox_mod._sanitize_host_error_text = sanitize_with_stderr
    return "sandbox_stderr: OK — real stderr (300 chars) surfaces on sandbox host crashes"


# ---------------------------------------------------------------------------
# Patch 3 — sandbox_imports
# ---------------------------------------------------------------------------

_SAFE_MODULES_MARKER = '"bisect",'
_SAFE_MODULES_PATCHED = '"bisect",\n        "difflib",'
_PROMPT_MODULES_OLD = "bisect, collections, copy, fractions"
_PROMPT_MODULES_NEW = "bisect, collections, copy, difflib, fractions"


def patch_sandbox_imports() -> str:
    if not _enabled("sandbox_imports"):
        return "sandbox_imports: SKIP (flag off)"
    try:
        from inference.agent import python_tool_sandbox as sandbox_mod
    except Exception as exc:  # noqa: BLE001
        return f"sandbox_imports: SKIP (module missing: {exc!r})"
    bootstrap = getattr(sandbox_mod, "_SANDBOX_BOOTSTRAP", None)
    if not isinstance(bootstrap, str):
        return "sandbox_imports: SKIP (missing _SANDBOX_BOOTSTRAP)"
    if '"difflib",' in bootstrap:
        return "sandbox_imports: SKIP (already applied)"
    if _SAFE_MODULES_MARKER not in bootstrap:
        return "sandbox_imports: SKIP (SAFE_MODULES marker absent — seam moved)"
    try:
        sandbox_mod._SANDBOX_BOOTSTRAP = bootstrap.replace(
            _SAFE_MODULES_MARKER, _SAFE_MODULES_PATCHED, 1
        )
        # Keep the advertised module list truthful (prompts + the by-name
        # copies inside tool_agent). Best-effort: the whitelist itself is the
        # functional half.
        prompt_note = ""
        try:
            from inference.agent import prompts as prompts_mod
            from inference.agent import tool_agent as agent_mod

            replaced = _patch_module_strings(
                [prompts_mod, agent_mod], _PROMPT_MODULES_OLD, _PROMPT_MODULES_NEW
            )
            prompt_note = f", prompt list updated in {replaced} binding(s)"
        except Exception:  # noqa: BLE001
            prompt_note = ", prompt list unchanged (seam absent)"
        return (
            "sandbox_imports: OK — difflib whitelisted in sandbox"
            f"{prompt_note} (sys stays blocked: sandbox stdio protocol)"
        )
    except Exception as exc:  # noqa: BLE001
        return f"sandbox_imports: FAIL-OPEN ({exc!r})"


# ---------------------------------------------------------------------------
# Patch 4 — runtime_state_cap
# ---------------------------------------------------------------------------

def patch_runtime_state_cap() -> str:
    if not _enabled("runtime_state_cap"):
        return "runtime_state_cap: SKIP (flag off)"
    try:
        from inference.agent import runtime_state as state_mod
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"runtime_state_cap: SKIP (modules missing: {exc!r})"
    original_write = getattr(state_mod, "write_runtime_state", None)
    original_view = getattr(agent_mod, "_ascii_history_view_payload", None)
    if original_write is None:
        return "runtime_state_cap: SKIP (missing write_runtime_state)"
    if original_view is None:
        return "runtime_state_cap: SKIP (missing _ascii_history_view_payload)"
    if getattr(original_write, "_bugfix_patched", False):
        return "runtime_state_cap: SKIP (already applied)"
    if not hasattr(state_mod, "history_entry_to_payload") or not hasattr(
        state_mod, "frame_to_payload"
    ):
        return "runtime_state_cap: SKIP (payload helpers missing — seam moved)"

    def write_runtime_state_capped(path, *, current_frame, history):  # type: ignore[no-untyped-def]
        try:
            if not _enabled("runtime_state_cap"):
                return original_write(
                    path, current_frame=current_frame, history=history
                )
            capped = list(history)[-RUNTIME_STATE_HISTORY_CAP:]
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "current_frame": state_mod.frame_to_payload(current_frame),
                "history": [
                    state_mod.history_entry_to_payload(entry) for entry in capped
                ],
            }
            tmp_path = path.with_suffix(f"{path.suffix}.tmp")
            tmp_path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            tmp_path.replace(path)
            return None
        except Exception:  # noqa: BLE001 — fall back to the stock writer
            return original_write(path, current_frame=current_frame, history=history)

    def ascii_history_view_capped(history_entries):  # type: ignore[no-untyped-def]
        try:
            if _enabled("runtime_state_cap"):
                history_entries = list(history_entries)[-RUNTIME_STATE_HISTORY_CAP:]
        except Exception:  # noqa: BLE001
            pass
        return original_view(history_entries)

    write_runtime_state_capped._bugfix_patched = True  # type: ignore[attr-defined]
    ascii_history_view_capped._bugfix_patched = True  # type: ignore[attr-defined]
    state_mod.write_runtime_state = write_runtime_state_capped
    agent_mod._ascii_history_view_payload = ascii_history_view_capped
    # Dual-namespace: solver.py:31-35 imports write_runtime_state BY NAME.
    note = ""
    try:
        from inference.framework import solver as solver_mod

        solver_binding = getattr(solver_mod, "write_runtime_state", None)
        if solver_binding is original_write:
            solver_mod.write_runtime_state = write_runtime_state_capped
        elif getattr(solver_binding, "_bugfix_patched", False):
            pass  # solver imported after the rebind: by-name binding is already the wrapper
        else:
            note = " (solver binding not stock — left untouched)"
    except Exception:  # noqa: BLE001
        note = " (solver namespace unavailable)"
    return (
        f"runtime_state_cap: OK — serialized history capped at last "
        f"{RUNTIME_STATE_HISTORY_CAP} entries, indent dropped{note}"
    )


# ---------------------------------------------------------------------------
# Patch 5 — analyzer_timeout_cap
# ---------------------------------------------------------------------------

def _probe_alive(base_url: str, headers: dict[str, str]) -> bool:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return False
    req = urllib.request.Request(f"{base}/models", headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=ANALYZER_PROBE_TIMEOUT_S) as resp:
            status = int(getattr(resp, "status", None) or resp.getcode())
            return 200 <= status < 300
    except Exception:  # noqa: BLE001 — any probe failure means "dead"
        return False


def patch_analyzer_timeout_cap() -> str:
    if not _enabled("analyzer_timeout_cap"):
        return "analyzer_timeout_cap: SKIP (flag off)"
    try:
        from inference.agent import tool_agent as agent_mod
        import requests  # noqa: PLC0415 — same dependency the target uses
    except Exception as exc:  # noqa: BLE001
        return f"analyzer_timeout_cap: SKIP (modules missing: {exc!r})"
    agent_cls = getattr(agent_mod, "ToolAgent", None)
    original = getattr(agent_cls, "_chat_completion", None)
    if original is None:
        return "analyzer_timeout_cap: SKIP (missing ToolAgent._chat_completion)"
    if getattr(original, "_bugfix_patched", False):
        return "analyzer_timeout_cap: SKIP (already applied)"

    def chat_completion_gated(self, messages, *, tools, request_timeout_seconds=None):  # type: ignore[no-untyped-def]
        effective = request_timeout_seconds
        try:
            if _enabled("analyzer_timeout_cap") and getattr(
                self, "_bugfix_last_request_timed_out", False
            ):
                base_url = getattr(getattr(self, "_model", None), "base_url", "")
                try:
                    headers = self._headers()
                except Exception:  # noqa: BLE001
                    headers = {}
                if not _probe_alive(base_url, headers):
                    # Server demonstrably dead: fail this retry fast instead
                    # of blocking the full decaying budget.
                    if effective is None:
                        effective = ANALYZER_FAST_FAIL_TIMEOUT_S
                    else:
                        effective = min(
                            float(effective), ANALYZER_FAST_FAIL_TIMEOUT_S
                        )
        except Exception:  # noqa: BLE001 — guard must never break the request
            effective = request_timeout_seconds
        try:
            result = original(
                self, messages, tools=tools, request_timeout_seconds=effective
            )
        except (requests.Timeout, requests.ConnectionError):
            try:
                self._bugfix_last_request_timed_out = True
            except Exception:  # noqa: BLE001
                pass
            raise
        try:
            self._bugfix_last_request_timed_out = False
        except Exception:  # noqa: BLE001
            pass
        return result

    chat_completion_gated._bugfix_patched = True  # type: ignore[attr-defined]
    agent_cls._chat_completion = chat_completion_gated
    return (
        "analyzer_timeout_cap: OK — retries after a timeout probe /models and "
        f"fail-fast at {ANALYZER_FAST_FAIL_TIMEOUT_S:.0f}s while the server is dead"
    )


# ---------------------------------------------------------------------------
# Patch 6 — estimator_images
# ---------------------------------------------------------------------------

_IMAGE_PLACEHOLDER = "<data-url-image>"


def _strip_data_url_images(value: Any, counter: list[int]) -> Any:
    """Deep-copy `value` with every data: URL string swapped for a short
    placeholder, counting how many were replaced."""
    if isinstance(value, str):
        if value.startswith("data:image/"):
            counter[0] += 1
            return _IMAGE_PLACEHOLDER
        return value
    if isinstance(value, dict):
        return {k: _strip_data_url_images(v, counter) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_data_url_images(v, counter) for v in value]
    return value


def patch_estimator_images() -> str:
    if not _enabled("estimator_images"):
        return "estimator_images: SKIP (flag off)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"estimator_images: SKIP (module missing: {exc!r})"
    original = getattr(agent_mod, "_estimate_tokens", None)
    if original is None:
        return "estimator_images: SKIP (missing _estimate_tokens)"
    if getattr(original, "_bugfix_patched", False):
        return "estimator_images: SKIP (already applied)"

    def estimate_tokens_image_aware(value: Any) -> int:
        try:
            if not _enabled("estimator_images"):
                return original(value)
            counter = [0]
            stripped = _strip_data_url_images(value, counter)
            if counter[0] == 0:
                return original(value)
            return original(stripped) + IMAGE_TOKEN_ESTIMATE * counter[0]
        except Exception:  # noqa: BLE001 — fall back to the stock estimator
            return original(value)

    estimate_tokens_image_aware._bugfix_patched = True  # type: ignore[attr-defined]
    # Single-namespace: only tool_agent._estimate_request_input_tokens
    # (tool_agent.py:1596-1606) calls it, via module globals.
    agent_mod._estimate_tokens = estimate_tokens_image_aware
    return (
        f"estimator_images: OK — data: URL image parts priced at "
        f"{IMAGE_TOKEN_ESTIMATE} tokens + placeholder overhead instead of len/3"
    )


# ---------------------------------------------------------------------------
# Patch 7 — history_image_strip
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"Current state: step (\d+)")


def _strip_images_from_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for message in history:
        content = message.get("content") if isinstance(message, dict) else None
        if (
            not isinstance(message, dict)
            or str(message.get("role", "")).strip() != "user"
            or not isinstance(content, list)
        ):
            stripped.append(message)
            continue
        step_label = "unknown"
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                match = _STEP_RE.search(str(part.get("text", "")))
                if match:
                    step_label = match.group(1)
                    break
        new_content: list[Any] = []
        changed = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                new_content.append(
                    {
                        "type": "text",
                        "text": f"[frame image for step {step_label} omitted]",
                    }
                )
                changed = True
            else:
                new_content.append(part)
        if changed:
            stripped.append({**message, "content": new_content})
        else:
            stripped.append(message)
    return stripped


def patch_history_image_strip() -> str:
    if not _enabled("history_image_strip"):
        return "history_image_strip: SKIP (flag off)"
    try:
        from inference.agent import tool_agent as agent_mod
    except Exception as exc:  # noqa: BLE001
        return f"history_image_strip: SKIP (module missing: {exc!r})"
    agent_cls = getattr(agent_mod, "ToolAgent", None)
    original = getattr(agent_cls, "_persistent_history_messages", None)
    if original is None:
        return "history_image_strip: SKIP (missing ToolAgent._persistent_history_messages)"
    if getattr(original, "_bugfix_patched", False):
        return "history_image_strip: SKIP (already applied)"

    def persistent_history_text_only(self, messages, *, tools=None):  # type: ignore[no-untyped-def]
        # Never guard the inner call: a crash propagates exactly as stock.
        history = original(self, messages, tools=tools)
        try:
            if not _enabled("history_image_strip") or not isinstance(history, list):
                return history
            # Everything returned here is prepended as PRIOR history on the
            # next analyze(); that turn's own image is rebuilt fresh by
            # _build_user_message, so stripping all images here leaves
            # exactly the newest turn with an image.
            return _strip_images_from_history(history)
        except Exception:  # noqa: BLE001 — fall back to the stock history
            return history

    persistent_history_text_only._bugfix_patched = True  # type: ignore[attr-defined]
    agent_cls._persistent_history_messages = persistent_history_text_only
    return (
        "history_image_strip: OK — prior user turns keep text only; each "
        "dropped image becomes '[frame image for step N omitted]'"
    )


# ---------------------------------------------------------------------------
# Patch 8 — click_range_reject
# ---------------------------------------------------------------------------

def _mouse_range_error(arguments: dict[str, Any], to_engine_action) -> str | None:  # type: ignore[no-untyped-def]
    """Return an error string when any MOUSE action carries integer row/col
    outside 0..63; None means 'no objection, run the stock path'."""
    if bool(str(arguments.get("action", "")).strip()) and arguments.get("actions") is None:
        raw_actions: Any = [
            {
                "action": arguments.get("action"),
                "row": arguments.get("row"),
                "col": arguments.get("col"),
            }
        ]
    else:
        raw_actions = arguments.get("actions")
    if not isinstance(raw_actions, list):
        return None
    for index, raw_action in enumerate(raw_actions, start=1):
        if not isinstance(raw_action, dict):
            continue
        if to_engine_action(raw_action.get("action")) != "ACTION6":
            continue
        for field in ("row", "col"):
            try:
                coord = int(raw_action[field])
            except (KeyError, TypeError, ValueError):
                continue  # missing/non-int: stock path emits its own error
            if coord < 0 or coord > 63:
                return (
                    f"MOUSE action at index {index} was rejected: {field}={coord} "
                    "is outside the board. The board is 64x64; `row` and `col` "
                    "must be integers 0-63 in GRID units (the attached image is "
                    "an upscale — do not use image pixel coordinates). "
                    "No action was executed and no action budget was spent."
                )
    return None


_UPSCALE_SENTENCE_MARKER = "- The image and `current_frame.ascii` are two representations of the same current frame.\n"


def patch_click_range_reject() -> str:
    if not _enabled("click_range_reject"):
        return "click_range_reject: SKIP (flag off)"
    try:
        from inference.framework import solver as solver_mod
    except Exception as exc:  # noqa: BLE001
        return f"click_range_reject: SKIP (solver module missing: {exc!r})"
    session_cls = getattr(solver_mod, "_HarnessGameSession", None)
    original = getattr(session_cls, "_normalize_actions", None)
    to_engine_action = getattr(solver_mod, "to_engine_action", None)
    if original is None or to_engine_action is None:
        return "click_range_reject: SKIP (missing _normalize_actions/to_engine_action)"
    if getattr(original, "_bugfix_patched", False):
        return "click_range_reject: SKIP (already applied)"

    def normalize_actions_with_range_check(self, arguments):  # type: ignore[no-untyped-def]
        try:
            if _enabled("click_range_reject") and isinstance(arguments, dict):
                error = _mouse_range_error(arguments, to_engine_action)
                if error is not None:
                    return None, error
        except Exception:  # noqa: BLE001 — fall through to the stock path
            pass
        return original(self, arguments)

    normalize_actions_with_range_check._bugfix_patched = True  # type: ignore[attr-defined]
    session_cls._normalize_actions = normalize_actions_with_range_check

    # Prompt half: state the upscale factor + grid units in the multimodal
    # addendum (both the prompts module and tool_agent's by-name copy).
    prompt_note = ", prompt sentence not added (marker absent)"
    try:
        from inference.agent import prompts as prompts_mod
        from inference.agent import tool_agent as agent_mod
        from inference.agent import vision_context as vision_mod

        try:
            scale = int(vision_mod.current_grid_image_upscale())
        except Exception:  # noqa: BLE001
            scale = 4
        sentence = (
            f"- The attached image is a {scale}x upscale of the 64x64 board; "
            "MOUSE `row`/`col` are GRID units 0-63, never image pixels.\n"
        )
        if any(
            isinstance(getattr(m, a, None), str) and sentence in getattr(m, a)
            for m in (prompts_mod, agent_mod)
            for a in dir(m)
            if not a.startswith("__")
        ):
            prompt_note = ", prompt sentence already present"
        else:
            replaced = _patch_module_strings(
                [prompts_mod, agent_mod],
                _UPSCALE_SENTENCE_MARKER,
                _UPSCALE_SENTENCE_MARKER + sentence,
            )
            if replaced:
                prompt_note = f", upscale sentence added in {replaced} binding(s)"
    except Exception:  # noqa: BLE001
        pass
    return (
        "click_range_reject: OK — out-of-range MOUSE coords now return an "
        f"error payload (zero engine actions){prompt_note}"
    )


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

_PATCHES = (
    ("animation_doc", patch_animation_doc),
    ("sandbox_stderr", patch_sandbox_stderr),
    ("sandbox_imports", patch_sandbox_imports),
    ("runtime_state_cap", patch_runtime_state_cap),
    ("analyzer_timeout_cap", patch_analyzer_timeout_cap),
    ("estimator_images", patch_estimator_images),
    ("history_image_strip", patch_history_image_strip),
    ("click_range_reject", patch_click_range_reject),
)


def install() -> str:
    if not _pack_enabled():
        return "bugfix_pack: SKIP (BUGFIX_PACK=0)"
    lines = ["bugfix_pack:"]
    for name, patch in _PATCHES:
        try:
            lines.append(f"  {patch()}")
        except Exception as exc:  # noqa: BLE001 — a patch must never sink install
            lines.append(f"  {name}: FAIL-OPEN ({exc!r})")
    return "\n".join(lines)
