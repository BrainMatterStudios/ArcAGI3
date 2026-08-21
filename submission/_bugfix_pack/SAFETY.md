# bugfix_pack — safety sheet (2026-08-22)

Graft: `graft_bugfix.py` (house style per `submission/_retry_guard/graft_retry.py`).
Sources: `docs/RESEARCH-2026-08-21-bug-lever-hunt.md` + wave-2 corrections.
Target: June stock harness (the thtennant/jeroencottaar source-share tree that
pack-v22 mounts; the f53fb37a `bundles/june` reference copy was evicted by tmp
cleanup — the wave-2 `srcpull` copy is byte-identical to the original share).
On the Aug-07 anim-lineage tree every seam also exists; `animation_doc` only
fires there (it reports SKIP on June stock).

## Per-patch classification

| # | patch | behavior-affecting? | envelope effect | notes |
|---|-------|---------------------|-----------------|-------|
| 1 | animation_doc | **YES** (prompt text the model reads changes) | none | Text-only; return type untouched. Seam exists only in anim-lineage trees; SKIP on June stock. Corrected text verified against the shipping anim bundle's `build_animation_view` bytecode (`steps` entries = dicts of scalars + preformatted strings). |
| 2 | sandbox_stderr | no (mechanical) | none | Only fires on sandbox host crashes, which today return a constant string; the model then sees the real stderr (300 chars) instead of a blind constant. Strictly more information on an already-error path. |
| 3 | sandbox_imports | no (mechanical) | none | `difflib` added to the sandbox whitelist + the advertised module list. `sys` deliberately NOT added: the sandbox JSON protocol runs on the child's real stdin/stdout (`_recv` reads `sys.stdin`), so user code touching `sys.stdin`/`sys.__stdout__` could consume or corrupt protocol frames — not trivially safe. |
| 4 | runtime_state_cap | no (mechanical)¹ | **shortens** | Measured (this machine, 64x64 frames, best-of-k): write per action 618ms→5.8ms at n=1000, 1245ms→6.0ms at n=2000; state file 64.71/129.37MB→0.51MB; per-`action()` sandbox pipe 18.29/36.58MB→0.91MB; view+dump 569/1170ms→27ms. Reproduces the hunt's 65/129MB figures. ¹The sandbox `history` variable now holds the last 50 entries instead of all — model-visible in principle, but the prompt never promises full depth and deep history is exactly the measured pathology. |
| 5 | analyzer_timeout_cap | no (mechanical) | **shortens** | NOT a blanket 10s cap — that would kill healthy long decodes (non-streaming completions send nothing until decode ends). Health-gated: only after a Timeout/ConnectionError, and only while a 2s `GET /models` probe says the server is dead, retries fail-fast at 10s. Any success or alive-probe restores the full decaying budget. Healthy runs never enter the gated path. |
| 6 | estimator_images | no (mechanical)² | none | `data:` URL image parts priced at 64 tokens + placeholder overhead instead of len/3 (~10-40k chars). ²Indirectly behavior-adjacent: correct pricing keeps MORE history in context (the overcharge was accidentally evicting it) — which is why 7 must ride along. |
| 7 | history_image_strip | **YES** (request content changes) | none | Prior user turns keep text only; every stale board PNG (each captioned "Current grid image:", a measured misgrounding channel) becomes `[frame image for step N omitted]`. The newest turn's image is rebuilt fresh each `analyze()` and always survives. **Must ship with 6** (hunt doc interaction: 6 alone would let more stale images survive trimming). |
| 8 | click_range_reject | **YES** (new error path + prompt sentence) | none | Out-of-range MOUSE row/col now returns an error payload via `_error_payload` — rejected actions cost ZERO engine actions (vs. today's silent clamp to the border, which both spends an action and feeds the model a false probe result). Upscale sentence reads the live `MULTIMODAL_UPSCALE` (4 in shipping arms) at install. |

No patch can extend run duration: 1/2/3/6/7/8 are O(1) text or bookkeeping
changes; 4 and 5 strictly remove measured dead time.

## Recommended flag grouping for the first single-variable smoke

- **Arm A (mechanical-only, safe default):** 2,3,4,5,6 ON; 1,7,8 OFF
  (`BUGFIX_ANIMATION_DOC=0 BUGFIX_HISTORY_IMAGE_STRIP=0 BUGFIX_CLICK_RANGE_REJECT=0`).
  Caveat: this leaves 6 without 7, which per the hunt's interaction note lets
  more stale images survive trimming — acceptable for a SMOKE (reading is
  crash/latency/parity, not score), but do NOT ship A live as-is; either add
  7 or also disable 6 (`BUGFIX_ESTIMATOR_IMAGES=0`) for a live mechanical arm.
- **Arm B (behavioral stage):** A + 7 (with 6), then + 8, then + 1 (1 only
  matters on anim-lineage bundles). Prompt levers have nulled twice at
  27B/3.6 — pre-register, read per-level actions not score.
- Master kill: `BUGFIX_PACK=0`. Per-patch: `BUGFIX_<NAME>=0` (checked at
  install AND, for all callable patches, at call time).

## Fail-open posture (tested)

`test_bugfix_pack.py` (7 pytest cases, all subprocess-isolated): patched
behavior correct on the real June tree (incl. an end-to-end sandbox `difflib`
run); flag-off leaves stock byte-behavior intact; master kill switch; call-time
toggles; every patch target renamed/deformed → install still succeeds with
that patch reported SKIP; animation_doc rewrite verified against the exact
recovered anim-bundle bullet; the timeout gate's four-phase scenario
(healthy → timeout → dead-capped-10s → alive-full-budget → cleared) against a
real local HTTP server.
