"""Synthetic ARC-AGI-3-style game generator (license-independent SFT leg).

Built ONLY on the MIT-licensed public arcengine primitives (Sprite/Level/
Camera/ARCBaseGame) plus this repo's own code. No Schema traces, no Claude
outputs, no external data.

v1 modules (commit 0553b61 — kept intact and reusable):
    families     -- samplers + mechanical reference solvers for 3 mechanic families
    render_game  -- GameSpec -> self-contained single-subclass game .py file
    validate     -- acceptance test: load via arc_agi local_wrapper, replay, determinism
    gen          -- CLI orchestrator (sample -> render -> solve -> validate -> manifest)
    corpus       -- replay reference traces through the real harness observation
                    pipeline into the corpus_v3 JSONL format the LoRA kernel consumes

v2 modules (2026-08-04 — deliberation-preserving targets + revision arcs +
diagnosis-mapped families F2/F4/F5/F7):
    families_v2  -- replay (F2 g50t-class), carry (F4 wa30-class), mirror
                    (F5 m0r0-class), rules (F7 tr87-class): sims, samplers,
                    mechanical solvers
    render_v2    -- game templates for the 4 new families (v1 delegation for
                    nav/click/push)
    episodes     -- scripted-teacher episodes: wrong-hypothesis -> contradiction
                    -> revision -> win, from a per-family confusion library
                    modeled on the duck's actual failures (~25% clean solves)
    narrate      -- long-form programmatic deliberation; length distribution
                    gated against corpus_v3 (mean/P10/P90 +-25%), anti-template
                    phrasing banks
    corpus_v2    -- corpus builder (windowed history, kernel-compatible records)
    validate_v2  -- episode replay gate + corpus gates (length distribution,
                    8-gram overlap, split integrity, kernel encode/cap)
    gen_v2       -- CLI orchestrator for the v2 pipeline
"""
