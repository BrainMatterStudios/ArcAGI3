"""Synthetic ARC-AGI-3-style game generator (license-independent SFT leg).

Built ONLY on the MIT-licensed public arcengine primitives (Sprite/Level/
Camera/ARCBaseGame) plus this repo's own code. No Schema traces, no Claude
outputs, no external data.

Modules:
    families     -- samplers + mechanical reference solvers for 3 mechanic families
    render_game  -- GameSpec -> self-contained single-subclass game .py file
    validate     -- acceptance test: load via arc_agi local_wrapper, replay, determinism
    gen          -- CLI orchestrator (sample -> render -> solve -> validate -> manifest)
    corpus       -- replay reference traces through the real harness observation
                    pipeline into the corpus_v3 JSONL format the LoRA kernel consumes
"""
