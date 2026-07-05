# Kaggle 27B Fine-Tune — Phase Scope (2026-07-05)

**Why we're here:** We reached 1.05 (winning tier) by adopting the field's public method — base **Qwen3.6-27B + duck harness**. Fine-tuning is the one lever *nobody in the field has pulled*. The local PoC validated the **data pipeline** and showed SFT generalizes on **loss (−35.7% held-out)**, but the behavioral eval revealed the **7B proxy can't act** in the harness (base included) — so behavioral validation must move to the **real 27B on the free RTX Pro 6000**, where the base model actually completes levels.

**Goal of this phase:** a fine-tuned 27B that completes **more levels than base 27B** in the duck harness on held-out games — measured behaviorally, not by loss.

---

## Method: RFT-first, RL-stretch

We already learned the hard lesson: **imitation-loss flatters; behavior is what counts.** So the method optimizes behavior directly.

### Primary — Rejection-sampling Fine-Tuning (RFT / STaR / self-improvement)
The tractable, proven-for-agents path:
1. **Generate** — run base 27B in the harness on the games, many seeds/temperatures/passes; save every trajectory + its outcome (levels completed). *(The reproduction already saves `events.jsonl` — this is a config away.)*
2. **Filter** — keep only trajectories from **completed levels / high reward** (the base's own *best* behavior).
3. **Train** — **QLoRA** the 27B on those success trajectories. 4-bit 27B ≈ 14GB + LoRA + activations → **fits 96GB comfortably**.
4. **Eval** — base-27B vs RFT-27B in the harness on **held-out games**, count levels.
5. **Iterate** — the improved model generates better trajectories → repeat (self-improvement loop).

### Stretch — GRPO (policy-gradient RL)
If RFT plateaus: group-relative policy optimization with LoRA, reward = levels/progress. More complex (rollouts + advantages + KL to a reference); single-GPU + harness-in-the-loop feasibility is an open question. Only attempt after RFT shows signal.

---

## Feasibility (RTX Pro 6000, 96GB, free)
- **QLoRA 27B training:** fits easily (4-bit base + adapter). ✅
- **Rollout generation:** vLLM 27B on the same GPU — already proven by the reproduction. ✅
- **Serving the tuned model at eval:** merge LoRA into weights, or serve base+adapter via vLLM. Standard. ✅
- **Per-iteration cost:** generate (hours) + train (~1h) + eval (hours) = a few kernel runs. The 12h/kernel limit means phases run as separate kernels.

---

## The honest ceilings (state these up front)
- **RFT amplifies, it doesn't invent.** It reinforces what the base *already sometimes does* — it **cannot teach games the base never solves** (no successful trajectories to learn from). Expected effect: **more consistent completion of already-solvable games**, not new capabilities.
- **Data scarcity is the core risk.** Only ~25 dev games; Tufa's `re-arc-3` generator is **private**. Many rollouts/seeds per game help, but overfitting to known games is real → **held-out games are the honest test**, and the hidden eval is the final judge.
- **The base is strong.** Beating it may yield modest gains; a winning-level jump is *not* guaranteed and likely needs exploration-RL or better/more data.

**Realistic expectation:** a modest-to-moderate behavioral lift (more reliable levels) that could push 1.05 upward. Anything larger is a stretch that depends on GRPO or solving the data-scarcity problem.

---

## Phases & decision gates
| Phase | Work | Gate to proceed |
|-------|------|-----------------|
| **0 · Generate** | Base 27B rollouts on dev games, save trajectories + outcomes | ≥ a few hundred success trajectories collected |
| **1 · RFT round 1** | Filter successes → QLoRA 27B → behavioral eval (base vs tuned, held-out) | **Tuned completes ≥ base levels on held-out games** |
| **2 · Iterate** | Self-improvement loop (2–3 rounds) | Monotone held-out gain |
| **3 · Submit** | Merge + submit; measure hidden score | **Hidden score > 1.05** |
| **4 · Stretch** | GRPO if RFT plateaus | RFT clearly capped |

**Non-negotiables (lessons banked):**
- Evaluate **behaviorally on held-out games**, never on training loss (loss lied once — not again).
- Keep the **1.05 base-27B build as the floor**; the fine-tune is additive/replaceable, never a regression.
- Anything "great on dev" gets a **hidden-eval confirm** before we believe it (efficiency-first and best-of-3 both cratered live).

---

## Assets in hand
- Trajectory extractor + SFT pipeline (`scratchpad/extract_sft.py` v2, clean single-wrap).
- The harness runs locally + on Kaggle (`submission/_adopt/taaf-src`); reproduction kernel proven at dev-mean 1.78.
- Free RTX Pro 6000 confirmed; vLLM + QLoRA both fit 96GB.
- Held-out game split (ft09, re86, sb26, sc25, tu93) established.
