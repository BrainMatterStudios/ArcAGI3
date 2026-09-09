# PRE-REGISTRATION — the model axis (test-time training), written 2026-09-09 BEFORE any corpus exists

## Why this lane, and why now

Every behavioural, memory, budget and tooling lever is now dead on this model:
A1 carry (+0.71 sd), A2 workspace (0 uptake in 358 calls), NOOA (clock contract), Polyphony
(one verified model = 79 % of a game's decode), and 09-09 CADENCE (ENGAGED every gate, −2.28 sd).
The KV10 elasticity (+47 % calls → +19 % levels) closes the throughput family as a source of a step.
Six independent deaths on behaviour-shaping is the signature of a **capability ceiling**, not a
harness defect.

The model axis is the only major lane the plan opened and never closed. **No fine-tune has ever been
served AND scored in this campaign** — every previous attempt died at serving, not at training
([[arcagi3-silent-serving-failures-2026-07-31]]: "an env toggle is not a shipped arm").

Field support and counter-evidence, both recorded: the largest attributed, reproducible gain anyone
has published in this competition is a rejection-sampling LoRA on winning trajectories, **1.25 → 1.94**
(rank ~300 → ~133), with the author's caveats that curation ≫ quantity and that offline levels do not
predict the leaderboard. Against it: our own 08-29 review called the model axis unsupported, our
distillation probe measured imitation compressing deliberation −18.9 %, and one competitor reports a
LoRA backfiring. **Prior is genuinely mixed. This is a gate sequence, not a plan to ship.**

## The corpus (STaR / rejection sampling on our OWN trajectories)

Source: `offkaggle/results/*/transcripts/*.txt`. Each transcript carries, per model call,
`[SYSTEM PROMPT]`, `[USER PROMPT]`, `[THINKING]`, optional `[ASSISTANT]`, and `[MODEL RESPONSE META]`
whose `raw_tool_calls` block holds the emitted python verbatim. A training record is therefore
**(system + user) → (reasoning + tool call)**, exactly the shape the model emits live.

**Selection rule (the rejection-sampling step), fixed now:** keep a call **only if it was made while
on a level that the run went on to CLEAR**. Calls on the wall level — the level the run died on — are
the failures and are discarded. This is ~37.5 % of calls by the measured action split (62.5 % of
actions land on never-cleared levels).

## THE RISK THAT DECIDES THE DESIGN

All 25 games we can train on are the **public** set. The hidden 110 are different games, and this
campaign's own standing law is that **public-25 does not predict the leaderboard**. Training on public
games could buy public-set performance that does not transfer, which is the single most likely way
this lane produces a false positive.

**Mitigation, binding:** a **held-out game split**. Train on 20 games, never showing the model a
single transcript from the other 5. Every read below is on the **held-out** games first. A gain that
appears only on trained games is a memorisation artifact and kills the arm.

## GATES, in order. Each is fatal; none may be skipped or reordered.

**M0 — CORPUS.** Built, deduplicated, counted, with the train/held-out split recorded by game id and
committed before training. Report: records, tokens, games, levels covered, and the exact holdout list.
Fail if fewer than ~2,000 records survive selection (too thin for a useful LoRA).

**M1 — TRAINING.** A LoRA trains to completion on Flash-Next with loss decreasing and no NaN. An
adapter file exists. (This gate has passed before; it is not where the lane dies.)

**M2 — THE SERVING GATE. This is where every previous fine-tune died.** The merged + re-quantised
model must actually serve on Modal and produce coherent, tool-calling output at the keith V14 regime.
**Known hard part: the base is NVFP4.** The proven recipe (`fp8_scaled_linear_inplace`) is FP8;
NVFP4 merge-and-requantise is NOT proven here and may be the real blocker. If M2 cannot be passed, the
lane is dead for engineering reasons and must be reported as such — **not** as evidence about training.

**M3 — HELD-OUT RIG READ (the honest one).** A wave on the **5 held-out games only**, adapter vs base,
same geometry, same seed discipline. Pre-registered: the adapter must be **≥ base on held-out levels**.
Below base on held-out = DEAD regardless of anything else.

**M4 — FULL RIG WAVE.** 25 games vs pooled base **39.33, sd 2.34**: **≥ 48 step candidate / 45–47
redraw / ≤ 44 dead** — the same band every other arm was held to. Report trained-vs-held-out split
separately; if the gain is concentrated on trained games, it is memorisation and the arm is dead.

**M5 — SLOT.** Only after M4 clears, and only with Ahmed's explicit go. Three live draws before any
selection claim, per the standing rule.

## What each outcome means

M2 fails → the lane is blocked on NVFP4 tooling, and the honest report is "unbuildable here", not
"training does not work". M3 fails → training on our own trajectories does not transfer between games,
which given the public/hidden gap is close to fatal for the whole idea. M4 in band → engaged-and-flat,
the seventh replication, and the campaign has no lever left before Oct 1; fall back to Track D
absorption as the plan's designated safety net. M4 ≥ 48 → the first step of the campaign.

## Cost and stop rule

M0–M1 are local and cost only time. M2 is a Modal serving session (~$5–15). M3–M4 are rig waves
(~$9 each). **Stop rule: if M2 is not passed within two working sessions, stop and report it as an
engineering blocker rather than grinding** — the Oct 1 absorption window is the higher-EV use of the
remaining time, and its tooling is already built.
