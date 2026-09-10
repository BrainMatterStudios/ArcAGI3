# The budget-controller family, measured on our own data — 2026-09-10

The 09-10 sweep ranked an **external cross-game budget controller** as the best-evidenced idea
available (ReD 81 % vs 34 % coverage; CLEAR +11.6 to +24.0 pts at tight budgets; ZEBRA −5.14 to
−5.92 pts for a uniform split; BAGEN 28–64 % token savings on failed trajectories). Every one of
those results depends on a prerequisite that is **domain-dependent and was unmeasured for us**:
can you tell early which games deserve the budget? Published AUCs run 0.86 (TextCraft) down to
0.59 (WebShop) and below 0.60 for deep research.

Both questions below were settled **offline on 247 recorded runs at zero rig cost.**

## 1. PREREQUISITE — PASSES, at the top of the published range

`src/modelaxis/early_predictability.py`. Every feature is computable from what the agent can see
live; nothing baseline-relative, because `baseline_actions` is excluded from the eval API.

Predicting **final levels ≥ 3** (base rate 0.14), AUC by decision point:

| after | levels cleared so far | actions on current level |
|---|---|---|
| 20 actions | 0.771 | 0.766 |
| 40 actions | **0.884** | 0.816 |
| 60 actions | **0.944** | 0.825 |

Predicting ≥ 2 (base rate 0.34): 0.680 at 40 actions, 0.869 at 80.

**Our domain is on the predictable side** — better than TextCraft's 0.86, far above the WebShop
regime where this family fails. Caveat stated: "levels cleared so far" is partly self-fulfilling for
a levels-based target; it is nonetheless exactly the information a controller would hold.

## 2. WHAT IT WOULD BUY — bounded at roughly +11 %, NOT a step

`src/modelaxis/triage_simulation.py`. Two halves, and only one is measured:
* **LOSS is exact** — levels an abandoned game cleared after the decision point, scored properly
  through the depth-weighted rule.
* **GAIN is an estimate** resting entirely on **one** number: the KV10 wave's elasticity, +47 % calls
  → +19 % levels, so ≈ 0.4.

Within the range where that elasticity was actually measured (uplift ≤ ~50 %):

| decision point | keep | uplift to survivors | net score | vs base 7.85 |
|---|---|---|---|---|
| 60 actions | top 60 % | +49 % | **+0.99** | +13 % |
| 30 actions | top 60 % | +59 % | +0.88 | +11 % |
| 60 actions | top 75 % | +24 % | +0.42 | +5 % |
| 40 actions | drop zero-level games | +72 % | +1.36 | +17 % |

**HONEST WARNING ABOUT THE AGGRESSIVE END.** The sweep's best cell — keep the top 25 % at 30 actions —
reports net **+7.70**, a doubling. **Do not believe it.** It applies a linear elasticity to a **+289 %**
uplift when the elasticity was measured once, at +47 %. That extrapolation turns a 2-level game into
4.3 levels with no evidence whatsoever, and elasticity of this kind almost certainly decays. The
credible read is the in-range rows: **~+11 %**, below the step bar.

## 3. THE EXPERIMENT THIS NOW JUSTIFIES

The whole family's value collapses to one unknown: **how does budget convert to levels at 2–3×,
not at +47 %?** A triage controller can genuinely deliver that much uplift to survivors, so the
question is no longer academic.

That is exactly the **diagnostic arm** already specified in the cadence pre-registration:
`--arm keith --games all --concurrency 9 --per-game-s 7920` — same server, same clock, ~3× the calls
per game because the wave runs ~3× longer. Not deployable, but it measures the elasticity at the
uplift a controller would actually produce. ~6 h, ≈$25–40.

* If levels/game scales near 0.4 elasticity out to 3×, the aggressive policy is real and the family
  is worth **building** — it would be the first credible route to a step.
* If it flattens (the likely outcome, and what the KV10/half-concurrency/cadence results all hint at),
  the family caps at ~+11 % and the honest answer is that it is not a step either.

## 4. DESIGN CONSTRAINT, if it is ever built

Three independent results say **the triage decision must not live in the model**: TRIAGE finds triage
efficiency negative for most of 20 model configs with no scaling in parameter count; BAGEN finds
models predict feasibility only after 60 % of budget is burned; ZEBRA finds an LLM allocating its own
budget costs −4.2 to −4.3 pts. Every positive result used an **external, code-side** controller.
ZEBRA also shows the controller tolerates a **50 % noisy** difficulty signal (−1.4 pp, n.s.), so the
signal above is more than good enough — the binding constraint is the conversion, not the ranking.
