# Failure-mode forensics — stock duck harness (Qwen3.8-27B), 16 games

**Data**: arc3-tp2c-smoke STOCK phase (graft installed, all knobs neutral: 60s yield, 31,744-token context budget, 1024-token tool cap, history_messages=10). 16 transcripts fetched to `submission/_tp_smoke/results_tp2c/transcripts/` — all 9 zero-level games (cd82 dc22 g50t ka59 sc25 sk48 tn36 tr87 wa30) + 6 one-to-two-level games + ft09 (4 levels, positive control). Every game ran its full ~7,920s (2.2h) and ended `gave_up` (time), never action-cap. Parser: `scratchpad/forensics/parse2.py`; stats: `per_game_stats2.json`. Counts below are **measured** from transcripts/JSON unless marked *inferred*.

## Headline numbers (measured)

- **803 env actions in 35.2 game-hours = 22.8 actions/hour/game** (3.0 actions/turn, 271 turns). Turn cadence 244-2,260 s/turn.
- **47% of all model calls (353/752) end on the 60s `turn_time_budget` yield with zero actions**; **63% of total wall time (79,323s of 126,775s) sits in model-call blocks that execute zero env actions** (lower bound — action blocks also contain thinking).
- **Stall-level action deficit**: 14/16 games stalled having spent LESS than ONE human-baseline play of actions on the stall level (median ratio 0.41x; s5i5 0.06x, vc33 0.07x, re86 0.10x, ft09 0.00x). Only dc22 (1.76x) and sc25 (1.97x) exceeded baseline — both have per-level move budgets and died to GAME_OVER instead.
- **The stall level eats the run while actions stop**: ar25 L2 = 33 acts/6,659s; ls20 L2 = 49/5,828s; m0r0 L2 = 45/5,924s; s5i5 L2 = 5/5,441s; vc33 L3 = 3/5,814s; re86 L3 = 9/5,012s; ft09 L5 = **0 acts/4,337s**.
- **World-model carry failed completely in 4/16 games** (tn36, tr87, g50t, re86: 0 "Working world model carried" blocks in the whole stock phase — these runs emitted thinking+tool calls with empty/unprefixed assistant text, so the harvest got nothing). 3 of these 4 are zero-level games. re86: 54 responses with `content_chars: 0`.
- **tn36 pathology**: 50 model calls, ALL at analysis_step=1, byte-identical user prompts, **0 actions in 2.2h** — every 60s window restarted scene analysis from absolute zero (yield discards in-flight reasoning; empty world model means nothing survives).
- Tool errors (tracebacks): 43 across 752 calls (5.7%) — real friction (re86 10, wa30 9) but not the binding failure. Zero-tool-call responses: 8/752. Format collapse: none.
- GAME_OVERs: 4 events (dc22 x1 at action 84, sc25 x1 at step 51 [its own wm: "the timer expired"], s5i5 x2) — all in games with per-level move/timer budgets.
- Clicks are **aimed, not scattered** (measured): requested clicks concentrate on few object coordinates with heavy repeats — dc22: (19,48)x21 + (36,48)x14 (its two identified toggles); sc25: (50,30)x18, (55,25)x10; ft09: 38 distinct grid cells. No spray pattern anywhere.

## Per-game table (stock phase)

| game | levels | actions | turns | stall-lv acts / human base | dominant failure | evidence one-liner |
|---|---|---|---|---|---|---|
| tn36 | 0/7 | **0** | 1 | 0 / 32 | **G analysis-paralysis (harness)** | 50 identical-prompt calls, 46 yields, zero actions ever; wm never carried |
| tr87 | 0/6 | 51 | 7 | 51 / 54 (0.94x) | **G paralysis** | 51 acts in first ~25 min, then ~105 min of cipher analysis + code-bug fixing, 0 acts; wm carry = 0 |
| g50t | 0/7 | 21 | 11 | 21 / 78 (0.27x) | **A verb/mechanic not found** (+G) | step 22 after 2.2h; single-arrow probes; wm carry = 0 |
| cd82 | 0/6 | 27 | 25 | 27 / 55 (0.49x) | **A verb not found** | arrows tick only the timer cell (63,63); final wm holds two contradictory mechanics side by side |
| dc22 | 0/6 | 104 | 18 | 104 / 59 (1.76x) | **A goal not found** (+D budget death) | systematically enumerated 36 knob x toggle states, none wins; GAME_OVER at action 84 (countdown HUD) |
| sc25 | 0/6 | 71 | 29 | 71 / 36 (1.97x) | **A goal not found** (+D) | wm post-mortem verbatim: "spent the whole budget probing ... without identifying the exact winning condition"; timer GAME_OVER at step 51 |
| sk48 | 0/8 | 50 | 19 | 50 / 61 (0.82x) | **B mechanic found too late** | decoded correct push-by-6/row-overlap rule near end of 2.2h; plan formed, clock out |
| wa30 | 0/9 | 62 | 32 | 62 / 71 (0.87x) | **C re-probe loop** | after 2.2h and 62 acts final plan is still "probe LEFT once to check the box's behavior" |
| ka59 | 0/7 | 20 | 20 | 20 / 28 (0.71x) | **E cadence starvation** | all 20 actions board-changing (bc_true=20, bc_false=0), coherent probing, 1 action per 6.6 min |
| ar25 | 1/8 | 57 | 13 | 33 / 50 (0.66x) | **B slow + goal-shape uncertainty** | coherent mirror-alignment geometry on L2, several hypotheses tried, 111 min not enough |
| ls20 | 1/7 | 63 | 11 | 49 / 123 (0.40x) | **C verification loop** | BFS found a 20-move path, then region-by-region re-verification for the rest of the run; final carried wm still describes the L1 board |
| m0r0 | 1/6 | 62 | 20 | 45 / 111 (0.41x) | **E slow but progressing** | correct eye/J-position action model, still exploring at cutoff |
| s5i5 | 1/8 | 69 | 18 | 5 / 89 (0.06x) | **G paralysis at frontier** | L1 cleared (64 acts, 2 GAME_OVERs), then 5 acts in 91 min on L2; re-deriving ascii parsing at the end |
| re86 | 2/8 | 73 | 20 | 9 / 86 (0.10x) | **G paralysis** (+ii amnesia) | zero assistant text all run (54x content_chars=0), wm carry = 0; L3 = 84 min of SPACE-teleport pattern archaeology, 9 acts |
| vc33 | 2/7 | 20 | 13 | 3 / 44 (0.07x) | **A wrong affordance** (+G) | L3: clicks passenger, "Only HUD pixel changed — passenger isn't clickable", 3 acts in 97 min |
| ft09 | 4/6 | 53 | 14 | 0 / 65 (0.00x) | **G paralysis at frontier** | L1-4 cleared UNDER human baseline (5,7,14,27 vs 43,12,23,28); L5 = 72 min of icon decoding, plan written, zero actions |

## Aggregate failure counts (dominant class, 16 games)

- **G — analysis-paralysis / action-emission failure** (model keeps analyzing, harness yield discards it, actions stop): **5** dominant (tn36, tr87, s5i5, re86, ft09) + secondary in 4 more (g50t, vc33, ls20, sk48). Not in the given a-f taxonomy; it is the modal failure. Fuel: 60s yield + context wipe + 1024-token tool cap forcing many tiny read cycles.
- **A — mechanic/goal never found**: **5** dominant (cd82, g50t = verb; dc22, sc25 = goal; vc33 = affordance).
- **B — mechanic found but executed too slowly/too late**: **2** dominant (sk48, ar25).
- **C — loop (re-probe / re-verify)**: **2** dominant (wa30, ls20); loop-evidence also in sc25's x18 identical clicks.
- **E — wall-clock starvation while producing novel states**: **2** dominant (ka59, m0r0).
- **D — GAME_OVER spiral**: 0 dominant; contributing in 3 games (dc22, sc25, s5i5 — per-level move/timer budgets).
- **F — tool/format failures**: 0 dominant; 5.7% of calls have tracebacks, 8/752 empty tool-call responses.

## Ranked interventions (by failures addressed)

1. **Action-emission floor + yield-survival** (require >=1 env action per model call once >=1 turn of analysis has banked, or auto-extend the 60s yield when a tool call is mid-flight; persist in-flight conclusions across yields). Addresses: G x5 dominant + G-secondary x4 + E x2 -> **11/16 games**; directly attacks the 63%-of-wall zero-action time and tn36's 0-action run. Feasibility: pure harness change; `TP_YIELD_SECONDS` knob already exists in the graft — this is exactly the axis the tp2c arms probe.
2. **Guaranteed world-model persistence** (harvest notes from THINKING when assistant text is empty/unprefixed; auto-summarize on yield; stamp the level number and invalidate on level change). Addresses: the 4 zero-carry games (tn36, tr87, g50t, re86 — 3 of them zero-level) + staleness (ls20's L1 wm on L2, cd82's contradictory wm) -> **~6 games**. Feasibility: prompt/graft change, no model change; dc22 proves the model USES carried notes well when they exist (36-state enumeration without repeats).
3. **Hypothesis-ledger goal search** (structured tested/untested win-condition list carried in notes; mandate a discriminating probe per turn; forbid re-testing logged states). Addresses: A x5 + C x2 -> **7 games**. Feasibility: medium — prompt scaffold + notes format; the model already does this ad hoc when wm carry works (dc22, sc25's own post-mortem).
4. **Plan-commit rule** (when a concrete multi-action plan exists — BFS path, batch click list — execute it in ONE batched `action(...)` call now, verify after; no pre-verification loops). Addresses: ls20, ft09, sk48, ar25 (+ trims G everywhere) -> **4+ games**. Feasibility: prompt-level; `action()` already batches; per-action 27B cost makes batching the single cheapest throughput lever.
5. **Move-budget/timer awareness** (detect countdown HUD, estimate per-level action budget, cheap-probe policy, fast replay of known-good prefix after GAME_OVER). Addresses: D-contrib x3 (dc22, sc25, s5i5) and makes the two above-baseline games (dc22, sc25) survivable. Feasibility: `probe_budget`-style detector already exists in repo lore; modest.

## Counter-evidence audit

- **(i) "more actions would yield more levels"** — COUNTER: dc22 (1.76x) and sc25 (1.97x) exceeded the human-baseline action count on their stall level and still cleared nothing (comprehension/budget-death, not volume); wa30 0.87x and tr87 0.94x near-baseline with zero clears. SUPPORT: 12/16 stall levels got <0.9x baseline actions before time expired — but the proximate cause was non-emission (paralysis/yield), not an action cap; no game hit an action cap (all `gave_up` on time). Net: raw action volume alone is not sufficient (measured), though it is necessary for the starved 12 (inferred).
- **(ii) "the model forgets earlier discoveries"** — SUPPORT (measured): 4/16 games carried ZERO world model all run; tn36 restarted identical analysis 50x; ls20's final carried wm describes level 1 while it played level 2 for 97 min; cd82's wm holds a refuted mechanic next to its replacement. COUNTER (measured): when the carry works the model retains and uses discoveries well — dc22 accumulated a correct action model over 38 carries and enumerated 36 states without repetition; ka59/wa30/m0r0 carried 33/46/42 wm blocks with consistent action models. Forgetting is a harness-channel failure (empty assistant channel -> empty harvest), not a model-memory failure per se.
- **(iii) "the model never states correct goals"** — COUNTER (measured): sk48 stated the correct push-mechanic + plausible collection-order goal; ft09 stated and executed level rules for 4 cleared levels; dc22/sc25 stated accurate ACTION models and explicitly flagged the goal as unknown (sc25's wm literally diagnoses its own failure). Goal statements are usually internally consistent and evidence-cited; we found no case of stating a goal then acting against it (cd82's contradictory wm is stale-note accretion, not contradictory action). The gap is goal-DISCOVERY on ~5 games and action-EMISSION on ~5, not goal hallucination.

*Caveat*: classifications B/C/E/G per game are judgments over measured signals (action deltas from header counters, timestamps, board_changed prints, executed-action echoes, wm text); the counts and ratios themselves are mechanical. Single stock play per game — per-game class labels carry draw noise; the aggregate pattern (63% zero-action wall time, 47% yield rate, 14/16 under-baseline stalls) is consistent across all 16 and unlikely to be draw artifact.
