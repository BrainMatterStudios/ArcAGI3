# STRATEGY 2026-08-29 — Independent review: how to get from 1.7 to 6+

Written after a from-scratch audit that deliberately did **not** trust the 08-29 handoff, memory, or
prior docs. Four investigations ran in parallel (all preserved in `docs/research-2026-08-29/`):

- **R1** Kaggle ecosystem — top public notebooks pulled and read, discussions, leaderboard history, top-team footprint
- **R2** Literature — Tycho, AVO, VISTA, OPINE, AERA, Retrodict, PRO-LONG, Prime Agent, ARC Prize's own analyses, TTT
- **R3** Scoring audit — scorer code vs our recorded plays
- **R4** Harness audit — where the 9 hours actually go (code, vLLM logs, transcripts)

Everything below cites those reports or a direct check I ran. Where the handoff is contradicted, it says so.

---

## 1. Where we actually stand (verified 2026-08-29)

| fact | value | source |
|---|---|---|
| Public LB top | cstl **5.99**, Lord Han Solo 4.99, Tufa Labs 4.67, Tong Hui Kang 4.27, rfbr 3.37 | `kaggle competitions leaderboard`, `research-2026-08-29/leaderboard-2026-08-29.csv` |
| Us | **1.74** best draw, rank **291 / 2,603**; series of 12 Qwen3.8 draws mean 1.41, sd 0.245 | ledger, R3 |
| Field | 160 teams >= 2.0, 29 >= 2.5, 7 >= 3.0, 4 >= 4.0 | leaderboard CSV |
| Best PUBLIC notebook | **2.23** — FOYSAL's "duck v12 + Qwen3.8" — which is **byte-for-byte our own v12 base** | R1 §1.1 |
| Daily submissions | **1 per UTC day** (`_max_daily_submissions: 1` from the Kaggle API; R1's "5/day" was wrong). ~65 days left | Kaggle API |
| When the 4–6 band appeared | **08-16 → 08-28**, right after Qwen3.8-27B (08-14). cstl 2.52 (08-11, *before* 3.8) → 5.99 (08-24); Tufa 1.62 → 4.67; Tong Hui Kang 0.80 → 4.27 in six days | R1 §0.2, `lb_history.json` |
| Has any 3+ team published? | **No.** Tufa's public Duck repo/notebooks are frozen at 07-01. All public notebooks above 1.9 are the Duck + a model swap | R1 §1, `gh api` check |

**Reading:** we are at parity with the public frontier (same code, same model; their best draw 2.23 vs ours 1.74 is
selection noise). The leaders are doing *private harness work on the same model class* and got ~2.5× above the public
Duck in two weeks. cstl was already 1.6× the Duck on Qwen3.6 and then roughly doubled again with 3.8 — the gains
**multiply**: model × harness.

---

## 2. Corrections to the handoff (evidence, not opinion)

| handoff claim | verdict | evidence |
|---|---|---|
| "Efficiency is worth exactly ZERO; score == cap; every efficiency lever dead by arithmetic" | **WRONG** | On the only full Qwen3.8 wave, **9/21 cleared levels were slower than the human baseline** (5 scored <50). ~**20% of the completion cap is forfeited** to efficiency (2.53 vs 3.16 local). The handoff applied a *median* ratio (0.86) to a *per-level* cap. (R3 §2) |
| "L1 on all games -> 2.82 max"; "perfect efficiency -> 1.95 max" | unsourced / wrong | At cap, L1-on-all is >= 3.52; perfect efficiency recomputes to 1.76. (R3 §6) |
| "Public-25 does not predict LB (3–5×)" | overstated | On **identical bytes** our local/LB ratio is **1.7–1.8×**. The 3–5× figures are other teams' harnesses. Local-25 is fine for reading *large* effects, which is what we need. (R3 §4) |
| "252 plays, zero 3-level games — the two-level wall" | true only for full waves | Seven 3+-level runs (one 4-level) exist in `scratchpad/multirole_corpus/` from 08-18..20. (R3 §3) |
| "The live hypothesis is the MODEL axis (TTT); next action = serving gate" | **not supported** | No published evidence anywhere that weight-level TTT helps on ARC-AGI-3; Tong Hui Kang's own published attempt failed; the one online-learning system that placed (StochasticGoose) collapsed 12.6 → 0.25 on the private set. Every 30–100 RHAE system learns into *workspace/memory*, not weights. (R2 §7) |
| "The crack/specialist lane" (ft09 etc.) | ceiling confirmed, deprioritize | Le Grand's 7/25-game solver = 0.9 LB; AERA's BFS = 0.30 LB; inventory is one game. Inert on 109/110 games. (R1 §2, R4 §1.3) |
| "1 slot/day" | **correct** | Kaggle API |
| "AVO proves our architecture reaches 100 with Opus 5" | true but irrelevant | Every 78–100 system is a frontier API model on the public 25. Not Kaggle-eligible; only their *mechanisms* transfer. (R2) |

---

## 3. Anatomy of the gap — three multiplicative deficits

### 3a. Throughput: we get ~88 actions per game, and every play dies on the clock

Measured on the real Kaggle RTX Pro 6000 with the shipped serving stack (R4 §0, §2):

- **100% of plays end by the 7,920 s per-game wall clock. 0 wins.** Median **85 actions/game**, ~105 s per action.
- Each LLM call carries a **16–20k-token prompt** and generates ~2.4k tokens. Prompt (prefill) throughput runs at
  2,000–6,000 tok/s while generation is only **150–330 tok/s** — the GPU spends most of its time *re-reading the same
  prompt*. Prefix-cache hit rate decays from 58% to 0–20% as contexts fill (KV 70–98% full), because
  `_trim_messages_for_context` drops the oldest message every turn, so the cached prefix is invalidated every call.
  (verified in `scratchpad/ft09_ablation/ft09-base/vllm-openai-server.log`)
- Per-stream decode: **40 tok/s at concurrency 1, 27 at 8, 8.6 at 28**; aggregate ~300 tok/s at 28 with 20k prompts
  (`_serving_lab3`). Context length, not batch size, throttles the GPU.
- The **60 s yield** (`LOCAL_ANALYZER_YIELD_SECONDS=60`) is *shorter than one call* (118 s median at conc 28), so nearly
  every "slice" is a single call; an investigation-only call ⇒ yield ⇒ fresh prompt + image ⇒ re-grounding.
  **26% of play wall time is in slices that end with no action**; stuck steps of 17–59 slices (bp35: 73 min, 1 action).
- Two of 28 slots idle for all of wave 4 (5% of capacity); ~90 s of slack before the 9 h kill.

Why this caps score even for a perfect player (human baselines, 25 public games: L1 median 30 actions; cumulative
L1–2 = 83, L1–3 = 152): **100 actions at human efficiency ≈ 1.9 levels; 200 ≈ 3.5; 400 ≈ 5.3.** We are at 88.
Levels 4+ — where most of the weight is — are unreachable at today's throughput no matter how smart the model is.

### 3b. Comprehension and memory: 43% of plays clear zero levels, and they are NOT action-starved

- Level distribution per play: **43% zero, 39% one, 18% two, 0% three** (R3 §3).
- Zero-level plays spend 91–1,304 actions on L1 vs baselines of 22–78 (sk48: 1,304 vs 61; ar25: 291 vs 32). They
  fire 20–140-action blind movement batches, hit GAME_OVER 7×, and never form a goal model (R4 §4).
- The harness has **no stagnation detector** and **no level-start probe**; the only loop-breakers are the 60 s yield and
  the clock (R4 §1.2).
- Memory is a rolling eviction window of ~4–9 assistant turns (16–20k tokens re-feeding prior reasoning), plus
  self-written notes that (a) require the model to volunteer a literal `World model:` prefix — carried on **33/481
  turns** in thtennant's archive, **0/1,482** cross-level notes in ours — and (b) are **wiped on every GAME_OVER**
  and level-up (`tool_agent.py:1113-1126`). The model "keeps starting over", the exact failure OpenAI diagnosed.
- ARC Prize's failure taxonomy (Opus 4.7/GPT-5.5): "true local effect, false world model", "wrong game prior imported
  from training", "solved the level, didn't learn the game" — all visible in our transcripts (sk48 period-3 limit cycle).

### 3c. Efficiency (second-order but real): ~20% of the cap

9/21 cleared levels slower than human; the worst offenders are blind action batches. Cheap to fix alongside 3b
(batch caps + expectation checks). Never the priority, but not "zero".

**A prior counter-result to keep honest about.** The two-level-wall note records that "patch 21" bought ×1.40
actions/game and cleared 25% *fewer* levels, with 88% of the extra actions going into levels never cleared. That is
consistent with 3b, not against 3a: extra actions spent by a loop that has lost its state are wasted. It is why Packs
1 and 2 below are a pair, and why the offline gate for every pack is **levels per game**, never actions per game.

### Counterfactual arithmetic (R3 §3, local per-play mean 2.53, LB ≈ local/1.75)

| scenario | local | ≈ LB |
|---|---|---|
| observed | 2.53 | 1.45 |
| every zero-level play clears L1 | 4.16 (×1.65) | 2.4 |
| every L1+ play gains one more level | 7.53 (×3.0) | 4.3 |
| both | 10.5 (×4.1) | 6.0 |

**6.0 = "every game reaches L1 and every game that reaches L1 reaches L2."** That is the whole target, in one line.
The leaders' trajectories (Tufa 1.62 → 4.67 in 13 days on their own code) say it is reachable with this model.

---

## 4. What the evidence says works (ranked by evidence × transferability to an offline 27B, 9 h)

From R2 (mechanism-level, with URLs) and R1 (what the Duck lacks):

1. **Retained state + compaction instead of eviction.** OpenAI: 13.3 → 38.3 on the public set with 6× fewer output
   tokens by keeping reasoning across tool calls and compacting rather than truncating. Retrodict `playbook.md`,
   VISTA `GUIDE.md`/`WORKING.md`, Reki's reflection memory every ~10 steps. *Strongest evidence, cheapest change.*
2. **Action batching with harness-executed plans and interrupt-on-surprise.** LLM emits a short sequence + expected
   effect; harness executes, halts on `board_changed==False`, level change, game over, or mismatch. Reki (2nd, 31B),
   Retrodict, Tycho plan-gating, PRO-LONG. It is what makes 300+ actions/game affordable.
3. **Deterministic level-start probe before any LLM call**: each available action once, ACTION6 on the top-k salient
   components; hand the model a table of "action → what changed". AERA: forcing ACTION6 first turned 0/5 → 5/5 on
   three games; StochasticGoose wasted ~350 actions/level learning clickables. Zero LLM cost.
4. **Observation as diffs + connected components with the HUD/timer strip masked**, raw grid hidden (Retrodict `[DIFF]`,
   Duck segmentation, just-explore). Fixes `board_changed` being always-true on 10/25 public games.
5. **Model-free stagnation supervisor with escalation tiers** (AVO's supervisor; Retrodict's 300-action directives;
   Continual Harness refine-on-stall): frame-hash novelty, actions since last level-up, repeated loops ⇒ inject a
   binding redirect; still stuck ⇒ hand control to (6).
6. **Graph-frontier explorer as the fallback policy** (just-explore: 3rd in the preview with *zero* LLM calls, median 16
   private levels in 8 h). Converts zero-level games into one-level games when the LLM stalls.
7. **Effect-signature tables** (OPINE's ontology error without the synthesizer; Reki's dead-signature): counts per
   (component type, action) → dead objects banned for the level, high-entropy objects probed.
8. **Level-boundary context reset with a carried summary + ruled-out list** (Tycho, VISTA). Targets "solved the level,
   didn't learn the game".
9. **Predict-before-act with predicate verification** (not full simulators — Tycho's ablation: 88% transition accuracy
   scored *worse* than 16%; repair-heavy loops hurt).
10. **Test-time learning of the workspace, not the weights** (PRO-LONG +18 pts from python/grep over a log; NOOA +11.8
    from structured memory). Weight-TTT: no evidence, stays off the list.

Explicitly **not** worth building: program-synthesized simulators with CEGIS at 27B (600–1,000 frontier calls/game);
image-first perception (VISTA's own ablation: text works); hand-built game-specific tools (Tufa: they hurt);
search-only agents (≤0.54 LB); the null-coordinate crash-win (local library artifact); more thinking per action
(we are throughput-bound; `effort_medium` was worse, `xhigh` is the shipped default — leave it).

---

## 5. The plan — four packs, each measured offline before it flies

**Measurement rule.** Every pack is first run at eval geometry (28 concurrent, 7,920 s/game) on Kaggle GPU quota
(zero slot cost) against the 25 public games + a holdout slice of the 249 MIT `arc-interactive` games, reading
**actions/game, levels/game, prefill:decode ratio, prefix-hit rate** — not just score. Local-25 reads ×1.5+ effects
reliably (identical-bytes ratio 1.7–1.8×, R3). Only a pack that moves *levels* locally gets a slot. Because the
objective is max-over-draws and every day unused is a lost lottery ticket, the current best config keeps flying on
days when nothing new is ready.

### Pack 1 — Throughput (plumbing only; no prompt semantics change). Target: ≥2.5× actions/game. ~2–3 days.
1. **Prefix-stable trimming**: replace drop-one-message-per-turn with hysteresis (when over budget, drop to ~55% in
   one cut; keep the system prompt + carried summary as a fixed prefix). Expected prefix hit 20% → 80%+, which
   removes most prefill and hands the GPU to decode. (`tool_agent.py:1608-1690`)
2. **Context window 32k → 16k** (real memory is already 4–9 turns; halves KV pressure; lab: fewer resident tokens ⇒
   3× per-stream decode).
3. **`LOCAL_ANALYZER_YIELD_SECONDS` 60 → 900** (no more sub-call slices; `should_stop` still checked every call) and
   bound the tool loop (`LOCAL_ANALYZER_TOOL_STEPS` ≈ 6).
4. **Keep notes across GAME_OVER** (one line, `tool_agent.py:1113-1126`); keep the level-up wipe for now.
5. **Reclaim wave-4 idle slots / remove the 90 s cliff**: set `max_runtime_s_per_game` from measured setup time.
6. Cap `action([...])` batches at ~10 per call (kills the 1,304-action blind batches; efficiency + GAME_OVER spam).

### Pack 2 — Memory & control (the loop shape every 30+ system shares). Target: zero-level games 43% → <25%. ~4–6 days.
1. **Harness-owned compaction**: when trimming, the harness asks the model for a structured summary (verified rules /
   assumptions / ruled-out / plan) and pins it as the carried prefix — no reliance on a volunteered `World model:` line.
2. **Level-start probe table** (mechanism 3) injected as the first user turn of every level.
3. **Stagnation supervisor** (mechanism 5): novelty + actions-since-progress ⇒ tiered directives; last tier hands the
   game to the frontier explorer (Pack 4) for a bounded budget.
4. **Expectation-checked batches**: the model states expected effect per batch; halt on mismatch; feed the diff.
5. **HUD-masked `board_changed` + `[DIFF]` line** in the tool result.

### Pack 3 — Serving upgrade. Target: +1.5–2× decode. ~3–4 days, parallel to Pack 2.
- Newer vLLM wheelhouses already exist on Kaggle (`saltb0x/arc3-vllm-wheelhouse-v0271-cu129`, jacquesbuis 0.24);
  post-0.19 vLLM fixes the MTP + prefix-cache corruption on GDN (PR #47861), so **MTP speculative decoding** (the
  repacked model ships `mtp.safetensors`) and prefix caching can coexist. z-lab's DFlash2 drafter claims 2–3.4×.
  Measure with the `_serving_lab` kernel pattern (acceptance rate, tok/s at conc 28) before any gameplay.

### Pack 4 — Model-free floor. ~3–4 days.
- Port just-explore's masked-frame-hash graph + click-priority tiers as the stagnation fallback and as the policy for
  the last ~15 min of every game (a never-cleared L1 is worth 2–5 points on that game; a cheap frontier walk clears
  L1 on many keyboard/click games per AERA's table).

### Sequencing and honest expectations
- Week 1: Pack 1 measured + flown; Pack 2 built. Week 2: Pack 2 flown; Pack 3 measured. Week 3: Pack 3/4 flown;
  then farm the best config daily (max-over-draws) while iterating.
- If the packs deliver what they measured elsewhere: throughput ×2.5–3 turns our "understood" games from 1–2 levels
  into 2–4 (the ×3 counterfactual), memory/probes/fallback convert a share of the 43% zero-level games (the ×1.65).
  Realistic band: **3–4 LB within three weeks; 6 requires the full stack plus draw selection.** The leaders did
  exactly this magnitude in two weeks, so it is not speculative that the model supports it — the risk is execution.
- **Backstop:** Milestone 2 (Sept 30) requires public notebooks for prize eligibility; if cstl/Tufa/THK publish,
  33 days remain to adopt and farm. Plan for it but do not wait for it.

---

## 6. What to stop doing

- Stop treating the serving gate / fine-tune lane as the next action. It is a half-day well spent *only after* Packs 1–2,
  as a diagnostic, and there is no evidence it moves score.
- Stop spending on the specialist/crack inventory (0.9 LB ceiling, one game, inert on 109/110).
- Stop reading single draws as verdicts (CV 0.17 ⇒ a draw reads only ±0.5); read offline levels/actions instead and fly
  for the lottery ticket.
- Stop citing "efficiency is worth zero".

---

## 7. Open questions worth one cheap check each
1. Does the gateway enforce the tech report's "5× human median" per-level action cutoff? Not in local `arc_agi`/`arcengine`
   code; matters only once we go action-heavy. Log `actions_per_level` vs `baseline_actions` if visible.
2. Is `environment_info.baseline_actions` visible at eval (Cottaar's thread, unanswered)? One log line inside an
   existing run settles it; if yes, per-level action budgeting becomes exact.
3. Is the LB denominator 55 or 110? Changes the value of a full win (+0.91 vs +1.82) and nothing else in this plan.
