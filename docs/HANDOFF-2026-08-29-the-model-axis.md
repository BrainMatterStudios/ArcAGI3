# HANDOFF 2026-08-29 — START HERE

Supersedes `HANDOFF-2026-08-28-comprehension-and-the-crack-lane.md`.
Working record: `docs/RESEARCH-2026-08-28-the-screen-false-negative.md`.

Everything below was measured on 08-28/29 against the live Kaggle API, the
`arcengine` source, the 25 public games, our own recorded waves, or published
external sources with URLs. **Section 6 lists claims from the previous handoff
that are now RETRACTED — read it before trusting anything upstream of it.**

---

## 0. FIRST: the clock, and the live state

`date -u` said **2026-08-29 09:15 UTC** while the harness notice said 08-29.
**Always verify with `date -u`.** This has now cost us three times.

| thing | state |
|---|---|
| submission **55861885** | PENDING, submitted 07:11Z 08-29, byte-identical v8 harvest (svid 345333330) |
| daily slot | **used** (0 remaining today) |
| kernel `arc3-oracle-probe` v2 | **RUNNING** — zero-slot GPU, the oracle-model probe |
| branch `winning/duck-patched` | `b1dfd40`, **not pushed** (push gate) |

## 1. THE STRATEGIC PICTURE, quantified

### 1a. Efficiency is worth exactly ZERO
Scoring, re-derived and validated **bit-exact against the real scorer** (224
rows, 0 mismatches, `scripts/rescore_waves.py --validate`):

```
total_w = N(N+1)/2 ;  s_i = min(115, 100(b_i/a_i)^2)
score   = min( sum_i w_i*s_i / total_w ,  100*k(k+1)/(N(N+1)) )
```

The second term — the **completion-share cap** — binds. At our measured 0.86x
human pace `s_i = 115 > 100`, so **score == cap exactly**. Playing faster than
the human baseline buys *literally nothing*. Efficiency is a one-sided penalty
we are already past. **Every efficiency lever is dead by arithmetic.**

Caps per game (mean over the 25 public games, N~7.3):
`k=1 -> 3.52 | k=2 -> 10.57 | k=3 -> 21.14 | all -> 100`

### 1b. The ceilings foreclose most of the strategy space
- perfect efficiency forever on levels we already clear -> **1.95 max**
- clearing level 1 on **all 110 games** -> **2.82 max**
- to reach 3.0: **28 games** must go 1->2. To reach cstl's 5.99: **~76 games**
- one game banked as a **full win** -> **+0.91** (+1.82 if the eval denominator
  is 55 rather than 110)

**We cannot reach 3.0 by playing better at what we already do.** Only depth on
games we currently fail moves anything.

### 1c. Most experiments we can run are UNMEASURABLE
Series n=12, mean **1.409**, sd **0.245**, **CV 0.174**. A single-draw effect
must exceed **~+0.49** to be readable. Detecting **+0.20** at 80% power costs
**~48 slots**; ~66 remain before Nov 2. This explains five weeks of unreadable
single-lever arms: insufficient power, not bad luck.

### 1d. Our main offline instrument does NOT predict the leaderboard
Measured independently by several teams (Kaggle discussion #732854, #736578):
Nick Pellegrin 5.0-5.4% local -> **still 1.4 LB**; OverfitOracle "5.0+ on 25
games, 1.6 on the 110"; daoviet 6.8 local -> 1.19 LB. Ratio is **3-5x and not
constant across harnesses**. The AERA paper (arxiv 2605.25931) shows all 25
public games are reachable by non-intelligent strategies (10 in a single blind
step). **Do not buy a lever on a public-25 delta.**

## 2. THE EXTERNAL PICTURE — what the leaders actually do

Four systems now score **78-100% RHAE on the public 25**:

| system | score | model | loop shape | code |
|---|---|---|---|---|
| VISTA (MIT) | 100.00 | Opus 5.0 | per-action VLM, 512px PNG | none |
| **NVIDIA AVO** | **100.00** | Opus 5 | **per-action LLM, TEXT-ONLY** | none |
| Tycho | 100 / 88.49 on Opus 4.8 | Opus 4.8/5 | actor + world-model builder | **Apache-2.0** |
| OPINE-World | 78.4 | Opus 4.8 | two agents, CEGIS | — |

**AVO's loop shape is OURS** — per-action, text-only, exact 64x64 grid — and it
reaches 100. **So the architecture is not the ceiling.** Its deltas are
persistent memory and a stagnation supervisor.

**But every one of the four runs a FRONTIER API MODEL.** We are locked to a 27B,
offline. That is the central unresolved question of the campaign (§3).

### Externally measured negatives — do not rebuild these
- **Neural latent world models = 0.0** (Dreamer/MuZero family clear NO level);
  WorldCoder = 0.0. Retires JEPA/latent-dynamics.
- **A better simulator is NOT a better player.** Tycho's ablation: a policy with
  **88.1% transition-match accuracy scored WORSE** than one with 16.2%.
- **Specialist/solver lane ceiling, measured by someone who ran it harder than
  us**: Scott Le Grand solved 7/25 public games, ~38% RHAE local — **the same
  agent submitted scored 0.9 LB.** AERA's BFS+pre-solve kernel: **0.30 LB**.
- **Animation frames**: +1.4%, p=0.92 (TAAF's A/B, thread #734369). Wall-clock
  bound, not action bound.
- **Context compaction in OUR stack**: already built twice (`depth_pack.py` D1b
  shipped -> 0.78; `TAAF_COMPACT` pinned off). 28/28 plays die at the WALL
  CLOCK, median 14 actions into a 52-baseline level — we are time-starved, not
  context-starved. OpenAI's 2.9x compaction result was on a frontier model with
  a huge window and does not obviously transfer.

## 3. THE LIVE HYPOTHESIS: it is the MODEL axis, not the harness

Put §2 together: AVO proves our architecture reaches 100 **with Opus 5**; all
four leaders use frontier models; our own six-point adoption law says the 27B
ignores every channel we offer it:

| channel | asked | delivered |
|---|---|---|
| cross-level notes | every turn | **0 / 1,482** |
| `run_probe` | every game | 1 / 28 |
| `plan_queue` | every run | 0 / 20 |
| compaction call | on trigger | 0 / 303 |
| `world_model.py` edits | mandated | 0 / 209, 0 / 140 |

**LAW: advertised = performative; only structural enforcement is adopted.**

And the transcripts (36 real Qwen3.8 plays, `scratchpad/multirole_corpus/`) say
the failure is **memory, not perception or hypothesis formation**: on sk48 the
agent makes 16 sharp, mostly-correct mechanistic discoveries, retracts true
ones, and ends in a **byte-identical period-3 limit cycle**. In `dc22` all 81
turns show "current world model from the previous turn" followed by **nothing**
— the channel is a monotone latch keyed on a literal `"world model:"` prefix
the model stopped emitting.

⇒ If the gap is capability rather than scaffolding, the only lever available
offline is **test-time training**, and three of the top nine teams (Tong Hui
Kang, Franzen/LLM ARChitect, MindsAI) are TTT houses.

**This is the one axis with a KNOWN UNREPAIRED HOLE rather than a measured
negative.** Verified 08-29 from the ledger: nine fine-tune-tagged submissions
exist and **not one ever served an adapter and scored** — 54554985 (1.26) and
55122039 (0.95) were both later shown to be BASE draws; 55160933, the one that
explicitly merged in-kernel, ERRORed.

## 4. THE NEXT ACTION — the serving gate (half a day, ZERO slots)

Merge a LoRA, FP8 re-quant, then **diff logits / top-k of the served endpoint
against base on fixed prompts, before any gameplay.**

- identical distributions ⇒ the adapter is not being served ⇒ **every
  fine-tuning verdict in this campaign is uninformative about fine-tuning**,
  which is worth knowing in half a day
- gate passes ⇒ TTT becomes a real, testable lane

Do this **before** any further general-agent scaffolding.

## 5. THE ORACLE-MODEL PROBE — running, unresolved

`submission/_oracle_probe/`. Hands the model a hand-written **verified-correct**
world model + goal + exact state every turn on wa30 L3, and asks only for the
next action. Three arms: `control` (board only), `oracle` (+ mechanics),
`guided` (+ per-turn instruction naming WHICH block and WHICH cell).

wa30 L3 is the only fair test bed we have: mechanics engine-verified by sprite
trace, `wa30_carriersim.py` lockstep-exact vs the engine on L1-L4, and an
**82-move solution engine-verified against the level's own 100-move budget** —
so "no plan fits" is excluded by construction.

**Local pilots (qwen3:8b n=5/arm, qwen3:14b n=3/arm): 0 wins in 24 runs, all
three arms, every run ending at exactly 3 blocks left** (= the two the carriers
ferry free; the LLM completed ZERO handoffs). Action channel independently
validated — hand-picked sequences move the avatar exactly as commanded.

**NO VERDICT YET on the shipped 27B.** Kernel v2 is RUNNING. v1 was an
INSTRUMENT FAILURE (§6).

Pre-registered reading, fixed before the run:
- oracle >= 50% -> can act on a model it did not build; gap is
  construction/persistence; TTT / structural memory live
- oracle < 50% AND guided < 50% -> cannot execute a correct model even when
  handed one ⇒ notes, digests, latch repairs, compaction and behaviour
  fine-tunes ALL dead by construction; only code-selects-actions remains
- oracle < 50% BUT guided >= 50% -> deficit is comprehension/planning, not
  execution

## 6. RETRACTIONS — claims from 08-28 that are now FALSE

1. **"cx01 cracks 7/7 levels; deployable inventory is 2."** FALSE. cx01 reports
   `levels_completed=6` immediately after `warmup_and_freeze` (42 probe
   actions) because it is a 7-level game with a **6-action human baseline per
   level**. Warmup clears 6 levels by accident; every specialist `solved=True`
   was `is_win(start)` firing on an already-satisfied target. The specialist won
   **zero** levels and FAILS cx01 at L7 (`bad_cycle`).
   **Inventory is still ONE (ft09).** The screen fix is real (cx01 IS
   ft09_gf2-class, old bars were overfit to ft09's dense board) but buys no
   crack.
2. **The 0/13 false-positive rate on `scratchpad/holdout_arcint` generalises.**
   WEAKENED. A corpus whose levels fall to 42 random actions is exactly AERA's
   non-intelligent-strategy class; it is easier than the competition set.
3. **"tu93 passes the live gate"** (from `live_sim_seconds`). FALSE —
   `live_sim_seconds` is the OPTIMISTIC offline model and is SUPERSEDED. The
   authoritative figure applies the **measured x2.66 competition guard tax**
   (ENVELOPE-2026-08-26-v8 §D1). tu93: nbfs_macros 1,733 s (15.5% over the
   1,500 s cap); **nbfs 1,510.1 s (0.67% over)**.
4. **"`qthdiggudy` is a carrier-only obstacle wrongly counted as an avatar
   wall."** FALSE, refuted by my own regression test — it appears in
   `fuykgiiwit` too, so it blocks the avatar. Subtracting it regressed wa30 L3
   from an 82-move win to NO PLAN in 495 s. `is_collidable` is NOT the
   passability predicate.

## 7. WHAT WAS BUILT AND VERIFIED (08-28)

- **wa30 L3 SOLVED** — 82 moves, engine-verified clean replay, reproduced
  twice; chain clears L1-L3 of 9. *Caveat: per §2 this lane's ceiling is 0.9 LB.*
- **`wa30_carriersim.py`** — reproduces the carrier drive EXACTLY in lockstep
  with the engine on L1-L4 (every carrier/block position and carry relation,
  every tick).
- **`nbfs`-first dispatch** — 12.9% cheaper on tu93 (84,685 -> 73,804 actions)
  at **zero level cost** (48 levels either way). Closes a gap three prior
  attempts missed. Still 0.67% over the cap.
- **`prescreen.py` branch-A thresholds corrected** (`MIN_STRICT=3`,
  `MIN_STRICT_LATTICE=2`); `n_strict` is categorical across 38 games (ft09 32,
  cx01 3, all 36 negatives 0). 44/44 tests pass.
- **`EXPLORER_OWNED_TIME_S` 1500 -> 1700** — envelope-checked (`remaining_s`
  takes the MIN incl. the unchanged cumulative 2700 s pool, so total grind wall
  time cannot rise). **INERT while `EXPLORER_V8_STALL_GRIND=0`.**

## 8. LAWS ADDED

1. **Read levels off the ENGINE, never off a solver's return dict.** The cx01
   error came from trusting `solved=True` without checking `levels_completed`.
2. **`is_collidable` is not the passability predicate.** Read the predicate the
   engine actually calls.
3. **A clone that executed ZERO actions is INVALID, not a failure.** It tested
   the harness, not the model.
4. **Validate a guard on the path that FAILS, not the path that works.** The
   08-29 slot aborted because `WINDOW_END_UTC` was left at 08-28 while
   `TARGET_UTC` moved — and `--mock` *skips* the window check, so the mock
   passed.
5. **A result that confirms your prior is when the instrument needs checking
   hardest.** The 27B v1 run returned a tidy 0/30 that agreed with the pilots
   and was pure instrument failure.
6. **`kaggle kernels output` returns the COMMIT run, not the scored rerun.**
   Check `benchmark.json:label` first. There is NO channel out of a scored
   rerun.

## 9. INSTRUMENT FAILURES THIS SESSION (so they are not repeated)

- probe printed a fabricated `FAIL` on an LLM timeout -> errored clones now
  excluded, `NO VERDICT` when none are valid
- **27B kernel v1 burned ~4 h GPU and measured nothing**: server runs with
  `preserve_thinking: True`, `max_tokens=512` was consumed entirely by the
  reasoning block, so 400 calls produced no parsable `ACTIONS` line. Fixed four
  ways in v2 (validity guard, preflight assert, max_tokens 3000 +
  `preserve_thinking: False`, raw-reply capture). v1's numbers kept as
  `v1_INVALID_results.json` so they can never be cited.
- **08-29 slot missed at 00:01Z** by the window bug above; recovered and
  submitted 07:11Z the same UTC day.

## 10. OPEN DECISIONS

1. **`EXPLORER_V8_STALL_GRIND`** — the raised cap is inert until this flips.
   Prior evidence is negative (292k engine actions, zero cracks in the earlier
   smoke). Not taken.
2. **Whether to keep spending on the specialist/crack lane at all**, given
   Le Grand's measured 0.9 LB ceiling for exactly that approach.
3. **Whether wa30 deployment is worth finishing** — the port needs macro
   evaluation moved into `wa30_carriersim` so only the committed plan touches
   the engine (backend port priced at **92,517 reset-replay actions for L3
   alone**, 27% over the whole budget; committing only the final plan costs 82).
