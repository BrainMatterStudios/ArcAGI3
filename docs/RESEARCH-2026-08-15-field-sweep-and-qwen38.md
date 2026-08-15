# RESEARCH 2026-08-15 — Full-field sweep after the leaderboard surge

Three parallel research lanes (Franzen deep-dive, field/community intelligence,
academic/technique sweep) run 2026-08-15 under Ahmed's directive: *"do
everything possible to win — research the competition and identify ideas and
techniques we can adopt to beat them."* Goal restated: **beat cstl's 2.70** on
the public LB (milestone-2, Sept 30, is judged on the PUBLIC leaderboard) and
win the final (private 55, Nov 2).

## 1. The single most important finding: Qwen 3.8-27B (2026-08-14 14:44Z)

**Verified first-hand** (HF API + config.json, not just agent reports):

- `Qwen/Qwen3.8-27B-FP8` published 2026-08-14T14:44Z, Apache-2.0, official FP8.
- Architecture **identical in class and shape** to the 3.6-27B we serve:
  `Qwen3_5ForConditionalGeneration`, model_type `qwen3_5`, 64 layers, 5120
  hidden, full_attention_interval 4, same GDN linear-attention geometry, same
  attn_output_gate. It is a **retrain of the architecture our pinned wheelhouse
  (vllm 0.19.0 / torch 2.10.0 / flashinfer 0.6.6) already serves**.
- Deltas: quant_method `fp8` (native block-128; vLLM 0.19 supports it; H100 and
  the Kaggle RTX Pro 6000 both have hardware FP8) vs vrfai's compressed-tensors;
  transformers stamp 5.8.0.dev0; **vision encoder included** (multimodal),
  262k native context.
- Benchmark jumps over 3.6 (press coverage, unverified by us): Terminal-Bench
  63.4→73.0, OSWorld 63.9→84.3, SWE-MM 25.7→38.6.

**Circumstantial but strong LB evidence**: Franzen 2.58 (08-14 21:37Z),
Sorokin 2.10 (08-14 19:30Z, only 6 entries total), AbeLincoln1865 1.90
(08-15 00:22Z, 7 entries) — **all within ~10h of the model drop**, after
Kojima's 1.86 had stood for weeks. Kaggle forum thread 735243: one member
reports a **consistent 2× on the local 25** for 3.8-27B-8bit vs 3.6-27B-8bit
in a duck-class harness. cstl's 2.70 PRE-dates the release (08-13 20:08Z) —
their edge is something else; the chasing pack is likely the model.

**Why this matters to our math**: base distribution mean 0.9650 (n=11), and
every incremental lever we built died under measurement. A brain swap that
multiplies the mean is exactly the "+0.4 minimum, architecture-class" lever
MEMORY says is mandatory — and it composes with everything else.

**Adoption lane (in flight as of this doc)**:
1. Modal `arc3-vllm38` app: byte-identical scored serving stack, model swapped —
   empirical serve-compat + smoke + tool-calling check, then an A/B wave.
2. Local snapshot download → Kaggle dataset `qwen3-8-27b-fp8-hf-snapshot`
   (upload_qwen38_snapshot.sh, pattern proven by the RedHatAI lane).
3. `duck-38` arm: duck-base v2 + dataset swap + serving-config model-path swap.
   Single variable. Slot candidate as soon as commit-attested.

Risks to pre-register: (a) serving may need flags 0.19.0 lacks for the new fp8
scheme → the Modal test settles it before any Kaggle spend; (b) tool-calling
template drift (qwen3_coder parser) → smoke includes a tool-call check;
(c) the 2× forum report is one person, n=1 local; our A/B wave is the evidence
that counts; (d) 3.8's press numbers are agentic-computer-use benchmarks, not
ARC; transfer is plausible (the duck is a tool-calling agent loop) not proven.

## 2. What the top teams are actually doing

- **cstl 2.70** (pre-3.8): still zero public footprint. Their profile
  (bot-competition winners, harness iteration 1.46→2.70 in ~9 subs) plus the
  field's convergent evidence says: engineered harness discipline on top of a
  local brain, iterated against the public LB. No shortcut to steal before
  Sept 30's forced open-source.
- **Franzen 2.58**: dark entry, but his entire published toolkit is
  model-agnostic and cheap (see §4): PoE likelihood scoring over augmented
  views (+5.1pp in ICML'25 paper), test-time LoRA training (+26pp there — but
  see §4 caveat), DFS decoding with probability-threshold pruning, radical
  vocab constraint. Publishes after competitions; open-sourced for prize money
  both prior years → expect his methods Sept 30.
- **Sorokin 2.10 / AbeLincoln1865 1.90**: zero public footprint; entry-count
  and timing patterns say strong operators riding the model release.
- **Tufa Labs 1.62 @ 107 entries**: grinding, not leaping. Their public Duck
  harness IS our base — their research page has nothing new for ARC-3 since.

## 3. Field techniques ranked (external evidence × our internal measurements)

Ordering principle: our binding constraints are **wall-clock** (9/10 rig games
exhaust it), the **(b/a)² efficiency term** (15/19 completed levels in
>3×-budget games ≈ 0), and **depth zeros** (most games score 0 levels).

**Tier A — act now**
1. **Qwen 3.8 swap** (§1). Mean-multiplier candidate; composes with all below.
2. **Duck memory-capture bug audit** (forum 734843, Jason Feng): duck captures
   persistent-memory updates only from VISIBLE output; reasoning models put
   them in hidden reasoning — one instrumented run: 66.8% of tool responses
   had hidden reasoning + zero visible content. **Bears directly on our killed
   duck-mem bundle (pooled −0.29 may have been measured on a
   dropped-writes harness) and on every memory-class lever we ever test.**
   Audit our shipped bytes + rig traces; fix is trivial (capture from
   reasoning channel or force visible echo). Fixed notebook is public.
3. **Retrodict-style explore-then-commit + expectation-checked action queues**
   (repo public, official scorecard 99.86% RHAE at 5.5× fewer tokens than
   baseline1): probe single actions while uncertain → the moment a mechanic is
   confirmed, batch all predictable actions into a queue where each step
   carries expected board cells; runner plays it mechanically, LLM re-invoked
   only on mismatch (with diff) or exhaustion. Attacks tokens-per-action AND
   actions-per-level — both binding. Frontier-model evidence, but the
   mechanism is harness-side. Our struct plan-channel was adjacent (plans
   adopted 2.4× but score-flat) — the NEW ingredient is expected-state
   verification per queued step + LLM-free playout. Medium build; rig-testable
   offline in days.

**Tier B — build behind the Tier-A readouts**
4. **Zero-LLM graph-frontier explorer floor** (just-explore, MIT, 3rd in
   preview with ZERO LLM calls; post-fix median 17/25 levels): frame-hash
   graph + frontier walk converts 0-level games into 1-2-level games — attacks
   the depth zeros that dominate our mean. CAUTION: sonpham's LLM-coupled
   state-graph measured negative 3×; the pure floor variant is untested by us.
   Single-variable rig arm.
5. **Effect-signature uncertainty steering** (OPINE-World §core): per-object
   Dirichlet effect posteriors + noisy-OR entropy → probe highest-uncertainty
   objects. O(1) Python per transition, no LLM calls. Sub-human-budget
   completions are the only regime where (b/a)² pays (our tn36 0.62× proved
   it matters). Low-medium build.
6. **Animation-frame metadata tier** (Taaf Anim, 16th): duck drops all
   mid-animation frames (`raw.frame[-1]`); 13/25 public games are multi-frame;
   compact metadata (counts/diff-bbox/transient pixels) is a few dozen tokens.
   Their public A/B was flat (+1.4%, p=0.92) but they credit it on private.
   Cheap; token-cost caution under wall-clock.
7. **Hard no-op guard** (two independent implementations): block exact
   (level, board, action) repeats that already no-op'd. Trivial; small
   non-negative expectation; overlaps our struct ground (measured ≈0) — only
   as a rider, never a slot.

**Tier C — watch / probe / decline**
8. **`environment_info.baseline_actions` exposure claim** (Cottaar, 18 votes,
   hosts silent 4 months): if the 2nd-best-human action count is really
   visible at eval, per-game action budgeting can target the (b/a)² term
   directly. One cheap read inside an existing arm's logs settles it.
   Log-probe only until confirmed.
9. **Franzen PoE scoring / DFS decoding**: mechanism verified in his papers,
   but our best-of-N law ("pass 0 eats full budget; candidate generation was
   never the binding constraint") and the wall-clock budget make 16×-scoring
   passes expensive. Revisit if 3.8 makes per-token cost cheaper (faster
   decode) or after queue-batching frees tokens.
10. **Test-time training / test-time RL** (Franzen's biggest ARC-1/2 lever):
    per-game gradients don't fit 5-min/game; serving-side LoRA hot-swap is
    exactly where we've been burned (silent-serving law); an independent
    replication (Nosumina) confirms ≤96GB local world-model synthesis fails.
    DECLINE for now; revisit only as cross-game LoRA trained offline between
    submissions (that's just fine-tuning, which our serving gate governs).
11. **Policy-superposition / value-ledger prompt packs** (Feng, +83%/+139%
    self-reported single runs, author ranked 216th): exactly the class our
    four-lever campaign killed. Rig A/B at most, never a slot on this evidence.
12. **VISTA-style vision channel** (frontier: 100 RHAE w/ Opus 5; 64→512px
    upscale is the key trick; Qwen 3.6 failed coordinate inference): dormant
    UNTIL the 3.8 swap lands — 3.8's vision gains may unlock it locally. Pair
    with `MULTIMODAL_CONTEXT` plumbing that already exists in the setup env.
13. **Tycho community fix**: dead (0 issues/PRs; nobody fixing 64k defect).
    CLOSED, matches our kill.

## 4. Nulls and cross-checks banked this sweep

- Nobody in the community explains cstl as a paradigm; the public set is
  saturated at frontier-API scale (Retrodict 99.86% @ $654) — differentiation
  there is tokens-per-point, our exact axis.
- Nosumina independently replicates our Track-1/2 core finding (≤96GB local
  models can't sustain executable-world-model synthesis).
- AERA confirms the explore-floor thesis but underperforms just-explore.
- OpenAI's "two settings" post (retained reasoning + compaction = 3×, 6× fewer
  tokens on GPT-5.6 Sol) independently confirms the duck knowledge-wipe defect
  class we identified; converges with Prime-Agent compaction findings. The
  duck-mem P2 (middle-drop) was our version; audit item 2 (memory capture) may
  explain why our measurement disagreed with everyone else's mechanism prior.
- yw8837's public 11-submission patched-duck series (0.55-1.29) independently
  replicates our base-variance band; their 300-game diagnostics ledger is free
  external control data (download queued).
- Official model LB: Opus 5 tops harness-free at 30.2% — the gap between that
  and VISTA's 100 RHAE is ALL harness; harness remains the fight.

## 5. Slot plan (each slot = pre-registered experiment, per doctrine)

- **08-16 00:01Z (armed)**: duck-p3 v2 — single-variable prompt arm, attested
  (hash 06e74d28, svid 342532400). Reading rule pre-registered (band 0.69-1.30).
- **08-17 (target)**: duck-38 — Qwen 3.8 swap, single variable vs duck-base v2,
  contingent on Modal serve-compat PASS + Kaggle dataset ready + commit attest.
  If the local-25 A/B replicates anything like 2×, this is the new floor and
  EVERY subsequent lever re-tests on top of it.
- **Parallel, no slot**: memory-capture audit (bears on past verdicts);
  Retrodict queue-batching build on the rig; explorer-floor rig arm.

## 6. Standing corrections this sweep forces

- "Nothing open ≤96GB out-benches the 27B" (Track-1 brain scout) is now
  **stale as of 08-14**: it predates Qwen 3.8-27B. The scout conclusion was
  correct when written; the 3.8 lane is the update path.
- The duck-mem kill (pooled −0.29) is **under audit** pending the
  memory-capture bug check: if our harness dropped 2/3 of memory writes, the
  bundle was never actually tested. Do not cite that kill as final until the
  audit lands.
