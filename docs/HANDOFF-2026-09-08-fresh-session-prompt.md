# Fresh-session prompt — 2026-09-08 (execute the revised plan, starting with Track A1)

Paste the block below as the first message of the new session. Everything it needs is committed on
`winning/duck-patched` (last commit b42bb94 + this file). Nothing is pushed.

```
CONTEXT — ARC-AGI-3 Kaggle competition (arc-prize-2026-arc-agi-3), repo /Users/ahmed/Documents/ArcAGI3,
branch winning/duck-patched, use .venv/bin/python for everything. Deadline Nov 2 2026. Today's date: run
`date -u` first. Read, in this order, before doing anything:
  1. docs/PLAN-2026-09-08-revised-plan-to-7plus.md  (the plan you are executing)
  2. docs/research-2026-09-08/R-score-arithmetic-0908.md, R-field-refresh-0908.md, R-reopen-audit-0908.md,
     R-model-axis-0908.md  (the evidence behind it)
  3. offkaggle/REGIME_WAVE_STATUS.md  (the Modal rig: every arm ever run, pre-registrations, results)
  4. submission/_throughput_v1/PROBE_STATUS.md  (the graft convention and the last arm, as a worked example)
  5. memory: /Users/ahmed/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-revised-plan-2026-09-08.md
     and arcagi3-modal-regime-rig.md. Memory is a catalog of what was TRIED, not what is TRUE; anything you
     rely on for a decision, re-verify against code, transcripts or the Kaggle API.

STATE (verified 2026-09-08 ~16:00Z):
- Live: our family = byte-copy of keithtyser V14 (Flash-Next NVFP4, profile kv5-bf16-mtp3-c8-cg32, stock
  June duck) with draws 3.25 / 2.58 (base v4, svid 347562879) and 4.31 / 2.45 (yield900 knob, svid
  347926973). Mean 3.15, sd 0.85. Leaderboard: Tufa 11.04, Third Intelligence 8.21, Franzen 7.63,
  mostik.ai 7.51, NVARC3 5.96 (3 entries). Nothing public scores above 4.33 (a byte copy of what we fly).
- Armed: yield900 draw 3, runner scripts/submit_keithyield900_20260909.py (pid 51806, log
  logs/keithyield900_runner_20260909.log), fires 2026-09-09 02:00Z; it completes the pre-registered 3-draw
  rule; read ≤ 4.04 = dead, revert to v4 as the control. Record its score in docs/submission-ledger.json
  and docs/HANDOFF-2026-08-29-packs-in-flight.md (§33 addendum) when it lands (~11:15Z).
- Modal rig: offkaggle/modal_flashnext_serve.py is deployed (base, no overrides; scale-to-zero); run the
  modal CLI as `SSL_CERT_FILE=$(.venv/bin/python -m certifi) /Users/ahmed/Library/Python/3.14/bin/modal ...`.
  Wave runner: offkaggle/run_regime_wave.py (arms, --knob, --draws, --games, --max-calls, --dry-run;
  identity gate; VOID rules). Base for comparison: pooled six-draw 39.3 levels (sd 2.34) per 25 games.
  Same-regime waves cost ≈ $9 and 2.25 h; 3-game kill tests ≈ $3. Modal preemptions VOID a wave.
- Kaggle: 60 h/week GPU quota (resets Sat 00:00Z); ONE ARC GPU session at a time; never stop Ahmed's
  non-ARC (rsna) kernels; both mount layouts must be handled (see submission/_keith_copy/push_bothmounts).

THESIS you are executing (from the plan): the agent is model-starved, not action- or time-starved. 98 % of
lost points are levels never reached; 60 % of each game's clock is spent stuck at a wall after a
baseline-sized action budget; efficiency is capped (≤ 0.8 LB headroom; 2× baseline actions ⇒ ×0.25).
ARC Prize showed GPT-6 Astra 62.7 → 99.9 purely from keeping reasoning state between calls + compaction;
NVIDIA's and Polyphony's open agents persist an executable, verified game model. The stock duck rebuilds
the sandbox every call (python_tool_sandbox.py main() builds fresh runtime_globals) and the Qwen chat
template drops earlier-turn reasoning (assistant messages carry "reasoning", tool_agent.py ~1890, but the
template keeps it only after the last user message; each duck turn opens a new user message), so at
1.7 calls/turn the model's thoughts and computed state are wiped about every second call.

FIRST TASK — Track A1 "carry + compact" graft, then its 3-game kill test on the rig:
  a. Build submission/_throughput_v1/graft_carry.py in the graft convention (module _STATE/_STOCK,
     install() wraps ToolAgent methods from inference.agent.tool_agent, flags read at call time, master flag
     CARRY_ENABLE "0" = byte-identical stock, status() counters, transcript markers, exceptions never crash
     a turn). What it does: (1) carry the model's prior-turn reasoning forward in a form the template
     will not drop — verify FIRST, from the rig's prompt logs / a rendered-prompt probe against the live
     endpoint, whether reasoning in earlier assistant messages reaches the model under the Qwen3.8 template;
     if not, re-inject it as assistant text (or a bounded "[prior reasoning]" block); (2) replace the
     oldest-turn eviction (_drop_oldest_history_block / _keep_recent_history_turns, tool_agent.py ~1608-1650)
     with a compaction step: when the window is near 32k, ask the model (one extra call, counted) to write a
     compact world/goal/action-model summary of the turns about to be dropped, and keep that summary as a
     persistent block; (3) budget: prompt tokens/call must stay ≤ 32k (context_budget_tokens 31744) — log
     them. Tests (unittest, both bundles: scratchpad/bundles/june_stock/... and
     submission/_inspect_replay/assets_build/ARC3-Inference): flag-off byte-identical; reasoning present in
     the next request; compaction fires and the summary persists; window never exceeded; exception safety.
     A real-engine dry run like dry_run_probe.py with a mock brain.
  b. Add arm `keith_carry` to offkaggle/run_regime_wave.py = keith_yield900 env + CARRY_* flags; extend the
     summary with a CARRY line (reasoning-carried calls, compactions, prompt tokens/call, window overflows)
     and an ENGAGED gate; tests green (offkaggle/test_run_regime_wave.py).
  c. Pre-register in offkaggle/REGIME_WAVE_STATUS.md BEFORE launch: engagement gate, PRIMARY = levels vs
     39.3 (sd 2.34) with the bands ≥ 48 step candidate / 45–47 counterbalanced redraw / ≤ 44 or
     engaged-but-flat dead, co-primary = walls passed among the 12 six-draw-never-passed walls (list in
     REGIME_WAVE_STATUS.md), safety = GAME_OVERs/run and live-cap score, fit-the-clock (calls × e2e ≤ 7,920 s),
     VOID rules. Then a 3-game kill test (tu93, vc33, ft09 or the ledger's wall games; 2 draws, ≈ $3)
     first in a fresh boot; only if it is engaged and not worse, the 25-game wave (≈ $9), first in a fresh boot,
     with a Monitor. Read strictly by the pre-registration. Commit results (offkaggle/results/<stamp>-keith_carry)
     and the RESULT block.
  Then continue with A2 (persistent workspace: sandbox globals survive across calls within a game, a queryable
  transition log, files the model writes persist, a backtest tool), and A3/A4 (port NVIDIA NOOA
  github.com/NVIDIA-NeMo/labs-OO-Agents and Polyphony github.com/Mininglamp-AI/polyphony-arc-3 to the Modal
  endpoint in Kaggle geometry; the clock gate is the first read). Track B/C/D per the plan.

LAWS (do not re-learn these):
- A 25-game total has sd ≈ 2.3 levels; a single wave ≤ 42 is flat; never read a lever from one draw.
  Behaviour-shaping levers on this loop are CLOSED (yield900, probe discipline, retry, evidence, hypothesis,
  upscale-8, half-concurrency); do not rebuild them. Everything in R-reopen-audit's "stays dead" list stays dead.
- Every wave runs first in a fresh Modal boot; identity gate must print the kv5 profile on an RTX PRO 6000;
  vLLM KV preemptions ~110–180 are a regime constant (not a VOID); Modal container preemptions are a VOID.
- Never launch a Kaggle submission, push, merge or deploy without Ahmed's explicit go in this session.
  Never write a secret into a committed file, transcript or log (the Modal bearer token lives in
  ~/.config/arc3/vllm_token; never print it).
- Verify after patching: assert the file exists / grep the change; compile-check every notebook cell.
- The harness writes compact JSON in events files; never pre-filter on '"key": value' spacing.
- Narrate progress; commit to the working branch; stop and report at each gate.
```
