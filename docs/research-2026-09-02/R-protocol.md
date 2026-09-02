# protocol — final report (2026-09-02)

## A. The frontier dataset — what the protocol is
- hf_traces/ (1,058 files, 50 trajectories: 25 Claude Opus 4.8/Fable 5, 25 GPT-5.6 Sol). README: Claude 183/183 levels, 98.98% RHAE; GPT 182/183, 95.35%.
- Harness "Schema" (Impossible Research; schema-harness.github.io) = Claude Code / Codex session with MCP tools write_file/edit_file/read_file/run_python/run_shell/read_history/run_backtest/run_bfs/commit_actions. Every user message = full 64x64 grid as hex + notes.md verbatim + result of last commit ("committed 4 action(s) — executed 1; MISPREDICTED … rest dropped"). Memory = notes.md + world_model_vN.py on disk. Framework source REDACTED in the dataset (25/25 event logs).
- World-model contract: init_state(entry_grid), predict(state, grid, action, x, y) -> (state, grid, flags), is_goal(state, grid). run_backtest replays over every recorded transition, demands exact grid + flags. run_bfs(target, max_depth, max_nodes) searches inside the model (sk48 L0: 16 steps, 33,254 nodes). Only commit_actions touches the game; a mispredict voids the queue.
- Census (characterize.json): Claude per game median 4.6 h, 63 turns, 403 tool calls, 334 actions, 2.49M output tokens, 448M prompt tokens (98% cache-read), peak context up to 858k. Sum over 25 Claude games: 81.2M output tokens. GPT median 2.7 h, 77 turns, 504 tool calls.
- World-model code 9–92 KB per game (GPT bp35 170 KB). Mispredictions per game median ~35 (up to 328).

## B. Games our stock stack almost never clears (8 phases)
No game is 0/8: sk48 1/8, dc22 1/8, g50t 1/8, tn36 3/8, cn04 3/8 (tr87 5/8). Pooled 9/40 = 22.5%.
| game | frontier L0: actions/turns/min/output tok | what cleared it | stock 27B (stock_b) |
|---|---|---|---|
| sk48 | 42/9/30/281k | probe with no-change placeholder model; encode rope ±6, chain-push; backtest 26/26; BFS 16-17 steps; 243-line predict | 41 actions, 18 turns, 0 — prose model reached the same mechanic but never planned |
| dc22 | 50/10/31/305k | movement+click probes → pour/swap rules; backtest 31/31; 18-action route; 105-line model | 138 actions, 32 turns, 0 |
| g50t | 138/37/167/2.05M (GPT: 71 actions) | 13 turns single-action probes; 8 mismatches fixed by backtest | 123 actions, 13 turns, 0 |
| tn36 | 12/6/26/164k | 3 clicks → program&run model (114 lines) | 48 actions, 29 turns, 1 |
| cn04 | 18/5/27/355k | backtest 2/2 → BFS 15-step plan | 41 actions, 14 turns, 1 |
Action budget is not the difference. The difference = (a) certified executable model + (b) search inside it — both absent: prompts.py "Every python tool call starts fresh"; no numpy/files; tool_agent.py:1113-1126 WIPES world/goal/action model text at every level transition; context window 32768, 30 assistant turns retained; raw grid hidden (letter ASCII + segmentation only).

## C. Cost profile vs our box
Stock (arc3-v22-aa stock_b): 25 games 2h12m, 2,570 actions, 3.89M generated tokens (~156k/game); vLLM gen 563 tok/s aggregate at ~24 running, prefill 4.3k tok/s, prefix hit 14%. Per game 34–93 calls, median 94–192 s per call (2–3k reasoning tokens).
Frontier median LEVEL 0 alone: 109 calls, 322k output tokens, 11.3M prompt tokens at 90–180k context. At ~22 tok/s/game that is ~4 h of generation for one L0. 95% of frontier output tokens are hidden thinking. A transplant must fit ≤~140k generated tokens and ≤32k context per game (~1/18 of frontier per-game generation).
Cheap protocol parts: backtest (ms–s CPU), BFS (s), file persistence, diff rendering, commit-halt. Expensive: long-context deliberation.

## D. Direct prior evidence the protocol does not transfer to a 27B
scratchpad/tycho_eval/tycho/results/: Tycho run locally 2026-08-14 with vrfai/Qwen3.6-27B-FP8: ft09 → 0 levels (120 calls, 22 actions, builder "inert"); sb26_v2 → 0 levels; ls20/sb26 HTTPError partials; KAT-Coder-V2.5 sb26 → 0 levels in 90 actions. Stock duck clears ft09 2–4 and sb26 1–6 in the same wall time. Caveats: Qwen3.6 not 3.8, effort=medium, 4,500 s, n=1.
Tycho ablation on frontier: Opus 4.8 no world model 79.07 RHAE vs 88.49 with. Retrodict beats baseline1 on the same model by 0.89 pp. The protocol adds ~10 points at the frontier; the model supplies ~80. Schema's own comparison: plain Claude Code 42.83% vs Schema 98.98% (but that baseline is still a frontier model writing code with files).

## E. Fine-tune route
Claude sessions: 26,692 assistant records, 10,285 tool_use, 7,928 text blocks; all 8,486 thinking blocks EMPTY; GPT reasoning encrypted. Corpus teaches format/tool protocol only. No license (cardData None, no LICENSE). Not a measurable step-change lane.

## Ranked step-change candidates
**1. "Protocol-lite" graft into the stock ToolAgent** — keep 27B/32k/June loop; add (i) persistent world_model.py across calls/turns/levels (stop the wipe at tool_agent.py:1113); (ii) backtest() over all recorded transitions with exact-diff report; (iii) bfs(is_goal, max_nodes) inside the model; (iv) commit(actions) halting on first mispredicted frame. Raw numeric grids + numpy in sandbox.
- Ceiling: L1 on the five hard games always = only +0.16 lv/game (under the bar). The bar is reachable only if certified models CHAIN levels on games we already clear once (frontier: 14/25 games with exact models used 1.6–5× fewer actions than humans). Hypothesis to test = "the 27B can certify a backtest-green model", not "clear one more level".
- Contradicting: D (0/4 local Tycho-27B runs, worse than stock); 27B forms correct prose hypotheses but has 2–3k tokens per 2-minute call; frontier L0 needs ~109 calls, we afford ~60 per game.
- Cheapest decisive experiment:
  - Stage 0 (30–60 min GPU, zero game runs): feed the 27B the frontier's recorded sk48/tn36/cn04 transitions (before/after grids + actions) with the predict/is_goal contract and a backtest tool, 32k context, ≤12 tool steps/turn, ≤20 turns. KILL: cannot reach backtest-green on ≥2 of 3 with the answers handed to it → lane dead.
  - Stage 1 (one GPU kernel 2.2h, 0 slots): paired phase, concurrency 20: protocol-lite × {sk48, dc22, g50t, tn36, cn04} × 2 draws + stock × same 10 in the same phase. RULE: protocol arm clears L1 on ≥6/10 (P=0.012 under 22.5%) AND leads stock by ≥3 AND backtest-green model on ≥2/5 games.
**2. SFT on the schema traces** — not a step-change lane (no reasoning text, no license, serving gate never passed).
Everything else (long-context transplant, Tycho/Retrodict as-is, 150k-reset playbooks) throughput-infeasible by C and refuted by D.

## Could not verify
Schema harness source (redacted); whether local Tycho-27B runs had thinking enabled; Kaggle discussion pages (JS-rendered); GPT token usage.
Files: scratchpad/search/protocol/{stock_aa/, characterize.py, characterize.json, schema_blog.html, hf_traces/}.
