# R11 — Mining the Schema traces (HF `schema-harness/arc-agi-3-schema-traces`)

**Date:** 2026-08-31. **Local copy:** `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/336208eb-4625-4ff9-b10d-1664cc846e5d/scratchpad/schematraces/ds` (739 MB, full snapshot via `huggingface_hub.snapshot_download`; the plain `git clone` leaves LFS pointers — git-lfs is not installed on this host).

> Ops note: the download initially failed because the Mac was at 100% disk (116 MB free). Freed ~12 GB by purging the `uv`/`pip` package caches. Watch disk before large downloads.

---

## 1. Verified scores and provenance

### 1.1 Scorer run (their `score_trajectories.py`, stdlib-only, run locally — verified)

`python3 score_trajectories.py --compact` exits clean with: **"Validation: all event-derived scores match both evaluation manifests."** The scorer streams all 50 `events.jsonl` files, reconstructs per-level action counts, and recomputes RHAE with the exact competition formula (`min(115, 100·(h/a)²)` per level, level-index-weighted mean, completion cap).

| Collection | Trajectories | Wins | Levels | Mean RHAE |
|---|---:|---:|---:|---:|
| `gpt_5_6_sol` (GPT-5.6 Sol via codex-cli) | 25 | 24 | 182/183 | **95.35%** |
| `claude_fable_opus` (Opus 4.8 ×14, Fable 5 ×11, via Claude Code) | 25 | 25 | 183/183 | **98.98%** |
| ALL | 50 | 49 | 365/366 | **97.17%** |

Per-game (RHAE %): every game is 100.00 except — GPT: bp35 60.93, ka59 65.34 (STOPPED at 6/7 levels — the single non-win), sc25 82.72, sk48 87.80, tn36 87.02; Claude: bp35 93.51, dc22 98.70, s5i5 89.87, sb26 98.63, tn36 94.74, vc33 99.10. Notably GPT's 16 `xhigh`-effort runs are all 100.00; all five sub-100 GPT games ran at effort `max`. Sub-100 scores are pure inefficiency (e.g. GPT bp35 used 1,347 actions vs human 651), not failure — 49/50 runs WIN all levels.

Runs executed 2026-07-15 → 07-19; released ~Jul 16; ~2,520 downloads/month, 36 likes as of today.

### 1.2 Who they are

- HF org `schema-harness` members: **Xiuyu Li** and **Guanning Zeng**. The project ("[schema]") is an academic effort from **Andrea Zanette's lab (Berkeley/CMU)**, led by Guanning Zeng (Zanette's incoming PhD student), with Haven Feng (@HavenFeng — the "havenfang" lead in the task is this person), Jiani Wang, Wenjie Ma, Shaofeng Yin, Chenyang Wang, Angjoo Kanazawa, et al. Blog: `schema-harness.github.io` ("Frontier Models with Our Harness Achieve ~99% on ARC-AGI-3 Public").
- **They are NOT on the Kaggle leaderboard.** I downloaded the full public LB today (2,671 teams, via `kaggle competitions leaderboard --download`): no team matching schema/Zeng/Zanette/Feng/Xiuyu/Guanning/Berkeley. Their result is public-25, unlimited wall-clock, frontier-API models — all three of which the Kaggle eval forbids/lacks.
- Field context (today's LB): cstl **7.51**, Lord Han Solo 4.99, **Tufa Labs 4.71**, Son Pham & Mark Barney 4.42, Tong Hui Kang 4.27, Franzen 4.05 — a large surge since our 08-15 notes (then four teams ≥1.86). This dataset has been public since mid-July with heavy downloads; assume the leaders have mined it.
- Mirror `JBrightmanAI/arc-agi-3-schema-traces` exists, marked "Duplicated from schema-harness/…", also **no license**.

---

## 2. The Schema harness protocol (dissected from primary data)

Trajectories examined in depth: Claude ls20 (100.0), Claude wa30 (100.0), Claude s5i5 (89.87, worst Claude, 32 h), GPT tu93 (100), GPT re86 (100), GPT bp35 (60.93, worst finished GPT). Sources per trajectory: `run.json`, `events.jsonl` (full event stream incl. every 64×64 grid), `sessions/session-001.jsonl` (the raw Claude Code / codex_exec session log, sanitized), `notes.md`, `world_model_v5.py`, `cand_*.py`, `snapshots/cleared_level_N.py`.

**Architecture: it is Claude Code / Codex CLI as the agent runtime**, connected to a custom MCP server ("locus" on the Claude side, "game-server" on the GPT side) that owns the live game connection. The developer/system instructions are `[redacted: runtime instructions]` in the release, and the harness `framework/` source (mounted read-only in the agent workdir) is NOT included — the protocol below is reconstructed from prompts, tool calls, and events, and is complete enough to reimplement.

### 2.1 The turn loop

1. **Observation (user message, plain text, no images).** Exact format:
   - Header: `State: NOT_FINISHED | level 4/7`, `Legal actions: [1, 2, 3, 4]`, `World model: installed; history: 248 transitions.`, `Files: workdir (read/write) = ~/agent-<game>; framework source (read-only) = ~/agent-<game>/framework.`
   - **Last-turn accounting**: `committed 3 action(s) [1 1 4] — executed 3; stopped because the world model MISPREDICTED this step … rest of the plan dropped. Net: level 4→4, state NOT_FINISHED→NOT_FINISHED.`
   - **Echo of the model's own stated intent** from the previous commit (`Your stated intent was: "…"`).
   - **Echo of the model's note-to-self** (`Your note-to-self from last turn (reconsider, don't just obey): …`) — this is the `suggestion` field of the previous `commit_actions` call.
   - On surprise: `NOTE: world model MISPREDICTED the step just taken (action 4); … Run run_backtest to see the mismatch and fix the model before planning again.`
   - **The agent's own `notes.md` inlined, truncated at 4,000 chars** (`… (notes.md truncated at 4000 chars — prune it with edit_file)`).
   - **`Current grid:` as 64 rows of 64 hex digits** (`shape=64x64 (values 0-15 as hex)`).
2. **Agent works in-tools** (any number of calls): `run_python` (plain inline python, prints back), `read_file`/`write_file`/`edit_file`/`grep`/`cp`/`mv` over the workdir, `read_history` (recorded transitions), `run_backtest`, `run_bfs`.
3. **`commit_actions({actions: [{action, x?, y?}…], reason, suggestion})`** ends the turn. `action` 0=RESET, 1-4=directions, 5=?, 6=click(x,y), 7=?. The harness then executes the batch **one action at a time against the live game, checking each real next frame against `world_model.predict()`**; on any mismatch it drops the remainder of the plan and raises `model_mispredicted`. A turn ending without a commit logs `turn_fallback` (no action taken, warned next turn).

### 2.2 The world-model contract (the core mechanism)

The agent maintains an executable model in its workdir (numpy allowed): `init_state(entry_grid)` and `predict(state, grid, action, x=None, y=None)` (plus optionally `is_goal`), predicting **the full next 64×64 grid AND level_up/dead/win flags**. Three harness services close the loop:

- **`run_backtest(level='current'|…, max_details)`** — replays every recorded real transition through the current model file: `backtest [level 4]: 51/52 transitions fully correct …; 1 mismatch(es)`, then prints the mismatching transitions in full (before-grid, action, predicted vs actual, cell diff count). Agents iterate `edit_file` → `run_backtest` until "backtest green", and keep **alternative hypothesis models as `cand_*.py`** files backtested side by side (wa30 has ~20 candidate models).
- **`run_bfs(target='advance', max_depth, max_nodes)`** — harness-side BFS **over the agent's own model** from the current live state: `BFS: goal in 13 step(s) via level_up; expanded 7558 nodes, 2357 distinct states (actions=[1,2,3,4] + RESET-first option)` and prints the plan ready to paste into `commit_actions` plus the predicted final grid. Search never touches the live game.
- **Step-wise execution validation** (above) — the live game itself is the test of the model, one action at a time; a plan is never blindly flushed.

Snapshots: on each level clear the harness copies the current model to `snapshots/cleared_level_N.py`. `notes.md` is the long-term memory (mechanics discovered, budgets, failure lessons — e.g. ls20's notes pin per-level move budgets by brute-force against HUD pixel readings, document fog-of-war radius R=20 exactly, and warn about display-rounding artifacts).

### 2.3 Context and budgets

- `max_actions: 3000` per game (run.json); actual usage 208–1,847 actions/game.
- Claude ran with a **1M-token context, auto-compacting** (observed `compact_boundary` at preTokens=1,003,141); GPT with a 258,400-token window and compaction events. Sessions carry `deferred_tools`/ToolSearch — this is stock Claude Code.
- **Cost of one game (ls20, a mid-sized 100% run): 4.36M output tokens, 692M cache-read input tokens, 13.4 h wall.** Across the release: mean **5.9 h/game**, median 3.7 h, max 32.1 h (s5i5), **~288 h total** of frontier agentic compute for 49 timed runs.
- Deaths/resets: RESET is action 0 (257 uses in the corpus), usable inside a batch (e.g. `[[0,-,-],[4,-,-],[4,-,-]]` — "Reset the unrecoverable dead-end, replay the fully backtested L3 route…"). The model must predict `dead` flags too; backtests check flags on every step. New level ⇒ `init_state(entry_grid)` re-run; the first move on a new level is expected to mispredict (unknown per-level constants) — accepted as the price of learning.

### 2.4 Quantitative shape of play (all 50 trajectories)

- 5,460 turns; 5,423 commits; **43,162 actions committed but only 22,184 executed** — ~49% of committed steps are dropped by step-wise misprediction aborts. 3,627 mispredict events (16% of executed actions; s5i5 worst at 328 mispredicts/376 turns).
- Commit sizes: median 1–11 per game, means 3.5–15.0, max 127 (wa30). Action mix: 6=click 7,381; 4=right 3,870; 3=left 3,453; 1=up 3,363; 2=down 2,949; 5: 821; 0=RESET 257; 7: 90.
- Tool profile per game (Claude ls20): 192 `edit_file`, 185 `run_python`, 77 `commit_actions`, 59 `run_backtest`, 36 `run_bfs`, 22 `read_file`. GPT tu93: 139 edits, 70 backtests, 45 commits, 42 bfs, 39 run_python. **Editing/validating the world model dominates; acting is the minority activity.**

### 2.5 Searched-plan vs incremental clears

- Commits whose `reason` explicitly cites BFS: 343/5,423 (6%) covering 4,187/43,162 committed actions (10%); **level-ups delivered by a BFS-citing commit: 81/316 detected level-up transitions (26%)**.
- The rest of the clears come from **hand-derived plans over the (backtested) world model** and incremental probing: plan length of the level-clearing commit across 6 dissected games ranges 1–48 (ls20: [1,28,22,35,13,11]; tu93: [11,3,12,14,29,18,5,21]; s5i5: [1,1,3,1,2,19,3]).
- So: roughly a quarter of clears are literally machine-searched; but effectively **all clears are model-planned** — the signature move is "backtest green → plan (by hand or BFS) → commit batch → per-step verification", not blind incremental play. Incremental single-action commits dominate the *learning* phase (probing mechanics, revealing fog), then a decisive batch closes the level.

---

## 3. Lane (a) — protocol adoption into the duck harness (27B, 32k ctx, per-action LLM + python sandbox)

### What transfers (ranked by cost/benefit for a 27B)

1. **The memory spine: `notes.md` + note-to-self echo (cheapest, aim first).** Three ingredients, all model-agnostic plumbing: (i) a persistent per-game `notes.md` the model maintains and that is re-injected (truncated to ~2-4KB) every turn; (ii) `commit_actions` carrying `reason` + `suggestion`, with the harness echoing both back verbatim next turn ("reconsider, don't just obey"); (iii) honest last-turn accounting ("committed N — executed M; stopped because X; Net: level a→b"). This attacks exactly what the 08-29 independent review found binding: **memory/control — 43% zero-level plays that are not action-starved**. Fits 32k easily.
2. **Batch-with-abort ("optimistic execution, pessimistic verification").** Duck already batches; Schema's twist is that a batch carries an *expectation* and the harness aborts the remainder the moment reality diverges. A 27B cannot write a full `predict()` — but a **degraded contract works without one**: the model commits a batch plus a cheap machine-checkable expectation per step (e.g. "frame changes", "avatar region moves right", "score/level unchanged", or a predicted frame from a harness-provided generic motion model). Abort on violation + report the diff next turn. This converts our current fire-and-forget batches into self-correcting ones and is the second A/B.
3. **Hex-grid observation text.** 64 rows × 64 hex chars ≈ 4.2KB ≈ ~1-2k tokens — cheaper and more regular than most encodings; frontier models handled all perception via `run_python` connected-component scans on it. Duck's masks/HUD handling already exists; low priority.
4. **`run_backtest` as a harness service over recorded transitions** — only pays if some executable model exists (see below); the *service* is trivial to build since we already log frames.
5. **`run_bfs` over a model** — last; worthless without a trustworthy `predict()`.

### What breaks at 27B (be honest)

- **Model-writing is the load-bearing skill.** ls20 took 192 file edits and 59 backtests by Opus 4.8 to keep a 687-line numpy model green; wa30 kept ~20 candidate models. Qwen3.8-27B will produce buggy predictors, and the backtest→edit loop will thrash (each cycle costs turns we don't have). Expect the full contract to *reduce* our score if adopted wholesale.
- **Budget mismatch is fatal for full fidelity: 5.9 h and ~4M output tokens per game** vs our ~13 min/game share of the 12 h Kaggle wall for 55+ games. Schema buys correctness with compute we do not have.
- 1M-token compaction ⇒ our 32k window forces the notes.md/echo mechanism to carry ALL memory — which is fine, because that mechanism is exactly the part that is context-cheap.
- Their runs also benefit from unlimited retries on *public* games during development; the trace method itself (in-run, no game-specific priors beyond notes) is legitimately transferable though.

### Minimal A/B to fly first

**A/B #1: "suggestion echo + notes spine"** — add to duck: persistent notes.md injected each turn (≤2KB), `suggestion` field on the action-batch tool echoed back next turn, and last-turn net-accounting line. Zero new model skills required, pure protocol; measurable on the offline 25 (zero-level-play rate, levels/game) before spending a slot. **A/B #2 (if #1 moves):** batch-abort-on-surprise with heuristic expectations (frame-changed / no-change). Defer executable world models entirely at 27B.

---

## 4. Lane (b) — fine-tune corpus verdict

### What the data actually contains (checked, not assumed)

- **Raw chain-of-thought is ABSENT on both sides.** Claude sessions contain 413 `thinking` blocks per game but every one is `{"thinking": ""}` (stripped in sanitization). GPT `reasoning` items have empty `summary` arrays. **SFT cannot clone the private deliberation.**
- **Visible method text IS present and substantive on the Claude side: 2.37M chars (~600-650k tokens) across 25 games** — real hypothesis-test narration ("522/522 green, bar at 30px exactly as predicted — B=21 continues to hold… the chute is a one-action trip into terrain I can't predict, and I'd rather not be flung blind…"). GPT visible text is thin (158k chars total; typical `agent_message`: "Probed movement to the right.") — GPT trajectories are usable mostly for (state→tool-call/code) pairs, not narration.
- Structured signal: 5,460 turn observations (exact prompts preserved in sessions), 26,692 Claude assistant content blocks, 10,285 Claude tool calls (incl. 100s of world-model edit sequences), 5,423 commit `reason`s (1.44M chars), the full `notes.md` evolution (594KB final states; intermediate states recoverable from prompt echoes), 28.7MB of world-model/candidate/snapshot python, and 22,184 (grid, action, next-grid) environment transitions in events.jsonl (430MB, grids dominate).

### Viability verdict: **viable for METHOD-lite + FORMAT, from the Claude half only; not a deliberation corpus.**

The decisive check ("does the data contain reasoning or only actions") lands in between: no hidden CoT, but the Claude visible text + commit reasons + notes evolution genuinely encode the *procedure* (probe → encode hypothesis in code → backtest → plan → verify per step) at ~650k tokens. That is enough for LoRA-scale SFT to teach a 27B the **protocol behaviors** (maintain notes, state expectations, react to mispredict reports, write candidate hypotheses) — it is NOT enough, and not the right kind of signal, to teach frontier-grade model-coding ability. 25 games of mechanics also invites overfit to public-game priors; hidden games differ.

### Corpus spec (if we build it)

- **Unit = one harness turn**: input = reconstructed user prompt (state header + last-turn accounting + intent/suggestion echo + notes.md excerpt + hex grid) ≈ 2.5-4k tokens; target = the assistant's full turn (visible text + tool-call sequence + final commit_actions) — median turn fits well under 32k; clip the few monster turns at tool-call boundaries. Claude-only for narration-bearing examples: ~2,556 turns; add GPT turns (~2,904) as action/code-only examples if wanted.
- Yield estimate: **~5,400 turn-level examples (~2,500 with rich narration), ~40-80M training tokens** if tool outputs are included, ~10-20M if outputs are elided/truncated. Enough for a LoRA epoch or two, not for shifting base capability.
- Precondition that dominates everything: **the duck harness must first speak the same protocol** (notes injection, suggestion echo, batch+abort), or the fine-tune teaches behaviors the runtime can't express. Sequence: Lane (a) A/B #1 first; only build the corpus if the protocol lands. And per standing law: no fine-tune has ever been served AND scored — the serving gate comes before any training bet.

---

## 5. License / competition-rules status

- **No license is declared anywhere**: HF metadata is empty ("YAML Metadata Warning: empty or missing yaml metadata") on both `schema-harness/arc-agi-3-schema-traces` and the `JBrightmanAI` mirror; no LICENSE file in the repo; README silent. Default = all rights reserved, despite the authors' evident intent to share ("sanitized session data", public blog inviting exploration).
- **Publicly available at no cost: yes** (open HF repo, ~2.5k downloads/mo) — that satisfies the usual Kaggle external-data availability test. I could not render the JS-only Kaggle rules page to quote this competition's exact external-data clause; verify the clause text once from a browser before relying on it.
- Frictions to flag before any training use in a submission: (1) prize-eligible solutions must open-source under **CC0/MIT-0** (our memory of the rules) — shipping a model fine-tuned on an *unlicensed* dataset is a gray area for "reproducible, open" claims; (2) the traces are frontier-model outputs (OpenAI/Anthropic terms restrict training competing models on outputs — a risk the *dataset authors* took, but which transfers murkily to downstream trainers). **Cheap fix: email the authors (Zanette lab, Berkeley/CMU) asking them to add a license tag (CC-BY-4.0 or similar)** — they respond publicly and clearly want reuse. Reading/adopting protocol ideas (Lane a) carries no license risk at all.

---

## 6. Key file paths

- Dataset: `/private/tmp/claude-501/-Users-ahmed-Documents-ArcAGI3/336208eb-4625-4ff9-b10d-1664cc846e5d/scratchpad/schematraces/ds` (scratchpad — will not survive cleanup; re-fetch with `snapshot_download(repo_id='schema-harness/arc-agi-3-schema-traces', repo_type='dataset')`, 739MB)
- Scorer: `ds/score_trajectories.py` (stdlib-only; run `python3 score_trajectories.py --compact` from `ds/`)
- Richest dissection targets: `ds/claude_fable_opus/claude-opus-4-8_max_ls20_100.0/` (notes.md + 687-line world model + full session), `ds/gpt_5_6_sol/gpt_5_6_sol_xhigh_wa30/` (20 candidate models), `ds/claude_fable_opus/claude-opus-4-8_max_s5i5_89.87/` (32 h grind, misprediction-heavy)
- Today's full leaderboard CSV: `/tmp/lb/arc-prize-2026-arc-agi-3-publicleaderboard-2026-08-31T20:16:20.csv` (2,671 teams)
