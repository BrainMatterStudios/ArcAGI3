# THE MECHANICS COMPENDIUM — all 25 dev games, from primary source
2026-07-25. Sources: `environment_files/<game>/<hash>/<game>.py` + `metadata.json` (line refs below), arcengine API (`.venv/.../arcengine/base_game.py`: games call `self.next_level()` / `self.win()` / `self.lose()`), K3 results from `scratchpad/rl_gate/sft_data/stats.json`, June decodes from memory `arcagi3-all-dev-games-cracked.md` (marked [June] where relied on).

Capability letters: **P** perception, **H** hypothesis/rule-discovery, **S** search/lookahead, **M** hidden state/memory, **X** execution precision. Upper-case = primary blocker, lower-case = present. Confidence: ✓ = win predicate read directly from source; ~ = partially decoded (marked).

## 1. Per-game table

| game | mechanic family | win condition (source) | caps | baselines/level | K3 (best win actions) |
|---|---|---|---|---|---|
| ar25 ✓ | piece-fit / fill-silhouette (A1-A7, A7=undo, A5=cycle selected) | all target-tag cells rendered non-empty — `vplrhaovhr` ar25.py:1688: every `0001sruqbuvukh` sprite pos has pixel ≥0 in composed frame | P h S x | 32,50,75,37,89,159,233,73 | 5/8 — stalls L6 |
| bp35 ✓ | digger w/ INVERTED gravity (dy=-1, avatar=color-11) [June] | land on gem → `self.win()` bp35.py:4257; spikes/fall → lose | H! P s | 21,48,44,38,33,87,86,131,163 | 1/9 (L1 in 49 vs 21) — stalls L2 |
| cd82 ✓ | template-copy under action budget | left 10×10 == right 10×10 excluding both diagonals (mask), cd82.py:748-753; lose when `_action_count >= budget` :632 | X P (budget!) | 55,8,41,21,23,23 | 1/6 — L2 baseline=8, K3 spent 121 |
| cn04 ✓ | click-paint cells on (rotating) objects | no cell of color 8 or 13 remains unpainted on any visible sprite (rotation-remapped) — `sjwqloivve` cn04.py:1023-1049 | X p (budget MaxSteps) | 29,54,85,300,208,113 | 1/6 — stalls L2 |
| dc22 ✓ | maze navigate + keys/doors (`buezna` tags), step energy | avatar pos == goal pos — `smxyfelexa` dc22.py:10891; lose on steps exhausted | S! M (endurance) | 59,102,67,98,324,578 | 0/6 — L1: 300 acts, no win |
| ft09 ✓ | neighbor-constraint coloring (corner-flag = must-equal/must-differ) | every clue sprite's corner-pixel flags satisfied vs 4/8 neighbors' center colors — `cgj` ft09.py:2436 | H p x | 43,12,23,28,65,37 | 4/6 (twice) — stalls L5 |
| g50t ✓ | clone-recorder sokoban (A5 banks ghost replaying your path) [June] | inner-sim win flag (reach goal; ghosts hold buttons/gates) g50t.py:2806,2834 | H! S m | 78,175,179,230,96,54,67 | 1/7 — stalls L2 (clone needed) |
| ka59 ✓ | push blocks AND launch-as-projectile onto targets; chasing enemies [June] | all tag-`0010` on tag-`0022` targets AND all tag-`0027` on `0001` — `dbmlcqbquh` ka59.py:41262; enemy contact → lose :41219 | H! S x | 28,109,51,51,33,132,326 | 1/7 — stalls L2 |
| lf52 ~ | tile-sim w/ red/blue image_groups, per-level step caps (64×1/5/10) lf52.py:5771-5780 — inner engine, PARTIALLY DECODED | inner `win()` on specific collision `tdcblgbfxw` lf52.py:5644 | ? S | 32,81,60,71,205,148,244,109,164,225 | 1/10 — stalls L2 |
| lp85 ✓ | click-sokoban, two piece types onto two goal types | all tag-`bghvgbtwcb` on `goal` AND all `fdgmtkfrxl` on `goal-o` — `khartslnwa` lp85.py:21442 | S x | 17,38,31,16,41,60,26,159 | 2/8 — stalls L3 |
| ls20 ✓ | rotation-gated collect-all (rotation shown ONLY on HUD indicator) [June] | all goal cells collected with matching gate `bejndxqqzf` — `pbznecvnfr` ls20.py:2042; countdown lose | P! H m | 22,123,73,84,96,192,186 | 1/7 — stalls L2 |
| m0r0 ✓ | modal select-then-move: click piece (`mosdlc`), arrows move it into matching socket (`unobxw-X`↔`gayktr-X`) m0r0.py:730-795 | all 4 `pikgci-*` pieces matched (count==0) m0r0.py:863-865; 150-action cap :726 | H! X m | 30,111,203,26,500,237 | 0/6 — ZERO on L1 (baseline 30) |
| r11l ✓ | timing windows / whack-a-mole: entities open (`flgzyjcqcspeg`) on internal per-action clocks, act while open; 5 misses→strike, 5 strikes→lose r11l.py:1667-1750 | all windows satisfied (none open-pending) r11l.py:1699-1706 | M! h (autonomous dynamics) | 22,33,51,26,52,49 | 1/6 — stalls L2 |
| re86 ✓ | paint canvas to match target sprites (color 4 = wildcard) | composed canvas == every target-tag sprite where target≠-1,≠4 — `jeiavrvavi` re86.py:1894; step-counter lose | P X | 26,42,86,108,189,139,424,241 | 2/8 — stalls L3 |
| s5i5 ✓ | indirect global controls: color-button click transforms ALL pieces of that color; slider clicks resize piece groups s5i5.py:2181-2242 | every tag-`0087` target pos has a tag-`0064` piece exactly on it — `neurwiqfry` s5i5.py:2080 | H! S | 20,89,106,54,162,38,86,83 | 2/8 — stalls L3 |
| sb26 ✓ | palette→slot pattern-match + A5 submit, limited attempts [June sb26-cracked] | submitted pattern correct (anim then `next_level` sb26.py:805); attempts 0 → lose :837 | P x (attempts) | 18,28,18,19,31,23,58,18 | **8/8 FULL CLEAR** |
| sc25 ✓ | 3×3 spell-combo: click spell icon→demo pattern; toggle 3×3 grid to match; cast; move budget `eyxbonasvgm` sc25.py:1890-1898, 2603-2666 | level goal via cast effects (win path :1856); over-budget → lose | H M (budget) | 36,6,32,83,143,50 | 2/6 — stalls L3 |
| sk48 ✓ | snake/chain: forward grows (clone segment inserted), backward retracts (pop), turn pads `irkeobngyh` rotate — sk48.py:768-810 | targets reached → flash-anim → `next_level` :716; steps 0 → lose :765 | H! S x | 61,177,101,103,230,181,125,92 | 0/8 — ZERO on L1 |
| sp80 ✓ | two-phase dam-building: click-select block, arrows move ("change" phase); A5 → water-spill gravity sim; ≥4 spills or health 0 → lose sp80.py:674-747,861-866 | spill contained / goal met → `next_level` :747 | H S m (budget) | 39,58,25,148,96,152 | 1/6 — stalls L2 |
| su15 ✓ | shape placement into zones (A6 clicks + A7), lives sprites (`zmlxwcvwb`), two-phase state machine su15.py:1048-1108 | placed set matches level spec — `cbdhpcilgb` su15.py:2039 | X p m | 22,42,26,115,36,31,8,40,41 | 3/9 — stalls L4 |
| tn36 ✓ | program-the-robot: click-place instruction tiles (level data `Programs`/`Positions`/`Rotations`), then run inner sim tn36.py:2612-2637 | inner sim win flag `vklyonlcrw` :2637 | H! S! m | 32,72,26,40,30,55,62 | 0/7 — ZERO |
| tr87 ✓ | symbol rewrite-rules: A3/A4 cycle glyph variants (name-digit ±1 mod N tr87.py:1024-1026); hidden rules `cifzvbcuwqe`, per-level `alter_rules`; limited attempts `upmkivwyrxz`→lose :1020 | full rule-application sequence completes — `bsqsshqpox` :1044, win :993 | H!! (attempts) | 54,58,40,45,71,146 | 0/6 — ZERO |
| tu93 ✓ | move ALL units simultaneously (coupled movement) onto exits | every tag-`0017` unit on some tag-`0015` exit — tu93.py:1261-1264; unit lost or steps 0 → lose | S! p | 19,16,34,42,123,80,14,23,111 | 2/9 — stalls L3 |
| vc33 ✓ | gravity-drop items into matching-color containers by clicking (removing supports; level data `Gravity`) | per-item: matching-color receptacle in correct column/region — `ielczunthe` vc33.py:1937 | S h | 7,18,44,61,131,34,152 | 3/7 — stalls L4 |
| wa30 ✓ | herd/route all items to target cell-set (A1-A5) | all tag-`geezpjgiyd` at positions ∈ `wyzquhjerd` and not excluded — `ymzfopzgbq` wa30.py:1194 | S m | 71,119,183,98,368,68,79,442,415 | 1/9 — stalls L2 |

Notes: 12 games carry an explicit `StepCounter`/budget lose; RESET restarts level free (engine `handle_reset`). Levels within a game share the win predicate — level data varies layout + parameters (ls20: `Fog`/`GoalRotation`/`StepsDecrement`; cn04: `GreyMasking`; vc33: `Gravity`; tr87: `alter_rules`/`double_translation`; tn36: `Programs`; su15: `steps`; sp80: `steps`).

## 2. Why K3 fails where it fails (blocker letters)

**Zeros (5):**
- **m0r0 — H (modal affordance).** Win needs click-to-SELECT then arrow-to-MOVE (two-stage control, m0r0.py:730-795). Nothing on screen teaches "selection"; without it every arrow press is a no-op. Compound/modal control schemes are engine-idiomatic → **likely shared with hidden games.**
- **sk48 — H (non-standard avatar).** Forward = grow chain, backward = retract, turns only on pads (sk48.py:768). The "avatar" changes shape every move; standard "move the agent" prior actively misleads. Baseline is only 61 — one insight away.
- **tn36 — H+S (program composition).** Must place instruction tiles then run; reward only after a full correct program. No incremental feedback → duck-class agents can't hill-climb. Structurally hard for any duck.
- **tr87 — H (hidden rewrite rules) + punishing attempt budget.** Rules are latent (`alter_rules` mutates them per level); wrong cycles burn `upmkivwyrxz` attempts → lose before hypothesis converges. Structurally hard.
- **dc22 — S/endurance, not concept.** Win predicate is trivial (reach goal). L1 baseline 59 but branching maze + keys/doors; K3 burned 300 actions without finishing. Blocker = exploration length vs token/time budget. **This blocker (long-horizon mazes) certainly recurs in hidden games.**

**Early stalls:** bp35 L2 / ls20 L2 / ka59 L2 / g50t L2 are each ONE mechanic insight away (inverted gravity; HUD-rotation link; block-launch; clone-recorder) — all four were our own June "impossibility" reversals, i.e. exactly the class of insight a better hypothesis engine (or a prompt prior) unlocks. cd82 L2 is an efficiency wall (baseline 8; must act near-perfectly). tu93 L3+ is coupled-unit search. dc22/wa30/m0r0 deep levels (baselines 324-578) are budget walls for ANY LLM-turn agent.

**Blocker distribution over the 5 zeros: H×4, S×1.** The dominant missing capability is rule/affordance discovery, not perception. This matches Law 2's world: scaffolding can't fix H; better priors/brains can.

## 3. The depth map

Scoring: level ℓ of an L-level game is worth 100·ℓ/Σ(1..L) game-points at baseline efficiency; dev-mean point = game-points/25.

**If the tuned duck inherited ALL of K3's dev wins (best win per level, actual K3 action counts): dev mean = 12.36/100** (at perfect s=100 on the same levels: 12.24 — K3's win efficiency is already ≈baseline). Versus our live 1.26 (hidden) — not comparable directly, see §5.

**Frontier (next unclaimed level) per game, ranked by value-per-level:**
ft09 L5 (+0.95 dev-mean pts/level, only 2 left), ar25 L6 (+0.67, 3 left), sc25 L3 / vc33 L4 (+0.57), cd82 L2 / cn04 L2 / r11l L2 / sp80 L2 (+0.38), su15 L4 (+0.36), lp85 L3 / re86 L3 / s5i5 L3 (+0.33), g50t L2 / ka59 L2 / ls20 L2 (+0.29), tu93 L3 (+0.27).

**Top winnable-looking targets** (frontier value × judged tractability):
1. **ft09 L5-L6** (+2.10 tail) — rule already demonstrated by K3 at L1-4; just bigger instances. Most valuable tractable tail on the board.
2. **ar25 L6-L8** (+2.33 tail) — K3 already 5/8; scaling, no new mechanic.
3. **g50t L2+** (+3.86 tail) — single insight (A5 = clone recorder) unlocks the game.
4. **ls20 L2+** (+3.86 tail) — single insight (HUD rotation indicator).
5. **ka59 L2+** (+3.86 tail) — single insight (bump = launch projectile).
6. **bp35 L2+** (+3.91 tail) — single insight (gravity inverted; avatar = color 11).
7. **sk48 L1-L8** (+4.00 full game) — single insight (chain grow/retract); K3 zero now.
8. **m0r0 L1-L4** (~+1.7 of its 4.00) — single insight (click-select mode); L5 (baseline 500) likely budget-walled.
9. **vc33 L4+** (+3.14 tail) — K3 already 3/7, mechanic understood.
10. **sc25 L3+** (+3.43 tail) — spell-book is learnable in-context.
Structurally hard for duck-class regardless of insight: tn36, tr87, dc22 L4+, wa30 L5/L8 (368/442 baselines), m0r0 L5.

## 4. Mechanic priors (recur across ≥5 games → plausible in hidden set)

- **ALL-quantified coincidence win** ("every X on some Y"): lp85, ka59, s5i5, tu93, wa30, m0r0, ls20, dc22(singleton) — ≥8/25. The single strongest engine prior: **wins require completing ALL subgoals, not one.**
- **Pattern match/copy/paint-to-target**: cd82, sb26, re86, cn04, sc25, ft09 — 6/25.
- **Budgeted actions/attempts/strikes** (efficiency IS survival): cd82, cn04, m0r0, sb26, sc25, sp80, r11l, tr87 + StepCounter in 12 — >half.
- **Click-to-act (A6)**: ~19/25 (known June finding, re-confirmed: 7 click-only).
- **Modal / multi-phase controls** (select-then-move, build-then-run, record-then-replay): m0r0, sp80, sc25, g50t, tn36, su15 — 6/25, and the #1 K3-killer.
- **Physics/gravity sim**: bp35, sp80, vc33, lf52(~) — 4 (weak prior).
- **Autonomous per-action dynamics** (world ticks when you act): ka59 enemies, r11l windows, g50t ghosts, sp80 water — 4 (weak).

### Prompt-injectable "mechanic field guide" (~340 tokens — needs offline A/B per Law 2 before any scored use)

> ARC-AGI-3 games are 64×64 sprite puzzles. Common families and tells:
> **REACH/NAVIGATE** — an avatar moves with A1-A4; walls block. Win: reach a marked cell. Watch for doors that need keys/buttons.
> **COVER-ALL** — several movable pieces + equally many target cells. Win: EVERY piece on a target (one is never enough). Movement may be coupled (all pieces move together) or via push.
> **MATCH/COPY** — two panels: an editable one and a reference. Win: make them equal (sometimes ignoring marked cells/diagonals). Click cycles colors; a submit button may finalize.
> **CONSTRAINT-COLOR** — cells with glyph corners/edges encode "neighbor must equal/differ". Satisfy all clues.
> **MODAL CONTROLS** — clicking an object can SELECT it (highlight); arrows then move the selection. A5/A7 often = submit, undo, phase-switch, or RECORD/REPLAY a clone. If arrows seem dead, try click-then-arrow.
> **BUILD-THEN-RUN** — place tiles/program steps, then a run action executes them; no feedback until run.
> **TIMING** — objects open/close on their own every few actions; act while open; misses cost strikes.
> **PHYSICS** — things fall (gravity may be INVERTED — check which way loose objects drift); water/projectiles propagate.
> Universal: budgets are tight (a step counter usually shows remaining actions; running out = lose, RESET restarts the level free). Wins are almost always ALL-quantified. The avatar may be an unusual color/shape, may grow like a snake, or may not exist (click-only). Hidden state is often shown on a small HUD indicator — read the HUD, not just the arena.

## 5. Honesty — what leans on dev specifics

- Per-game mechanics/win predicates are FACTS of the dev set only. Hidden games are different games; only the ENGINE-level priors (§4: all-quantified wins, A6 dominance, step budgets, modal controls, free level-reset, HUD indicators) plausibly transfer — same arcengine, same designers.
- The 12.36 K3-inherit dev mean is NOT a hidden-set prediction. Field #1 is 1.86 on hidden; K3's 26-ish raw dev mean implies hidden games are far harder and/or longer than dev, or that dev familiarity (25 public games, human-documented) leaks into any dev number. Pure-BC hidden ceiling: the student ≤ teacher, and the teacher's dev wins skew to P/X-family games (sb26, ft09, ar25, vc33) — the H-family blockers (§2) are untouched by imitation of win turns. Expect BC to transfer turn-discipline and the §4 priors, not insights; a 2-4× improvement over 1.26 is the optimistic band, unmeasurable except over many scored draws (sd≈0.3).
- The "one insight away" list (§3) uses our June solve knowledge — a hidden game's analogous insight must be DISCOVERED, not recalled. What transfers is the meta-lesson: when stuck, hypothesize a non-obvious affordance (launch/clone/invert/select-mode) and test it cheaply — RESET is free.
- lf52 marked partially decoded; its inner engine (5.9K lines) was not fully read. sc25/su15 win-spec details simplified.
- K3 numbers are single-to-double-run samples per game (retries exist for some); stall levels have their own variance.
