# HANDOFF — Crack ls20's invisible-state wall (perception sub-project)

Date: 2026-06-21. Branch: `winning/mechanic-model-search` (committed, **NOT pushed**).
Read this whole file first. It is self-contained on purpose.

---

## 0. One-paragraph summary

ls20 L1 is unsolved by every agent because the avatar's **rotation** gates the goal but is **invisible
in the frame** (the avatar renders identically across rotations). We *proved* (engine source + an
active state-injection probe) that this is a **representation-completeness** problem, and we *built and
unit-proved* the fix mechanism: a `HistoryAugmentedExplorer` that folds a `mod-4` "visit counter" for
the rotation tile into the graph node key, so the 4 rotations become 4 distinct, separately-explorable
states. **The only remaining blocker is perception: reliably counting when the avatar steps on the rot
tile**, from pixels. Two detector designs failed for the same reason (avatar segmentation is hard on
these frames). Your job: build a robust visit detector (recommended: use the dormant `ObjectTracker`),
validate it to ~0 mismatches against the ground-truth `cklxociuu` oracle, then confirm ls20 cracks with
no dev regression.

---

## 1. CRITICAL: do not trust stale memory; verify the baseline first

The auto-loaded memory files (`~/.claude/.../memory/arcagi3-*.md`) are **partly stale** — they describe
an old "SalienceExplorer v6 / 0.33" era. The **real** current state:
- **Banked best agent = `TransferExplorer` (v13), public score 0.33.** (`submission/my_agent.py`.)
- `TransferExplorer` **inherits `SalienceExplorer._key` verbatim** — so anything built on `SalienceExplorer`'s
  state-key (like the work below) uses the banked agent's exact node identity. The 0.33 floor is real.
- There is a large explorer fleet (`transfer_*`, `discovery_explorer`, `prior_explorer`, …) and ~331
  tests. **`tests/test_bakeoff_metrics.py::test_human_baseline_level1_is_optimal` HANGS forever** — never
  run a bare full `pytest`; run targeted test files.
- Run interpreter: `.venv/bin/python` (3.12). Always prefix commands with `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src`.

The most reliable, current record of THIS work is the bottom of
`~/.claude/projects/-Users-ahmed-Documents-ArcAGI3/memory/arcagi3-research-findings.md` (Phase P + Phase Q
entries) and the docs listed in §3.

---

## 2. The goal and the working agreements

**Goal:** crack ls20 L1 (and ideally L2+) efficiently — today only brute-blind salience reaches ls20 L1
at ~7900 actions (≈ 0 efficiency under the squared metric). ls20 is a HOLDOUT game, so an efficient
crack contributes to the score, and it would be the **first break of an invisible-state wall**.

**Working agreements (the user's standing rules — follow them):**
- Use the **superpowers** flow for non-trivial work: `brainstorming` → `writing-plans` →
  `subagent-driven-development` (or inline). Design before code.
- **Validate-or-kill** with evidence; **always have a ground-truth/control** (here: the `cklxociuu` oracle).
- **Never regress the banked 0.33.** The agent under construction is a `SalienceExplorer` subclass behind
  an `augment=False` default that is **byte-identical** to banked — keep that firewall (there's a test).
- **Approval gate:** implement/test/commit on the branch, then **STOP** before any `git push`, Kaggle
  submission, or PR merge. Do not push or submit without explicit user say-so.
- **Honesty:** report failures/ties plainly; don't claim a run you didn't do. Don't thrash — if a
  perception approach isn't converging after a few measured iterations, checkpoint (that's what happened).

---

## 3. What to read (in order)

1. `docs/superpowers/specs/2026-06-21-history-augmented-state-design.md` — the spec, **including the
   "Interim result", "Iteration update", and "Redesign v2 result" sections at the bottom** (the full
   diagnosis of why it's stuck).
2. `docs/superpowers/specs/2026-06-21-invisible-state-prevalence-map-design.md` — Phase P (why this is an
   invisible-state problem; the active-probe proof).
3. `~/.claude/.../memory/arcagi3-ls20-mechanic-ground-truth.md` — the reverse-engineered ls20 mechanic.
4. Code: `src/arcagi3/history_augmented_explorer.py` (current detector), `scripts/ls20_crack.py` (the
   oracle harness), `scripts/invisible_state_probe.py` (the Phase P active probe — a working example of
   driving the offline engine + reading `env._game`).
5. `src/arcagi3/tracking.py` — the **dormant `ObjectTracker`** (your recommended tool; see §6).

---

## 4. ls20 ground truth (verified from engine source `environment_files/ls20/9607627b/ls20.py`)

- **Rotation-gated**, NOT paint/moving-goal (the repo's `experiment-overview.html` is WRONG/superseded).
- Win predicate `bejndxqqzf(i)` requires `cklxociuu == ehwheiwsk[i]` (plus shape/color, already matched at
  L1 start). `cklxociuu` = rotation index into `dhksvilbb=[0,90,180,270]`. **L1 GoalRotation=0, StartRotation=270 (idx 3).**
- The **rotation tile** (tag `rhsxkxzdjz`) is at engine `(x=19, y=30)` = **grid `(row=32, col=19)`**
  (`grid_row = engine_y + 2`, `grid_col = engine_x`). Stepping onto it: `cklxociuu = (cklxociuu + 1) % 4`.
  So from idx 3, **1 step on the tile → idx 0 = goal rotation**.
- The tile's **visible arrow glyph** renders at grid rows 31–33, cols 20–22 (colors 0 and 1), i.e.
  **offset by ~1 cell from the trigger cell (32,19)**, and **is occluded** when the avatar steps on the
  trigger (`n_small` objects drop 12→9 at that step — confirmed).
- **Avatar** = a 5×5 block: head color **12** (top 2 rows) + body color **9** (bottom 3 rows). Moves on a
  **5-cell pitch** (`act1=up(-5,0), act2=down(+5,0), act3=left(0,-5), act4=right(0,+5)` in grid terms).
  **Color 9 is ALSO used by static maze objects** (this is the core segmentation trap).
- Known winning solution (13 actions, hits the rot tile once): `[3,3,3,1,1,1,1,4,4,4,1,1,1]`.
- **The engine resets `cklxociuu` to StartRotation on every life-loss / level restart** (StepCounter
  42/life, 3 lives). Your counter MUST reset then too (already handled — see §5).
- Engine introspection (offline only): `env._game.cklxociuu`, `env._game.dhksvilbb`,
  `env._game.current_level.get_data("StartRotation")`, `env._game.gudziatsk` (avatar sprite, `.x/.y`),
  `env._game.current_level.get_sprites_by_tag("rhsxkxzdjz")` (rot tile sprite).

---

## 5. What is already built (and works) — do NOT rebuild

- `src/arcagi3/history_augmented_explorer.py` = `HistoryAugmentedExplorer(SalienceExplorer)`:
  - `_key` appends a sorted `(sig, count % counter_mod)` tuple → the mechanism. **Proven by unit tests**
    (`tests/test_history_augmented_explorer.py`, 6 tests incl. the `augment=False` byte-identical FIREWALL).
  - `decide` resets history on terminal/not-played/level-up (mirrors the engine's per-life rotation
    reset — this was a real bug, now fixed) and otherwise calls `_update_history`.
  - `_update_history` currently = the **disappearance detector** (count a confirmed-static cluster that
    disappears). It is the cleaner foundation but does NOT yet work on ls20 (see §7).
  - `histaug` policy is wired into `scripts/eval_efficiency.py` (`make_policy`) for the dev A/B.
- `scripts/ls20_crack.py` = **the oracle harness**. Runs banked vs augmented on the OFFLINE ls20 engine,
  and per step compares `sum(pol._counts.values()) % 4` to `(env._game.cklxociuu - start_idx) % 4`,
  printing `rot_oracle: <mismatch>/<checks>`. **This is your optimization target: drive mismatches → ~0.**

Run it: `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/ls20_crack.py 8000`

---

## 6. THE TASK: build a robust visit detector (recommended approach)

The mechanism is done. You must make `_update_history` produce a counter that **increments exactly once
each time the avatar steps onto the rot-tile trigger cell**, so `sum(counts) % 4` tracks `cklxociuu`.

**Recommended approach — use `ObjectTracker` (`src/arcagi3/tracking.py`, currently dormant/unused):**
1. **Segment the avatar explicitly** and EXCLUDE it: the avatar is the single **rigidly-translating
   multi-cell block**. Identify its cells via a translation mask (cells `c` with `grid[c]!=bg`,
   `prev[c-Δ]==grid[c]`, `prev[c]!=grid[c]` for the avatar's per-step delta Δ; learn Δ from the head
   color 12 which translates cleanly). `ObjectTracker` may give you stable avatar identity across frames.
2. **Track static glyphs with persistence through occlusion** (`ObjectTracker` is built for exactly this —
   typed events, object permanence). The rot glyph is a small static cluster at grid (31–33, 20–22).
3. **Count a "visit"** when the avatar's position coincides with the rot-tile trigger (grid (32,19)) — or,
   equivalently, when the tracked rot-glyph transitions visible→occluded **by the avatar specifically**
   (not by a generic disappearance). Edge-triggered (once per step-on).
4. **Validate against the oracle FIRST**, before judging the crack: drive the known solution and/or run
   `ls20_crack.py`; require `rot_mismatch → ~0`. Only once the counter tracks `cklxociuu` does it make
   sense to expect the explorer to crack the level.

**Keep:** the `mod-4` counter, the per-life `_reset_history`, the `augment=False` firewall.

---

## 7. What was already TRIED and FAILED — do not repeat

- **Avatar-overlap counting** (count when avatar cells overlap a remembered glyph cell): failed because
  `infer_translation`/`infer_all_translations` learns only the head (color 12) — the body color 9 is
  shared with static maze objects, so the full color-9 mask never translates rigidly. Head-only footprint
  → unreliable overlap. (Oracle ~66% mismatch.)
- **Translation-mask avatar segmentation** (take all cells that arrived by the avatar's delta): noisier,
  **regressed** ls20 to L0. Reverted. (May be salvageable with `ObjectTracker` + cleanup, but it wasn't
  robust as a quick inline fix.)
- **Disappearance/occlusion-event detection** (count a confirmed-static glyph cluster that vanishes):
  failed because the avatar's OWN parts (head 12 + body 9, each ≤ `OBJ_MAX_SIZE=16`) cluster as small
  objects and get counted when the avatar PAUSES ≥2 frames against walls, and 8-adjacency clustering
  merges the avatar with adjacent glyphs. **The avatar must be explicitly segmented and excluded** — the
  problem is intrinsic to detectors that don't do that.
- **Cumulative counter without reset:** real bug, already FIXED (engine resets rotation per life/level).

The recurring lesson: **you cannot avoid segmenting the avatar.** Its colors are shared with the maze
and its parts look like small objects. Solve avatar segmentation robustly (ObjectTracker) and the rest
follows.

---

## 8. Definition of done (pre-registered)

1. **Oracle:** on `scripts/ls20_crack.py`, `rot_oracle` mismatches ≈ 0 (the counter tracks `cklxociuu`
   within each life).
2. **Crack:** augmented `best_level` ≥ 1 on ls20 at **far fewer** actions than banked's ~7900 (target:
   low hundreds), ideally L2+ (other GoalRotations).
3. **No dev regression:** `histaug` drops no TUNE/HOLDOUT levels vs `salience` on
   `scripts/eval_efficiency.py` at budgets 6000 and 30000 (the augmentation must not explode other games;
   if it does, tighten the object filter / avatar exclusion).
4. **Firewall intact:** `augment=False` byte-identical test still green; banked 0.33 untouched.
Then STOP and report (promotion to the submission default needs a real, user-gated Kaggle submission —
dev ≠ Kaggle).

---

## 9. Concrete first moves for the fresh session

1. Verify baseline: `cd /Users/ahmed/Documents/ArcAGI3 && git --no-pager log --oneline -12` (you should see
   the `phase-q` commits) and `ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_history_augmented_explorer.py -q` (6 pass).
2. Run `scripts/ls20_crack.py 8000` and read the `rot_oracle` + `visit counts` (you'll see it counting
   avatar blocks — confirm the §7 diagnosis for yourself).
3. Read `src/arcagi3/tracking.py` (`ObjectTracker`) and `scripts/invisible_state_probe.py` (working
   offline-engine + `env._game` example).
4. Use `brainstorming` to scope the ObjectTracker-based detector (avatar segmentation + occlusion-aware
   rot-tile visit), then `writing-plans`, then build — TDD, validating to the oracle first.
5. A useful scratch diagnostic pattern (drive the known solution with engine introspection, compare your
   perception to `env._game.cklxociuu` / `gudziatsk` per step) cracked the diagnosis last time — recreate
   it early.

Good luck. The mechanism is proven and the wall is provably breakable; this is now a focused perception
problem with a precise oracle to optimize against.
