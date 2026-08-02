# Determinism sweep (E2) + HUD-mask no-op validation — 2026-08-02

Environment: arc_agi 0.9.9 / arcengine 0.9.3, OFFLINE mode, all 25 public games
from `/Users/ahmed/Documents/ArcAGI3/environment_files`. Scripts in this
directory: `hud_detector.py`, `test_a_determinism.py`, `test_b_noop_mask.py`.
Raw outputs: `results_a.json`, `results_b.json`, `frames_<gid>_run{1,2}.npz`.

## TEST A — Determinism sweep

Protocol: per game, one fixed 100-action script (seeded per-game RNG over the
game's initially-available actions; ~35% ACTION6 clicks at fixed coords, rest
ACTION1-5/7), played twice from completely fresh instances (fresh `Arcade`
client, `make(seed=0)`). Deterministic auto-RESET policy when a terminal state
is hit mid-script (identical in both runs). Per-step comparison of the settled
(last-layer) 64x64 grid, raw and with wavefront/HUD strips masked (mask =
union of strips detected on both runs' own frames).

| game | steps | raw div | masked div | strips found | mask cells | note |
|------|-------|---------|------------|--------------|------------|------|
| ar25 | 101 | 0 | 0 | 1 | 60 | |
| bp35 | 101 | 0 | 0 | 0 | 0 | |
| cd82 | 101 | 0 | 0 | 1 | 64 | hit GAME_OVER, auto-RESET, still 0 div |
| cn04 | 101 | 0 | 0 | 0 | 0 | |
| dc22 | 101 | 0 | 0 | 1 | 60 | |
| ft09 | 101 | 0 | 0 | 0 | 0 | |
| g50t | 101 | 0 | 0 | 1 | 50 | |
| ka59 | 101 | 0 | 0 | 1 | 64 | hit GAME_OVER, auto-RESET, still 0 div |
| lf52 | 101 | 0 | 0 | 0 | 0 | |
| lp85 | 101 | 0 | 0 | 0 | 0 | |
| ls20 | 101 | 0 | 0 | 0 | 0 | |
| m0r0 | 101 | 0 | 0 | 2 | 86 | |
| r11l | 101 | 0 | 0 | 0 | 0 | |
| re86 | 101 | 0 | 0 | 1 | 64 | hit GAME_OVER, auto-RESET, still 0 div |
| s5i5 | 101 | 0 | 0 | 0 | 0 | |
| sb26 | 101 | 0 | 0 | 1 | 37 | |
| sc25 | 101 | 0 | 0 | 0 | 0 | |
| sk48 | 101 | 0 | 0 | 1 | 15 | |
| sp80 | 101 | 0 | 0 | 0 | 0 | |
| su15 | 101 | 0 | 0 | 1 | 58 | |
| tn36 | 101 | 0 | 0 | 0 | 0 | |
| tr87 | 101 | 0 | 0 | 1 | 50 | |
| tu93 | 101 | 0 | 0 | 0 | 0 | |
| vc33 | 101 | 0 | 0 | 0 | 0 | |
| wa30 | 101 | 0 | 0 | 1 | 32 | |

Also checked per game: action-execution logs identical (no divergent
exception paths) and per-step state sequences identical — true for all 25.

**Verdict: 25/25 games are replay-safe on this evidence.** Zero divergent
steps in 101 compared frames per game, raw — the masked column never even
mattered. Three games (cd82, ka59, re86) died and auto-RESET mid-script and
still replayed bit-identically, so the death+reset path is deterministic too.
Fewer HUD strips were detected here than in the dedicated probe (Test B)
because a mid-episode RESET refills the bar, breaking the
"changes-exactly-once" wavefront statistic — a known detector limitation, not
a game difference.

Caveats, honestly stated: (a) this is a 100-action prefix with `seed=0` on one
machine/process — it does not rule out divergence deeper into episodes,
under a different seed argument, or on the gateway's server build; (b) the
scripted prefix is shallow play (no level transitions beyond what random play
reaches in 100 actions).

## TEST B — HUD-mask no-op validation

Protocol: per game, a clean probe episode (seeded random policy, up to 60
steps, stopped at first terminal so the bar never refills) builds the
wavefront mask. Then each candidate action type (ACTION1-5/7 as available,
plus an ACTION6 click at corner (0,63)) is applied 8 times from a fresh
instance; each step classified FROZEN (no pixel change), HUD_ONLY (raw diff
non-empty, masked diff empty), CONTENT (masked diff non-empty).

Verdicts:
- **HOLDS** — a no-op action exists whose raw diff is non-empty on (nearly)
  every step while the masked diff is empty (HUD_ONLY >= 4/8, CONTENT = 0).
- **HOLDS-slow** — same property but the bar ticks only every 2-3 actions
  (CONTENT = 0, HUD_ONLY >= 1, remainder FROZEN). The assertion holds on every
  step where the raw diff is non-empty.
- **FROZEN-noop** — a no-op exists but the frame is bit-identical under it
  (bar does not tick per-action for that action type): assertion untestable,
  raw frame-equality already works there; no evidence against the mask.
- **NO-PURE-NOOP** — every candidate action changed real (unmasked) content.
- **NO-HUD** — detector found no wavefront strips.

| game | verdict | validated no-op action(s) | mask cells | full-edge-strip risk | multi-change cells in mask |
|------|---------|--------------------------|------------|----------------------|-----------------------------|
| ar25 | HOLDS | ACTION5 | 35 | no | 0 |
| bp35 | HOLDS | ACTION7, A6(corner) | 88 | no | 0 |
| cd82 | HOLDS | ACTION1, ACTION2, A6(corner) | 96 | no | 0 |
| cn04 | HOLDS-slow | A6(corner) 3/8 ticks | 26 | no | 0 |
| dc22 | HOLDS | A6(corner) | 30 | no | 0 |
| ft09 | FROZEN-noop | A6(corner) frozen 8/8 | 10 | no | 0 |
| g50t | HOLDS | ACTION1, ACTION3, ACTION5 | 30 | no | 0 |
| ka59 | HOLDS | A6(corner) | 38 | no | 0 |
| lf52 | HOLDS | ACTION1-4, ACTION7, A6(corner) | 60 | no | 0 |
| lp85 | FROZEN-noop | A6(corner) frozen 8/8 | 20 | no | 0 |
| ls20 | NO-HUD | — | 0 | — | — |
| m0r0 | HOLDS-slow | ACTION5, A6(corner) 3/8 ticks | 52 | no | 0 |
| r11l | HOLDS | A6(corner) | 64 | **yes** | 0 |
| re86 | NO-PURE-NOOP | — (all of A1-A5 move content) | 49 | no | 0 |
| s5i5 | HOLDS | A6(corner) | 64 | **yes** | 0 |
| sb26 | HOLDS | ACTION5 | 20 | no | 0 |
| sc25 | FROZEN-noop | A6(corner) frozen 8/8 | 104 | no | 0 |
| sk48 | HOLDS-slow | ACTION2 2/8 ticks (A7/A6 frozen) | 8 | no | 0 |
| sp80 | HOLDS | ACTION5, A6(corner) | 64 | **yes** | 0 |
| su15 | FROZEN-noop | ACTION7 + A6(corner) frozen 8/8 | 38 | no | 0 |
| tn36 | HOLDS | A6(corner) | 60 | no | 0 |
| tr87 | NO-PURE-NOOP | — (all of A1-A4 move content) | 30 | no | 0 |
| tu93 | HOLDS | ACTION1, ACTION2, ACTION3 | 64 | **yes** | 0 |
| vc33 | HOLDS | A6(corner) | 64 | **yes** | 0 |
| wa30 | HOLDS-slow | ACTION5 3/8 ticks | 19 | no | 0 |

Counts: HUD detected on **24/25** (all but ls20). Mask assertion (masked diff
empty while raw diff non-empty) **directly validated on 18/25** (14 HOLDS + 4
HOLDS-slow). 4 FROZEN-noop games could not exercise the assertion (the bar
does not tick under the no-op action, so plain frame equality already
suffices there). 2 games (re86, tr87) had no scriptable pure no-op among the
candidates tried — every action moves content — so the assertion is untested
there, not falsified. **Zero false-mask evidence anywhere**: in no game did a
masked cell change more than once during the clean probe, and no validated
no-op ever changed unmasked content.

Guard check: the full-width edge-strip risk flagged in the research
(sp80/lf52/vc33/r11l) reproduces partially: sp80, vc33, r11l flag (full
64-cell strip on a border row/col), plus s5i5 and tu93; lf52's strip is 60
cells, not full-width, so it escapes the geometric flag. On all five flagged
games the probes showed no real content inside the strip (0 multi-change
cells), but a 60-step probe cannot prove content never enters that row —
treat these 5 masks as usable-with-monitoring (e.g. unmask if a masked cell
ever changes twice without a RESET).

## Gate decisions

- **Replay-harvest lever (Test A gate): OPEN.** Local engine replays are
  bit-exact over 100-action mixed prefixes on all 25 games, including through
  death+reset. Remaining risk is gateway-side parity, which this test cannot
  observe.
- **HUD mask (Test B gate): PASS with per-game caveats.** Mask precision is
  clean everywhere it was testable (18/25 direct, 0 false positives); ship the
  mask with the multi-change unmask guard, and expect no benefit on ls20
  (no HUD) and reduced no-op-veto value on re86/tr87 (no pure no-op exists).
