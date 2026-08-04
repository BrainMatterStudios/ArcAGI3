"""Long-form programmatic deliberation for scripted-teacher turns.

Turns corpus_v2 turn semantics (episodes.py) into reasoning text whose LENGTH
DISTRIBUTION matches corpus_v3 (target-token mean ~1487, P10 ~354, P90 ~2937)
and whose content is genuinely grounded: state reading with real coordinates,
explicit hypothesis statements, per-action predictions, observation-vs-
prediction comparison, and explicit revision. Nothing here is padding — every
sentence is derived from ground truth the generator knows.

Anti-template design: every semantic slot renders through a bank of phrasing
variants chosen by a per-record seeded rng, core nouns go through per-record
synonym pools, and most sentences interpolate coordinates/counts, so shared
8-gram runs across records are rare (gated in validate_v2).

HARD CONSTRAINT honoured: corpus_v3 was consulted ONLY for length/format
statistics. All phrasing below is written from scratch; no sentence of
corpus_v3 (a DO_NOT_TRAIN Claude-teacher corpus) is copied or paraphrased.
"""

from __future__ import annotations

import math
import random
from typing import Any

CHARS_PER_TOKEN = 3.0  # measured on corpus_v3 targets (mean 3.02)

# corpus_v3 reference target-token distribution (measured 2026-08-04 from
# submission/_sft_k3/corpus_v3 meta.qwen_target; lengths only, no text):
REF_TARGET = {"mean": 1487, "p10": 354, "p50": 1250, "p90": 2937, "max": 4883}


def _tok(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def draw_target_budget(rng: random.Random) -> int:
    """Two-piece lognormal around the corpus_v3 median: sigma below/above the
    median chosen so P10/P90 land near 354/2937 tokens."""
    z = rng.gauss(0.0, 1.0)
    sigma = 1.30 if z < 0 else 0.72
    val = math.exp(math.log(2000) + sigma * z)
    return int(min(max(val, 150), 4600))


# ---------------------------------------------------------------------------
# realization machinery
# ---------------------------------------------------------------------------


class Lex:
    """Per-record phrasing chooser: bank variants + consistent synonym picks."""

    SYN = {
        "avatar": ["avatar", "player square", "controlled sprite", "agent marker"],
        "grid": ["grid", "board", "playfield", "arena"],
        "cell": ["cell", "tile", "square"],
        "seems": ["appears", "seems", "looks like it", "evidently"],
        "hypothesis": ["hypothesis", "working theory", "candidate model", "reading"],
        "goal_word": ["goal", "objective", "win condition", "target state"],
    }

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.syn = {k: rng.choice(v) for k, v in self.SYN.items()}

    def w(self, key: str) -> str:
        return self.syn[key]

    def pick(self, variants: list[str], **slots: Any) -> str:
        t = self.rng.choice(variants)
        # explicit slots override the per-record synonym pool on collision
        return t.format(**{**self.syn, **slots}) if slots or "{" in t else t


def _rc(cell) -> str:
    return f"({cell[0]},{cell[1]})"


def _md(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# hypothesis library (statement / prediction / rejection realizations)
# ---------------------------------------------------------------------------

HYP = {
    # -- replay -------------------------------------------------------------
    "direct_walk": {
        "stmt": [
            "Simplest {hypothesis}: this is a plain corridor {grid} and the far chamber is reachable on foot; the single-{cell} gap in the dividing wall is just an opening.",
            "My first {hypothesis} treats the level as ordinary navigation — walk through the gap in the wall line and touch the far marker.",
            "Assume vanilla maze rules for now: the wall has one gap, so the route to the far chamber runs through it.",
            "Initial model: nothing here is stateful; the break in the vertical wall should behave like floor.",
        ],
        "reject": [
            "That collapsed when the gap refused to admit the {avatar}.",
            "Refuted: the gap is not an opening — movement into it does nothing.",
            "Dead — the gap blocks like a wall despite looking like a doorway.",
        ],
    },
    "plate_latch": {
        "stmt": [
            "Second {hypothesis}: the lone tinted floor {cell} is a toggle switch — step on it once and the barrier in the wall opens for good.",
            "New {hypothesis}: that isolated coloured {cell} is a pressure switch that latches the door open permanently after one touch.",
            "Maybe the odd {cell} is a one-shot opener: touch it, then the barrier stays open.",
        ],
        "reject": [
            "Wrong — the barrier snapped shut the moment I stepped off, so it is held open, not latched.",
            "Refuted: door state tracks occupancy of the {cell}, not history.",
            "That failed: leaving the tinted {cell} re-sealed the doorway immediately.",
        ],
    },
    "a5_probe": {
        "stmt": [
            "The plate needs a continuous occupant, and I cannot stand here and be at the door at once. SPACE is the only unexplored input — try it while standing on the plate.",
            "Something must HOLD this {cell} while I walk. The untested action is SPACE; pressing it right here is the cheapest probe.",
            "I need a second body. SPACE has done nothing obvious so far, so I will fire it while on the plate and watch every pixel.",
        ],
    },
    "ghost_replay": {
        "stmt": [
            "Confirmed model: SPACE banks every move made so far and spawns a copy at the spawn {cell}; the copy replays the recorded route one step per action I take. Because I banked while standing on the plate, the copy will finish ON the plate and hold the door for me.",
            "The mechanic is record-and-replay: my walk was being recorded; SPACE deposited a ghost that re-walks that exact route in lockstep with my new moves, ending where I banked — the plate. The teleport back to spawn is the price, not a punishment.",
            "Working model, now solid: this is a two-phase tool. Phase one records a path ending on the plate; SPACE banks it and resets me; phase two the ghost replays it step-for-step while I walk the freed route.",
        ],
    },
    # -- carry --------------------------------------------------------------
    "space_probe": {
        "stmt": [
            "Arrows move the {avatar} and the last pressed arrow leaves it oriented that way even when the move is blocked. SPACE is untested; I will face the block and press it.",
            "Movement is plain 4-way; the leftover unknown is SPACE. Best probe: stand beside a block, face it, press SPACE, diff the frame.",
        ],
    },
    "space_destroy": {
        "stmt": [
            "Current read: SPACE deletes whatever the {avatar} is facing — the block did disappear outright.",
            "Tentative {hypothesis}: SPACE is a destroy tool; the faced block was removed from the {grid}.",
        ],
        "reject": [
            "Refuted — the \"destroyed\" block came back when I pressed SPACE on open floor, so it was never destroyed at all.",
            "Wrong: deletion cannot explain the block re-materialising in the faced {cell}.",
        ],
    },
    "carry": {
        "stmt": [
            "Corrected model: SPACE picks the faced block up into an invisible carry, and a second SPACE sets it down in the currently faced {cell}. The vanish/reappear pair was pick-up and put-down, not destruction.",
            "The verbs are grab and place: facing a block, SPACE lifts it (it leaves the board); facing empty floor, SPACE plants the carried block there. So the delivery loop is face-grab-walk-face-place.",
            "This is a carry mechanic. The block is not gone while carried — it travels with me unseen until I release it. {goal_word}: every block parked inside the tinted zone, hands empty.",
        ],
    },
    # -- mirror -------------------------------------------------------------
    "single_dot": {
        "stmt": [
            "Default {hypothesis}: one of the two identical dots is mine and the other is scenery or an NPC; walk mine onto the other one.",
            "First read: arrows steer a single dot; the twin should hold still while I close the gap.",
        ],
        "reject": [
            "Refuted immediately: BOTH dots moved on every arrow.",
            "Wrong — the second dot is not independent; it echoed every input.",
        ],
    },
    "coupled_mirror": {
        "stmt": [
            "Correct model: every arrow drives BOTH dots, with the twin's {ax_word} component negated — {mirror_desc}. A dot blocked by a wall stays put while the other still moves, so walls are the only lever that changes the offset. {goal_word}: both dots on the same {cell}.",
            "The dots are one control system: my input is applied to both, mirrored along the {ax_word} axis. Wall contact is asymmetric and therefore the offset-editing tool. Merge them to clear the level.",
            "Coupled twin movement, {ax_word}-mirrored: same rows logic, opposite columns logic (or vice versa). Since only wall-jams desynchronise them, the plan is to steer one dot against a wall the other does not touch, shrink the offset to zero, and land both on one {cell}.",
        ],
    },
    # -- rules --------------------------------------------------------------
    "copy_target": {
        "stmt": [
            "Obvious {hypothesis}: the editable bottom row must reproduce the fixed row above it, {cell} by {cell}.",
            "First guess: this is a copy task — cycle each bottom glyph until it equals the glyph directly above.",
        ],
        "reject": [
            "Refuted: a perfect copy earned no confirmation mark, so equality with the row above is not the criterion.",
            "Wrong — matching the upper glyph produced nothing; the check must involve the legend, not identity.",
        ],
    },
    "legend_map": {
        "stmt": [
            "Correct model: the paired glyphs at the top are rewrite rules LHS→RHS. For each column the working glyph must equal rule(target glyph), not the target glyph itself. The little bar under a column confirms that column.",
            "The legend is the specification: each top pair says \"this glyph maps to that glyph\". Apply the mapping to every fixed-row glyph and dial the editable row to the mapped values; per-column bars confirm progress.",
            "This is symbol rewriting. Read the fixed glyph in a column, look it up in the legend, cycle the editable glyph to the legend's right-hand partner. All columns confirmed = level complete.",
        ],
    },
    # -- nav ----------------------------------------------------------------
    "direct_goal": {
        "stmt": [
            "Baseline {hypothesis}: plain maze — walk the shortest corridor to the {goal_word} {cell}; the odd coloured tiles are decoration.",
            "Treat it as vanilla pathfinding first: the coloured singleton tiles get ignored and the {goal_word} is reached on foot.",
        ],
        "reject": [
            "Refuted: the gap in the wall line rejects movement — it is a closed door, and the coloured tiles are probably its key.",
            "Wrong — a barrier {cell} blocks the direct corridor; the decoration theory dies here.",
        ],
    },
    "keys_doors": {
        "stmt": [
            "Model: doors embedded in the wall lines stay shut until their matching key/switch tile is stepped on; hazards kill on contact; the {goal_word} tile ends the level. Route = keys in band order, then the {goal_word}.",
            "The maze is gated: each wall-line has one door {cell} keyed to a pickup earlier in the corridor. Collect the pickup, the door clears, proceed; never touch a hazard {cell}.",
            "Confirmed structure: key-before-door progression with lethal tiles to route around. The solver line is pickup(s) first, then through each opened door to the {goal_word}.",
        ],
    },
    # -- click --------------------------------------------------------------
    "click_match": {
        "stmt": [
            "Read: the framed swatch top-left is a legend naming the wanted property; clicking every object that matches it should clear them.",
            "{hypothesis}: match-the-legend — the boxed sample defines the target class, and clicks on members remove them.",
        ],
        "reject": [
            "But the first matching object shrugged the click off entirely — something upstream gates clicks.",
        ],
    },
    "gate_arm": {
        "stmt": [
            "Corrected model: the small corner button must be clicked FIRST; it arms the board, and only then do object clicks register.",
            "The board has an arming switch — the odd corner square. Sequence is arm, then harvest the matching objects.",
        ],
    },
    "misread_legend": {
        "stmt": [
            "Read: the legend seems to point at one class of objects; I will start with the nearest object that plausibly fits.",
        ],
        "reject": [
            "That cost a life pip — the picked object was NOT in the target class; re-reading the legend more literally.",
        ],
    },
    "legend_target": {
        "stmt": [
            "Model: the legend defines the exact target class ({rule_desc}); clicking a member removes it, clicking anything else burns a life pip. Clear all members to advance.",
            "Confirmed: {rule_desc} is the selection criterion. Only precise clicks on class members count; misses are punished via the pip row.",
        ],
    },
    # -- push ---------------------------------------------------------------
    "pushpull": {
        "stmt": [
            "Assumption to test: boxes move when walked into, and plausibly FOLLOW when I back away — push and pull. If pulling works, any layout is trivial.",
            "{hypothesis}: boxes are draggable both ways. First shove one, then reverse to see if it trails the {avatar}.",
        ],
        "reject": [
            "Pull refuted: the box ignored my retreat. Push-only physics — order and angles now matter a lot.",
            "No drag: the box stayed put when I backed off. This is one-way pushing.",
        ],
    },
    "push_plan": {
        "stmt": [
            "Model: strictly push-only Sokoban. A box moves one {cell} when walked into, only if the {cell} behind it is clear; boxes against walls can deadlock, so the pad order must be planned, not greedy.",
            "Push-only physics confirmed. That means irreversible mistakes exist: never shove a box flush to a wall unless its pad lies along that wall. Plan the whole sequence before committing.",
            "One-way box physics: approach from the side opposite the intended direction. The route below was searched to keep every box pushable until it parks on a pad.",
        ],
    },
}

WRONG_PRED = {
    "pass_gap": [
        "Prediction: the last step passes through the gap {gap} into the far chamber.",
        "If the gap is open floor, this sequence ends with the {avatar} inside the right-hand chamber at {gap}.",
        "Committing to the corridor theory: expect the final move to cross {gap} without resistance.",
    ],
    "latch_stays": [
        "Prediction under the latch theory: the door stays open now that the plate was touched, so the same gap {gap} admits me this time.",
        "If the switch latched, the barrier should remain absent while I walk away — the approach to {gap} will succeed.",
    ],
    "copy_confirm": [
        "Prediction: once column 0 shows the same glyph as the row above, some confirmation appears (a mark, a colour change, or the level ends).",
        "If copy-the-target is right, matching the upper glyph in column 0 triggers visible acknowledgement.",
    ],
    "target_vanish": [
        "Prediction: the clicked object disappears and the remaining count drops by one.",
        "Expect the object at {cell} to be removed on this click.",
    ],
    "close_distance": [
        "Prediction: my dot steps toward the other one each move while the twin stays put — the gap should shrink to {want}.",
        "Under the single-dot theory the separation drops by one per step; after this batch it should be {want}.",
    ],
    "box_follows": [
        "Prediction for the last step: the box trails me — it should end adjacent to me again after I back away.",
        "If drag physics exist, the box follows the retreat and lands where I just stood.",
    ],
    "space_noop_empty": [
        "Prediction: SPACE away from any block should do nothing at all — a strict destroy tool has no effect on empty floor.",
        "If SPACE is deletion, pressing it while facing open floor is a no-op; any frame change refutes deletion.",
    ],
}

REACT = {
    "blocked_door": [
        "Contradiction logged: I predicted the gap would admit the {avatar}, but the position did not change — the gap {cell} is a solid, stateful barrier.",
        "The frame diff shows no movement on the final step: the \"opening\" refused entry. My corridor {hypothesis} is dead.",
        "Predicted a pass-through; observed a stall. That gap is a closed door, which means something in this room opens it.",
    ],
    "door_open_on_plate": [
        "Observation matched the probe: the instant the {avatar} stood on the tinted {cell}, the barrier sprite vanished from the wall line.",
        "The plate works: door pixels disappeared exactly when I occupied the plate. Cause and effect are tight.",
    ],
    "door_needs_holder": [
        "Prediction failed: the door re-materialised on my very first step OFF the plate, and the gap blocked again. The plate is a HOLD-to-open control, not a toggle.",
        "Contradiction: barrier back the moment the plate emptied. Occupancy, not history, drives that door.",
    ],
    "ghost_marching": [
        "Confirmed frame over frame: the copy advances one recorded step for each action I spend, retracing my earlier walk exactly.",
        "The ghost is replaying my route in lockstep — every move I make, it makes my old move.",
    ],
    "block_vanished": [
        "Result: the faced block vanished from the {grid} outright. Under a destruction reading that is progress; but nothing else changed — no score, no zone reaction.",
        "The block at {cell} is gone after SPACE. Destroyed? Suspicious that the {goal_word} region did not react at all.",
    ],
    "block_reappeared": [
        "Contradiction: SPACE on open floor made the earlier block REAPPEAR in the faced {cell}. It was never destroyed — it was being carried invisibly.",
        "Prediction (no-op on empty floor) failed spectacularly: the vanished block popped back one {cell} ahead of me. Vanish+return = pick-up and put-down.",
    ],
    "twin_mirrored": [
        "Contradiction: both dots moved on every input, and horizontally they moved in OPPOSITE directions — the gap changed by two columns, not one.",
        "Prediction failed: the \"other\" dot is not independent. It mirrors my column moves and copies my row moves.",
    ],
    "no_mark": [
        "Contradiction: column 0 now equals the glyph above it and yet no mark appeared anywhere. Copying is not the criterion.",
        "The copy completed and the board stayed silent — no bar, no unlock. The equality theory is out.",
    ],
    "mark_appeared": [
        "Confirmation: the bar under column 0 lit up the moment the working glyph matched the legend's right-hand partner. The mapping model is correct.",
        "Mark observed under the corrected column — legend-mapping validated.",
    ],
    "click_no_effect": [
        "Contradiction: the click landed on a legend-matching object and the frame did not change at all — no removal, no penalty. Clicks are being ignored wholesale.",
        "Nothing happened. Not even a life pip moved. The board seems deaf until something enables it.",
    ],
    "gate_armed": [
        "The corner button flipped colour on click — the board is now armed, exactly as the arming theory predicted.",
        "Confirmation: button changed state; retrying the object click should now register.",
    ],
    "life_lost": [
        "Penalty observed: a life pip vanished after that click. The clicked object was NOT a target; my legend reading was off.",
        "That click cost a pip — clear negative signal that the chosen object is outside the target class.",
    ],
    "box_no_follow": [
        "Contradiction: after the push I stepped back and the box stayed exactly where the push left it. There is no pulling in this {grid}.",
        "Prediction failed: no drag. The box only moves when displaced from behind.",
    ],
}

PLAN_NOTE = {
    "head_for_gap": ["Head straight for the gap in the dividing wall."],
    "try_plate": ["Detour to the isolated tinted {cell} and watch the wall line while standing on it."],
    "press_space_on_plate": ["Stand on the plate and fire SPACE, diffing the frame for ANY change."],
    "record_then_bank": ["Walk a deliberate route that ends on the plate, then bank it with SPACE."],
    "wait_then_cross": [
        "The copy needs {ghost_len} actions to finish its route; I am {fillers} spare actions short, so burn exactly {fillers} harmless back-and-forth moves before crossing the doorway, then walk to the {goal_word}.",
        "Timing plan: ghost route length {ghost_len}; my path reaches the doorway too early, so insert {fillers} filler moves, then cross while the plate is held and finish at the {goal_word}.",
    ],
    "deliver_blocks": ["Loop per block: approach, face, grab, walk to the zone, face a free zone {cell}, place."],
    "probe_space_on_block": ["Approach the nearest block, face it with a deliberate bump, then press SPACE."],
    "jam_then_merge": ["Use wall contacts to edit the offset between the dots, then walk them onto a common {cell}."],
    "approach_twin": ["Steer toward the second dot along the shorter axis."],
    "match_row_above": ["Cycle column 0 until it shows the same glyph as the fixed row."],
    "apply_rule_cell0": ["Cycle column 0 to the legend's mapped glyph and watch for a mark."],
    "apply_rules_all": ["Sweep the cursor left-to-right, dialing each column to rule(target)."],
    "straight_to_goal": ["Shortest corridor toward the {goal_word} {cell}."],
    "key_then_goal": ["Collect the opener tile(s) first, then pass each door in order, then the {goal_word}."],
    "click_matching": ["Click the first object that fits the legend."],
    "probe_button": ["Click the odd corner button — it is the only element not explained by the matching theory."],
    "arm_first": ["Arm the board via the corner button before any object click."],
    "click_targets_order": ["Click the remaining targets in reading order, avoiding everything else."],
    "shove_and_drag": ["Push the box once, then back away to test whether it follows."],
    "push_order": ["Execute the searched push sequence; it keeps every box clear of deadlocks."],
}

EVIDENCE = {
    "door_solid": ["the wall gap at {door} rejects movement while unlit"],
    "door_opened_on_plate": ["standing on the plate {plate} removes the door sprite"],
    "door_closed_off_plate": ["the door re-seals the instant the plate is vacated"],
    "a5_ghost_spawn": ["SPACE spawned a copy at the spawn point {start} and teleported me back"],
    "block_vanished": ["SPACE while facing the block at {cell} removed it from view"],
    "block_reappeared": ["a later SPACE on open floor made that block reappear in the faced cell"],
    "twin_mirrors": ["both dots respond to every arrow, columns negated (axis {axis})"],
    "no_mark_on_copy": ["copying the fixed glyph into column {cell} produced no mark"],
    "mark_on_rule": ["setting column {cell} to the legend-mapped glyph produced a mark"],
    "click_ignored": ["a click on the matching object at {cell} had zero effect pre-arming"],
    "gate_armed": ["the corner button at {cell} flipped state when clicked"],
    "life_lost": ["clicking the object at {cell} burned a life pip"],
    "box_no_follow": ["a box never follows the {avatar}; only pushes move it"],
    "level_done": ["level {level} cleared with the current model"],
}


# ---------------------------------------------------------------------------
# per-family state reading (units with real coordinates)
# ---------------------------------------------------------------------------


def _state_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    st = sem["state"]
    fam = sem["family"]
    u: list[str] = []
    if fam == "replay":
        u.append(lex.pick([
            "The {grid} is a {rows}x{cols} lattice split by a vertical wall line; the {avatar} sits at {pos} in the left chamber.",
            "Layout check: {rows}x{cols} cells, one dividing wall; {avatar} at {pos}, left of the divider.",
        ], rows=st["rows"], cols=st["cols"], pos=_rc(st["pos"])))
        u.append(lex.pick([
            "The far chamber holds the presumed {goal_word} marker at {goal}; the only break in the divider is at {door}.",
            "Across the wall, a distinct marker at {goal}; the divider's single irregular {cell} sits at {door}.",
        ], goal=_rc(st["goal"]), door=_rc(st["door"][0])))
        u.append(lex.pick([
            "A lone tinted floor {cell} sits at {plate}, {d} steps from the {avatar} — the only unexplained furniture on this side.",
            "Also on the left side: an isolated coloured {cell} at {plate} (Manhattan distance {d} from here).",
        ], plate=_rc(st["plate"]), d=_md(st["pos"], st["plate"])))
        if st["banked"]:
            u.append(lex.pick([
                "The banked copy currently stands at {g}; {rec} recorded moves are queued for replay.",
                "Ghost status: at {g} with a {rec}-move recording to walk.",
            ], g=_rc(st["ghost"]) if st["ghost"] else "spawn", rec=st["recorded"]))
        u.append(lex.pick([
            "Door state this frame: {ds}.",
            "The barrier is currently {ds}.",
        ], ds="open" if st["door_open"] else "closed"))
    elif fam == "carry":
        u.append(lex.pick([
            "{rows}x{cols} room; {avatar} at {pos}, facing {f}.",
            "The {grid} spans {rows}x{cols} cells with the {avatar} at {pos} (facing {f}).",
        ], rows=st["rows"], cols=st["cols"], pos=_rc(st["pos"]),
            f="nothing yet" if not st["facing"] else _rc(st["facing"])))
        u.append(lex.pick([
            "Loose blocks on the floor: {blocks}; a tinted zone covers {zone}.",
            "Inventory of movables: blocks at {blocks}; the shaded region {zone} is the likely destination.",
        ], blocks=", ".join(_rc(b) for b in st["blocks"]) or "none visible",
            zone=", ".join(_rc(z) for z in st["zone"])))
        if st["carrying"]:
            u.append(lex.pick([
                "One block is currently OFF the board — consistent with it being carried.",
                "Note: a block is absent from the frame; under the carry model it travels with me.",
            ]))
        if st["blocks"]:
            near = min(st["blocks"], key=lambda b: _md(st["pos"], b))
            u.append(lex.pick([
                "Nearest block: {b}, {d} steps away.",
                "Closest movable sits at {b} (distance {d}).",
            ], b=_rc(near), d=_md(st["pos"], near)))
    elif fam == "mirror":
        u.append(lex.pick([
            "Two identical dots on a {rows}x{cols} {grid}: one at {a}, one at {b}.",
            "The {grid} ({rows}x{cols}) holds two same-coloured dots at {a} and {b}.",
        ], rows=st["rows"], cols=st["cols"], a=_rc(st["a"]), b=_rc(st["b"])))
        u.append(lex.pick([
            "Current offset between them: {gap} (rows,cols); Manhattan separation {d}.",
            "Separation vector {gap}; total distance {d}.",
        ], gap=_rc(st["gap"]), d=abs(st["gap"][0]) + abs(st["gap"][1])))
    elif fam == "rules":
        u.append(lex.pick([
            "Top of the frame: {k} glyph pairs laid out as LHS=RHS rows — a legend. Bottom: a fixed glyph row and an editable row of {n} columns with a cursor box on column {cur}.",
            "Frame anatomy: {k} paired-glyph legend rows up top; below, a {n}-column fixed row over a {n}-column editable row; cursor currently on column {cur}.",
        ], k=len(st["rules"]), n=st["n"], cur=st["cursor"]))
        marks = st["marks"]
        u.append(lex.pick([
            "Columns confirmed so far: {m} of {n} ({which}).",
            "Mark bars present under {m}/{n} columns ({which}).",
        ], m=sum(marks), n=st["n"],
            which=", ".join(str(i) for i, m in enumerate(marks) if m) or "none"))
        u.append(lex.pick([
            "Fixed row reads: {t}; editable row currently: {w}.",
            "Row contents — target: {t}; working: {w}.",
        ], t=" ".join(st["target"]), w=" ".join(st["working"])))
    elif fam == "nav":
        u.append(lex.pick([
            "Maze {rows}x{cols}; {avatar} at {pos}, {goal_word} tile at {goal} ({d} steps as the crow flies).",
            "{avatar} at {pos} in a {rows}x{cols} maze; the {goal_word} {cell} sits at {goal}, straight-line budget {d}.",
        ], rows=st["rows"], cols=st["cols"], pos=_rc(st["pos"]), goal=_rc(st["goal"]),
            d=_md(st["pos"], st["goal"])))
        if st["doors"]:
            u.append(lex.pick([
                "Wall-line breaks (door candidates) at {doors}; singleton tiles at {keys}{sw}.",
                "Doors: {doors}. Pickup-looking tiles: {keys}{sw}.",
            ], doors=", ".join(_rc(d) for d in st["doors"]),
                keys=", ".join(_rc(k) for k in st["keys"]) or "none",
                sw=("; switch at " + ", ".join(_rc(s) for s in st["switches"])) if st["switches"] else ""))
        if st["hazards"]:
            u.append(lex.pick([
                "Hazard tiles to route around: {h}.",
                "Lethal-looking cells at {h} — excluded from every path below.",
            ], h=", ".join(_rc(h) for h in st["hazards"])))
        if st["opened"]:
            u.append(lex.pick([
                "Already opened this level: {o}.",
            ], o=", ".join(st["opened"])))
    elif fam == "click":
        u.append(lex.pick([
            "{n} objects remain on the board; rule in play: {rule}; {rem} presumed targets left; {lv} life pips.",
            "Board census: {n} objects, {rem} still to clear under the {rule} reading, {lv} pips in the corner.",
        ], n=len(st["objects"]), rule=st["rule"], rem=st["remaining"], lv=st["lives"]))
        tg = [o for o in st["objects"] if o["target"]]
        if tg:
            u.append(lex.pick([
                "Candidate targets at {cells}.",
                "The matching objects sit at {cells}.",
            ], cells=", ".join(_rc(o["cell"]) for o in tg)))
        if st["gate"]:
            u.append(lex.pick([
                "The odd corner button is {a}.",
                "Arming state: {a}.",
            ], a="armed" if st["armed"] else "not armed yet"))
    elif fam == "push":
        u.append(lex.pick([
            "Sokoban-style room {rows}x{cols}: {avatar} at {pos}, boxes at {boxes}, pads at {pads}.",
            "Push layout — {avatar} {pos}; movable boxes {boxes}; destination pads {pads} ({rows}x{cols} room).",
        ], rows=st["rows"], cols=st["cols"], pos=_rc(st["pos"]),
            boxes=", ".join(_rc(b) for b in st["boxes"]),
            pads=", ".join(_rc(p) for p in st["pads"])))
        on = [b for b in st["boxes"] if b in st["pads"]]
        u.append(lex.pick([
            "Boxes already parked on pads: {k} of {n}.",
            "Progress: {k}/{n} pads covered.",
        ], k=len(on), n=len(st["pads"])))
    return u


def _axis_words(axis: str) -> dict[str, str]:
    if axis == "x":
        return {"ax_word": "column",
                "mirror_desc": "LEFT/RIGHT push them apart or together while UP/DOWN moves them in parallel"}
    return {"ax_word": "row",
            "mirror_desc": "UP/DOWN push them oppositely while LEFT/RIGHT moves them in parallel"}


def _hyp_slots(sem: dict[str, Any]) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    st = sem["state"]
    if sem["family"] == "mirror":
        slots.update(_axis_words(st["axis"]))
    if sem["family"] == "click":
        rule = st.get("rule")
        slots["rule_desc"] = {
            "match_color": "same colour as the legend swatch",
            "match_shape": "same outline as the legend shape",
            "odd_one_out": "the one object whose colour breaks the pattern",
        }.get(rule, str(rule))
    return slots


def _steps_units(lex: Lex, spec: dict[str, Any], sem: dict[str, Any]) -> list[str]:
    units = []
    for i, s in enumerate(sem["steps"]):
        kind = s.get("kind", "move")
        if kind in ("move",) and s.get("frm") and s.get("to"):
            note = s.get("note")
            if note == "through_gap":
                units.append(lex.pick([
                    "Step {i}: {d} — from {a} into the gap at {b}.",
                    "Step {i}: {d} takes the {avatar} from {a} through the opening at {b}.",
                    "Step {i}: {d}, crossing the wall line at {b}.",
                ], i=i + 1, d=s["disp"], a=_rc(s["frm"]), b=_rc(s["to"])))
            else:
                units.append(lex.pick([
                    "Step {i}: {d} — {a} to {b}.",
                    "Step {i}: {d} moves the {avatar} {a} -> {b}.",
                    "{i}. {d}: expect {a} -> {b}, nothing else changing.",
                    "Step {i}: {d}; the {avatar} should land on {b}.",
                ], i=i + 1, d=s["disp"], a=_rc(s["frm"]), b=_rc(s["to"])))
                if s.get("ghost_to"):
                    units.append(lex.pick([
                        "  (simultaneously the copy advances to {g})",
                        "  — and the replaying copy steps to {g};",
                    ], g=_rc(s["ghost_to"])))
        elif kind == "pair":
            units.append(lex.pick([
                "Step {i}: {d} — dot one {a}->{b}, dot two {c}->{e}; offset becomes {g}.",
                "Step {i}: {d}; expected joint move: {a}->{b} and {c}->{e}, separation {g}.",
                "{i}. {d}: my dot {a} to {b} while the twin goes {c} to {e} (gap {g}).",
            ], i=i + 1, d=s["disp"], a=_rc(s["frm"]), b=_rc(s["to"]),
                c=_rc(s["b_frm"]), e=_rc(s["b_to"]), g=_rc(s["gap"])))
        elif kind == "bump":
            note = s.get("note", "")
            if note == "into_block" or sem["family"] == "carry":
                units.append(lex.pick([
                    "Step {i}: {d} into the block — expected to be BLOCKED; its only job is to set my facing.",
                    "Step {i}: {d} is a deliberate bump; position should not change, orientation should.",
                    "Step {i}: {d}, a facing-setter — the block stops the move but the {avatar} turns.",
                ], i=i + 1, d=s["disp"]))
            else:
                units.append(lex.pick([
                    "Step {i}: {d} — expected BLOCKED at {a} (no position change).",
                    "Step {i}: {d} should not move the {avatar}; the {cell} ahead is solid.",
                ], i=i + 1, d=s["disp"], a=_rc(s["frm"]) if s.get("frm") else "the current cell"))
        elif kind == "bank":
            units.append(lex.pick([
                "Step {i}: SPACE, pressed while standing on the plate.",
                "Step {i}: SPACE — the probe itself, fired from the plate.",
                "Step {i}: SPACE with the {avatar} parked on the tinted {cell}.",
            ], i=i + 1))
        elif kind == "probe":
            units.append(lex.pick([
                "Step {i}: SPACE with the current facing.",
                "Step {i}: fire SPACE and diff the frame for any change.",
                "Step {i}: SPACE — watching the faced {cell} specifically.",
            ], i=i + 1))
        elif kind == "grab":
            units.append(lex.pick([
                "Step {i}: SPACE — the faced block should leave the board into carry.",
                "Step {i}: SPACE lifts the faced block; expect it to vanish from the frame.",
            ], i=i + 1))
        elif kind == "release":
            units.append(lex.pick([
                "Step {i}: SPACE — the carried block should materialise at {b}.",
                "Step {i}: SPACE places the carried block into {b}.",
            ], i=i + 1, b=_rc(s["to"])))
        elif kind == "filler":
            units.append(lex.pick([
                "Then the filler shuffle near spawn — pairs of opposed moves that spend actions without changing my net position, purely to let the copy finish its route.",
                "Next a run of deliberate time-burners: alternating opposite moves beside the spawn {cell} so the copy keeps marching while I stay put.",
            ]))
        elif kind == "push":
            units.append(lex.pick([
                "Step {i}: {d} — walk into the box, shoving it to {bx}; I take {b}.",
                "Step {i}: {d} displaces the box to {bx} while the {avatar} moves {a} -> {b}.",
                "{i}. {d}: box slides to {bx}, {avatar} follows into {b}.",
            ], i=i + 1, d=s["disp"], a=_rc(s["frm"]), b=_rc(s["to"]),
                bx=_rc(s.get("box_to", s["to"]))))
        elif kind == "pickup":
            units.append(lex.pick([
                "Step {i}: {d} onto the pickup {cell} at {b} — its door should clear.",
                "Step {i}: {d} lands on the opener tile at {b}; expect the matching door to vanish.",
            ], i=i + 1, d=s["disp"], b=_rc(s["to"])))
        elif kind == "cursor":
            units.append(lex.pick([
                "Press {i}: {d} — cursor slides to column {c}.",
                "Press {i}: {d} moves the selection box to column {c}.",
            ], i=i + 1, d=s["disp"], c=s["to"][1]))
        elif kind == "cycle":
            if s.get("g0") is not None:
                units.append(lex.pick([
                    "Press {i}: {d} — column {c} cycles {g0} -> {g1}{mk}.",
                    "Press {i}: {d} turns column {c}'s glyph from {g0} to {g1}{mk}.",
                ], i=i + 1, d=s["disp"], c=s.get("col", "?"), g0=s["g0"], g1=s["g1"],
                    mk=" (mark expected)" if s.get("mark") else ""))
            else:
                units.append(lex.pick([
                    "Press {i}: {d} — one glyph step on the selected column.",
                    "Press {i}: {d} advances the column's glyph by one alphabet slot.",
                ], i=i + 1, d=s["disp"]))
        elif kind in ("click", "click_arm"):
            units.append(lex.pick([
                "Click {i}: MOUSE on {b}.",
                "Click {i}: target the object at {b}.",
                "Click {i}: MOUSE aimed at grid {b}.",
            ], i=i + 1, b=_rc(s["to"])))
        else:
            units.append(lex.pick([
                "Step {i}: {d}.",
                "Step {i}: {d} (direction input; every coupled entity responds).",
            ], i=i + 1, d=s.get("disp", "?")))
        # pixel-space grounding + expected-diff sentence per informative step
        if s.get("to") and s.get("frm") and kind in ("move", "push", "pair", "release", "pickup"):
            if i % 2 == 0 or i < 12:
                units.append(lex.pick([
                    "  ({b} renders at {box})",
                    "  — on the raw frame that {cell} is {box};",
                    "  (pixel check: destination {cell} = {box})",
                ], b=_rc(s["to"]), box=_box_str(spec, s["to"])))
            if s["frm"] != s["to"]:
                units.append(lex.pick([
                    "  Expected diff: the moving colour block leaves {fa} and fills {fb}; any other change would be unmodelled.",
                    "  Frame delta to verify: {fa} reverts to background while {fb} takes the mover's colour.",
                    "  The diff should be exactly two {cell} regions: vacated {fa}, occupied {fb}.",
                ], fa=_box_str(spec, s["frm"]), fb=_box_str(spec, s["to"])))
    return units


def _macro_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    """Family-specific plan arithmetic: per-entity sub-goals with distances."""
    st = sem["state"]
    fam = sem["family"]
    out: list[str] = []
    if fam == "rules":
        rules = st["rules"]
        alpha = list(GLYPH_ORDER)
        out.append(lex.pick([
            "Per-column derivation table:",
            "Working through every column against the legend:",
        ]))
        for i in range(st["n"]):
            t = st["target"][i]
            need = rules.get(t)
            cur = st["working"][i]
            if need is None:
                continue
            if cur == need:
                out.append(lex.pick([
                    "- column {i}: target {t} maps to {need}; already showing {need} — done;",
                    "- column {i}: {t}->{need} per the legend, and the cell is correct;",
                ], i=i, t=t, need=need))
            else:
                d = (alpha.index(need) - alpha.index(cur)) % 7
                presses = d if d <= 3 else 7 - d
                key = "UP" if d <= 3 else "DOWN"
                out.append(lex.pick([
                    "- column {i}: target {t} -> rule says {need}; currently {cur}, so {p}x {k};",
                    "- column {i}: legend maps {t} to {need}; cell shows {cur} = {p} presses of {k};",
                ], i=i, t=t, need=need, cur=cur, p=presses, k=key))
    elif fam == "carry":
        zone = [tuple(z) for z in st["zone"]]
        for j, b in enumerate(st["blocks"]):
            if tuple(b) in zone:
                out.append(lex.pick([
                    "Block {j} at {b} is already inside the zone — leave it.",
                    "- block {j} ({b}): parked in the zone, no work left;",
                ], j=j + 1, b=_rc(b)))
            else:
                dz = min(_md(b, z) for z in zone) if zone else 0
                out.append(lex.pick([
                    "Sub-goal for block {j} at {b}: reach an adjacent {cell}, bump to face it, grab, ferry roughly {d} steps to the zone, then face a free zone {cell} and place.",
                    "- block {j} ({b}): approach + face-bump + grab, then a ~{d}-step carry into the zone and a facing release;",
                ], j=j + 1, b=_rc(b), d=dz))
        if st["carrying"]:
            out.append(lex.pick([
                "A block is in hand right now, so the immediate move is the delivery leg, not another grab.",
                "Currently carrying: finish the drop before touching anything else.",
            ]))
    elif fam == "push":
        pads = [tuple(p) for p in st["pads"]]
        for j, b in enumerate(st["boxes"]):
            if tuple(b) in pads:
                out.append(lex.pick([
                    "- box {j} ({b}): already on a pad — do not disturb it;",
                    "Box {j} at {b} is parked; route around it.",
                ], j=j + 1, b=_rc(b)))
            else:
                dp = min(_md(b, p) for p in pads) if pads else 0
                out.append(lex.pick([
                    "- box {j} ({b}): nearest pad is {d} pushes away at best; each push needs me on the opposite side;",
                    "Box {j} at {b}: lower bound {d} pushes to a pad, plus repositioning walks.",
                ], j=j + 1, b=_rc(b), d=dp))
    elif fam == "replay":
        out.append(lex.pick([
            "Timing ledger: the recording is {rec} moves long, so after banking, the copy needs exactly {rec} of my actions to finish and sit on the plate.",
            "Bookkeeping that matters here: {rec} recorded moves = {rec} actions of replay before the plate is held.",
        ], rec=st["recorded"] if not st["banked"] else st["recorded"]))
        out.append(lex.pick([
            "Distances: {avatar} to plate {d1}, spawn to doorway column {d2}, doorway to far marker {d3} — the crossing must start no earlier than the replay's completion.",
            "Route arithmetic: plate leg {d1} steps; spawn-to-door {d2}; door-to-marker {d3}. The door is only usable while the plate is pressed.",
        ], d1=_md(st["pos"], st["plate"]), d2=_md(st["start"], st["door"][0]),
            d3=_md(st["door"][0], st["goal"])))
    elif fam == "mirror":
        gap = st["gap"]
        out.append(lex.pick([
            "Offset analysis: the dots differ by {g} — rows off by {dr}, columns by {dc}. Parallel moves preserve this; only a one-sided wall block changes it.",
            "Gap arithmetic: {g}. Every unblocked joint move keeps it constant, so the plan must include exactly the wall-jams that cancel it.",
        ], g=_rc(gap), dr=abs(gap[0]), dc=abs(gap[1])))
        if st["axis"] == "x":
            out.append(lex.pick([
                "Column logic is inverted for the twin: LEFT/RIGHT change the column gap by 2 or 0 (never 1); UP/DOWN slide both rows together.",
                "Because columns are mirrored, the column separation has fixed parity under free movement — a wall touch is required to break it.",
            ]))
        else:
            out.append(lex.pick([
                "Row logic is inverted for the twin: UP/DOWN change the row gap by 2 or 0; LEFT/RIGHT slide both columns together.",
            ]))
    elif fam == "nav":
        if st["doors"]:
            out.append(lex.pick([
                "Gate structure: {nd} door(s) at {doors} against {nk} pickup(s) — the route must thread pickup-then-door in band order.",
                "Sequencing constraint: each of the {nd} doors ({doors}) needs its opener first; {nk} opener tiles are visible.",
            ], nd=len(st["doors"]), doors=", ".join(_rc(d) for d in st["doors"]),
                nk=len(st["keys"]) + len(st["switches"])))
        out.append(lex.pick([
            "Lower bound: {d} steps of pure Manhattan distance to the {goal_word}; detours for openers and hazard avoidance come on top.",
            "The straight-line cost is {d} moves; the actual path pays extra for gates and hazards.",
        ], d=_md(st["pos"], st["goal"])))
    elif fam == "click":
        tg = [o for o in st["objects"] if o["target"]]
        for j, o in enumerate(tg[:6]):
            out.append(lex.pick([
                "- target {j}: the {sh} at {c} — one precise click, centre of its drawn pixels;",
                "Target {j}: {sh}-shaped object at {c}; aim inside the shape mask, not its bounding box.",
            ], j=j + 1, sh=o["shape"], c=_rc(o["cell"])))
    return out


GLYPH_ORDER = ("full", "ring", "plus", "diamond", "notch", "tee", "u")


def _frame_anatomy_units(lex: Lex, spec: dict[str, Any], sem: dict[str, Any]) -> list[str]:
    out = []
    fam = sem["family"]
    if fam == "rules":
        out.append(lex.pick([
            "The frame is native 64x64 with no letterboxing; legend rows sit in the top third, the two glyph rows in the bottom third.",
            "No scaling to worry about here — logical and display coordinates coincide.",
        ]))
        return out
    if fam == "click":
        g = spec["grid"]
        scale = 64 // g
        out.append(lex.pick([
            "The board is a {g}x{g} logical grid scaled x{s} to the 64x64 display; clicks are issued in display coordinates.",
            "Coordinate bookkeeping: {g}-cell grid, scale factor {s}; the MOUSE row/col below are display pixels.",
        ], g=g, s=scale))
        return out
    cell = spec.get("cell_px")
    if not cell:
        return out
    rows, cols = spec["rows"], spec["cols"]
    w, h = cols * cell, rows * cell
    scale = min(64 // w, 64 // h)
    ox, oy = (64 - w * scale) // 2, (64 - h * scale) // 2
    out.append(lex.pick([
        "Geometry: {r}x{c} logical cells at {px}px each, letterboxed with a {oy}-row top margin and {ox}-column left margin on the 64x64 frame.",
        "The playfield is {r}x{c} cells ({px}px per {cell}); margins of {oy} rows / {ox} cols of letterbox surround it.",
    ], r=rows, c=cols, px=cell * scale, oy=oy, ox=ox))
    if spec.get("hud"):
        out.append(lex.pick([
            "Display row 63 is a step-budget bar that ticks down — it changes on its own and must be excluded from any no-op/effect judgement.",
            "Note the HUD strip on the bottom display row: its countdown is not gameplay signal and would corrupt board_changed readings if included.",
        ]))
    else:
        out.append(lex.pick([
            "No HUD strip on this game — every pixel change is gameplay.",
            "There is no budget bar overlay here, so frame diffs are pure game state.",
        ]))
    return out


def _evidence_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    out = []
    ev = sem["evidence"]
    if not ev:
        return out
    intro = lex.pick([
        "Evidence ledger so far:",
        "Accumulated observations this run:",
        "What the log establishes so far:",
    ])
    items = []
    for eid, params in ev[-8:]:
        bank = EVIDENCE.get(eid)
        if not bank:
            continue
        try:
            items.append("- " + lex.pick(bank, **{k: (_rc(v) if isinstance(v, list) else v)
                                                  for k, v in params.items()}))
        except (KeyError, IndexError):
            continue
    if items:
        out.append(intro)
        out.extend(items)
    return out


def _considered_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    out = []
    slots = _hyp_slots(sem)
    for hid in sem["considered"]:
        h = HYP.get(hid)
        if not h:
            continue
        stmt = lex.pick(h["stmt"], **slots)
        rej = lex.pick(h.get("reject", ["Set aside for now."]), **slots)
        out.append(lex.pick([
            "Considered and rejected — {s} {r}",
            "Earlier {hypothesis}: {s} {r}",
        ], s=stmt, r=rej))
    return out


# ---------------------------------------------------------------------------
# world model / plan one-liners (assistant `content` + carried knowledge)
# ---------------------------------------------------------------------------

WM_BANK = {
    "direct_walk": [
        "corridor room; far marker reachable through the wall gap (unverified).",
        "single divider wall with a gap; assuming plain walking reaches the far side.",
        "treating the level as open navigation; the wall break should be passable.",
    ],
    "plate_latch": [
        "gap is a door; lone tinted cell may toggle it permanently.",
        "the break is a shut door; testing the isolated tile as its latch.",
        "door in the wall line; candidate opener = the odd floor tile.",
    ],
    "a5_probe": [
        "door is held open only while the plate is occupied; testing SPACE as the missing tool.",
        "plate must stay occupied for the door; probing SPACE from the plate.",
        "hold-to-open plate confirmed; SPACE is the remaining unknown input.",
    ],
    "ghost_replay": [
        "SPACE banks the recorded walk as a ghost that replays it step-per-action and can hold the plate; goal = far marker.",
        "record-and-replay tool: banked ghost re-walks my route and will sit on the plate while I cross.",
        "two-phase mechanic — record a plate-ending path, bank with SPACE, then walk out while the copy holds the door.",
    ],
    "space_probe": [
        "4-way movement with persistent facing; SPACE untested.",
        "arrows move and orient the avatar (even when blocked); probing SPACE next.",
        "movement mapped; facing persists on blocked moves; SPACE is the open question.",
    ],
    "space_destroy": [
        "SPACE appears to delete the faced block (tentative).",
        "working guess: SPACE removes whatever is directly ahead.",
        "faced block vanished on SPACE; holding a deletion reading loosely.",
    ],
    "carry": [
        "SPACE grabs the faced block into carry / releases it into the faced cell; goal = all blocks inside the zone.",
        "grab-and-place verbs on SPACE; deliver every block into the tinted region, hands empty.",
        "carry mechanic: face+SPACE lifts, face+SPACE drops; win = all blocks zoned.",
    ],
    "single_dot": [
        "two dots; assuming direct control of one, other independent.",
        "one dot presumed mine, the other scenery; walking toward it.",
        "single-avatar reading of the twin dots (unverified).",
    ],
    "coupled_mirror": [
        "both dots move on every arrow, one axis negated; walls desync them; goal = merge on one cell.",
        "twin dots share every input with a mirrored axis; wall-jams edit the offset; merging wins.",
        "coupled mirrored control confirmed; use blocking geometry to zero the offset and overlap the dots.",
    ],
    "copy_target": [
        "editable row should replicate the fixed row (unverified).",
        "assuming a copy task: bottom row mirrors the row above.",
        "working row presumed to need the target row's glyphs verbatim.",
    ],
    "legend_map": [
        "legend pairs are rewrite rules; working[i] must equal rule(target[i]); marks confirm columns.",
        "top pairs define glyph mappings; apply them per column to the editable row; per-column bars confirm.",
        "rewrite-rule board: dial each working glyph to the legend image of the glyph above it.",
    ],
    "direct_goal": [
        "plain maze to the goal tile; coloured singletons assumed decorative.",
        "treating the maze as ungated; walking the shortest corridor to the goal.",
        "no stateful tiles assumed yet; direct route to the goal cell.",
    ],
    "keys_doors": [
        "key/switch tiles open matching doors; hazards lethal; goal tile ends the level.",
        "gated maze: pick up openers in band order, doors clear, avoid hazard cells, finish on the goal.",
        "keys-before-doors structure confirmed; hazards are no-go cells; route ends at the goal tile.",
    ],
    "click_match": [
        "legend names the target class; clicking members should remove them.",
        "boxed sample = selection rule; clicks on matches presumed to clear them.",
        "match-the-legend reading; unverified whether clicks register unconditionally.",
    ],
    "gate_arm": [
        "corner button arms the board; only then do object clicks register.",
        "arming switch in the corner gates all clicks; arm first, then harvest.",
        "board starts deaf; the odd corner button enables object clicks.",
    ],
    "misread_legend": [
        "legend read loosely; starting with a plausible object.",
        "selection rule half-understood; probing with a likely candidate.",
    ],
    "legend_target": [
        "legend defines the exact target class; wrong clicks burn pips.",
        "target class pinned by the legend; misclicks cost life pips, so precision first.",
        "confirmed selection rule from the legend; only class members get clicked.",
    ],
    "pushpull": [
        "boxes move when walked into; testing whether they also follow.",
        "push confirmed; drag hypothesis under test.",
    ],
    "push_plan": [
        "push-only boxes, deadlocks possible; executing a searched push order.",
        "one-way pushes with deadlock risk; following a pre-searched order.",
        "Sokoban physics confirmed; the committed sequence keeps all boxes live.",
    ],
}

PLAN_BANK = {
    "orient": [
        "map the room and test the cheapest informative action.",
        "survey the frame, then spend the fewest actions that discriminate.",
        "orient first; probe only what the map cannot answer.",
    ],
    "misstep": [
        "act on the current best guess and watch for contradiction.",
        "commit to the leading hypothesis; treat any deviation as data.",
        "run the plan implied by the current model and check its prediction.",
    ],
    "probe": [
        "run the single probe that best splits the surviving hypotheses.",
        "one targeted experiment, then re-evaluate the model.",
        "isolate the unknown with a minimal test.",
    ],
    "revise": [
        "adopt the corrected model and re-plan from the current state.",
        "rebuild the plan on the revised model, from where things actually stand.",
        "with the fixed model, re-derive the route and commit.",
    ],
    "execute": [
        "execute the committed sequence; stop on any surprise.",
        "run the verified sequence; abort on the first unexpected diff.",
        "carry out the plan; any surprise pauses execution.",
    ],
}


# ---------------------------------------------------------------------------
# deep expanders (grounded, coordinate-heavy — supply for the long tail)
# ---------------------------------------------------------------------------


def _cell_box(spec: dict[str, Any], r: int, c: int) -> tuple[int, int, int, int]:
    """(row0, row1, col0, col1) display-pixel box of a logical cell."""
    if spec["family"] == "rules":
        return (r, r, c, c)
    if spec["family"] == "click":
        g = spec["grid"]
        scale = min(64 // g, 64 // g)
        ox = oy = (64 - g * scale) // 2
        return (r * scale + oy, (r + 1) * scale + oy - 1,
                c * scale + ox, (c + 1) * scale + ox - 1)
    cell = spec["cell_px"]
    w, h = spec["cols"] * cell, spec["rows"] * cell
    scale = min(64 // w, 64 // h)
    ox, oy = (64 - w * scale) // 2, (64 - h * scale) // 2
    return ((r * cell) * scale + oy, ((r + 1) * cell) * scale + oy - 1,
            (c * cell) * scale + ox, ((c + 1) * cell) * scale + ox - 1)


def _box_str(spec, cell) -> str:
    r0, r1, c0, c1 = _cell_box(spec, cell[0], cell[1])
    return f"display rows {r0}-{r1}, cols {c0}-{c1}"


def _geometry_units(lex: Lex, spec: dict[str, Any], sem: dict[str, Any]) -> list[str]:
    """Pixel-space grounding: where the key entities render in the 64x64 frame."""
    st = sem["state"]
    fam = sem["family"]
    ents: list[tuple[str, Any]] = []
    if fam == "replay":
        ents = [("avatar", st["pos"]), ("plate", st["plate"]),
                ("doorway", st["door"][0]), ("far marker", st["goal"])]
        if st["ghost"]:
            ents.append(("copy", st["ghost"]))
    elif fam == "carry":
        ents = [("avatar", st["pos"])] + [(f"block {i+1}", b) for i, b in enumerate(st["blocks"])]
        if st["zone"]:
            ents.append(("zone corner", st["zone"][0]))
    elif fam == "mirror":
        ents = [("dot one", st["a"]), ("dot two", st["b"])]
    elif fam == "nav":
        ents = [("avatar", st["pos"]), ("goal tile", st["goal"])]
        ents += [(f"door {i+1}", d) for i, d in enumerate(st["doors"])]
        ents += [(f"pickup {i+1}", k) for i, k in enumerate(st["keys"])]
    elif fam == "push":
        ents = [("avatar", st["pos"])] + [(f"box {i+1}", b) for i, b in enumerate(st["boxes"])]
        ents += [(f"pad {i+1}", p) for i, p in enumerate(st["pads"])]
    elif fam == "click":
        ents = [(f"object at {_rc(o['cell'])}", o["cell"]) for o in st["objects"][:6]]
    elif fam == "rules":
        return []
    out = []
    if ents:
        out.append(lex.pick([
            "Pixel mapping for the letter-coded view (each logical {cell} is a scaled block):",
            "Anchoring the lattice to the raw frame:",
        ]))
        for name, cell in ents[:8]:
            out.append(lex.pick([
                "- {n} occupies {box};",
                "- {n}: {box};",
                "- {n} sits at {box};",
            ], n=name, box=_box_str(spec, cell)))
    return out


def _hyp_space_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    """Discuss the family's other plausible-but-unconfirmed readings."""
    slots = _hyp_slots(sem)
    fam = sem["family"]
    pool = {
        "replay": ["direct_walk", "plate_latch"],
        "carry": ["space_destroy"],
        "mirror": ["single_dot"],
        "rules": ["copy_target"],
        "nav": ["direct_goal"],
        "click": ["click_match"],
        "push": ["pushpull"],
    }.get(fam, [])
    seen = set(sem["considered"]) | {sem["hyp"]["id"]}
    out = []
    for hid in pool:
        if hid in seen:
            continue
        h = HYP.get(hid)
        if not h:
            continue
        out.append(lex.pick([
            "Alternative kept on the shelf: {s} No evidence forces it yet.",
            "A rival reading exists — {s} It stays parked until something contradicts the current model.",
        ], s=lex.pick(h["stmt"], **slots)))
    return out


def _consistency_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    """For a true-model turn: show each logged observation is explained."""
    if sem["hyp"]["kind"] != "true" or not sem["evidence"]:
        return []
    out = [lex.pick([
        "Cross-checking the model against everything observed so far:",
        "Does the current model explain the whole log? Walking it:",
    ])]
    for eid, params in sem["evidence"][-6:]:
        bank = EVIDENCE.get(eid)
        if not bank:
            continue
        try:
            fact = lex.pick(bank, **{k: (_rc(v) if isinstance(v, list) else v)
                                     for k, v in params.items()})
        except (KeyError, IndexError):
            continue
        out.append(lex.pick([
            "- {f} — consistent with the model;",
            "- {f}: the model predicts exactly this;",
            "- {f} (explained);",
        ], f=fact))
    return out


def _step_check_units(lex: Lex, sem: dict[str, Any]) -> list[str]:
    out = []
    for s in sem["steps"]:
        if s.get("kind") == "move" and s.get("frm") and s.get("to"):
            dr = s["to"][0] - s["frm"][0]
            dc = s["to"][1] - s["frm"][1]
            out.append(lex.pick([
                "Check: {a} -> {b} is a delta of ({dr},{dc}), which is exactly one {d} step — consistent.",
                "Continuity check {a} to {b}: ({dr},{dc}) matches {d}.",
            ], a=_rc(s["frm"]), b=_rc(s["to"]), dr=dr, dc=dc, d=s["disp"]))
    return out[:10]


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def render_turn(
    spec: dict[str, Any], sem: dict[str, Any], rng: random.Random, budget_tokens: int
) -> tuple[str, str, str, str]:
    """-> (reasoning_content, content, wm_line, plan_line).

    Units carry a section index so the assembled text keeps narrative order:
    react -> state -> revision -> hypothesis -> alternatives -> plan -> steps
    -> prediction/commit -> evidence -> checks.
    """
    lex = Lex(rng)
    slots = _hyp_slots(sem)
    hyp = sem["hyp"]
    hyp_bank = HYP.get(hyp["id"], {"stmt": ["(model under construction)"]})

    units: list[tuple[int, bool, str]] = []  # (section, mandatory, text)

    # 0. reaction to the previous result (observation vs prediction)
    if sem.get("react"):
        bank = REACT.get(sem["react"]["id"])
        if bank:
            params = {k: (_rc(v) if isinstance(v, list) else v)
                      for k, v in sem["react"].get("params", {}).items()}
            try:
                units.append((0, True, lex.pick(bank, **params)))
            except (KeyError, IndexError):
                units.append((0, True, lex.pick(bank)))

    # 1. state reading (headline demoted to optional on very small budgets so
    # short turns can match corpus_v3's short-turn regime)
    tight = budget_tokens < 220
    state_units = _state_units(lex, sem)
    if state_units:
        units.append((1, not tight, state_units[0]))
        for su in state_units[1:]:
            units.append((1, False, su))
    for gu in _geometry_units(lex, spec, sem):
        units.append((1, False, gu))
    for fu in _frame_anatomy_units(lex, spec, sem):
        units.append((1, False, fu))

    # 2. explicit revision sentence
    if sem.get("revision"):
        frm_bank = HYP.get(sem["revision"]["frm"], {})
        rej = lex.pick(frm_bank.get("reject", ["it did not survive contact with the frame."]), **slots)
        units.append((2, True, lex.pick([
            "Revision: dropping the previous {hypothesis} — {rej}",
            "Model update. The old {hypothesis} is out: {rej}",
            "I am explicitly revising here. {rej}",
        ], rej=rej)))

    # 3. current hypothesis statement
    units.append((3, True, lex.pick(hyp_bank["stmt"], **slots)))

    # 4. considered-and-rejected + shelved alternatives
    for cu in _considered_units(lex, sem):
        units.append((4, False, cu))
    for au in _hyp_space_units(lex, sem):
        units.append((4, False, au))

    # 5. plan rationale + family plan arithmetic
    if sem.get("plan_note"):
        bank = PLAN_NOTE.get(sem["plan_note"]["id"])
        if bank:
            params = sem["plan_note"].get("params", {})
            units.append((5, not tight, lex.pick(bank, **params)))
    for mu in _macro_units(lex, sem):
        units.append((5, False, mu))

    # 6. per-step enumeration
    step_units = _steps_units(lex, spec, sem)
    if step_units:
        units.append((6, False, lex.pick([
            "Concretely, the batch below should play out as:",
            "Spelling the sequence out move by move:",
            "Predicted trajectory for this batch:",
        ])))
        for su in step_units:
            units.append((6, False, su))
        for cu in _step_check_units(lex, sem):
            units.append((6, False, cu))

    # 7. the committed (possibly wrong) prediction
    if sem.get("wrong_pred"):
        bank = WRONG_PRED.get(sem["wrong_pred"]["id"])
        if bank:
            params = dict(sem["wrong_pred"].get("params", {}))
            st = sem["state"]
            if sem["family"] == "replay":
                params.setdefault("gap", _rc(st["door"][0]))
            elif sem["family"] == "nav" and st.get("doors"):
                params.setdefault("gap", _rc(st["doors"][0]))
            if sem["wrong_pred"]["id"] == "close_distance":
                gap = st["gap"]
                params.setdefault("want", max(0, abs(gap[0]) + abs(gap[1]) - len(sem["steps"])))
            if sem["wrong_pred"]["id"] == "target_vanish" and sem["steps"]:
                params.setdefault("cell", _rc(sem["steps"][0]["to"]))
            try:
                units.append((7, True, lex.pick(bank, **params)))
            except (KeyError, IndexError):
                units.append((7, True, lex.pick(bank)))
        units.append((7, True, lex.pick([
            "If that prediction fails, the {hypothesis} is wrong and I will say so and change it.",
            "A miss here falsifies the model — I will treat any deviation as a signal to revise, not to repeat.",
            "This is a commitment: deviation between prediction and result forces a model change next turn.",
        ])))

    # 8. evidence ledger + model-consistency walk
    for eu in _evidence_units(lex, sem):
        units.append((8, False, eu))
    for cu in _consistency_units(lex, sem):
        units.append((8, False, cu))

    # 9. uncertainty / self-checks / result-protocol / level-close expectations
    if sem["phase"] in ("orient", "probe", "misstep"):
        units.append((9, False, lex.pick([
            "Open questions: which frame elements are stateful, and which action is the state-changer? The cheapest disambiguation comes first.",
            "Uncertainty remains over what is decorative versus mechanical; the batch above is chosen to maximise information per action.",
            "I am deliberately spending few actions per probe — the budget bar punishes brute force.",
            "Two things could still surprise me: an element I have classified as decor turning out stateful, and an action with a delayed effect. Both would show up as an unexplained frame diff, which is exactly what I will scan for.",
            "Probe economics: each action spent here must either confirm or kill a {hypothesis}; batches that cannot change my mind are wasted budget.",
        ])))
    if sem["phase"] in ("revise", "execute"):
        units.append((9, False, lex.pick([
            "Sanity check before committing: every step above was verified against the walls in the current frame, and none enters a lethal or blocked {cell}.",
            "The sequence was re-checked against the current positions — no step relies on stale coordinates.",
            "Cross-check done: path continuity holds (each step moves exactly one {cell} from the previous position).",
            "Before running it I re-validated the batch against the frame: start position matches, every intermediate {cell} is free under the current model, and the endpoint is the intended one.",
            "Commit check: the plan uses only verified mechanics — nothing in it depends on an untested interaction.",
        ])))
    units.append((9, False, lex.pick([
        "Reading protocol for the result: `board_changed` separates real effects from no-ops (modulo the HUD strip), `level` and `score` reveal completions, and `game_over`/`done` are hard stops.",
        "After execution I will parse the result dict: score/level movement means a completion, a False board_changed on a supposed move means I mis-modelled a collision, and any game_over halts everything.",
        "Result-diff plan: compare the reported fields against the per-step predictions above; a single mismatch is enough to stop and re-derive.",
    ])))
    if sem.get("closes_level"):
        units.append((9, False, lex.pick([
            "If the model is right this batch finishes the level; I expect a level-completion flag in the result and will re-orient on the next frame from scratch.",
            "Expecting the level counter to tick after this sequence — if it does not, the model is missing a precondition and I will hunt for it before spending more actions.",
            "This should be the closing batch for level {lv}: the win condition as modelled is satisfied on the final step. A silent result instead would mean a hidden precondition, and finding it becomes the next turn's whole job.",
        ], lv=sem["level"])))
    else:
        units.append((9, False, lex.pick([
            "Stop conditions for this batch: any game_over, any unpredicted frame change, or a level transition — each hands control back to analysis immediately.",
            "Stopping rules: surprise diff, death flag, or an early completion all end the batch; otherwise continue to the planned end.",
        ])))

    # assemble to budget: mandatory first, then optional in listed order
    chosen = [(i, sec, txt) for i, (sec, mand, txt) in enumerate(units) if mand]
    used = sum(_tok(t) + 1 for _i, _s, t in chosen)
    for i, (sec, mand, txt) in enumerate(units):
        if mand:
            continue
        if used >= budget_tokens:
            break
        chosen.append((i, sec, txt))
        used += _tok(txt) + 1
    chosen.sort(key=lambda t: (t[1], t[0]))
    reasoning = "\n".join(t for _i, _s, t in chosen)

    wm = lex.pick(WM_BANK.get(hyp["id"], ["model under construction."]))
    plan = lex.pick(PLAN_BANK.get(sem["phase"], ["proceed carefully."]))
    content = f"World model: {wm}\nPlan: {plan}"
    return reasoning, content, wm, plan
