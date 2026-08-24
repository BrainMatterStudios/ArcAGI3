#!/usr/bin/env python3
"""ROUND 2 of the bankruptcy-judge falsifier — prompt-pack + key builder.

Context (task order 2026-08-24 round 2): round-1 stage-3 read FLAG 1/16=6.3%
vs bar 70%, RESCUE 0/11 vs bar 40% -> generative form killed. Post-hoc audit of
stage3_raw.jsonl found the INSTRUMENT was broken: 87/114 samples returned EMPTY
content and most of the rest truncated mid-ruling (max_tokens=2048 was consumed
by reasoning before the verdict line). Round 2 therefore (a) fixes the budget
(max_tokens 8192, finish_reason recorded) and (b) runs four arms:

  V1 MENU DISCRIMINATION (36 cases -> 27 scored): K=4 menu = neutral phrasing of
     the true mechanic class + 3 archetype-plausible distractors from OTHER
     games' classes; deterministic assignment + fixed-seed shuffle. The 9 cases
     on sk48/m0r0 (games UNRESOLVED in the answer key) cannot have a true option
     and are excluded from the metric (documented deviation; denominators
     reported). Controls' correct option = the mechanic class the carried
     (correct) model instantiates, phrased neutrally — style-matched to the
     distractors so nothing marks it.
     METRIC MENU_PICK (majority of valid samples picks the correct letter):
     PRE-REGISTERED pass >= 60%, kill < 40% (chance 25%).

  V2 PREDICTION-FORCING (10 cases): the 9 spiral fires with >=3 recorded
     subsequent actions (2 of the 11 spirals fire at end-of-history) + the
     first eligible deserving stuck case by id. The model must predict, UNDER
     the carried model, the outcome (changed-px band + board zone) of the next
     3 recorded actions; compared mechanically to the record.
     METRIC PRED_DIVERGENCE (majority of valid samples has >=2/3 predictions
     mismatching the record): PRE-REGISTERED pass >= 60%.

  V3 EVIDENCE LEGIBILITY (6 cases): round-1 FLAG-failed deserving cases rerun
     with a RICH digest — before/after ascii frames for the 3 most contradicting
     transitions per the answer-key rationale (key TEXT never enters the prompt;
     only the selection is key-guided, as the task order specifies).
     METRIC FLAG_RICH: >= 3/6 flip to majority-REJECT with a shown-transition
     citation = round-1 failure was evidence format; <= 1/6 = capacity-dead.

  R1X ROUND-1 REPLAY (38 cases, verbatim prompts) at the fixed budget:
     attribution arm. If FLAG_R1X >= 50%, the round-1 kill is reclassified as
     an instrument artifact (truncation), not a capability verdict; it also
     serves as the same-budget tuple-digest baseline for V3.

ANSWER-KEY HYGIENE: this builder reads answer_key.jsonl (it is not a model
context). No key text, no correctness marking, and no 'deserves_rejection'
signal enters any prompt; round2_key.jsonl is consumed only by run_round2.py's
metric step after all completions return.

Run:  .venv/bin/python submission/_bankruptcy_judge/falsifier/round2/build_round2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FALS = os.path.dirname(HERE)
sys.path.insert(0, FALS)
import corpus_lib as cl  # noqa: E402

# ------------------------------------------------------------------ V1 material
# Neutral phrasings of the answer key's mechanic classes: game names, decode
# provenance and campaign jargon stripped; style-matched lengths.
NEUTRAL_CLASS = {
    "sb26": "Selection-then-placement: one click selects a source color or item "
            "from a palette area, a second click places it into a slot; the goal is "
            "to reproduce a target pattern shown elsewhere on the board, and a "
            "separate submit action commits the attempt with no per-slot feedback "
            "when the attempt is wrong.",
    "tu93": "Cover-all with a hidden rotation gate: every object of a given class "
            "must be brought onto designated exit cells, and completion is "
            "additionally gated on a rotation state that is rendered only by a "
            "small indicator sprite on the status area.",
    "ft09": "Neighborhood toggle (lights-out family): clicking an icon flips the "
            "on/off state of that icon plus a fixed mapped neighborhood of other "
            "icons; the goal is to reach a target on/off pattern, and results are "
            "order-independent — only each icon's flip parity matters.",
    "vc33": "Click accumulation: repeated clicks on the same cell are required by "
            "the mechanic; each click accumulates state on that cell rather than "
            "being one-shot, and the win condition is composed from these "
            "accumulated per-cell click effects.",
    "tn36": "Pattern matching by direct edits: the board must be brought to match "
            "a target pattern; clicks edit cells or objects toward that target, "
            "and the level completes when the board matches it.",
    "wa30": "Grab-and-drag transport: the avatar must grab an object and drag it "
            "along a navigable path to a goal location; plain movement without "
            "holding the object accomplishes nothing useful.",
    "dc22": "Cover-with-a-twist: moves paint or cover cells (each move covers a "
            "small number of cells), but plain full coverage is not the real win "
            "condition — an additional non-obvious predicate must also be "
            "satisfied; the action set mixes movement and clicks.",
    "ka59": "Push-and-launch physics: bumping a block launches it as a projectile "
            "that passes through interior walls and stops only at the arena "
            "boundary; interior walls are not barriers for launched blocks.",
}
# Which archetypes each class is a plausible hypothesis for.
CLASS_ARCH = {
    "sb26": {"CLICK", "MIXED"}, "ft09": {"CLICK", "MIXED"},
    "vc33": {"CLICK", "MIXED"}, "tn36": {"CLICK", "MIXED"},
    "wa30": {"AVATAR", "MIXED"}, "dc22": {"MIXED", "AVATAR"},
    "ka59": {"AVATAR", "MIXED"}, "tu93": {"AVATAR", "MIXED"},
}

V1_INSTRUCTION = """\
You are auditing mechanic hypotheses for a multi-level grid-puzzle game on a 64x64 board.
Below is a transition record reconstructed from the environment's ground-truth log: the
last observed transitions as tuples (action, coords[x,y], changed_px, level, level_delta)
plus the recorded no-op events on the current level. After the record you are given four
candidate mechanic hypotheses, labeled A-D. Exactly one of them is the game's true
mechanic; the others belong to different games.

Rank the four candidates by how well they fit the record, best fit first. Then answer
with the best-fit letter and cite one transition from the record that supports it.

Answer in exactly this format (three lines, nothing else after them):
RANKING: <letter> > <letter> > <letter> > <letter>
ANSWER: <letter>
CITE: #<transition number> — <one sentence on how this transition supports the answer>"""

V2_INSTRUCTION = """\
You are stress-testing a working world model for a multi-level grid-puzzle game on a
64x64 board. EXHIBIT A below is the agent's carried world model. EXHIBIT B is a
transition record from the environment's ground-truth log: tuples of (action,
coords[x,y], changed_px, level, level_delta) plus recorded no-op events.

For this exercise, ASSUME EXHIBIT A IS COMPLETELY CORRECT, even if you doubt it.
Under that assumption, predict the observable outcome of each of the three upcoming
actions listed after the exhibits.

For each prediction give:
  band — the number of pixels that will change on the board:
         ZERO (0 pixels), SMALL (1-19), MEDIUM (20-99), LARGE (100 or more)
  zone — where most of the changed pixels will fall, on a 3x3 partition of the board:
         columns x=0-21 left / x=22-42 center / x=43-63 right,
         rows y=0-21 top / y=22-42 middle / y=43-63 bottom;
         NW=top-left N=top-center NE=top-right W=middle-left C=center E=middle-right
         SW=bottom-left S=bottom-center SE=bottom-right; use NONE if band is ZERO.

Answer in exactly this format (three lines, nothing else after them):
P1: band=<ZERO|SMALL|MEDIUM|LARGE> zone=<NW|N|NE|W|C|E|SW|S|SE|NONE>
P2: band=<ZERO|SMALL|MEDIUM|LARGE> zone=<NW|N|NE|W|C|E|SW|S|SE|NONE>
P3: band=<ZERO|SMALL|MEDIUM|LARGE> zone=<NW|N|NE|W|C|E|SW|S|SE|NONE>"""

# V2 case set: pre-registered rule — the 9 spiral fires with >=3 recorded
# subsequent actions, plus the first (by id) deserving stuck case with >=3.
V2_EXCLUDED_SPIRALS = {  # fire at end of recorded history — nothing to predict
    "spiral__packv22__sk48-d8078629-dup__L1__b22",
    "spiral__xd__dc22-fdcac232__L2__b58",
}
V2_FILL_STUCK = ["stuck__packv22__m0r0-492f87ba__L1__b22"]

# V3 case set + per-case transition selectors (key-guided SELECTION; no key text).
# Chosen by hand 2026-08-24 from the 15 round-1 FLAG-failed deserving cases:
# the 6 whose answer-key rationale names contradictions that are visible in
# individual before/after frames.
V3_CASES = [
    "spiral__digest1__ft09-0d8bbf25__L3__b31",     # ~36-37 px flips vs per-cell recolor model
    "spiral__digest1__sb26-7fbdac44__L2__b29",     # 1-px failed ACTION5 submits
    "spiral__depthdiag__tn36-ef4dde99__L2__b31",   # GAME_OVER/RESET tail under cup-catch model
    "spiral__packv22__sk48-d8078629-dup__L1__b22", # no-ops at extension limits + 36/12 oscillation
    "spiral__xd__dc22-fdcac232__L2__b36",          # knob route fully executed, no completion
    "spiral__xpl2__vc33-5430563c__L3__b32",        # clicks move poles (incl. saturation no-op), no level_delta
]


def v3_select(cid: str, tuples: list[dict]) -> list[int]:
    """3 most-contradicting transition indices per the answer-key rationale
    (deterministic; operates on the same last-20 window round 1 showed)."""
    by_i = {t["i"]: t for t in tuples}

    def top_px(pred, k, exclude=()):
        cands = [t for t in tuples if pred(t) and t["i"] not in exclude]
        cands.sort(key=lambda t: (-t["changed_px"], t["i"]))
        return [t["i"] for t in cands[:k]]

    if cid.startswith("spiral__digest1__ft09"):
        sel = top_px(lambda t: t["action"] == "ACTION6", 3)
    elif cid.startswith("spiral__digest1__sb26"):
        subs = sorted([t["i"] for t in tuples
                       if t["action"] == "ACTION5" and t["changed_px"] <= 2])[:3]
        sel = subs + top_px(lambda t: t["action"] == "ACTION5", 3 - len(subs),
                            exclude=set(subs))
    elif cid.startswith("spiral__depthdiag__tn36"):
        sel = [t["i"] for t in tuples[-3:]]  # the GAME_OVER/RESET tail
    elif cid.startswith("spiral__packv22__sk48"):
        fixed = [i for i in (35, 41) if i in by_i]
        sel = fixed + top_px(lambda t: True, 3 - len(fixed), exclude=set(fixed))
    elif cid.startswith("spiral__xd__dc22"):
        sel = [i for i in (31, 38, 45) if i in by_i]
    elif cid.startswith("spiral__xpl2__vc33"):
        fixed = [i for i in (56,) if i in by_i]
        sel = fixed + top_px(lambda t: t["action"] == "ACTION6", 3 - len(fixed),
                             exclude=set(fixed))
    else:
        raise KeyError(cid)
    assert len(sel) == len(set(sel)) == 3, (cid, sel)
    return sorted(sel)


HEX = "0123456789abcdef"


def frame_ascii(grid) -> str:
    g = np.asarray(grid)
    ruler = "    " + "".join(str(c % 10) for c in range(g.shape[1]))
    lines = [ruler]
    for r in range(g.shape[0]):
        lines.append(f"{r:3d} " + "".join(HEX[int(v) & 15] for v in g[r]))
    return "\n".join(lines)


def zones_of(diff_rc) -> list[str]:
    """diff_rc: (rows, cols) arrays of changed cells -> zones containing changes."""
    names = [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]]

    def band(v):
        return 0 if v <= 21 else (1 if v <= 42 else 2)

    zs = {names[band(r)][band(c)] for r, c in zip(*diff_rc)}
    return sorted(zs)


def px_band(n: int) -> str:
    return "ZERO" if n == 0 else "SMALL" if n < 20 else "MEDIUM" if n < 100 else "LARGE"


def rng_for(cid: str) -> random.Random:
    return random.Random(int(hashlib.md5(cid.encode()).hexdigest(), 16))


def main() -> None:
    r1_prompts = [json.loads(l) for l in open(os.path.join(FALS, "judge_prompts.jsonl"))]
    key_rows = [json.loads(l) for l in open(os.path.join(FALS, "answer_key.jsonl"))][1:]
    key = {k["id"]: k for k in key_rows}
    instruction_r1 = open(os.path.join(FALS, "judge_instruction.txt")).read().strip()
    assert len(r1_prompts) == 38 and len(key) == 38

    transcripts = {t.sid: t for t in cl.list_corpus()}

    prompts, keys, dump = [], [], []

    # ---------------- V1 menu discrimination ---------------------------------
    v1_built = v1_skipped = 0
    for p in r1_prompts:
        game, cid = p["game"], p["id"]
        if p["kind"] == "on_completed_level":
            continue  # neither a deserving case nor a control in the task order's 36
        if game not in NEUTRAL_CLASS:  # sk48 / m0r0: no ground-truth class exists
            v1_skipped += 1
            continue
        digest = cl.evidence_digest(p["sid"].split("/")[0], cid.split("__")[2],
                                    p["upto_action"], n=20)
        digest_text = cl.digest_as_text(digest)
        rng = rng_for("V1__" + cid)
        pool = [g for g in sorted(NEUTRAL_CLASS) if g != game
                and p["archetype"] in CLASS_ARCH[g]]
        rest = [g for g in sorted(NEUTRAL_CLASS) if g != game and g not in pool]
        while len(pool) < 3:
            pool.append(rest.pop(0))
        distractors = rng.sample(pool, 3)
        options = [(game, NEUTRAL_CLASS[game])] + [(g, NEUTRAL_CLASS[g]) for g in distractors]
        rng.shuffle(options)
        letters = "ABCD"
        correct_letter = letters[[g for g, _ in options].index(game)]
        menu = "\n\n".join(f"{letters[i]}) {txt}" for i, (_, txt) in enumerate(options))
        content = (f"{V1_INSTRUCTION}\n\nTRANSITION RECORD:\n---\n{digest_text}\n---\n\n"
                   f"CANDIDATE MECHANICS:\n\n{menu}\n\n"
                   f"Deliver your three-line answer now.")
        rid = "V1__" + cid
        prompts.append({"id": rid, "variant": "V1", "src": cid, "kind": p["kind"],
                        "game": game, "archetype": p["archetype"],
                        "messages": [{"role": "user", "content": content}]})
        keys.append({"id": rid, "variant": "V1", "src": cid, "kind": p["kind"],
                     "game": game, "deserves_rejection": key[cid]["deserves_rejection"],
                     "correct_letter": correct_letter,
                     "option_games": [g for g, _ in options]})
        dump.append(f"{'='*100}\nV1 {cid} correct={correct_letter} "
                    f"options={[g for g, _ in options]}\n{content}\n")
        v1_built += 1
    # 27 scored = 11 resolved-game deserving + 16 resolved-game controls; the 10
    # skipped = 9 sk48/m0r0 deserving+controls + the non-deserving m0r0 stuck case.
    assert v1_built == 27 and v1_skipped == 10, (v1_built, v1_skipped)

    # ---------------- V2 prediction-forcing ----------------------------------
    v2_ids = sorted(p["id"] for p in r1_prompts
                    if p["kind"] == "spiral" and p["id"] not in V2_EXCLUDED_SPIRALS)
    v2_ids += V2_FILL_STUCK
    assert len(v2_ids) == 10, v2_ids
    by_id = {p["id"]: p for p in r1_prompts}
    for cid in v2_ids:
        p = by_id[cid]
        run_dir, game_key = p["sid"].split("/")[0], cid.split("__")[2]
        t = transcripts[p["sid"]]
        wm_text = t.blocks[p["block"]].wm_text or "(no carried world model)"
        digest = cl.evidence_digest(run_dir, game_key, p["upto_action"], n=20)
        digest_text = cl.digest_as_text(digest)
        idx, gr = cl.game_run(run_dir, game_key)
        states = cl.load_states(run_dir)[idx]
        hist = gr["history"]
        upto = p["upto_action"]
        assert upto + 3 <= len(hist) and upto + 3 < len(states), cid
        nxt, truth = [], []
        for j in range(upto, upto + 3):
            a = hist[j]["action"]
            coords = None
            if a.get("data") and "x" in a["data"]:
                coords = [a["data"]["x"], a["data"]["y"]]
            c = f" at (x={coords[0]}, y={coords[1]})" if coords else ""
            nxt.append(f"  UPCOMING ACTION {j - upto + 1}: {a['id']}{c}")
            g0, g1 = np.asarray(cl._grid(states[j])), np.asarray(cl._grid(states[j + 1]))
            diff = np.where(g0 != g1)
            changed = int(diff[0].size)
            truth.append({"i": j, "action": a["id"], "coords": coords,
                          "changed_px": changed, "band": px_band(changed),
                          "zones": zones_of(diff) if changed else ["NONE"]})
        content = (f"{V2_INSTRUCTION}\n\nEXHIBIT A — carried working world model:\n"
                   f"---\n{wm_text}\n---\n\n"
                   f"EXHIBIT B — transition record:\n---\n{digest_text}\n---\n\n"
                   f"The next three actions that will be taken are:\n" + "\n".join(nxt) +
                   "\n\nDeliver your three-line prediction now, under the assumption "
                   "that EXHIBIT A is correct.")
        rid = "V2__" + cid
        prompts.append({"id": rid, "variant": "V2", "src": cid, "kind": p["kind"],
                        "game": p["game"], "archetype": p["archetype"],
                        "messages": [{"role": "user", "content": content}]})
        keys.append({"id": rid, "variant": "V2", "src": cid, "kind": p["kind"],
                     "game": p["game"],
                     "deserves_rejection": key[cid]["deserves_rejection"],
                     "truth": truth})
        dump.append(f"{'='*100}\nV2 {cid} truth={json.dumps(truth)}\n{content}\n")

    # ---------------- V3 rich-evidence rerun ---------------------------------
    v3_instruction = instruction_r1.replace(
        "the last observed transitions as tuples (action, coords[x,y], changed_px,\n"
        "              level, level_delta) plus the recorded no-op events on the current level\n"
        "              (actions that changed zero pixels).",
        "summary statistics for the current level plus BEFORE/AFTER board frames\n"
        "              (64x64, one hex digit per cell = the cell's color index) for three\n"
        "              selected recorded transitions, each annotated with its action,\n"
        "              coordinates and changed-pixel count.")
    assert v3_instruction != instruction_r1, "instruction splice failed"
    for cid in V3_CASES:
        p = by_id[cid]
        run_dir, game_key = p["sid"].split("/")[0], cid.split("__")[2]
        t = transcripts[p["sid"]]
        wm_text = t.blocks[p["block"]].wm_text or "(no carried world model)"
        digest = cl.evidence_digest(run_dir, game_key, p["upto_action"], n=20)
        sel = v3_select(cid, digest["tuples"])
        idx, gr = cl.game_run(run_dir, game_key)
        states = cl.load_states(run_dir)[idx]
        hist = gr["history"]
        all_zero_delta = all(tt["level_delta"] == 0 for tt in digest["tuples"])
        parts = [
            f"Current level: {digest['level']}. Actions taken on this level so far: "
            f"{digest['actions_this_level']}. No-op actions on this level (changed zero "
            f"pixels): {digest['noop_count_this_level']}."
            + (" level_delta was 0 on every recorded transition in the observed window "
               "(no level was completed)." if all_zero_delta else ""),
            "Three recorded transitions are shown below in full (BEFORE and AFTER board "
            "frames, 64x64, one hex digit per cell = color index; row numbers left, "
            "column ruler on top).",
        ]
        for j in sorted(sel):
            a = hist[j]["action"]
            coords = None
            if a.get("data") and "x" in a["data"]:
                coords = f"(x={a['data']['x']}, y={a['data']['y']})"
            g0, g1 = np.asarray(cl._grid(states[j])), np.asarray(cl._grid(states[j + 1]))
            diff = np.where(g0 != g1)
            changed = int(diff[0].size)
            cells = list(zip(diff[1].tolist(), diff[0].tolist()))[:40]  # (x,y)
            parts.append(
                f"TRANSITION #{j}: action={a['id']}"
                + (f" coords={coords}" if coords else "")
                + f" changed_px={changed}"
                + (f" changed cells (x,y), first {len(cells)}: {cells}" if changed else
                   " — NO pixels changed (no-op)")
                + f"\nBEFORE (#{j}):\n{frame_ascii(g0)}\nAFTER (#{j}):\n{frame_ascii(g1)}")
        digest_text = "\n\n".join(parts)
        content = (f"{v3_instruction}\n\nEXHIBIT A — carried working world model:\n"
                   f"---\n{wm_text}\n---\n\n"
                   f"EXHIBIT B — evidence digest:\n---\n{digest_text}\n---\n\n"
                   f"Deliver your ruling now, in the exact format specified.")
        rid = "V3__" + cid
        prompts.append({"id": rid, "variant": "V3", "src": cid, "kind": p["kind"],
                        "game": p["game"], "archetype": p["archetype"],
                        "messages": [{"role": "user", "content": content}]})
        keys.append({"id": rid, "variant": "V3", "src": cid, "kind": p["kind"],
                     "game": p["game"], "deserves_rejection": True,
                     "shown_transitions": sorted(sel),
                     "match_keywords": key[cid]["match_keywords"]})
        dump.append(f"{'='*100}\nV3 {cid} shown={sorted(sel)}\n"
                    f"(content {len(content)} chars — frames elided in dump)\n"
                    + content[:2000] + "\n...\n")

    # ---------------- R1X verbatim replay ------------------------------------
    for p in r1_prompts:
        cid = p["id"]
        rid = "R1X__" + cid
        prompts.append({"id": rid, "variant": "R1X", "src": cid, "kind": p["kind"],
                        "game": p["game"], "archetype": p["archetype"],
                        "messages": p["messages"]})
        k = key[cid]
        keys.append({"id": rid, "variant": "R1X", "src": cid, "kind": k["kind"],
                     "game": k["game"], "deserves_rejection": k["deserves_rejection"],
                     "match_keywords": k["match_keywords"]})

    # ---------------- hygiene + write ----------------------------------------
    forbidden = ["deserves_rejection", "answer_key", "correct_letter", "rationale",
                 "ANSWER KEY", "match_keywords"]
    for p in prompts:
        blob = p["messages"][0]["content"]
        for w in forbidden:
            assert w not in blob, (p["id"], w)
    counts = {}
    for p in prompts:
        counts[p["variant"]] = counts.get(p["variant"], 0) + 1
    assert counts == {"V1": 27, "V2": 10, "V3": 6, "R1X": 38}, counts

    with open(os.path.join(HERE, "round2_prompts.jsonl"), "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    with open(os.path.join(HERE, "round2_key.jsonl"), "w") as fh:
        fh.write(json.dumps({
            "_warning": "ROUND-2 ANSWER KEY — must NEVER enter a model context. "
                        "Written before any model saw round2_prompts.jsonl. "
                        "Consumed only by run_round2.py metric computation.",
            "_written": "2026-08-24"}) + "\n")
        for k in keys:
            fh.write(json.dumps(k) + "\n")
    with open(os.path.join(HERE, "review_dump_round2.txt"), "w") as fh:
        fh.write("\n".join(dump))
    total_chars = sum(len(p["messages"][0]["content"]) for p in prompts)
    print(f"cases: {counts} total prompts {len(prompts)} "
          f"({total_chars/1e6:.2f} MB of prompt text)")
    print("wrote round2_prompts.jsonl, round2_key.jsonl, review_dump_round2.txt")


if __name__ == "__main__":
    main()
