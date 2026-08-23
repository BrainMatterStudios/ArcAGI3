#!/usr/bin/env python3
"""Stage 2 — build the judge prompt pack + the answer key (written BEFORE any
model ever sees the prompts).

Inputs : stage1_results.json (the tuned trigger's fire points), the corpus, and
         judge_instruction.txt (frozen; verbatim from the §B design law).
Outputs: judge_prompts.jsonl   — one case per line: the exact fresh-context judge
                                 input (single user message: instruction + carried
                                 world model + evidence digest). NO ground truth.
         answer_key.jsonl      — the pre-registered key: true mechanic class,
                                 deserves_rejection, rationale, match keywords for
                                 the automatic RESCUE metric. MUST NEVER enter a
                                 judge context (first record carries the warning).
         review_dump.txt       — human-readable dump of every case (world model +
                                 digest) used to adjudicate the key by hand.

Cases: all Stage-1 fires (spiral / stuck / on-completed) + 20 healthy controls
sampled deterministically from eventually-completed level segments (block at
~70% of the segment, carried model present — a model that led to a completed
level shortly after is presumptively sound).

Run:  .venv/bin/python stage2_build.py
"""
from __future__ import annotations

import json
import os

import corpus_lib as cl

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ ground truth
# Sources: memory arcagi3-sb26-cracked, arcagi3-goal-inference-stage1 (dev-source
# win predicates), arcagi3-all-dev-games-cracked, human fixtures
# (scratchpad/testing_20260822/human_fixtures/), design doc §A specialist tier.
GROUND_TRUTH = {
    "sb26": {
        "mechanic_class": "select-then-place pattern sorting (the design's 'select-then-swap'): "
                          "click a palette/source color to SELECT it, click a slot to PLACE it; "
                          "reproduce the target pattern shown in the top panel; ACTION5 submits; "
                          "a wrong submit gives no per-slot feedback. L1+ uses a non-obvious "
                          "two-box slot<->target correspondence.",
        "keywords": ["select", "palette", "slot", "submit", "pattern", "swap", "place"],
    },
    "tu93": {
        "mechanic_class": "COVER-ALL with a rotation-gated goal (dev-source predicate): every "
                          "class-X object must sit on the exit cells, and the win is gated on a "
                          "rotation state that renders only on a HUD indicator sprite.",
        "keywords": ["cover", "exit", "rotation", "rotate", "gate", "all objects"],
    },
    "ft09": {
        "mechanic_class": "GF(2) lights-out on an icon-neighborhood map: clicking an icon toggles "
                          "itself plus a fixed mapped neighborhood of icons; reach the target "
                          "on/off pattern (linear over GF(2), order-independent, each icon's "
                          "parity is what matters).",
        "keywords": ["toggle", "lights-out", "neighborhood", "parity", "xor", "flip"],
    },
    "vc33": {
        "mechanic_class": "click-composition (dev-source predicate class): the win predicate "
                          "composes click effects; repeated same-cell clicks are mechanic-required "
                          "(human replays: 50-61% same-cell click runs), i.e. clicks accumulate "
                          "state on a cell rather than being one-shot.",
        "keywords": ["repeat", "same cell", "accumulate", "composition", "multiple clicks", "again"],
    },
    "tn36": {
        "mechanic_class": "pattern-goal (specialist-tier decode): the board must be brought to "
                          "match a target pattern; clicks edit cells/objects toward the target.",
        "keywords": ["pattern", "match", "target", "copy"],
    },
    "wa30": {
        "mechanic_class": "grab-and-drag transport (specialist-tier decode: wa30 grab-drag A*): "
                          "the avatar must grab an object and drag it along a navigable path to "
                          "a goal location; plain movement without the grab does nothing useful.",
        "keywords": ["grab", "drag", "carry", "push", "transport"],
    },
    "dc22": {
        "mechanic_class": "cover-class with a non-plain-cover twist (dev-source: induction finds "
                          "COVER — a move covers +4 cells — but the real win predicate is NOT "
                          "plain cover; mixed movement+click action set).",
        "keywords": ["cover", "fill", "paint", "trail"],
    },
    "ka59": {
        "mechanic_class": "block push-AND-LAUNCH physics: bumping a block launches it as a "
                          "projectile that ignores the inner wall and stops only at the arena "
                          "boundary (the wall is not a barrier for launched blocks).",
        "keywords": ["launch", "projectile", "push", "momentum", "slide"],
    },
    "m0r0": {
        "mechanic_class": "UNRESOLVED (no fixture/dev-source decode in the campaign record); "
                          "movement-family game. Key is adjudicated on evidence-consistency only.",
        "keywords": [],
    },
    "sk48": {
        "mechanic_class": "UNRESOLVED (zero-game, never cracked; movement+click action set). "
                          "Key is adjudicated on evidence-consistency only.",
        "keywords": [],
    },
}

# ------------------------------------------------------------- hand adjudication
# deserves_rejection per fire case, decided 2026-08-24 by reading review_dump.txt
# (carried model vs ground truth + digest) BEFORE any judge model saw the prompts.
ADJUDICATION: dict[str, tuple[bool, str]] = {
    "spiral__depthdiag__tn36-ef4dde99__L2__b31": (True,
        "cup/ball-catch goal model self-described as unconfirmed after 62 actions on the level; "
        "no level_delta ever; run hit GAME_OVER at ~88 actions under this model; RESET visible at #87."),
    "on_completed_level__depthdiag__wa30-ee6fef47__L2__b28": (False,
        "grab-core->route->drop model matches wa30 grab-drag ground truth and the level was "
        "completed shortly after this point — the trigger's one false fire; correct verdict is KEEP."),
    "spiral__depthdiag__wa30-ee6fef47__L3__b49": (True,
        "mechanic FAMILY (carry cores into the rectangle) is correct, but the operative submodel "
        "(cross the wall via the c-box doors) produced 42 actions with zero progress to session "
        "end; rejection deserved on the routing/gate hypothesis — a good rebuild keeps grab-drag."),
    "spiral__digest1__ft09-0d8bbf25__L3__b31": (True,
        "model treats clicks as single-cell recolors ('toggle all 17 W cells, 18 clicks') but every "
        "observed click flips ~36-37 px — the GF(2) neighborhood-toggle ground truth directly "
        "contradicts the carried per-cell model; 43 actions, no progress."),
    "spiral__digest1__sb26-7fbdac44__L2__b29": (True,
        "carried model degenerated to 'Plan: press SPACE to confirm' with no mechanic content; "
        "three ACTION5 submits each changed 1 px (failed submit) and the level never completed — "
        "the L2 slot<->target mapping hypothesis is bankrupt (ground truth: non-obvious two-box "
        "correspondence, no per-slot feedback)."),
    "stuck__packv22__m0r0-492f87ba__L1__b22": (True,
        "no goal model after 27 actions ('What is the goal?'); internally contradictory motion "
        "claims (mirror motion vs both-left); the harness's own goal-evidence block says nothing "
        "has scored; rebuild call warranted."),
    "stuck__packv22__sk48-d8078629__L1__b20": (True,
        "'bring all three colors to the left wall' goal never produced any progress in 47 actions "
        "aimed at 4 targets (11.8x each); push/pull plan repeats without a scoring event."),
    "stuck__packv22__sk48-d8078629__L1__b36": (True,
        "same carried model at 119 actions with 29 no-ops — the model's own push-right plan "
        "(ACTION3) is now mostly a no-op (#100-#118) yet the model is unchanged; clear bankruptcy."),
    "spiral__packv22__sk48-d8078629-dup__L1__b22": (True,
        "'bridge extension persists across moves' contradicted by no-ops at extension limits "
        "(#35, #41 ACTION3/4 changed_px=0) and oscillating 36/12 px cycles; 42 actions, "
        "no progress, collection never observed."),
    "stuck__packv22__tn36-ef4dde99__L1__b20": (True,
        "checkerboard-encoding guess with brute-force toggle patterns; 49 actions, no progress on "
        "a level other runs complete quickly; the specific bars/stems encoding was never validated."),
    "stuck__packv22__tn36-ef4dde99__L1__b39": (True,
        "the carried model itself concedes 'the approach (brute-forcing 10-bit patterns) was "
        "wrong' yet continues pattern-guessing; 95 actions, one game-over already burned."),
    "spiral__xd__dc22-fdcac232__L2__b36": (True,
        "the plan's 15-action knob route is visible fully executed in the digest (#31-#45, 8-9 px "
        "moves) without the predicted completion; table-toggle goal model unvalidated and frozen."),
    "spiral__xd__dc22-fdcac232__L2__b58": (True,
        "same verbatim model 22+ blocks later: 51 actions on the level, repeated (52,22) 129-px "
        "toggles and a no-op click (#72), zero progress to session end."),
    "spiral__xpl2__sk48-d8078629__L2__b40": (True,
        "beam-grab-and-pull model never observed grabbing anything; 33 actions of 36/12-px "
        "oscillation plus unexplained 240-337 px ACTION1 transitions (#63-#66) the model does not "
        "account for; no progress."),
    "spiral__xpl2__vc33-5430563c__L3__b32": (True,
        "goal model 'align paired colored markers' executed as planned (repeated same-cell clicks "
        "DID move the poles, incl. saturation no-op #56) but 50 actions produced no level_delta — "
        "the goal predicate, not the click mechanics, is bankrupt (ground truth: click-composition "
        "predicate, alignment guess unconfirmed)."),
    "stuck__xpl4__m0r0-492f87ba__L2__b42": (False,
        "plan-only model 25 actions into a fresh L2 with movement visibly working (64-66 px "
        "changes) and exploration ongoing — evidence at fire time does not yet convict the model; "
        "a fair judge KEEPs (the trigger fired via the stall branch on a thin but young model)."),
    "spiral__xpl5__dc22-fdcac232__L2__b33": (True,
        "model asserts the run is over ('no further actions are possible') while the digest shows "
        "actions continuing after RESET #135 — directly contradicted; 113 actions, exhaustive "
        "64-state sweep already failed under this goal model."),
    "spiral__xpl5__dc22-fdcac232__L2__b55": (True,
        "same dead model at 144 actions; repeated (52,40)/(52,22) toggle cycling with 5 no-ops "
        "and zero progress to session end."),
}


def control_points(corpus, n=20):
    """Deterministic healthy controls: completed-level segments, block at ~70%."""
    cands = []
    for t, arch in corpus:
        levels = sorted({b.level for b in t.blocks if b.level is not None})
        for L in levels:
            if not t.level_eventually_completed(L):
                continue
            seg = [b for b in t.blocks if b.level == L]
            if len(seg) < 4:
                continue
            pick = seg[int(len(seg) * 0.7)]
            if pick.wm_hash == "-":
                withwm = [b for b in seg[1:] if b.wm_hash != "-"]
                if not withwm:
                    continue
                pick = withwm[len(withwm) * 7 // 10]
            cands.append((t, arch, L, pick))
    # spread: at most 2 per game, at most 1 per (sid, level); favor deeper levels
    cands.sort(key=lambda c: (-c[2], c[0].sid))
    picked, per_game = [], {}
    for (t, arch, L, b) in cands:
        g = t.game_key.split("-")[0]
        if per_game.get(g, 0) >= 3:
            continue
        picked.append((t, arch, L, b))
        per_game[g] = per_game.get(g, 0) + 1
        if len(picked) >= n:
            break
    return picked


def build_prompt(instruction, wm_text, digest_text):
    return (
        f"{instruction}\n\n"
        f"EXHIBIT A — carried working world model:\n"
        f"---\n{wm_text}\n---\n\n"
        f"EXHIBIT B — evidence digest:\n"
        f"---\n{digest_text}\n---\n\n"
        f"Deliver your ruling now, in the exact format specified."
    )


def main():
    instruction = open(os.path.join(HERE, "judge_instruction.txt")).read().strip()
    stage1 = json.load(open(os.path.join(HERE, "stage1_results.json")))
    fires = stage1["selected"]["all_fires"]
    label_keys = {(r["sid"], r["level"]) for r in stage1["labels"]}

    corpus = [(t, cl.archetype(t.run_dir, t.game_key)) for t in cl.list_corpus()]
    by_sid = {t.sid: t for t, _ in corpus}

    cases = []
    for f in sorted(fires, key=lambda f: (f["sid"], f["block"])):
        t = by_sid[f["sid"]]
        b = t.blocks[f["block"]]
        kind = ("spiral" if (f["sid"], f["level"]) in label_keys
                else "on_completed_level" if f["eventually_completed"] else "stuck")
        cases.append(dict(kind=kind, t=t, block=b, level=f["level"], arch=f["arch"]))
    for (t, arch, L, b) in control_points(corpus):
        cases.append(dict(kind="control", t=t, block=b, level=L, arch=arch))

    prompts, keys, dump = [], [], []
    for c in cases:
        t, b = c["t"], c["block"]
        game = t.game_key.split("-")[0]
        cid = f"{c['kind']}__{t.run_dir}__{t.game_key}__L{c['level']}__b{b.idx}"
        digest = cl.evidence_digest(t.run_dir, t.game_key, b.cum_actions, n=20)
        digest_text = cl.digest_as_text(digest)
        wm_text = b.wm_text or "(no carried world model)"
        prompts.append({
            "id": cid,
            "kind": c["kind"],
            "game": game,
            "archetype": c["arch"],
            "sid": t.sid,
            "level": c["level"],
            "block": b.idx,
            "upto_action": b.cum_actions,
            "messages": [{"role": "user",
                          "content": build_prompt(instruction, wm_text, digest_text)}],
        })
        gt = GROUND_TRUTH.get(game, {"mechanic_class": "UNKNOWN", "keywords": []})
        if cid in ADJUDICATION:
            deserves, rationale = ADJUDICATION[cid]
        elif c["kind"] == "control":
            deserves, rationale = False, ("level was completed shortly after this point with "
                                          "this carried model — presumptively sound")
        else:
            raise RuntimeError(
                f"fire case {cid} has no hand adjudication — re-read review_dump.txt and "
                f"extend ADJUDICATION before regenerating the answer key")
        keys.append({
            "id": cid,
            "kind": c["kind"],
            "game": game,
            "sid": t.sid,
            "level": c["level"],
            "mechanic_class": gt["mechanic_class"],
            "match_keywords": gt["keywords"],
            "deserves_rejection": deserves,
            "rationale": rationale,
        })
        dump.append(
            f"{'='*100}\nCASE {cid}  kind={c['kind']} arch={c['arch']} "
            f"actions_this_level={digest['actions_this_level']} "
            f"noops={digest['noop_count_this_level']}\n"
            f"--- carried world model ---\n{wm_text}\n"
            f"--- digest ---\n{digest_text}\n"
        )

    with open(os.path.join(HERE, "judge_prompts.jsonl"), "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    with open(os.path.join(HERE, "answer_key.jsonl"), "w") as fh:
        fh.write(json.dumps({
            "_warning": "ANSWER KEY — must NEVER enter a judge context or any prompt. "
                        "Written before any model saw judge_prompts.jsonl. "
                        "Consumed only by run_stage3.py metric computation.",
            "_written": "2026-08-24",
        }) + "\n")
        for k in keys:
            fh.write(json.dumps(k) + "\n")
    with open(os.path.join(HERE, "review_dump.txt"), "w") as fh:
        fh.write("\n".join(dump))

    n = {}
    for p in prompts:
        n[p["kind"]] = n.get(p["kind"], 0) + 1
    print(f"cases: {len(prompts)} {n}")
    print(f"adjudicated by hand: {len(ADJUDICATION)}")
    print("wrote judge_prompts.jsonl, answer_key.jsonl, review_dump.txt")


if __name__ == "__main__":
    main()
