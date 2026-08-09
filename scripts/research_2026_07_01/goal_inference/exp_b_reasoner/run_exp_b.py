"""EXP-B: can a small LOCAL model NAME a game's win condition from a structured
object-list + probe transitions (no source leakage)?

For each game: reset the offline env, apply a handful of probe actions (moves,
ACTION5, salient ACTION6 clicks), build a compact FAITHFUL text description of the
start state + observed deltas, and ask a local ollama model for a predicate-style
win-condition hypothesis. Saves prompt, raw output, latency per (game, model).

Ground truth is written by hand in ground_truth.py and NEVER enters the prompt.
Run: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/research_2026_07_01/goal_inference/exp_b_reasoner/run_exp_b.py
"""
import os, sys, json, time, urllib.request
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from goal_harness import _mk, _apply
from arcagi3 import perception as P

OUT = os.path.dirname(__file__)
MODELS = ["qwen2.5:14b", "qwen2.5-coder:14b", "llama3.1:latest"]
GAMES = ["cd82", "tu93", "sb26"]


def describe_objects(grid, bg, max_objs=40):
    objs = P.connected_components(grid, background=bg)
    objs = sorted(objs, key=lambda o: -o.size)
    lines = []
    for o in objs[:max_objs]:
        r0, c0, r1, c1 = o.bbox
        lines.append(f"  color={o.color} bbox=(r{r0}-{r1},c{c0}-{c1}) size={o.size} "
                     f"shape={o.height}x{o.width}")
    if len(objs) > max_objs:
        lines.append(f"  ...(+{len(objs)-max_objs} more small objects)")
    # color histogram
    vals, counts = np.unique(grid, return_counts=True)
    hist = ", ".join(f"c{int(v)}:{int(n)}" for v, n in zip(vals, counts))
    return hist, objs, "\n".join(lines)


def diff_summary(before, after):
    mask = before != after
    n = int(mask.sum())
    if n == 0:
        return "no change", 0
    rr, cc = np.where(mask)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    # group by (from->to)
    pairs = {}
    for r, c in zip(rr, cc):
        k = (int(before[r, c]), int(after[r, c]))
        pairs[k] = pairs.get(k, 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:6]
    parts = "; ".join(f"{a}->{b}({cnt} cells)" for (a, b), cnt in top)
    return f"{n} cells changed in region (r{r0}-{r1},c{c0}-{c1}); recolorings: {parts}", n


def probe(game):
    env = _mk(game)
    obs = env.reset()
    grid0 = P.to_grid(obs.frame)
    bg = P.detect_background(grid0)
    avail = list(obs.available_actions or [])
    transitions = []

    def step_tok(tok, label):
        nonlocal obs
        prev = P.to_grid(obs.frame)
        obs = _apply(env, tok)
        nxt = P.to_grid(obs.frame)
        summ, n = diff_summary(prev, nxt)
        transitions.append((label, summ, int(obs.levels_completed or 0)))
        return n

    # moves
    for a in (1, 2, 3, 4):
        if a in avail:
            step_tok(("S", a), f"MOVE action{a}")
    # action5 (submit/interact)
    if 5 in avail:
        step_tok(("S", 5), "ACTION5")
    # salient clicks (ACTION6) -- fresh reset so clicks act on the start state
    if 6 in avail or 7 in avail:
        targets = P.salient_click_targets(grid0, background=bg, max_targets=6)
        for (x, y, _p) in targets[:5]:
            obs = env.reset()  # click each on a clean start to isolate its effect
            prev = P.to_grid(obs.frame)
            obs = _apply(env, ("C", x, y))
            nxt = P.to_grid(obs.frame)
            summ, n = diff_summary(prev, nxt)
            transitions.append((f"CLICK ACTION6 at (col={x},row={y})", summ,
                                int(obs.levels_completed or 0)))
    return grid0, bg, avail, transitions


def build_prompt(game, grid0, bg, avail, transitions):
    hist, objs, objlines = describe_objects(grid0, bg)
    tlines = []
    for label, summ, lvl in transitions:
        tlines.append(f"  - {label}: {summ}" + (f"  [LEVEL COMPLETED!]" if lvl >= 1 else ""))
    action_legend = ("Actions: action1-4 = directional moves; action5 = interact/confirm; "
                     "action6 = click at a grid cell (x=col,y=row); action7 = alt interact.")
    prompt = f"""You are analyzing an abstract grid-based puzzle game to infer its WIN CONDITION.
You are given the START state as a list of colored objects, and a set of PROBE
transitions (an action and how the grid changed). You do NOT know the game's rules.

The grid is 64x64. Colors are integers 0-15. Background color = {bg}.
{action_legend}
Available actions this level: {avail}

START STATE color histogram (color:cell_count): {hist}
START STATE objects (largest first; connected same-color regions):
{objlines}

PROBE TRANSITIONS observed (each = one action from the start state and its effect):
{chr(10).join(tlines)}

TASK: Infer the most likely WIN CONDITION -- the specific state change or configuration
that triggers level-completion. Answer ONLY in this format, concise:

WIN_CONDITION: <one-sentence predicate, e.g. "all X objects are moved onto Y tiles" or
"the editable region's colors match the target region">
MECHANIC: <one phrase for how the player acts to achieve it>
CONFIDENCE: <low|medium|high>

Do not explain further. Be specific about WHICH objects/colors/regions."""
    return prompt


def query_ollama(model, prompt, timeout=300):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d.get("response", "").strip(), time.time() - t0


def main():
    results = []
    for game in GAMES:
        grid0, bg, avail, transitions = probe(game)
        prompt = build_prompt(game, grid0, bg, avail, transitions)
        with open(os.path.join(OUT, f"{game}_prompt.txt"), "w") as f:
            f.write(prompt)
        print(f"\n===== {game} =====\nprompt saved ({len(prompt)} chars), "
              f"{len(transitions)} probe transitions", flush=True)
        for model in MODELS:
            try:
                resp, dt = query_ollama(model, prompt)
            except Exception as e:
                resp, dt = f"ERROR: {e}", -1.0
            fn = os.path.join(OUT, f"{game}__{model.replace(':','_').replace('/','_')}_output.txt")
            with open(fn, "w") as f:
                f.write(f"MODEL: {model}\nLATENCY_S: {dt:.1f}\n\n{resp}\n")
            print(f"  [{model}] {dt:.1f}s", flush=True)
            print("   " + resp.replace("\n", "\n   "), flush=True)
            results.append(dict(game=game, model=model, latency_s=round(dt, 1)))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nDONE. artifacts in", OUT)


if __name__ == "__main__":
    main()
