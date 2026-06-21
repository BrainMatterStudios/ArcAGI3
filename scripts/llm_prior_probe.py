"""LLM-affordance-prior probe (Increment 1 of the serious on-device rich-prior build).

NOT code-synthesis (which failed before). The LLM's job is the one thing the learning-starved
wall games can't provide and the LLM uniquely has from training: GUESS which object/action is
the likely goal-trigger on a novel grid puzzle, using world-knowledge priors (buttons are small
distinctive marks, keys/doors pair, the odd-one-out is interactive...). We give it a STRUCTURED
scene description (object inventory + observed action effects) and ask for a target.

This increment validates the CORE question: does the on-device LLM identify the KNOWN trigger?
(vc33 -> color 9 edge-buttons; we also try ls20/sc25.) If yes, the rich prior is real and worth
wiring into an explorer; if no, on-device rich-prior is confirmed below the bar.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/llm_prior_probe.py <prefix> [model]
"""
from __future__ import annotations
import json, logging, sys
import urllib.request
import numpy as np
from collections import Counter
from dotenv import load_dotenv; load_dotenv()
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from arcagi3 import perception as P

logging.basicConfig(level=logging.ERROR)
OLLAMA = "http://localhost:11434/api/generate"


def describe_scene(grid, bg):
    objs = P.connected_components(grid, background=bg)
    counts = Counter(o.color for o in objs)
    H, W = grid.shape
    lines = []
    by_color = {}
    for o in objs:
        by_color.setdefault(o.color, []).append(o)
    for color, os in sorted(by_color.items()):
        sizes = [o.size for o in os]
        # positions: edge/corner/center
        locs = []
        for o in os:
            r0, c0, r1, c1 = o.bbox
            cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
            edge = r0 <= 2 or c0 <= 2 or r1 >= H - 3 or c1 >= W - 3
            locs.append("edge" if edge else "interior")
        loc = Counter(locs).most_common(1)[0][0]
        lines.append(f"  color {color}: {len(os)} region(s), sizes {min(sizes)}-{max(sizes)} px, "
                     f"mostly {loc}")
    return f"Background is color {bg}. Distinct objects/regions:\n" + "\n".join(lines)


def query_llm(prompt, model):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["response"]


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "vc33"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:14b"
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("p"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["llmprior"]))
    obs = env.reset()
    grid = P.to_grid(obs.frame); bg = P.detect_background(grid)
    avail = list(obs.available_actions or [])
    scene = describe_scene(grid, bg)
    actions_desc = ("Arrows (1=up,2=down,3=left,4=right) move; 5=interact; "
                    "6=click any (x,y) cell.")
    avail_desc = f"Available actions this turn: {avail} (6 means clicking is allowed)."
    prompt = (
        "You are analyzing an unknown 64x64 grid puzzle game (ARC-AGI-3). Each level is solved "
        "by triggering a specific change. You must GUESS, from visual/structural priors, which "
        "element is most likely the interactive trigger to make progress.\n\n"
        f"{scene}\n\n{actions_desc}\n{avail_desc}\n\n"
        "Using general game-design intuition (small distinctive marks are often buttons; rare "
        "colors are often interactive; odd-one-out objects matter; keys pair with doors), answer "
        "STRICTLY as JSON: {\"target_color\": <int>, \"action\": \"click\"|\"move\"|\"interact\", "
        "\"reason\": \"<short>\"}. Pick the single most likely trigger.")
    print(f"--- {prefix} (model={model}) ---")
    print(scene)
    resp = query_llm(prompt, model)
    print("\nLLM response:\n", resp.strip()[:600])


if __name__ == "__main__":
    main()
