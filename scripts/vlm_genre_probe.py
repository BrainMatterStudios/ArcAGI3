"""Track-B probe (Option 1) — can a SMALL embeddable VLM read the GENRE (front-end task)?

The earlier VLM probe (vlm_prior_probe.py) asked for the exact interactive trigger — the
"terminal" task that even a strong VLM can't do from a static frame. This tests the FRONT-END
task instead: classify the game's GENRE (closed set) + spot the player-controlled agent. That is
the layer where Exp 51 showed 5/8 holdouts die, and the layer a strong VLM (Claude) read 5/6
correctly. If the small on-device qwen2.5vl:7b — the model that fits the eval T4 — can also do it,
Option 1-offline is viable now; if not, it's the fine-tuning target for Track A's synthetic games.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/vlm_genre_probe.py [model] [games...]
"""
from __future__ import annotations
import base64, io, json, logging, re, sys, urllib.request
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from arc_agi import Arcade, OperationMode
from arcagi3 import perception as P

logging.basicConfig(level=logging.ERROR)
OLLAMA = "http://localhost:11434/api/generate"
PAL = np.array([
    [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64], [255, 220, 0],
    [170, 170, 170], [240, 18, 190], [255, 133, 27], [127, 219, 255], [135, 12, 37],
    [100, 70, 30], [200, 100, 200], [60, 180, 180], [180, 180, 60], [120, 120, 255],
    [255, 255, 255]], dtype=np.uint8)

GENRES = ["NAVIGATE", "COLLECT", "PUSH", "AIM", "MATCH", "SYMMETRY", "CLICK"]
# ground-truth genre per game (sets allow visually-defensible alternatives), from Exp 48/49/51 + frames
TRUTH = {
    "ls20": {"NAVIGATE"}, "collect": {"COLLECT"}, "su15": {"AIM"}, "m0r0": {"SYMMETRY"},
    "sk48": {"MATCH"}, "wa30": {"PUSH"}, "re86": {"AIM", "MATCH"}, "tr87": {"MATCH"},
}
PROMPT = (
    "You are looking at the start screen of a 64x64 grid puzzle game (each cell a colored square).\n"
    "Classify the GAME GENRE as exactly ONE of these labels:\n"
    "  NAVIGATE = move a small avatar to a target location/marker\n"
    "  COLLECT  = move an avatar to gather/touch all scattered items\n"
    "  PUSH     = push/carry blocks onto marked target squares (sokoban)\n"
    "  AIM      = aim a crosshair/cursor at targets\n"
    "  MATCH    = arrange/align pieces to match a shown pattern or color sequence\n"
    "  SYMMETRY = make the two halves of the board mirror each other\n"
    "  CLICK    = click a special element; no moving avatar\n"
    "Reply with EXACTLY this format:\n"
    "GENRE: <one label>\n"
    "AGENT: <describe the player-controlled object and its location, or 'none visible'>\n"
    "GOAL: <one short sentence>"
)


def render_png(grid, scale=8):
    rgb = PAL[np.clip(grid, 0, 15)]
    big = np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(big).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_vlm(prompt, img_b64, model):
    body = json.dumps({"model": model, "prompt": prompt, "images": [img_b64],
                       "stream": False, "options": {"temperature": 0.0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def parse_genre(text):
    m = re.search(r"GENRE:\s*([A-Z]+)", text.upper())
    if m and m.group(1) in GENRES:
        return m.group(1)
    for g in GENRES:  # fallback: first genre keyword mentioned
        if g in text.upper():
            return g
    return "?"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5vl:7b"
    games = sys.argv[2:] if len(sys.argv) > 2 else list(TRUTH)
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("p"))
    envs = {e.game_id: e for e in client.get_environments()}
    print(f"=== VLM genre probe — model={model} ===\n", flush=True)
    hits = 0; total = 0
    rows = []
    for prefix in games:
        gid = next((g for g in envs if g.startswith(prefix)), None)
        if gid is None:
            print(f"[{prefix}] not a live game (bundled/dev) — skipped", flush=True)
            continue
        env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["vlm-genre"]))
        grid = P.to_grid(env.reset().frame)
        ans = query_vlm(PROMPT, render_png(grid), model)
        g = parse_genre(ans)
        ok = g in TRUTH[prefix]
        hits += ok; total += 1
        rows.append((prefix, g, "/".join(sorted(TRUTH[prefix])), ok))
        agent_line = next((l for l in ans.splitlines() if l.upper().startswith("AGENT")), "")
        print(f"[{prefix}] VLM={g:9} truth={'/'.join(sorted(TRUTH[prefix])):14} "
              f"{'HIT' if ok else 'miss'}   {agent_line[:80]}", flush=True)
    print("\n" + "-" * 70)
    print(f"{'game':9}{'VLM':10}{'truth':16}result")
    for prefix, g, t, ok in rows:
        print(f"{prefix:9}{g:10}{t:16}{'HIT' if ok else 'miss'}")
    print("-" * 70)
    print(f"GENRE top-1 accuracy: {hits}/{total} = {hits/total:.0%}  (model={model})")


if __name__ == "__main__":
    main()
