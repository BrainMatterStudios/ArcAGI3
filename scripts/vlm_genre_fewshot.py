"""Direction #2 (cheap path) — does FEW-SHOT prompting lift the small embeddable VLM to ~75% genre?

The zero-shot qwen2.5vl:7b scored 57%. Few-shot in-context examples are fully offline-compatible
(embed a handful of labelled exemplars in the submission) and often lift small VLMs a lot. Test:
give the VLM ONE real exemplar frame per genre (6 genres), then classify the OTHER 19 public games.
If this clears ~75%, the perception front-end is viable with a small fast model — no 32B needed.

Uses cached real frames (scripts/.cache/all25_frames.npz); no live API.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src .venv/bin/python scripts/vlm_genre_fewshot.py [model]
"""
from __future__ import annotations

import base64, io, json, re, sys, urllib.request
from pathlib import Path

import numpy as np

from genre_distill_logo import GAME_GENRE, accept_of

OLLAMA = "http://localhost:11434/api/generate"
PAL = np.array([
    [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64], [255, 220, 0], [170, 170, 170],
    [240, 18, 190], [255, 133, 27], [127, 219, 255], [135, 12, 37], [100, 70, 30], [200, 100, 200],
    [60, 180, 180], [180, 180, 60], [120, 120, 255], [255, 255, 255]], dtype=np.uint8)
GENRES = ["NAVIGATE", "COLLECT", "PUSH", "AIM", "MATCH", "SYMMETRY", "CLICK"]
# one clear real exemplar game per genre (COLLECT has no public game -> described only in text)
EXEMPLARS = {"NAVIGATE": "ls20", "AIM": "r11l", "MATCH": "ft09", "CLICK": "tn36",
             "SYMMETRY": "m0r0", "PUSH": "wa30"}


def png_b64(grid, scale=6):
    rgb = PAL[np.clip(grid, 0, 15)]
    big = np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(big).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query(prompt, imgs, model):
    body = json.dumps({"model": model, "prompt": prompt, "images": imgs,
                       "stream": False, "options": {"temperature": 0.0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]


def parse(text):
    m = re.search(r"GENRE:\s*([A-Z]+)", text.upper())
    if m and m.group(1) in GENRES:
        return m.group(1)
    for g in GENRES:
        if g in text.upper():
            return g
    return "?"


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5vl:7b"
    d = np.load(Path(__file__).resolve().parent / ".cache" / "all25_frames.npz")
    X = d["X"]; games = np.array([str(g) for g in d["games"]])
    first = {g: X[np.where(games == g)[0][0]] for g in set(games)}

    # ollama qwen2.5vl is effectively single-image; few-shot-with-images is unavailable, so the
    # only offline lever for the 7B is a chain-of-thought single-image prompt with rich genre cues.
    intro = (
        "This is the start screen of a 64x64 grid puzzle game. Classify its GENRE as ONE label:\n"
        "  NAVIGATE = a small avatar (plus/dot) to move to a target marker inside a large region\n"
        "  COLLECT  = an avatar gathering many scattered small items\n"
        "  PUSH     = push/carry blocks (often a bar) onto bordered marker squares\n"
        "  AIM      = a crosshair or a dotted aim-LINE pointing a cursor at ringed/bordered targets\n"
        "  MATCH    = arrange/align coloured pieces to a shown reference sequence, OR a grid of icon pairs\n"
        "  SYMMETRY = the board split into two halves that should mirror each other\n"
        "  CLICK    = large flat regions or a checkerboard console; NO moving avatar\n"
        "First reason about the key visual structure, then answer. Format EXACTLY:\n"
        "REASON: <one line>\nGENRE: <one label>")

    test_games = sorted(set(games))
    print(f"=== CoT single-image genre probe — model={model} — {len(test_games)} games ===\n", flush=True)
    hits = 0
    for gm in test_games:
        ans = query(intro, [png_b64(first[gm])], model)
        g = parse(ans); ok = g in accept_of(gm); hits += ok
        print(f"  {gm:7}{GAME_GENRE[gm]:10}-> {g:10}{'HIT' if ok else 'miss'}", flush=True)
    print(f"\nCoT genre acc: {hits}/{len(test_games)} = {hits/len(test_games):.0%}", flush=True)
    print("(compare: zero-shot 7B 57%, strong-VLM 83%, gate 75%, always-MATCH baseline 52%)", flush=True)


if __name__ == "__main__":
    main()
