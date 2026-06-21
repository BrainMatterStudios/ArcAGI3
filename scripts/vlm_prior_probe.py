"""VLM-affordance-prior probe — the serious visual rich-prior (Increment 1b).

The text-LLM picked vc33's trigger wrong (color 11, a no-op) because "which small rare object
is the button" is VISUAL information a text description loses. The serious version is a VISION
model that SEES the grid and uses visual affordance priors. We render the grid to an image and
ask an on-device VLM which colored element is the likely interactive trigger.

Usage: ARCAGI3_ALLOW_CPU=1 PYTHONPATH=src python scripts/vlm_prior_probe.py <prefix> [model]
"""
from __future__ import annotations
import base64, io, json, logging, sys
import urllib.request
import numpy as np
from dotenv import load_dotenv; load_dotenv()
from arc_agi import Arcade, OperationMode
from arcagi3 import perception as P

logging.basicConfig(level=logging.ERROR)
OLLAMA = "http://localhost:11434/api/generate"
# ARC-ish 16-color palette (approx)
PAL = np.array([
    [0, 0, 0], [0, 116, 217], [255, 65, 54], [46, 204, 64], [255, 220, 0],
    [170, 170, 170], [240, 18, 190], [255, 133, 27], [127, 219, 255], [135, 12, 37],
    [100, 70, 30], [200, 100, 200], [60, 180, 180], [180, 180, 60], [120, 120, 255],
    [255, 255, 255]], dtype=np.uint8)


def render_png(grid, scale=8):
    h, w = grid.shape
    rgb = PAL[np.clip(grid, 0, 15)]
    big = np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8))
    try:
        from PIL import Image
    except Exception:
        return None
    buf = io.BytesIO(); Image.fromarray(big).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def query_vlm(prompt, img_b64, model):
    body = json.dumps({"model": model, "prompt": prompt, "images": [img_b64],
                       "stream": False, "options": {"temperature": 0.2}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["response"]


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "vc33"
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma3:latest"
    client = Arcade(operation_mode=OperationMode.NORMAL, logger=logging.getLogger("p"))
    gid = next(e.game_id for e in client.get_environments() if e.game_id.startswith(prefix))
    env = client.make(game_id=gid, scorecard_id=client.open_scorecard(tags=["vlm"]))
    obs = env.reset()
    grid = P.to_grid(obs.frame)
    img = render_png(grid)
    if img is None:
        print("PIL not available; cannot render"); return
    prompt = (
        "This is a 64x64 grid puzzle (each cell a colored square). It is solved by clicking the "
        "ONE special interactive element (like a button/switch) that triggers a big change. "
        "Looking at the image, describe where the most likely interactive trigger is (its color "
        "and rough location), using visual intuition (small distinctive marks, odd-one-out, "
        "things that look like buttons). Answer with the color and location.")
    print(f"--- {prefix} (VLM={model}) --- (truth: vc33 trigger = color 9, dark-red edge marks)")
    print(query_vlm(prompt, img, model).strip()[:700])


if __name__ == "__main__":
    main()
