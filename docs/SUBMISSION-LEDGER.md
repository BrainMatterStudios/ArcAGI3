# SUBMISSION LEDGER — arc-prize-2026-arc-agi-3

**Built 2026-08-04** (forensic reconstruction ordered by PLAN-REVIEW-2026-08-04: "submission →
bytes → patch-set identity is forensic, not ledgered"). Machine-readable twin:
`docs/submission-ledger.json` (full descriptions, hashes, recovery provenance per row).

**Prime finding of the reconstruction: the submission → kernel-version binding was never
actually lost.** The Kaggle REST endpoint
`GET /api/v1/competitions/submissions/list/arc-prize-2026-arc-agi-3` returns, for every
submission, a `urlNullable` of the form
`/code/<user>/<kernel>?scriptVersionId=<id>` — Kaggle's own record of exactly which kernel
version each submission bound. Every identity below labeled *(Kaggle binding)* comes from
that record, not from inference. The kaggle CLI's CSV hides both this and `totalBytes`.

## Method / provenance

| Fact | Source |
|---|---|
| submission ↔ kernel + scriptVersionId | REST `submissions/list` `urlNullable` (authoritative) |
| totalBytes | same REST rows (`totalBytes`) |
| notebook content identity | `sha256[:16]` over newline-joined **code-cell sources** of the `.ipynb`, from `kaggle kernels pull` — **latest version only**; version-specific pulls are refused (HTTP 403), and `kernels logs <slug>/<n>` silently ignores the version suffix |
| "latest == bound version" | `kaggle kernels list -m` `lastRunTime`: every push triggers a run, so a kernel whose last run is the bound version's (possibly queue-delayed) commit run has had **no later push**. Checked per kernel; all 14 involved kernels pass. |
| version *numbers* | recoverable only where documented (submit commands in docs, submission text, DIAGNOSIS/REVIEW docs). Where not documented they are marked unknown — scriptVersionId is the identity that matters. |
| never-played fingerprint | 3411-byte duck-era gateway parquet (all three 0.00 COMPLETE subs); genuine duck runs 3642–3728 bytes. **Not valid for the June era** — see Surprises #5. |

Content recovery score: **21 of 45 submissions byte-recovered** (hash from actually-pulled
notebook bytes; for 12 of these rows the pulled bytes are identical to the repo
working-tree notebook — duck-base v2 and hybrid-explorer have no exact local twin);
**24 UNRECOVERABLE**
(bound version superseded by later pushes; no API path to old bytes; no logged-in web
session available). UNRECOVERABLE rows carry description + repo-history correlates only,
explicitly labeled — never guessed.

## Master table

Scores: public LB (RHAE %). `nb` = code-cells sha256[:16]; `—` = UNRECOVERABLE. All times UTC.

| sub id | date | score | bytes | kernel @ version | scriptVersionId | nb | arm |
|---|---|---|---|---|---|---|---|
| 55224522 | 08-04 | 0.80 | 3686 | duck-patched @ **v7** (doc) | 339976750 | `ffb8f677cb5027d7` | WMR trio + slow-tick HUD fix + anti-freeze; graph/compact/playbook pinned OFF (byte-verified) |
| 55197631 | 08-03 | **0.00** | **3411** | duck-patched @ **v5** (doc) | 339790398 | — | "WMR on depth-pack v1" — depth claim false; never played (push-race) |
| 55187670 | 08-02 | 0.78 | 3670 | **duck-depth** @ ? | 339708857 | `a59eabbddd5c6a6a` | depth pack D1a-d |
| 55163445 | 08-01 | 0.76 | 3628 | duck-levers @ ? | 339499554 | `7ba37358755f7ad1` | context-starvation arm (est. //4, window 49152) |
| 55160933 | 08-01 | ERROR | 0 | duck-sft @ **v4** (doc) | 339369623 | `70d68cdbccb5c380` | run-8 LoRA merged in-kernel, hard asserts (assert fired) |
| 55122039 | 07-31 | 0.95 | 3690 | duck-sft @ ? | 339165974 | — | claimed adapter — **served BASE** (refuted desc) |
| 55121691 | 07-30 | 0.69 | 3687 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base farm draw #9 |
| 55102674 | 07-30 | ERROR | 0 | duck-sft @ ? | 338999171 | — | merge attempt (ERROR) |
| 55067420 | 07-29 | 1.27 | 3673 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base farm draw #7 |
| 55040465 | 07-28 | 0.88 | 3664 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base farm draw #6 |
| 55013450 | 07-27 | 0.96 | 3680 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base farm draw #5 |
| 54986953 | 07-26 | 0.75 | 3672 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base farm draw #4 |
| 54962014 | 07-25 | 0.82 | 3642 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base farm draw #3 |
| 54938402 | 07-24 | 1.14 | 3704 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | pinned rerun of the 0.92 bytes (drift experiment → drift dead) |
| 54914567 | 07-23 | 0.85 | 3669 | duck-l2 @ ? | 337252267 | `71f59ccb8ff93601` | death-safe memory + image-aware estimator |
| 54897987 | 07-22 | 0.85 | 3728 | duck-tplite @ ? | 337115905 | `481773a354b9565d` | ACTION7 + KV retune 40960 + adaptive budget |
| 54887708 | 07-21 | 0.81 | 3690 | duck-patched @ v1 (inf) | 337022107 | — | ACTION7 round-trip + animation metadata |
| 54847434 | 07-20 | **0.00** | **3411** | ewm-rtx @ ? | 336424468 | `ad4595bf610acd27` | EWM probe (never banked despite 22 h settle) |
| 54818336 | 07-19 | 0.98 | 3666 | duck-shadow @ ? | 336252609 | `dda8ac868c5836d2` | concurrent shadow, max-over-plays |
| 54808852 | 07-18 | 0.92 | 3711 | duck-base @ **v2** | 336252059 | `468aa314e61ead74` | base on RTX Pro 6000 (draw #1) |
| 54807766 | 07-18 | ERROR | 0 | duck-base @ v1 (inf) | 336229761 | — | base isolation test |
| 54806690 | 07-18 | ERROR | 0 | ewm-probe @ ? | 336229137 | `ad4595bf610acd27` | EWM probe (raced push by 4 min) |
| 54802596 | 07-18 | ERROR | 0 | duck-shadow @ ? | 336193335 | — | duck+shadow first attempt |
| 54648872 | 07-13 | 0.83 | 3670 | duck-bestofn @ ? | 334814185 | `b09e302b0cddbfdd` | LEAP-2 replay-banking allocator |
| 54603982 | 07-12 | 0.73 | 3646 | duck-dietcap @ ? | 334520755 | `03b1ad17d9ef4688` | thinking capped 1200 tok |
| 54554985 | 07-11 | **1.26** | 3672 | duck-bestofn @ ? (early) | **332341256** | — | claimed "Fine-tuned 27B RFT LoRA" — **REFUTED, see Surprises #1** |
| 54376711 | 07-06 | 1.10 | 3714 | duck-bestofn @ ? | 332973855 | — | pure-duck sequential best-of-N |
| 54344660 | 07-05 | **0.00** | **3411** | duck-hybrid @ ? | 332696731 | `85bc5384cfd4001d` | best-of-3 hybrid + geodesic (never played) |
| 54312141 | 07-04 | 1.05 | 3674 | duck-hybrid @ ? | 332460722 | — | duck + geodesic post-pass |
| 54280283 | 07-03 | 0.33 | 3848 | hybrid-explorer @ ? | 332231713 | `32ad2820e5dc99f2` | anchor-first revert |
| 54267178 | 07-02 | 0.01 | **3430** | hybrid-explorer @ ? | 332127994 | — | v25 efficiency-first (regressed live; bytes legit, see Surprises #5) |
| 54226502 | 07-01 | 0.33 | 3839 | hybrid-explorer @ ? | 331772154 | — | v24 portfolio + archetype solver |
| 54183663 | 06-30 | 0.33 | 3839 | hybrid-explorer @ ? | 331389774 | — | v23 geodesic double-reset fix |
| 54161711 | 06-29 | 0.33 | 3848 | hybrid-explorer @ ? | 331215339 | — | TransferExplorer DENSE re-submit |
| 53988518 | 06-23 | 0.33 | 3848 | hybrid-explorer @ ? | 329850434 | — | transfer-dense click lattice |
| 53934829 | 06-22 | 0.28 | 3825 | hybrid-explorer @ ? | 329384984 | — | v15 TransferCAI combo |
| 53917868 | 06-21 | 0.33 | 3840 | hybrid-explorer @ ? | 329241130 | — | v13 TransferExplorer |
| 53876618 | 06-20 | 0.33 | 3840 | hybrid-explorer @ ? | **328228290** | — | v12 click-grid (identical bytes to 53823223) |
| 53823223 | 06-19 | 0.33 | 3840 | hybrid-explorer @ ? | **328228290** | — | v12 click-grid (same version resubmitted) |
| 53790346 | 06-18 | 0.33 | 3832 | hybrid-explorer @ ? | 327943912 | — | v11 SalienceExplorer |
| 53757373 | 06-17 | 0.18 | 3831 | hybrid-explorer @ ? | 327723136 | — | v10 multiseed salience |
| 53728818 | 06-16 | 0.33 | 3856 | hybrid-explorer @ ? | 327367468 | — | v6 salience + suspicion + HUD band |
| 53691982 | 06-15 | 0.22 | 3532 | hybrid-explorer @ ? | 327249858 | — | v3 self-contained + fail-safe |
| 53691334 | 06-15 | ERROR | 0 | hybrid-explorer @ ? | 327246609 | — | v2 package-import fix |
| 53690786 | 06-15 | ERROR | 0 | hybrid-explorer @ ? | 327243653 | — | first submission |

(June-era "v.." labels in descriptions are internal build labels, **not** Kaggle version numbers.)

## The identities the review asked to verify — verdicts

- **55224522 = duck-patched v7, 0.80** — VERIFIED. Kaggle binds 339976750 = the kernel's
  latest; pulled bytes hash `ffb8f677cb5027d7`, identical to the repo notebook at HEAD, and
  contain `patch14` / `TAAF_ANTIFREEZE` / `TAAF_PLAYBOOK` with pins
  `TAAF_COMPACT=0, TAAF_GRAPH=0, TAAF_PLAYBOOK=0`. **The review's original `bcef400`→"0.80=v6"
  claim is REFUTED** (its own addendum agrees): the bundle-internal `git_status.txt` tracks
  the upstream duck repo, not ours. v7 has exactly one scored draw: this one.
- **55197631 = duck-patched v5, 0.00, 3411 bytes** — VERIFIED (Kaggle binding 339790398;
  v5 number per DIAGNOSIS-2026-08-03). Never played; description's "on depth-pack v1" was
  false about its own artifact (no depth markers) — the incident that produced
  `scripts/submit_gated.py`.
- **55187670 = 0.78 depth pack — kernel determined: `arc-agi-3-duck-depth`**, scriptVersionId
  339708857, bytes recovered (`a59eabbddd5c6a6a`, = repo `_duck_depth` at db48923).
- **55163445 = 0.76 context arm** — VERIFIED: duck-levers 339499554, bytes recovered,
  estimator `(len+3)//4` + window 49152 + budget assert present in the bytes.
- **55160933 = duck-sft v4, ERROR** — VERIFIED; bytes recovered; the hard serving assert
  design is present (merge-or-raise), i.e. it errored instead of silently scoring base.
- **The 8 base draws (0.92, 1.14, 0.82, 0.75, 0.96, 0.88, 1.27, 0.69)** — CONFIRMED
  identical: every one binds **the same scriptVersionId 336252059** (duck-base v2, the
  documented `-v 2` farm command). The identical-bytes premise of the yardstick
  distribution (n=8 mean 0.929 sd 0.195) holds in Kaggle's own record.

## Surprises found (things the docs did not already say)

1. **The 1.26 (sub 54554985, "Fine-tuned 27B + RFT LoRA") bound a notebook version that
   predates the fine-tune's existence.** Kaggle binds it to duck-bestofn scriptVersionId
   **332341256**, created ~07-03 (IDs are globally monotonic: it sits between 332231713,
   run 07-02 21:44, and 332460722, submitted 07-04) — i.e. **older than the version the
   07-06 "pure duck" 1.10 sub bound (332973855)** and *before the RFT training kernels
   first ran (07-05)*. The same-day retraction ("1.26 = BASE; LoRA never served", memory
   2026-07-11) is therefore corroborated by Kaggle's own binding, independently of the
   `setup_commands.json` analysis. Whatever the auto-fire script pushed on 07-10/11, the
   submission bound an early, pre-fine-tune version.
2. **55122039 (0.95) description refuted** (previously known, now ledgered): duck-sft bytes,
   but the model-path swap was a no-op — a base draw on non-base-notebook bytes. Its bytes
   are UNRECOVERABLE (superseded by v4); classification rests on commit 7bc3cfc's analysis.
3. **The 3411 fingerprint is not only a push-race artifact**: 54847434 (EWM probe) settled
   22 h after its push and still produced the 3411-byte never-played parquet — a run that
   banks nothing produces the same fingerprint as a run that never happened.
4. **June-era identical-bytes pair confirmed**: 53823223 and 53876618 bind the same
   scriptVersionId 328228290 — a true resubmission (0.33 both times, consistent with the
   June agent's determinism, unlike the duck's 0.69–1.27 spread on identical bytes).
5. **The ≤3500-byte never-played rule is duck-era-only**: 54267178 scored 0.01 with 3430
   bytes and 53691982 scored 0.22 with 3532 bytes — both genuinely played June-era runs.
   `submit_gated.py`'s threshold is correct for the current gateway parquet but must not be
   applied to pre-July submissions (or future format changes) without recalibration.
6. **ERROR submissions report totalBytes = 0**, distinct from the 3411 fingerprint; 0.00 +
   3411 = never-played, ERROR + 0 = never scored. All three 0.00-COMPLETE subs are 3411.
7. **Old kernel-version bytes are unrecoverable through the API** (403 on version pulls;
   `kernels logs <slug>/<n>` silently ignores `<n>`). Recovery for 21/45 rows was only
   possible because the bound versions happen to still be their kernels' latest — one more
   push to any of these kernels destroys the evidence. The repo working tree currently
   equals the pulled bytes for duck-patched/depth/levers/sft/l2/tplite/shadow/bestofn/
   dietcap/ewm — those hashes are now pinned here and in the JSON.

## UNRECOVERABLE accounting (24 rows)

- duck-patched v5 (55197631) and v1-era 337022107 (54887708) — superseded by later pushes.
- duck-sft 338999171, 339165974 (55102674, 55122039) — superseded by v4.
- duck-base v1 336229761 (54807766) — plausible correlate: repo `_duck_base/duck-base.ipynb`
  (`886dbc8a…`) differs from pulled v2 by exactly the gpu-probe cell; **unproven**, so
  marked unrecovered.
- duck-shadow 336193335 (54802596), duck-bestofn 332341256 + 332973855 (54554985, 54376711),
  duck-hybrid 332460722 (54312141) — superseded.
- 15 June-era hybrid-explorer submissions (14 unique versions) — only the final version
  (332231713) is still latest and recovered.

## Going-forward rule (proposal — NOT yet applied)

`scripts/submit_gated.py` already queries the submissions API in gate 3 and sees `ref`,
`totalBytes`, and (unused today) `urlNullable`. It should append a ledger row the moment it
first identifies the submission, so identity is recorded at birth, never reconstructed.
Proposed patch (follow-up work; do not let a ledger failure kill the watch):

```diff
--- a/scripts/submit_gated.py
+++ b/scripts/submit_gated.py
@@ -52,6 +52,8 @@ from pathlib import Path
 DEFAULT_COMPETITION = "arc-prize-2026-arc-agi-3"
 
+LEDGER_PATH = Path(__file__).resolve().parent.parent / "docs/submission-ledger.json"
+
 # The never-played fingerprint. Three independent 0.00 submissions (55197631,
@@ -186,6 +188,36 @@ def find_our_submission(competition: str, message: str, not_before: datetime) ->
     return max(candidates, key=lambda pair: pair[0])[1]
 
 
+def append_ledger_row(row: dict, args, notebook_hash: str) -> None:
+    """Record submission -> bytes -> kernel-version identity at birth (best-effort)."""
+    try:
+        url = str(row.get("urlNullable") or "")
+        entry = {
+            "submission_id": row.get("ref"),
+            "date_utc": row.get("date"),
+            "status": str(row.get("status")),
+            "public_score": None,
+            "total_bytes": row.get("totalBytes"),
+            "kernel_slug": url.split("?")[0].split("/code/")[-1] or args.kernel,
+            "script_version_id": (int(url.split("scriptVersionId=")[-1])
+                                  if "scriptVersionId=" in url else None),
+            "kernel_version_number": {"n": args.version, "basis": "submit_gated --version"},
+            "notebook_identity": {"code_cells_sha256_16": notebook_hash,
+                                  "recovery": "recorded at submit time from --notebook"},
+            "description": args.message,
+            "notes": "auto-appended by submit_gated.py",
+        }
+        data = json.loads(LEDGER_PATH.read_text())
+        if not any(r.get("submission_id") == entry["submission_id"]
+                   for r in data["submissions"]):
+            data["submissions"].insert(0, entry)
+            LEDGER_PATH.write_text(json.dumps(data, indent=1) + "\n")
+            log(f"ledger row appended for submission {entry['submission_id']}")
+    except Exception as exc:  # noqa: BLE001 — the ledger must never kill the watch
+        alarm(f"LEDGER APPEND FAILED ({exc!r}) — add the row by hand")
+
+
 def watch_submission(competition: str, message: str, submitted_at: datetime) -> int:
@@ -196,6 +228,7 @@ def watch_submission(competition: str, message: str, submitted_at: datetime) ->
         if row is None:
             log("submission not visible in the list yet")
         else:
+            append_ledger_row(row, watch_submission.args, watch_submission.notebook_hash)
             ref = row.get("ref")
```

(with `watch_submission.args`/`.notebook_hash` set in `main()` after gate 1, where the
notebook source is already parsed — hash it with the same
`sha256(code_cells)[:16]` recipe as this ledger, and pass `args` through. Public score is
back-filled later by re-running the ledger builder or by hand when the score lands.)

Also worth adopting: **never push a new version to a kernel whose bound bytes are not yet
ledgered** — a push permanently destroys the only recovery path for the previous version's
bytes (Surprise #7).

## Honesty note

Nothing in this ledger is guessed. Every kernel/version binding is Kaggle's own record;
every content hash is from actually-pulled bytes; version *numbers* are only stated where a
document or submission text pins them (basis given per row in the JSON); everything else is
marked unknown or UNRECOVERABLE. The point of this file is that nobody ever again argues
from a misattributed draw.
