# TRACK D — 24-hour absorption checklist (Oct 1 Milestone-2 releases)

**Written 2026-09-09, before any release exists.** Track D of
`docs/PLAN-2026-09-08-revised-plan-to-7plus.md`. The plan calls this "certain
value": the byte-copy pipeline already turned a competitor release into our live
base in one day (keith V14 -> **3.25**, our current base, sub 56042273).

Track D is now the **only open track**, not a fallback. The plan's §9 rule says
"A1 and A2 both flat on the rig (<= 42) -> fall back to A3/A4 and to D". A1 read
41, A2 closed at zero uptake, and A3/A4 (NOOA, Polyphony) are both gated out
(`docs/research-2026-09-09/GATE-polyphony-clock.md`). So D is where the remaining
score is.

Tooling built for this: **`offkaggle/absorb_kernel.py`** (13 tests, and verified
end-to-end against the live Kaggle API on 09-09: it reproduces
`submission/_keith_copy/ATTEST.json` and the hand-built push metadata exactly).

---

## Before Oct 1 — standing preparation

- [ ] **Keep >= 30 h of the weekly GPU quota unspent in the week of Oct 1.** The
      quota is 60 h/week and is shared with Ahmed's non-ARC work. An absorption
      that cannot get a GPU session is not an absorption.
- [ ] **One Kaggle GPU session at a time.** Never stop a non-ARC (rsna) kernel to
      make room. Ask.
- [ ] Keep the Modal rig warm (`offkaggle/modal_flashnext_serve.py` +
      `offkaggle/run_regime_wave.py`). It is what lets us measure a release in
      Kaggle geometry the same day, without spending a submission slot.
- [ ] Watch for releases. Milestone-2 write-ups usually name the kernel; if not,
      the leaderboard entry links it.

---

## H+0 to H+1 — capture and attest (no GPU, no slot, ~10 min)

**Added 2026-09-14 (rehearsal findings).** (1) Pull the source's own public output FIRST
(`kaggle kernels output <slug> -p <dir>`, then delete `vllm-site-packages/`): a
`vllm-setup-failure.json` means their commit never served (amanatar's "hybrid REPL agent" OOMed at
16 GB KV + MTP and never played) — do NOT spend a commit run on it; the log also gives the cost gate
(levels, calls, tokens) for free. (2) Script kernels (`kernel_type: script`, one `.py`) are supported by
`absorb_kernel.py`; the copy keeps the `.py` suffix and the hash treats it as one cell. (3) Every
TAAF-lineage release keeps its AGENT in the dataset bundle (`src/`, `deploy_target.pkl`,
`benchmark_initial.pkl`) and its SERVING in the same bundle's `serving_setup.py`, launched by
`setup_commands.json`; the notebook is a thin runner. Absorbing a harness therefore means mounting THEIR
bundle; transplanting our serving means running OUR bundle's `serving_setup.py` instead (see the
transplant kit).

```bash
python3 offkaggle/absorb_kernel.py stage <owner>/<kernel> --as arc3-absorb-<name>
```

This pulls the kernel, writes a **byte-identical** copy under
`submission/_absorb_<name>/push/`, and writes `ATTEST.json`.

- [ ] Record `code_cell_sha256`. That is the identity to quote from here on. It
      ignores outputs and execution counts, so it survives a re-pull.
- [ ] **License gate (added 2026-09-12).** Record the license the release ships under
      (kernel page + any GitHub/HF source). Milestone-prize releases must be CC0/MIT-0
      to claim the prize, but a GitHub-only or non-claiming release can ship under
      anything, and OUR top-5 eligibility requires open-sourcing what we fly. A
      non-permissive or missing license = do not stage; ask on the forum first.
- [ ] **Pull immediately.** Version-specific pulls are refused (HTTP 403), so you
      can only ever get the LATEST version. If they push again you lose the bytes
      you were reading. `pulled_utc` is your only version anchor.
- [ ] Diff the serving regime against our current base:

```bash
python3 offkaggle/absorb_kernel.py diff \
  submission/_keith_copy/kernel-metadata.json \
  submission/_absorb_<name>/push/kernel-metadata.json
```

  Exit 0 means same rig, and the difference is **pure harness** — the fast,
  high-confidence path. A non-zero exit lists exactly which of
  `dataset_sources` / `model_sources` / `docker_image` / `machine_shape`
  changed. A changed model or docker digest means you are absorbing a serving
  change too, which is slower and needs the H+1..H+6 step below.

## H+1 to H+6 — read it, then measure it on Modal (no slot)

**Rehearsed 2026-09-14 on the June milestone-winner kernel:** stage → (source output pulled; their 27B run
made 18 levels) → `transplant_serving.py` → commit run 2 h 12 m → 39 levels on Flash-Next under their
harness. If the release's regime differs, prefer the transplant + Kaggle commit run over a Modal
reproduction; the tool's attestation lists exactly what changed (1 pin, 1 cell, 2 substitutions).

- [ ] Read the notebook's code cells. You want three things and only three:
      **(a)** the loop shape (what the agent does per turn), **(b)** the serving
      regime (vLLM flags, KV, MTP, context, concurrency, per-game clock),
      **(c)** anything that would not survive our box.
- [ ] **Cost gate first, before any port enthusiasm.** Compute the release's
      completion tokens per game and compare to ours (~73,600-74,200 per game for true stock V14 waves; the earlier ~83,094 figure was the carry75 arm — corrected 2026-09-12).
      This is the metric that killed Polyphony and it is hardware-independent.
      If their loop needs materially more decode per game than one RTX PRO 6000
      can serve at concurrency 28, it will not reproduce here no matter how good
      it looks.
- [ ] If the regime changed, reproduce it on Modal and run the standard wave:
      `--games all --draws 1 --concurrency 28 --per-game-s 7920`. Read against
      the corrected base (2026-09-12 audit): **38.5 levels, wave sd ~3.5**; one
      wave resolves only ±7 levels, so a copy reading 31-46 needs a second
      counterbalanced wave; two-wave mean >= 45 step candidate, <= 42 dead.
      The Kaggle commit run of the absorbed copy is itself a 25-public-game
      read at the live geometry (used 09-12..09-14: 36-40 levels for the V14
      family) and needs no Modal rig — prefer it when quota allows (~2.3 h).
- [ ] **Pre-register the read before you look at it.** Write the gate into
      `offkaggle/REGIME_WAVE_STATUS.md` first. Every honest verdict this campaign
      produced came from doing this; three of five A2 runs failed on our own
      instrument and only the pre-registration stopped them being reported as
      verdicts on the idea.

## H+6 to H+8 — commit the copy (needs Ahmed's go)

Pushing is shared, irreversible state. **Stop here and ask.** The tool never
pushes; it prints the command.

```bash
python3 -m kaggle kernels push -p submission/_absorb_<name>/push
```

- [ ] Push the copy **unmodified first**. A byte-copy that scores is a fact; a
      copy-plus-graft that scores is two hypotheses tangled together.
- [ ] Every push triggers a run. That commit run is what `kaggle kernels output`
      returns — **not** the scored rerun. There is no channel out of a scored
      rerun. Plan any telemetry you need into the commit run.
- [ ] The rerun's hardware follows the **version's** accelerator. A CPU version
      never scores. Confirm `machine_shape` survived the push.
- [ ] `kaggle kernels output` hangs on kernels with big outputs. Fetch specific
      files, not the whole output.
- [ ] Poll: `kaggle kernels status ahmedmobasher86/arc3-absorb-<name>`.

## H+8 to H+24 — attest the submission and bank it

- [ ] After submitting, record the binding from the authoritative source, not the
      CSV: `GET /api/v1/competitions/submissions/list/arc-prize-2026-arc-agi-3`
      returns `urlNullable` = `/code/<user>/<kernel>?scriptVersionId=<id>`. The
      kaggle CLI's CSV hides both this and `totalBytes`.
- [ ] Add the row to `docs/SUBMISSION-LEDGER.md` and `docs/submission-ledger.json`
      with the `code_cell_sha256`, the scriptVersionId, and the ATTEST.
- [ ] Re-run `absorb_kernel.py verify submission/_absorb_<name>` to prove the
      pushed bytes are still the attested bytes.

---

## After the copy scores — composing our own work on top

- [ ] Re-run the diff after **every** later pull of the same kernel. Authors keep
      pushing; a silent regime change is how an unexplainable score happens.
- [ ] Apply grafts the duck way: rebind methods **in memory only**, never edit the
      stock bytes, so the stock sha stays pinned and attributable
      (`submission/_throughput_v1/graft_carry.py` is the worked example).
- [ ] Then `verify` again: `file_sha256` will change, and that change is the graft.
      Being able to say exactly which hash moved is the point.
- [ ] Three live draws before selection. The two final selections are duplicates
      of the best-mean config, not two different configs.

## What is already known and should not be re-derived

| Fact | Consequence |
|---|---|
| `kernels output` returns the COMMIT run, not the scored rerun | build telemetry into the commit run or lose it |
| version-specific pulls -> HTTP 403 | pull the moment you see a release |
| rerun hardware follows the version's accelerator | a CPU version scores 0 |
| every push triggers a run | pushes cost quota; batch them |
| public-25 does not predict LB | never select on a local wave alone |
| per-game sd ~1.2 levels, live sd ~0.3/draw | +10-30% levers are unmeasurable; only step changes are visible |
| pooled base 39.33 levels sd 2.34 | the rig's null; anything inside is dead |
| ~73,600-74,200 completion tokens per game for stock V14 (~83,094 was the carry75 arm; corrected 2026-09-12) | the decode budget; the cost gate for any port |
