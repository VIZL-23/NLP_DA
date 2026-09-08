# Phase 7 — Evaluation & Ablations

**Status:** Scripts complete, smoke-tested, and (unusually for this project so far) actually run end to end at toy scale — a real bug was found and fixed in the process. Real-scale numbers still blocked on Phase 6 (needs a GPU).
**Goal:** mAP, zero-shot, FPS + the 7 ablations.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| A way to run the seven planned ablations, three of them flagged priority | Done — `scripts/run_ablations.py`, and actually executed (§3a) |
| Negative control re-check (Phase 3's `eval_yoloworld.py` bug, re-verified against a trained TG-FEM checkpoint rather than assumed fixed) | Done — `scripts/eval_tgfem.py` |
| A compact tgfem vs tgfem_identity vs cbam_worlddetect comparison table, including the falsifiable per-class prediction from `PHASE-3.md` §3 | Done, plus a speed/FPS column (was carried forward from an earlier pass, now added) |
| DeepCrack OOD zero-shot evaluation | Done — `scripts/eval_deepcrack.py` (§5). This did not exist at all until this pass - see README's original gap list |
| Zero-shot per-class AP on held-out classes (within-dataset) | Comes for free from Phase 6's openvocab-protocol runs — no separate script needed, see §2 |
| Actual full-scale ablation results | **Blocked — depends on Phase 6 runs at the report's schedule, which need a GPU** |

---

## 2. Zero-shot evaluation doesn't need a separate script

The openvocab data YAMLs (`neu_openvocab.yaml`, etc.) declare **all**
classes in `names` — held-out ones simply have zero training images (Phase
1 design, and the README's "Locked decisions" section). So a checkpoint
trained with `--protocol openvocab` already has WorldDetect comparing
against the full class vocabulary, including the held-out ones, at
inference time — the model was just never shown a training image containing
one. Running `scripts/train_tgfem.py --protocol openvocab`'s test-split
evaluation (test pool includes every held-out-class instance, by
construction — Phase 1 leakage guard) *is* the zero-shot evaluation. Its
`per_class_AP50` output already separates seen from held-out classes by
name. A dedicated "zero-shot eval script" would just be re-deriving numbers
Phase 6 already produces; `eval_tgfem.py` instead covers what that run does
**not**: a negative control and a cross-variant comparison (§3, §4).

---

## 3. Negative control, re-verified against a real checkpoint

Phase 3's `eval_yoloworld.py` caught a real bug: `WorldValidator.__call__`
silently overwrites whatever prompts `set_classes()` was given with the
dataset YAML's own class names, so every prompt style was secretly
evaluating the same vocabulary. That bug was specific to `WorldValidator`;
this project's Phase 6 pipeline uses a plain `DetectionValidator` (Phase 5,
`train_tgfem.py`), which has no `set_classes` override at all — but "the
class most likely to reintroduce this exact failure mode" is different
here: TG-FEM's context tokens are **learned jointly with detection**, so a
broken or short-circuited text path could still fit the seen classes during
training and only reveal itself at held-out-class query time.

`scripts/eval_tgfem.py`'s `negative_control()` loads a **trained** checkpoint,
strips its embedded `TextConditioner` hook, and re-attaches a fresh one with
absurd prompts (`"banana"`, `"elephant"`, ...) — no retraining. If real and
nonsense prompts score similarly, the classification head isn't actually
being driven by text. Smoke-tested against a 1-epoch toy checkpoint (both
scored 0.0, as expected at that scale — the check is written to require
`real mAP@0.5 > 0.01` before it treats a close absurd/real gap as a failure,
so it doesn't false-positive on an undertrained model with genuinely near-
zero accuracy).

---

## 4. Ablation plan (`scripts/run_ablations.py`)

Of the seven planned, three are load-bearing (README) — the script runs
these first:

| # | What | How |
|---|---|---|
| **(a)** | TG-FEM removed (identity check) | `variant=tgfem_identity`, same param budget as active |
| **(d)** | context-token count M | `--n-ctx 0/4/8/16`, `--tier` for the M=0 hand-written-prompt arm |
| **(g)** | gates driven by image vs text — **"this one is the paper"** | `variant=cbam_worlddetect` vs `variant=tgfem`, same WorldDetect head both times |

The other four (dataset/protocol/seed variations rather than new
architectures) are `--dataset`/`--protocol`/`--seed` flags on the same
`train_tgfem.py`, not separate configs — not enumerated individually since
they're combinations of flags already exposed, not new code.

Each variant is a separate `python` subprocess (not an in-process loop), so
one failing/OOMing run can't corrupt another's state, and already-completed
runs (`results/phase6_*.json` present) are skipped — the script is
resumable.

---

## 4a. ⭐ A real bug, found by actually running the dispatcher

`run_ablations.py --priority-only` had never been executed before this pass
- only its dependency `train_tgfem.py` had. Running all 6 priority variants
end to end (toy scale: 1 epoch, batch 4, imgsz 64) surfaced a genuine
failure: `cbam_worlddetect` crashed immediately.

**Cause.** `TGFEMTrainer.get_model()` (`src/tgfem/trainer.py`) asserted at
least one `TGFEM` layer had to exist in the model, on the assumption every
variant this trainer runs uses TG-FEM. That's false for `cbam_worlddetect`
- ablation (g)'s entire point is that it has **zero** TGFEM layers, using
CBAM gates instead, while still needing the WorldDetect text-threading this
same trainer provides. The config had simply never been trained before, so
the bad assertion had never fired.

**Fix.** Check for the `WorldDetect` head instead - that's the actual
requirement (it's what the language branch threads text into), and TGFEM
layer count is incidental. Re-ran after the fix: all 6 variants train
successfully, the resume/skip logic works (a second run skips all 6
instantly), and `eval_tgfem.py --compare-only` correctly reads and tabulates
all 6 results, including `cbam_worlddetect`'s.

This is worth stating plainly: **ablation (g) - "this one is the paper" -
was completely broken until this pass.** Nothing about the code review or
the earlier structural gate checks would have caught it; only running the
actual dispatcher did.

---

## 5. Comparison table (`eval_tgfem.py --compare-only`)

Reads every `results/phase6_{dataset}_{protocol}_*.json`, tabulates mAP@0.5
(and now inference ms - §4a's fix made a real 6-variant run possible, which
is what surfaced that this was missing) per variant, and computes:

- **ablation (g):** `tgfem.mAP50 - cbam_worlddetect.mAP50`, plus **per-class**
  delta with the two texture-confusable, NEU-DET-held-out classes
  (`crazing`, `rolled-in_scale`) flagged explicitly — this is Phase 3 §3's
  falsifiable prediction: if text conditioning works, TG-FEM's gain over
  CBAM should concentrate on exactly these two classes, the same ones
  attention already helped most versus the from-scratch anchor.
- **ablation (a):** `tgfem.mAP50 - tgfem_identity.mAP50`.

Verified against a real (if toy-scale) 6-variant run after the §4a fix - the
table reads all 6 JSONs correctly, computes both deltas, and lists the
per-class breakdown. Numbers themselves are still noise (1 epoch, random
CLIP) - only the machinery is confirmed correct.

---

## 6. DeepCrack OOD zero-shot evaluation (`scripts/eval_deepcrack.py`)

The one piece of Phase 7 that was not just untested but **entirely
unwritten** until this pass. Phase 1b held DeepCrack out entirely for
out-of-distribution zero-shot evaluation, but nothing ever actually queried
a trained checkpoint against it.

**What makes this a harder test than the within-dataset zero-shot in §2.**
The NEU/GC10 openvocab held-out classes still come from the *same dataset*
- same imaging regime, same general domain, just an unseen class name.
DeepCrack is a different dataset entirely (concrete/asphalt cracks vs steel
surface defects); the checkpoint has never seen a DeepCrack image at all.

**Design.** Loads a trained checkpoint and reuses its **learned** context
tokens throughout - including for the negative control - rather than
resetting to `n_ctx=0` the way `eval_tgfem.py`'s negative control does.
CoOp's premise is that a learned context generalises to new class names
(`demo.py` relies on the same premise for live queries), so this exercises
that premise under the hardest available condition. Reports per-tier AP
(bare/natural/material/alias - the corpus has 10 phrases for "crack") plus
an absurd-prompt (`"banana"`) control.

**The measured caveat gets carried into the output, not just the docs.**
DeepCrack's boxes are loose by construction (Phase 1b: median fill ratio
16.6%). The script prints this reminder alongside every result: a low
number is not by itself proof the model can't localise cracks - it could be
an IoU/geometry disagreement even when the model is pointing at the right
pixels. This was written into the script *because* Phase 1b already
measured and documented the limitation - the eval script's job is to not
let that finding get silently lost once a number exists to argue with.

**Verification.** No real DeepCrack data is available where this was built,
so a tiny synthetic dataset (10 images, drawn diagonal lines as fake cracks)
was generated matching `prepare_deepcrack.py`'s expected input layout, run
through that script to produce a real `deepcrack-yolo/` directory, and
`eval_deepcrack.py` was run against a real (toy-scale) trained checkpoint.
It loaded the checkpoint, found the embedded `TextConditioner`, ran all
requested tiers plus the negative control, and wrote a results JSON -
mechanically correct. As with everything else in this project, the numbers
themselves mean nothing until run with real CLIP weights, a GPU, and real
DeepCrack data.

---

## 7. Files created

```
scripts/eval_tgfem.py       negative control + comparison table (+ speed column)
scripts/eval_deepcrack.py   DeepCrack OOD zero-shot evaluation - NEW
scripts/run_ablations.py    resumable dispatcher over train_tgfem.py
src/tgfem/trainer.py        bugfix: WorldDetect check replaces the wrong TGFEM-layer assertion
```

---

## 8. Gate check

| Criterion | Status |
|---|---|
| Ablation plan covers all 3 priority ablations (a, d, g) | Pass |
| Zero-shot evaluation path identified (no redundant script needed) | Pass |
| Negative control re-implemented against a real checkpoint, not assumed | Pass |
| Comparison table includes the falsifiable per-class prediction + speed | Pass |
| Dispatcher is resumable (skips completed variants) | Pass |
| All 6 priority-ablation variants actually run successfully | Pass — after fixing §4a's bug |
| DeepCrack OOD eval script exists and runs end to end | Pass (synthetic data) |
| Real, full-scale ablation and DeepCrack numbers produced | **Blocked — needs a GPU and real CLIP weights** |

**Phase 7 gate: PASSED, including live execution of the dispatcher and both
eval scripts — not just structurally. Only real-scale numbers remain
blocked on compute.**

---

## 9. Carried forward

1. Once Phase 6 has run the three priority variants on a real GPU machine:
   `python scripts/eval_tgfem.py --checkpoint runs/phase6_neu_closed_tgfem/weights/best.pt --dataset neu --protocol closed --device 0`
   then `python scripts/eval_tgfem.py --compare-only --dataset neu --protocol closed`.
2. Download real DeepCrack data (README) and run `scripts/prepare_deepcrack.py`
   for real, then `scripts/eval_deepcrack.py` against a real trained
   checkpoint - everything so far used synthetic placeholder data.
3. Report per-class zero-shot AP for held-out classes specifically (Phase 3
   §2's finding: they're the hardest classes for a fully-supervised model
   too, so a modest zero-shot number isn't automatically a method failure) —
   already in `per_class_AP50`, just needs pulling out into the write-up
   once real numbers exist.
