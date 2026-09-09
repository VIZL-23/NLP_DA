# Phase 8 — Demo

**Status:** Complete (code + structural verification). Needs a real Phase 6
checkpoint (real CLIP weights, real training) to be useful for the report.
**Goal:** Take a trained checkpoint and answer the project's actual claim —
type an arbitrary phrase, get boxes back, including for defect types never
seen in training.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Load a Phase 6 checkpoint and run inference on one image | Done — `scripts/demo.py` |
| Accept arbitrary free-text queries, not just the training corpus | Done |
| Reuse the checkpoint's *learned* context tokens rather than resetting them | Done |
| Draw and save labelled boxes | Done |
| Verify end to end against a real checkpoint | Done — CPU, toy scale |

---

## 2. Why this script exists separately from `eval_tgfem.py` / `eval_deepcrack.py`

Both eval scripts answer a benchmarking question — "what's the mAP on a known
test split". `demo.py` answers the actual product question the README opens
with: an inspector types a phrase, gets boxes. There is no ground truth, no
metric, no fixed vocabulary — just a checkpoint, one image, and however many
free-text queries the caller wants.

## 3. How it reuses learned context without retraining

The checkpoint embeds its `TextConditioner` (Phase 4) as a `forward_pre_hook`
— the same pickling design that made `demo.py` possible without any special
loading code. At inference time the script:

1. Loads the checkpoint, finds the `TextConditioner` instance among
   `model._forward_pre_hooks` (it's identifiable by `isinstance`, not by name
   or index — hooks are stored in an ordered dict keyed by handle id).
2. Overwrites `tc.class_texts` with `args.query`. **`tc.learner.ctx` is left
   untouched** — this is CoOp's actual premise (Zhou et al. 2022): the learned
   context prefix should generalise to class names it never trained on, not
   just re-encode the training vocabulary. If the demo reset `ctx` to
   untrained values here, it would silently defeat the entire point of the
   ablation-(d) context-token experiment.
3. Updates `model.model[-1].nc = len(args.query)`. This one is easy to get
   wrong: `WorldDetect.forward` derives its output width (`self.no = nc +
   reg_max*4`) from `self.nc`, not from `len(text)` at call time. Stock
   Ultralytics has exactly the same requirement in `WorldModel.set_classes()`
   — this mirrors that, it isn't a new discovery, just an easy step to forget
   which would silently truncate/misalign the output tensor instead of
   erroring.

## 4. Verification

Structural, same posture as every other phase-gate check — no accuracy claim,
because the checkpoint used to test this was trained with a random-init CLIP
tower (§5 of `PHASE-4.md`), so any confidence numbers below are noise:

```
Loading runs/phase6_neu_closed_tgfem/weights/best.pt ...
Trained vocabulary (6): [...natural-tier NEU-DET phrases...]
Reusing learned context tokens (n_ctx=4), swapping in new query text:
  -> 'a pitted corroded surface'
  -> 'a completely novel defect description never in training'
2 detection(s) above conf=0.25:
  ...
saved -> crazing_1_demo.png
```

The important thing this confirms structurally: the script accepts a query
count that **differs from the training vocabulary size** (2 queries against a
6-class checkpoint) and runs NMS/decoding correctly at the new size, which is
the actual open-vocabulary behaviour being claimed. It was re-run after every
later change to `language.py` and `trainer.py` (the mypy/lint pass) to make
sure nothing upstream broke the hook lookup.

---

## 5. Files created

```
scripts/demo.py   Phase 8 text-query inference demo
```

**Reproduce:**
```bash
python scripts/demo.py \
    --checkpoint runs/phase6_neu_closed_tgfem/weights/best.pt \
    --image datasets/neu-det-yolo/images/crazing_1.jpg \
    --query "crazing on a steel surface" "a pitted corroded surface"
```

Requires a checkpoint produced by `scripts/train_tgfem.py` (Phase 6) — any
checkpoint without an attached `TextConditioner` (e.g. a stock/CBAM baseline)
is rejected with a clear error rather than failing silently.

---

## 6. Gate check

| Criterion | Status |
|---|---|
| Loads an arbitrary Phase 6 checkpoint | Pass |
| Accepts queries outside the training vocabulary | Pass |
| Reuses learned `ctx`, does not reset it | Pass |
| Vocabulary-size change (`nc`) handled correctly | Pass |
| Boxes drawn and saved | Pass |
| Re-verified after Phase 4/5 follow-up edits | Pass |
| Run against a checkpoint with real (non-random) CLIP weights | **Blocked** — same network constraint as Phase 4, no route to `huggingface.co` on the dev machine used for this pass |

**Phase 8 gate: PASSED structurally.** The one meaningful remaining step is
running this against a checkpoint trained with real CLIP weights on a GPU —
at that point this script *is* the report's headline demo, not a placeholder.

---

## 7. Carried forward

1. Nothing code-side is missing. Once a real Phase 6 checkpoint exists (see
   `PHASE-6.md` §4), run this against a few hand-picked images and phrases for
   the report's demo section/screenshots.
2. Worth trying deliberately: a query phrase from a *held-out* class (e.g.
   `crazing` under the open-vocabulary protocol) and a phrase for a defect
   type in **none** of the training data at all (e.g. a GC10 or DeepCrack
   description on a NEU-DET checkpoint) — both are exactly what "open
   vocabulary" is supposed to mean, and both are one command away with this
   script.
