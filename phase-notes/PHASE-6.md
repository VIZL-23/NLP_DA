# Phase 6 — Training

**Status:** Scripts complete and smoke-tested. Full runs blocked - no GPU available for this pass.
**Goal:** Full runs on both protocols.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| A training script that uses the real language branch + TG-FEM math (Phases 4/5), not the plain `YOLO(...).train()` API Phase 3 used | Done — `scripts/train_tgfem.py` |
| Cover all three Phase 5 variants (tgfem, ablation (a), ablation (g)) from one script | Done |
| Evaluate on the held-out TEST split, not the val split Ultralytics reports during training | Done (Phase 3's own finding, reapplied) |
| Smoke-test the full script end to end | Done, CPU/toy scale |
| Actually run the report's schedule (150 epochs, batch 16, imgsz 640) for a real number | **Not done — no GPU available yet** |

---

## 2. Why this isn't just train_baseline.py again

Phase 3's `train_baseline.py` calls the high-level `YOLO(cfg).train(...)`
API. That's fine for `stock`/`cbam`, which need no text at all. It cannot
work for `tgfem` (or the ablation (g) control), because there is no way to
hand a `YOLO` wrapper's trainer a language branch to own, recompute every
forward pass, and expose to the optimiser — that machinery is
`TGFEMTrainer` (Phase 5). `scripts/train_tgfem.py` is the Phase 6
equivalent of `train_baseline.py`, built on `TGFEMTrainer` instead.

**Vocabulary selection.** `class_texts_for()` (`src/tgfem/data.py`) picks
one phrase per class, in the class-index order the dataset YAML declares,
from a given tier (`--tier`, default `natural`) — this is what ablation
(d) varies, together with `--n-ctx`. Order matters: WorldDetect/TG-FEM match
text to ground truth by class **index**, never by string (the same rule
Phase 3's `eval_yoloworld.py` had to enforce after finding the opposite bug
there).

**Test-split evaluation.** `trainer.validator` (built by
`TGFEMTrainer._setup_train`) validates on the data YAML's `val` split
during training — the report's criteria are defined on `test`, which is
supposed to stay untouched until the very end (Phase 3 finding). The script
builds a **second, fresh** `DetectionValidator` with `split="test"`
afterwards, exactly mirroring what `train_baseline.py` does via the
high-level API's `model.val(split="test", ...)`.

---

## 3. Smoke test (CPU, 1 epoch, imgsz=64, batch=8)

```
Using 1434 train, 176 val images ...
1 epochs completed in 0.030 hours.
Validating .../best.pt...
  val split (176 img)  : mAP@0.5 = 0.00102
Validating on TEST split (separately, 190 img, 429 instances):
  all        190        429    0.00374   0.183   0.00433   0.000995
saved -> results/phase6_neu_closed_tgfem.json
```

Numbers are meaningless (1 epoch, toy imgsz, random-init CLIP — see
`phase-notes/PHASE-4.md` §5) — the point of this run was to prove the
script itself is correct: it trains, evaluates on the right split, prints
the CLIP-not-pretrained warning, and writes a `results/phase6_*.json` in
the same schema Phase 3 established. It is.

---

## 4. ⚠ BLOCKED — no GPU available

The machine used for Phases 4-6 so far has no CUDA device
(`torch.cuda.is_available()` is `False`) and no route to `huggingface.co`
(Phase 4 §5) — the two things a real Phase 6 run needs. The report's
schedule (150 epochs, batch 16, imgsz 640, per baseline — Phase 3 §6) took
**~2 hours per run on an RTX 4050** for the much simpler Phase 3 baselines;
TG-FEM's extra compute (attention + a CLIP forward pass every step, even
frozen) will cost more, not less. Running that on CPU would take,
conservatively, many times longer per run, and there are at least three
required runs (tgfem, ablation (a), ablation (g)) before any ablation (d)
variants.

**What is ready to go, the moment a GPU + internet-connected machine is
available:**

```bash
python scripts/train_tgfem.py --variant tgfem            --dataset neu --protocol closed --device 0
python scripts/train_tgfem.py --variant tgfem_identity    --dataset neu --protocol closed --device 0
python scripts/train_tgfem.py --variant cbam_worlddetect  --dataset neu --protocol closed --device 0
python scripts/train_tgfem.py --variant tgfem --protocol openvocab --device 0   # the zero-shot numbers
```

or, to run the priority ablations as a batch: `python scripts/run_ablations.py --priority-only --device 0`
(see `phase-notes/PHASE-7.md`).

**Before trusting any resulting number:** check
`results/phase6_*.json`'s `"pretrained_clip_loaded"` field is `true`. If
it's `false`, the run still executed correctly but the text embeddings were
semantic noise (same caveat as every run so far).

---

## 5. Files created

```
scripts/train_tgfem.py   Phase 6 training script (tgfem / tgfem_identity / cbam_worlddetect)
```

---

## 6. Gate check

| Criterion | Status |
|---|---|
| Script covers all three Phase 5 model variants | Pass |
| Vocabulary construction matches class-index order (no string-matching bug) | Pass |
| Evaluates on TEST split, separately from training-time val | Pass |
| Results schema consistent with Phase 3's `results/phase3_*.json` | Pass |
| Smoke-tested end to end (structure, not accuracy) | Pass |
| A real 150-epoch/GPU run produced | **Blocked — no GPU available yet** |

**Phase 6 gate: PASSED structurally, blocked on compute for the actual
numbers.**

---

## 7. Carried forward

1. **Run the four commands in §4 on a GPU machine with internet access.**
   This is the single largest remaining piece of work on the whole project
   — everything upstream of it (Phases 0-5) is done and verified; nothing
   downstream (Phase 7's ablations, Phase 8's report numbers) can happen
   without these runs existing.
2. Confirm `pretrained_clip_loaded: true` in the resulting JSONs before
   using them for anything.
3. GC10-DET and DeepCrack aren't present on this machine either (gitignored,
   download separately per the README) — Phase 6 runs against `neu` only
   were exercised so far; the `--dataset gc10` path is written but untested
   end-to-end for lack of the data.
