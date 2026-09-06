# Phase 3 — Baselines

**Status:** Complete
**Goal:** Produce the three reference numbers that define the report's success criteria, before any TG-FEM claim can be made.

---

## 1. Headline results

All numbers are on the **held-out test split** of the NEU-DET closed protocol
(190 images, 429 instances), untouched until this point.

| Baseline | Role | mAP@0.5 | mAP@0.5:0.95 | P | R | ms/img |
|---|---|---|---|---|---|---|
| **YOLOv11n (stock)** | closed-vocab upper bound | **0.7717** | 0.4336 | 0.788 | 0.686 | 4.87 |
| **YOLOv11n + CBAM** | visual-attention control | **0.7372** | 0.4150 | 0.606 | 0.738 | 5.31 |
| **YOLO-World-S** | open-vocab competitor (zero-shot) | **0.0394** | 0.0131 | — | — | 10.5 |

### The two criteria are now concrete numbers

| Criterion (report §3) | Threshold |
|---|---|
| **Retention** — within 3 mAP@0.5 of the closed baseline | TG-FEM ≥ **0.7417** |
| **Gain** — +5 mAP@0.5 over zero-shot YOLO-World-S | TG-FEM ≥ **0.0894** on held-out classes |

The distance between those two numbers — **0.77 vs 0.04** — is the whole project
in one line: a closed detector is excellent on classes it was trained on, and a
zero-shot open-vocabulary model is close to useless on steel defects. TG-FEM has
to live somewhere in between.

---

## 2. Per-class AP@0.5

| class | stock | CBAM | Δ |
|---|---|---|---|
| scratches | 0.938 | 0.899 | −0.039 |
| patches | 0.877 | **0.905** | **+0.028** |
| pitted_surface | 0.829 | 0.768 | −0.061 |
| inclusion | 0.800 | 0.782 | −0.018 |
| rolled-in_scale | 0.658 | 0.590 | −0.069 |
| crazing | 0.528 | 0.480 | −0.048 |

**`crazing` (0.53) and `rolled-in_scale` (0.66) are the two weakest classes for
both models — and they are exactly the two classes held out in the NEU-DET
open-vocabulary protocol.**

That is an awkward coincidence worth stating plainly in the report: we are asking
TG-FEM to find, from text alone, the two categories that a *fully supervised*
detector handles worst. It makes the open-vocab task harder than the headline
numbers suggest, and it means a modest zero-shot result is not automatically a
failure of the method. Consider reporting zero-shot AP per class rather than only
the mean, so this is visible rather than buried.

---

## 3. ⚠ The retention criterion is currently measuring pretraining, not architecture

**CBAM scores 0.7372 — which already FAILS the 0.7417 retention threshold**,
despite containing no text conditioning at all and being a plain attention
variant of the same backbone.

The reason is a confound in how the baselines had to be trained:

| | initialisation |
|---|---|
| YOLOv11n stock | **COCO-pretrained** (`yolo11n.pt`) |
| YOLOv11n + CBAM | **from scratch** — no pretrained checkpoint exists for this architecture |
| YOLOv11n + TG-FEM | will also be **from scratch** |

Inserting modules at layers 5/8/13 shifts every downstream index, so the stock
state dict no longer maps onto the modified architecture. Any custom-module
variant therefore starts from random init while the stock baseline starts with
a large head start.

**Consequence:** as written, the 3-point retention criterion compares a
from-scratch model against a pretrained one. It measures the value of COCO
pretraining, not the value of the architecture. CBAM demonstrates this: it
tracked stock almost exactly on *validation* (0.788 vs 0.791) yet lands 3.4
points behind on *test*.

### Recommended fix — run a fourth baseline
Train **stock YOLOv11n from scratch** (`pretrained=None`, everything else
identical). That gives a like-for-like retention anchor, and costs one ~2-hour
run. Then:

- retention is measured against the **from-scratch** stock number;
- the **pretrained** number stays in the table as the absolute ceiling.

Without this, a TG-FEM result below 0.7417 is uninterpretable — we could not say
whether the module underperformed or simply lacked pretraining.

### What is already a fair comparison
**TG-FEM vs CBAM** — both from scratch, both at layers 5/8/13, both shape
preserving, same optimiser/seed/schedule. This is exactly ablation **(g)** in the
report ("gates driven by F rather than by S"), and it is the load-bearing
comparison for the paper's actual claim. That one needs no correction.

---

## 4. YOLO-World-S zero-shot — and a bug that would have invalidated ablation (d)

| prompt style | example | mAP@0.5 |
|---|---|---|
| bare | `"crazing"` | 0.0385 |
| natural | `"crazing on a steel surface"` | **0.0394** |
| **absurd** (control) | `"banana"`, `"elephant"` … | **0.0000** |

YOLO-World performs near-zero on steel defects, which is expected — it was
pretrained on COCO/LVIS everyday objects. That is precisely the gap (G1) the
project exists to address.

### The bug
The first run produced **identical mAP for both prompt styles** (0.0401 to four
decimal places). Testing with deliberately nonsensical prompts gave the *same*
0.04011 — proving the prompts were being ignored entirely.

Cause, in `ultralytics/models/yolo/world/val.py`:

```python
names = [... for name in check_det_dataset(self.args.data)["names"].values()]
current = model.names.values() if isinstance(model.names, dict) else model.names
if list(current) != names:
    model.set_classes(names, cache_clip_model=False)   # discards your prompts
```

`model.set_classes(prompts)` followed by `model.val()` **silently overwrites the
prompts with the dataset YAML's class names.** Every run was secretly evaluating
the same vocabulary.

**Why this mattered beyond today:** ablation **(d)** — hand-written prompts
(M=0) versus learned context tokens — runs through this exact path. Undetected,
every prompt variant would have returned identical numbers and we would have
concluded "prompt wording does not matter". That would have been an artifact,
not a result, and it directly contradicts gap G4.

### The fix
The validator is the only thing permitted to choose the vocabulary, so we choose
it *through* the validator: write the prompts into the dataset YAML's `names`.
Ground-truth labels match on class **index**, never on the string, so metrics
remain correct as long as prompt order matches label order.

The absurd-prompt control is now a permanent guard in `scripts/eval_yoloworld.py`
that **exits non-zero** if nonsense ever scores the same as real descriptions.

Small positive signal: natural prompts (0.0394) beat bare (0.0385). Tiny, but
the direction G4 predicts.

---

## 5. Speed — the >30 FPS constraint is not binding

| model | inference | ≈ FPS |
|---|---|---|
| YOLOv11n stock | 4.87 ms | **205** |
| YOLOv11n + CBAM | 5.31 ms | **188** |
| YOLO-World-S | 10.5 ms | 95 |

Measured on an **RTX 4050 Laptop GPU (6 GB)**, not the T4 the report names —
disclose the actual hardware.

CBAM costs 0.44 ms (~9%) for +99,110 parameters. TG-FEM is projected at 0.21 M
parameters, roughly 2× CBAM, so expect ~1 ms and ~170 FPS. **The >30 FPS
constraint has a ~6× margin and will not be the binding constraint** — worth
saying, since the report uses that constraint to justify rejecting Grounding DINO.

---

## 6. Reproduce

```bash
python scripts/eval_yoloworld.py                       # zero-shot, ~5 min
python scripts/train_baseline.py --variant stock       # ~1h50m
python scripts/train_baseline.py --variant cbam        # ~1h55m
```

Results land in `results/phase3_*.json`; weights and curves in `runs/phase3_*/`.
Settings shared by both training runs: 150 epochs, batch 16, 640×640, seed 42,
AMP on, `neu_closed.yaml`.

---

## 7. Gate check

| Criterion | Status |
|---|---|
| Closed-vocabulary upper bound measured | Pass — 0.7717 |
| Open-vocabulary competitor measured | Pass — 0.0394 |
| Visual-attention control measured | Pass — 0.7372 |
| Prompt pipeline verified against a negative control | Pass — absurd = 0.0000 |
| Both criteria expressed as concrete thresholds | Pass |
| Latency budget checked | Pass — 6× margin |

**Phase 3 gate: PASSED.**

---

## 8. Carried forward

1. **Run a from-scratch stock baseline** (§3). Without it the retention
   criterion is not interpretable. One ~2-hour run; do it before Phase 6.
2. **Report per-class zero-shot AP**, not just the mean — the held-out classes
   are the hardest ones (§2).
3. **Report corrections:** FPS is measured on an RTX 4050, not a T4; the
   >30 FPS constraint has a 6× margin.
4. The `set_classes` override (§4) will bite again in Phase 4 when we cache
   CLIP embeddings — the same "prompts get overwritten" trap applies to any
   evaluation path that goes through a World validator.
