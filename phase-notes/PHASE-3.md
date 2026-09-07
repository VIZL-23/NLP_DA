# Phase 3 — Baselines

**Status:** Complete
**Goal:** Produce the three reference numbers that define the report's success criteria, before any TG-FEM claim can be made.

---

## 1. Headline results

All numbers are on the **held-out test split** of the NEU-DET closed protocol
(190 images, 429 instances), untouched until this point.

| Baseline | Init | Role | mAP@0.5 | mAP@0.5:0.95 | P | R | ms/img |
|---|---|---|---|---|---|---|---|
| **YOLOv11n (stock)** | pretrained | absolute ceiling | **0.7717** | 0.4336 | 0.788 | 0.686 | 4.87 |
| **YOLOv11n + CBAM** | scratch | visual-attention control | **0.7372** | 0.4150 | 0.606 | 0.738 | 5.31 |
| **YOLOv11n (scratch)** | scratch | **like-for-like retention anchor** | **0.7069** | 0.3947 | 0.631 | 0.692 | 5.27 |
| **YOLO-World-S** | — | open-vocab competitor (zero-shot) | **0.0394** | 0.0131 | — | — | 10.5 |

### The two criteria are now concrete numbers

| Criterion (report §3) | Threshold |
|---|---|
| **Retention** — within 3 mAP@0.5 of the closed baseline, measured against the **from-scratch** anchor | TG-FEM ≥ **0.6769** |
| **Gain** — +5 mAP@0.5 over zero-shot YOLO-World-S | TG-FEM ≥ **0.0894** on held-out classes |

**But the threshold that actually matters is CBAM's 0.7372**, not the retention
floor — see §3.

The distance between those two numbers — **0.77 vs 0.04** — is the whole project
in one line: a closed detector is excellent on classes it was trained on, and a
zero-shot open-vocabulary model is close to useless on steel defects. TG-FEM has
to live somewhere in between.

---

## 2. Per-class AP@0.5

| class | stock (pretrained) | stock_scratch | CBAM |
|---|---|---|---|
| scratches | 0.938 | 0.904 | 0.899 |
| patches | 0.877 | 0.850 | **0.905** |
| pitted_surface | 0.829 | 0.807 | 0.768 |
| inclusion | 0.800 | 0.775 | 0.782 |
| rolled-in_scale | 0.658 | 0.519 | 0.590 |
| crazing | 0.528 | 0.386 | 0.480 |

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

## 3. ⭐ The confound, resolved — and CBAM is not a weak control

### The problem
Inserting modules at layers 5/8/13 shifts every downstream index, so the stock
`yolo11n.pt` state dict no longer maps onto the modified architecture. Every
custom-module variant (CBAM, TG-FEM) must therefore train **from scratch**,
while plain `stock` starts **COCO-pretrained**.

On that basis CBAM (0.7372) appeared to *fail* the original 0.7417 retention
threshold, despite containing no text conditioning at all. The criterion was
measuring pretraining, not architecture.

### The fix
A fourth baseline was run: **stock YOLOv11n, `pretrained=None`**, architecture,
schedule and seed otherwise identical. It isolates the two effects cleanly.

| comparison | Δ mAP@0.5 | what it isolates |
|---|---|---|
| stock **−** stock_scratch | **+0.065** | value of COCO pretraining |
| CBAM **−** stock_scratch | **+0.030** | value of attention, init controlled |

### The result reverses the earlier reading

**CBAM (0.7372) BEATS the fair anchor (0.7069) by 3.0 points.** Attention is not
costing accuracy — it was carrying a 6.5-point pretraining handicap that made it
*look* worse than it is.

Two consequences:

1. **The retention floor drops to 0.6769** (0.7069 − 0.03) when measured
   like-for-like. CBAM passes it comfortably.
2. **The bar that actually matters is CBAM's 0.7372, not the retention floor.**
   Attention alone already buys +3.0 points. For TG-FEM to support the paper's
   claim it must beat **CBAM**, not merely clear a retention threshold — because
   anything between 0.677 and 0.737 would be explained by attention, with text
   conditioning contributing nothing.

**This is exactly why the control was worth two hours.** Without it the honest
conclusion would have been "CBAM hurts", which is false, and TG-FEM would have
been measured against the wrong bar.

### Where attention helps — and it is the interesting place

| class | stock_scratch | CBAM | Δ |
|---|---|---|---|
| **crazing** | 0.386 | **0.480** | **+0.094** |
| **rolled-in_scale** | 0.519 | **0.590** | **+0.070** |
| patches | 0.850 | 0.905 | +0.055 |
| gc10_inclusion → inclusion | 0.775 | 0.782 | +0.007 |
| scratches | 0.904 | 0.899 | −0.005 |
| pitted_surface | 0.807 | 0.768 | −0.039 |

Attention delivers almost all of its gain on **`crazing` and
`rolled-in_scale`** — the two hardest, most texture-confusable classes, and
**the exact pair held out in the NEU-DET open-vocabulary protocol.**

This is direct support for the report's **G3** narrative. MPA-YOLO's authors
report that attention "struggles with interference" when background texture
resembles the defect; here attention is precisely where the texture-confusable
classes improve most. It also sharpens the TG-FEM hypothesis: if
text-conditioned gates beat feature-conditioned gates, the gain should appear
*on these same two classes*. That is a specific, falsifiable prediction to test
in Phase 6 — much stronger than a mean-mAP comparison.

### The load-bearing comparison
**TG-FEM vs CBAM** — both from scratch, both at layers 5/8/13, both shape
preserving, same optimiser/seed/schedule. This is ablation **(g)** and it needs
no correction. Report per-class AP for it, not just the mean.

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

1. **TG-FEM must beat CBAM (0.7372), not the retention floor (0.6769)** (§3).
   Anything in between is explained by attention alone.
2. **Falsifiable prediction to test in Phase 6:** if text conditioning works,
   the gain over CBAM should concentrate on `crazing` and `rolled-in_scale` —
   the texture-confusable classes where attention already helps most. Report
   per-class AP, not just the mean.
3. **Report per-class zero-shot AP** too — the held-out classes are the hardest
   ones (§2).
4. **Report corrections:** the retention criterion must name which baseline it
   is measured against (pretrained vs scratch); FPS is measured on an RTX 4050,
   not a T4; the >30 FPS constraint has a 6× margin.
5. The `set_classes` override (§4) will bite again in Phase 4 when we cache
   CLIP embeddings — the same "prompts get overwritten" trap applies to any
   evaluation path that goes through a World validator.
