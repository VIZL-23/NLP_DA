# Phase 2 — Walking Skeleton

**Status:** Complete
**Goal:** Prove the whole pipeline is wired end to end with TG-FEM in the graph — *before* any of the real maths exists. Kill integration risk while the module is still trivial to debug.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Register a custom module with Ultralytics | Done — no fork, no parser patch |
| Place it at all three pyramid levels | Done — layers 5, 8, 13 |
| Verify exact identity behaviour | Done — `torch.equal` passes |
| Train end to end without shape/autograd errors | Done — 2 epochs |
| Prove boxes come out of the far end | Done — 70 boxes emitted |
| Verify the module survives a checkpoint round-trip | Done (unplanned, important) |

---

## 2. How registration works

Ultralytics' `parse_model` (in `ultralytics/nn/tasks.py`) resolves a module name
from a YAML row using **`globals()[m]`**, evaluated inside that module. So making
a custom layer usable needs exactly one line:

```python
import ultralytics.nn.tasks as tasks
tasks.TGFEM = TGFEM
```

That is all `src/tgfem/register()` does. **No fork of Ultralytics, no patching of
the parser.** This was the single biggest unknown in the build plan and it turned
out to be cheap.

### The channel contract
`parse_model` has a special case for every known block type. A class it does not
recognise falls through to:

```python
else:
    c2 = ch[f]          # output channels = input channels
```

and receives the YAML `args` verbatim. So a custom module is safe **provided it
preserves shape** — which is exactly TG-FEM's contract:

```
input  (B, C, H, W)  ->  output (B, C, H, W)
```

`[-1, 1, TGFEM, [256]]` therefore constructs `TGFEM(d=256)`.

### Passing text in (Phase 4+)
`parse_model` builds a single-input layer chain, so a second (text) argument
cannot be threaded through the YAML. TG-FEM instead exposes `self.txt`, which the
parent model populates immediately before the forward pass — the same pattern
YOLO-World uses for `txt_feats`. It is deliberately **not** a buffer or
parameter, so it never enters the state dict or the autograd graph.

---

## 3. Model structure

`cfg/yolo11-tgfem.yaml` — derived from stock `yolo11.yaml`, with TG-FEM inserted
on each backbone pyramid output **before** the PAN-FPN neck (report §4.2).

Inserting three layers shifts every downstream index. The rewiring:

| stock index | becomes | meaning |
|---|---|---|
| 4 (P3) | **5** | TG-FEM @ P3 (80×80) |
| 6 (P4) | **8** | TG-FEM @ P4 (40×40) |
| 10 (P5) | **13** | TG-FEM @ P5 (20×20) |
| 13 | 16 | head P4 |
| 16 / 19 / 22 | 19 / 22 / 25 | Detect inputs |

Every `Concat` and the `Detect` head now reference the **conditioned** tensors,
not the raw backbone ones — which is the entire point of the placement.

---

## 4. Verification results

```
TGFEM at layers : [5, 8, 13]
identity check  : PASS   (torch.equal(mod(t), t) is True)
parameters      : 2,591,010     (fused: 2,583,322 — 6.4 GFLOPs)
output shape    : (1, 10, 8400)
```

- `10` = 4 box coords + 6 NEU-DET classes ✓
- `8400` = 80² + 40² + 20² anchor positions ✓

**The identity check matters beyond this phase.** Ablation (a) in the report is
"TG-FEM removed" — but removing a module also removes its parameters, which
confounds the comparison. Because TG-FEM is an *exact* pass-through in identity
mode, ablation (a) can be run at the **same parameter budget**, making it a
genuine controlled test rather than an architectural confound.

### Training run (2 epochs, 143 images, batch 8, 640×640)

| Metric | Value |
|---|---|
| Peak GPU memory | **1.57 GB** |
| Inference speed | **20.3 ms/image ≈ 49 FPS** |
| Wall clock | 32 seconds |
| mAP50 | 5.3e-05 (meaningless — that is fine) |

**Accuracy here is noise and should be ignored.** Two epochs from scratch on 143
images learns nothing. The gate is purely structural.

### Boxes genuinely emitted
```
prediction on crazing_1.jpg (conf>=0.001): 70 boxes
  patches   conf=0.0022  xyxy=[0.0, 0.0, 89.8, 79.6]
  patches   conf=0.0022  xyxy=[109.8, 0.0, 200.0, 79.6]
  ...
```

> A default `predict()` returns **zero** boxes here, because nothing clears the
> 0.25 confidence threshold after 2 epochs. The first version of the gate script
> printed "PASSED" alongside "0 boxes" — a check that proved nothing. It now
> predicts at `conf=0.001` and **fails loudly** if the head emits nothing.

### Checkpoint round-trip
Reloading `runs/phase2_skeleton/weights/best.pt` returns `TGFEM` still at layers
`[5, 8, 13]`. The custom class serialises and deserialises correctly — worth
knowing now rather than in Phase 6.

---

## 5. Two useful findings for the report

1. **We are already above the >30 FPS constraint.** 20.3 ms/image ≈ **49 FPS**
   on an RTX 4050 laptop GPU, before any optimisation. Note the report cites a
   T4; disclose the actual hardware. TG-FEM's Phase 5 maths (+0.21 M params,
   +0.4 GFLOPs) will cost some of this headroom but not all of it.
2. **6 GB VRAM is not the constraint we feared.** Peak was 1.57 GB at batch 8 /
   640×640. We can comfortably raise to **batch 16**, matching the report's
   stated protocol without gradient accumulation.

---

## 6. Files created

```
src/tgfem/
├── __init__.py            register() - injects TGFEM into ultralytics.nn.tasks
└── module.py              TGFEM (identity mode; Phase 5 fills in the maths)
cfg/
└── yolo11-tgfem.yaml      YOLO11 + TG-FEM at P3/P4/P5, indices rewired
scripts/
└── train_skeleton.py      structural assertions + tiny training run + box check
```

**Reproduce:**
```bash
python scripts/train_skeleton.py                       # 2 epochs, 10% of train
python scripts/train_skeleton.py --epochs 5 --fraction 0.2
```

Outputs land in `runs/phase2_skeleton/` (gitignored).

---

## 7. Gate check

| Criterion | Status |
|---|---|
| Custom module resolves from YAML | Pass |
| Present at all three pyramid levels | Pass |
| Exact identity (ablation (a) is valid) | Pass |
| Trains without shape or autograd errors | Pass |
| Detection head emits boxes | Pass (70) |
| Survives checkpoint save/load | Pass |

**Phase 2 gate: PASSED.** Integration risk is eliminated — Phase 5 now only has
to replace the identity `forward` with real maths.

---

## 8. Carried forward

1. `self.txt` plumbing is defined but unused until Phase 4 — the parent model
   still needs code to populate it before each forward.
2. Batch size can rise to 16 given the memory headroom.
3. Phase 3 (baselines) is next and needs **no** new code from this phase.
