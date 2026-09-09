# Phase 6 — Training

**Status:** Complete. All five runs executed on GPU with real CLIP weights.
**Goal:** Full runs on both protocols.

---

## 1. Results (NEU-DET, held-out TEST split)

Every run: 150 epochs, batch 16, 640x640, seed 42, `pretrained_clip_loaded: true`.

| variant | init | head | gates | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| stock (Phase 3) | pretrained | Detect | — | **0.7717** | 0.4336 |
| CBAM (Phase 3) | scratch | Detect | image | **0.7372** | 0.4150 |
| **TG-FEM** | scratch | WorldDetect | **text** | **0.7268** | 0.3998 |
| **CBAM+WorldDetect** [abl g] | scratch | WorldDetect | image | **0.7240** | 0.3954 |
| **TG-FEM identity** [abl a] | scratch | WorldDetect | none | **0.7183** | 0.3846 |
| stock scratch (Phase 3) | scratch | Detect | — | **0.7069** | 0.3947 |

---

## 2. The two ablations that decide the paper

```
ablation (g)  TG-FEM  -  CBAM+WorldDetect  =  +0.0028
ablation (a)  TG-FEM  -  identity           =  +0.0085
```

**Both are inside the noise floor.** Per-epoch validation mAP swung by ±0.10
during training (e.g. stock_scratch went 0.754 -> 0.560 -> 0.760 across epochs
80-90); a 0.28-point difference on a 190-image test split is far below that.

With the detection head now held constant, **there is no evidence that
text-derived gates outperform image-derived gates.**

### The falsifiable prediction fails

Phase 3 §3 predicted: if text conditioning works, TG-FEM's gain over CBAM should
concentrate on `crazing` and `rolled-in_scale` — the texture-confusable classes
where attention already helped most.

| class | TG-FEM | CBAM+WD | delta |
|---|---|---|---|
| **crazing** | 0.4036 | 0.4386 | **−0.0350** |
| **rolled-in_scale** | 0.5556 | 0.5696 | **−0.0139** |
| pitted_surface | 0.8257 | 0.7724 | +0.0532 |
| scratches | 0.9151 | 0.8990 | +0.0161 |
| patches | 0.9000 | 0.8848 | +0.0152 |
| inclusion | 0.7608 | 0.7797 | −0.0188 |

TG-FEM is **worse** on both predicted classes. The prediction was specific and
falsifiable, and it was falsified. The small positive total comes from
`pitted_surface`, which the mechanism gives no reason to expect.

---

## 3. ⚠ The open-vocabulary claim fails outright

Openvocab protocol, TEST split = 600 images / 1321 instances:

| class | images | instances | AP@0.5 |
|---|---|---|---|
| **crazing** (held out) | 300 | 688 | **0.000** |
| **rolled-in_scale** (held out) | 300 | 628 | **0.000** |
| inclusion (seen) | 1 | 1 | 0.000 |
| patches (seen) | 4 | 4 | 0.564 |

**Both held-out classes score exactly zero, with 688 and 628 instances present.**
Not a sparse-data artifact — the model detects none of them.

### The reported number is misleading
`results/phase6_neu_openvocab_tgfem.json` shows `mAP50 = 0.1409`. That is the
mean over four classes where only `patches` scores:
`(0 + 0 + 0.564 + 0) / 4 = 0.141`. It is driven by **4 instances of a class the
model was trained on.**

**Do NOT compare it to YOLO-World-S's 0.0394** and claim the +5-point criterion
is met. The test sets differ (600-image held-out pool vs the 190-image closed
test split) and the number reflects a seen class. On the criterion's actual
subject — held-out categories queried by text — the score is **0.000**.

CBAM+WorldDetect openvocab behaves identically (0.0793, also 0.000 on both
held-out classes), so this is a property of the setup, not of TG-FEM.

### Why, and why it is not a bug
`WorldDetect` classifies open-vocabulary, but that ability comes from
**large-scale region-text pretraining** — YOLO-World used 27M grounding pairs.
This project trains from scratch on **1,075 images**.

The vision branch only ever learns to place *seen* classes' region embeddings
near their text embeddings. Nothing in the objective teaches it that an unseen
defect's appearance should land near unseen text. The architecture is sound; it
is data-starved by roughly four orders of magnitude.

The risk register anticipated this exactly (row 1: "the zero-shot gain does not
materialise").

---

## 4. The text path IS live — verified, not assumed

Before accepting a negative result, `scripts/eval_tgfem.py`'s negative control
loaded the trained checkpoint, stripped its `TextConditioner`, and re-attached
one carrying absurd prompts ("banana", "elephant", ...):

```
real prompts   mAP@0.5 = 0.3994
absurd prompts mAP@0.5 = 0.0243     <- 16x collapse
```

The model genuinely depends on its text input. **The weak ablation deltas are a
real finding, not a disconnected wiring.**

(Both figures sit below 0.7268 because the control re-attaches with `n_ctx=0`,
discarding the learned context tokens; only the real-vs-absurd contrast matters.)

---

## 5. Bugs found and fixed on GPU

Three defects that a CPU-only machine could not have surfaced:

1. **Device split in the CoOp path** (`src/tgfem/language.py`).
   `TextConditioner` holds the encoder as a plain attribute, not a submodule, so
   `model.to(device)` moved `ctx` to CUDA while the frozen CLIP tower stayed on
   CPU. Died at the first validation pass. `encode_frozen` already guarded
   itself, so the `n_ctx=0` arm would have survived while every run that matters
   failed. Fixed with a device-sync guard in `ContextTokenLearner.forward`.

2. **`TGFEMTrainer` rejected the ablation (g) control** (`src/tgfem/trainer.py`).
   A hard guard required TGFEM layers, but CBAM+WorldDetect has none — it still
   needs the language branch for `txt_feats`, it just has no text-gated
   attention. **This made the single comparison the paper rests on impossible to
   run.** Now rejects only models that consume no text at all.

3. **Device string in `eval_tgfem.py`.** Ultralytics accepts a bare `"0"`;
   `torch.load` and `Module.to` do not. Normalised once at function entry.

Also: **CLIP now loads from the local `weights/clip/ViT-B-32.pt`** that
Ultralytics already downloads for YOLO-World, instead of huggingface.co. The HF
route stalled at 0 bytes here too — the same failure that silently degraded
every Phase 4-6 run to a randomly-initialised encoder. Training now needs no
network at all.

---

## 6. Report corrections this phase adds

1. **TG-FEM costs 0.95 M parameters, not 0.21 M** (3,538,201 vs the 2,591,010
   identity baseline) — 4.5x the stated overhead. GFLOPs 6.4 -> 8.3 (+30%).
2. **The +5-point criterion is not met** on its actual subject (§3).
3. **The retention criterion must name its baseline.** TG-FEM (0.7268) clears
   the from-scratch anchor (0.7069) but not the pretrained one (0.7717).

---

## 7. Gate check

| Criterion | Status |
|---|---|
| All variants trained at the report's schedule | Pass |
| Real CLIP weights in every run | Pass |
| Evaluated on the untouched TEST split | Pass |
| Ablations (a) and (g) executed | Pass |
| Text path verified live by negative control | Pass |
| TG-FEM outperforms the CBAM control | **FAIL — +0.0028, within noise** |
| Zero-shot detection of held-out classes | **FAIL — 0.000 AP** |

**Phase 6 gate: PASSED procedurally. The scientific result is negative.**

---

## 8. Carried forward

1. The headline open-vocabulary claim is unsupported. The report needs
   restructuring around what the data shows — see PHASE-7.md.
2. Untested paths remain: GC10-DET, DeepCrack OOD, ablation (d) (`--tier`,
   `--n-ctx`), and TG-FEM at P5-only vs all three scales.
3. A negative result with this many controls (identity at equal parameters, an
   image-gated control at equal head, a live-text negative control, and a
   pre-registered falsifiable prediction) is a legitimate finding, not a failed
   project.
