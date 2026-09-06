# TG-FEM — Text-Guided Open-Vocabulary Surface Defect Detection

**Course:** BCSE409L Natural Language Processing · **Faculty:** Dr. Vijayaprabakaran K
**Team:** Ritvik Kaza (24BCE5351) · Vishal Narayan V (24BCE5427)

---

## What this project does

Conventional defect detectors are **closed-vocabulary**: their class list is
compiled into the model weights, so adding one new defect type means collecting
data, annotating it, and retraining.

This project makes the defect vocabulary a **runtime parameter**. An inspector
types a plain-English description — *"rolled-in scale on hot-rolled steel"* — and
the model returns bounding boxes for matching regions, **including defect types
it was never trained on**.

**The contribution is TG-FEM** (Text-Guided Feature Enhancement Module): an
attention module whose channel and spatial gates are computed from the *text*
rather than from the image's own pooled statistics. In short — the model is told
*what to look for* before it decides *where to look*. This is the one thing that
distinguishes it from CBAM-family attention and from YOLO-World's channel-only
max-sigmoid attention.

### How the pieces fit

| Component | Role | Trained? |
|---|---|---|
| YOLOv11n backbone + PAN-FPN neck | finds **where** things are; produces boxes | yes |
| CLIP text encoder | turns a phrase into a meaning-fingerprint | **frozen** |
| Learnable context tokens | teaches CLIP civil/steel jargon without retraining it | yes |
| **TG-FEM** | uses the text to steer the detector's attention | yes |
| Region-text contrastive head | labels each box by nearest text fingerprint | yes |

CLIP understands language but cannot localise; YOLO localises but has no
language. Training teaches YOLO to describe regions in CLIP's space, so matching
a box to a phrase becomes a nearest-neighbour question.

---

## Quick start

```bash
# 1. Create the environment (reuses the system CUDA torch - no 2.5 GB re-download)
python -m venv .venv --system-site-packages
.venv/Scripts/activate
pip install -r requirements.txt

# 2. Verify toolchain + datasets
python scripts/check_env.py

# 3. Build the YOLO-format dataset and both split protocols
python scripts/prepare_neu_det.py

# 4. Sanity-check the conversion visually
python scripts/verify_labels.py     # -> phase-notes/assets/label_check.png
```

> **torch is not pinned** in `requirements.txt` — install it separately to match
> your CUDA driver. Developed against torch 2.11.0+cu128 on an RTX 4050 (6 GB).

### Datasets

`NEU-DET` and `SDNET2018` are committed to the repo. **`GC10-DET` and
`DeepCrack` are gitignored — download them separately:**

| Dataset | Source | Save to |
|---|---|---|
| GC10-DET | [kaggle: zhangyunsheng/defects-class-and-location](https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location) | `datasets/GC10-DET/` |
| DeepCrack | [github: yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack) → `./dataset` | `datasets/DeepCrack/` |

`python scripts/check_env.py` reports exactly what is present and what is wrong.

---

## Build phases

Detailed write-ups live in **`phase-notes/PHASE-N.md`** — read the relevant one
before starting a phase.

| # | Phase | Status | What it does |
|---|---|---|---|
| **0** | Foundation | ✅ **Done** | venv + CUDA verified, Ultralytics installed, datasets audited, repo hygiene |
| **1** | Data pipeline | ✅ **Done** | VOC→YOLO conversion, both split protocols, 60-phrase text corpus |
| **1b** | GC10-DET + DeepCrack | ⬜ Todo | Convert the two new datasets; pick 3 held-out classes |
| **2** | Walking skeleton | ⬜ **Next** | No-op module wired into Ultralytics, 2-epoch tiny run — proves the plumbing |
| **3** | Baselines | ⬜ Todo | YOLOv11n trained, YOLO-World-S zero-shot, YOLOv11n+CBAM control |
| **4** | Language branch | ⬜ Todo | Cache frozen CLIP embeddings; add learnable context tokens |
| **5** | TG-FEM | ⬜ Todo | Implement the module, insert at P3/P4/P5 |
| **6** | Training | ⬜ Todo | Full runs on both protocols |
| **7** | Evaluation & ablations | ⬜ Todo | mAP, zero-shot, FPS + the 7 ablations |
| **8** | Report & demo | ⬜ Todo | DA2 writeup, live demo |

### Build strategy

1. **Stand on Ultralytics, don't rewrite it.** It already ships YOLOv11,
   `YOLOWorld`, the `WorldDetect` contrastive head, and `MaxSigmoidAttnBlock`
   (YOLO-World's channel-only attention). TG-FEM slots into the same socket as
   that block, which makes the headline comparison a one-line YAML swap.
2. **Walking skeleton before novelty.** Phase 2 wires a *no-op* module through
   the whole pipeline first. Integration is the biggest risk; kill it early.
3. **Baselines before novelty.** You cannot claim "+5 mAP over YOLO-World-S"
   without a YOLO-World-S number — and if TG-FEM disappoints, solid baselines
   are still a report.

### Priority ablations
Of the seven planned, three are load-bearing — run these first:
**(a)** TG-FEM removed (identity check) · **(d)** context-token count M
· **(g)** gates driven by image vs by text — *this one is the paper*.

---

## Repository layout

```
scripts/
  check_env.py         toolchain + dataset audit (run this first)
  prepare_neu_det.py   VOC→YOLO, splits, dataset YAMLs, geometry stats
  verify_labels.py     draws converted boxes back onto images
  build_prompts.py     text-corpus validation + protocol filtering
prompts/
  defect_corpus.json   60 natural-language defect descriptions  (DRAFT)
datasets/              source data + generated neu-det-yolo/ (gitignored)
phase-notes/           per-phase write-ups and findings
```

---

## Locked decisions

**GC10-DET held-out classes (open-vocabulary protocol):**
> `crescent_gap` · `punching_hole` · `welding_line`

Chosen by exhaustively evaluating all 120 possible triples. The metric that
matters is **worst-case seen-class retention** — how much training data the
*remaining* classes lose to the test pool, because GC10 images are multi-label.

| Candidate | Train pool | Worst-case retention |
|---|---|---|
| **crescent_gap / punching_hole / welding_line** | **1572 (69%)** | **73%** |
| inclusion / oil_spot / water_spot | 1548 | 89% — *but see below* |
| crescent_gap / water_spot / punching_hole | 1420 | **30%** ❌ |

*Why this triple wins:* these three **co-occur heavily with each other**
(punching_hole+welding_line in 222 images, crescent_gap+welding_line in 150).
Holding them out **as a block** lets that cluster leave together. Splitting it —
as the third row does — strands `welding_line`, which loses 70% of its training
images. Result: every seen class now retains 92–100% (except `rolled_pit` at
73%, which is inherently rare at 44 images).

*Why not the 89% option:* it holds out `inclusion`, which also exists in
NEU-DET. Training on NEU-DET's `inclusion` and then querying GC10's with nearly
identical wording is **not a genuine zero-shot test** — it would inflate the
result. Never hold out a class whose twin is in the training set.

**Class-name collision — namespaced, not merged:**
> `neu_inclusion` vs `gc10_inclusion`

They are the same concept but different regimes (200×200 grayscale close-up vs
2048×1000 steel sheet), so they get distinct class IDs. Text descriptions stay
natural in the corpus; only the label-space identity is namespaced.

---

## Things you must know before touching the data

These were found by auditing and are easy to get wrong:

- **GC10-DET folders are NOT the labels.** Images are multi-label; the folder
  shows only the *dominant* defect. **The XML is ground truth.** Converting by
  folder name produces silently wrong labels.
- **GC10-DET class names are Chinese pinyin** in the XML (`3_yueyawan` =
  `crescent_gap`). The verified mapping is `GC10_NAME_MAP` in `check_env.py`.
  It also merges the `10_yaozhe`/`10_yaozhed` typo pair and drops a corrupt
  class named `d`.
- **SDNET2018 has no bounding boxes** — classification folders only. It cannot
  train a detector or produce mAP. Use it for backbone warm-up / hard negatives.
- **Open-vocab splits need a leakage guard.** 123 NEU-DET images contain more
  than one class, so a "seen-class" training image can hide a held-out object.
  `prepare_neu_det.py` excludes any such image and asserts this every run.
- **NEU-DET is 200×200 grayscale**; GC10-DET is 2048×1000. Training at 640
  upscales one and downscales the other.

---

## Known report corrections needed (before DA2)

Measured from our own data, so an examiner could reproduce them:

1. **Constraint (ii) is wrong.** The report claims defects are "<2% of image
   pixels, aspect ratios beyond 10:1". Actual NEU-DET median box area is
   **11.8%**; `pitted_surface` averages **55.5%**; only 3.2% of boxes exceed
   10:1. → Rescope as a **scale-range** problem (14× spread in object area),
   which also justifies multi-scale P3+P4+P5 conditioning better.
2. **The dataset scale claim is off.** "≈106,000 images / 55,000 instances"
   counts RDD2022 (not used) and SDNET2018 (no instances). Real detection
   instances: **4,189** (NEU-DET) + **3,542** (GC10-DET).
3. **§4.2 formula inconsistency.** `F' = F⊙g⊙s + F` gives `2F` when `g=s=1`, not
   the identity. Either drop the `+F` or say it reduces to identity as gates→0.
4. **CLIP text dim is 512, not 256.** `T` is declared `N×256`; a projection is
   needed, or `d` corrected.

See `phase-notes/PHASE-1.md` §4 for the full numbers.
