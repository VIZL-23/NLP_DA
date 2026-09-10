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
| **0** | Foundation | Done | venv + CUDA verified, Ultralytics installed, datasets audited, repo hygiene |
| **1** | Data pipeline | Done | VOC→YOLO conversion, both split protocols, 60-phrase text corpus |
| **2** | Walking skeleton | Done | TG-FEM registered + placed at P3/P4/P5, trains end to end, boxes emitted |
| **1b** | GC10-DET + DeepCrack | Done | Both converted; corpus extended to 170 phrases / 17 classes |
| **3** | Baselines | Done | stock 0.7717 · CBAM 0.7372 · YOLO-World-S 0.0394 |
| **4** | Language branch | Done | Frozen CLIP text encoder + CoOp learnable context tokens, wired via a `forward_pre_hook` |
| **5** | TG-FEM | Done | Real math (proj → region-text attention → dual gating → residual) + WorldDetect open-vocab head |
| **6** | Training | **Done, on GPU, real CLIP** | 5 runs at the report's schedule. `pretrained_clip_loaded: true` throughout. **The result is negative** — see below |
| **7** | Evaluation & ablations | **Done** | Ablations (a), (d), (g) + phrase augmentation, all on GPU. Text interface characterised. GC10 specialist 0.6379; DeepCrack run added |
| **8** | Report & demo | Scaffolded | `scripts/demo.py` verified end to end; the report itself needs restructuring around the Phase 6 result |
| **9** | Specialists + app | **Done** | 16-class steel merge **rejected**, 7-class steel+concrete merge **shipped**; concrete data grown 255 -> 8,126 images; single-class models shown not to be text-guided |

\* **Verified by actually running the code, not numerically yet.** All of
Phases 4-8 have been built and executed on a machine with no GPU and no live
connection to CLIP's weight host - every model variant, both closed and
open-vocab protocols, the ablation dispatcher, and both eval scripts have
been run end to end and produce correct output. Two real bugs were found and
fixed this way (a crash in the `cbam_worlddetect` ablation, a dead CLI flag)
- neither would have been caught by code review alone. None of this is an
accuracy number, though: `TextEncoder` (in `src/tgfem/language.py`) exposes
`pretrained_loaded`; if it's `False`, the run used a randomly-initialised
text tower instead of real CLIP weights. Check it's `True` before trusting
anything past Phase 3. The `gc10` dataset path and real DeepCrack data are
still completely untested (no data available where this was built). See
`phase-notes/PHASE-4.md` §5 and `PHASE-6.md`/`PHASE-7.md` for what's left
and the commands to run once a GPU is available.

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

### Baseline results (Phase 3, NEU-DET test split)

| baseline | init | mAP@0.5 | role |
|---|---|---|---|
| YOLOv11n | pretrained | **0.7717** | absolute ceiling |
| YOLOv11n + CBAM | scratch | **0.7372** | attention control |
| YOLOv11n | scratch | **0.7069** | like-for-like retention anchor |
| YOLO-World-S zero-shot | — | **0.0394** | open-vocab competitor |

Custom-module variants (CBAM, TG-FEM) **must** train from scratch — inserting
layers at 5/8/13 shifts the state-dict indices, so `yolo11n.pt` no longer maps
onto them. Comparing them against the pretrained number measures pretraining,
not architecture, so a from-scratch anchor was added.

Isolating the two effects:
- COCO pretraining is worth **+6.5** points (0.7717 − 0.7069)
- Attention is worth **+3.0** points (0.7372 − 0.7069)

**Targets for TG-FEM:**

| target | value | why |
|---|---|---|
| retention floor | ≥ 0.6769 | 3 pts below the from-scratch anchor |
| **the bar that matters** | **> 0.7372** | must beat CBAM — anything less is explained by attention alone |
| zero-shot gain | ≥ 0.0894 | +5 pts over YOLO-World-S |

> **Falsifiable prediction.** Attention's gain concentrates on `crazing`
> (+0.094) and `rolled-in_scale` (+0.070) — the texture-confusable classes, and
> the NEU-DET held-out pair. If text conditioning works, TG-FEM's gain over CBAM
> should appear on these same classes. Report per-class AP, not just the mean.
> See `phase-notes/PHASE-3.md` §3.

### Phase 6 results (NEU-DET test split) - THE RESULT IS NEGATIVE

| variant | init | head | gates | mAP@0.5 |
|---|---|---|---|---|
| stock | pretrained | Detect | - | 0.7717 |
| CBAM | scratch | Detect | image | 0.7372 |
| **TG-FEM** | scratch | WorldDetect | **text** | **0.7268** |
| CBAM+WorldDetect [abl g] | scratch | WorldDetect | image | 0.7240 |
| TG-FEM identity [abl a] | scratch | WorldDetect | none | 0.7183 |
| stock scratch | scratch | Detect | - | 0.7069 |

```
ablation (g)  TG-FEM - CBAM+WorldDetect = +0.0028   <- within noise
ablation (a)  TG-FEM - identity          = +0.0085   <- within noise
```

Per-epoch validation swung by +-0.10 during training, so neither delta is
distinguishable from noise. **With the head held constant, there is no evidence
that text-derived gates beat image-derived gates.**

**The falsifiable prediction failed.** It said TG-FEM's gain should concentrate
on `crazing` and `rolled-in_scale`; TG-FEM is *worse* on both
(-0.0350, -0.0139).

**Zero-shot fails outright.** On the openvocab test split both held-out classes
score **0.000 AP** despite 688 and 628 instances. The `mAP50 = 0.1409` in
`results/phase6_neu_openvocab_tgfem.json` is the mean over four classes where
only the *seen* class `patches` (4 instances) scores - do not compare it to
YOLO-World-S's 0.0394.

**This is not a wiring bug.** The negative control swaps in absurd prompts and
performance collapses 16x (0.3994 -> 0.0243), so the text path is genuinely live.

Diagnosis: `WorldDetect`'s open-vocabulary ability comes from large-scale
region-text pretraining (YOLO-World: 27M grounding pairs). Training from scratch
on 1,075 images cannot produce it. Architecture sound, data-starved by ~4 orders
of magnitude. The risk register anticipated this (row 1).

See `phase-notes/PHASE-6.md` for full detail.

### Priority ablations
Of the seven planned, three are load-bearing — run these first:
**(a)** TG-FEM removed (identity check) · **(d)** context-token count M
· **(g)** gates driven by image vs by text — *this one is the paper*.

---

## Running the demo app

A local web app: pick a model, upload an image (or pick a bundled sample), type
a description of the defect, get boxes back. This is the project deliverable -
the thing the whole pipeline exists to produce.

```bash
python app/server.py
```

Then open **http://localhost:8000**. Nothing is deployed and nothing is fetched
over the network; it runs entirely on your machine, so a demo cannot be broken
by the venue's wifi.

If a checkpoint has not been trained yet the app **skips it and starts anyway**,
so it is usable mid-project with whatever is ready.

### Options

```bash
python app/server.py --device cpu        # force CPU (e.g. while the GPU is training)
python app/server.py --only neu gc10     # load a subset of the models
python app/server.py --port 8080         # if 8000 is taken
```

The model list, each vocabulary and each sample set all come **from the server**,
which reads them from the checkpoints themselves. Adding a model is one entry in
`MODELS` in `app/server.py`; the frontend needs no change.

### The two models, and why these

| model | classes | domain | mAP@0.5 |
|---|---|---|---|
| `neu_crack` | 7 | steel strip + concrete cracks | 0.6867 |
| `gc10` | 10 | galvanised steel sheet | 0.6379 |

Merging datasets was measured **both ways**, and the result depends entirely on
how similar the classes are:

| merge | outcome | shipped? |
|---|---|---|
| NEU + GC10 (steel + steel, 16 classes) | GC10 0.598 vs 0.638, NEU 0.704 vs 0.720 | no |
| NEU + crack (steel + concrete, 7 classes) | NEU 0.7264 vs 0.7205, crack 0.448 vs 0.465 | **yes** |

Two steel datasets share near-synonymous classes (`inclusion` exists in both),
so the model has to separate genuinely confusable categories and every class
gets harder. Steel and concrete are not confusable, so that merge costs nothing
measurable - and it is the only configuration where a concrete query means
anything (see below).

`neu_crack` **replaces** the separate NEU and crack specialists, which stay in
`runs/` as the evidence for this decision.

### Which checkpoints it uses, and why

Every entry is a **phrase-augmented** checkpoint. This is **not** the
highest-mAP choice, and that is deliberate:

| checkpoint | mAP@0.5 | phrasings it responds to |
|---|---|---|
| `phase6_neu_closed_tgfem` | **0.7268** (best) | **2 of 6** |
| `phase6_neu_closed_tgfem_phraseaug` | 0.7205 | **5 of 6** |

Choosing by mAP would give a model that only answers to six memorised strings -
so in a live demo, where a user types in their own words, it returns nothing and
the project looks broken. The 0.006 mAP difference is well inside noise
(`phase-notes/PHASE-7.md` section 2); the phrasing difference is not.

### Sample images

```bash
python scripts/make_samples.py --dataset neu_crack  # -> app/samples/neu_crack/
python scripts/make_samples.py --dataset gc10      # -> app/samples/gc10/
```

One folder per model, so a steel checkpoint is never demoed on a concrete image
by accident. Samples are drawn from the **held-out test split** by fixed seed -
**not** selected by how well the model scores on them. "Random sample, fixed
seed, from data the model never saw" is a defensible answer when someone asks
how they were chosen; cherry-picked best cases are not.

### Known limits, worth stating up front in a demo

These are measured, not guesses (`phase-notes/PHASE-7.md` sections 3-4):

- **Only the trained classes work.** Held-out classes score 0.000 AP - there is
  no working zero-shot behaviour.
- **Roughly 1 in 6 genuinely novel phrasings works.** Wording close to the
  listed phrases is reliable; wording with no lexical overlap usually returns
  nothing. The UI shows the supported phrases for this reason.
- **Each model expects its own domain.** The steel models have only ever seen
  200x200 grayscale steel imagery; a cracked wall belongs to the `crack` model.
  Picking the wrong model is the most likely way to make the demo look broken.
- **A single-class model would ignore the query text entirely.** The retired
  crack specialist returned the same boxes for `banana` as for a real query,
  because with one class its contrastive head never had to separate one text
  embedding from another. This is why `neu_crack` ships instead
  (`phase-notes/PHASE-9.md` sections 4-5). Both shipped models reject `banana`.
- A wrong-class or nonsense query returning **nothing is correct behaviour**, and
  is worth demonstrating deliberately: it shows the model discriminating rather
  than boxing whatever text it is given.

---

## Repository layout

```
app/
  server.py            local FastAPI app (the deliverable)
  static/index.html    single-page UI
  samples/             bundled test-split images (generated)
scripts/
  check_env.py         toolchain + dataset audit (run this first)
  prepare_neu_det.py   VOC→YOLO, splits, dataset YAMLs, geometry stats
  prepare_gc10.py      GC10-DET VOC→YOLO + splits + YAMLs
  prepare_deepcrack.py DeepCrack mask→box (connected components) + YAMLs
  verify_labels.py     draws converted boxes back onto images
  build_prompts.py     text-corpus validation + protocol filtering
  train_baseline.py    Phase 3 baselines (stock / stock_scratch / cbam)
  eval_yoloworld.py    Phase 3 YOLO-World-S zero-shot baseline
  train_skeleton.py    Phase 2 walking-skeleton gate check (identity TG-FEM)
  train_tgfem_gate.py  Phase 5 gate check (real TG-FEM + language branch)
  train_tgfem.py       Phase 6 training (tgfem / tgfem_identity / cbam_worlddetect)
  run_ablations.py     Phase 7 ablation dispatcher (resumable)
  eval_tgfem.py        Phase 7 negative control + cross-variant comparison
  eval_deepcrack.py    Phase 7 DeepCrack OOD zero-shot evaluation
  demo.py              Phase 8 text-query inference demo
  probe_wording.py     Phase 7 wording-robustness probe (graded paraphrases)
  probe_generalisation.py  Phase 7 corpus-absent phrasings: coverage vs semantics
  prepare_combined.py  NEU + GC10 merged into one 16-class dataset
  make_samples.py      bundle demo samples from the test split (fixed seed)
src/tgfem/
  module.py            TGFEM - real math (Phase 5)
  language.py          TextEncoder, ContextTokenLearner, TextConditioner (Phase 4)
  detection_model.py   TGFEMModel - WorldDetect text threading (Phase 5)
  trainer.py           TGFEMTrainer - optimiser/checkpoint wiring (Phase 5)
  data.py              canonical taxonomies + corpus phrase selection
  inference.py         DefectDetector - framework-free, backs the app and demo
cfg/
  yolo11-tgfem.yaml            the model (TG-FEM gates + WorldDetect head)
  yolo11-tgfem-ablation-a.yaml ablation (a) - identity mode, same param budget
  yolo11-cbam.yaml             Phase 3 closed-vocab CBAM baseline (stock Detect)
  yolo11-cbam-worlddetect.yaml ablation (g) - CBAM gates + WorldDetect head
prompts/
  defect_corpus.json   170 natural-language defect descriptions, 17 classes  (DRAFT)
datasets/              source data + generated *-yolo/ dirs (gitignored)
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
| crescent_gap / water_spot / punching_hole | 1420 | **30% (rejected)** |

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

1. **Constraint (ii) — attached to the wrong dataset.** The report claims
   defects are "<2% of image pixels, aspect ratios beyond 10:1". False for
   NEU-DET (median box **11.8%**, only 3.2% above 10:1) — but **true** for the
   others. Reframe as **three scale regimes**:

   | dataset | median box area | <2% area | >10:1 |
   |---|---|---|---|
   | NEU-DET | 11.79% | 6.9% | 3.2% |
   | GC10-DET | 3.49% | 39.6% | 10.7% |
   | DeepCrack | 1.11% | — | — |

   This is stronger than the original claim *and* better justifies multi-scale
   P3+P4+P5 conditioning.
2. **The dataset scale claim is off.** "≈106,000 images / 55,000 instances"
   counts RDD2022 (not used) and SDNET2018 (no boxes at all). Real detection
   instances: **4,189** (NEU-DET) + **3,541** (GC10-DET) + **1,864** (DeepCrack).
3. **§4.2 formula inconsistency.** `F' = F⊙g⊙s + F` gives `2F` when `g=s=1`, not
   the identity. Either drop the `+F` or say it reduces to identity as gates→0.
4. **CLIP text dim is 512, not 256.** `T` is declared `N×256`; a projection is
   needed, or `d` corrected.
5. **Corpus size.** §4.1 says "approximately 60 defect descriptions"; it is now
   **170** across 17 classes.
6. **DeepCrack boxes are loose by construction** — median fill ratio 16.6%.
   State it as a known limitation of box-based crack detection.

See `phase-notes/PHASE-1.md` §4 and `PHASE-1b.md` §3–4 for the full numbers.
