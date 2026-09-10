# Phase 9 — Specialists, a real concrete dataset, and a multi-model app

Three things settled in this phase, all of them by measurement:

1. **One combined model loses to per-dataset specialists.** Ship specialists.
2. **The concrete model had 255 training images.** It now has 8,126.
3. **The app therefore needs a model picker**, not a single checkpoint.

---

## 1. Specialists beat the 16-class combined model

Phase 7 trained NEU+GC10 together into one 16-class model, hoping a richer
vocabulary would come free. It does not. Every number below is on the same
held-out test splits.

### GC10 classes

| class | specialist | inside combined | diff |
|---|---|---|---|
| crease | 0.460 | 0.422 | +0.038 |
| crescent_gap | 0.943 | 0.908 | +0.035 |
| gc10_inclusion | 0.164 | 0.131 | +0.033 |
| oil_spot | 0.571 | 0.571 | -0.001 |
| punching_hole | 0.978 | 0.966 | +0.011 |
| rolled_pit | 0.197 | 0.180 | +0.017 |
| silk_spot | 0.621 | 0.525 | +0.097 |
| waist_folding | 0.855 | 0.863 | -0.009 |
| water_spot | 0.672 | 0.671 | +0.000 |
| welding_line | 0.918 | 0.744 | +0.174 |
| **mean** | **0.638** | **0.598** | **+0.040** |

### NEU classes

| class | specialist | inside combined | diff |
|---|---|---|---|
| crazing | 0.366 | 0.396 | -0.029 |
| inclusion | 0.757 | 0.738 | +0.019 |
| patches | 0.881 | 0.872 | +0.009 |
| pitted_surface | 0.824 | 0.807 | +0.017 |
| rolled-in_scale | 0.604 | 0.593 | +0.011 |
| scratches | 0.891 | 0.820 | +0.071 |
| **mean** | **0.720** | **0.704** | **+0.016** |

8 of 10 GC10 classes and 5 of 6 NEU classes get **worse** when the datasets are
trained together. `welding_line` alone loses 17 points. On top of that the
combined model scored **0/6** on novel paraphrases against the NEU specialist's
1/6 (Phase 7 section 4), so the merge cost vocabulary robustness as well as mAP.

This is not surprising in hindsight: the two datasets are different steel
processes photographed under different conditions, and several classes are
near-synonyms across them (`inclusion` exists in both taxonomies, which is why
the GC10 one had to be renamed `gc10_inclusion`). Forcing one text-conditioned
head to separate them makes every class harder.

**Decision: ship three specialists.** The user picks a domain; the app exposes
the same 17-class vocabulary at strictly better accuracy. The combined model
stays in `runs/` as the evidence for this decision, and is not shipped.

---

## 2. The concrete model: 255 images → 8,126

`prepare_deepcrack.py` yields 255 images. That was always too small to claim
anything about concrete, and the 255-image run is kept only as the small-data
comparison point. It finished at **0.2466 mAP@0.5** (precision 0.360, recall
0.287) - by far the weakest model in the project, and the number this phase
exists to beat.

`scripts/prepare_crack_merged.py` converts a pooled public crack-segmentation
set (HuggingFace parquet, 4 files, 1,056 MB) into the same YOLO box format.

```
images converted : 10592
boxes written    : 20276  (avg 1.9/image)
  crack-free backgrounds (empty label): 1411
  row from a held-out source: 521
  mask with no component above min-area: 185

splits:  train 8126   val 881   test 1585
```

Source composition: Rissbilder 3822, CRACK500 3363, noncrack 1411, Volker 990,
DeepCrack 521, GAPS384 509, cracktree200 206, Sylvie 185, CFD 118, forest 118,
Eugen 55.

### Two decisions the composition forced

**The 1,411 `noncrack_*` rows are kept as background images.** They are
crack-free concrete with an all-zero mask. The first version of the converter
dropped them as "no boxes found" — that would have thrown away exactly the hard
negatives that stop the model boxing a blank wall. Ultralytics represents a
background image as an image with an empty label file, so that is what they get.
A crack-free row is a *label*, not a failure.

**The 521 `DeepCrack_*` rows are excluded.** DeepCrack is the held-out
out-of-distribution probe. Training on those rows would have quietly invalidated
every OOD number in the project — the probe would be reporting on data the model
had memorised. `--keep-source DeepCrack` overrides this deliberately.

### The mask threshold, verified rather than assumed

The masks are not clean binary — `np.unique` returns 15–17 grey levels. That
looked like it might silently produce zero boxes under the inherited `>127`
threshold, so it was checked on samples before converting 11k images:

| | fraction of pixels |
|---|---|
| `mask > 0` | 0.018 – 0.046 |
| `mask > 127` | 0.009 – 0.030 |

The extra levels are JPEG antialiasing on the crack edges, not classes. Both
thresholds produce boxes; `>127` takes the crack core and gives tighter boxes,
so it matches `prepare_deepcrack.py` and the two datasets stay comparable.

### The mask→box limitation, re-measured

| | |
|---|---|
| crack pixels per image | 3.55% (mean) |
| median box area | 3.71% of image |
| median box fill ratio | 22.2% |
| boxes filled < 25% | 55.4% |

Same story as DeepCrack: a diagonal crack cannot fill an axis-aligned rectangle,
so a box is mostly background by construction. Stated, not hidden.

Training: 60 epochs rather than 150. The dataset is ~32x larger than DeepCrack,
so the same number of gradient steps arrives far sooner, and 150 passes over
8,126 images does not fit in a night on a 4050.

---

## 3. The app now loads several models

`app/server.py` went from one hardcoded checkpoint to a `MODELS` registry:

| key | label | domain |
|---|---|---|
| `neu` | Steel · NEU-DET | hot-rolled steel strip, 6 classes |
| `gc10` | Steel · GC10-DET | galvanised steel sheet, 10 classes |
| `crack` | Concrete · cracks | concrete and pavement, 1 class |

- `/api/info` returns the model list, and each model's vocabulary and samples.
  The frontend hardcodes none of it — adding a checkpoint is one `MODELS` entry.
- `/api/sample/{model}/{name}` — samples are now **per model**
  (`app/samples/neu/`, `.../gc10/`, `.../crack/`), so a steel checkpoint is
  never demoed on a concrete image by accident. Path-traversal guard retained.
- `/api/detect` takes a `model` field.
- **A missing checkpoint is skipped, not fatal.** The app starts with whatever
  has finished training, which is the normal mid-project state.
- Switching model in the UI drops the current sample selection and any upload —
  a sample name from the previous model refers to a different folder on the
  server.

Verified end to end on CPU while the GPU was training: both loaded models return
boxes for their own samples, the traversal guard returns 404, and an unknown
model key returns a clean error rather than a stack trace.

---

## 4. The concrete specialist works — and is not text-guided

Trained: 60 epochs, 8,126 images, `pretrained_clip_loaded: true`.

| | mAP@0.5 | precision | recall |
|---|---|---|---|
| concrete specialist (8,126 img) | **0.4647** | 0.682 | 0.408 |
| DeepCrack specialist (255 img) | 0.2466 | 0.360 | 0.287 |

Those two numbers are on **different test splits**, so they are not a like-for-like
comparison. On the one shared test set — DeepCrack, which the specialist trained
on and the merged model never saw — the merged model scores **0.1927 zero-shot**
against the specialist's 0.2466 in-distribution. Reaching 78% of an
in-distribution model's score on a domain it has never seen is the honest read.

### The negative control failed

`eval_deepcrack.py` runs a nonsense prompt as a permanent control. It fired:

| query | mAP@0.5 |
|---|---|
| `crack` | 0.19262 |
| `a crack in the concrete surface` | 0.19267 |
| **`banana`** | **0.19268** |

Identical to four decimal places. Confirmed directly on individual images — the
same boxes come back for any query, with confidences differing by ~0.002:

```
crack sample, conf>=0.10
  a crack in the concrete surface     1 box  0.6967
  banana                              1 box  0.6941
  a happy elephant                    1 box  0.6940
```

Against a multi-class model on the same test, where the text plainly does work:

```
NEU (6 classes), scratches image
  scratches                           2 boxes  0.6631
  banana                              0 boxes
  a happy elephant                    0 boxes
```

**Cause: the vocabulary has one class.** The contrastive head was never asked to
separate one text embedding from another — "crack vs background" is the entire
discrimination the loss ever demanded — so the text contribution collapses into
a constant. This is structural, not a data problem: the 255-image DeepCrack
model shows the same behaviour, and it is predictable from `nc=1` alone.

**This is not a bug to fix in code.** It is a true statement about what a
single-class checkpoint does, so the app now states it: `DefectDetector`
exposes `text_discriminative` (read off the vocabulary size), `/api/info`
returns it, and selecting the crack model shows a warning that the text is
recorded but not used. Presenting it as text-guided would be a false claim
about the deliverable.

### The open decision

Making the crack model genuinely text-guided requires giving it something to
discriminate against — i.e. training crack **together with** the steel classes.
Section 1 showed that merging NEU+GC10 costs 1.6–4.0 mAP points, but those are
two *same-domain* steel datasets with near-synonymous classes. Crack is a
different domain entirely, so the interference may not transfer. That is a
testable hypothesis, not a known result, and it costs one ~6 h training run.

The trade-off is real either way: a 7-class NEU+crack model would probably lose
a little steel mAP, in exchange for being the only configuration where a text
query about concrete means anything. Left for the user to call.

---

## 5. Option A: joint NEU+crack training fixes the text interface, for free

The hypothesis in section 4 was that giving crack six steel classes to compete
against would make the text discriminate. It does, and it costs nothing.

`phase6_neu_crack_closed_tgfem_phraseaug` - 7 classes, 100 epochs, 3,584 train
images (NEU 1,434 + a seeded source-stratified 2,150-image crack subsample),
`pretrained_clip_loaded: true`. **Overall mAP@0.5 = 0.6867.**

### The negative control now passes

```
concrete image, conf >= 0.10
  a crack in the concrete surface     2 boxes   0.5592
  crack                               3 boxes   0.5402
  banana                              0 boxes
  a happy elephant                    0 boxes
  scratches on the steel surface      0 boxes    <- a TRAINED class, rejected

steel image (scratches), conf >= 0.25
  scratches on the steel surface      1 box     0.6779
  a crack in the concrete surface     0 boxes
  banana                              0 boxes
```

The `scratches`-on-concrete row is the load-bearing one. A model that returned
nothing for unfamiliar text would fail that test too - rejecting a *trained*
class because it does not match *this image* is cross-class discrimination, and
that is what the single-class model could not do at all.

### It costs nothing on steel

| class | joint | NEU specialist | diff |
|---|---|---|---|
| crazing | 0.430 | 0.366 | +0.064 |
| inclusion | 0.737 | 0.757 | -0.020 |
| patches | 0.912 | 0.881 | +0.031 |
| pitted_surface | 0.822 | 0.824 | -0.002 |
| rolled-in_scale | 0.577 | 0.604 | -0.027 |
| scratches | 0.880 | 0.891 | -0.010 |
| **NEU mean** | **0.7264** | **0.7205** | **+0.0059** |

| | joint | crack specialist | diff |
|---|---|---|---|
| crack | 0.4483 | 0.4647 | -0.0164 |

Steel is unchanged (+0.6 is inside the +/-0.10 per-class noise band established
in Phase 7; the honest claim is "no measurable cost", not "an improvement").
Crack loses 1.6 points **while training on 2,150 images instead of 8,126** - a
quarter of the data for a sixth of the score, which is a good trade and probably
recoverable by raising `--crack-train`.

### Why this contradicts section 1, and why both results stand

Section 1 found merging NEU+GC10 *hurt* both taxonomies. This section finds
merging NEU+crack hurts neither. The two are consistent: NEU and GC10 are two
photographs of **the same material** with near-synonymous classes (`inclusion`
exists in both taxonomies), so the model must separate genuinely confusable
categories. Steel and concrete are not confusable. Merging helps when the added
classes are far apart and hurts when they are close - which is a more useful
conclusion than either result alone.

### What ships

The app now loads `neu_crack` (7 classes) + `gc10` (10 classes). The `neu` and
`crack` specialists are retired from the app and kept in `runs/` as evidence.
The crack specialist scores 1.6 points higher on crack and is still the wrong
thing to ship: it ignores the query text entirely, and a text-guided detection
demo whose text does nothing is not the deliverable.

---

## 6. A real multi-class concrete model

Sections 4-5 solved the *text* problem for concrete by giving `crack` six steel
classes to compete against. It did not solve the *vocabulary* problem: the only
concrete question the app could answer was still "is there a crack".

Source: **Roboflow Universe, "Concrete defect detection" by SHM, CC BY 4.0** —
1,680 images, 3,565 boxes, 6 classes, already in YOLO format.
`scripts/prepare_concrete.py` renames the classes into the corpus's snake_case
(indices untouched, so no label file is rewritten) and writes manifests at the
dataset root.

### Choosing it was most of the work

Three candidates were rejected before this one, and the reasons generalise:

| dataset | why not |
|---|---|
| **GYU-DET** (11,123 imgs, 6 classes, YOLO, validated on YOLOv11) | 4608x3456 images - tens of GB, and the paper never states a size. Also "cracks significantly underrepresented", which is the class we already have. |
| **Roboflow `yolov11-iavvq`** (2.2k, 7 classes) | Four of its seven classes are literally named `0`, `1`, `2`, `3`. A class with no name has no text to query, which is fatal *here* specifically. It also had 0 generated versions, so nothing to download. |
| **CODEBRIM** (1,590 imgs, 5 classes) | Sound, but needs conversion and the archive is large. Kept as the fallback. |

The lesson worth keeping: for this project a dataset's **class names are part of
the data**. A taxonomy of integers is unusable no matter how good the boxes are.

### The audit came back clean

```
1,680 label files · 3,565 boxes
0 empty · 0 malformed · 0 out-of-range · 0 orphaned images
```

Better than expected for a community upload. The class balance is not:

| class | boxes | AP@0.5 |
|---|---|---|
| crack | 1,202 | 0.383 |
| efflorescence | 1,156 | 0.334 |
| scaling | 433 | 0.307 |
| spalling | 424 | **0.581** |
| exposed_reinforcement | 214 | 0.318 |
| **rust_stain** | **136** | **0.031** |

**Overall mAP@0.5 = 0.3256**, 150 epochs, `pretrained_clip_loaded: true`.

`rust_stain` was predicted to fail before training started, from box count alone
(`prepare_concrete.py` prints the warning), and it did. Note that AP does **not**
track box count beyond a floor: `spalling` has a third of `crack`'s boxes and
nearly twice its AP, because a spall is a compact high-contrast region while a
crack is a thin line in a loose box.

### Vocabulary phrasings had to be written

Six classes needed corpus entries before training could start - 10 phrasings
each across all five tiers (bare/natural/visual/material/alias), which is what
`--phrase-aug` resamples from. Without them `class_texts_for` has nothing to
look up and the model is not text-guided in any meaningful sense.

`crack` deliberately reuses the existing DeepCrack entry rather than getting a
duplicate: these *are* concrete cracks.

### Text discrimination passes, including against siblings

```
spalling image, conf >= 0.10
  spalling on the concrete surface        1 box   0.32
  efflorescence on the concrete surface   0 boxes    <- trained sibling class
  rust staining on the concrete surface   0 boxes    <- trained sibling class
  banana                                  0 boxes
  a happy elephant                        0 boxes
```

Rejecting a *trained sibling* is the strong form of the test - a model that
merely failed on unfamiliar words would also return nothing for `banana`.

### The confidence default was wrong, and it was measurable

Hit rate on the correct query, single-class test images:

| class | conf 0.25 | conf 0.10 |
|---|---|---|
| spalling | 6/6 | 6/6 |
| efflorescence | 5/6 | 6/6 |
| exposed_reinforcement | 3/3 | 3/3 |
| scaling | 2/6 | 6/6 |
| crack | 0/6 | 3/6 |
| rust_stain | 0/3 | 0/3 |
| **total** | **53%** | **80%** |

This model's scores run lower than the steel models' because its boxes are
looser and its imagery is field photography rather than lab micrographs. At the
app's global 0.25 default, `crack` found nothing at all and `scaling` found a
third of what it can - a working model presenting as broken.

`ModelSpec.default_conf` is therefore **per model** (steel 0.25, concrete 0.10),
returned by `/api/info` and applied by the UI on model switch. Measured, not
guessed: that table is the justification.

### Kept separate, deliberately

This model is **not** merged into `neu_crack`. Its classes sit on the same
material as `crack` and are plausibly confusable with it, which is structurally
the NEU+GC10 situation from section 1 that hurt both taxonomies. Training it
alone contains that risk, and the app's model picker turns it into a feature:
three genuine domains, 23 queryable phrases.

---

## What is still open

- **Decide on the 7-class NEU+crack experiment** (section 4). It is the only
  route to a text-guided concrete model, and it is one training run.
- `probe_wording.py` / `probe_generalisation.py` on the crack model would be
  **meaningless** — they measure which phrasings a model responds to, and this
  one responds identically to all of them. Run them only if a multi-class crack
  model gets trained.
- The false-positive rate on crack-free concrete is still worth measuring: the
  114 background images in the val split make it a real number rather than an
  anecdote. SDNET2018 is only worth downloading if that number comes back bad.
- `eval_deepcrack.py` had the same `device="0"` bug already fixed in
  `eval_tgfem.py`; it now shares `_resolve_device` from `tgfem.inference`.
