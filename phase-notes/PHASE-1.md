# Phase 1 — Data Pipeline

**Status:** Complete
**Goal:** Convert NEU-DET into a training-ready YOLO dataset, build both split protocols, and author the natural-language defect corpus.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Convert Pascal VOC XML → YOLO format | Done — 1800 images, 4189 boxes |
| Fix the Phase 0 off-by-one annotation mismatch | Done — resolved by filename pairing |
| Build closed-vocabulary split (80/10/10) | Done |
| Build open-vocabulary split with leakage guard | Done — leakage check passes |
| Author the ~60-phrase defect corpus | Done — 60 phrases (**draft, needs review**) |
| Visually verify box correctness | Done — see `assets/label_check.png` |

---

## 2. Conversion results

```
datasets/neu-det-yolo/
├── images/                  1800 .jpg  (flat pool)
├── labels/                  1800 .txt  (YOLO format, global 6-class indices)
├── closed_train.txt         \
├── closed_val.txt            |  image-path manifests
├── closed_test.txt           |
├── openvocab_train.txt       |
├── openvocab_val.txt         |
├── openvocab_test.txt       /
├── neu_closed.yaml          Ultralytics data config
└── neu_openvocab.yaml       Ultralytics data config
```

| Metric | Value |
|---|---|
| Images converted | 1800 / 1800 |
| Boxes written | 4189 |
| Degenerate boxes (xmax≤xmin) | 0 |
| Annotations without a matching image | 0 |

**Boxes per class:** inclusion 1011, patches 881, crazing 689, rolled-in_scale 628, scratches 548, pitted_surface 432.

### The off-by-one is resolved
Phase 0 found 1439/1440 train and 361/360 val annotations. The converter indexes
images by **filename stem across both splits** rather than trusting the folder
layout, so the misplaced XML pairs correctly. All 1800 annotations matched.

### Conversion maths verified
Hand-checked `scratches_1` against its source XML
(xmin=26, ymin=12, xmax=43, ymax=171, 200×200):
`cx=0.1725, cy=0.4575, w=0.085, h=0.795` — exact.
Visual check in `assets/label_check.png` confirms boxes land on the defects.

---

## 3. Split protocols

### Closed vocabulary — all 6 classes, stratified 80/10/10

| Subset | Images |
|---|---|
| train | 1434 |
| val | 176 |
| test | 190 |

Stratified by *class signature* (the sorted set of classes in an image), so
multi-class images stay proportionally distributed. Seed 42.

### Open vocabulary — `crazing` and `rolled-in_scale` held out

| Subset | Images |
|---|---|
| train | 1075 |
| val | 125 |
| test | 600 |

**Leakage guard (important).** 123 images contain more than one class. If a
"seen-class" training image also contained a `crazing` object, the model would
see a held-out class during training and the zero-shot claim would be invalid.

The splitter therefore applies **strict exclusion**: any image containing even
one held-out object goes to the test pool and is never used for training.
An automated assertion re-verifies this on every run:

```
leakage check : PASS (no held-out class in train/val)
```

Seen-class distribution in openvocab train: inclusion 341, patches 301,
scratches 269, pitted_surface 267.

### Design note: one label pool, global indices
Both protocols share the same label files, using stable global 6-class indices.
Protocols differ only in *which images* the manifests select. This avoids
maintaining two divergent label sets. In the openvocab YAML the held-out classes
are still declared in `names` — they simply have zero training examples.

---

## 4. FINDING: the problem statement's constraint (ii) does not hold for NEU-DET

The report's problem statement asserts that target defects:

> "occupy under 2% of image pixels, exhibit aspect ratios beyond 10:1, and are
> low-contrast against a textured background"

Measured across all 4189 boxes, this is **not true of NEU-DET**:

| class | n | median area % | median aspect | boxes >10:1 | boxes <2% area |
|---|---|---|---|---|---|
| crazing | 689 | 21.1 | 1.88 | 0% | 0% |
| inclusion | 1011 | 4.1 | 2.75 | 0% | 23% |
| patches | 881 | 9.3 | 1.50 | 0% | 3% |
| pitted_surface | 432 | **55.5** | 1.42 | 0% | 0% |
| rolled-in_scale | 628 | 12.3 | 1.46 | 0% | 1% |
| scratches | 548 | 7.5 | **7.02** | **24%** | 4% |
| **overall** | 4189 | **11.8** | — | **3.2%** | **6.9%** |

**What is actually true:**
- The median box covers **11.8%** of the image, not under 2%.
- `pitted_surface` boxes cover **more than half the image** on average.
- Only **`scratches`** is genuinely elongated (median 7:1, 24% above 10:1).
- Only **`inclusion`** has a meaningful small-object population (23% under 2%).

**Why this matters.** The "<2% area, >10:1 aspect" characterisation describes
**thin-crack segmentation datasets** (CRACK500, DeepCrack) — not NEU-DET steel
defects. Two parts of the report currently rest on it:

1. **Problem statement, constraint (ii)** — factually incorrect as written.
2. **Table 3, the neck-placement justification** — argues semantic conditioning
   belongs at P3 (80×80) *because* "thin defects occupy under 2% of pixels".
   On NEU-DET that premise fails, so the stated rationale for P3 placement loses
   its evidential basis.

**Recommended fixes (pick one):**

- **(A) Rescope the constraint per class.** State that NEU-DET spans an extreme
  *scale range* — from `inclusion` (median 4.1%) to `pitted_surface` (55.5%) —
  and that `scratches` alone carries the high-aspect-ratio challenge. This is
  defensible, still a genuine difficulty, and matches the data. Multi-scale
  conditioning (P3+P4+P5) is arguably better justified by a **14× spread in
  object area** than by uniform tininess.
- **(B) Keep the original claim but attach it to the crack datasets** (CRACK500 /
  DeepCrack), and state explicitly that NEU-DET tests a different regime.

Option (A) is the smaller edit and keeps the architecture argument intact.

**This must be corrected before DA2 submission** — the numbers above are
reproducible from our own data, so an examiner could check them.

---

## 5. Other data facts worth recording

- **All images are exactly 200×200 grayscale** (`depth=1`). Training at 640×640
  is a **3.2× upscale**, which adds no information. Worth stating honestly; it
  also means the ">30 FPS" budget is generous here.
- **123 images contain more than one class** — the dataset is not single-label
  per image, which is what made the leakage guard necessary.
- **81 boxes carry `difficult=1`.** We currently **include** them
  (`INCLUDE_DIFFICULT = True`), matching NEU-DET convention. Flip the flag in
  `scripts/prepare_neu_det.py` to test sensitivity.
- Class imbalance is mild (432–1011 boxes per class, ~2.3× spread).

---

## 6. The defect description corpus (NLP deliverable)

`prompts/defect_corpus.json` — **60 phrases, 10 per class**, organised in five tiers:

| tier | n | purpose |
|---|---|---|
| bare | 6 | class name alone — the zero-effort baseline prompt |
| natural | 12 | how an inspector would actually phrase it |
| visual | 18 | appearance-grounded (shape, texture, contrast) |
| material | 12 | grounded in the steel/rolling process |
| alias | 12 | domain synonyms and alternative industry terms |

The tiering is deliberate: it gives **ablation (d)** — hand-written prompts
(M=0 context tokens) vs learned context tokens — a structured axis to vary,
rather than an undifferentiated bag of strings.

Validation (`scripts/build_prompts.py`) enforces: exactly one `bare` phrase per
class, known tiers only, non-empty text, and **no phrase shared between two
classes** (which would make the contrastive target ambiguous).

Protocol filtering is built in — under `openvocab`, the held-out classes
contribute **0** phrases to the training vocabulary (60 → 40).

> **REVIEW NEEDED.** The corpus is marked `0.1-draft`. The wording was authored
> from general metallurgical description, not from an inspection manual. Before
> DA2, the team should verify the domain terminology — particularly the `material`
> and `alias` tiers — against an authoritative source, since the whole G4 claim
> ("prompt design for the structural-inspection vocabulary does not exist") rests
> on this corpus being credible.

---

## 7. Files created

```
scripts/
├── prepare_neu_det.py     VOC→YOLO conversion, splits, YAML generation, stats
├── verify_labels.py       draws converted boxes back onto images
└── build_prompts.py       corpus validation + protocol filtering
prompts/
└── defect_corpus.json     the 60-phrase corpus  (DRAFT)
phase-notes/assets/
└── label_check.png        visual verification grid
```

**Reproduce everything:**
```bash
python scripts/prepare_neu_det.py
python scripts/verify_labels.py
python scripts/build_prompts.py --protocol openvocab --list
```

`datasets/neu-det-yolo/` is gitignored — it is fully derived from
`datasets/NEU-DET` and must be rebuilt rather than committed.

---

## 8. Gate check — can we proceed to Phase 2?

| Criterion | Status |
|---|---|
| All annotations converted, none orphaned | Pass |
| Conversion maths verified numerically and visually | Pass |
| Both protocols built, sizes sane | Pass |
| Open-vocab leakage check automated and passing | Pass |
| Ultralytics-loadable data YAMLs generated | Pass |
| Text corpus authored and validated | Pass (draft wording) |

**Phase 1 gate: PASSED.**

---

## 9. Carried forward

1. **Report correction (§4 above)** — constraint (ii) must be rewritten. Blocking for DA2, not for Phase 2.
2. **Corpus wording review** — verify domain terminology with an authoritative source.
3. **`datasets/loader.py` has uncommitted edits** from before Phase 0; `scripts/check_env.py` now supersedes it. Decide whether to keep or remove.
4. Nothing blocks **Phase 2 (walking skeleton)**: register a no-op module in
   Ultralytics, train 2 epochs on a tiny subset, confirm boxes come out.
