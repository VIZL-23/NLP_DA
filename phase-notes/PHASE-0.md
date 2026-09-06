# Phase 0 — Foundation

**Status:** Complete
**Goal:** Get a reproducible, isolated environment running with GPU acceleration, and establish exactly what data we have before any modelling work.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Verify GPU / CUDA toolchain | Done |
| Isolate the project environment | Done — `.venv` created |
| Install Ultralytics (our build base) | Done — v8.4.142 |
| Repo hygiene (`.gitignore`) | Done |
| Audit both datasets | Done — issues found, logged below |
| End-to-end smoke test | Done — YOLOv11n inference on a real NEU-DET image |

---

## 2. Hardware & software baseline

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6.0 GB (6141 MiB) |
| Driver | 592.82 |
| Python | 3.10.0 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| CUDA available | Yes |
| Ultralytics | 8.4.142 |

**Smoke test result:** `yolo11n.pt` loaded and ran inference on
`datasets/NEU-DET/train/images/crazing/crazing_1.jpg`.
Peak VRAM during inference: **69 MB** — negligible, large headroom for training.

---

## 3. Environment decision: `--system-site-packages` venv

The machine's global Python already carries the **VITSIH robotics stack**
(`eclipse-zenoh`) plus a working CUDA build of torch.

Two problems to avoid:
1. Installing ML packages globally could break the VITSIH project.
2. A fully isolated venv would force a ~2.5 GB re-download of torch.

**Solution:** `python -m venv .venv --system-site-packages`

- The venv *sees* the global torch/numpy/opencv → no re-download.
- Anything we `pip install` lands **inside** the venv → global env stays clean.

**Always activate before working:**
```bash
.venv/Scripts/activate
```

---

## 4. Dataset audit

### NEU-DET — our primary detection dataset

```
datasets/NEU-DET/
├── train/
│   ├── images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}/  (240 each)
│   └── annotations/   *.xml  (Pascal VOC)
└── validation/
    ├── images/{...same 6 classes...}/  (60 each)
    └── annotations/   *.xml
```

| Split | Images | Annotations | Status |
|---|---|---|---|
| train | 1440 | 1439 | **off by one** |
| validation | 360 | 361 | **off by one** |
| **TOTAL** | **1800** | **1800** | totals match |

**Finding:** totals are correct (1800/1800, the standard NEU-DET size), but one
XML sits in the wrong split folder. **Action deferred to Phase 1** — the
conversion script will pair annotations to images by filename rather than
trusting the folder layout, which resolves this automatically.

**Structural note:** images are nested in **per-class subfolders** while
annotations are **flat**. The Phase 1 converter must handle this asymmetry.

### SDNET2018 — concrete, supporting role only

| Surface | Images | Cracked | Non-cracked |
|---|---|---|---|
| Decks | 13,620 | 2,025 | 11,595 |
| Pavements | 24,334 | 2,608 | 21,726 |
| Walls | 18,138 | 3,851 | 14,287 |
| **TOTAL** | **56,092** | 8,484 | 47,608 |

**Critical finding:** SDNET2018 has **classification labels only — no bounding
boxes.** It is organised as `Cracked/` vs `Non-cracked/` folders.

Consequences:
- It **cannot** be used for detection training or mAP evaluation.
- Valid uses: backbone warm-up, and as a source of **hard negatives**.
- The report must not imply these 56,092 images are detection data. This matches
  the limitation already noted for ref [1] in the literature table.

**Therefore NEU-DET is the working dataset for Phases 1–7.**
Class imbalance is also worth noting: only ~15% of SDNET2018 is cracked.

---

## 5. Files created

```
NLP_DA/
├── .gitignore              # venv, weights, runs/, caches, phase-notes/
├── requirements.txt        # ultralytics + pinned transitive deps
├── scripts/
│   └── check_env.py        # re-runnable environment + dataset audit
└── phase-notes/            # (gitignored) these documents
    └── PHASE-0.md
```

**Re-run the audit any time:**
```bash
.venv/Scripts/python.exe scripts/check_env.py
```

---

## 6. Open issue carried forward

**The datasets are committed to git (59,693 tracked files, 667 MB).**

We chose **not** to untrack them in Phase 0. Reasoning:

- The data is already in git *history*, so untracking now would **not** shrink
  the repository — only a history rewrite (`git filter-repo`) would, and that is
  disruptive to coordinate across the team.
- `git rm -r --cached datasets/` would delete the files from a teammate's working
  tree on their next `pull`, causing real data loss for them.
- 667 MB is within GitHub's tolerance. The only genuine cost is slower clones due
  to the high file *count*.

**Decision: leave `datasets/` tracked.** The new `.gitignore` prevents the
*future* bloat that actually matters — model weights, `runs/`, and caches.

Revisit only if the team agrees to coordinate a history rewrite.

---

## 7. Gate check — can we proceed to Phase 1?

| Criterion | Status |
|---|---|
| CUDA-enabled torch working | Pass |
| Ultralytics installed and importable | Pass |
| YOLOv11n runs inference on project data | Pass |
| Datasets located, counted, understood | Pass |
| Repo protected from artefact bloat | Pass |

**Phase 0 gate: PASSED.** Proceed to Phase 1 (data pipeline).

---

## 8. What Phase 1 will do

1. Convert NEU-DET Pascal VOC XML → YOLO `.txt` format (pairing by filename,
   which fixes the off-by-one split issue).
2. Build the two split protocols:
   - **Open-vocab split** — train on 4 classes, hold out `crazing` and
     `rolled-in_scale` for text-only querying.
   - **Standard split** — for the closed-vocabulary baseline.
3. Write the **~60-phrase defect description corpus** (the NLP deliverable).
4. Visually verify boxes render correctly on sample images.
