# Phase 1b — GC10-DET and DeepCrack

**Status:** Complete
**Goal:** Convert the two newly downloaded datasets, extend the text corpus to cover them, and quantify the mask→box limitation honestly.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Convert GC10-DET → YOLO | Done — 2280 images, 3541 boxes |
| Apply the pinyin→English mapping + data fixes | Done |
| Build GC10 closed + open-vocab protocols | Done — leakage check passes |
| Convert DeepCrack masks → boxes | Done — 537 images, 1864 boxes |
| Measure the mask→box limitation | Done — see §4 |
| Extend the text corpus to all 17 classes | Done — 170 phrases |
| Visual verification of both | Done |

---

## 2. GC10-DET conversion

```
datasets/gc10-det-yolo/
├── images/  2280 .jpg      ├── closed_{train,val,test}.txt
├── labels/  2280 .txt      ├── openvocab_{train,val,test}.txt
                            ├── gc10_closed.yaml
                            └── gc10_openvocab.yaml
```

| Metric | Value |
|---|---|
| Images converted | 2280 |
| Boxes written | 3541 |
| Annotations without an image | 0 |
| Degenerate boxes | 0 |
| Dropped (unmapped class `d`) | 1 |

### The three source traps, handled

1. **Class names are Chinese pinyin.** The XML says `3_yueyawan`; the folder says
   `crescent_gap`. Mapping lives in `src/tgfem/data.py::GC10_NAME_MAP`, derived
   empirically by cross-referencing every XML against its image folder.
2. **`10_yaozhe` and `10_yaozhed` are the same class** (typo variant, 11 + 131
   objects) — merged to `waist_folding`. A corrupt class literally named `d`
   (1 object) is dropped.
3. **The folder is NOT the label.** Measured: **73 images have a folder name
   that is not even among the classes their XML annotates.** GC10 images are
   multi-label and the folder records only the *dominant* defect. Converting by
   folder would produce silently wrong labels for a large fraction of the set.

### Splits

| Protocol | train | val | test |
|---|---|---|---|
| closed (10 classes) | 1800 | 214 | 266 |
| openvocab (3 held out) | 1406 | 166 | 708 |

Held out: **`crescent_gap`, `punching_hole`, `welding_line`** — see README
"Locked decisions" for why this triple and not another.

**Leakage check: PASS.** Seen-class retention in the openvocab *train* split:
silk_spot 85%, waist_folding 90%, gc10_inclusion 87%, water_spot 87%,
crease 83%, oil_spot 82%, rolled_pit 61%.

*(These read ~10 points below the figures quoted in the README because those
counted the whole seen pool; a further 10% of it goes to val.)*

### Namespacing applied
`7_yiwu` maps to **`gc10_inclusion`**, not `inclusion`, keeping it distinct from
NEU-DET's class of the same name. NEU-DET's stays plain `inclusion` inside its
own label space — unambiguous, since each dataset has its own YAML and indices.
Must be renamed `neu_inclusion` if the two are ever merged into one run.

---

## 3. GC10-DET rescues the report's constraint (ii)

Phase 1 found that the report's claim — defects "occupy under 2% of image
pixels, exhibit aspect ratios beyond 10:1" — is **false for NEU-DET**. GC10-DET
changes the picture:

| dataset | median box area | boxes < 2% area | boxes > 10:1 |
|---|---|---|---|
| NEU-DET | 11.79% | 6.9% | 3.2% |
| **GC10-DET** | **3.49%** | **39.6%** | **10.7%** |
| DeepCrack | 1.11% | (see §4) | — |

**The original claim was not wrong — it was attached to the wrong dataset.**
GC10-DET genuinely is a small-object, high-aspect-ratio regime, and DeepCrack
more so.

This gives the report a much stronger and fully honest framing: **three
datasets spanning three distinct scale regimes**

- **NEU-DET** — large-blob regime (median 11.8%, `pitted_surface` at 55%)
- **GC10-DET** — small-object regime (median 3.5%, 40% under 2%)
- **DeepCrack** — thin-crack regime (median 1.1%)

That is a far better justification for **multi-scale P3+P4+P5 conditioning** than
"defects are uniformly tiny" ever was, and every number is reproducible from our
own conversion.

---

## 4. DeepCrack: mask → box, and its measured limitation

DeepCrack ships **binary pixel masks**, not boxes. It is **eval-only** — report
§4.4 holds it out entirely for out-of-distribution zero-shot evaluation.

### Method
Naively taking one box per image would yield a near-full-image rectangle for a
long diagonal crack. Instead we run **connected-component analysis**
(`cv2.connectedComponentsWithStats`, 8-connectivity) and emit **one box per
crack segment**, dropping components under 20 px as noise.

| Metric | Value |
|---|---|
| Images converted | 537 (300 train + 237 test) |
| Boxes written | 1864 |
| Average boxes/image | 3.5 |

### The limitation, measured rather than hidden

| Metric | Value |
|---|---|
| Crack pixels per image | 3.54% (mean) |
| Median box area | 1.11% of image |
| **Median box fill ratio** | **16.6%** (crack px / box px) |
| Boxes filled < 25% | **66.3%** |

**A crack box is mostly background by construction** — a diagonal segment cannot
fill an axis-aligned rectangle. Visual check (`assets/label_check_deepcrack.png`)
confirms this: one long diagonal crack yields a huge loose box, while
connected components correctly splits a branching crack into tight segments.

**This belongs in the report.** It is a genuine limitation of box-based crack
detection, not a bug in our conversion — and stating it pre-empts the obvious
examiner question about why crack mAP is low.

Minor note: some DeepCrack images are watermarked stock photos.

---

## 5. Text corpus extended

`prompts/defect_corpus.json` is now **v0.2** — **170 phrases across 17 classes
and 3 datasets** (was 60 across 6).

| dataset | classes | held out |
|---|---|---|
| neu-det | 6 | crazing, rolled-in_scale |
| gc10-det | 10 | crescent_gap, punching_hole, welding_line |
| deepcrack | 1 | crack (OOD, always held out) |

Same five tiers (bare / natural / visual / material / alias), 10 phrases per
class. Validation passes, including the no-duplicate-phrase rule — GC10's
`inclusion` uses `"steel sheet inclusion"` as its bare tier so it cannot collide
with NEU-DET's.

> **Report correction:** §4.1 says "a corpus of approximately 60 defect
> descriptions". It is now **170**. The 60-figure needs updating.

> **REVIEW STILL NEEDED.** Wording was authored from general metallurgical
> description, not an inspection manual. The GC10 terms (`silk_spot`,
> `waist_folding`, `rolled_pit`) are translations of Chinese industry terms and
> deserve particular scrutiny, since gap G4 rests on this corpus being credible.

---

## 6. Files created / changed

```
src/tgfem/data.py            NEW - canonical taxonomies, one source of truth
scripts/prepare_gc10.py      NEW - GC10 VOC->YOLO + splits + YAMLs
scripts/prepare_deepcrack.py NEW - mask->box via connected components
scripts/verify_labels.py     generalised: --dataset {neu,gc10,deepcrack}
scripts/check_env.py         now imports taxonomies from tgfem.data
scripts/build_prompts.py     reports per-dataset grouping
prompts/defect_corpus.json   v0.2 - 170 phrases, 17 classes
phase-notes/assets/          label_check_{gc10,deepcrack}.png
```

**Reproduce:**
```bash
python scripts/prepare_gc10.py
python scripts/prepare_deepcrack.py
python scripts/verify_labels.py --dataset gc10
python scripts/verify_labels.py --dataset deepcrack
python scripts/build_prompts.py --protocol openvocab
```

All generated dataset directories are gitignored and must be rebuilt, not committed.

---

## 7. Gate check

| Criterion | Status |
|---|---|
| GC10 converted, all annotations paired | Pass |
| Pinyin mapping + typo merge + corrupt drop applied | Pass |
| Both GC10 protocols built, leakage check passing | Pass |
| DeepCrack masks converted to segment boxes | Pass |
| Mask→box limitation quantified | Pass |
| Boxes visually verified on both | Pass |
| All YAMLs load in Ultralytics | Pass |
| Corpus covers every class, validation clean | Pass (draft wording) |

**Phase 1b gate: PASSED.**

---

## 8. Carried forward

1. **Report fix — constraint (ii):** now has a much better resolution than the
   Phase 1 recommendation. Use the three-regime framing in §3.
2. **Report fix — corpus size:** "approximately 60 descriptions" → **170**.
3. **Report fix — dataset totals:** detection instances now
   **4,189 (NEU-DET) + 3,541 (GC10-DET) + 1,864 (DeepCrack)**, not the
   "55,000" quoted from RDD2022.
4. **DeepCrack box looseness** should be stated as a known limitation.
5. Corpus wording still needs domain review, GC10 terms especially.
