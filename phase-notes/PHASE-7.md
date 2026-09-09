# Phase 7 — Evaluation & Ablations

**Status:** Complete for NEU-DET. All priority ablations run on GPU with real CLIP weights.
**Goal:** mAP, zero-shot, FPS, and the ablations that decide whether the mechanism works.

---

## 1. What this phase establishes

Three independent ablations show the architecture's mechanisms do not move mAP.
A fourth line of investigation — probing the text interface directly — found the
mechanism that *is* broken, fixed it, and then measured the limits of that fix.

One sentence: **the detector works, the text interface matches words rather than
meaning, and 1,075 training images is roughly four orders of magnitude short of
what the open-vocabulary claim requires.**

---

## 2. Ablations (NEU-DET closed protocol, held-out TEST split)

Every run: 150 epochs, batch 16, 640x640, seed 42, `pretrained_clip_loaded: true`.

| run | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| TG-FEM (n_ctx=8, one fixed phrase) | **0.72681** | 0.39977 |
| CBAM + WorldDetect — ablation (g) | 0.72401 | 0.39541 |
| TG-FEM, n_ctx=0 — ablation (d) | 0.72142 | 0.38477 |
| TG-FEM identity — ablation (a) | 0.71830 | 0.38460 |
| TG-FEM + phrase augmentation | 0.72047 | 0.39368 |

```
ablation (a)  TG-FEM     - identity      = +0.0085
ablation (d)  n_ctx=8    - n_ctx=0       = +0.0054
ablation (g)  text gates - image gates   = +0.0028
phrase aug    augmented  - fixed phrase  = -0.0063
```

**All four deltas fall within ±0.01, against per-epoch validation swings of
±0.10.** None is distinguishable from noise.

The *consistency* is what makes this credible rather than inconclusive. Four
different interventions, four deltas in the same tight band, with a passing
negative control. A broken pipeline produces erratic numbers; this produces the
same answer every way we measure it.

### Ablation (d) answers gap G4 directly
G4 asked whether prompt design for the structural-inspection vocabulary matters.
Measured: **CoOp learnable context tokens gave no detection benefit over
hand-written prompts** (+0.0054). For an NLP deliverable this is a clean,
negative answer to a question the report itself posed.

### The falsifiable prediction failed
PHASE-3.md section 3 predicted that if text conditioning works, TG-FEM's gain
over CBAM should concentrate on `crazing` and `rolled-in_scale`. It is **worse**
on both (-0.0350, -0.0139). The prediction was specific, pre-registered, and
falsified.

---

## 3. Probing the text interface directly

mAP is a poor instrument for what this project is about. An inspector does not
type the exact string the model trained on. `scripts/probe_wording.py` holds the
image and the class fixed and varies only the *wording*.

Original model (one fixed phrase per class for all 150 epochs):

| query | conf |
|---|---|
| EXACT trained: "scratches on the steel surface" | 0.639 |
| one word changed: "...on the metal surface" | 0.561 |
| "long thin scratch marks on the metal" | **0.000** |
| bare "scratches" | 0.177 |
| "a long straight bright line running across the surface" | **0.000** |
| "long thin gouges scored into metal" | **0.000** |
| wrong class: "a pitted steel surface" | 0.000 |
| CONTROL: "a wooden door" | 0.000 |

**Changing one word costs 12% confidence; rephrasing at all collapses it to
zero.** The model keyed on six specific strings. This also explains the zero-shot
failure: a model that cannot handle a paraphrase of a *seen* class was never
going to handle an *unseen* one.

**Root cause.** `class_texts_for()` fixed one phrase per class for the entire
run, so the model saw the identical six strings 150 times. Nothing pressured it
to generalise across phrasing. Meanwhile the 170-phrase corpus built in Phase 1b
went unused.

---

## 4. Phrase augmentation — the fix, and its measured limit

Implemented `class_phrase_pools()` plus per-step resampling in
`TextConditioner`: each **training** step draws a fresh phrasing per class from
the corpus (~10 each). Evaluation stays deterministic on the fixed `--tier`
phrase so runs remain comparable.

### It worked, for listed phrasings

| query | original | phrase-aug |
|---|---|---|
| EXACT trained | 0.639 | 0.601 |
| one word changed | 0.561 | **0.599** |
| "long thin scratch marks on the metal" | 0.000 | **0.626** |
| bare "scratches" | 0.177 | **0.614** |
| "a long straight bright line..." | 0.000 | **0.606** |
| novel: "long thin gouges scored into metal" | 0.000 | 0.196 |
| wrong class | 0.000 | **0.000** |
| CONTROL nonsense | 0.000 | **0.000** |

From 2 of 6 phrasings working to 5 of 6, **for an mAP cost of -0.0063** — smaller
than every ablation delta above, i.e. effectively free. Crucially both control
rows stayed at exactly 0.000: the model became *robust*, not *indiscriminate*.

### But it is coverage, not generalisation

Four of those five phrasings were **in the training pool**. To separate "covers
what it was shown" from "understands meaning", `scripts/probe_generalisation.py`
queries only phrasings absent from the corpus entirely:

| query (none in corpus) | conf |
|---|---|
| "scoring marks left by a sharp edge" | **0.466** |
| "long thin gouges scored into metal" | 0.196 |
| "a slender vertical streak on the metal" | 0.076 |
| "a linear defect running down the plate" | **0.000** |
| "drag marks across the steel" | **0.000** |
| "a fine incision in the surface" | **0.000** |
| novel wrong class: "a surface covered in tiny holes" | 0.000 |
| CONTROL: "a bowl of soup" | 0.000 |

**1 of 6 novel paraphrases clears 0.25.** And *which* ones score is the finding:

- "**scoring** marks" → 0.466; corpus has "surface **scoring**", "abrasion **scoring**"
- "gouges **scored** into metal" → 0.196; corpus has "grooves **scored** into the metal"
- "vertical **streak**" → 0.076; corpus has "linear **streaks**"
- "drag marks", "incision", "linear defect" → **0.000**; semantically correct,
  **zero lexical overlap**

**The model matches words, not meaning.** Phrase augmentation widened the lookup
table and added fuzzy string matching; it did not create semantic understanding.

**Consequence for corpus size:** enlarging the corpus to 30 phrases per class
would extend coverage to those 30 phrasings and their near-lexical neighbours. It
would **not** produce a model that handles a phrasing nobody wrote down — which
is what open vocabulary means. This was tested, not assumed.

---

## 5. Negative control — the text path is genuinely live

Before accepting any negative result, `scripts/eval_tgfem.py` loaded the trained
checkpoint, stripped its `TextConditioner`, and re-attached one carrying absurd
prompts:

```
real prompts   mAP@0.5 = 0.3994
absurd prompts mAP@0.5 = 0.0243      (16x collapse)
```

Every probe in this phase also carried its own controls, and **wrong-class and
nonsense queries scored 0.000 in every single one.** The weak results are
measurements, not disconnected wiring.

---

## 6. The unified diagnosis

Four findings that look separate share one cause:

| finding | measurement |
|---|---|
| Ablations flat | ±0.01 across four interventions |
| Zero-shot fails | 0.000 AP on both held-out classes |
| No semantic generalisation | 1/6 novel paraphrases |
| Word-matching behaviour | scoring tracks lexical overlap |

**1,075 training images across 6 classes cannot teach the vision branch to
inhabit CLIP's semantic space.** It learns image → *these six specific embedding
vectors*. Anything far from those vectors falls off a cliff, whether "far" means
an unseen class or merely an unseen synonym.

YOLO-World reached open-vocabulary behaviour with 27M grounding pairs. The
architecture here is sound and correctly wired — it is data-starved by roughly
four orders of magnitude. PHASE-3.md's risk register anticipated exactly this
(row 1).

---

## 7. Files added this phase

```
scripts/probe_wording.py         wording robustness probe (graded paraphrases)
scripts/probe_generalisation.py  corpus-absent phrasings: coverage vs semantics
src/tgfem/data.py                + class_phrase_pools()
src/tgfem/language.py            + phrase_pools resampling (training only)
src/tgfem/trainer.py             + phrase_pools plumbing
scripts/train_tgfem.py           + --phrase-aug
```

**Reproduce:**
```bash
python scripts/train_tgfem.py --variant tgfem --dataset neu --protocol closed --phrase-aug --device 0
python scripts/probe_wording.py        --checkpoint runs/phase6_neu_closed_tgfem_phraseaug/weights/best.pt
python scripts/probe_generalisation.py --checkpoint runs/phase6_neu_closed_tgfem_phraseaug/weights/best.pt
```

### Implementation note worth keeping
Sampling assigns to a **local**, never to `self.class_texts`. An earlier version
mutated it, which silently left *evaluation* running on whatever phrase training
last sampled (observed: `'abrasion mark'` instead of the `--tier` phrase), and
would also have clobbered `scripts/demo.py`, which injects queries by assigning
to that same attribute.

---

## 8. Gate check

| Criterion | Status |
|---|---|
| Ablations (a), (d), (g) run at the report's schedule | Pass |
| Negative control confirms the text path is live | Pass |
| Text interface characterised, not merely scored | Pass |
| Coverage vs generalisation separated experimentally | Pass |
| TG-FEM outperforms the CBAM control | **FAIL — +0.0028** |
| Zero-shot detection of held-out classes | **FAIL — 0.000 AP** |
| Semantic generalisation to unlisted phrasings | **FAIL — 1/6** |

**Phase 7 gate: PASSED procedurally. The scientific result is negative, and now
mechanistically explained.**

---

## 9. Carried forward

1. Untested paths remain: GC10-DET (10 classes, ~2,300 images), DeepCrack OOD,
   a `--tier` sweep, and TG-FEM at P5-only vs all three scales.
2. **GC10-DET is the one worth running.** Nearly double the images and 10 classes
   instead of 6. If the diagnosis in section 6 is right, more classes and more
   data should move the text interface measurably — a testable prediction rather
   than a hope.
3. Phrase augmentation is a genuine, cheap improvement to the text interface and
   should stay on by default for any future run, despite the -0.0063 mAP cost.
