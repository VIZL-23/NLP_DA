# TG-FEM — Text-Guided Open-Vocabulary Surface Defect Detection

**Course:** BCSE409L Natural Language Processing · **Faculty:** Dr. Vijayaprabakaran K
**Team:** Ritvik Kaza (24BCE5351) · Vishal Narayan V (24BCE5427)

---

## What this project does

Conventional defect detectors are **closed-vocabulary**: their class list is
compiled into the model weights, so adding one new defect type means collecting
data, annotating it, and retraining.

This project makes the defect vocabulary a **runtime input**. An inspector picks
an image, types a plain-English description — *"scratches on the steel
surface"* — and the model returns bounding boxes for matching regions. The
deliverable is a local web app that does exactly that, across steel and concrete.

**The contribution is TG-FEM** (Text-Guided Feature Enhancement Module): an
attention module whose channel and spatial gates are computed from the *text*
rather than from the image's own pooled statistics. The model is told *what to
look for* before it decides *where to look*. This is what distinguishes it from
CBAM-family attention and from YOLO-World's channel-only max-sigmoid attention.

**What the measurements say, up front.** The text interface works: every model
returns boxes for trained phrasings and returns *nothing* for nonsense or for a
defect that isn't in the image. But it is **not** zero-shot — defect types never
seen in training score 0.000 AP — and it responds to the defect *noun*, not to
meaning. Both findings are measured, not guessed; see
[Known limits](#known-limits-worth-stating-up-front-in-a-demo).

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

## How to run the app

Commands are for **Windows PowerShell**, run from the project folder. The app
runs entirely on your machine — no internet needed once it's set up.

### 1. Install Python and PyTorch

You need **Python 3.10** and a **PyTorch build that matches your GPU driver**.
PyTorch is deliberately not in `requirements.txt`, because the right build
depends on your machine. Get the install command from
[pytorch.org](https://pytorch.org/get-started/locally/). This project was built
with `torch 2.11.0+cu128` on an RTX 4050 (6 GB). A CPU-only build also works for
the app — just slower.

### 2. Create the virtual environment and install dependencies

```powershell
python -m venv .venv --system-site-packages
```

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`--system-site-packages` lets the venv reuse the PyTorch you installed in step 1
instead of downloading ~2.5 GB again.

### 3. Put the trained models in place

The trained models are **not in git** — each is ~264 MB, over GitHub's 100 MB
file limit. Copy these three files from the machine that trained them (drive,
USB, etc.) to exactly these paths:

```
runs/phase6_neu_crack_closed_tgfem_phraseaug_syn/weights/best.pt
runs/phase6_gc10_closed_tgfem_phraseaug/weights/best.pt
runs/phase6_concrete_closed_tgfem_phraseaug/weights/best.pt
```

Only `best.pt` is needed; `last.pt` can be skipped. Each checkpoint carries its
own CLIP text encoder, so **nothing else needs downloading to run the app**. If
you don't have the files, [train them yourself](#training-the-models-from-scratch).

The sample images the app shows are already in the repo (`app/samples/`).

### 4. Start the server

```powershell
.venv\Scripts\python.exe app\server.py
```

When it's ready it prints something like:

```
  neu_crack  7 classes · 21 samples · cuda:0
  gc10   10 classes · 20 samples · cuda:0
  concrete  6 classes · 18 samples · cuda:0

  ->  http://127.0.0.1:8000
```

A model whose checkpoint is missing prints `skip <name> - not trained yet` and
the app starts without it. If none are found, it exits with a message saying so.

### 5. Use it

Open **http://localhost:8000** in a browser, then:

1. **Pick a model** matching your material.
2. **Pick an image** — a bundled sample, or *Upload your own image*.
3. **Describe what to find** — click a phrase chip, or type one description per
   line. The full list of phrases each model knows is in
   **[docs/query-phrases.md](docs/query-phrases.md)**.
4. Press **Detect**. Each description gets its own colour; the legend shows
   which ones matched and which weren't found.

Press `Ctrl+C` in the terminal to stop the server.

### Options

```powershell
.venv\Scripts\python.exe app\server.py --device cpu          # no GPU, or GPU busy training
.venv\Scripts\python.exe app\server.py --port 8080           # if 8000 is taken
.venv\Scripts\python.exe app\server.py --only neu_crack      # load a subset of models
```

> Prefer an activated venv? `.venv\Scripts\Activate.ps1`, then plain `python`.
> If PowerShell blocks the script, calling `.venv\Scripts\python.exe` directly
> (as above) avoids the problem entirely.

### Troubleshooting

| symptom | cause |
|---|---|
| `ModuleNotFoundError: fastapi`, or `Form data requires "python-multipart"` | step 2 didn't run in this venv |
| `no checkpoints found` | step 3 — the `best.pt` files aren't at those exact paths |
| `ModuleNotFoundError: torch` | step 1, or the venv was created without `--system-site-packages` |
| port already in use | another server is running — stop it, or use `--port 8080` |
| page looks stale after an update | hard-refresh (`Ctrl+F5`) |
| a query returns nothing | often correct — see [Known limits](#known-limits-worth-stating-up-front-in-a-demo); also try lowering the confidence slider |

---

## Query phrases

**[docs/query-phrases.md](docs/query-phrases.md)** lists **251 phrases** the
shipped models were trained on, grouped by model and then by class, with each
class's AP and a warning where a class is weak. Phrase 1 in each list is the one
the class is scored on — start there.

Two rules make the difference between a result and a blank:

- **Pick the right model.** Steel phrases do nothing on the concrete model.
- **Keep the defect noun.** Rewording around it is fine; swapping it for an
  unlisted synonym usually returns nothing.

The file is generated from `prompts/defect_corpus.json`. After changing the
corpus or the shipped models, regenerate it rather than editing by hand:

```powershell
.venv\Scripts\python.exe scripts\make_phrase_list.py
```

---

## The shipped models

| model | classes | domain | mAP@0.5 | default conf |
|---|---|---|---|---|
| `neu_crack` | 7 | steel strip + concrete cracks | 0.6756 | 0.25 |
| `gc10` | 10 | galvanised steel sheet | 0.6379 | 0.25 |
| `concrete` | 6 | concrete structural defects | 0.3256 | 0.10 |

**23 class entries across three domains, 251 trained phrasings.** All three
reject nonsense text, and also reject a trained *sibling* class that isn't in the
image.

The model list, each vocabulary and each sample set come **from the server**,
which reads them from the checkpoints. Adding a model is one entry in `MODELS` in
`app/server.py`; the frontend needs no change.

#### Merging was measured both ways, and the answer depends on class distance

| merge | outcome | shipped? |
|---|---|---|
| NEU + GC10 (steel + steel, 16 classes) | GC10 0.598 vs 0.638, NEU 0.704 vs 0.720 | no |
| NEU + crack (steel + concrete, 7 classes) | NEU 0.7264 vs 0.7205, crack 0.448 vs 0.465 | **yes** |

Two steel datasets share near-synonymous classes (`inclusion` exists in both),
so the model must separate genuinely confusable categories and every class gets
harder. Steel and concrete are not confusable, so that merge costs nothing
measurable. **Merging helps when the added classes are far apart and hurts when
they are close** — which is more useful than either result alone.

The `concrete` model is kept **separate** for the same reason: its classes sit on
the same material as `crack`, so folding it in would recreate the NEU+GC10
situation. `neu_crack` replaces the earlier NEU and crack specialists, which stay
in `runs/` as the evidence.

#### Why `neu_crack` exists at all

A single-class crack model **is not text-guided**. With `nc=1` its contrastive
head was never asked to separate one text embedding from another, so it returned
the same boxes for `banana` as for a real query (confidences differing by
~0.002). Giving crack six steel classes to compete against fixed it, at no
measurable cost to steel. See `phase-notes/PHASE-9.md` sections 4–5.

#### Checkpoints are chosen for vocabulary, not for the top mAP

Every shipped checkpoint is **phrase-augmented**, and `neu_crack` is also the
**synonym-corpus** run (`_syn`, 91 phrasings instead of 70). Both trade a little
mAP for words a user might actually type:

| change | mAP | vocabulary |
|---|---|---|
| phrase augmentation (NEU) | 0.7268 → 0.7205 | 2 of 6 → 5 of 6 phrasings answered |
| synonym corpus (`neu_crack`) | 0.6867 → 0.6756 | `drag marks` 0.028 → 0.685 |

Choosing by mAP would give a model that answers only to memorised strings, so in
a live demo it returns nothing and the project looks broken. The mAP differences
are small (`phase-notes/PHASE-7.md` section 2); the vocabulary differences are not.

#### Confidence defaults are per model, and measured

`concrete` starts at 0.10, not 0.25. Its boxes are looser and its imagery is
field photography, so its scores run lower than the steel models'. Hit rate on
correct queries, single-class test images:

| | conf 0.25 | conf 0.10 |
|---|---|---|
| all six classes | 53% | **80%** |

At a global 0.25 default, `crack` found nothing at all and `scaling` found a
third of what it can — a working model presenting as broken.
`ModelSpec.default_conf` carries the per-model value; the UI reads it from
`/api/info`.

#### Sample images

The bundled samples are drawn from each dataset's **held-out test split** by
fixed seed — **not** selected by how well the model scores on them. "Random
sample, fixed seed, from data the model never saw" is a defensible answer when
someone asks how they were chosen. One folder per model, so a steel model is
never demoed on a concrete image by accident. To regenerate:

```powershell
.venv\Scripts\python.exe scripts\make_samples.py --dataset neu_crack --per-class 3
.venv\Scripts\python.exe scripts\make_samples.py --dataset gc10
.venv\Scripts\python.exe scripts\make_samples.py --dataset concrete --per-class 3
```

---

## Known limits, worth stating up front in a demo

These are measured, not guesses (`phase-notes/PHASE-7.md` sections 3–4,
`phase-notes/PHASE-9.md` sections 6–8):

- **It is not zero-shot.** Classes held out of training score 0.000 AP.
- **The text interface is keyed on the defect NOUN.** Across both multi-domain
  models, 8 of 8 novel phrasings that keep the class noun score 0.246–0.674; all
  8 that replace it score at most 0.088, with no overlap. So
  `scratches on the metal surface` works, but `long thin gouges scored into metal`
  returns nothing — even though an inspector would call them the same defect.
  Every trained phrasing works; that's why the UI and
  [docs/query-phrases.md](docs/query-phrases.md) list them.
- **Vocabulary is bought by writing phrases, not by training longer.** Adding 21
  synonyms to the corpus and retraining moved `drag marks left along the steel`
  from 0.028 to **0.685** for −0.011 mAP; a synonym never added stayed silent. If
  a defect name matters, add it to `prompts/defect_corpus.json` and retrain.
- **Pick the right model.** The steel models have only seen steel imagery. A
  cracked wall belongs to `neu_crack` (cracks only) or `concrete` (six defect
  types). The wrong model is the most likely way to make the demo look broken.
- **`rust_stain` does not work.** 136 training boxes, AP 0.031. Don't
  demonstrate it. Lead the concrete model with `spalling` (AP 0.581, found in
  6 of 6 test images).
- **The concrete model is much weaker than the steel ones** (0.326 vs 0.68/0.64),
  and that is the task, not a bug: field photography, region-level defects whose
  boxes cover a median 12% of the image, and two classes under 250 boxes.
- **Nothing returned is often the right answer.** A wrong-class or nonsense query
  returning nothing is worth demonstrating on purpose: it shows the model
  discriminating rather than boxing whatever text it's given.

---

## Training the models from scratch

Only needed if you don't have the three `best.pt` files. Training needs an
NVIDIA GPU; each run takes roughly 2–5 hours on an RTX 4050.

### 1. CLIP weights (training only)

Training builds the text encoder from OpenAI's CLIP ViT-B-32. `src/tgfem/language.py`
looks for a local copy first, at any of:

```
weights/clip/ViT-B-32.pt
%APPDATA%\Ultralytics\weights\clip\ViT-B-32.pt
%USERPROFILE%\.cache\clip\ViT-B-32.pt
```

and otherwise downloads it from huggingface.co. If neither works it falls back
to a **randomly-initialised** encoder and warns loudly — every result from such a
run is meaningless. Check `pretrained_clip_loaded: true` in the run's
`results/*.json` before trusting it.

### 2. Datasets

Only **`NEU-DET`** is committed (37 MB). Everything else is gitignored and
downloaded per machine.

| Dataset | Source | Save to | Used for |
|---|---|---|---|
| GC10-DET | [kaggle: zhangyunsheng/defects-class-and-location](https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location) | `datasets/GC10-DET/` | `gc10` |
| DeepCrack | [github: yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack) → `./dataset` | `datasets/DeepCrack/` | held-out OOD probe |
| Merged crack set | `python scripts/download_cracks.py` (resumable) | `datasets/CRACK500/parquet/` | `neu_crack` |
| Concrete defects | [Roboflow Universe, SHM, CC BY 4.0](https://universe.roboflow.com/shm-agftj/concrete-defect-detection-pl8ed) → export **YOLOv11**, extract | `datasets/concrete-defect-roboflow/` | `concrete` |

The Roboflow export needs a free Roboflow account to download. You only need the
dataset zip — not the API key or the hosted-model snippet Roboflow offers.

Then build the YOLO layouts (order matters where noted):

```powershell
.venv\Scripts\python.exe scripts\check_env.py            # reports what's present / wrong
.venv\Scripts\python.exe scripts\prepare_neu_det.py
.venv\Scripts\python.exe scripts\prepare_gc10.py
.venv\Scripts\python.exe scripts\prepare_deepcrack.py
.venv\Scripts\python.exe scripts\download_cracks.py
.venv\Scripts\python.exe scripts\prepare_crack_merged.py # needs download_cracks.py
.venv\Scripts\python.exe scripts\prepare_neu_crack.py    # needs neu-det-yolo + crack-merged-yolo
.venv\Scripts\python.exe scripts\prepare_concrete.py     # needs the Roboflow export
```

> **Never commit a dataset, an archive, or a checkpoint.** A 363 MB parquet in a
> commit once made a push impossible — GitHub hard-rejects any file over 100 MB.
> `.gitignore` covers `runs/`, `weights/`, `*.pt`, every generated `*-yolo/`
> directory, and `datasets/*.zip|tar|7z|rar`.

### 3. Train the three shipped models

These reproduce the exact configurations the app loads. Results land in
`results/`, checkpoints in `runs/`.

```powershell
.venv\Scripts\python.exe scripts\train_tgfem.py --variant tgfem --dataset neu_crack --protocol closed --phrase-aug --epochs 100 --run-suffix syn
.venv\Scripts\python.exe scripts\train_tgfem.py --variant tgfem --dataset gc10 --protocol closed --phrase-aug
.venv\Scripts\python.exe scripts\train_tgfem.py --variant tgfem --dataset concrete --protocol closed --phrase-aug
```

`--run-suffix` keeps a retrain from overwriting an existing checkpoint (runs use
`exist_ok=True`). Don't run two trainings, or training and the app on GPU, at the
same time on a 6 GB card — start the app with `--device cpu` while training.

---

## Build phases

Detailed write-ups live in **`phase-notes/PHASE-N.md`**.

| # | Phase | Status | What it does |
|---|---|---|---|
| **0** | Foundation | Done | venv + CUDA verified, Ultralytics installed, datasets audited, repo hygiene |
| **1** | Data pipeline | Done | VOC→YOLO conversion, both split protocols, 60-phrase text corpus |
| **1b** | GC10-DET + DeepCrack | Done | Both converted; corpus extended to 170 phrases / 17 classes |
| **2** | Walking skeleton | Done | TG-FEM registered + placed at P3/P4/P5, trains end to end, boxes emitted |
| **3** | Baselines | Done | stock 0.7717 · CBAM 0.7372 · YOLO-World-S 0.0394 |
| **4** | Language branch | Done | Frozen CLIP text encoder + CoOp learnable context tokens, wired via a `forward_pre_hook` |
| **5** | TG-FEM | Done | Real math (proj → region-text attention → dual gating → residual) + WorldDetect open-vocab head |
| **6** | Training | Done | GPU, real CLIP throughout. **The architecture result is negative** — see below |
| **7** | Evaluation & ablations | Done | Ablations (a), (d), (g) + phrase augmentation. Text interface characterised |
| **8** | Demo | Done | `scripts/demo.py` (command line); superseded as the deliverable by the web app |
| **9** | Specialists + app | Done | Steel merge rejected, steel+concrete merge shipped; single-class models shown not to be text-guided; 6-class concrete model; noun-keying measured; synonym corpus shipped; app serves 3 models |

### Build strategy

1. **Stand on Ultralytics, don't rewrite it.** It already ships YOLOv11,
   `YOLOWorld`, the `WorldDetect` contrastive head, and `MaxSigmoidAttnBlock`
   (YOLO-World's channel-only attention). TG-FEM slots into the same socket as
   that block, which makes the headline comparison a one-line YAML swap.
2. **Walking skeleton before novelty.** Phase 2 wires a *no-op* module through
   the whole pipeline first. Integration is the biggest risk; kill it early.
3. **Baselines before novelty.** You can't claim a gain over YOLO-World-S without
   a YOLO-World-S number — and if TG-FEM disappoints, solid baselines are still a
   report.

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

- COCO pretraining is worth **+6.5** points (0.7717 − 0.7069)
- Attention is worth **+3.0** points (0.7372 − 0.7069)

### Phase 6 results (NEU-DET test split) — the architecture result is negative

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

Per-epoch validation swung by ±0.10 during training, so neither delta is
distinguishable from noise. **With the head held constant, there is no evidence
that text-derived gates beat image-derived gates.**

**The pre-registered prediction failed.** It said TG-FEM's gain would
concentrate on `crazing` and `rolled-in_scale`; TG-FEM is *worse* on both
(−0.0350, −0.0139).

**Zero-shot fails outright.** On the openvocab test split both held-out classes
score **0.000 AP** despite 688 and 628 instances. The `mAP50 = 0.1409` in
`results/phase6_neu_openvocab_tgfem.json` is the mean over four classes where
only the *seen* class `patches` (4 instances) scores — don't compare it to
YOLO-World-S's 0.0394.

**This is not a wiring bug.** The negative control swaps in absurd prompts and
performance collapses 16× (0.3994 → 0.0243), so the text path is genuinely live.

Diagnosis: `WorldDetect`'s open-vocabulary ability comes from large-scale
region-text pretraining (YOLO-World: 27M grounding pairs). Training from scratch
on 1,075 images can't produce it — the architecture is sound but data-starved by
~4 orders of magnitude. See `phase-notes/PHASE-6.md`.

---

## Repository layout

```
app/
  server.py            local FastAPI app (the deliverable)
  static/index.html    single-page UI
  samples/             bundled test-split images, one folder per model
docs/
  query-phrases.md     every phrase each shipped model was trained on (generated)
scripts/
  check_env.py         toolchain + dataset audit (run this first)
  prepare_neu_det.py   VOC→YOLO, splits, dataset YAMLs, geometry stats
  prepare_gc10.py      GC10-DET VOC→YOLO + splits + YAMLs
  prepare_deepcrack.py DeepCrack mask→box (connected components) + YAMLs
  download_cracks.py   resumable HTTP-Range fetch of the merged crack parquet
  prepare_crack_merged.py  parquet→YOLO boxes; keeps crack-free backgrounds,
                       excludes DeepCrack so the OOD probe stays clean
  prepare_neu_crack.py NEU + crack → 7 classes (seeded, source-stratified)
  prepare_concrete.py  Roboflow concrete export → project layout (rename only)
  prepare_combined.py  NEU + GC10 → 16 classes (REJECTED, kept as evidence)
  verify_labels.py     draws converted boxes back onto images
  build_prompts.py     text-corpus validation + protocol filtering
  train_baseline.py    Phase 3 baselines (stock / stock_scratch / cbam)
  eval_yoloworld.py    Phase 3 YOLO-World-S zero-shot baseline
  train_skeleton.py    Phase 2 walking-skeleton gate check (identity TG-FEM)
  train_tgfem_gate.py  Phase 5 gate check (real TG-FEM + language branch)
  train_tgfem.py       training for every TG-FEM model (--run-suffix avoids overwrites)
  run_ablations.py     Phase 7 ablation dispatcher (resumable)
  eval_tgfem.py        Phase 7 negative control + cross-variant comparison
  eval_deepcrack.py    DeepCrack OOD evaluation, with nonsense-prompt control
  probe_wording.py     wording robustness; marks each phrase seen/NOVEL itself
  probe_generalisation.py  corpus-absent phrasings: coverage vs semantics
  make_samples.py      bundle demo samples from the test split (fixed seed)
  make_phrase_list.py  regenerate docs/query-phrases.md from the corpus
  demo.py              Phase 8 command-line text-query demo
  run_crack_specialist.sh  queues a crack run behind a busy GPU
src/tgfem/
  module.py            TGFEM - real math (Phase 5)
  language.py          TextEncoder, ContextTokenLearner, TextConditioner (Phase 4)
  detection_model.py   TGFEMModel - WorldDetect text threading (Phase 5)
  trainer.py           TGFEMTrainer - optimiser/checkpoint wiring (Phase 5)
  data.py              class taxonomies (incl. GC10_NAME_MAP) + corpus phrase selection
  inference.py         DefectDetector - framework-free, backs the app and demo
cfg/
  yolo11-tgfem.yaml            the model (TG-FEM gates + WorldDetect head)
  yolo11-tgfem-ablation-a.yaml ablation (a) - identity mode, same param budget
  yolo11-cbam.yaml             Phase 3 closed-vocab CBAM baseline (stock Detect)
  yolo11-cbam-worlddetect.yaml ablation (g) - CBAM gates + WorldDetect head
prompts/
  defect_corpus.json   241 defect descriptions across 22 classes
results/               one JSON per training/eval run (committed; small)
datasets/              source data + generated *-yolo/ dirs (gitignored except NEU-DET)
runs/                  checkpoints and training logs (gitignored)
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
Holding them out **as a block** lets that cluster leave together. Splitting it
strands `welding_line`, which loses 70% of its training images.

*Why not the 89% option:* it holds out `inclusion`, which also exists in
NEU-DET. Training on NEU-DET's `inclusion` and then querying GC10's with nearly
identical wording is **not a genuine zero-shot test**. Never hold out a class
whose twin is in the training set.

**Class-name collision — namespaced, not merged:**
> `inclusion` (NEU-DET) vs `gc10_inclusion` (GC10-DET)

Same concept, different regimes (200×200 grayscale close-up vs 2048×1000 steel
sheet), so they get distinct class IDs. Descriptions stay natural in the corpus;
only the label-space identity is namespaced.

---

## Things you must know before touching the data

These were found by auditing and are easy to get wrong:

- **GC10-DET folders are NOT the labels.** Images are multi-label; the folder
  shows only the *dominant* defect. **The XML is ground truth.**
- **GC10-DET class names are Chinese pinyin** in the XML (`3_yueyawan` =
  `crescent_gap`). The verified mapping is `GC10_NAME_MAP` in
  `src/tgfem/data.py`. It also merges the `10_yaozhe`/`10_yaozhed` typo pair and
  drops a corrupt class named `d`.
- **Open-vocab splits need a leakage guard.** 123 NEU-DET images contain more
  than one class, so a "seen-class" training image can hide a held-out object.
  `prepare_neu_det.py` excludes any such image and asserts this every run.
- **NEU-DET is 200×200 grayscale**; GC10-DET is 2048×1000. Training at 640
  upscales one and downscales the other.
- **The merged crack set's masks encode annotation conventions, not severity.**
  Stroke width is 56% explained by which source dataset an image came from
  (`cracktree200` is exactly 2 px on every mask). Don't derive sub-classes from it.
- **A dataset's class names are part of the data here.** A Roboflow set whose
  classes were named `0`, `1`, `2`, `3` was rejected for that alone — there is no
  text to query.
- **SDNET2018 has no bounding boxes** — classification folders only, and it is no
  longer in the repo. It can't train a detector or produce mAP.

---

## Known report corrections needed

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

2. **The dataset scale claim is off.** "≈106,000 images / 55,000 instances"
   counts RDD2022 (not used) and SDNET2018 (no boxes). The shipped models train
   on NEU-DET, GC10-DET, the merged crack set (8,126 training images) and the
   Roboflow concrete set (1,680 images, 3,565 boxes).
3. **§4.2 formula inconsistency.** `F' = F⊙g⊙s + F` gives `2F` when `g=s=1`, not
   the identity. Either drop the `+F` or say it reduces to identity as gates→0.
4. **CLIP text dim is 512, not 256.** `T` is declared `N×256`; a projection is
   needed, or `d` corrected.
5. **Corpus size.** §4.1 says "approximately 60 defect descriptions"; it is now
   **241** across 22 classes.
6. **Crack boxes are loose by construction** — median fill ratio 16.6% on
   DeepCrack, 22.2% on the merged set. State it as a known limitation of
   box-based crack detection.
7. **The zero-shot claim doesn't hold.** Held-out classes score 0.000 AP. The
   report should present the text interface as closed-vocabulary with flexible
   wording, not as open-vocabulary detection.

See `phase-notes/PHASE-1.md` §4, `PHASE-1b.md` §3–4 and `PHASE-9.md` for the full numbers.
