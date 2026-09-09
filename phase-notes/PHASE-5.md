# Phase 5 — TG-FEM

**Status:** Complete (code + structural verification). Semantic verification blocked - see PHASE-4.md §5.
**Goal:** Implement the module, insert at P3/P4/P5.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Replace Phase 2's identity `forward` with report §4.2's real maths | Done — `src/tgfem/module.py` |
| Keep ablation (a) valid (same param budget, identity vs active) | Done — `_materialise` runs regardless of `identity` |
| Make the model genuinely open-vocabulary, not just attention-conditioned | Done — swapped head to `WorldDetect` (§3) |
| Wire text into the whole pipeline via a real Ultralytics training loop | Done — `TGFEMTrainer` / `TGFEMModel` (§3) |
| Verify end to end on real data, real Ultralytics trainer | Done — `scripts/train_tgfem_gate.py`, passes (§4) |

---

## 2. The maths, and the report correction it required

```
1. Projection            1x1 conv, C -> d                    (self.proj)
2. Region-text attention  A = softmax(F_q T^T / sqrt(d));  S = A T
3. Dual gating from S     channel gate g, spatial gate s
4. Residual output        F' = F + F (*) g (*) s
```

Both gates are derived from `S` — the text-conditioned region response —
never from `F`'s own pooled statistics. That is the entire structural
difference from CBAM and from YOLO-World's channel-only
`MaxSigmoidAttnBlock`, and it's what ablation (g) isolates (§3).

**README correction #3 applied.** The report writes `F' = F⊙g⊙s + F` and
claims it reduces to identity when `g=s=1` — it actually gives `2F`. The
correct identity condition is **gates → 0**. `channel_gate` and
`spatial_gate` are therefore zero-weight / `bias=-4` initialised
(`sigmoid(-4) ≈ 0.018`), so training starts near-identity — the same trick
used for zero-init residual branches elsewhere (ReZero, CBAM's own gate
init).

**A finding worth carrying into the report discussion.** Because the whole
gated branch starts at zero weight, the *first* backward pass on a fresh
model gives **exactly zero gradient** to `proj` (and to `channel_gate`'s and
`spatial_gate`'s own weight matrices' upstream contribution) — the chain
back through `g`/`s` into `S` multiplies by a zero weight matrix. Verified
directly:

```
step 0: proj_grad=0.000000  channel_gate_grad=118.115143
step 1: proj_grad=219.370331  channel_gate_grad=364.447083
step 2: proj_grad=3861.101074  channel_gate_grad=3086.170898
```

`channel_gate`/`spatial_gate`'s own weights *do* get a real gradient
immediately (linear layer: `dz/dW = input`, independent of `W`'s current
value) — only the path *back through* them is blocked at step 0. One step
later, the gates are non-zero and the whole module is live. This is the
same self-unsticking behaviour ReZero relies on (its scalar gate's own
gradient is `dL/dα = dL/dy · F(x)`, generically non-zero even though `α=0`
initially kills the branch's contribution). Not a bug — but worth stating
explicitly if training curves show a one-step "warm-up" before TG-FEM
engages, so it isn't mistaken for something wrong.

---

## 3. TG-FEM alone does not make the model open-vocabulary

The README's component table lists five pieces, including a
**region-text contrastive head** that "labels each box by nearest text
fingerprint." TG-FEM's attention only *conditions features* — it never
touches how a box gets assigned a class. With a stock `Detect` head (fixed
N-way softmax baked into weights), the model would still be closed-set
regardless of how good the attention is: a class never seen in training
simply has no output neuron for it.

**Fix — reuse Ultralytics' own `WorldDetect` + `ContrastiveHead`,** exactly
as `cfg/yolo11-cbam.yaml`'s docstring already anticipated
("`WorldDetect`, YOLO-World's channel-only attention... ships already").
`cfg/yolo11-tgfem.yaml`'s head is now `WorldDetect`, which compares each
region's projected embedding against the text embeddings via cosine
similarity + a learned scale/bias (`ContrastiveHead`) — genuinely
open-vocabulary classification, the same mechanism that made YOLO-World
work at all.

**The wiring problem this creates.** `WorldDetect.forward(x, text)` needs
`text` as an explicit second argument — `parse_model` builds a single-input
layer chain (Phase 2's finding, which is *why* TG-FEM itself uses the
`self.txt` attribute trick instead). Ultralytics' own `WorldModel` solves
this by overriding `predict()` to special-case `WorldDetect` inside the
layer loop. `src/tgfem/detection_model.py::TGFEMModel` does the same,
trimmed to what this architecture needs (no `C2fAttn`/`ImagePoolingAttn` —
those are YOLO-World neck blocks this project doesn't use). TG-FEM's own
layers need **no** special-casing in that loop — they still just do
`x = m(x)`, reading `self.txt`, which is the entire reason Phase 2 chose
that mechanism over threading a second argument.

**Getting the language branch into Ultralytics' optimiser.**
`src/tgfem/trainer.py::TGFEMTrainer` subclasses `DetectionTrainer`
(mirroring `WorldTrainer`'s pattern) and overrides `get_model()` to: build a
`TGFEMModel`, find its `TGFEM` layers, attach a `TextConditioner`, and
expose the learnable context vector as `model.tgfem_ctx` — the *same*
`nn.Parameter` object already owned by `ContextTokenLearner` (weight tying,
the same pattern as tied input/output embeddings). `Trainer.build_optimizer`
walks `model.named_modules()` generically with no notion of TG-FEM at all;
this is what makes it pick `ctx` up automatically without any change to
Ultralytics itself. Only `ctx` is exposed this way — not the whole frozen
CLIP encoder, which would otherwise add ~40M dead (`requires_grad=False`)
parameters to the optimiser's bookkeeping for nothing.

---

## 4. Verification — the Phase 5 gate (`scripts/train_tgfem_gate.py`)

Same philosophy as Phase 2's walking skeleton, extended to the real
pipeline. All pass on CPU, on real NEU-DET data, through Ultralytics' actual
`DetectionTrainer` loop (not a hand-rolled substitute):

```
TGFEM at layers      : [5, 8, 13]
WorldDetect head      : present
context tokens moved during training : True
TGFEM materialised    : proj=(512, 128, 1, 1)  channel_gate=(128, 512)
raw prediction tensor : shape (1, 10, 189)
checkpoint round-trip : TGFEM layers, materialised weights, WorldDetect
                        head, and the TextConditioner hook all survive
                        pickling and reload correctly
```

**A real trap found and fixed while building this gate.** The very first
run reported `context tokens moved during training : False` despite
`ctx.grad` being confirmed non-zero and growing every batch. Cause:
Ultralytics defaults `nbs=64` (nominal batch size) and only calls
`optimizer.step()` once every `round(nbs / batch)` batches — with a tiny
smoke-test batch size, that threshold is never reached within one epoch, so
gradients accumulate (un-zeroed) but nothing ever updates. **Not a bug in
TG-FEM's wiring** — confirmed by overriding `nbs=batch` (forcing
`accumulate=1`), after which `ctx` visibly moved (`max diff ≈ 0.0195` after
22 steps). `train_tgfem_gate.py` now does this automatically; a real
Phase 6 run at `batch=16` with the report's schedule does not need it
(`nbs=64` there is a legitimate ~4-batch accumulation, not a stall).

---

## 5. Ablation configs

| cfg | Gates | Head | Purpose |
|---|---|---|---|
| `yolo11-tgfem.yaml` | TG-FEM, text-conditioned | WorldDetect | the model |
| `yolo11-tgfem-ablation-a.yaml` | TG-FEM, `identity=True` | WorldDetect | ablation (a) — same param budget, mechanism disabled |
| `yolo11-cbam-worlddetect.yaml` | CBAM, image-conditioned | WorldDetect | ablation (g) — **the load-bearing comparison** |

`yolo11-cbam.yaml` (Phase 3) is **not** reused for ablation (g) — it pairs
CBAM with a stock closed-vocabulary `Detect`, which was the right control
for "does attention help at all vs. pretraining" (Phase 3's question) but
would confound "gates from text vs. gates from image" with "has an
open-vocab head at all vs. doesn't" (ablation (g)'s actual question). Its
existing results (`results/phase3_cbam.json`) stay valid for what they
were measuring; ablation (g) needs a fresh CBAM run against the new config.

---

## 6. Files created / changed

```
src/tgfem/module.py           real math, was identity-only
src/tgfem/detection_model.py  NEW - TGFEMModel (WorldDetect threading)
src/tgfem/trainer.py          NEW - TGFEMTrainer (optimiser/checkpoint wiring)
cfg/yolo11-tgfem.yaml         head -> WorldDetect, d 256 -> 512
cfg/yolo11-tgfem-ablation-a.yaml   NEW - ablation (a)
cfg/yolo11-cbam-worlddetect.yaml   NEW - ablation (g) control
scripts/train_tgfem_gate.py   NEW - Phase 5 gate check
```

**Reproduce:**
```bash
python scripts/train_tgfem_gate.py
python scripts/train_tgfem_gate.py --epochs 2 --fraction 0.2 --n-ctx 4
```

---

## 7. Gate check

| Criterion | Status |
|---|---|
| Real math replaces identity, matches report §4.2 (formula bug fixed) | Pass |
| Ablation (a) keeps identical parameter budget | Pass — verified (132,225 params either way at a test scale) |
| Region-text contrastive head present and wired (genuine open-vocab) | Pass |
| Trains end to end through Ultralytics' real trainer, no shape/autograd errors | Pass |
| Gradient reaches learnable context tokens through the whole pipeline | Pass |
| Checkpoint round-trip (real math, not just identity mode) | Pass |
| Boxes emitted at the far end | Pass |

**Phase 5 gate: PASSED structurally.** Same caveat as Phase 4: no semantic
content in any embedding produced so far (§5 there) and no GPU used for
this pass, so none of the above validates *accuracy* — only that the whole
pipeline is correct and trainable. Phase 6 needs real compute.

---

## 8. Carried forward

1. **No GPU was available for this pass.** All verification above ran on
   CPU, at toy scale (tiny fraction, small imgsz, 1-2 epochs).
   `scripts/train_tgfem.py` (Phase 6) is written for the report's real
   schedule (150 epochs, batch 16, imgsz 640) but was only smoke-tested at
   toy scale so far — see `phase-notes/PHASE-6.md`.
2. Ablation (g)'s CBAM+WorldDetect config exists but has never been trained
   — no results to compare against `tgfem` yet.
3. `train_tgfem.py`'s test-split evaluation uses a fresh `DetectionValidator`
   (not the `WorldValidator` YOLO-World uses), which sidesteps the
   `set_classes`-gets-overridden trap Phase 3 found in `eval_yoloworld.py` —
   worth re-confirming once real numbers exist, since that trap was subtle
   and specific to `WorldValidator`'s `__call__` override, not to
   `DetectionValidator` in general.
