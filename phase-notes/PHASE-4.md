# Phase 4 — Language Branch

**Status:** Complete (code + structural verification). Semantic verification blocked - see §5.
**Goal:** Cache frozen CLIP text embeddings; add learnable context tokens.

---

## 1. What we set out to do

| Task | Result |
|---|---|
| Frozen CLIP ViT-B/32 text encoder wrapper | Done — `src/tgfem/language.py::TextEncoder` |
| Learnable CoOp-style context tokens (ablation (d)) | Done — `ContextTokenLearner` |
| Wire text into every TGFEM layer without forking Ultralytics | Done — `TextConditioner` + a `forward_pre_hook` |
| Verify gradients actually reach the context tokens | Done — see §4 |

---

## 2. Design

`open_clip` (`ViT-B-32`, `openai` weights) supplies the frozen text tower —
**512-d output**, matching the README correction ("CLIP text dim is 512, not
256"). `TextEncoder` keeps the tower's sub-modules (`token_embedding`,
`positional_embedding`, `transformer`, `ln_final`, `text_projection`)
individually addressable rather than calling the black-box `encode_text`,
because CoOp needs to intervene *between* tokenising and running the
transformer.

**`ContextTokenLearner`** (CoOp, Zhou et al. 2022) holds `M` learnable
continuous vectors, shared across all classes. For each class phrase it
builds `[SOT] + [ctx_1..ctx_M] + [class tokens, incl. EOT] + [pad]`, splices
`ctx` into the frozen embedding lookup at the right positions, and runs the
result through CLIP's frozen transformer — **fresh, every call**, not
cached. That is the one design constraint the whole module exists to serve:
gradients from the detection loss must flow back through the frozen
transformer into `ctx`. `n_ctx=0` degrades to the pure frozen embedding
(ablation (d)'s M=0 arm — hand-written prompts, no learnable component).

**`TextConditioner`** owns the encoder + learner and exposes `attach(model,
tgfem_layers)`, which registers `self` (not a closure — see §3) as a
`forward_pre_hook` on the root model. Every time the model is called, the
hook recomputes `T` and pushes it into each TGFEM layer's `.txt` (Phase 2's
attribute mechanism) and into `model.txt_feats` (for WorldDetect — see
`phase-notes/PHASE-5.md`).

---

## 3. A pickling bug worth recording

Ultralytics checkpoints the **whole model object** via `torch.save`
(pickle), not just a state dict. A `forward_pre_hook` registered as a
closure — `def _hook(module, args): ...` defined inside `attach()` — cannot
be pickled (`AttributeError: Can't pickle local object`), which only shows
up the first time a training run tries to save `last.pt`/`best.pt`, not at
attach time.

**Fix:** make `TextConditioner` itself the hook, via `__call__`, and
register the bound method. A bound method of a picklable object pickles
fine. This does mean the checkpoint now embeds the entire CLIP text tower
(confirmed: `best.pt` is 264 MB, dominated by the frozen encoder) — large,
but self-contained, and the same trade-off `WorldModel.clip_model` already
makes in stock Ultralytics.

---

## 4. Verification

Structural — no accuracy claim, same posture as Phase 2:

```
T shape: torch.Size([N, 512])
ctx grad exists after loss.backward(): yes, norm > 0
M=0 encode() == TextEncoder.encode_frozen(): exact match
checkpoint round-trip: hook survives pickling, TextEncoder + ctx intact
```

Full detail (including a real trap: gradient into `ctx` looked like zero in
a small smoke test purely because of Ultralytics' `nbs` gradient-accumulation
default, not a bug) is in `phase-notes/PHASE-5.md` §4, since it only became
observable once the language branch was wired into an actual training loop.

---

## 5. ⚠ NETWORK CONSTRAINT — read before trusting any embedding

This sandbox's outbound egress is allowlisted to a small set of hosts
(pypi, npm, a few others) and **does not include `huggingface.co`**, which
is where `open_clip` fetches `ViT-B-32`'s `openai` weights. Every attempt
here fell back to `TextEncoder`'s randomly-initialised path:

```
Could not download real CLIP weights (... 403 Forbidden).
Falling back to a RANDOMLY-INITIALISED ViT-B-32 text tower.
```

`encoder.pretrained_loaded` is `False` in every run this phase produced.
The **architecture and gradient flow are verified correct** — a random
encoder still has a real, differentiable forward pass, so shape/gradient/
checkpoint checks are meaningful — but **no embedding produced in this
sandbox carries any semantic content**. On the original dev machine (which
has normal internet access, per `phase-notes/PHASE-0.md`), this should just
work: `python scripts/train_tgfem_gate.py` and check
`pretrained CLIP : True` in its output.

---

## 6. Files created

```
src/tgfem/language.py    TextEncoder, ContextTokenLearner, TextConditioner
requirements.txt          + open_clip_torch==3.3.0
```

---

## 7. Gate check

| Criterion | Status |
|---|---|
| Frozen CLIP wrapper, 512-d output | Pass |
| Learnable context tokens, gradient reaches them | Pass |
| M=0 ablation arm reproduces frozen encoding exactly | Pass |
| Hook survives Ultralytics' pickle-based checkpointing | Pass |
| Real (non-random) CLIP weights loaded | **Blocked** — no route to huggingface.co in this sandbox |

**Phase 4 gate: PASSED structurally.** Re-run on a machine with real
internet access before trusting any embedding-space result — see §5.

---

## 8. Carried forward

1. Re-run `scripts/train_tgfem_gate.py` on a machine with internet access to
   confirm `pretrained_clip_loaded: true` before Phase 6/7 numbers are used
   for anything.
2. `class_texts_for()` (`src/tgfem/data.py`) fixes one phrase per class per
   run (tier selectable, default `natural`) — ablation (d) varies `--tier`
   and `--n-ctx` together on `scripts/train_tgfem.py`, it does not sample a
   different phrase every batch. That's a scope decision, not a limitation
   discovered by accident; revisit only if per-batch phrase sampling turns
   out to matter for robustness.
