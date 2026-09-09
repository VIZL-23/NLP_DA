"""Phase 5/6 - wires the language branch into an Ultralytics training run.

`parse_model` builds a single-input layer chain (Phase 2 finding), so the
text embedding matrix `T` cannot be threaded through the model YAML, and
Ultralytics' high-level `YOLO(...).train(...)` has no concept of a second,
non-image input either. YOLO-World solves an analogous problem (see
`ultralytics/models/yolo/world/train.py`) by subclassing `DetectionTrainer`
and overriding `get_model`. TG-FEM follows the same pattern here, with one
difference: YOLO-World's text branch is fully frozen and its embeddings are
cached to disk once; TG-FEM's context tokens are *learnable* (report
ablation (d)), so `T` must be recomputed fresh on every forward pass for
gradients to reach them, and its parameter must be visible to the trainer's
optimiser.

How the pieces connect:
    1. `get_model` builds the DetectionModel from the YAML as usual, then
       locates every `TGFEM` layer in it.
    2. A `TextConditioner` (see `language.py`) is built once and its
       `attach()` registers a `forward_pre_hook` on the model: every forward
       call recomputes `T = ContextTokenLearner(class_texts)` and pushes it
       into each TGFEM layer's `.txt` before that layer runs.
    3. The learnable context vector `ctx` is additionally exposed as
       `model.tgfem_ctx` - a bare attribute assignment of the *same*
       `nn.Parameter` object already owned by the `ContextTokenLearner`
       (a standard PyTorch weight-tying pattern, e.g. tied embeddings).
       This is what makes `Trainer.build_optimizer` (which walks
       `model.named_modules()` generically - it has no notion of TG-FEM)
       pick `ctx` up automatically. Only `ctx` is exposed this way, not the
       whole frozen CLIP encoder - registering the encoder too would make
       the optimiser walk (and needlessly track state for) ~40M dead
       parameters with `requires_grad=False`.
"""

from __future__ import annotations

from typing import Any

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer

from .detection_model import TGFEMModel
from .language import TextConditioner
from .module import TGFEM


class TGFEMTrainer(DetectionTrainer):
    """DetectionTrainer + a language branch feeding every TGFEM layer.

    Args (via `overrides`, same as `DetectionTrainer`): the usual `model`,
    `data`, `epochs`, etc. Two TG-FEM-specific knobs are read directly from
    `overrides` (not `DEFAULT_CFG`, since they are not stock Ultralytics
    args): `class_texts` (list[str], required) and `n_ctx` (int, default 8 -
    ablation (d) varies this; `n_ctx=0` reproduces hand-written prompts with
    no learnable component).
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        overrides = dict(overrides or {})
        self.class_texts: list[str] = overrides.pop("class_texts")
        self.n_ctx: int = overrides.pop("n_ctx", 8)
        self.text_conditioner: TextConditioner | None = None
        self._hook_handle = None
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights: str | None = None, verbose: bool = True) -> TGFEMModel:
        model = TGFEMModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data.get("channels", 3),
            nc=self.data["nc"],
            verbose=verbose,
        )
        if weights:
            model.load(weights)

        tgfem_layers = [m for m in model.model if isinstance(m, TGFEM)]

        # A model with NO TGFEM layers is legitimate: ablation (g)'s control is
        # CBAM + WorldDetect (cfg/yolo11-cbam-worlddetect.yaml). It still needs
        # the language branch, because WorldDetect classifies by comparing
        # region embeddings against `txt_feats` - it just has no text-gated
        # attention. Requiring TGFEM layers here made the one comparison the
        # paper actually rests on impossible to run.
        #
        # What must be rejected is a model that needs no text at all (a stock
        # `Detect` head and no TGFEM) - that belongs in train_baseline.py,
        # where Phase 3's stock/cbam runs live.
        from ultralytics.nn.modules.head import WorldDetect

        head = model.model[-1]
        needs_text = isinstance(head, WorldDetect)
        if not tgfem_layers and not needs_text:
            raise RuntimeError(
                "TGFEMTrainer was given a model with neither TGFEM layers nor a "
                "WorldDetect head, so nothing in it consumes text. Use "
                "scripts/train_baseline.py for text-free variants."
            )

        device = next(model.parameters()).device
        self.text_conditioner = TextConditioner(self.class_texts, n_ctx=self.n_ctx, device=str(device))
        self._hook_handle = self.text_conditioner.attach(model, tgfem_layers)

        if self.text_conditioner.learner.ctx is not None:
            # Same Parameter object as `learner.ctx` (weight tying) - not a
            # copy - so gradients that land on either reference accumulate
            # onto the one underlying tensor. This is what makes it visible
            # to `Trainer.build_optimizer`, which only walks `model`.
            model.tgfem_ctx = self.text_conditioner.learner.ctx

        return model

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess_batch(batch)
        # Keep the text branch on the same device as the batch/model - a
        # no-op after the first call, cheap to check every time.
        device = batch["img"].device
        self.text_conditioner.encoder.to(device)
        self.text_conditioner.learner.to(device)
        return batch
