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
       locates every `TGFEM` layer in it (zero is valid - ablation (g)'s
       CBAM control uses this same trainer with no TGFEM layers at all,
       since it still needs the WorldDetect head fed).
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

from typing import Any, cast

from torch import nn
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import WorldDetect

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
        # Optional list[list[str]] - all corpus phrasings per class, in class
        # index order. When present the language branch resamples a phrasing
        # per training step (see TextConditioner). None = the pre-Phase-7
        # behaviour of one fixed phrase per class for the whole run.
        self.phrase_pools: list[list[str]] | None = overrides.pop("phrase_pools", None)
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

        # BaseModel.model is untyped upstream (mypy infers a Tensor|Module
        # union from unrelated call sites in Ultralytics' own code) - it is
        # always the parse_model-built nn.Sequential at runtime.
        layers = cast(nn.Sequential, model.model)
        tgfem_layers: list[nn.Module] = [m for m in layers if isinstance(m, TGFEM)]
        # TGFEM layers are the module ablation (a) toggles to identity and
        # ablation (g) replaces with CBAM entirely - zero is a legitimate
        # count for cbam_worlddetect (README, ablation (g): same head, only
        # the gate source differs). What every variant this trainer runs
        # DOES need is the WorldDetect head, since that's what the language
        # branch actually threads text into (see module docstring).
        if not isinstance(layers[-1], WorldDetect):
            raise RuntimeError(
                "TGFEMTrainer was asked to train a model with no WorldDetect head - "
                "wrong cfg? (expected cfg/yolo11-tgfem*.yaml or yolo11-cbam-worlddetect.yaml)"
            )

        device = next(model.parameters()).device
        self.text_conditioner = TextConditioner(
            self.class_texts,
            n_ctx=self.n_ctx,
            device=str(device),
            phrase_pools=self.phrase_pools,
        )
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
        # get_model() always runs (and sets this) before any batch is
        # preprocessed - DetectionTrainer._setup_train() calls setup_model()
        # ahead of building the dataloaders that produce batches.
        assert self.text_conditioner is not None
        # Keep the text branch on the same device as the batch/model - a
        # no-op after the first call, cheap to check every time.
        device = batch["img"].device
        self.text_conditioner.encoder.to(device)
        self.text_conditioner.learner.to(device)
        return batch
