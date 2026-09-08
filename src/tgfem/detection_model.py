"""TGFEMModel - the `DetectionModel` subclass that makes the region-text
contrastive head (README component table, row 5) actually work.

TG-FEM's own attention (module.py) only *conditions features* - it does not
by itself let the model classify a box into a class it never saw at train
time. That requires the classification branch itself to compare each
region's embedding against text embeddings rather than run a fixed N-way
softmax, i.e. Ultralytics' own `WorldDetect` + `ContrastiveHead` (already
shipped - "stand on Ultralytics, don't rewrite it", README build strategy
#1). `cfg/yolo11-tgfem.yaml`'s head is `WorldDetect`, not stock `Detect`.

`WorldDetect.forward(x, text)` needs `text` passed explicitly - `parse_model`
builds a single-input layer chain (Phase 2 finding), so the generic
`for m in self.model: x = m(x)` loop cannot thread a second argument to one
specific layer. Ultralytics' own `WorldModel.predict()` solves this by
special-casing `WorldDetect` inside the loop; this class does the same,
reusing `self.txt_feats` - refreshed every forward call by
`language.TextConditioner`'s pre-hook - instead of YOLO-World's frozen,
externally-cached `set_classes()` value.

TGFEM layers need no special-casing here at all: they read `self.txt`,
already populated by the same pre-hook, through their own single-input
`forward(x)` (Phase 2's whole point in choosing that mechanism).
"""

from __future__ import annotations

import torch

from ultralytics.nn.modules import WorldDetect
from ultralytics.nn.tasks import DetectionModel


class TGFEMModel(DetectionModel):
    """DetectionModel + WorldDetect text threading. See module docstring."""

    def __init__(self, cfg="yolo11-tgfem.yaml", ch=3, nc=None, verbose=True):
        self.txt_feats = torch.randn(1, nc or 1, 512)  # placeholder; overwritten before first real forward
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def predict(self, x, profile=False, txt_feats=None, augment=False, embed=None):
        """Same contract as `BaseModel.predict`, plus `txt_feats` threading
        into `WorldDetect`. Mirrors `ultralytics.nn.tasks.WorldModel.predict`,
        trimmed to what this architecture actually uses (no C2fAttn /
        ImagePoolingAttn - those are YOLO-World-specific neck blocks this
        project's cfg does not include)."""
        if augment:
            return self._predict_augment(x)

        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0]:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, WorldDetect):
                x = m(x, txt_feats)
            else:
                x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        if preds is None:
            # Deliberately `self.predict(...)`, not `self(...)`: the outer
            # `model(batch)` call that reached this method already fired the
            # TextConditioner pre-hook once (see language.py / trainer.py
            # module docstrings) - `self.txt_feats` and every TGFEM.txt are
            # already fresh for this step. Calling `self(...)` again here
            # would re-fire the hook and recompute T a second time for
            # nothing.
            preds = self.predict(batch["img"])
        return self.criterion(preds, batch)
