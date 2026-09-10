"""Standalone inference: image + text queries -> boxes.

Deliberately free of any web/CLI framework so the same object backs
`scripts/demo.py`, the local FastAPI app, and any future deployment. The web
layer should own HTTP and nothing else.

Usage:
    det = DefectDetector("runs/.../best.pt")
    for d in det.predict("image.jpg", ["scratches on the steel surface"]):
        print(d.query, d.conf, d.xyxy)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass
class Detection:
    """One box, in ORIGINAL image pixel coordinates."""

    query: str
    query_index: int
    conf: float
    xyxy: tuple[int, int, int, int]

    def as_dict(self) -> dict:
        x1, y1, x2, y2 = self.xyxy
        return {
            "query": self.query,
            "query_index": self.query_index,
            "conf": round(self.conf, 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        }


def _resolve_device(device: str | None) -> str:
    """Normalise a device spec to something torch accepts everywhere.

    Ultralytics tolerates a bare "0"; `torch.load(map_location=...)` and
    `Module.to(...)` do not - they raise "don't know how to restore data
    location" and "Invalid device string: '0'" respectively. Normalise once,
    here, so no caller has to remember.
    """
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    device = str(device)
    return f"cuda:{device}" if device.isdigit() else device


class DefectDetector:
    """A loaded TG-FEM checkpoint, queryable with arbitrary text.

    The checkpoint carries its own `TextConditioner` (Phase 4's pickling
    design), so the learned context tokens travel with it and inference reuses
    the exact `ctx` training produced - only the query strings are swapped.

    NOT thread-safe: `predict` mutates the model's vocabulary for the duration
    of the call. Fine for a single-user local server; a concurrent deployment
    needs one instance per worker.
    """

    def __init__(self, checkpoint: str | Path, device: str | None = None, imgsz: int = 640):
        from tgfem import register
        from tgfem.detection_model import TGFEMModel  # noqa: F401 - needed for unpickling
        from tgfem.language import TextConditioner

        register()
        self.checkpoint = Path(checkpoint)
        self.device = _resolve_device(device)
        self.imgsz = imgsz

        ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
        self.model = model.float().to(self.device).eval()

        hooks = [h for h in self.model._forward_pre_hooks.values()
                 if isinstance(h, TextConditioner)]
        if not hooks:
            raise ValueError(
                f"{self.checkpoint} has no TextConditioner attached - it was not "
                "trained by scripts/train_tgfem.py, so it cannot be queried with text."
            )
        self._tc = hooks[0]

        # The vocabulary the model was actually trained on. Read from the
        # checkpoint rather than hardcoded anywhere, so swapping in a model
        # with a different taxonomy needs no code change.
        self.trained_vocabulary: list[str] = list(self._tc.class_texts)
        self.n_ctx: int = self._tc.learner.n_ctx
        self.clip_pretrained: bool = self._tc.encoder.pretrained_loaded

    # -- internals --------------------------------------------------------

    @staticmethod
    def _nms():
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:  # older Ultralytics layout
            from ultralytics.utils.ops import non_max_suppression
        return non_max_suppression

    def _load_image(self, image: str | Path | np.ndarray) -> np.ndarray:
        import cv2

        if isinstance(image, np.ndarray):
            return image
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"could not read image: {image}")
        return img

    # -- public API -------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | np.ndarray,
        queries: Sequence[str],
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> list[Detection]:
        """Detect regions matching each query. Returns boxes in original pixels."""
        import cv2

        queries = [q.strip() for q in queries if q and q.strip()]
        if not queries:
            return []

        img = self._load_image(image)
        h0, w0 = img.shape[:2]

        resized = cv2.resize(img, (self.imgsz, self.imgsz))
        tensor = (
            torch.from_numpy(resized[:, :, ::-1].copy())
            .permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        )

        # Swap the vocabulary. `WorldDetect.forward` derives `self.no` from
        # `self.nc`, which is only updated when the vocabulary size changes -
        # mirroring WorldModel.set_classes()'s own `self.model[-1].nc = len(text)`.
        self._tc.class_texts = list(queries)
        self.model.model[-1].nc = len(queries)

        y, _ = self.model(tensor)
        dets = self._nms()(y, conf_thres=conf, iou_thres=iou, nc=len(queries))[0]

        sx, sy = w0 / self.imgsz, h0 / self.imgsz
        out = []
        for *xyxy, score, cls in dets.tolist():
            idx = int(cls)
            out.append(Detection(
                query=queries[idx],
                query_index=idx,
                conf=float(score),
                xyxy=(
                    max(0, int(xyxy[0] * sx)), max(0, int(xyxy[1] * sy)),
                    min(w0, int(xyxy[2] * sx)), min(h0, int(xyxy[3] * sy)),
                ),
            ))
        out.sort(key=lambda d: d.conf, reverse=True)
        return out

    @property
    def text_discriminative(self) -> bool:
        """Whether the query text actually changes this model's output.

        A single-class checkpoint's contrastive head never had to separate one
        text embedding from another - the loss only ever demanded "object vs
        background" - so at inference every query, including nonsense, returns
        the same boxes. Measured on the crack specialist: 'a crack in the
        concrete surface' 0.6967, 'banana' 0.6941, 'a happy elephant' 0.6940 on
        the same image, against a 6-class model where 'banana' correctly returns
        nothing at all (phase-notes/PHASE-9.md section 4).

        This is a property of the vocabulary size, not of the training data, so
        it can be read straight off the checkpoint. Callers should surface it -
        presenting a single-class detector as text-guided would be a false
        claim about what the model does.
        """
        return len(self.trained_vocabulary) > 1

    def info(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "imgsz": self.imgsz,
            "n_ctx": self.n_ctx,
            "clip_pretrained": self.clip_pretrained,
            "trained_vocabulary": self.trained_vocabulary,
            "text_discriminative": self.text_discriminative,
        }
