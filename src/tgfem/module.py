"""TG-FEM - Text-Guided Feature Enhancement Module.

PHASE 2 STATUS: identity pass-through only.

This is deliberately a no-op. Phase 2 is the "walking skeleton": we prove the
module can be registered with Ultralytics, placed in a model YAML at all three
pyramid levels, and trained end-to-end - *before* any of the real maths exists.
That way integration risk is eliminated while the module is still trivial to
debug.

Phase 5 fills in the four steps from report section 4.2:
    1. Projection            1x1 conv, C -> d, flattened to (HW, d)
    2. Region-text attention A = softmax(F_q T^T / sqrt(d));  S = A T
    3. Dual gating from S    channel gate g, spatial gate s
    4. Residual output       F' = F (*) g (*) s + F

The contribution is that BOTH gates are derived from S (the text), not from F's
own pooled statistics - which is the sole structural difference from CBAM and
from YOLO-World's channel-only MaxSigmoidAttnBlock.
"""

import torch
import torch.nn as nn


class TGFEM(nn.Module):
    """Text-guided feature enhancement, inserted at one pyramid level.

    Shape contract (must hold in every phase):
        input  : (B, C, H, W)
        output : (B, C, H, W)   - identical shape

    Ultralytics' `parse_model` has no special case for this class, so it falls
    through to `else: c2 = ch[f]`, which preserves the channel count and passes
    the YAML args verbatim. The shape contract above is what makes that safe.

    Args:
        d: text-embedding dimensionality used by the Phase 5 attention.
        identity: when True the module is an exact pass-through. This is also
            how ablation (a) in the report is run - the module stays in the
            graph at the same parameter budget but contributes nothing.

    Text input:
        `self.txt` is populated by the parent model immediately before the
        forward pass (the same pattern YOLO-World uses for `txt_feats`),
        because `parse_model` builds a single-input layer chain and cannot
        thread a second argument through. It stays None until Phase 4.
    """

    def __init__(self, d: int = 256, identity: bool = True):
        super().__init__()
        self.d = d
        self.identity = identity
        # Not a buffer or parameter: it is cached text, set per-forward, and
        # must never be captured in the state dict or the autograd graph.
        self.txt = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity or self.txt is None:
            return x
        raise NotImplementedError(
            "TG-FEM maths lands in Phase 5. Until then run with identity=True."
        )

    def extra_repr(self) -> str:
        return f"d={self.d}, identity={self.identity}"
