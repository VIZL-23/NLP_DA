"""TG-FEM - Text-Guided Feature Enhancement Module.

PHASE 5 STATUS: real maths (report section 4.2), implemented as:

    1. Projection            1x1 conv, C -> d              (self.proj)
    2. Region-text attention A = softmax(F_q T^T / sqrt(d));  S = A T
    3. Dual gating from S    channel gate g (self.channel_gate),
                             spatial gate s (self.spatial_gate)
    4. Residual output       F' = F + F (*) g (*) s

The contribution is that BOTH gates are derived from S (the text-conditioned
region response), not from F's own pooled statistics - the sole structural
difference from CBAM (feature-derived gates) and from YOLO-World's
channel-only MaxSigmoidAttnBlock.

REPORT CORRECTION APPLIED (README "Known report corrections", item 3)
-----------------------------------------------------------------------
The report states `F' = F(*)g(*)s + F`, which is mathematically the same
formula used here, but was previously mis-described as reducing to identity
when g=s=1 (it reduces to 2F instead). The correct identity condition is
**gates -> 0**, not gates -> 1. This module is built consistently with that:
`channel_gate` and `spatial_gate` are zero-weight/negative-bias initialised
(sigmoid(-4) ~= 0.018), so at the start of training F' ~= F - a near-identity
initialisation, the same trick used for zero-init residual branches elsewhere
(e.g. ReZero, CBAM's own gate init). It is *not* an exact identity (ablation
(a) below covers that case separately).

CHANNEL COUNT: parse_model has no special case for TGFEM (Phase 2 finding),
so it never receives c1 (input channels) - only the YAML args (`d`, and
optionally `identity`). `proj` (C->d) and `channel_gate` (d->C) are therefore
built lazily, from the actual input shape, the first time the module is
called - see `_materialise`. Both `identity=True` and `identity=False` call
`_materialise` unconditionally, so ablation (a) ("TG-FEM removed") can be run
at the *same parameter budget* by toggling `identity` rather than deleting
the layer - the whole reason this matters is explained in
phase-notes/PHASE-2.md section 4.
"""

import torch
import torch.nn as nn


class TGFEM(nn.Module):
    """Text-guided feature enhancement, inserted at one pyramid level.

    Shape contract (must hold in every phase):
        input  : (B, C, H, W)
        output : (B, C, H, W)   - identical shape

    Args:
        d: text-embedding / query dimensionality (512 for CLIP ViT-B/32 -
            see the README correction "CLIP text dim is 512, not 256").
        identity: when True, materialises the same parameters as the active
            module (see CHANNEL COUNT above) but the forward pass is an exact
            pass-through. This is ablation (a) in the report.

    Text input:
        `self.txt` is populated by the parent model immediately before the
        forward pass (the same pattern YOLO-World uses for `txt_feats`, and
        the mechanism is `tgfem.language.TextConditioner.attach`), because
        `parse_model` builds a single-input layer chain and cannot thread a
        second argument through. It stays None until a TextConditioner is
        attached (e.g. during Phase 2-style structural checks) - in that
        case the module also falls back to identity, since there is no text
        to condition on.
    """

    def __init__(self, d: int = 512, identity: bool = False):
        super().__init__()
        self.d = d
        self.identity = identity
        # Not a buffer or parameter: it is cached text, set per-forward, and
        # must never be captured in the state dict or the autograd graph
        # beyond the single forward pass that uses it.
        self.txt = None

        # Built lazily in `_materialise` once the real input channel count
        # is known (see module docstring, CHANNEL COUNT).
        self.proj: nn.Conv2d | None = None
        self.channel_gate: nn.Linear | None = None
        # d -> 1 has no dependence on C, so this one can be built eagerly.
        self.spatial_gate = nn.Linear(d, 1)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.constant_(self.spatial_gate.bias, -4.0)
        self._c: int | None = None

    def _materialise(self, c: int, device, dtype) -> None:
        if self.proj is not None:
            if c != self._c:
                raise RuntimeError(
                    f"TGFEM was built for {self._c} input channels but saw {c}. "
                    "Each TGFEM instance is placed at exactly one pyramid level "
                    "(one fixed channel count) - this indicates a YAML wiring bug."
                )
            return
        self._c = c
        self.proj = nn.Conv2d(c, self.d, kernel_size=1).to(device=device, dtype=dtype)
        self.channel_gate = nn.Linear(self.d, c).to(device=device, dtype=dtype)
        nn.init.zeros_(self.channel_gate.weight)
        nn.init.constant_(self.channel_gate.bias, -4.0)  # sigmoid(-4) ~= 0.018 -> near-identity at init
        self.spatial_gate.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        self._materialise(c, x.device, x.dtype)

        if self.identity or self.txt is None:
            return x

        text = self.txt.to(device=x.device, dtype=x.dtype)  # (N, d)

        f_q = self.proj(x).flatten(2).transpose(1, 2)  # (B, HW, d)
        attn = torch.softmax(f_q @ text.t() / (self.d ** 0.5), dim=-1)  # (B, HW, N)
        s = attn @ text  # (B, HW, d) - the text-conditioned region response

        g = torch.sigmoid(self.channel_gate(s.mean(dim=1))).view(b, c, 1, 1)  # (B, C, 1, 1)
        sp = torch.sigmoid(self.spatial_gate(s)).transpose(1, 2).view(b, 1, h, w)  # (B, 1, H, W)

        return x + x * g * sp

    def extra_repr(self) -> str:
        return f"d={self.d}, identity={self.identity}, c={self._c}"
