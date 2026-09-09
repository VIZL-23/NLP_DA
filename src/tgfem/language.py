"""Phase 4 - the language branch.

Wraps a frozen CLIP text encoder (open_clip, ViT-B/32, 512-d output - see the
README correction "CLIP text dim is 512, not 256") plus a small number of
learnable context tokens (CoOp, Zhou et al. 2022: "Learning to Prompt for
Vision-Language Models"), and exposes the machinery that keeps every TGFEM
module's `.txt` populated with a fresh, differentiable text-embedding matrix
before each forward pass.

NETWORK NOTE - READ BEFORE TRUSTING ANY EMBEDDING-SPACE RESULT
----------------------------------------------------------------
Loading real CLIP weights requires reaching huggingface.co (open_clip's
weight host). Some machines have no route to it - this code was itself
developed on one. When the download fails, `TextEncoder` falls back to a
**randomly-initialised** transformer of the
identical open_clip ViT-B-32 architecture and output dimensionality, and
prints a loud warning instead of failing silently.

The fallback keeps the whole pipeline mechanically testable end to end
(shapes, gradients, checkpointing - exactly the Phase 2 "walking skeleton"
philosophy, applied to the language branch) on a machine with no internet
access. It produces embeddings with **no semantic content whatsoever** -
`encoder.pretrained_loaded` is False in that case, and every downstream
script must check it before reporting a number as a real result. Run once on
a machine with normal internet access to get real CLIP weights before Phase
6/7 numbers mean anything.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED_TAG = "openai"
CLIP_EMBED_DIM = 512  # matches the README correction: T is N x 512, not N x 256



def _find_local_openai_clip():
    """Return a path to a cached OpenAI CLIP ViT-B/32 checkpoint, or None.

    Searched in priority order; the first two are where Ultralytics puts it.
    """
    import os
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "weights" / "clip" / "ViT-B-32.pt",
        Path(os.path.expanduser("~")) / "AppData" / "Roaming" / "Ultralytics" / "weights" / "clip" / "ViT-B-32.pt",
        Path(os.path.expanduser("~")) / ".cache" / "clip" / "ViT-B-32.pt",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 100_000_000:  # guard against a truncated download
            return c
    return None


class TextEncoder(nn.Module):
    """Frozen CLIP text tower, exposed at the layer granularity CoOp needs.

    Ordinary `open_clip` usage only exposes `encode_text(token_ids)`, which
    is a black box from the tokenizer to the pooled feature. CoOp needs to
    intervene *between* those two steps - substitute the embeddings at a few
    token positions with learnable vectors, then run the (frozen) transformer
    on the result. So this class keeps the sub-modules (`token_embedding`,
    `positional_embedding`, `transformer`, `ln_final`, `text_projection`)
    individually addressable rather than calling `encode_text` directly.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        import open_clip

        self.pretrained_loaded = True
        try:
            # Prefer a locally cached OpenAI checkpoint over the network.
            # Ultralytics already downloads exactly this file (the original
            # OpenAI CLIP ViT-B/32, ~354 MB) for YOLO-World's own text tower,
            # so on a machine that has ever run YOLO-World the weights are
            # already on disk. Reaching huggingface.co for a second copy is
            # both redundant and, on restricted networks, the single point of
            # failure that silently degraded every Phase 4-6 run to a
            # RANDOMLY-INITIALISED encoder (phase-notes/PHASE-4.md section 5).
            local = _find_local_openai_clip()
            if local is not None:
                model = open_clip.load_openai_model(str(local), device="cpu")
            else:
                model, _, _ = open_clip.create_model_and_transforms(
                    CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED_TAG
                )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
            warnings.warn(
                f"\n{'!' * 70}\n"
                f"Could not download real CLIP weights ({exc.__class__.__name__}: {exc}).\n"
                f"Falling back to a RANDOMLY-INITIALISED {CLIP_MODEL_NAME} text tower.\n"
                f"Embeddings from this encoder carry NO semantic meaning - use only to\n"
                f"verify shapes/gradients/checkpointing. Re-run with internet access to\n"
                f"huggingface.co before trusting any Phase 6/7 number.\n"
                f"{'!' * 70}\n",
                stacklevel=2,
            )
            model, _, _ = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=None
            )
            self.pretrained_loaded = False

        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.context_length = model.context_length
        self.embed_dim = CLIP_EMBED_DIM
        self.token_dim = model.token_embedding.embedding_dim

        # Keep only the text tower; the vision tower is unused and dropped so
        # it is not carried around (and so it cannot be trained by accident).
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.register_buffer("attn_mask", model.attn_mask, persistent=False)

        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Frozen: always eval, regardless of what the parent Module.train() does.
        return super().train(False)

    @torch.no_grad()
    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return self.tokenizer(texts)

    def forward_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the frozen transformer on already-embedded token ids.

        Kept separate from `encode_frozen` so `ContextTokenLearner` can
        substitute a slice of the embedding sequence before calling this.
        """
        x = self.token_embedding(token_ids) + self.positional_embedding[: token_ids.shape[1]]
        x = self.transformer(x, attn_mask=self.attn_mask[: token_ids.shape[1], : token_ids.shape[1]])
        x = self.ln_final(x)
        pooled = x[torch.arange(x.shape[0], device=x.device), token_ids.argmax(dim=-1)]
        return pooled @ self.text_projection

    def forward_embeds(self, embeds: torch.Tensor, token_ids_for_pooling: torch.Tensor) -> torch.Tensor:
        """Same as `forward_ids`, but the caller has already built the
        embedding sequence (used by `ContextTokenLearner` to splice in
        learnable context vectors)."""
        x = embeds + self.positional_embedding[: embeds.shape[1]]
        x = self.transformer(x, attn_mask=self.attn_mask[: embeds.shape[1], : embeds.shape[1]])
        x = self.ln_final(x)
        pooled = x[torch.arange(x.shape[0], device=x.device), token_ids_for_pooling.argmax(dim=-1)]
        return pooled @ self.text_projection

    @torch.no_grad()
    def encode_frozen(self, texts: list[str]) -> torch.Tensor:
        """Bare CLIP embedding, M=0 context tokens - the hand-written-prompt
        arm of ablation (d). No gradient, safe to cache to disk."""
        ids = self.tokenize(texts).to(self.positional_embedding.device)
        return self.forward_ids(ids)


class ContextTokenLearner(nn.Module):
    """CoOp: M learnable continuous vectors, shared across all classes,
    prepended to the class-name tokens inside CLIP's frozen embedding space.

    forward() must be called fresh (not cached) on every training step:
    the point is that gradients from the detection loss flow back through
    the frozen CLIP transformer into `self.ctx`, so the text-embedding
    matrix `T` used by TG-FEM changes as training proceeds even though CLIP
    itself never updates.
    """

    def __init__(self, encoder: TextEncoder, n_ctx: int = 8):
        super().__init__()
        self.encoder = encoder
        self.n_ctx = n_ctx
        if n_ctx > 0:
            self.ctx = nn.Parameter(torch.empty(n_ctx, encoder.token_dim))
            nn.init.normal_(self.ctx, std=0.02)  # matches CLIP's own token-embedding init scale
        else:
            self.ctx = None  # M=0 ablation: identical to TextEncoder.encode_frozen

    def _build_ids(self, class_texts: list[str]) -> torch.Tensor:
        """[SOT] + M placeholder slots (overwritten below) + class tokens (+EOT), padded.

        The placeholder ids only need to (a) not collide with the EOT id used
        for pooling and (b) exist so the tensor has the right shape - their
        embeddings are replaced with `self.ctx` before the transformer runs.
        """
        L = self.encoder.context_length
        device = self.ctx.device if self.ctx is not None else "cpu"
        out = torch.zeros(len(class_texts), L, dtype=torch.long, device=device)
        for i, text in enumerate(class_texts):
            full = self.encoder.tokenize([text])[0]
            eot_pos = int(full.argmax(dim=-1))
            core = full[1 : eot_pos + 1]  # class tokens through EOT, SOT stripped
            core = core[: L - 1 - self.n_ctx]  # leave room for SOT + ctx
            out[i, 0] = full[0]  # SOT
            start = 1 + self.n_ctx
            out[i, start : start + len(core)] = core
        return out

    def forward(self, class_texts: list[str]) -> torch.Tensor:
        if self.n_ctx == 0 or self.ctx is None:
            return self.encoder.encode_frozen(class_texts)

        # DEVICE SYNC - do not remove.
        # `TextConditioner` holds the encoder as a plain attribute, not as a
        # registered submodule, so Ultralytics' `model.to(device)` never
        # reaches it: `ctx` (exposed to the optimiser as `model.tgfem_ctx`)
        # migrates to CUDA while the frozen CLIP tower stays on CPU. The next
        # `token_embedding(ids)` then raises
        #   "Expected all tensors to be on the same device, but got index is
        #    on cuda:0, different from other tensors on cpu".
        # `encode_frozen` already guards itself this way; the CoOp path did
        # not, which is why the n_ctx=0 ablation arm would have survived and
        # every n_ctx>0 run died at the first validation pass on GPU.
        if self.encoder.positional_embedding.device != self.ctx.device:
            self.encoder.to(self.ctx.device)

        ids = self._build_ids(class_texts)
        x = self.encoder.token_embedding(ids)  # (N, L, token_dim), frozen lookup
        ctx = self.ctx.unsqueeze(0).expand(len(class_texts), -1, -1)  # (N, M, token_dim)
        x = torch.cat([x[:, :1, :], ctx, x[:, 1 + self.n_ctx :, :]], dim=1)
        return self.encoder.forward_embeds(x, ids)


@dataclass
class TextConditionerConfig:
    n_ctx: int = 8
    device: str = "cpu"


class TextConditioner:
    """Owns the language branch and keeps every TGFEM.txt current.

    `parse_model` builds a single-input layer chain (Phase 2 finding), so a
    text tensor cannot be threaded through the model YAML. This mirrors
    YOLO-World's own solution: populate an attribute on each conditioned
    layer immediately before the forward pass, via a `forward_pre_hook` on
    the root model. That keeps Ultralytics completely unmodified - no fork,
    no patch - matching the project's Phase 2 registration approach.
    """

    def __init__(self, class_texts: list[str], n_ctx: int = 8, device: str = "cpu"):
        self.class_texts = list(class_texts)
        self.encoder = TextEncoder(device=device)
        self.learner = ContextTokenLearner(self.encoder, n_ctx=n_ctx)
        self.last_T: torch.Tensor | None = None

    def parameters(self):
        """Trainable parameters to hand to the optimiser (empty list if n_ctx=0)."""
        return [] if self.learner.ctx is None else [self.learner.ctx]

    def encode(self) -> torch.Tensor:
        self.last_T = self.learner(self.class_texts)
        return self.last_T

    def __call__(self, module: nn.Module, _args, _kwargs=None):
        """The forward_pre_hook body. A bound method rather than a nested
        closure deliberately - Ultralytics checkpoints the whole model via
        `torch.save` (pickle), including its `_forward_pre_hooks`, and a
        closure captured over local variables cannot be pickled. A bound
        method of this (picklable) object can."""
        T = self.encode()
        for layer in self._tgfem_layers:
            layer.txt = T
        if hasattr(module, "txt_feats"):
            # WorldDetect's contract (see detection_model.py): (1, N, d),
            # broadcast to batch size inside TGFEMModel.predict.
            module.txt_feats = T.unsqueeze(0)

    def attach(self, root_module: nn.Module, tgfem_layers: list[nn.Module]):
        """Register a pre-forward hook on the *root* model (the Ultralytics
        `DetectionModel`, i.e. `YOLO(...).model`) that recomputes T fresh -
        so gradients reach `self.ctx` - and pushes it into every TGFEM layer
        before that forward call runs. Returns the removable hook handle."""
        self._tgfem_layers = tgfem_layers
        return root_module.register_forward_pre_hook(self, with_kwargs=True)
