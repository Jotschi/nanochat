"""
Vision Transformer (ViT) encoder for the nanochat VLM.

Converts an image into a fixed-length sequence of visual token embeddings
that are compatible with the LLaMA-style GPT language model.

Design follows nanochat's conventions:
- no bias in linear layers
- RMSNorm (no learnable params) via F.rms_norm
- relu^2 activation in MLP
- nanochat's Linear class (casts weights to input dtype)
- meta-device compatible: __init__ does shapes only, init_weights() does data
- full (bidirectional) self-attention, no causal mask
- uses F.scaled_dot_product_attention (the ViT is small, no FA3 needed)

The output of the ViT is a sequence of (n_patches,) embeddings, each of
dimension llama_n_embd, ready to be concatenated with text embeddings.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Linear, norm


@dataclass
class ViTConfig:
    image_size: int = 128
    patch_size: int = 16
    embed_dim: int = 256
    n_layer: int = 4
    n_head: int = 4

    @property
    def n_patches(self):
        assert self.image_size % self.patch_size == 0, "image_size must be divisible by patch_size"
        grid = self.image_size // self.patch_size
        return grid * grid

    @property
    def head_dim(self):
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by n_head"
        return self.embed_dim // self.n_head


class ViTSelfAttention(nn.Module):
    """Full (bidirectional) multi-head self-attention with QK norm, matching GPT's style."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.c_q = Linear(self.embed_dim, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.embed_dim, self.n_head * self.head_dim, bias=False)
        self.c_v = Linear(self.embed_dim, self.n_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        # QK norm (matches GPT's CausalSelfAttention)
        q, k = norm(q), norm(k)
        # SDPA expects (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)  # full attention, no mask
        y = y.transpose(1, 2)  # back to (B, T, H, D)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class ViTBlock(nn.Module):
    """Pre-norm transformer block: attention + relu^2 MLP, both with residual connections."""

    def __init__(self, config):
        super().__init__()
        self.attn = ViTSelfAttention(config)
        self.c_fc = Linear(config.embed_dim, 4 * config.embed_dim, bias=False)
        self.c_proj = Linear(4 * config.embed_dim, config.embed_dim, bias=False)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.c_proj(F.relu(self.c_fc(norm(x))).square())
        return x


class ViT(nn.Module):
    """
    Vision Transformer encoder with a linear projector into the LLaMA hidden dim.

    forward(x): x is (B, 3, image_size, image_size) -> (B, n_patches, llama_n_embd)
    """

    def __init__(self, config, llama_n_embd):
        """
        NOTE: this __init__ runs in meta device context (like GPT.__init__).
        Only shapes/dtypes are computed here; actual data init happens in init_weights().
        """
        super().__init__()
        self.config = config
        self.llama_n_embd = llama_n_embd
        self.n_patches = config.n_patches
        # Patch embedding: a strided conv that turns (3, H, W) into (n_patches, embed_dim)
        self.patch_embed = nn.Conv2d(3, config.embed_dim, kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        # Learnable positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, config.embed_dim))
        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(config) for _ in range(config.n_layer)])
        # Final norm before projection
        self.ln_out = nn.Identity()  # we use F.rms_norm (no params) in forward
        # Projector: ViT hidden dim -> LLaMA hidden dim
        self.projector = Linear(config.embed_dim, llama_n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize all parameters (called after to_empty, like GPT.init_weights)."""
        # Patch embedding: uniform with std=1/sqrt(3 * patch^2) (like a conv init)
        p = self.config.patch_size
        s = 3**0.5 * (3 * p * p) ** -0.5
        torch.nn.init.uniform_(self.patch_embed.weight, -s, s)
        # Positional embeddings: small normal
        torch.nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        # Blocks: same scheme as GPT (uniform for q/k/v/fc, zeros for projections)
        s = 3**0.5 * self.config.embed_dim ** -0.5
        for block in self.blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.c_proj.weight)
        # Projector: small normal (like lm_head init) so initial visual tokens are small
        torch.nn.init.normal_(self.projector.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (B, 3, image_size, image_size)
        returns: (B, n_patches, llama_n_embd)
        """
        B = x.size(0)
        # Patch embed: (B, 3, H, W) -> (B, embed_dim, grid, grid) -> (B, n_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.to(COMPUTE_DTYPE)
        # Add positional embeddings
        x = x + self.pos_embed.to(x.dtype)
        x = norm(x)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = norm(x)
        # Project into LLaMA hidden dim
        x = self.projector(x)
        return x
