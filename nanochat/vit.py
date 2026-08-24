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


# -----------------------------------------------------------------------------
# DINOv2 ViT-B/14 encoder (native PyTorch, no HF dependency)
#
# Mirrors the exact structure of the official DINOv2 ViT-B/14 checkpoint so the
# pretrained weights load directly. Uses LayerNorm + GELU + fused QKV + LayerScale
# (the DINOv2 recipe), unlike the from-scratch ViT above which uses nanochat's
# RMSNorm + relu^2 + separate QKV.
# -----------------------------------------------------------------------------

@dataclass
class DinoViTConfig:
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 768
    n_layer: int = 12
    n_head: int = 12
    mlp_ratio: int = 4
    weights_path: str = ""  # path to dinov2_vitb14_pretrain.pth

    @property
    def n_patches(self):
        assert self.image_size % self.patch_size == 0, "image_size must be divisible by patch_size"
        grid = self.image_size // self.patch_size
        return grid * grid

    @property
    def head_dim(self):
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by n_head"
        return self.embed_dim // self.n_head


class DinoMLP(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, dim, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DinoAttention(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DinoBlock(nn.Module):
    def __init__(self, dim, n_head, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DinoAttention(dim, n_head)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = DinoMLP(dim, dim * mlp_ratio)
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DinoViT(nn.Module):
    """
    DINOv2 ViT-B/14 encoder with a linear projector into the LLaMA hidden dim.

    forward(x): x is (B, 3, image_size, image_size) -> (B, n_patches, llama_n_embd)
    The CLS token is dropped; only the 256 patch tokens are projected.
    """

    def __init__(self, config, llama_n_embd):
        super().__init__()
        self.config = config
        self.llama_n_embd = llama_n_embd
        self.n_patches = config.n_patches
        self.embed_dim = config.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_patches + 1, config.embed_dim))
        self.patch_embed = nn.Conv2d(3, config.embed_dim, kernel_size=config.patch_size,
                                     stride=config.patch_size, bias=True)
        self.blocks = nn.ModuleList(
            [DinoBlock(config.embed_dim, config.n_head, config.mlp_ratio) for _ in range(config.n_layer)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.projector = Linear(config.embed_dim, llama_n_embd, bias=False)
        self._weights_loaded = False

    @torch.no_grad()
    def init_weights(self):
        """Only the projector is randomly initialized; the backbone is loaded from DINOv2."""
        torch.nn.init.normal_(self.projector.weight, mean=0.0, std=0.02)

    @staticmethod
    def _interpolate_pos_embed(pos_embed, n_patches):
        """Interpolate DINOv2's stored pos_embed (for 518px / 37x37) to the target grid."""
        N, D = pos_embed.shape[1] - 1, pos_embed.shape[2]
        old_size = int(N ** 0.5)
        new_size = int(n_patches ** 0.5)
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :].reshape(1, old_size, old_size, D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, D)
        return torch.cat([cls_pos, patch_pos], dim=1)

    @torch.no_grad()
    def load_dinov2_weights(self, path):
        """Load the official DINOv2 ViT-B/14 pretrained weights into the backbone."""
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.pos_embed.data.copy_(self._interpolate_pos_embed(sd["pos_embed"], self.n_patches))
        self.cls_token.data.copy_(sd["cls_token"])
        self.patch_embed.weight.data.copy_(sd["patch_embed.proj.weight"])
        self.patch_embed.bias.data.copy_(sd["patch_embed.proj.bias"])
        for i, block in enumerate(self.blocks):
            p = f"blocks.{i}."
            block.norm1.weight.data.copy_(sd[p + "norm1.weight"])
            block.norm1.bias.data.copy_(sd[p + "norm1.bias"])
            block.attn.qkv.weight.data.copy_(sd[p + "attn.qkv.weight"])
            block.attn.qkv.bias.data.copy_(sd[p + "attn.qkv.bias"])
            block.attn.proj.weight.data.copy_(sd[p + "attn.proj.weight"])
            block.attn.proj.bias.data.copy_(sd[p + "attn.proj.bias"])
            block.ls1.data.copy_(sd[p + "ls1.gamma"])
            block.norm2.weight.data.copy_(sd[p + "norm2.weight"])
            block.norm2.bias.data.copy_(sd[p + "norm2.bias"])
            block.mlp.fc1.weight.data.copy_(sd[p + "mlp.fc1.weight"])
            block.mlp.fc1.bias.data.copy_(sd[p + "mlp.fc1.bias"])
            block.mlp.fc2.weight.data.copy_(sd[p + "mlp.fc2.weight"])
            block.mlp.fc2.bias.data.copy_(sd[p + "mlp.fc2.bias"])
            block.ls2.data.copy_(sd[p + "ls2.gamma"])
        self.norm.weight.data.copy_(sd["norm.weight"])
        self.norm.bias.data.copy_(sd["norm.bias"])
        self._weights_loaded = True

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, D, g, g)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, D)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x[:, 1:, :]  # drop CLS token -> (B, n_patches, D)
        x = self.projector(x)
        return x
