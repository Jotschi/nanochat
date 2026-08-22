"""
Test the ViT encoder.

Run: python -m pytest tests/test_vit.py -v

All tests run on CPU with tiny configs.
"""

import torch
import pytest

from nanochat.vit import ViT, ViTConfig


def build_vit(config, llama_n_embd, device="cpu"):
    """Build a ViT on meta device, move to device, init weights (nanochat pattern)."""
    with torch.device("meta"):
        vit = ViT(config, llama_n_embd)
    vit.to_empty(device=device)
    vit.init_weights()
    return vit


def tiny_config():
    # 32x32 image, 8x8 patches -> 4x4 grid = 16 patches
    return ViTConfig(image_size=32, patch_size=8, embed_dim=32, n_layer=2, n_head=4)


class TestViTConfig:
    def test_n_patches(self):
        cfg = ViTConfig(image_size=128, patch_size=16)
        assert cfg.n_patches == 64  # 8x8 grid

    def test_n_patches_tiny(self):
        cfg = tiny_config()
        assert cfg.n_patches == 16  # 4x4 grid

    def test_head_dim(self):
        cfg = tiny_config()
        assert cfg.head_dim == 8  # 32 // 4

    def test_invalid_patch(self):
        cfg = ViTConfig(image_size=30, patch_size=8)
        with pytest.raises(AssertionError):
            _ = cfg.n_patches


class TestViTForward:
    def test_output_shape(self):
        vit = build_vit(tiny_config(), llama_n_embd=64)
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        assert out.shape == (2, 16, 64)  # (B, n_patches, llama_n_embd)

    def test_output_shape_default_config(self):
        # Use a smaller llama dim but the default ViT config dimensions
        cfg = ViTConfig(image_size=32, patch_size=8, embed_dim=32, n_layer=1, n_head=4)
        vit = build_vit(cfg, llama_n_embd=32)
        x = torch.randn(1, 3, 32, 32)
        out = vit(x)
        assert out.shape == (1, 16, 32)

    def test_deterministic(self):
        vit = build_vit(tiny_config(), llama_n_embd=64)
        vit.eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out1 = vit(x)
            out2 = vit(x)
        assert torch.allclose(out1, out2)


class TestViTGradients:
    def test_gradient_flow(self):
        vit = build_vit(tiny_config(), llama_n_embd=64)
        vit.train()
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        loss = out.sum()
        loss.backward()
        # All parameters should have a gradient tensor (flow reaches every param).
        for name, param in vit.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
        # Params with non-zero init must receive non-zero gradients.
        # (c_proj / mlp.c_proj are zero-init like GPT, so their upstream weights
        # legitimately get zero grad on the very first step.)
        for name in ["patch_embed.weight", "pos_embed", "projector.weight"]:
            g = dict(vit.named_parameters())[name].grad
            assert g.abs().sum() > 0, f"Zero gradient for {name}"

    def test_patch_embed_gradient(self):
        vit = build_vit(tiny_config(), llama_n_embd=64)
        x = torch.randn(1, 3, 32, 32)
        out = vit(x)
        out.sum().backward()
        assert vit.patch_embed.weight.grad is not None


class TestViTInit:
    def test_weights_initialized(self):
        vit = build_vit(tiny_config(), llama_n_embd=64)
        # patch_embed should not be all zeros
        assert vit.patch_embed.weight.abs().sum() > 0
        # pos_embed should be small but non-zero
        assert vit.pos_embed.abs().sum() > 0
        # projector should be non-zero
        assert vit.projector.weight.abs().sum() > 0

    def test_meta_device_init(self):
        """Verify the meta-device init pattern works (like GPT)."""
        cfg = tiny_config()
        with torch.device("meta"):
            vit = ViT(cfg, llama_n_embd=64)
        # On meta device, params have no storage
        assert vit.patch_embed.weight.device.type == "meta"
        vit.to_empty(device="cpu")
        vit.init_weights()
        assert vit.patch_embed.weight.device.type == "cpu"
        # Forward should work after init
        x = torch.randn(1, 3, 32, 32)
        out = vit(x)
        assert out.shape == (1, 16, 64)


class TestViTAttention:
    def test_full_attention_not_causal(self):
        """ViT uses full (bidirectional) attention, so output at position 0
        should depend on later positions (unlike causal attention)."""
        from nanochat.vit import ViTSelfAttention
        cfg = tiny_config()
        attn = ViTSelfAttention(cfg)
        attn.to_empty(device="cpu")
        # init
        s = 3**0.5 * cfg.embed_dim ** -0.5
        torch.nn.init.uniform_(attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(attn.c_proj.weight)
        attn.eval()
        x = torch.randn(1, 4, cfg.embed_dim)
        with torch.no_grad():
            out = attn(x)
        assert out.shape == (1, 4, cfg.embed_dim)
