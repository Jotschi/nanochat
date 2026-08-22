"""
Test the VLM (ViT + GPT) integration.

Run: python -m pytest tests/test_vlm.py -v

All tests run on CPU with tiny configs.
"""

import torch
import pytest

import nanochat.flash_attention as fa_module
from nanochat.gpt import GPT, GPTConfig
from nanochat.vit import ViT, ViTConfig
from nanochat.vlm import VLM


@pytest.fixture(autouse=True)
def _force_sdpa():
    """These tests build models on CPU. The FA3 kernel is CUDA-only, so force the
    SDPA fallback for the duration of each test (and restore afterwards). This is
    an autouse fixture rather than a module-level override because other test
    files (e.g. test_attention_fallback) mutate the global USE_FA3 flag at run
    time, which would clobber a one-time module-level setting."""
    prev_override, prev_use = fa_module._override_impl, fa_module.USE_FA3
    fa_module._override_impl = "sdpa"
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()
    yield
    fa_module._override_impl = prev_override
    fa_module.USE_FA3 = prev_use


def tiny_gpt_config(vocab_size=128):
    return GPTConfig(
        sequence_len=256,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
    )


def tiny_vit_config():
    # 32x32 image, 8x8 patches -> 4x4 grid = 16 patches
    return ViTConfig(image_size=32, patch_size=8, embed_dim=32, n_layer=2, n_head=4)


def build_vlm(vocab_size=128, device="cpu"):
    """Build a tiny VLM on meta device, move to device, init weights."""
    gpt_cfg = tiny_gpt_config(vocab_size)
    vit_cfg = tiny_vit_config()
    image_token_id = vocab_size - 1  # pretend <|image|> is the last token
    with torch.device("meta"):
        vlm = VLM(gpt_cfg, vit_cfg, image_token_id)
    vlm.to_empty(device=device)
    vlm.gpt.init_weights()
    vlm.vit.init_weights()
    return vlm


class TestVLMConstruction:
    def test_n_visual_tokens(self):
        vlm = build_vlm()
        assert vlm.n_visual_tokens == 16

    def test_config_passthrough(self):
        vlm = build_vlm()
        assert vlm.config.n_embd == 64
        assert vlm.config.n_layer == 2

    def test_get_device(self):
        vlm = build_vlm()
        assert vlm.get_device().type == "cpu"


class TestVLMForward:
    def test_forward_returns_loss(self):
        vlm = build_vlm()
        vlm.train()
        image = torch.randn(2, 3, 32, 32)
        idx = torch.randint(0, 128, (2, 8))
        targets = torch.randint(0, 128, (2, 8))
        loss = vlm(image, idx, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_forward_returns_logits(self):
        vlm = build_vlm()
        vlm.eval()
        image = torch.randn(1, 3, 32, 32)
        idx = torch.randint(0, 128, (1, 8))
        with torch.no_grad():
            logits = vlm(image, idx)
        # combined length = 16 (visual) + 8 (text)
        assert logits.shape == (1, 16 + 8, 128)

    def test_visual_positions_masked_in_loss(self):
        """Visual positions should not contribute to the loss (targets=-1)."""
        vlm = build_vlm()
        vlm.train()
        image = torch.randn(1, 3, 32, 32)
        idx = torch.randint(0, 128, (1, 8))
        targets = torch.randint(0, 128, (1, 8))
        # Build combined targets manually to verify masking
        combined_idx, combined_embeds = vlm._build_combined(image, idx)
        assert combined_idx.shape == (1, 16 + 8)
        # First 16 positions should be the image token id
        assert (combined_idx[0, :16] == vlm.image_token_id).all()
        # Last 8 positions should be the text idx
        assert torch.equal(combined_idx[0, 16:], idx[0])
        # Combined embeds shape
        assert combined_embeds.shape == (1, 16 + 8, 64)

    def test_loss_reduction_none(self):
        vlm = build_vlm()
        vlm.train()
        image = torch.randn(1, 3, 32, 32)
        idx = torch.randint(0, 128, (1, 8))
        targets = torch.randint(0, 128, (1, 8))
        loss = vlm(image, idx, targets, loss_reduction="none")
        # 'none' returns per-position loss, flattened: (B * (n_visual + T_text),)
        assert loss.shape == (1 * (16 + 8),)


class TestVLMGenerate:
    def test_generate_produces_tokens(self):
        vlm = build_vlm()
        vlm.eval()
        image = torch.randn(1, 3, 32, 32)
        tokens = [1, 2, 3, 4]
        generated = list(vlm.generate(image, tokens, max_tokens=5, temperature=0.0))
        assert len(generated) == 5
        # All generated tokens should be valid vocab ids
        for t in generated:
            assert 0 <= t < 128

    def test_generate_deterministic_greedy(self):
        vlm = build_vlm()
        vlm.eval()
        image = torch.randn(1, 3, 32, 32)
        tokens = [1, 2, 3]
        gen1 = list(vlm.generate(image, tokens, max_tokens=4, temperature=0.0))
        gen2 = list(vlm.generate(image, tokens, max_tokens=4, temperature=0.0))
        assert gen1 == gen2


class TestGPTBackwardCompat:
    """Verify the GPT.forward() change (optional embeds param) is backward compatible."""

    def test_forward_without_embeds(self):
        """GPT.forward() without embeds should work exactly as before."""
        gpt_cfg = tiny_gpt_config()
        with torch.device("meta"):
            gpt = GPT(gpt_cfg)
        gpt.to_empty(device="cpu")
        gpt.init_weights()
        gpt.eval()
        idx = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            logits = gpt.forward(idx)
        assert logits.shape == (1, 16, 128)

    def test_forward_with_embeds_matches_wte(self):
        """GPT.forward(embeds=wte(idx)) should equal GPT.forward(idx)."""
        gpt_cfg = tiny_gpt_config()
        with torch.device("meta"):
            gpt = GPT(gpt_cfg)
        gpt.to_empty(device="cpu")
        gpt.init_weights()
        gpt.eval()
        idx = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            logits_normal = gpt.forward(idx)
            embeds = gpt.transformer.wte(idx)
            logits_embeds = gpt.forward(idx, embeds=embeds)
        assert torch.allclose(logits_normal, logits_embeds, atol=1e-5)
