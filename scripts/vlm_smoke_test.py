"""
Self-contained VLM pipeline smoke test.

Exercises the complete vision->LLaMA path end-to-end WITHOUT requiring COCO data
or a pretrained base model, using a tiny from-scratch model and synthetic data:

    1. Train a tiny BPE tokenizer (with the <|image|> special token)
    2. Build a tiny VLM (ViT + GPT) from scratch
    3. Train it for a few steps on synthetic image->caption data
    4. Save a VLM checkpoint (model + meta)
    5. Load it back via checkpoint_manager.build_vlm
    6. Run autoregressive generation on an image + prompt

Run as:
    python -m scripts.vlm_smoke_test

This is the "definition of done" check from spec/GOAL.md: a contributor can
train or load a small multimodal checkpoint, provide an image plus a textual
prompt, and obtain a generated textual response.
"""

import os
import tempfile
import torch

from nanochat.common import get_base_dir, print0
from nanochat.tokenizer import RustBPETokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.vit import ViT, ViTConfig
from nanochat.vlm import VLM
from nanochat.checkpoint_manager import save_checkpoint, build_vlm
from dataclasses import asdict

# Force SDPA (this smoke test runs on CPU; FA3 is CUDA-only)
import nanochat.flash_attention as fa_module
fa_module._override_impl = "sdpa"
fa_module.USE_FA3 = fa_module._resolve_use_fa3()


def make_tiny_tokenizer(vocab_size=512):
    """Train a tiny BPE tokenizer on synthetic text and save it to the base dir."""
    # Synthetic corpus: diverse random words so BPE has enough merges to reach
    # the target vocab size. (A tiny repetitive corpus only yields ~58 merges.)
    import random
    rng = random.Random(42)
    letters = "abcdefghijklmnopqrstuvwxyz"
    def rand_word():
        return "".join(rng.choice(letters) for _ in range(rng.randint(2, 8)))
    corpus = []
    for i in range(20000):
        sentence = " ".join(rand_word() for _ in range(rng.randint(5, 20)))
        corpus.append(sentence)
    tokenizer = RustBPETokenizer.train_from_iterator(iter(corpus), vocab_size)
    # Save to the standard base dir location
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer.save(tokenizer_dir)
    # Also write token_bytes.pt (needed by get_token_bytes, though not used here)
    import torch as _t
    special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(tokenizer.get_vocab_size()):
        if token_id in special_ids:
            token_bytes.append(0)
        else:
            token_bytes.append(len(tokenizer.decode_single_token_bytes(token_id)))
    token_bytes = _t.tensor(token_bytes, dtype=_t.int32)
    with open(os.path.join(tokenizer_dir, "token_bytes.pt"), "wb") as f:
        _t.save(token_bytes, f)
    return tokenizer


def build_tiny_vlm(tokenizer, device="cpu"):
    """Build a tiny VLM from scratch on the meta device, then init."""
    vocab_size = tokenizer.get_vocab_size()
    image_token_id = tokenizer.encode_special("<|image|>")
    gpt_cfg = GPTConfig(
        sequence_len=128, vocab_size=vocab_size,
        n_layer=2, n_head=4, n_kv_head=4, n_embd=64, window_pattern="L",
    )
    vit_cfg = ViTConfig(image_size=32, patch_size=8, embed_dim=32, n_layer=2, n_head=4)
    with torch.device("meta"):
        vlm = VLM(gpt_cfg, vit_cfg, image_token_id)
    vlm.to_empty(device=device)
    vlm.gpt.init_weights()
    vlm.vit.init_weights()
    return vlm, gpt_cfg, vit_cfg, image_token_id


def synthetic_batch(B, T_text, image_size, vocab_size, tokenizer, device, seed=0):
    """Create a synthetic (image, ids, targets) batch for image->caption training.

    Uses a FIXED image->caption mapping (deterministic per image index) so the
    task is learnable: the same image always maps to the same caption. This lets
    the smoke test verify the model actually learns the image->text mapping.

    ids and targets are both (B, T_text). targets[i] = ids[i] shifted left by 1
    (position t predicts token t+1), with the last position masked (-1).
    """
    bos = tokenizer.get_bos_token_id()
    ids_list = []
    images_list = []
    for i in range(B):
        # Deterministic per-image rng: image i always gets the same caption + pixels
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + i)
        caption = [int(torch.randint(0, vocab_size, (1,), generator=g).item()) for _ in range(T_text - 1)]
        ids_row = [bos] + caption  # exactly T_text tokens
        ids_list.append(ids_row)
        # Deterministic image pixels (same image every time for index i)
        img = torch.randn(3, image_size, image_size, generator=g)
        images_list.append(img)
    images = torch.stack(images_list).to(device=device)
    ids = torch.tensor(ids_list, dtype=torch.int32, device=device)  # (B, T_text)
    targets = torch.full((B, T_text), -1, dtype=torch.int64, device=device)
    for i in range(B):
        targets[i, :T_text - 1] = torch.tensor(ids_list[i][1:T_text], dtype=torch.int64, device=device)
    return images, ids, targets


def main():
    device = torch.device("cpu")
    print0("=" * 70)
    print0("VLM PIPELINE SMOKE TEST")
    print0("=" * 70)

    # 1) Tiny tokenizer
    print0("\n[1/6] Training tiny tokenizer...")
    tokenizer = make_tiny_tokenizer(vocab_size=512)
    vocab_size = tokenizer.get_vocab_size()
    image_token_id = tokenizer.encode_special("<|image|>")
    print0(f"  vocab_size={vocab_size}, image_token_id={image_token_id}")
    assert image_token_id is not None and image_token_id < vocab_size

    # 2) Tiny VLM
    print0("\n[2/6] Building tiny VLM...")
    vlm, gpt_cfg, vit_cfg, image_token_id = build_tiny_vlm(tokenizer, device)
    n_params = sum(p.numel() for p in vlm.parameters())
    print0(f"  VLM params: {n_params:,} (gpt n_embd={gpt_cfg.n_embd}, vit patches={vit_cfg.n_patches})")

    # 3) Train on a learnable synthetic mapping (fixed image -> caption)
    print0("\n[3/6] Training on a learnable synthetic image->caption mapping...")
    vlm.train()
    optimizer = torch.optim.AdamW(vlm.parameters(), lr=3e-3)
    B, T_text, image_size = 4, 8, 32
    losses = []
    for step in range(40):
        images, ids, targets = synthetic_batch(B, T_text, image_size, vocab_size, tokenizer, device)
        loss = vlm(images, ids, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 10 == 0 or step == 39:
            print0(f"  step {step:2d}: loss={loss.item():.4f}")
    # The mapping is learnable, so loss should drop meaningfully from init.
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "Loss must be finite"
    assert losses[-1] < losses[0] - 0.5, f"Loss should decrease on learnable data: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print0(f"  loss went {losses[0]:.4f} -> {losses[-1]:.4f} (learned the mapping)")

    # 4) Save checkpoint
    final_step = len(losses)
    print0(f"\n[4/6] Saving VLM checkpoint (step {final_step})...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "vlm_checkpoints", "smoke")
        save_checkpoint(
            checkpoint_dir,
            step=final_step,
            model_data=vlm.state_dict(),
            optimizer_data=optimizer.state_dict(),
            meta_data={
                "step": final_step,
                "val_loss": losses[-1],
                "gpt_config": asdict(gpt_cfg),
                "vit_config": asdict(vit_cfg),
                "image_token_id": image_token_id,
                "user_config": {"smoke_test": True},
            },
            rank=0,
        )
        model_path = os.path.join(checkpoint_dir, f"model_{final_step:06d}.pt")
        assert os.path.exists(model_path), f"Model checkpoint not saved: {model_path}"
        print0(f"  Saved to {model_path}")

        # 5) Load it back
        print0("\n[5/6] Loading VLM checkpoint back...")
        loaded_vlm, loaded_tokenizer, loaded_meta = build_vlm(checkpoint_dir, final_step, device, phase="eval")
        assert loaded_vlm.config.n_embd == gpt_cfg.n_embd
        assert loaded_vlm.n_visual_tokens == vit_cfg.n_patches
        assert loaded_vlm.image_token_id == image_token_id
        # Verify weights match (cast to float: build_vlm upcasts bf16->float on CPU)
        for (n1, p1), (n2, p2) in zip(vlm.named_parameters(), loaded_vlm.named_parameters()):
            assert n1 == n2, f"Param name mismatch: {n1} vs {n2}"
            assert torch.allclose(p1.float(), p2.float(), atol=1e-3), f"Param value mismatch for {n1}"
        print0("  Checkpoint round-trip OK (weights match)")

        # 6) Generate
        print0("\n[6/6] Running generation on a synthetic image + prompt...")
        loaded_vlm.eval()
        image = torch.randn(1, 3, image_size, image_size, device=device)
        bos = tokenizer.get_bos_token_id()
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        prompt_tokens = [bos, user_start, image_token_id]
        prompt_tokens.extend(tokenizer.encode("Describe this image."))
        prompt_tokens.append(user_end)
        prompt_tokens.append(assistant_start)
        generated = list(loaded_vlm.generate(image, prompt_tokens, max_tokens=10, temperature=0.0))
        assert len(generated) == 10, f"Expected 10 tokens, got {len(generated)}"
        for t in generated:
            assert 0 <= t < vocab_size, f"Generated token {t} out of vocab range"
        decoded = tokenizer.decode(generated)
        print0(f"  Generated {len(generated)} tokens: {decoded!r}")

    print0("\n" + "=" * 70)
    print0("VLM PIPELINE SMOKE TEST PASSED")
    print0("=" * 70)


if __name__ == "__main__":
    main()
