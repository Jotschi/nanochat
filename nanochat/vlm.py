"""
Vision-Language Model (VLM): wraps a GPT language model with a ViT vision encoder.

The ViT converts an image into a fixed-length sequence of visual token embeddings
(n_patches of them, each of dimension n_embd). These are prepended as a contiguous
prefix to the text token embeddings, and the combined sequence is fed through the
GPT transformer.

The language model architecture is unchanged (RMSNorm, RoPE, SwiGLU/relu², causal
self-attention, GQA, smear, value embeddings, etc.). The only modification to the
GPT is the optional `embeds` parameter in forward(), which allows us to inject
pre-computed embeddings (visual + text) instead of looking up text embeddings.

Visual positions use a special `<|image|>` token ID in the `idx` tensor, which is
used for the value-embeddings lookup. The actual visual content comes from the
`embeds` tensor (ViT output), not from the token embedding table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.vit import ViT, ViTConfig
from nanochat.common import COMPUTE_DTYPE


class VLM(nn.Module):
    """
    Vision-Language Model: ViT encoder + GPT language model.

    forward(image, idx, targets):
        image: (B, 3, H, W) float tensor
        idx: (B, T_text) long tensor - text token IDs
        targets: (B, T_text) long tensor - target token IDs (for loss)
        returns: scalar loss (if targets) or logits (B, 64+T_text, vocab)

    generate(image, tokens, max_tokens, ...):
        image: (1, 3, H, W) float tensor (batch size 1)
        tokens: list[int] - text token IDs
        yields: int tokens one at a time
    """

    def __init__(self, gpt_config, vit_config, image_token_id):
        super().__init__()
        self.gpt = GPT(gpt_config)
        self.vit = ViT(vit_config, gpt_config.n_embd)
        self.image_token_id = image_token_id
        self.n_visual_tokens = vit_config.n_patches

    @property
    def config(self):
        return self.gpt.config

    def get_device(self):
        return self.gpt.get_device()

    def set_llm_trainable(self, trainable):
        """Freeze (or unfreeze) all GPT parameters. Used in stage 1 (ViT pretrain)."""
        for param in self.gpt.parameters():
            param.requires_grad = trainable
        # Also toggle the module mode so frozen GPT doesn't update any state
        self.gpt.eval() if not trainable else self.gpt.train()

    def _encode_image(self, image):
        """Encode an image into visual token embeddings. (B, 3, H, W) -> (B, n_visual, n_embd)"""
        return self.vit(image)

    def _build_combined(self, image, idx):
        """
        Build the combined (visual + text) embeddings and index tensor.

        Returns:
            combined_idx: (B, n_visual + T_text) long
            combined_embeds: (B, n_visual + T_text, n_embd) float
        """
        B, T_text = idx.size()
        device = idx.device
        # Encode image
        visual_embeds = self._encode_image(image)  # (B, n_visual, n_embd)
        # Embed text tokens
        text_embeds = self.gpt.transformer.wte(idx)  # (B, T_text, n_embd)
        # Build combined index (image_token_id for visual positions)
        visual_ids = torch.full((B, self.n_visual_tokens), self.image_token_id,
                                dtype=torch.long, device=device)
        combined_idx = torch.cat([visual_ids, idx], dim=1)
        # Build combined embeddings
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        return combined_idx, combined_embeds

    def forward(self, image, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """
        Forward pass for training or evaluation.

        Args:
            image: (B, 3, H, W) float tensor
            idx: (B, T_text) long tensor - text token IDs
            targets: (B, T_text) long tensor - target token IDs (shifted by 1)
            kv_cache: KV cache for inference (not used in training)
            loss_reduction: 'mean' or 'none'

        Returns:
            loss (scalar) if targets provided, else logits (B, n_visual+T_text, vocab)
        """
        combined_idx, combined_embeds = self._build_combined(image, idx)
        # Mask visual positions in targets (they are not supervised)
        if targets is not None:
            B = idx.size(0)
            visual_targets = torch.full((B, self.n_visual_tokens), -1,
                                        dtype=targets.dtype, device=targets.device)
            combined_targets = torch.cat([visual_targets, targets], dim=1)
        else:
            combined_targets = None
        return self.gpt.forward(combined_idx, combined_targets, kv_cache,
                                loss_reduction, embeds=combined_embeds)

    @torch.inference_mode()
    def generate(self, image, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive generation conditioned on an image and text prompt.
        Mirrors GPT.generate() but with a visual prefix.

        Args:
            image: (1, 3, H, W) float tensor (batch size 1)
            tokens: list[int] - text token IDs (the prompt)
            max_tokens: int - max number of tokens to generate
            temperature: float - sampling temperature (0 = greedy)
            top_k: int or None - top-k sampling
            seed: int - random seed

        Yields:
            int - one token ID at a time
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Encode image once
        image = image.to(device)
        visual_embeds = self._encode_image(image)  # (1, n_visual, n_embd)

        # Build initial combined sequence
        text_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        visual_ids = torch.full((1, self.n_visual_tokens), self.image_token_id,
                                dtype=torch.long, device=device)
        combined_idx = torch.cat([visual_ids, text_ids], dim=1)
        text_embeds = self.gpt.transformer.wte(text_ids)
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        for _ in range(max_tokens):
            logits = self.gpt.forward(combined_idx, embeds=combined_embeds)
            logits = logits[:, -1, :]  # (1, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            # Append the new token to the combined sequence
            new_embed = self.gpt.transformer.wte(next_ids)
            combined_idx = torch.cat([combined_idx, next_ids], dim=1)
            combined_embeds = torch.cat([combined_embeds, new_embed], dim=1)
            token = next_ids.item()
            yield token

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, scalar_lr=0.5, vit_lr=0.02):
        """
        Set up the optimizer for the full VLM (GPT + ViT).
        Follows the same pattern as GPT.setup_optimizer but adds ViT params.
        """
        model_dim = self.config.n_embd

        # GPT params (same groups as GPT.setup_optimizer)
        gpt_matrix_params = list(self.gpt.transformer.h.parameters())
        gpt_value_embeds_params = list(self.gpt.value_embeds.parameters())
        gpt_embedding_params = list(self.gpt.transformer.wte.parameters())
        gpt_lm_head_params = list(self.gpt.lm_head.parameters())
        gpt_resid_params = [self.gpt.resid_lambdas]
        gpt_x0_params = [self.gpt.x0_lambdas]
        gpt_smear_params = [self.gpt.smear_gate.weight, self.gpt.smear_lambda, self.gpt.backout_lambda]

        # ViT params
        # Muon only supports 2D matrix params. The Conv2d patch_embed weight is 4D
        # (out, in, kH, kW) and must go to AdamW, not Muon.
        vit_matrix_params = []
        vit_embedding_params = []
        for name, param in self.vit.named_parameters():
            if 'pos_embed' in name:
                vit_embedding_params.append(param)
            elif param.ndim == 2:
                vit_matrix_params.append(param)
            else:
                vit_embedding_params.append(param)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        param_groups = [
            # GPT AdamW groups
            dict(kind='adamw', params=gpt_lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=gpt_embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=gpt_value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=gpt_resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=gpt_x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=gpt_smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
            # ViT AdamW group (pos_embed and small params)
            dict(kind='adamw', params=vit_embedding_params, lr=vit_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        ]
        # GPT Muon groups (matrix params, grouped by shape)
        for shape in sorted({p.shape for p in gpt_matrix_params}):
            group_params = [p for p in gpt_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
        # ViT Muon groups (matrix params, grouped by shape)
        for shape in sorted({p.shape for p in vit_matrix_params}):
            group_params = [p for p in vit_matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=vit_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        from nanochat.optim import MuonAdamW
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
