"""
Stage 2: Supervised fine-tuning of the full VLM (ViT + projector + LLM).

Loads a VLM checkpoint (from stage 1, or a base model with a fresh ViT) and
fine-tunes the entire model on image-text conversation data. The loss is computed
only on the assistant's response tokens (the image and user prompt are masked).

Run as:
    python -m scripts.vlm_sft

Or a tiny CPU smoke test:
    python -m scripts.vlm_sft --num-iterations=10 --device-batch-size=2 \
        --total-batch-size=8 --eval-every=-1 --device-type=cpu

Requires:
    python -m scripts.coco_data          # prepare COCO data (one-time)
    a VLM checkpoint in ~/.cache/nanochat/vlm_checkpoints/ (from stage 1)
    OR a base model in ~/.cache/nanochat/base_checkpoints/ (fresh ViT)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import argparse
import wandb
import torch
import torch.distributed as dist

from nanochat.common import (compute_init, compute_cleanup, print0, DummyWandb,
                             get_base_dir, autodetect_device_type, get_peak_flops,
                             COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized)
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_vlm
from nanochat.gpt import GPTConfig
from nanochat.vit import ViTConfig, DinoViTConfig
from nanochat.vlm import VLM
from scripts.coco_data import COCODataset, collate_coco_batch, IMAGE_SIZE

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Stage 2: VLM supervised fine-tuning")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# VLM source: "vlm" (from stage 1) or "base" (fresh ViT on a base model)
parser.add_argument("--source", type=str, default="vlm", choices=["vlm", "base"],
                    help="where to load the starting VLM from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="model step to load")
# Encoder choice (only used when --source=base; with --source=vlm it comes from the checkpoint)
parser.add_argument("--encoder", type=str, default="vit", choices=["vit", "dinov2"],
                    help="vision encoder (for --source=base)")
parser.add_argument("--dinov2-weights", type=str, default="",
                    help="path to dinov2_vitb14_pretrain.pth (for --source=base --encoder dinov2)")
# ViT config (only used when --source=base, to build a fresh ViT)
parser.add_argument("--image-size", type=int, default=128, help="ViT input image size")
parser.add_argument("--patch-size", type=int, default=16, help="ViT patch size")
parser.add_argument("--vit-dim", type=int, default=256, help="ViT embedding dim")
parser.add_argument("--vit-layers", type=int, default=4, help="ViT num layers")
parser.add_argument("--vit-heads", type=int, default=4, help="ViT num heads")
# Image normalization: "unit" (from-scratch ViT) or "imagenet" (DINOv2)
parser.add_argument("--normalize", type=str, default="unit", choices=["unit", "imagenet"],
                    help="image normalization mode")
# Scale the ViT LR by this factor (use <1 to gently fine-tune a pretrained DINOv2 backbone)
parser.add_argument("--vit-lr-scale", type=float, default=1.0,
                    help="multiplier applied to all ViT param-group LRs")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=2000, help="number of optimization steps")
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=256, help="total batch size (images)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.05, help="LR for embedding params (AdamW)")
parser.add_argument("--unembedding-lr", type=float, default=0.001, help="LR for lm_head (AdamW)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="LR for matrix params (Muon)")
parser.add_argument("--vit-lr", type=float, default=0.0003, help="LR for ViT params")
parser.add_argument("--init-lr-frac", type=float, default=0.5, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.05, help="fraction of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="fraction of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--max-coco-images", type=int, default=None, help="cap on COCO images (for tiny runs)")
# Output
parser.add_argument("--output-tag", type=str, default=None, help="output model tag (default: vlmsft_d{depth})")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# The DINOv2 backbone has many distinct param shapes; the fused AdamW kernel
# recompiles per shape, so raise the dynamo recompile limit to avoid hitting it.
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-vlm-sft", name=args.run, config=user_config)

# -----------------------------------------------------------------------------
# Load the starting VLM
from dataclasses import asdict
if args.source == "vlm":
    # Load a full VLM checkpoint from stage 1
    vlm, tokenizer, meta = load_vlm("vlm", device, phase="train",
                                    model_tag=args.model_tag, step=args.model_step)
    gpt_config = vlm.config
    vit_config = vlm.vit.config
    image_token_id = vlm.image_token_id
    encoder = vlm.encoder
    gpt_config_kwargs = asdict(gpt_config)
    vit_config_kwargs = asdict(vit_config)
    # DINOv2 checkpoints use ImageNet normalization
    if encoder == "dinov2" and args.normalize == "unit":
        args.normalize = "imagenet"
    print0(f"Loaded VLM from stage 1: depth={gpt_config.n_layer}, n_embd={gpt_config.n_embd}, encoder={encoder}")
else:
    # Build a fresh VLM on top of a base model
    base_model, tokenizer, base_meta = load_model("base", device, phase="train",
                                                  model_tag=args.model_tag, step=args.model_step)
    gpt_config = base_model.config
    vocab_size = tokenizer.get_vocab_size()
    image_token_id = tokenizer.encode_special("<|image|>")
    encoder = args.encoder
    if encoder == "dinov2":
        assert args.dinov2_weights, "--dinov2-weights required when --encoder dinov2"
        vit_config = DinoViTConfig(weights_path=args.dinov2_weights)
        if args.normalize == "unit":
            args.normalize = "imagenet"
    else:
        vit_config = ViTConfig(
            image_size=args.image_size, patch_size=args.patch_size,
            embed_dim=args.vit_dim, n_layer=args.vit_layers, n_head=args.vit_heads,
        )
    gpt_config_kwargs = asdict(gpt_config)
    vit_config_kwargs = asdict(vit_config)
    with torch.device("meta"):
        vlm = VLM(GPTConfig(**gpt_config_kwargs), vit_config, image_token_id, encoder=encoder)
    vlm.to_empty(device=device)
    vlm.gpt.init_weights()
    vlm.vit.init_weights()
    vlm.gpt.load_state_dict(base_model.state_dict(), strict=True, assign=True)
    if encoder == "dinov2":
        vlm.vit.load_dinov2_weights(args.dinov2_weights)
    del base_model
    print0(f"Built fresh VLM on base model: depth={gpt_config.n_layer}, n_embd={gpt_config.n_embd}")

# Unfreeze everything for full fine-tuning
vlm.set_llm_trainable(True)
n_trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in vlm.parameters())
print0(f"Trainable params: {n_trainable:,} / {n_total:,} total")

# -----------------------------------------------------------------------------
# Optimizer (MuonAdamW: Muon for matrix params, AdamW for embeddings/scalars)
optimizer = vlm.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    vit_lr=args.vit_lr,
    weight_decay=0.0,
)
# Override initial LR as a fraction of base; scale ViT groups by vit_lr_scale
# (use <1 to gently fine-tune a pretrained DINOv2 backbone)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    if group.get("vit", False):
        group["lr"] = group["lr"] * args.vit_lr_scale
    group["initial_lr"] = group["lr"]

# GradScaler for fp16
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

# -----------------------------------------------------------------------------
# Data: image-text conversations
# The conversation is: user asks to describe the image, assistant gives the caption.
# We render it with the tokenizer so the loss mask only covers the assistant part.
train_dataset = COCODataset(split="train", max_rows=args.max_coco_images * 5 if args.max_coco_images else None)
val_dataset = COCODataset(split="val", max_rows=args.max_coco_images * 5 if args.max_coco_images else None)
print0(f"Train pairs: {len(train_dataset):,} | Val pairs: {len(val_dataset):,}")

user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
bos = tokenizer.get_bos_token_id()

def render_image_conversation(caption):
    """
    Render an image-description conversation into (ids, mask).
    The image is represented by a single <|image|> placeholder token in the user turn.
    mask=1 only on the assistant's caption tokens (what we supervise).
    """
    ids, mask = [], []
    def add(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))
    add(bos, 0)
    add(user_start, 0)
    add([image_token_id], 0)  # the image placeholder
    add(tokenizer.encode("Describe this image."), 0)
    add(user_end, 0)
    add(assistant_start, 0)
    add(tokenizer.encode(caption), 1)  # supervise the caption
    add(assistant_end, 1)
    return ids, mask

def vlm_data_generator(dataset, buffer_size=100):
    """Infinite generator of (images, ids, targets) batches for VLM SFT."""
    n = len(dataset)
    cursor = ddp_rank
    while True:
        rows = []
        for _ in range(args.device_batch_size):
            rows.append(dataset[cursor % n])
            cursor += ddp_world_size
            if cursor >= n:
                cursor = cursor % n
        images, captions = collate_coco_batch(rows, vit_config.image_size, args.normalize)
        # Render each conversation
        rendered = [render_image_conversation(c) for c in captions]
        max_len = max(len(ids) for ids, _ in rendered)
        ids = torch.full((args.device_batch_size, max_len), bos, dtype=torch.long)
        mask = torch.zeros((args.device_batch_size, max_len), dtype=torch.int8)
        for i, (ids_row, mask_row) in enumerate(rendered):
            ids[i, :len(ids_row)] = torch.tensor(ids_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)
        # targets = ids shifted by 1, masked where mask (shifted) is 0
        targets = torch.full((args.device_batch_size, max_len), -1, dtype=torch.long)
        for i in range(args.device_batch_size):
            # position t predicts ids[t+1]; supervise where mask[t+1] == 1
            shifted_mask = mask[i, 1:]
            shifted_ids = ids[i, 1:]
            targets[i, :max_len - 1] = torch.where(shifted_mask == 1, shifted_ids, torch.tensor(-1))
        yield (
            images.to(device=device, non_blocking=True),
            ids.to(device=device, dtype=torch.int32, non_blocking=True).contiguous(),
            targets.to(device=device, dtype=torch.int64, non_blocking=True).contiguous(),
        )

train_loader = vlm_data_generator(train_dataset)
build_val_loader = lambda: vlm_data_generator(val_dataset)

# -----------------------------------------------------------------------------
# LR schedule (linear warmup, constant, linear warmdown) based on progress
def get_lr_multiplier(progress):
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# Training loop
x, y_ids, y_targets = next(train_loader)  # prefetch first batch
min_val_loss = float("inf")
best_val_loss = float("inf")
best_step = 0
best_state = None  # CPU copy of the best-validation weights
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
progress = 0.0
while True:
    last_step = step == args.num_iterations
    flops_so_far = step * args.total_batch_size  # rough proxy

    # Evaluate val loss
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        vlm.eval()
        val_loader = build_val_loader()
        val_losses = []
        for _ in range(10):
            vx, vy_ids, vy_targets = next(val_loader)
            with torch.no_grad():
                vloss = vlm(vx, vy_ids, vy_targets)
            val_losses.append(vloss.item())
        val_loss = sum(val_losses) / len(val_losses)
        print0(f"Step {step:05d} | Val loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        # Track the best-validation weights so we can save them at the end
        # (SFT often overfits after the val-loss minimum, so the final step
        # is not necessarily the best checkpoint).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_step = step
            best_state = {k: v.detach().cpu() for k, v in vlm.state_dict().items()}
            print0(f"  -> new best val loss {best_val_loss:.4f} at step {best_step}")
        wandb_run.log({"step": step, "val/loss": val_loss})
        vlm.train()

    # Save the best-validation checkpoint at the end
    if last_step:
        output_dirname = args.output_tag if args.output_tag else f"vlmsft_d{gpt_config.n_layer}"
        checkpoint_dir = os.path.join(get_base_dir(), "vlm_sft_checkpoints", output_dirname)
        save_state = best_state if best_state is not None else vlm.state_dict()
        save_checkpoint(
            checkpoint_dir,
            best_step if best_state is not None else step,
            save_state,
            optimizer.state_dict(),
            {
                "step": best_step if best_state is not None else step,
                "val_loss": best_val_loss if best_state is not None else val_loss,
                "final_val_loss": val_loss,
                "gpt_config": gpt_config_kwargs,
                "vit_config": vit_config_kwargs,
                "encoder": encoder,
                "image_token_id": image_token_id,
                "user_config": user_config,
            },
            rank=ddp_rank,
        )
        print0(f"Saved best checkpoint (step {best_step}, val loss {best_val_loss:.4f})")
        break

    # -------------------------------------------------------------------------
    # single training step
    synchronize()
    t0 = time.time()
    world_tokens_per_fwdbwd = args.device_batch_size * ddp_world_size
    assert args.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
    for micro_step in range(grad_accum_steps):
        loss = vlm(x, y_ids, y_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y_ids, y_targets = next(train_loader)
        progress = max(progress, (step + 1) / args.num_iterations)
    # step the optimizer
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    vlm.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    step += 1
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / args.num_iterations
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.4f} | dt: {dt * 1000:.2f}ms | total time: {total_training_time / 60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({"step": step, "train/loss": debiased_smooth_loss, "train/lrm": lrm, "train/dt": dt})

    if step == 1:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
print0(f"Minimum validation loss: {min_val_loss:.4f}")

# cleanup
wandb_run.finish()
compute_cleanup()
