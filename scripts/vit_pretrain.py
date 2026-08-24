"""
Stage 1: Pretrain the ViT vision encoder with the LLM frozen.

The LLM (base model) is loaded from a pretrained checkpoint and frozen. Only the
ViT encoder + projector are trained, on COCO image->caption pairs. This teaches the
vision encoder to produce visual token embeddings that the (frozen) LLM can use to
predict caption text.

Run as:
    python -m scripts.vit_pretrain

Or a tiny CPU smoke test:
    python -m scripts.vit_pretrain --num-iterations=10 --device-batch-size=2 \
        --total-batch-size=8 --eval-every=-1 --device-type=cpu

Requires:
    python -m scripts.coco_data          # prepare COCO data (one-time)
    a pretrained base model in ~/.cache/nanochat/base_checkpoints/
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
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.vit import ViTConfig, DinoViTConfig
from nanochat.vlm import VLM
from scripts.coco_data import COCODataset, collate_coco_batch, IMAGE_SIZE

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Stage 1: ViT pretraining (frozen LLM)")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Base model to load (the frozen LLM)
parser.add_argument("--model-tag", type=str, default=None, help="base model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="base model step to load")
# Encoder choice: "vit" (from scratch) or "dinov2" (pretrained DINOv2 ViT-B/14)
parser.add_argument("--encoder", type=str, default="vit", choices=["vit", "dinov2"],
                    help="vision encoder: from-scratch ViT or pretrained DINOv2")
parser.add_argument("--dinov2-weights", type=str, default="",
                    help="path to dinov2_vitb14_pretrain.pth (required if --encoder dinov2)")
# ViT config (used for the from-scratch ViT; DINOv2 uses its own fixed config)
parser.add_argument("--image-size", type=int, default=128, help="ViT input image size")
parser.add_argument("--patch-size", type=int, default=16, help="ViT patch size")
parser.add_argument("--vit-dim", type=int, default=256, help="ViT embedding dim")
parser.add_argument("--vit-layers", type=int, default=4, help="ViT num layers")
parser.add_argument("--vit-heads", type=int, default=4, help="ViT num heads")
# Image normalization: "unit" (from-scratch ViT) or "imagenet" (DINOv2)
parser.add_argument("--normalize", type=str, default="unit", choices=["unit", "imagenet"],
                    help="image normalization mode")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=2000, help="number of optimization steps")
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=16, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=256, help="total batch size (images)")
# Optimization
parser.add_argument("--vit-lr", type=float, default=0.0003, help="learning rate for ViT params (AdamW)")
parser.add_argument("--warmup-steps", type=int, default=50, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="fraction of steps for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--max-coco-images", type=int, default=None, help="cap on COCO images (for tiny runs)")
# Output
parser.add_argument("--output-tag", type=str, default=None, help="output model tag (default: vlm_d{depth})")
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-vit-pretrain", name=args.run, config=user_config)

# -----------------------------------------------------------------------------
# Load the frozen base model (the LLM)
base_model, tokenizer, base_meta = load_model("base", device, phase="eval",
                                              model_tag=args.model_tag, step=args.model_step)
gpt_config = base_model.config
vocab_size = tokenizer.get_vocab_size()
image_token_id = tokenizer.encode_special("<|image|>")
print0(f"Loaded base model: depth={gpt_config.n_layer}, n_embd={gpt_config.n_embd}, vocab={vocab_size}")
print0(f"image_token_id={image_token_id}")

# -----------------------------------------------------------------------------
# Build the VLM: wrap the loaded GPT + a vision encoder (from-scratch ViT or DINOv2)
if args.encoder == "dinov2":
    assert args.dinov2_weights, "--dinov2-weights required when --encoder dinov2"
    vit_config = DinoViTConfig(weights_path=args.dinov2_weights)
    # DINOv2 expects ImageNet normalization and 224px input
    if args.normalize == "unit":
        args.normalize = "imagenet"
    print0(f"DINOv2 config: {vit_config} (n_patches={vit_config.n_patches})")
else:
    vit_config = ViTConfig(
        image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.vit_dim, n_layer=args.vit_layers, n_head=args.vit_heads,
    )
    print0(f"ViT config: {vit_config} (n_patches={vit_config.n_patches})")

# We reuse the loaded GPT directly (no copy) by constructing a VLM that shares it.
# Build VLM on meta, then swap in the loaded GPT's state.
from nanochat.gpt import GPTConfig
from dataclasses import asdict
gpt_config_kwargs = asdict(gpt_config)
with torch.device("meta"):
    vlm = VLM(GPTConfig(**gpt_config_kwargs), vit_config, image_token_id, encoder=args.encoder)
vlm.to_empty(device=device)
vlm.gpt.init_weights()
vlm.vit.init_weights()
# Copy the pretrained GPT weights into the VLM's GPT
vlm.gpt.load_state_dict(base_model.state_dict(), strict=True, assign=True)
del base_model  # free the standalone base model
# For DINOv2, load the pretrained backbone weights and freeze it (train only the projector)
if args.encoder == "dinov2":
    vlm.vit.load_dinov2_weights(args.dinov2_weights)
    vlm.freeze_vit_backbone()
    print0("Loaded DINOv2 pretrained weights; backbone frozen (training projector only)")

# Freeze the LLM, train only the ViT
vlm.set_llm_trainable(False)
n_trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in vlm.parameters())
print0(f"Trainable params (ViT only): {n_trainable:,} / {n_total:,} total")

# -----------------------------------------------------------------------------
# Optimizer: AdamW over the (trainable) ViT params only
vit_params = [p for p in vlm.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(vit_params, lr=args.vit_lr, betas=(0.9, 0.95), weight_decay=0.01)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# GradScaler for fp16
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

# -----------------------------------------------------------------------------
# Data
train_dataset = COCODataset(split="train", max_rows=args.max_coco_images * 5 if args.max_coco_images else None)
val_dataset = COCODataset(split="val", max_rows=args.max_coco_images * 5 if args.max_coco_images else None)
print0(f"Train pairs: {len(train_dataset):,} | Val pairs: {len(val_dataset):,}")

# Data generator: yields (images, caption_ids, caption_targets)
def coco_data_generator(dataset, buffer_size=100):
    """Infinite generator of (images, ids, targets) batches for image->caption training."""
    n = len(dataset)
    cursor = ddp_rank
    epoch = 1
    while True:
        rows = []
        for _ in range(args.device_batch_size):
            rows.append(dataset[cursor % n])
            cursor += ddp_world_size
            if cursor >= n:
                cursor = cursor % n
                epoch += 1
        images, captions = collate_coco_batch(rows, vit_config.image_size, args.normalize)
        # Tokenize captions: prepend BOS, targets = ids shifted by 1
        bos = tokenizer.get_bos_token_id()
        ids_list = [tokenizer.encode(c, prepend=bos) for c in captions]
        max_len = max(len(ids) for ids in ids_list)
        pad_id = bos  # pad with BOS (harmless, masked in targets)
        ids = torch.full((args.device_batch_size, max_len), pad_id, dtype=torch.long)
        targets = torch.full((args.device_batch_size, max_len), -1, dtype=torch.long)
        for i, ids_row in enumerate(ids_list):
            ids[i, :len(ids_row)] = torch.tensor(ids_row)
            # targets: predict next token; last position masked
            targets[i, :len(ids_row) - 1] = torch.tensor(ids_row[1:])
        yield (
            images.to(device=device, non_blocking=True),
            ids.to(device=device, dtype=torch.int32, non_blocking=True).contiguous(),
            targets.to(device=device, dtype=torch.int64, non_blocking=True).contiguous(),
        )

train_loader = coco_data_generator(train_dataset)
build_val_loader = lambda: coco_data_generator(val_dataset)

# -----------------------------------------------------------------------------
# LR schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmdown_iters = round(args.warmdown_ratio * args.num_iterations)
    if it < args.warmup_steps:
        return (it + 1) / args.warmup_steps
    elif it <= args.num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (args.num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# -----------------------------------------------------------------------------
# Training loop
x, y_ids, y_targets = next(train_loader)  # prefetch first batch
min_val_loss = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
while True:
    last_step = step == args.num_iterations
    flops_so_far = step * args.total_batch_size  # rough proxy (not exact FLOPs)

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
        wandb_run.log({"step": step, "val/loss": val_loss})
        vlm.train()
        vlm.set_llm_trainable(False)  # keep LLM frozen + in eval mode

    # Save checkpoint at the end
    if last_step:
        output_dirname = args.output_tag if args.output_tag else f"vlm_d{gpt_config.n_layer}"
        checkpoint_dir = os.path.join(get_base_dir(), "vlm_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            vlm.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "val_loss": val_loss,
                "gpt_config": gpt_config_kwargs,
                "vit_config": asdict(vit_config),
                "encoder": args.encoder,
                "image_token_id": image_token_id,
                "user_config": user_config,
            },
            rank=ddp_rank,
        )
        break

    # -------------------------------------------------------------------------
    # single training step
    synchronize()
    t0 = time.time()
    tokens_per_fwdbwd = args.device_batch_size
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
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
    # step the optimizer
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
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
