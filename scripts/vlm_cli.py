"""
VLM inference / demo CLI: image + text prompt -> generated response.

Run as:
    python -m scripts.vlm_cli --image photo.jpg --prompt "Describe this image."

Or with a specific checkpoint:
    python -m scripts.vlm_cli --image photo.jpg --prompt "What is in this image?" \
        --source vlm_sft --model-tag vlmsft_d12 --temperature 0.7

Requires a VLM checkpoint (from scripts.vit_pretrain / scripts.vlm_sft).
"""

import argparse
import torch

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_vlm
from scripts.coco_data import load_image, IMAGE_SIZE

parser = argparse.ArgumentParser(description='VLM: image + prompt -> response')
parser.add_argument('--image', type=str, required=True, help='Path to the input image')
parser.add_argument('-p', '--prompt', type=str, default='Describe this image.',
                    help='Text prompt to condition on')
parser.add_argument('-i', '--source', type=str, default="vlm_sft",
                    help="Source of the model: vlm_sft|vlm")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens to generate')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'],
                    help='Device type for inference: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

# Init compute
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

# Load the VLM and tokenizer
vlm, tokenizer, meta = load_vlm(args.source, device, phase="eval",
                                 model_tag=args.model_tag, step=args.step)
vlm.eval()

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
image_token_id = vlm.image_token_id

# Load and preprocess the image
print(f"Loading image: {args.image}")
image = load_image(args.image, IMAGE_SIZE).unsqueeze(0).to(device)  # (1, 3, H, W)

# Build the prompt token sequence:
#   <bos> <user_start> <image> {prompt} <user_end> <assistant_start>
prompt_tokens = [bos, user_start, image_token_id]
prompt_tokens.extend(tokenizer.encode(args.prompt))
prompt_tokens.append(user_end)
prompt_tokens.append(assistant_start)

print(f"\nPrompt: {args.prompt}")
print("-" * 50)
print("Assistant: ", end="", flush=True)

# Generate
generate_kwargs = {
    "max_tokens": args.max_tokens,
    "temperature": args.temperature,
    "top_k": args.top_k,
}
response_tokens = []
for token in vlm.generate(image, prompt_tokens, **generate_kwargs):
    response_tokens.append(token)
    token_text = tokenizer.decode([token])
    print(token_text, end="", flush=True)
    # Stop early on assistant_end or bos
    if token == tokenizer.encode_special("<|assistant_end|>") or token == bos:
        break
print()
print("-" * 50)
print(f"Generated {len(response_tokens)} tokens")
