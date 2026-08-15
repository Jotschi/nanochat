"""
Convert a nanochat checkpoint into HuggingFace format so vLLM can serve it.

    python vllm/convert_to_hf.py                     # sft checkpoint -> $NANOCHAT_BASE_DIR/hf_export/d12-sft
    python vllm/convert_to_hf.py --source base
    python vllm/convert_to_hf.py --check             # report mappability and exit

Read vllm/README.md first. In short: transformers' NanoChatForCausalLM targets an
older nanochat architecture than this repo trains, so several tensors in a current
checkpoint have no destination. This script REFUSES to write a model in that case
rather than dropping them, because a model missing its value embeddings still
loads and still generates -- just nonsense. Pass --force only if you know why you
want that.
"""

import os
import sys
import json
import glob
import argparse

import torch

# Run as a path (`python vllm/convert_to_hf.py`) rather than a module, so put the
# repo root on sys.path ourselves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Weight mapping to transformers/models/nanochat

DIRECT = {
    "transformer.wte.weight": "model.embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
}

PER_LAYER = {
    "attn.c_q.weight": "self_attn.q_proj.weight",
    "attn.c_k.weight": "self_attn.k_proj.weight",
    "attn.c_v.weight": "self_attn.v_proj.weight",
    "attn.c_proj.weight": "self_attn.o_proj.weight",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_proj.weight": "mlp.fc2.weight",
}

# Tensors this repo's architecture has and transformers' NanoChat does not.
# Keeping the reason next to each one, because "unmapped" alone invites someone
# to delete the check.
UNSUPPORTED = {
    "value_embeds.": "per-layer value embeddings added to V; no equivalent",
    ".attn.ve_gate.": "gate blending the value embedding into V; no equivalent",
    "smear_gate": "token-smearing gate over the previous token",
    "smear_lambda": "scale for the smear gate",
    "backout_lambda": "backout of the first block's contribution",
    "resid_lambdas": "per-layer residual stream scaling",
    "x0_lambdas": "per-layer re-blending of the initial embedding",
}


def classify(state_dict, n_layer):
    """Split checkpoint keys into (mapped, unsupported, unknown)."""
    mapped, unsupported, unknown = {}, [], []
    for key, tensor in state_dict.items():
        reason = next((r for pat, r in UNSUPPORTED.items() if pat in key), None)
        if reason is not None:
            unsupported.append((key, reason))
            continue
        if key in DIRECT:
            mapped[DIRECT[key]] = tensor
            continue
        if key.startswith("transformer.h."):
            rest = key[len("transformer.h."):]
            idx, suffix = rest.split(".", 1)
            if suffix in PER_LAYER:
                mapped[f"model.layers.{idx}.{PER_LAYER[suffix]}"] = tensor
                continue
        unknown.append(key)
    return mapped, unsupported, unknown


def add_norm_weights(mapped, n_layer, hidden_size, head_dim):
    """
    nanochat normalises with parameter-free F.rms_norm; transformers' NanoChat
    uses learnable RMSNorm. An all-ones weight makes them numerically identical,
    so these are safe to synthesise (unlike anything in UNSUPPORTED).
    """
    for i in range(n_layer):
        for name, size in (
            ("input_layernorm", hidden_size),
            ("post_attention_layernorm", hidden_size),
            ("self_attn.q_norm", head_dim),
            ("self_attn.k_norm", head_dim),
        ):
            mapped[f"model.layers.{i}.{name}.weight"] = torch.ones(size, dtype=torch.float32)
    mapped["model.norm.weight"] = torch.ones(hidden_size, dtype=torch.float32)


def build_config(meta):
    mc = meta["model_config"]
    n_embd, n_head = mc["n_embd"], mc["n_head"]
    return {
        "architectures": ["NanoChatForCausalLM"],
        "model_type": "nanochat",
        "vocab_size": mc["vocab_size"],
        "hidden_size": n_embd,
        "intermediate_size": n_embd * 4,
        "num_hidden_layers": mc["n_layer"],
        "num_attention_heads": n_head,
        "num_key_value_heads": mc["n_kv_head"],
        "max_position_embeddings": mc["sequence_len"],
        "hidden_act": "relu2",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "final_logit_softcapping": 15.0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 100000.0},
        "torch_dtype": "bfloat16",
    }


def find_checkpoint(base_dir, source, depth, step):
    subdir = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[source]
    ckpt_dir = os.path.join(base_dir, subdir, f"d{depth}")
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"No checkpoint directory {ckpt_dir}")
    if step is None:
        steps = sorted(
            int(os.path.basename(p)[len("model_"):-len(".pt")])
            for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt"))
        )
        if not steps:
            raise SystemExit(f"No model_*.pt in {ckpt_dir}")
        step = steps[-1]
    model_path = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(ckpt_dir, f"meta_{step:06d}.json")
    for p in (model_path, meta_path):
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}")
    return model_path, meta_path, step


def main():
    parser = argparse.ArgumentParser(description="Convert a nanochat checkpoint to HuggingFace format")
    parser.add_argument("--source", default="sft", choices=["base", "sft", "rl"])
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--step", type=int, default=None, help="default: highest step present")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--check", action="store_true", help="report mappability and exit")
    parser.add_argument("--force", action="store_true",
                        help="write the model even though tensors will be dropped (produces a broken model)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    model_path, meta_path, step = find_checkpoint(base_dir, args.source, args.depth, args.step)
    meta = json.load(open(meta_path))
    state_dict = torch.load(model_path, map_location="cpu")
    n_layer = meta["model_config"]["n_layer"]

    print(f"Checkpoint : {model_path}")
    print(f"Step       : {step}   val_bpb {meta.get('val_bpb')}")
    print(f"Tensors    : {len(state_dict)}")

    mapped, unsupported, unknown = classify(state_dict, n_layer)
    print(f"Mapped     : {len(mapped)}")

    if unknown:
        print(f"\nUnrecognised tensors ({len(unknown)}) - the mapping needs updating:")
        for k in unknown:
            print(f"  {k}")

    if unsupported:
        print(f"\nUnsupported by transformers' NanoChat ({len(unsupported)} tensors):")
        seen = set()
        for key, reason in unsupported:
            if reason not in seen:
                seen.add(reason)
                print(f"  {reason}")
        print("\n  These carry trained weights. Dropping them yields a model that loads")
        print("  and generates, but is not the model you trained. See vllm/README.md.")

    if args.check:
        return

    blocked = unsupported or unknown
    if blocked and not args.force:
        raise SystemExit(
            "\nRefusing to write an incomplete model. Re-run with --check to inspect, "
            "or --force if you specifically want the truncated conversion."
        )
    if blocked and args.force:
        print("\n--force given: writing anyway. The result will NOT match the trained model.")

    mc = meta["model_config"]
    head_dim = mc["n_embd"] // mc["n_head"]
    add_norm_weights(mapped, n_layer, mc["n_embd"], head_dim)

    out_dir = args.out_dir or os.path.join(base_dir, "hf_export", f"d{args.depth}-{args.source}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from safetensors.torch import save_file
    except ImportError:
        raise SystemExit("safetensors is required: uv pip install safetensors")

    tensors = {k: v.to(torch.bfloat16).contiguous() for k, v in mapped.items()}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(build_config(meta), f, indent=2)

    print(f"\nWrote {len(tensors)} tensors to {out_dir}")
    print("Tokenizer files are NOT written yet - see vllm/README.md.")


if __name__ == "__main__":
    main()
