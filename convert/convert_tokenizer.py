#!/usr/bin/env python3
import argparse
import json
import pickle
import torch
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def extract_from_tiktoken(tokenizer_obj):
    print("→ Extracting from tokenizer.pkl ...")

    # Find vocab
    if hasattr(tokenizer_obj, "encoder"):
        encoder = tokenizer_obj.encoder
        print(f"  Found encoder with {len(encoder)} tokens.")
    else:
        raise ValueError("tokenizer.pkl has no 'encoder' attribute.")

    # Find merges
    if hasattr(tokenizer_obj, "bpe_ranks"):
        print("  Found bpe_ranks.")
        ranks = tokenizer_obj.bpe_ranks
        merges = sorted(ranks.items(), key=lambda x: x[1])
        merges = [list(k) for k, _ in merges]
        print(f"  Found {len(merges)} merge rules.")
    else:
        raise ValueError("tokenizer.pkl has no 'bpe_ranks' attribute.")

    # Special tokens
    special_tokens = {}
    if hasattr(tokenizer_obj, "special_tokens"):
        special_tokens = tokenizer_obj.special_tokens
        print(f"  Found special tokens: {special_tokens}")

    return encoder, merges, special_tokens


def extract_from_token_bytes(token_bytes_path):
    print("→ Extracting vocab from token_bytes.pt ...")

    tbl = torch.load(token_bytes_path, map_location="cpu")

    encoder = {}
    for i, row in enumerate(tbl):
        token = bytes(row.tolist()).decode("utf-8", errors="replace")
        encoder[token] = i

    print(f"  Constructed vocab from {len(encoder)} entries.")

    merges = []  # cannot recover merges
    print("  WARNING: No merges from token_bytes.pt; tokenizer will not match original.")

    return encoder, merges, {}


def save_hf_tokenizer(encoder, merges, output_dir, special_tokens={}):
    print("→ Building HuggingFace tokenizer ...")

    # Convert merges to HuggingFace string format if needed
    hf_merges = []
    for m in merges:
        if isinstance(m[0], (bytes, bytearray)):
            a = m[0].decode("utf-8", "replace")
            b = m[1].decode("utf-8", "replace")
        else:
            a, b = m
        hf_merges.append(f"{a} {b}")

    # Build HF tokenizer
    tokenizer = Tokenizer(BPE(encoder=encoder, merges=hf_merges))

    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    # Add special tokens
    for tok, tok_id in special_tokens.items():
        tokenizer.add_special_tokens([tok])

    os.makedirs(output_dir, exist_ok=True)
    tok_json_path = os.path.join(output_dir, "tokenizer.json")

    tokenizer.save(tok_json_path)
    print(f"✔ Saved tokenizer.json to {tok_json_path}")

    # Create tokenizer_config.json
    config = {
        "model_type": "gpt2",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": len(encoder),
        "add_prefix_space": False
    }

    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"✔ Saved tokenizer_config.json")

    print("\nConversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert TikToken tokenizer.pkl → HF tokenizer.json")
    parser.add_argument("--tokenizer_pkl", type=str, help="Path to tokenizer.pkl")
    parser.add_argument("--token_bytes", type=str, help="Path to token_bytes.pt", default=None)
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    encoder = None
    merges = None
    special_tokens = {}

    # Try tokenizer.pkl first
    if args.tokenizer_pkl and os.path.exists(args.tokenizer_pkl):
        print(f"Loading tokenizer.pkl from {args.tokenizer_pkl}")
        with open(args.tokenizer_pkl, "rb") as f:
            obj = pickle.load(f)

        try:
            encoder, merges, special_tokens = extract_from_tiktoken(obj)
        except Exception as e:
            print("Failed to extract from tokenizer.pkl:", e)

    # Fallback to token_bytes.pt
    if encoder is None and args.token_bytes and os.path.exists(args.token_bytes):
        encoder, merges, special_tokens = extract_from_token_bytes(args.token_bytes)

    if encoder is None:
        raise RuntimeError("No valid tokenizer.pkl or token_bytes.pt found; cannot build tokenizer.")

    save_hf_tokenizer(encoder, merges, args.out, special_tokens)


if __name__ == "__main__":
    main()