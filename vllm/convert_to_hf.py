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


def expected_tensor_count(n_layer):
    """
    transformers' NanoChat keeps nanochat's parameter-free RMSNorm, so a converted
    model is exactly six weights per layer plus embed_tokens and lm_head - no norm
    tensors at all. Verified against nanochat-students/nanochat-d20, which has
    20*6+2 = 122. Synthesising all-ones norm weights (a reasonable-sounding guess)
    makes vLLM reject the load with "There is no module or parameter named
    model.layers.0.input_layernorm.weight in TransformersForCausalLM".
    """
    return n_layer * len(PER_LAYER) + len(DIRECT)


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


# -----------------------------------------------------------------------------
# Tokenizer: tiktoken (rustbpe) -> HuggingFace tokenizers

# nanochat's render_conversation, as Jinja. BOS, then each message wrapped in its
# role's start/end specials, and a trailing <|assistant_start|> to prime a reply
# (the equivalent of render_for_completion).
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "<|user_start|>{{ message['content'] }}<|user_end|>"
    "{%- elif message['role'] == 'assistant' %}"
    "<|assistant_start|>{{ message['content'] }}<|assistant_end|>"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant_start|>{%- endif %}"
)


def _byte_to_unicode():
    """GPT-2's reversible byte <-> printable-unicode map, as used by ByteLevel."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _recover_merges(mergeable_ranks):
    """
    tiktoken stores only token->rank, not the merge list HF needs. Recover each
    merge by re-running BPE on the token restricted to lower-rank merges: the two
    parts left at that point are exactly the pair that produced it.
    """
    merges = []
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        parts = [bytes([b]) for b in token]
        while len(parts) > 1:
            min_idx, min_rank = None, None
            for i in range(len(parts) - 1):
                r = mergeable_ranks.get(parts[i] + parts[i + 1])
                if r is not None and r < rank and (min_rank is None or r < min_rank):
                    min_idx, min_rank = i, r
            if min_idx is None:
                break
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        if len(parts) != 2:
            raise ValueError(f"Could not recover a single merge for {token!r} (got {len(parts)} parts)")
        merges.append((parts[0], parts[1]))
    return merges


def write_tokenizer(out_dir, base_dir):
    import pickle
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    pkl = os.path.join(base_dir, "tokenizer", "tokenizer.pkl")
    if not os.path.exists(pkl):
        raise SystemExit(f"No tokenizer at {pkl}; train one with scripts.tok_train")
    with open(pkl, "rb") as f:
        enc = pickle.load(f)

    b2u = _byte_to_unicode()
    to_str = lambda bs: "".join(b2u[b] for b in bs)

    ranks = enc._mergeable_ranks
    vocab = {to_str(tok): rank for tok, rank in ranks.items()}
    merges = [(to_str(a), to_str(b)) for a, b in _recover_merges(ranks)]
    specials = dict(enc._special_tokens)
    for name, tid in specials.items():
        vocab[name] = tid

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges, fuse_unk=False))
    # Same two-stage split as tiktoken: the GPT-4-style regex, then byte-level
    # mapping. use_regex=False because the Split above already did that job.
    tokenizer.pre_tokenizer = Sequence([
        Split(Regex(enc._pat_str), behavior="isolated", invert=False),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens(list(specials.keys()))
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))

    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|bos|>",
            "eos_token": "<|assistant_end|>",
            "pad_token": "<|bos|>",
            "model_max_length": 2048,
            "clean_up_tokenization_spaces": False,
            "chat_template": CHAT_TEMPLATE,
            "added_tokens_decoder": {
                str(tid): {"content": name, "special": True, "single_word": False,
                           "lstrip": False, "rstrip": False, "normalized": False}
                for name, tid in specials.items()
            },
        }, f, indent=2)

    return len(vocab), len(merges), specials


def verify_tokenizer(out_dir, base_dir, samples):
    """Round-trip a few strings through both tokenizers and compare ids."""
    import pickle
    from tokenizers import Tokenizer

    with open(os.path.join(base_dir, "tokenizer", "tokenizer.pkl"), "rb") as f:
        enc = pickle.load(f)
    hf = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))

    mismatches = 0
    for text in samples:
        want = enc.encode_ordinary(text)
        got = hf.encode(text, add_special_tokens=False).ids
        if want != got:
            mismatches += 1
            if mismatches <= 2:
                print(f"  MISMATCH on {text[:60]!r}\n    tiktoken: {want[:16]}\n    hf:       {got[:16]}")
    return mismatches


def find_checkpoint(base_dir, source, model_tag, step):
    subdir = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[source]
    ckpt_dir = os.path.join(base_dir, subdir, model_tag)
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
    parser.add_argument("--model-tag", default="d12hf",
                        help="checkpoint directory name, e.g. d12 or d12hf (the --model-tag used at training time)")
    parser.add_argument("--step", type=int, default=None, help="default: highest step present")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--check", action="store_true", help="report mappability and exit")
    parser.add_argument("--force", action="store_true",
                        help="write the model even though tensors will be dropped (produces a broken model)")
    args = parser.parse_args()

    base_dir = get_base_dir()
    model_path, meta_path, step = find_checkpoint(base_dir, args.source, args.model_tag, args.step)
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
    expected = expected_tensor_count(n_layer)
    if len(mapped) != expected:
        raise SystemExit(f"Expected {expected} tensors for a {n_layer}-layer NanoChat, got {len(mapped)}")

    out_dir = args.out_dir or os.path.join(base_dir, "hf_export", f"{args.model_tag}-{args.source}")
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

    n_vocab, n_merges, specials = write_tokenizer(out_dir, base_dir)
    print(f"Wrote tokenizer: {n_vocab:,} vocab, {n_merges:,} merges, {len(specials)} special tokens")

    # A tokenizer that silently disagrees with the one the model was trained on is
    # the kind of bug that shows up as "the model is just bad", so check it here.
    samples = [
        "Es war einmal ein kleiner Astronaut namens Mira.",
        "Schreib ein Abenteuer von Mira in das Raumschiff nur mit Aris",
        "Größe, Übermut und weiße Füße - äöüß!",
        "The quick brown fox jumps over the lazy dog 1234567890.",
        "  leading spaces\tand\ttabs\nand newlines\n\n",
    ]
    mismatches = verify_tokenizer(out_dir, base_dir, samples)
    if mismatches:
        raise SystemExit(f"\nTokenizer round-trip FAILED on {mismatches}/{len(samples)} samples - "
                         "the exported tokenizer does not match the trained one.")
    print(f"Tokenizer round-trip matches tiktoken on all {len(samples)} samples")


if __name__ == "__main__":
    main()
