"""
Materialize the Kleiner Astronaut story corpus as parquet shards for pretraining.

The stories are produced by the Java generator in `dataset-processor/` and land
in jsonl files with a `text` field and an MD5 `hash`. Rather than teaching the
dataloader to read jsonl, we write the corpus out in the shape nanochat already
expects: parquet shards with a single `text` column. Everything downstream --
`nanochat/dataloader.py`, `scripts/tok_train.py`, `scripts/tok_eval.py` -- then
works unmodified, resolving shards through `nanochat.dataset.list_parquet_files`.

The held-out split is written as the *last* shard, because `_document_batches`
in dataloader.py uses `parquet_paths[:-1]` for train and `parquet_paths[-1:]`
for val. It uses the same hash-bucket predicate as the Java `Split` class, so a
story held out here is also held out of the chat conversations.

Usage:
    python -m nanochat.kleiner_astronaut            # write the shards
    python -m nanochat.kleiner_astronaut --stats    # just report corpus size
"""

import os
import glob
import json
import math
import argparse
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------

# Must match de.jotschi.ai.converter.stage3.Split
VAL_FRACTION = 0.05
_BUCKETS = 10_000

# Rows per parquet row group. The dataloader shards across DDP ranks at row-group
# granularity, so this wants to be small enough to give every rank work.
ROW_GROUP_SIZE = 1024
# Stories per shard. ~130k stories at 16k/shard is a handful of files.
SHARD_SIZE = 16384

DEFAULT_STORY_GLOBS = [
    "dataset-processor/dataset/stories*.jsonl",
    "dataset-processor/dataset/stories_done/stories*.jsonl",
]
DEFAULT_OUT_DIRNAME = "base_data_astronaut"


def is_held_out(story_hash, val_fraction=VAL_FRACTION):
    """Deterministic story-level split, mirroring the Java Split class."""
    if not story_hash or len(story_hash) < 8:
        raise ValueError(f"Need at least an 8 character hex hash, got: {story_hash!r}")
    bucket = int(story_hash[:8], 16) % _BUCKETS
    return bucket < round(val_fraction * _BUCKETS)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iter_stories(patterns, include_hf=False):
    """
    Yield (hash, text) for every unique story, deduplicated by hash.

    The story files overlap: `stories_done/` holds earlier batches and the newer
    ones were appended over time. 133,464 rows collapse to 133,204 unique.
    """
    root = _repo_root()
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern if os.path.isabs(pattern) else os.path.join(root, pattern))))
    if not paths:
        raise FileNotFoundError(
            f"No story files matched {patterns}. Expected the Java generator's output under "
            "dataset-processor/dataset/ -- see spec/DATA.md."
        )

    seen = set()
    for path in paths:
        print(f"Reading {os.path.relpath(path, root)}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get("text") or record.get("story")
                story_hash = record.get("hash")
                if not text or not story_hash:
                    continue
                if story_hash in seen:
                    continue
                seen.add(story_hash)
                # NFC-normalize so the tokenizer sees one encoding of each umlaut.
                yield story_hash, unicodedata.normalize("NFC", text)

    if include_hf:
        yield from _iter_hf_stories(seen)


def _iter_hf_stories(seen):
    """
    The published Jotschi/kleiner-astronaut dataset. A separate, older generation
    (Phi-3, April 2024) with a different length distribution -- avg 1957 chars vs
    963 -- so it is opt-in rather than merged by default.
    """
    from tasks.common import load_hub_dataset

    import hashlib
    for split in ("train", "test"):
        dataset = load_hub_dataset("Jotschi/kleiner-astronaut", split=split)
        print(f"Read {len(dataset):,} rows from Jotschi/kleiner-astronaut:{split}")
        for i in range(len(dataset)):
            text = dataset[i].get("text")
            if not text:
                continue
            # The HF rows carry no hash; derive one the same way the Java side does.
            story_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            if story_hash in seen:
                continue
            seen.add(story_hash)
            yield story_hash, unicodedata.normalize("NFC", text)


def _write_shard(rows, path):
    table = pa.table({"text": pa.array(rows, type=pa.string())})
    pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE, compression="zstd")
    return len(rows)


def materialize(out_dir, patterns, val_fraction=VAL_FRACTION, include_hf=False, shard_size=SHARD_SIZE):
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, "*.parquet")):
        os.remove(stale)

    train_buffer, val_rows = [], []
    shard_index = 0
    n_train = n_val = 0
    chars_train = chars_val = 0

    for story_hash, text in iter_stories(patterns, include_hf=include_hf):
        if is_held_out(story_hash, val_fraction):
            val_rows.append(text)
            n_val += 1
            chars_val += len(text)
        else:
            train_buffer.append(text)
            n_train += 1
            chars_train += len(text)
            if len(train_buffer) >= shard_size:
                _write_shard(train_buffer, os.path.join(out_dir, f"shard_{shard_index:05d}.parquet"))
                shard_index += 1
                train_buffer = []

    if train_buffer:
        _write_shard(train_buffer, os.path.join(out_dir, f"shard_{shard_index:05d}.parquet"))
        shard_index += 1
    if not val_rows:
        raise RuntimeError("Validation split is empty; check --val-fraction")

    # The val shard must sort last: list_parquet_files() sorts by filename and
    # the dataloader takes the final one as validation.
    _write_shard(val_rows, os.path.join(out_dir, f"shard_{shard_index:05d}_val.parquet"))

    total_chars = chars_train + chars_val
    print()
    print(f"Train stories : {n_train:,} in {shard_index} shard(s)  ({chars_train:,} chars)")
    print(f"Val stories   : {n_val:,} in 1 shard  ({chars_val:,} chars)")
    print(f"Total         : {n_train + n_val:,} stories, {total_chars:,} chars")
    print(f"Output        : {out_dir}")
    return n_train, n_val, total_chars


def _measured_chars_per_token(sample_texts):
    """
    Compression ratio from the trained tokenizer, or None if there isn't one yet.

    Measuring beats guessing: on this corpus the real ratio is ~4.6, so the 3.8
    fallback overestimates the token count by about 20% and would stretch the
    training horizon by the same amount.
    """
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    except Exception:
        return None
    chars = sum(len(t) for t in sample_texts)
    tokens = sum(len(ids) for ids in tokenizer.encode(sample_texts))
    return chars / tokens if tokens else None


def report_stats(patterns, include_hf, total_batch_size, epochs, chars_per_token):
    n = 0
    total_chars = 0
    sample = []
    for _, text in iter_stories(patterns, include_hf=include_hf):
        n += 1
        total_chars += len(text)
        if len(sample) < 2000:
            sample.append(text)

    measured = _measured_chars_per_token(sample)
    if measured is not None:
        chars_per_token = measured
        source = "measured with the trained tokenizer"
    else:
        source = f"estimate; train the tokenizer for a real number"

    tokens = total_chars / chars_per_token
    print()
    print(f"Unique stories       : {n:,}")
    print(f"Total characters     : {total_chars:,}")
    print(f"Average chars/story  : {total_chars // max(n, 1):,}")
    print(f"Corpus tokens        : {tokens:,.0f}  ({chars_per_token:.2f} chars/token, {source})")
    print()
    print("Training horizon -- see spec/TRAINING_STRATEGY.md. Set --num-iterations")
    print("explicitly; --target-param-data-ratio would imply far more epochs than")
    print("this corpus supports.")
    print(f"  total_batch_size = {total_batch_size:,} tokens")
    for e in range(1, epochs + 1):
        iters = math.ceil(e * tokens / total_batch_size)
        print(f"  {e} epoch(s) -> --num-iterations {iters:,}")


def main():
    parser = argparse.ArgumentParser(description="Materialize the Kleiner Astronaut corpus as parquet shards")
    parser.add_argument("--out-dir", type=str, default=None,
                        help=f"output directory (default: $NANOCHAT_BASE_DIR/{DEFAULT_OUT_DIRNAME})")
    parser.add_argument("--stories", type=str, action="append", default=None,
                        help="glob for story jsonl files, repeatable (default: dataset-processor/dataset)")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION, help="held-out fraction")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="stories per shard")
    parser.add_argument("--include-hf", action="store_true",
                        help="also pull the published Jotschi/kleiner-astronaut dataset")
    parser.add_argument("--stats", action="store_true", help="report corpus size and exit")
    parser.add_argument("--total-batch-size", type=int, default=524288,
                        help="tokens per step, used by --stats to suggest --num-iterations")
    parser.add_argument("--epochs", type=int, default=6, help="how many epoch rows --stats prints")
    parser.add_argument("--chars-per-token", type=float, default=3.8,
                        help="estimate used by --stats before the tokenizer exists")
    args = parser.parse_args()

    patterns = args.stories if args.stories else DEFAULT_STORY_GLOBS

    if args.stats:
        report_stats(patterns, args.include_hf, args.total_batch_size, args.epochs, args.chars_per_token)
        return

    out_dir = args.out_dir or os.path.join(get_base_dir(), DEFAULT_OUT_DIRNAME)
    materialize(out_dir, patterns, val_fraction=args.val_fraction,
                include_hf=args.include_hf, shard_size=args.shard_size)
    print()
    print("Point the training scripts at it with:")
    print(f"  export NANOCHAT_DATA_DIR={out_dir}")


if __name__ == "__main__":
    main()
