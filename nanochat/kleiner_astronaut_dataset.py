"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import json
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
index_to_filename = lambda index: f"{index:05d}.jsonl" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "astronaut_basedata")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_jsonl_files(split, data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    #print(data_dir)
    jsonl_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.jsonl') and split in f and not f.endswith('.tmp')
    ])
    jsonl_path = [os.path.join(data_dir, f) for f in jsonl_files]
    return jsonl_path

def jsonl_iter_batched(split, start=0, step=1):
    if split == "val":
        split="test"

    assert split in ["train", "test"], "split must be 'train' or 'test'"
    jsonl_path = list_jsonl_files(split)

    for filepath in jsonl_path:
        lines = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start:
                    continue
                if (i - start) % step != 0:
                    continue

                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                text = entry['request'] + " " + entry['story']
                #+ " " + entry['question'] + " " + entry['answer']
                lines.append(text)

        if lines:
            yield lines
# -----------------------------------------------------------------------------
def download_single_file(index):
    print("Noop")    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Kleiner Astronaut dataset")
    #parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    #parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    #args = parser.parse_args()

    #print(f"Target directory: {DATA_DIR}")
    #print()
    #download_single_file
    #print(list_jsonl_files())
    #for batch in jsonl_iter_batched(split="train"):
    #    #print("BATCH")
    #    print(batch)
    #    #print(next(batch))
    #print(f"JSONL Dir {DATA_DIR}")
