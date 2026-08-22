"""
COCO image-caption dataset for VLM training.

Downloads a small subset of COCO val2014 (5000 images, ~37K captions) and
pairs each image with its captions. Provides:
- prepare_coco(): download images + captions, build a parquet index
- COCODataset: iterable of (image_path, caption) pairs
- load_image(): PIL -> normalized tensor (3, image_size, image_size)

Run to prepare the data (one-time):
    python -m scripts.coco_data

The data is stored under ~/.cache/nanochat/coco/:
    val2014/            # image files
    captions_val2014.json
    coco_captions.parquet  # (image_path, caption, image_id)
"""

import os
import json
import shutil
import zipfile
import argparse
import urllib.request

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from nanochat.common import get_base_dir, print0

# -----------------------------------------------------------------------------
# COCO val2014 specifics
BASE_DIR = os.path.join(get_base_dir(), "coco")
IMAGES_DIR = os.path.join(BASE_DIR, "val2014")
IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"
CAPTIONS_URL = "http://images.cocodataset.org/annotations/captions_val2014.json"
PARQUET_PATH = os.path.join(BASE_DIR, "coco_captions.parquet")

# Image preprocessing constants (must match ViTConfig.image_size)
IMAGE_SIZE = 128
MEAN = 0.5
STD = 0.5


def _download(url, dest):
    """Download url to dest, skipping if it already exists."""
    if os.path.exists(dest):
        print0(f"Already exists, skipping: {dest}")
        return
    print0(f"Downloading {url} -> {dest}")
    tmp = dest + ".tmp"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    os.replace(tmp, dest)


def prepare_coco(max_images=None):
    """
    Download COCO val2014 images and captions, build the parquet index.

    Args:
        max_images: optional cap on the number of images to include (for tiny runs)
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1) Download and extract images
    images_zip = os.path.join(BASE_DIR, "val2014.zip")
    _download(IMAGES_ZIP_URL, images_zip)
    if not os.listdir(IMAGES_DIR):
        print0("Extracting images...")
        with zipfile.ZipFile(images_zip, "r") as z:
            z.extractall(IMAGES_DIR)
        os.remove(images_zip)  # free space after extraction

    # 2) Download captions
    captions_path = os.path.join(BASE_DIR, "captions_val2014.json")
    _download(CAPTIONS_URL, captions_path)
    with open(captions_path, "r") as f:
        captions_data = json.load(f)

    # 3) Build image_id -> caption list mapping
    captions_by_image = {}
    for ann in captions_data["annotations"]:
        captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

    # 4) Enumerate image files on disk
    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if max_images is not None:
        image_files = image_files[:max_images]

    # 5) Build rows: one row per (image, caption) pair
    rows = []
    for fname in image_files:
        # COCO filenames are like COCO_val2014_000000123456.jpg
        stem = os.path.splitext(fname)[0]
        image_id = int(stem.rsplit("_", 1)[-1])
        for caption in captions_by_image.get(image_id, []):
            rows.append({
                "image_path": fname,
                "caption": caption,
                "image_id": image_id,
            })

    # 6) Write parquet
    table = pa.table({
        "image_path": [r["image_path"] for r in rows],
        "caption": [r["caption"] for r in rows],
        "image_id": [r["image_id"] for r in rows],
    })
    pq.write_table(table, PARQUET_PATH)
    n_images = len(set(r["image_id"] for r in rows))
    print0(f"Wrote {len(rows):,} (image, caption) pairs for {n_images:,} images to {PARQUET_PATH}")
    return PARQUET_PATH


# -----------------------------------------------------------------------------
# Image loading / preprocessing

def load_image(image_path, image_size=IMAGE_SIZE):
    """
    Load an image from disk and preprocess it into a normalized tensor.

    Args:
        image_path: path to the image file
        image_size: target square size (resized to image_size x image_size)

    Returns:
        tensor of shape (3, image_size, image_size), float32, in [-1, 1]
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    # (H, W, 3) uint8 -> (3, H, W) float in [0, 1]
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    # normalize to [-1, 1]
    x = (x - MEAN) / STD
    return x


def load_image_from_bytes(data, image_size=IMAGE_SIZE):
    """Same as load_image but from raw bytes (e.g. from a parquet column)."""
    import io
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    x = (x - MEAN) / STD
    return x


# -----------------------------------------------------------------------------
# Dataset

class COCODataset:
    """
    Iterable over (image_path, caption) pairs from the COCO parquet index.

    Each __getitem__ returns a dict:
        {"image_path": str, "caption": str, "image_id": int}
    """

    def __init__(self, parquet_path=None, split="train", max_rows=None):
        self.parquet_path = parquet_path or PARQUET_PATH
        assert os.path.exists(self.parquet_path), \
            f"COCO parquet not found at {self.parquet_path}. Run `python -m scripts.coco_data` first."
        self.table = pq.read_table(self.parquet_path)
        # Simple train/val split: last 10% of rows for val
        n = len(self.table)
        if split == "val":
            start = int(n * 0.9)
            self.table = self.table.slice(start)
        else:
            self.table = self.table.slice(0, start)
        if max_rows is not None:
            self.table = self.table.slice(0, max_rows)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        return {
            "image_path": self.table["image_path"][idx].as_py(),
            "caption": self.table["caption"][idx].as_py(),
            "image_id": self.table["image_id"][idx].as_py(),
        }

    def image_full_path(self, image_path):
        return os.path.join(IMAGES_DIR, image_path)


def collate_coco_batch(batch, image_size=IMAGE_SIZE):
    """
    Collate a list of COCODataset rows into a batch of tensors.

    Args:
        batch: list of dicts from COCODataset
        image_size: target image size

    Returns:
        images: (B, 3, image_size, image_size) float tensor
        captions: list[str]
    """
    images = torch.stack([
        load_image(os.path.join(IMAGES_DIR, r["image_path"]), image_size)
        for r in batch
    ])
    captions = [r["caption"] for r in batch]
    return images, captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare COCO val2014 image-caption data")
    parser.add_argument("--max-images", type=int, default=None,
                        help="cap on number of images (for tiny runs)")
    args = parser.parse_args()
    prepare_coco(max_images=args.max_images)
