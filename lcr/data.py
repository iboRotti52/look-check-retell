"""Flickr8k loading (HF mirror `jxie/flickr8k`, 6,000/1,000/1,000 split) and the A/B halves."""
from __future__ import annotations

import glob
import io
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from . import config

SPLITS = ("train", "validation", "test")
CAP_COLS = [f"caption_{i}" for i in range(5)]


@lru_cache(maxsize=None)
def _frame(split: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(config.DATA / "flickr8k" / "data" / f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files for split '{split}'. Run scripts/00_download.py first."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def num_images(split: str) -> int:
    return len(_frame(split))


def image_id(split: str, idx: int) -> str:
    return f"{split}_{idx:04d}"


def parse_id(iid: str) -> tuple[str, int]:
    split, idx = iid.rsplit("_", 1)
    return split, int(idx)


def load_image(iid: str) -> Image.Image:
    split, idx = parse_id(iid)
    cell = _frame(split).iloc[idx]["image"]
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def references(iid: str) -> list[str]:
    split, idx = parse_id(iid)
    row = _frame(split).iloc[idx]
    return [str(row[c]).strip() for c in CAP_COLS]


def ids(split: str) -> list[str]:
    """Image ids of 'train', 'validation', 'test', or the halves 'A' / 'B' of train."""
    if split in ("A", "B"):
        perm = np.random.default_rng(config.SPLIT_SEED).permutation(num_images("train"))
        half = len(perm) // 2
        part = perm[:half] if split == "A" else perm[half:]
        return [image_id("train", int(i)) for i in sorted(part)]
    return [image_id(split, i) for i in range(num_images(split))]
