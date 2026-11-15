"""On-disk caches: BLIP patch embeddings and OWLv2 proposals, one file set per split."""
from __future__ import annotations

import json

import numpy as np

from . import config, data

K = config.TOP_PROPOSALS
OWL_FIELDS = {"boxes": (K, 4), "objectness": (K,), "embeds": (K, 512), "shift": (K,), "scale": (K,)}


def _dir(kind: str):
    d = config.CACHE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def blip_path(split: str):
    return _dir("blip") / f"{split}.f16.npy"


def open_blip(split: str, mode: str = "r") -> np.memmap:
    n = data.num_images(split)
    return np.lib.format.open_memmap(blip_path(split), mode=mode, dtype=np.float16, shape=(n, 577, 768)) \
        if mode == "w+" else np.load(blip_path(split), mmap_mode="r")


def owl_path(split: str):
    return _dir("owl") / f"{split}.npz"


def save_owl(split: str, props: list[dict[str, np.ndarray]], sizes: list[tuple[int, int]]):
    n = len(props)
    arrs = {k: np.zeros((n, *shape), dtype=np.float16) for k, shape in OWL_FIELDS.items()}
    count = np.zeros(n, dtype=np.int16)
    for i, p in enumerate(props):
        c = len(p["objectness"])
        count[i] = c
        for k in OWL_FIELDS:
            arrs[k][i, :c] = p[k]
    np.savez(owl_path(split), count=count, sizes=np.array(sizes, dtype=np.int32), **arrs)


class OwlCache:
    def __init__(self, split: str):
        z = np.load(owl_path(split))
        self.z = {k: z[k] for k in z.files}

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        c = int(self.z["count"][idx])
        return {k: self.z[k][idx, :c] for k in OWL_FIELDS}

    def size(self, idx: int) -> tuple[int, int]:
        w, h = self.z["sizes"][idx]
        return int(w), int(h)


def split_index(iid: str) -> tuple[str, int]:
    """(cache split, row) for an image id; halves A/B live inside the 'train' cache."""
    return data.parse_id(iid)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)
