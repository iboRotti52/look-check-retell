"""Cache frozen features once: BLIP patch embeddings and OWLv2 top-100 proposals.

    python scripts/02_cache_features.py --splits validation train [--what blip owl]

The test split is deliberately NOT a default: the protocol reserves it for the
final evaluation, so nothing is computed on it until the group decides to.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lcr import cache, check, config, data  # noqa: E402
from lcr.captioner import Captioner  # noqa: E402


def run_blip(split: str, bs: int):
    if cache.blip_path(split).exists():
        print(f"[blip] {split}: cached, skipping"); return
    cap = Captioner()
    n = data.num_images(split)
    tmp = cache.blip_path(split).with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, 577, 768))
    for s in tqdm(range(0, n, bs), desc=f"blip {split}"):
        imgs = [data.load_image(data.image_id(split, i)) for i in range(s, min(n, s + bs))]
        mm[s:s + len(imgs)] = cap.encode_images(imgs).float().cpu().numpy().astype(np.float16)
    mm.flush(); del mm
    tmp.rename(cache.blip_path(split))


def run_owl(split: str, bs: int, only: str | None = None):
    """`only` = A or B restricts the train split to one half (the other rows stay empty
    and can be filled later by re-running without --only after deleting the file)."""
    if cache.owl_path(split).exists():
        print(f"[owl] {split}: cached, skipping"); return
    det = check.Detector()
    n = data.num_images(split)
    todo = list(range(n))
    if only and split == "train":
        todo = [data.parse_id(i)[1] for i in data.ids(only)]
    empty = {k: np.zeros((0, *shape[1:]), np.float16) for k, shape in cache.OWL_FIELDS.items()}
    props, sizes = [empty] * n, [(0, 0)] * n
    for s in tqdm(range(0, len(todo), bs), desc=f"owl {split}"):
        rows = todo[s:s + bs]
        imgs = [data.load_image(data.image_id(split, i)) for i in rows]
        for i, p, im in zip(rows, det.proposals(imgs), imgs):
            props[i], sizes[i] = p, im.size
    cache.save_owl(split, props, sizes)
    np.save(config.CACHE / "owl" / "class_queries.npy", det.text_embeddings(check.CLASSES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "train"])
    ap.add_argument("--what", nargs="+", default=["blip", "owl"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--only", default=None, choices=["A", "B"], help="OWLv2 on one half of train only")
    a = ap.parse_args()
    t0 = time.time()
    for what in a.what:
        for split in a.splits:
            if what == "blip":
                run_blip(split, 4 * a.bs)
            else:
                run_owl(split, a.bs, a.only)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    print(f"done in {(time.time() - t0) / 60:.1f} min")
