"""Look E3 extraction (docs/LOOK_E3_PROTOKOL.md): per-head sink maps and candidate maps for B or validation.

    python scripts/25_look_e3_extract.py --split train        # half B (selection)
    python scripts/25_look_e3_extract.py --split validation   # report split
Writes data/cache/look_e3/{stem}_heads.f16.npy (N,12,12,576), {stem}_maps.npz, {stem}_info.json. Resumable.
"""
import argparse
import time

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, data
from lcr.look_e3 import MAPS, E3Decoder

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["train", "validation"], required=True)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
stem = {"train": "A_B_look2", "validation": "A_validation_look2"}[a.split]
rows = cache.load_json(config.CACHE / "pipeline" / f"{stem}.json")
assert all(data.parse_id(r["id"])[0] == a.split for r in rows)
folder = config.CACHE / "look_e3"; folder.mkdir(exist_ok=True)
n = sum(len(r["claims"]) for r in rows)
hpath, mpath, ipath = folder / f"{stem}_heads.f16.npy", folder / f"{stem}_maps.npz", folder / f"{stem}_info.json"
state = cache.load_json(ipath) if ipath.exists() else dict(done_rows=0, info=[], owner=[])
heads = np.lib.format.open_memmap(hpath, mode="r+" if hpath.exists() else "w+", dtype=np.float16,
                                  shape=(n, 12, 12, 576))
maps = {k: list(v) for k, v in np.load(mpath).items()} if mpath.exists() and state["done_rows"] else {k: [] for k in MAPS}
import torch
dec = E3Decoder(); torch.set_num_threads(a.threads)
feats = np.load(config.CACHE / "blip" / f"{a.split}.f16.npy", mmap_mode="r")
k = len(state["owner"]); t0 = time.monotonic()


def save():
    heads.flush()
    np.savez_compressed(mpath, **{m: np.asarray(v, np.float16) for m, v in maps.items()})
    cache.save_json(state, ipath)


for ri in range(state["done_rows"], len(rows)):
    r = rows[ri]
    h, mm, info = dec.run_row(r, feats[data.parse_id(r["id"])[1]])
    for ci, (m, inf) in enumerate(zip(mm, info)):
        heads[k] = h[ci]; k += 1
        for name in MAPS:
            maps[name].append(m[name])
        state["owner"].append([ri, ci]); state["info"].append(inf)
    state["done_rows"] = ri + 1
    if (ri + 1) % 200 == 0 or ri + 1 == len(rows):
        save()
        print(f"{a.split} {ri+1}/{len(rows)} rows, {(time.monotonic()-t0)/60:.1f} min", flush=True)
assert k == n
