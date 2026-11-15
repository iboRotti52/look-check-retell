"""Run Draft -> Look -> Check with the frozen captioner and shared calibrated checker.

    python scripts/04_draft_look_check.py --split validation --decoder A
    python scripts/04_draft_look_check.py --split B --decoder A        # Retell training data

--decoder zero uses the public checkpoint without Flickr8k fine-tuning.
Default output: <decoder>_<split>_lcr (sink, decoder A, frozen bundle).
Use --region-rule v1|v2|v3|v4 explicitly for historical methods.
"""
import argparse

import numpy as np
from tqdm import tqdm

import _common  # noqa: F401
from lcr import cache, config, data, pipeline
from lcr.captioner import Captioner

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=["A", "B", "validation"])
ap.add_argument("--decoder", default="A", choices=["A", "B", "zero"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--attn", default="sink", choices=["raw", "sink"], help="sink = sink-corrected attention maps")
ap.add_argument("--region-rule", default=config.LOOK_VERSION, choices=["v1", "v2", "v3", "v4", "lcr"])
ap.add_argument("--overwrite", action="store_true", help="explicitly replace an existing output")
a = ap.parse_args()
assert a.split != "test", "the test split is reserved for the final evaluation"
assert a.split != a.decoder, "drafts must come from images held out from the captioner's fine-tuning"
if a.limit < 0:
    ap.error("--limit must be nonnegative")
if a.region_rule in ("v4", "lcr") and (a.attn != "sink" or a.decoder != "A"):
    ap.error("Frozen attention requires --attn sink and --decoder A")
stem = pipeline.output_stem(a.decoder, a.split, a.attn == "sink", a.region_rule)
if a.limit:
    stem += f"_limit{a.limit}"
out = config.CACHE / "pipeline" / f"{stem}.json"
if not a.overwrite and (out.exists() or out.with_suffix(".maps.npz").exists()):
    ap.error(f"Output already exists: {out}. Use --overwrite to replace it.")

if a.region_rule == "lcr":
    from lcr.region_evidence import RegionChecker
    RegionChecker.load()  # fail before loading a captioner if calibration is unavailable
weights = None if a.decoder == "zero" else str(config.CACHE / "ckpt" / f"decoder_{a.decoder}.pt")
cap = Captioner(decoder_weights=weights)
ids = data.ids(a.split)[: a.limit or None]
rows, maps, owner = [], [], []
for row, m in pipeline.run(cap, ids, progress=tqdm, sink=a.attn == "sink", region_rule=a.region_rule):
    rows.append(row)
    for j, x in enumerate(m):
        maps.append(x); owner.append((len(rows) - 1, j))
cache.save_json(rows, out)
np.savez_compressed(out.with_suffix(".maps.npz"), maps=np.stack(maps) if maps else np.zeros((0, 24, 24)),
                    owner=np.array(owner, dtype=np.int32))
n = sum(len(r["claims"]) for r in rows)
print(f"{len(rows)} images, {n} claims ({n / len(rows):.2f} per image) -> {out}")
