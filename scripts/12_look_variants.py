"""Look development: alternative ways to turn the saved attention maps into a region, evaluated offline.

    python scripts/12_look_variants.py --split validation --attn sink            # evaluate only
    python scripts/12_look_variants.py --split B --attn sink --write top3        # also write a pipeline file

No GPU is needed: the 24x24 maps (.maps.npz) and the OWLv2 proposal cache are reused.
Variants (all keep attention as the localiser; only the detector score inside the chosen region is read):
  argmax : the proposal with the highest excess attention mass (round-1 design)
  top{k} : among the k proposals with the highest excess mass, the one that best matches the claim.
           This tolerates attention that lands on the right object but on a neighbouring box.
  det    : B3, the detector's best box for the claim anywhere in the image (reference, not attention-based)
  image  : the claim's best detector score over the whole image (no region at all; reference)
Quality is threshold-free: AUROC of the region's claim score s_claim for separating thing/stuff claims that
a reference mentions from those it does not (reference PROXY, not human labels).
--write <variant> writes <decoder>_<split>_<attn>_<variant>.json with rec_att replaced, so that scripts
06 and 11 can use it via --attn <attn>_<variant>.
"""
import argparse

import numpy as np

import _common  # noqa: F401
from lcr import cache, check, config, look, metrics, pipeline
from lcr.vocab import CLASS_INDEX, KIND

ap = argparse.ArgumentParser()
ap.add_argument("--decoder", default="A")
ap.add_argument("--split", default="validation")
ap.add_argument("--attn", default="sink", choices=["raw", "sink"])
ap.add_argument("--ks", default="1,2,3,5,10")
ap.add_argument("--write", default="", help="variant to write as a pipeline file, e.g. top3")
a = ap.parse_args()
sfx = "" if a.attn == "raw" else "_sink"
base = config.CACHE / "pipeline" / f"{a.decoder}_{a.split}{sfx}"
rows = cache.load_json(base.with_suffix(".json"))
z = np.load(base.with_suffix(".maps.npz"))
idx = {(int(r), int(c)): i for i, (r, c) in enumerate(z["owner"])}
owl = cache.OwlCache("validation" if a.split == "validation" else "train")
q = pipeline.class_queries()
ks = [int(k) for k in a.ks.split(",")]

s_claim = {f"top{k}": [] for k in ks}
s_claim.update(det=[], image=[])
label = []
new_rows = []
for ri, r in enumerate(rows):
    from lcr import data
    prop = owl[data.parse_id(r["id"])[1]]
    boxes = prop["boxes"].astype(np.float32)
    scores = check.score(prop, q)
    r2 = dict(r); r2["claims"] = []
    for ci, c in enumerate(r["claims"]):
        att = z["maps"][idx[(ri, ci)]].astype(np.float32)
        att = att / max(att.sum(), 1e-12)
        ex = np.where(look.size_ok(boxes), look.excess_mass(att, boxes), -np.inf)
        order = np.argsort(-ex)
        j = CLASS_INDEX[c["cls"]]
        chosen = {}
        for k in ks:
            cand = [i for i in order[:k] if np.isfinite(ex[i])]
            chosen[k] = max(cand, key=lambda i: scores[i, j]) if cand else None
        if KIND[c["cls"]] != "person":
            for k in ks:
                s_claim[f"top{k}"].append(float(scores[chosen[k], j]) if chosen[k] is not None else 0.0)
            s_claim["det"].append(c["rec_det"]["s_claim"])
            s_claim["image"].append(float(scores[:, j].max()) if len(scores) else 0.0)
            label.append(bool(c["in_refs"]))
        if a.write:
            k = int(a.write.replace("top", ""))
            reg = chosen[k]
            rec = pipeline.region_record(c["cls"], reg, float(ex[reg]) if reg is not None else 0.0, prop, scores, c["blip_alts"])
            rec["by"] = "attention"
            c2 = dict(c); c2["rec_att"] = rec
            r2["claims"].append(c2)
    new_rows.append(r2)

label = np.array(label)
res = {}
for name, v in s_claim.items():
    v = np.array(v)
    res[name] = dict(auroc=float(metrics.auroc(v[label], v[~label])), mean_pos=float(v[label].mean()),
                     mean_neg=float(v[~label].mean()))
res["argmax"] = res.pop("top1")
out = dict(split=a.split, attn=a.attn, n=int(len(label)), unsupported_rate=float(1 - label.mean()), variants=res)
cache.save_json(out, config.RESULTS / f"look_variants_{a.split}{sfx}.json")
print(f"{a.split} {a.attn}: {len(label)} thing/stuff claims")
for name, v in res.items():
    print(f"  {name:7s} AUROC {v['auroc']:.3f}   mean s_claim ref-supported {v['mean_pos']:.3f} / unsupported {v['mean_neg']:.3f}")
if a.write:
    dst = config.CACHE / "pipeline" / f"{a.decoder}_{a.split}{sfx or '_raw'}_{a.write}.json"
    cache.save_json(new_rows, dst)
    print("->", dst)
