import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, pickle
sys.path.insert(0, ROOT + "/scripts")
import _common
import numpy as np
from lcr import cache, config, claims, data, calibrate
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX, KIND
P = config.CACHE / "pipeline"
rows = cache.load_json(P / "A_B_look4.json")
assert all(r["id"].startswith("train_") for r in rows)
maps = np.load(P / "A_B_look4.maps.npz"); print(maps.files, maps["maps"].shape)
owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
owl = cache.OwlCache("train")
q = np.load(config.CACHE / "owl" / "class_queries_single.npy").astype(np.float32)
rule = dict(min_patches=0, alpha=0.5)
image_max, xs, pooled, per = [], [], [], []
refs = {}
for ri, row in enumerate(rows):
    idx = int(row["id"].rsplit("_", 1)[1])
    prop = owl[idx]
    lg = re.proposal_logits(prop, q)
    image_max.append(lg.max(0))
    refs[row["id"]] = claims.reference_classes(data.references(row["id"]))
    for ci, c in enumerate(row["claims"]):
        att = maps["maps"][owner[(ri, ci)]].astype(np.float32)
        top, w = re.candidates(att, prop, c["kind"], rule)
        boxes = prop["boxes"].astype(np.float32)
        xs.append(dict(ri=ri, ci=ci, id=row["id"], cls=c["cls"], kind=c["kind"], ok=bool(c["in_refs"]),
                       fold=calibrate.fold_of(row["id"]), top=top.tolist(), w=w.tolist(),
                       boxes=boxes[top].tolist(), lg3=lg[top].astype(np.float32),
                       argimg=int(lg[:, CLASS_INDEX[c["cls"]]].argmax()),
                       imgbox=boxes[int(lg[:, CLASS_INDEX[c["cls"]]].argmax())].tolist(),
                       blip_alts=c["blip_alts"], nv=claims.non_visual(row["draft"], c["cls"]), draft=row["draft"]))
image_max = np.asarray(image_max)
pickle.dump(dict(xs=xs, image_max=image_max, refs=refs, ids=[r["id"] for r in rows]),
            open(ROOT + "/data/cache/review_fable_B_feat.pkl", "wb"))
print(len(xs), image_max.shape)
