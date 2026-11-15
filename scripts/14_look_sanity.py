"""Sanity gates for a Look map file (see docs/LITERATUR_TARAMASI.md, operation 5a).

    python scripts/14_look_sanity.py A_validation_sink

orientation  pointing with the map transposed / flipped must drop (else the grid->image mapping is wrong)
specificity  pointing on claim c's human boxes with the map of ANOTHER claim (different class, same image)
             must be well below the claim's own map (else the map only shows salient objects, not the word)
"""
import sys
import numpy as np
import _common  # noqa
from lcr import cache, config, grounding
pipe = sys.argv[1]
assert "test" not in pipe, "the test split is not used during development"
P = config.CACHE / "pipeline"
rows = cache.load_json(P / f"{pipe}.json"); z = np.load(P / f"{pipe}.maps.npz")
map_values = z["maps"]
midx = {(int(r), int(c)): i for i, (r, c) in enumerate(z["owner"])}
g = config.BLIP_GRID
def point(att, gt):
    i, j = np.unravel_index(int(np.argmax(att)), att.shape)
    return grounding.point_in_boxes((i + .5) / g, (j + .5) / g, gt)
def energy(att, gt):
    a = att / max(att.sum(), 1e-12); return float((a * grounding.coverage_mask(gt)).sum())
own, tr, flip, swap, swap_own, e_own, e_swap, area = [], [], [], [], [], [], [], []
for ri, r in enumerate(rows):
    gtd = grounding.class_boxes(r["id"])
    cl = [(ci, c) for ci, c in enumerate(r["claims"]) if gtd.get(c["cls"]) is not None and len(gtd[c["cls"]])]
    for ci, c in cl:
        gt = gtd[c["cls"]]; att = map_values[midx[(ri, ci)]].astype(np.float32)
        own.append(point(att, gt)); tr.append(point(att.T, gt)); flip.append(point(att[::-1], gt))
        # swap: another claim of a different class in the same image whose GT does not overlap this point
        others = [(cj, d) for cj, d in cl if d["cls"] != c["cls"]]
        if others:
            cj, d = others[0]
            att2 = map_values[midx[(ri, cj)]].astype(np.float32)
            swap.append(point(att2, gt)); swap_own.append(point(att, gt))
            e_swap.append(energy(att2, gt)); e_own.append(energy(att, gt)); area.append(float(grounding.coverage_mask(gt).mean()))
m = lambda x: 100 * float(np.mean(x))
print(f"{pipe}: own {m(own):.1f}  transposed {m(tr):.1f}  vflip {m(flip):.1f}  (n={len(own)})")
print(f"  swap test (n={len(swap)} claims with another-class claim in image): pointing own {m(swap_own):.1f} vs other-claim map {m(swap):.1f}; "
      f"energy own {m(e_own):.1f} vs other {m(e_swap):.1f}; chance {m(area):.1f}")
cache.save_json(dict(pipe=pipe, metric_version="exact_box_union_v2", n=len(own), pointing=m(own), transposed=m(tr), vflip=m(flip),
                    swap_n=len(swap), swap_own=m(swap_own), swap_other=m(swap),
                    swap_energy_own=m(e_own), swap_energy_other=m(e_swap), swap_chance=m(area)),
                config.RESULTS / f"look_sanity_{pipe}.json")
