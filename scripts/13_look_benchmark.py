"""Look benchmark against human boxes (Flickr30k Entities). See docs/LOOK_V2_PLAN.md section 4.

    python scripts/13_look_benchmark.py --pipe A_validation            # 1st-round raw maps
    python scripts/13_look_benchmark.py --pipe A_validation_sink       # sink-corrected maps

For every draft claim whose class has F30kE boxes in that image:
  map level    pointing game (argmax patch centre inside a GT box) and energy (map mass on GT boxes);
               chance = share of the image the GT boxes cover
  region level the chosen OWLv2 box has IoU >= 0.5 with some GT box, for
               attention (rec_att), detector (rec_det, B3), a random proposal, and the best proposal (oracle:
               do the OWLv2 proposals contain the object at all?)
Never run on the test split during development (the script refuses).
"""
import argparse
from collections import defaultdict

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, data, grounding, look

ap = argparse.ArgumentParser()
ap.add_argument("--pipe", default="A_validation", help="pipeline file stem under data/cache/pipeline")
ap.add_argument("--examples", type=int, default=0, help="write an overlay figure with this many claims")
a = ap.parse_args()
assert "test" not in a.pipe, "the test split is not used during development"
P = config.CACHE / "pipeline"
rows = cache.load_json(P / f"{a.pipe}.json")
if any(r.get("map_only") for r in rows):
    raise ValueError("This artifact has maps only; use the frozen look3 records for region metrics")
z = np.load(P / f"{a.pipe}.maps.npz")
midx = {(int(r), int(c)): i for i, (r, c) in enumerate(z["owner"])}
split = data.parse_id(rows[0]["id"])[0]
owl = cache.OwlCache(split)
rng = np.random.default_rng(config.SEED)
g = config.BLIP_GRID


def best_iou(box, gt):
    return float(look.iou(np.asarray(box, np.float32), gt).max()) if len(box) else 0.0


recs = []
n_all = n_gt = 0
for ri, r in enumerate(rows):
    gtd = grounding.class_boxes(r["id"])
    boxes = owl[data.parse_id(r["id"])[1]]["boxes"].astype(np.float32)
    ok = look.size_ok(boxes)
    for ci, c in enumerate(r["claims"]):
        n_all += 1
        gt = gtd.get(c["cls"])
        if gt is None or not len(gt):
            continue
        n_gt += 1
        att = z["maps"][midx[(ri, ci)]].astype(np.float32)
        att = att / max(att.sum(), 1e-12)
        mask = grounding.coverage_mask(gt)
        i, j = np.unravel_index(int(np.argmax(att)), att.shape)
        rand = boxes[rng.choice(np.flatnonzero(ok))] if ok.any() else []
        recs.append(dict(
            kind=c["kind"], in_refs=c["in_refs"],
            point=grounding.point_in_boxes((i + 0.5) / g, (j + 0.5) / g, gt),
            energy=float((att * mask).sum()), area=float(mask.mean()),
            att_iou=best_iou(c["rec_att"]["box"], gt), det_iou=best_iou(c["rec_det"]["box"], gt),
            rand_iou=best_iou(rand, gt) if len(rand) else 0.0,
            oracle_iou=float(max(look.iou(b, boxes[ok]).max() for b in gt)) if ok.any() else 0.0,
        ))


def summarise(rs):
    if not rs:
        return {}
    m = lambda k: float(np.mean([x[k] for x in rs]))
    acc = lambda k: 100 * float(np.mean([x[k] >= 0.5 for x in rs]))
    return dict(n=len(rs), pointing=100 * m("point"), pointing_chance=100 * m("area"), energy=100 * m("energy"),
                energy_chance=100 * m("area"), region_att=acc("att_iou"), region_det=acc("det_iou"),
                region_random=acc("rand_iou"), region_oracle=acc("oracle_iou"))


by_kind = defaultdict(list)
for x in recs:
    by_kind[x["kind"]].append(x)
out = dict(pipe=a.pipe, metric_version="exact_box_union_v2", claims=n_all, claims_with_gt=n_gt, all=summarise(recs),
           **{k: summarise(v) for k, v in sorted(by_kind.items())})
cache.save_json(out, config.RESULTS / f"look_benchmark_{a.pipe}.json")
print(f"{a.pipe}: {n_gt}/{n_all} claims have F30kE boxes ({100 * n_gt / n_all:.1f}%)")
hdr = f"{'':8s}{'n':>6s} {'point':>7s} {'chance':>7s} {'energy':>7s} {'reg.att':>8s} {'reg.det':>8s} {'random':>7s} {'oracle':>7s}"
print(hdr)
for k in ["all", *sorted(by_kind)]:
    s = out[k]
    print(f"{k:8s}{s['n']:6d} {s['pointing']:7.1f} {s['pointing_chance']:7.1f} {s['energy']:7.1f} "
          f"{s['region_att']:8.1f} {s['region_det']:8.1f} {s['region_random']:7.1f} {s['region_oracle']:7.1f}")

if a.examples:
    from _common import plt, savefig
    pool = [(ri, ci) for ri, r in enumerate(rows) for ci, c in enumerate(r["claims"])
            if c["cls"] in grounding.class_boxes(r["id"])]
    pick = [pool[i] for i in rng.choice(len(pool), size=min(a.examples, len(pool)), replace=False)]
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(pick) / cols)), cols, figsize=(7, 1.9 * np.ceil(len(pick) / cols)))
    for ax, (ri, ci) in zip(np.ravel(axes), pick):
        r = rows[ri]; c = r["claims"][ci]
        img = data.load_image(r["id"]); W, H = img.size
        ax.imshow(img); ax.imshow(np.kron(z["maps"][midx[(ri, ci)]].astype(np.float32), np.ones((1, 1))),
                                  extent=(0, W, H, 0), alpha=0.45, cmap="jet")
        for b in grounding.class_boxes(r["id"])[c["cls"]]:
            ax.add_patch(plt.Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H,
                                       fill=False, ec="lime", lw=1.2))
        if c["rec_att"]["box"]:
            b = c["rec_att"]["box"]
            ax.add_patch(plt.Rectangle((b[0] * W, b[1] * H), (b[2] - b[0]) * W, (b[3] - b[1]) * H,
                                       fill=False, ec="red", lw=1, ls="--"))
        ax.set_title(f"{c['word']}  ({c['cls']})", fontsize=6); ax.axis("off")
    for ax in np.ravel(axes)[len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, f"look_benchmark_examples_{a.pipe}")
