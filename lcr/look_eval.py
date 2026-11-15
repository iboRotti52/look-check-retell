"""Shared development-only measurements for the bounded E1/E2 Look experiments."""
from collections import defaultdict

import numpy as np

from . import cache, config, data, grounding, look

KINDS = ("person", "thing", "stuff")


def normalise(att):
    a = np.asarray(att, np.float32)
    total = float(a.sum())
    return a / total if total > 0 else np.full_like(a, 1 / a.size)


def transformed(att, sigma=0):
    return look.smooth_map(att, sigma)


def load_bank(stem, selected_ids=None):
    if "test" in stem:
        raise ValueError("No test artifacts during development")
    p = config.CACHE / "pipeline"
    rows = cache.load_json(p / f"{stem}.json")
    split = data.parse_id(rows[0]["id"])[0]
    if split not in ("train", "validation") or any(data.parse_id(r["id"])[0] != split for r in rows):
        raise ValueError("Only a single development split is allowed")
    maps = np.load(p / f"{stem}.maps.npz")
    midx = {tuple(map(int, owner)): i for i, owner in enumerate(maps["owner"])}
    map_values = maps["maps"]
    z = np.load(config.CACHE / "owl" / f"{split}.npz")
    boxes_all, objectness_all, counts = z["boxes"], z["objectness"], z["count"]
    entries = []
    for ri, r in enumerate(rows):
        if selected_ids is not None and r["id"] not in selected_ids:
            continue
        idx = data.parse_id(r["id"])[1]; n = int(counts[idx])
        boxes = boxes_all[idx, :n].astype(np.float32)
        obj = objectness_all[idx, :n].astype(np.float32)
        gt_dict = grounding.class_boxes(r["id"])
        for ci, claim in enumerate(r["claims"]):
            gt = gt_dict.get(claim["cls"])
            att = normalise(map_values[midx[ri, ci]])
            entries.append(dict(iid=r["id"], ri=ri, ci=ci, kind=claim["kind"], cls=claim["cls"],
                att=att, boxes=boxes, obj=obj, area=look.box_area(boxes), valid=look.size_ok(boxes), gt=gt,
                gtiou=np.max([look.iou(b, boxes) for b in gt], axis=0) if gt is not None and len(gt) else None,
                coverage=grounding.coverage_mask(gt) if gt is not None and len(gt) else None))
    return rows, entries


def pick(entry, att, rule):
    return look.select_region(att, entry["boxes"], entry["obj"], alpha=rule["alpha"], gamma=rule["gamma"])[0]


def measure(entries, maps=None, rules=None):
    """Pointing/energy/class-level region IoU. Preserve iid for paired bootstrap."""
    records = []
    for j, e in enumerate(entries):
        att = normalise(e["att"] if maps is None else maps[j])
        rule = (rules or {}).get(e["kind"], dict(alpha=.5, gamma=.5, sigma=0))
        k = pick(e, att, rule)
        if e["gtiou"] is None:
            continue
        y, x = np.unravel_index(att.argmax(), att.shape)
        records.append(dict(iid=e["iid"], kind=e["kind"], hit=bool(k is not None and e["gtiou"][k] >= .5),
            iou=float(e["gtiou"][k]) if k is not None else 0.,
            point=grounding.point_in_boxes((y+.5)/24, (x+.5)/24, e["gt"]),
            energy=float((att * e["coverage"]).sum()), chance=float(e["coverage"].mean())))
    return summarise(records), records


def summarise(records):
    def one(rs):
        if not rs:
            return dict(n=0)
        return dict(n=len(rs), region=100*float(np.mean([r["hit"] for r in rs])),
                    miou=float(np.mean([r["iou"] for r in rs])),
                    pointing=100*float(np.mean([r["point"] for r in rs])),
                    energy=100*float(np.mean([r["energy"] for r in rs])),
                    chance=100*float(np.mean([r["chance"] for r in rs])))
    return dict(all=one(records), **{k: one([r for r in records if r["kind"] == k]) for k in KINDS})


def paired_bootstrap(before, after, image_ids, n_boot=2000):
    assert [(r["iid"], r["kind"]) for r in before] == [(r["iid"], r["kind"]) for r in after]
    ids = list(image_ids); lookup = {iid: i for i, iid in enumerate(ids)}
    out = {}
    for kind in ("all", *KINDS):
        counts = np.zeros(len(ids)); delta = np.zeros(len(ids))
        for a, b in zip(before, after):
            if kind == "all" or a["kind"] == kind:
                j = lookup[a["iid"]]; counts[j] += 1; delta[j] += int(b["hit"]) - int(a["hit"])
        if not counts.sum():
            out[kind] = dict(n=0); continue
        rng = np.random.default_rng(config.SEED)
        draws = []
        for _ in range(n_boot):
            sample = rng.integers(0, len(ids), len(ids)); den = counts[sample].sum()
            if den:
                draws.append(100 * delta[sample].sum() / den)
        out[kind] = dict(delta_pp=100*float(delta.sum()/counts.sum()),
                         ci95_pp=np.quantile(draws, [.025, .975]).tolist(), n_boot=len(draws), unit="image")
    return out
