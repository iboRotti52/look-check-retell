"""Read-only audit of frozen Look v2 on half B; no parameter search or test access.

Run: LCR_DEVICE=cpu .venv/bin/python scripts/review_look_v2.py
Writes descriptive diagnostics and a deterministic, stratified gallery of person
failures. Controls isolate the map from the shared OWLv2 proposal/objectness prior.
"""
from collections import defaultdict

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, data, grounding, look


def summary(items):
    if not items:
        return {"n": 0}
    keys = [k for k in items[0] if k.endswith("_iou")]
    return dict(n=len(items), **{
        k.removesuffix("_iou"): {
            "accuracy_pct": 100 * float(np.mean([r[k] >= .5 for r in items])),
            "miou": float(np.mean([r[k] for r in items])),
        } for k in keys})


def main():
    stem = "A_B_look2"
    p = config.CACHE / "pipeline"
    rows = cache.load_json(p / f"{stem}.json")
    allowed = set(data.ids("B"))
    assert all(r["id"] in allowed for r in rows), "Only half B may be audited"
    z = np.load(p / f"{stem}.maps.npz")
    midx = {(int(r), int(c)): i for i, (r, c) in enumerate(z["owner"])}
    owl = cache.OwlCache("train")
    records, failures = [], []
    metadata = defaultdict(int)
    uniform = np.full((config.BLIP_GRID, config.BLIP_GRID), 1 / config.BLIP_GRID**2)
    for ri, row in enumerate(rows):
        prop = owl[data.parse_id(row["id"])[1]]
        boxes = prop["boxes"].astype(np.float32)
        obj = prop["objectness"].astype(np.float32)
        ok = look.size_ok(boxes)
        gt_dict = grounding.class_boxes(row["id"])
        for ci, c in enumerate(row["claims"]):
            gt = gt_dict.get(c["cls"])
            if gt is None or not len(gt):
                continue
            att = z["maps"][midx[ri, ci]].astype(np.float32)
            att /= max(att.sum(), 1e-12)
            reg, _ = look.select_region(att, boxes, obj)
            obj_reg = int(np.argmax(np.where(ok, obj, -np.inf))) if ok.any() else None
            uni_reg, _ = look.select_region(uniform, boxes, obj)
            no_obj, _ = look.select_region(att, boxes, obj, gamma=0)
            ious = np.max([look.iou(b, boxes) for b in gt], axis=0)
            val = lambda k: float(ious[k]) if k is not None and k >= 0 else 0.0
            r = dict(id=row["id"], kind=c["kind"], gt_boxes=len(gt),
                     own_iou=val(c["rec_att"]["region"]),
                     recomputed_iou=val(reg), detector_iou=val(c["rec_det"]["region"]),
                     objectness_only_iou=val(obj_reg), uniform_v2_iou=val(uni_reg),
                     attention_no_objectness_iou=val(no_obj),
                     oracle_iou=float(ious[ok].max()) if ok.any() else 0.0)
            yy, xx = np.unravel_index(att.argmax(), att.shape)
            r["point_correct"] = grounding.point_in_boxes((yy + .5) / 24, (xx + .5) / 24, gt)
            others = [j for j, d in enumerate(row["claims"]) if d["cls"] != c["cls"]]
            if others:
                other = z["maps"][midx[ri, others[0]]].astype(np.float32)
                swapped, _ = look.select_region(other, boxes, obj)
                r["swapped_iou"] = val(swapped)
            metadata["recomputed_region_differs"] += reg != c["rec_att"]["region"]
            current_box = np.asarray(c["rec_att"]["box"], np.float32)
            dbox = np.asarray(c["rec_det"]["box"], np.float32)
            if len(current_box) and len(dbox):
                actual_agree = float(look.iou(current_box, dbox[None])[0])
                metadata["stale_iou_att_det_gt_1e_5"] += abs(actual_agree - c["iou_att_det"]) > 1e-5
            k4 = c.get("regions_by_k", {}).get("4")
            metadata["regions_by_k4_differs_from_rec_att"] += k4 is not None and k4 != c["rec_att"]["region"]
            records.append(r)
            if c["kind"] == "person" and r["own_iou"] < .5 and len(current_box):
                near = gt[int(np.argmax(look.iou(current_box, gt)))]
                ar = float(look.box_area(current_box[None])[0] / max(look.box_area(near[None])[0], 1e-12))
                inter = np.maximum(0, np.minimum(current_box[2:], near[2:]) - np.maximum(current_box[:2], near[:2])).prod()
                pred_cover = float(inter / max(look.box_area(current_box[None])[0], 1e-12))
                gt_cover = float(inter / max(look.box_area(near[None])[0], 1e-12))
                # Descriptive geometry labels, not human error annotations.
                label = ("no_overlap" if inter == 0 else "small_part" if ar < .5 and pred_cover >= .8
                         else "large_container" if ar > 2 and gt_cover >= .8 else "other_overlap")
                r.update(error_geometry=label, pred_gt_area_ratio=ar, pred_coverage=pred_cover, gt_coverage=gt_cover)
                failures.append((r, ri, ci))
    def matched_summary(rs):
        # Swap exists only for a subset; keep own and swapped on the same items.
        return {"n": len(rs), "own_accuracy_pct": 100 * float(np.mean([r["own_iou"] >= .5 for r in rs])),
                "swapped_accuracy_pct": 100 * float(np.mean([r["swapped_iou"] >= .5 for r in rs]))} if rs else {"n": 0}
    base_records = [{k: v for k, v in r.items() if k != "swapped_iou"} for r in records]
    person = [r for r in base_records if r["kind"] == "person"]
    out = dict(stem=stem, split="B", purpose="descriptive audit; no configuration selected",
               metric="class-level best IoU against any matching F30kE box, threshold .5",
               all=summary(base_records),
               by_kind={k: summary([r for r in base_records if r["kind"] == k]) for k in ("person", "thing", "stuff")},
               person_by_gt_box_count={k: summary([r for r in person if (min(r["gt_boxes"], 3) == n)])
                                       for k, n in (("1", 1), ("2", 2), ("3+", 3))},
               matched_map_swap=matched_summary([r for r in records if "swapped_iou" in r]),
               person_failures={"n": len(failures),
                                "with_correct_point": sum(r["point_correct"] for r, _, _ in failures),
                                "geometry_counts": {k: sum(r["error_geometry"] == k for r, _, _ in failures)
                                                    for k in ("no_overlap", "small_part", "large_container", "other_overlap")},
                                "area_ratio_median": float(np.median([r["pred_gt_area_ratio"] for r, _, _ in failures]))},
               cache_metadata=dict(metadata))
    path = config.RESULTS / "look_v2_review_B.json"
    cache.save_json(out, path)
    print(path)
    print(out)

    # Up to five random failures per geometry bucket, deliberately stratified.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(config.SEED)
    picks = []
    for label in ("small_part", "large_container", "other_overlap", "no_overlap"):
        pool = [x for x in failures if x[0]["error_geometry"] == label]
        picks.extend(pool[i] for i in rng.choice(len(pool), size=min(5, len(pool)), replace=False))
    fig, axes = plt.subplots(max(1, int(np.ceil(len(picks) / 4))), 4, figsize=(16, 4 * max(1, int(np.ceil(len(picks) / 4)))))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (r, ri, ci) in zip(np.ravel(axes), picks):
        row = rows[ri]; c = row["claims"][ci]
        im = data.load_image(row["id"]); w, h = im.size
        ax.imshow(im)
        att = z["maps"][midx[ri, ci]].astype(np.float32)
        ax.imshow(att, extent=(0, w, h, 0), cmap="magma", alpha=.35)
        for b in grounding.class_boxes(row["id"])[c["cls"]]:
            ax.add_patch(plt.Rectangle((b[0]*w, b[1]*h), (b[2]-b[0])*w, (b[3]-b[1])*h, fill=False, ec="lime", lw=1.5))
        for key, color in (("rec_att", "red"), ("rec_det", "cyan")):
            b = c[key]["box"]
            if b:
                ax.add_patch(plt.Rectangle((b[0]*w, b[1]*h), (b[2]-b[0])*w, (b[3]-b[1])*h, fill=False, ec=color, lw=1.5, ls="--"))
        ax.set_title(f'{row["id"]}: {r["error_geometry"]}\nLook IoU={r["own_iou"]:.2f}, box/GT area={r["pred_gt_area_ratio"]:.2f}', fontsize=10)
    fig.suptitle("Look v2 person failures on B (stratified): GT green, Look red, detector cyan", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(config.RESULTS / "look_v2_person_review_B.png", dpi=120)
    plt.close(fig)
    cache.save_json([r for r, _, _ in picks], config.RESULTS / "look_v2_person_review_examples_B.json")


if __name__ == "__main__":
    main()
