"""Look coverage of human boxes on B_fit, with the small-object count corrected (2026-11-19). B only.

scripts/37 counted a claim as "small object found" when a top-3 candidate hit ANY human box of the class,
also when the class had a large box too and the hit was on the large one. Here a small-object hit has to land
on a small box (shorter side < 2 BLIP patches; lcr/grounding.coverage_hits). Same images (B_fit), same maps
(A_B_look4), same two Look rules as script 37's comparison (with and without the 2-patch minimum side), same
candidate ranking (region_evidence.candidates). The old definition is reported next to the new one.
Does not refit or rewrite anything; scripts/38 reads this file for the report's small-object numbers.

    python scripts/49_look_small_object.py
"""
import numpy as np

import _common  # noqa: F401
from lcr import cache, config, grounding
from lcr import region_evidence as re

P = config.CACHE / "pipeline"
RULES = dict(with_filter=dict(min_patches=2, alpha=.5), without_filter=dict(min_patches=0, alpha=.5))
KEYS = ("n", "top1", "top3", "small_n", "small_top3", "small_top3_any_box")


def main():
    rows = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows), "B only; refuses validation/test"
    maps = np.load(P / "A_B_look4.maps.npz")
    owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    owl = cache.OwlCache("train")
    counts = {name: dict.fromkeys(KEYS, 0) for name in RULES}
    fit = 0
    for ri, row in enumerate(rows):
        if re.partition(row["id"]) != "fit":
            continue
        fit += 1
        gtd = grounding.class_boxes(row["id"])
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        for ci, c in enumerate(row["claims"]):
            gt = gtd.get(c["cls"])
            if c["kind"] == "person" or gt is None or not len(gt):
                continue
            att = maps["maps"][owner[(ri, ci)]]
            for name, rule in RULES.items():
                top, _ = re.candidates(att, prop, c["kind"], rule)
                h = grounding.coverage_hits(prop["boxes"][top].astype(np.float32), gt)
                s = counts[name]
                s["n"] += 1; s["top1"] += h["top1"]; s["top3"] += h["top3"]
                s["small_n"] += h["has_small"]; s["small_top3"] += h["small_top3"]
                s["small_top3_any_box"] += h["small_top3_any_box"]
    pct = lambda a, b: round(100 * a / b, 2) if b else None
    out = dict(images=fit, small_side=grounding.SMALL_SIDE, rules=RULES, counts=counts,
               rates={name: dict(top3=pct(s["top3"], s["n"]), small_top3=pct(s["small_top3"], s["small_n"]),
                                 small_top3_any_box=pct(s["small_top3_any_box"], s["small_n"]))
                      for name, s in counts.items()})
    cache.save_json(out, config.RESULTS / "look_small_object_B.json")
    r = out["rates"]
    lines = ["# Look kapsaması: küçük nesne sayımı düzeltildi (B_fit)", "",
             "Üreten: `scripts/49_look_small_object.py`. Küçük kutu: kısa kenar < 2 BLIP yaması. Yeni tanım: isabet "
             "küçük kutuya gelmeli; eski tanım (script 37'nin ilk sayımı) sınıfın herhangi bir kutusunu sayıyordu.", "",
             f"Görüntü: {fit}.", "", "| Look kuralı | n | top-3 | küçük n | küçük top-3 (yeni) | küçük top-3 (eski) |",
             "|---|---|---|---|---|---|"]
    for name, s in counts.items():
        lines.append(f"| {name} | {s['n']} | {r[name]['top3']} | {s['small_n']} | {r[name]['small_top3']} | "
                     f"{r[name]['small_top3_any_box']} |")
    (config.ROOT / "docs/generated/LOOK_SMALL_OBJECT_B.md").write_text("\n".join(lines) + "\n")
    print(out["rates"])


if __name__ == "__main__":
    main()
