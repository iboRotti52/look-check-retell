"""Look L1 (docs/LOOK_L1_PROTOKOL.md): referent signals from the cached Look E3 signed heads, B only.

Reads data/cache/look_e3/A_B_look2_heads.f16.npy (per-claim signed sink maps at the step that predicts the
claim word, all 144 heads) with the rows they were extracted from (A_B_look2.json, ownership checked as in
script 29). Each claim's map is rebuilt exactly as the live path does (look_v4.selected_map on the frozen
heads, then look.smooth_map with the kind's sigma), and its candidates are ranked with
region_evidence.candidates and the S4 bundle's own Look rule -- the same single ranking script 47 uses, so
top-1 is always the first of the top-3. Nothing here changes which box is chosen.

Label: reference-supported vs reference-unsupported thing/stuff claims. That split is an AUTOMATIC PROXY
(not a hallucination label): a claim can be correct without being in any of the five references (docs/LCR.md).

Size control (Revizyon 2): the size covariate is the area of Look's own top-1 box, which every claim with a
candidate has. Flickr30k Entities boxes cannot be used for this: they are attached to reference phrases, so
a reference-unsupported claim almost never has one, and a GT-box split leaves almost no negatives (the
first B run had 1 and 0). The GT coverage is still reported, to make that visible. The top-1 area is also
scored as a signal of its own: a referent signal that does no better than box area is only tracking size.
Cells with fewer than --min-per-class positives or negatives get no AUROC.

    python scripts/46_look_referent.py
"""
import argparse

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, data, grounding, look, look_referent, look_v4, metrics
from lcr import region_evidence as re_

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train", choices=["train"], help="L1 is B-only; no other value is accepted")
ap.add_argument("--min-per-class", type=int, default=25, help="minimum positives AND negatives per AUROC cell")
a = ap.parse_args()

STEM = "A_B_look2"
E3 = config.CACHE / "look_e3"
P = config.CACHE / "pipeline"
SIGNALS = ("strength", "top1_concentration", "neg_entropy", "null_margin", "top1_area")


def load():
    rows = cache.load_json(P / f"{STEM}.json")
    assert all(r["id"].startswith("train_") for r in rows), "Look L1 is B-only; refuses validation/test"
    info = cache.load_json(E3 / f"{STEM}_info.json")
    heads = np.load(E3 / f"{STEM}_heads.f16.npy", mmap_mode="r")
    owner = [[ri, ci] for ri, r in enumerate(rows) for ci in range(len(r["claims"]))]
    if owner != info["owner"] or len(heads) != len(owner):
        raise ValueError("E3 cache ownership mismatch")
    s4 = cache.load_json(re_.BUNDLE_PATH)
    return rows, heads, owner, look_v4.selection()[0]["heads"], s4["look"]


def score_claims(rows, heads, owner, sel_heads, rule):
    owl = cache.OwlCache("train")
    maps, strengths = [], []
    for k, (ri, ci) in enumerate(owner):
        sigma = config.LOOK_RULES_BY_KIND[rows[ri]["claims"][ci]["kind"]]["sigma"]
        maps.append(look.smooth_map(look_v4.selected_map(heads[k], sel_heads), sigma))
        strengths.append(look_v4.residual_strength(heads[k], sel_heads))
    by_row: dict[int, list[int]] = {}
    for k, (ri, _ci) in enumerate(owner):
        by_row.setdefault(ri, []).append(k)
    records = []
    for k, (ri, ci) in enumerate(owner):
        row, claim = rows[ri], rows[ri]["claims"][ci]
        prop = owl[data.parse_id(row["id"])[1]]
        top, _ = re_.candidates(maps[k], prop, claim["kind"], rule, n=3)
        others = [maps[j] for j in by_row[ri] if j != k]
        if len(top):
            box = prop["boxes"][top[0]].astype(np.float32)
            sig = look_referent.signals(strengths[k], maps[k], box, others)
            area = float(look.box_area(box[None])[0])
        else:                                       # no positive-attention box: the plainest "no referent"
            sig = dict(strength=float(strengths[k]), entropy=look_referent.entropy(maps[k]), top1_concentration=0.,
                       null_margin=float("nan"), null_rank=-1, null_n_other_claims=len(others))
            area = float("nan")
        gt = grounding.class_boxes(row["id"]).get(claim["cls"])
        records.append(dict(id=row["id"], cls=claim["cls"], kind=claim["kind"], in_refs=claim["in_refs"],
                            n_candidates=int(len(top)), top1_area=area, has_gt_box=gt is not None and len(gt) > 0,
                            neg_entropy=-sig["entropy"], **sig))
    return records


def labelled(records):
    return [r for r in records if r["kind"] in ("thing", "stuff") and r["in_refs"] is not None]


def auroc_table(rows, min_n):
    out = {}
    for name in SIGNALS:
        pos = np.array([r[name] for r in rows if r["in_refs"] is True and np.isfinite(r[name])])
        neg = np.array([r[name] for r in rows if r["in_refs"] is False and np.isfinite(r[name])])
        ok = len(pos) >= min_n and len(neg) >= min_n
        out[name] = dict(auroc=metrics.auroc(pos, neg) if ok else None, n_pos=int(len(pos)), n_neg=int(len(neg)))
    return out


def by_size(rows, min_n):
    """Terciles of Look's top-1 box area; every cell reports its own n_pos/n_neg."""
    sized = [r for r in rows if np.isfinite(r["top1_area"])]
    if not sized:
        return dict(note="no claim has a top-1 box")
    edges = np.quantile([r["top1_area"] for r in sized], [1 / 3, 2 / 3]).tolist()
    cells = {"small": [], "medium": [], "large": []}
    for r in sized:
        cells["small" if r["top1_area"] <= edges[0] else "medium" if r["top1_area"] <= edges[1] else "large"].append(r)
    return dict(edges=edges, **{k: auroc_table(v, min_n) for k, v in cells.items()})


def gt_coverage(rows):
    """Why F30kE box area cannot be the size covariate: positives have boxes, negatives almost never."""
    return {lab: dict(n=sum(r["in_refs"] is v for r in rows),
                      with_gt_box=sum(r["in_refs"] is v and r["has_gt_box"] for r in rows))
            for lab, v in (("supported", True), ("unsupported", False))}


def cell(v):
    return f"{v['auroc']:.3f}" if v["auroc"] is not None else "–"


def write_markdown(out, path):
    L = ["# Look L1 — referans sinyalleri (B, otomatik proxy)", "",
         "Üreten: `scripts/46_look_referent.py`. Protokol: `docs/LOOK_L1_PROTOKOL.md`. "
         "Etiket (`in_refs`) otomatik referans proxy'sidir, insan etiketi değil. Harita: kelimeyi üreten adım "
         "(E3 önbelleği); sıralama: `region_evidence.candidates`, S4 Look kuralı.", "",
         f"Skorlanan claim: {out['n_claims']}; Look adayı olmayan (top-1 yok): {out['n_no_candidate']}. "
         f"Hücre başına en az {out['min_per_class']} pozitif ve negatif yoksa AUROC verilmez (–).", "",
         "## Genel (thing + stuff, kişi hariç)", "", "| Sinyal | AUROC | n destekli | n desteksiz |", "|---|---|---|---|"]
    for name, v in out["overall"].items():
        L.append(f"| {name} | {cell(v)} | {v['n_pos']} | {v['n_neg']} |")
    L += ["", "`top1_area` yalnız boyutun ne kadar ayırdığını gösterir; bir sinyal bunu geçmiyorsa boyutu izliyor olabilir.", "",
          "## Look top-1 kutu alanına göre üçte birlikler", ""]
    sz = out["by_size"]
    if "note" in sz:
        L.append(sz["note"])
    else:
        L += [f"Sınırlar: {sz['edges'][0]:.4f}, {sz['edges'][1]:.4f}", "",
              "| Sinyal | küçük | orta | büyük |", "|---|---|---|---|"]
        for name in SIGNALS:
            L.append(f"| {name} | " + " | ".join(f"{cell(sz[c][name])} ({sz[c][name]['n_pos']}/{sz[c][name]['n_neg']})"
                                                for c in ("small", "medium", "large")) + " |")
        L.append("\nParantez içi: (n destekli / n desteksiz).")
    g = out["gt_coverage"]
    L += ["", "## Neden F30kE kutu alanı kullanılmıyor", "",
          f"F30kE kutuları referans öbeklerine bağlı: destekli {g['supported']['with_gt_box']}/{g['supported']['n']}, "
          f"desteksiz {g['unsupported']['with_gt_box']}/{g['unsupported']['n']} claim'in kutusu var.", "",
          "Bu dosya yalnız B üzerindedir; validation/test okunmaz (`docs/LOOK_L1_PROTOKOL.md` §5)."]
    path.write_text("\n".join(L) + "\n")


def main():
    rows, heads, owner, sel_heads, rule = load()
    records = score_claims(rows, heads, owner, sel_heads, rule)
    lab = labelled(records)
    out = dict(protocol="docs/LOOK_L1_PROTOKOL.md", look_rule=rule, min_per_class=a.min_per_class,
               n_claims=len(records), n_no_candidate=sum(r["n_candidates"] == 0 for r in records),
               overall=auroc_table(lab, a.min_per_class), by_size=by_size(lab, a.min_per_class),
               gt_coverage=gt_coverage(lab))
    cache.save_json(out, config.RESULTS / "look_referent_B.json")
    write_markdown(out, config.ROOT / "docs" / "generated" / "LOOK_REFERENT_B.md")
    print(f"{out['n_claims']} claims scored -> results/look_referent_B.json, docs/generated/LOOK_REFERENT_B.md")


if __name__ == "__main__":
    main()
