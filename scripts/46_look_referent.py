"""Look L1 (docs/LOOK_L1_PROTOKOL.md): referent signals from the cached Look E3 signed heads, B only.

Reads data/cache/look_e3/A_B_look2_heads.f16.npy (per-claim signed sink maps, all 144 heads) and the
frozen data/cache/pipeline/A_B_look4.json (for the top-1 box v4 actually chose, and each claim's
kind/in_refs/cls). Computes lcr.look_referent's three signal families per claim -- this does not change
which box is chosen -- and reports how well they separate reference-supported from reference-unsupported
thing/stuff claims. That split is an AUTOMATIC PROXY (not a hallucination label): a claim can be correct
without being in any of the five references, and vice versa (docs/LCR.md). Results are also split by
Flickr30k Entities object-box size, to check whether a signal is really tracking "wrong object" rather
than just "small object". Never run on validation or test (asserted below).

    python scripts/46_look_referent.py
"""
import argparse

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, grounding, look, look_referent, look_v4, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="train", choices=["train"], help="L1 is B-only; no other value is accepted")
a = ap.parse_args()

STEM = "A_B_look2"
E3 = config.CACHE / "look_e3"
P = config.CACHE / "pipeline"
SIGNALS = ("strength", "top1_concentration", "neg_entropy", "null_margin")


def load():
    heads = np.load(E3 / f"{STEM}_heads.f16.npy", mmap_mode="r")
    info = cache.load_json(E3 / f"{STEM}_info.json")
    owner = info["owner"]                                   # [[row_index, claim_index_in_row], ...]
    frozen = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in frozen), "Look L1 is B-only; refuses validation/test"
    sel, _ = look_v4.selection()
    return heads, owner, frozen, sel["heads"]


def score_claims(heads, owner, frozen, sel_heads):
    by_row: dict[int, list[int]] = {}
    for k, (ri, _ci) in enumerate(owner):
        by_row.setdefault(ri, []).append(k)
    maps = [look_v4.selected_map(heads[k], sel_heads) for k in range(len(owner))]
    strengths = [look_v4.residual_strength(heads[k], sel_heads) for k in range(len(owner))]
    records = []
    for k, (ri, ci) in enumerate(owner):
        row = frozen[ri]
        claim = row["claims"][ci]
        box = np.asarray(claim["rec_att"]["box"], np.float32)
        if box.size != 4:
            continue                                        # Look marked this claim's box empty: no top-1 to score
        others = [maps[j] for j in by_row[ri] if j != k]
        sig = look_referent.signals(strengths[k], maps[k], box, others)
        area = None
        gt = grounding.class_boxes(row["id"]).get(claim["cls"])
        if gt is not None and len(gt):
            area = float(look.box_area(np.asarray(gt, np.float32)).mean())
        records.append(dict(id=row["id"], cls=claim["cls"], kind=claim["kind"], in_refs=claim["in_refs"],
                            gt_box_area=area, neg_entropy=-sig["entropy"], **sig))
    return records


def auroc_table(records):
    """AUROC per signal, separating in_refs=True from in_refs=False, kind in {thing, stuff} only."""
    rows = [r for r in records if r["kind"] in ("thing", "stuff") and r["in_refs"] is not None]
    out = {}
    for name in SIGNALS:
        pos = np.array([r[name] for r in rows if r["in_refs"] is True and not np.isnan(r[name])])
        neg = np.array([r[name] for r in rows if r["in_refs"] is False and not np.isnan(r[name])])
        out[name] = dict(auroc=metrics.auroc(pos, neg), n_pos=len(pos), n_neg=len(neg))
    return out


def by_size(records):
    """Same AUROC table, restricted to claims with an F30kE box, split at the median box area."""
    rows = [r for r in records if r["gt_box_area"] is not None]
    if len(rows) < 20:
        return {"n": len(rows), "note": "too few F30kE-boxed claims in B for a size split"}
    med = float(np.median([r["gt_box_area"] for r in rows]))
    small = [r for r in rows if r["gt_box_area"] <= med]
    large = [r for r in rows if r["gt_box_area"] > med]
    return dict(median_area=med, small=auroc_table(small), large=auroc_table(large),
               n_small=len(small), n_large=len(large))


def write_markdown(out, path):
    L = ["# Look L1 — referans sinyalleri (B, otomatik proxy)", "",
         "Üreten: `scripts/46_look_referent.py`. Protokol: `docs/LOOK_L1_PROTOKOL.md`. "
         "Etiket (`in_refs`) bir otomatik referans proxy'sidir, insan etiketi değil; "
         "gerçek insan altın standardı için adım 9'un pilot etiketlemesi gerekir.", "",
         f"Toplam skorlanan claim: {out['n_claims']} (top-1 kutusu olmayan {out['n_no_box']} claim dışlandı).", "",
         "## Genel (thing + stuff, kişi hariç)", "", "| Sinyal | AUROC | n destekli | n desteksiz |", "|---|---|---|---|"]
    for name, v in out["overall"].items():
        L.append(f"| {name} | {v['auroc']:.3f} | {v['n_pos']} | {v['n_neg']} |")
    L += ["", "## Nesne boyutuna göre (yalnız F30kE kutusu olan claim'ler, medyanda ikiye bölünmüş)", ""]
    sz = out["by_size"]
    if "note" in sz:
        L.append(sz["note"])
    else:
        L.append(f"Medyan kutu alanı: {sz['median_area']:.4f} (küçük n={sz['n_small']}, büyük n={sz['n_large']})")
        L += ["", "| Sinyal | AUROC (küçük) | AUROC (büyük) |", "|---|---|---|"]
        for name in SIGNALS:
            L.append(f"| {name} | {sz['small'][name]['auroc']:.3f} | {sz['large'][name]['auroc']:.3f} |")
    L += ["", "Bu dosya yalnız B üzerindedir; validation/test okunmaz (bkz. `docs/LOOK_L1_PROTOKOL.md` §5)."]
    path.write_text("\n".join(L) + "\n")


def main():
    heads, owner, frozen, sel_heads = load()
    records = score_claims(heads, owner, frozen, sel_heads)
    out = dict(n_claims=len(records), n_no_box=len(owner) - len(records),
              overall=auroc_table(records), by_size=by_size(records))
    cache.save_json(out, config.RESULTS / "look_referent_B.json")
    write_markdown(out, config.ROOT / "docs" / "generated" / "LOOK_REFERENT_B.md")
    print(f"{out['n_claims']} claims scored -> results/look_referent_B.json, "
         f"docs/generated/LOOK_REFERENT_B.md")


if __name__ == "__main__":
    main()
