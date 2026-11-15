"""POST-HOC ablation of the positive-anchored Check (written after scripts/43 failed its pre-registered criterion).

    OMP_NUM_THREADS=2 python scripts/44_check_fix_ablation.py
Diagnoses why S3 detects fewer swaps than S2 at matched FCR: varies only the abstention rule (C4). B only, same
folds and swap set as script 43. Not a selection: the default Check is not changed from these numbers.
Writes results/check_fix_ablation_B.json and docs/generated/CHECK_FIX_ABLATION_B.md.
"""
import importlib
import json

import numpy as np

import _common  # noqa: F401
from lcr import calibrate, claims, config, data
from lcr.vocab import CLASS_INDEX, CLASSES

s43 = importlib.import_module("43_check_fix")
VARIANTS = dict(full=dict(), abstain_few_positives_only=dict(abstain_auroc=None),
                no_abstention=dict(abstain_auroc=None, abstain_min=0))


def main():
    rows, maps, owl, q, hist, IM, xs, pooled, top1 = s43.build()
    ids = [r["id"] for r in rows]
    refs = {i: claims.reference_classes(data.references(i)) for i in ids}
    men = np.array([[c in refs[i] for c in CLASSES] for i in ids])
    ok = np.array([x["ok"] for x in xs]); nv = np.array([x["non_visual"] for x in xs]); nv0 = np.zeros(len(xs), bool)
    k = np.array([CLASS_INDEX[x["cls"]] for x in xs])
    sw = s43.swap_labels(xs, refs); m = sw >= 0; kk_sw = np.where(m, sw, k)
    npos = men.sum(0); hibase = m & (npos[np.maximum(sw, 0)] > npos[k])
    folds = np.array([x["fold"] for x in xs]); img_folds = np.array([calibrate.fold_of(i) for i in ids])
    prev = json.loads((config.RESULTS / "check_fix_B.json").read_text())
    out = dict(post_hoc=True, S2=prev["S2_five_way_z_oof"], variants={})
    for name, fit in VARIANTS.items():
        out["variants"][name] = {}
        for alpha in s43.ALPHAS:
            a3 = np.empty(len(xs), object); v3 = np.empty(len(xs), object); sub3 = np.empty(len(xs), object)
            a3s = np.full(len(xs), "", object); sub3s = np.full(len(xs), "", object)
            for f in range(5):
                te = folds == f
                idx, v, a, s, _, _ = s43.s3(IM, men, pooled, top1, xs, k, alpha, img_folds != f, te, nv, **fit)
                v3[idx], a3[idx], sub3[idx] = v, a, s
                idx, _, a, s, _, _ = s43.s3(IM, men, pooled, top1, xs, kk_sw, alpha, img_folds != f, te & m, nv0, **fit)
                a3s[idx], sub3s[idx] = a, s
            r = s43.summarise(a3, sub3, xs, ok, nv, sw, a3s, sub3s, hibase)
            r["proxy"] = calibrate.verdict_quality(v3, ok)
            out["variants"][name][str(alpha)] = r
            print(f"{name} alpha {alpha}: FCR {r['FCR']:.3f} det {r['swap']['detected']:.3f}", flush=True)
    (config.RESULTS / "check_fix_ablation_B.json").write_text(json.dumps(out, indent=1) + "\n")
    f = lambda v: "–" if v is None or v != v else f"{v:.3f}"
    L = ["# Check düzeltmesi — SONRADAN ablation (otomatik tablo)", "",
         "Üreten: `scripts/44_check_fix_ablation.py`. **Ön kayıtlı değil**: script 43 ölçütü tutmadıktan sonra, nedenini "
         "anlamak için. Yalnız çekimserlik kuralı (C4) değişir. Yalnız B, dış-kat; varsayılan bu sayılara göre seçilmedi.", "",
         "| varyant | α | FCR | FCR giysi | FCR diğer thing | swap yakalama | yüksek-taban | swap skip | gerçek sınıfla replace | ikame kesinliği |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    s2 = out["S2"]
    L.append(f"| S2 beş-yollu z | – | {f(s2['FCR'])} | {f(s2['FCR_by_group']['clothing'])} | {f(s2['FCR_by_group']['thing_other'])} | "
             f"{f(s2['swap']['detected'])} | {f(s2['swap']['detected_hibase'])} | {f(s2['swap']['skipped'])} | "
             f"{f(s2['swap']['replaced_with_true'])} | {f(s2['swap']['substitute_precision'])} |")
    for name, curve in out["variants"].items():
        for a, r in curve.items():
            w = r["swap"]; g = r["FCR_by_group"]
            L.append(f"| {name} | {a} | {f(r['FCR'])} | {f(g['clothing'])} | {f(g['thing_other'])} | {f(w['detected'])} | "
                     f"{f(w['detected_hibase'])} | {f(w['skipped'])} | {f(w['replaced_with_true'])} | {f(w['substitute_precision'])} |")
    (config.ROOT / "docs/generated/CHECK_FIX_ABLATION_B.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
