"""Inspect the three frozen Look proposals separately, using B only.

The frozen rule's OOF thresholds were fitted for weighted scores. Per-box verdicts
below only diagnose that rule; they are not a newly calibrated competing method.
No validation/test access, selection or writes to frozen pipeline records.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from lcr import cache, calibrate, config, data, look
from lcr.vocab import CLASSES, CLASS_INDEX


def main():
    spec = importlib.util.spec_from_file_location("check4_review", Path(__file__).with_name("35_check_v4.py"))
    c4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c4)
    folder = config.CACHE / "pipeline"
    rows = cache.load_json(folder / "A_B_look4_check4.json")
    selected = cache.load_json(config.RESULTS / "check_v4_selection.json")
    cfg = selected["table"][selected["choice"]]["cfg"]
    assert cfg == dict(Q="Q1", C="C2", A="A1", R="R2", G="G0", E="E0")
    key = ("Q1", "C2", "R2", "A1")
    xs = []
    for ri, row in enumerate(rows):
        assert row["id"].startswith("train_")
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            rec = c["rec_att"]
            xs.append(dict(ri=ri, ci=ci, id=row["id"], cls=c["cls"], kind=c["kind"],
                           ok=bool(c["in_refs"]), fold=calibrate.fold_of(row["id"]),
                           excess=rec["excess"], has_region=rec["region"] >= 0,
                           v={key: (rec["s_claim_cal"], rec["s_alt_cal"], rec["alt"], rec["s_image_cal"])},
                           verdict=rec["verdict"]))
    folds = np.array([x["fold"] for x in xs])
    thresholds = {}
    for f in range(5):
        v, th = c4.run(xs, cfg, folds != f, folds == f)
        assert all(v[i] == xs[i]["verdict"] for i in np.flatnonzero(folds == f))
        thresholds[f] = th
    print("Frozen B OOF verdicts reproduced from stored scores.", flush=True)
    q = np.load(config.CACHE / "owl" / c4.QUERIES["Q1"])
    mu, sd = c4.calib_stats(q)
    owl = cache.OwlCache("train")
    maps = np.load(folder / "A_B_look4.maps.npz")
    owners = {tuple(o): i for i, o in enumerate(maps["owner"])}
    counts, patterns = Counter(), Counter()
    frisbees = []
    for x in xs:
        if not x["has_region"]:
            continue
        _, idx = data.parse_id(x["id"])
        prop = owl[idx]
        att = maps["maps"][owners[(x["ri"], x["ci"])]]
        reg, rs, _ = look.select_frozen(att, prop["boxes"].astype(np.float32),
                                        prop["objectness"].astype(np.float32), x["kind"])
        rec = rows[x["ri"]]["claims"][x["ci"]]["rec_att"]
        assert reg == rec["region"]
        top = np.argsort(-np.where(np.isfinite(rs), rs, -np.inf))[:3]
        lg = c4.logits(prop, q)
        S = (lg - mu) / sd
        sc = S[top, CLASS_INDEX[x["cls"]]]
        w = np.clip(rs[top], 0, None)
        w = w / w.sum() if w.sum() else np.full(len(top), 1 / len(top))
        assert np.isclose(float(w @ sc), rec["s_claim_cal"], atol=2e-5)
        eligible = c4.eligible(x["cls"], "A1")
        js = np.argmax(np.where(eligible[None, :], S[top], -np.inf), axis=1)
        sa = S[top, js]
        ts, tc, ta = thresholds[x["fold"]][x["kind"]]
        vv = c4.rule(sc, sa, sc, np.ones(len(top), bool), ts, tc, False, ta)
        patterns[(x["verdict"], "/".join(vv))] += 1
        counts["claims_with_region"] += 1
        if x["verdict"] != "supported" and np.any(sc >= ts):
            counts["weighted_not_supported_but_one_box_above_support"] += 1
            counts["same_reference_supported" if x["ok"] else "same_reference_unsupported"] += 1
        if "supported" in vv and "contradicted" in vv:
            counts["boxes_give_both_support_and_contradiction"] += 1
        if x["cls"] == "frisbee":
            k = CLASS_INDEX["frisbee"]
            best = int(lg[:, k].argmax())
            raw = 1 / (1 + np.exp(-lg[:, k]))
            d = dict(id=x["id"], reference_support=x["ok"], weighted_verdict=x["verdict"],
                     weighted_claim_z=rec["s_claim_cal"], thresholds=[ts, tc, ta],
                     top3=[dict(region=int(t), weight=float(wi), claim_raw=float(raw[t]),
                                claim_z=float(z), alternative=CLASSES[j], alternative_z=float(a), verdict=str(v))
                           for t, wi, z, j, a, v in zip(top, w, sc, js, sa, vv)],
                     image_best_region=best, image_best_raw=float(raw[best]),
                     image_best_passes_look_size_filter=bool(look.size_ok(prop["boxes"].astype(np.float32))[best]),
                     image_best_box=prop["boxes"][best].astype(float).tolist(),
                     image_best_is_in_top3=bool(best in top))
            frisbees.append(d)
    out = dict(split="B", label="automatic reference proxy; exploratory diagnostics only",
               note="Per-box verdicts reuse weighted-score OOF thresholds; no independent per-box calibration.",
               counts=counts, patterns=[dict(weighted=a, per_box=b, n=n) for (a, b), n in patterns.most_common()],
               frisbee=frisbees)
    cache.save_json(out, config.RESULTS / "check_v4_boxes_B.json")
    render_and_summarise(out, owl)
    print(json.dumps(dict(counts=counts, frisbee=frisbees), indent=2))


def render_and_summarise(out, owl):
    failures = [d for d in out["frisbee"] if d["reference_support"] and d["weighted_verdict"] != "supported"]
    for d in out["frisbee"]:
        prop = owl[int(d["id"].rsplit("_", 1)[1])]
        best = d["image_best_region"]
        d["image_best_passes_look_size_filter"] = bool(look.size_ok(prop["boxes"].astype(np.float32))[best])
        d["image_best_box"] = prop["boxes"][best].astype(float).tolist()
    cache.save_json(out, config.RESULTS / "check_v4_boxes_B.json")
    lines = ["# Check v4: üç kutunun ayrı tanısı (yalnız B)", "",
             "Üreten: `scripts/review_check_v4_boxes_B.py`. Yeni seçim veya validation/test erişimi yok.",
             "Referans desteği otomatik proxy'dir. Kutu kararları ortalama için fit edilen OOF eşiklerini kullanır;",
             "yeni bir kutu-başına yöntemin kalibrasyonu veya performans iddiası değildir.", "",
             "| Tanı | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in out["counts"].items()]
    lines += ["", "Referansta desteklenen fakat v4'ün desteklemediği frizbi iddiaları:", "",
              "| B görüntüsü | v4 | ilk 3 kutuda ham frisbee skorları | görüntüde en yüksek | boyut filtresini geçer mi | ilk 3'te mi |",
              "|---|---|---|---:|---|---|"]
    for d in failures:
        scores = ", ".join(f"{b['claim_raw']:.3f}" for b in d["top3"])
        lines.append(f"| {d['id']} | {d['weighted_verdict']} | {scores} | {d['image_best_raw']:.3f} | "
                     f"{'evet' if d['image_best_passes_look_size_filter'] else 'hayır'} | "
                     f"{'evet' if d['image_best_is_in_top3'] else 'hayır'} |")
    lines += ["", "İlk dört örnek görsel olarak denetlendi. Yeşil kutu frizbiye oturuyor;",
              "diğer dört örnekte bu tablo yalnız skor ve filtre tanısıdır.", ""]
    (config.ROOT / "docs/generated/CHECK_V4_BOXES_B.md").write_text("\n".join(lines))
    plot_examples(failures[:4], owl, (2, 2), "check_v4_frisbee_B.png")
    plot_examples(failures[:1], owl, (1, 1), "check_v4_frisbee_example_B.png")


def plot_examples(examples, owl, shape, filename):
    fig, axes = plt.subplots(*shape, figsize=(12, 10) if shape == (2, 2) else (8, 6), squeeze=False)
    for ax, d in zip(axes.flat, examples):
        prop = owl[int(d["id"].rsplit("_", 1)[1])]
        boxes = prop["boxes"][[b["region"] for b in d["top3"]]]
        best_box = prop["boxes"][d["image_best_region"]]
        im = data.load_image(d["id"])
        ax.imshow(im)
        W, H = im.size
        for box, color, label, style in [
                *[(box, color, f"Look {i+1}", "--") for i, (box, color) in enumerate(zip(boxes, ["red", "orange", "magenta"]))],
                (best_box, "lime", "OWL best frisbee", "-")]:
            x0, y0, x1, y1 = box * [W, H, W, H]
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, ec=color, lw=2, ls=style))
            ax.text(x0, max(0, y0-3), label, color=color, fontsize=8,
                    bbox=dict(facecolor="black", alpha=.6, pad=1))
        raw3 = ", ".join(f"{p['claim_raw']:.3f}" for p in d["top3"])
        ax.set_title(f"{d['id']} | v4: {d['weighted_verdict']}\nfrisbee raw: Look=[{raw3}] | image best={d['image_best_raw']:.3f}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Look kutuları ve OWLv2'nin bulduğu frizbi (yeşil)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    path = config.RESULTS / filename
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        render_and_summarise(cache.load_json(config.RESULTS / "check_v4_boxes_B.json"), cache.OwlCache("train"))
    else:
        main()
