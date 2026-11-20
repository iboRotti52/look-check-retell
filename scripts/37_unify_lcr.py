"""B-only selection/calibration of the one attention-object evidence pipeline.

Protocol: docs/LCR_BIRLESTIRME_PROTOKOL.md. No validation/test data are loaded.
Old experiments remain untouched. Output: lcr_bundle.json, lcr_development.json,
A_B_lcr.json and an automatically generated development table.
"""
import hashlib
import itertools
import json
import shutil

import _common  # noqa: F401
import numpy as np

from lcr import cache, calibrate, config, grounding, look
from lcr import region_evidence as re
from lcr.vocab import CLASS_INDEX, CLASSES


def run():
    if re.HISTORICAL_BUNDLE_PATH.exists():
        raise FileExistsError("LCR selection is locked; do not overwrite the existing bundle")
    p = config.CACHE / "pipeline"
    rows = cache.load_json(p / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows)
    maps = np.load(p / "A_B_look4.maps.npz")
    owners = {tuple(o): i for i, o in enumerate(maps["owner"])}
    owl = cache.OwlCache("train")
    qfile = "class_queries_single.npy"
    queries = np.load(config.CACHE / "owl" / qfile).astype(np.float32)
    rules = [dict(min_patches=m, alpha=a) for m, a in itertools.product((2, 0), (.5, 1.))]
    xs, raw_features, image_max, look_counts = [], [], [], [dict(n=0, top1=0, top3=0, small_n=0, small_top3=0, small_top3_any_box=0) for _ in rules]
    fit_images = np.array([re.partition(r["id"]) == "fit" for r in rows])
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, queries)
        image_max.append(lg.max(0))
        gt_dict = grounding.class_boxes(row["id"]) if fit_images[ri] else {}
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            att = maps["maps"][owners[(ri, ci)]]
            one = []
            gt = gt_dict.get(c["cls"])
            for rj, rule in enumerate(rules):
                top, weights = re.candidates(att, prop, c["kind"], rule)
                one.append([re.pool_logits(lg, top, weights, pool) for pool in ("weighted", "max")])
                if gt is not None and len(gt):
                    # small_top3 needs the hit on a SMALL box (2026-11-19); the first count credited any box of the
                    # class and is kept as small_top3_any_box. The Look choice below uses top3 only.
                    h = grounding.coverage_hits(prop["boxes"][top].astype(np.float32), gt)
                    s = look_counts[rj]
                    s["n"] += 1; s["top1"] += h["top1"]; s["top3"] += h["top3"]
                    s["small_n"] += h["has_small"]; s["small_top3"] += h["small_top3"]
                    s["small_top3_any_box"] += h["small_top3_any_box"]
            xs.append(dict(ri=ri, ci=ci, cls=c["cls"], kind=c["kind"], ok=c["in_refs"],
                           id=row["id"], fold=calibrate.fold_of(row["id"])))
            raw_features.append(one)
        if (ri + 1) % 500 == 0:
            print(f"B features {ri+1}/{len(rows)}", flush=True)
    raw_features, image_max = np.asarray(raw_features), np.asarray(image_max)
    top3 = [s["top3"] / max(s["n"], 1) for s in look_counts]
    near = [j for j, rate in enumerate(top3) if rate >= max(top3) - .005]
    chosen_look = min(near, key=lambda j: (rules[j]["min_patches"] != 0, rules[j]["alpha"] != .5))
    print("Look rules", list(zip(rules, look_counts)), "chosen", rules[chosen_look], flush=True)
    # All remaining choices see only B_fit. Image-level scale is fit inside each training fold.
    image_indices = np.array([x["ri"] for x in xs])
    fit = fit_images[image_indices]; cal = ~fit
    kinds = np.array([x["kind"] for x in xs]); ok = np.array([x["ok"] for x in xs], bool)
    cls = np.array([CLASS_INDEX[x["cls"]] for x in xs]); folds = np.array([x["fold"] for x in xs])
    image_folds = np.array([calibrate.fold_of(r["id"]) for r in rows])

    def scaled_features(pool_index, alternatives, train_images):
        mx = image_max[train_images]
        mu, sd = mx.mean(0), mx.std(0) + 1e-6
        z = (raw_features[:, chosen_look, pool_index] - mu) / sd
        masks = np.array([re.alternative_mask(x["cls"], alternatives) for x in xs])
        sc = z[np.arange(len(xs)), cls]
        sa = np.where(masks, z, -np.inf).max(1)
        return sc, sa, mu, sd

    def fit_and_decide(sc, sa, train, test):
        v = np.full(len(xs), "uncertain", object); th = {}
        for kind in ("thing", "stuff"):
            km = kinds == kind
            ts, tc = re.fit_thresholds(sc[km & train], sa[km & train], ok[km & train])
            tm = km & test
            v[tm] = re.decide(sc[tm], sa[tm], np.isfinite(sc[tm]), ts, tc)
            th[kind] = re.encode_thresholds(ts, tc)
        return v, th

    checks = []
    for pi, pool in enumerate(("weighted", "max")):
        for alternatives in ("all", "group"):
            v = np.full(len(xs), "uncertain", object)
            for f in range(5):
                sc, sa, _, _ = scaled_features(pi, alternatives, fit_images & (image_folds != f))
                train, test = fit & (folds != f), fit & (folds == f)
                vf, _ = fit_and_decide(sc, sa, train, test)
                v[test] = vf[test]
            quality = calibrate.verdict_quality(v[fit], ok[fit])
            checks.append(dict(pool=pool, alternatives=alternatives, quality=quality))
    feasible = [c for c in checks if np.isfinite(c["quality"]["contradicted_precision"]) and c["quality"]["contradicted_precision"] >= .45]
    selection_gate = bool(feasible)
    eligible = feasible or checks
    best = max(c["quality"]["J"] for c in eligible)
    near = [c for c in eligible if c["quality"]["J"] >= best - .01]
    choice = min(near, key=lambda c: (c["pool"] != "max", c["alternatives"] != "group"))
    print("Check options", checks, "chosen", choice, flush=True)
    pi = ("weighted", "max").index(choice["pool"])
    sc, sa, mu, sd = scaled_features(pi, choice["alternatives"], ~fit_images)
    final_v, th = fit_and_decide(sc, sa, cal, np.ones(len(xs), bool))
    bundle = dict(schema_version=1, method="lcr", detector=config.OWL_NAME, classes=CLASSES,
                  attention=re.attention_provenance(),
                  query_file=qfile, query_sha256=hashlib.sha256(queries.tobytes()).hexdigest(),
                  look=rules[chosen_look], pool=choice["pool"], alternatives=choice["alternatives"],
                  mu=mu.tolist(), sd=sd.tolist(), thresholds=th,
                  fit_ids=[r["id"] for i, r in enumerate(rows) if fit_images[i]],
                  calibration_ids=[r["id"] for i, r in enumerate(rows) if not fit_images[i]],
                  label="automatic reference proxy; not human correctness", protocol="docs/LCR_BIRLESTIRME_PROTOKOL.md",
                  selection_gate=selection_gate)
    bundle["id"] = re.fingerprint(bundle)
    checker = re.RegionChecker(bundle, queries)
    # Apply the exact same class used by live inference and synthetic relabelling.
    generated = []
    for ri, row in enumerate(rows):
        out = dict(row, evidence_partition="fit" if fit_images[ri] else "cal", bundle_id=bundle["id"])
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        lg = re.proposal_logits(prop, queries)
        claims_out = []
        for ci, c in enumerate(row["claims"]):
            att = maps["maps"][owners[(ri, ci)]]
            rec = checker.record(c["cls"], att, prop, logits=lg,
                                 person_record=c["rec_att"] if c["kind"] == "person" else None)
            claims_out.append(dict(c, rec_att=rec, regions_by_k={"4": rec["region"]},
                                   excess_by_k={"4": rec["excess"]}))
        out.update(claims=claims_out, check_metadata=dict(method="lcr", bundle_id=bundle["id"],
                   label=bundle["label"], calibration="B_cal only", retell_training="B_fit only"),
                   look_metadata=dict(row["look_metadata"], region_rule="lcr", proposal_rule=bundle["look"]))
        generated.append(out)
    # Production class and selection calculations must agree, including their verdicts.
    assert all(generated[x["ri"]]["claims"][x["ci"]]["rec_att"]["verdict"] == final_v[i]
               for i, x in enumerate(xs))
    alternative_stats = {}
    for name, mask in (("fit", fit), ("cal", cal)):
        # Image references are read only from train, using caption columns only.
        from lcr import claims as claim_tools
        import pandas as pd
        frames = [pd.read_parquet(f, columns=data_columns()) for f in sorted((config.DATA / "flickr8k/data").glob("train-*.parquet"))]
        references = pd.concat(frames, ignore_index=True)
        hit = n = 0
        for i in np.flatnonzero(mask & ~ok & (final_v == "contradicted")):
            x = xs[i]; rec = generated[x["ri"]]["claims"][x["ci"]]["rec_att"]
            idx = int(x["id"].rsplit("_", 1)[1])
            hit += rec["alt"] in claim_tools.reference_classes(references.iloc[idx].tolist()); n += 1
        alternative_stats[name] = dict(n=n, alternative_in_refs=hit, rate=hit / n if n else None)
    result = dict(bundle_id=bundle["id"], look_options=[dict(rule=r, **s) for r, s in zip(rules, look_counts)],
                  look_choice=rules[chosen_look], check_options=checks, check_choice=choice,
                  selection_gate=selection_gate, n_fit_images=int(fit_images.sum()), n_cal_images=int((~fit_images).sum()),
                  final_fit_descriptive=calibrate.verdict_quality(final_v[fit], ok[fit]),
                  calibration_resubstitution=calibrate.verdict_quality(final_v[cal], ok[cal]),
                  alternatives_reference_proxy=alternative_stats,
                  caveat="B development, not independent validation; Look selection sees B_fit GT; calibration score is resubstitution")
    re.HISTORICAL_BUNDLE_PATH.write_text(json.dumps(bundle, indent=1, allow_nan=False) + "\n")
    cache.save_json(result, config.RESULTS / "lcr_development.json")
    cache.save_json(generated, p / "A_B_lcr.json")
    shutil.copyfile(p / "A_B_look4.maps.npz", p / "A_B_lcr.maps.npz")
    lines = ["# Birleşik LCR — yalnız B geliştirme sonuçları", "", "Üreten: `scripts/37_unify_lcr.py`.",
             "Validation/test okunmadı. Etiket otomatik referans proxy'si. B sonuçları yeni bağımsız başarı iddiası değildir.", "",
             "## Look (B_fit insan kutuları; kişi dışı)", "",
             "| min patch | alpha | n | ilk kutu | ilk üçten biri | küçük GT n | küçük GT ilk üç |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r, s in zip(rules, look_counts):
        lines.append(f"| {r['min_patches']} | {r['alpha']} | {s['n']} | {s['top1']/s['n']:.3f} | {s['top3']/s['n']:.3f} | {s['small_n']} | {s['small_top3']/max(s['small_n'],1):.3f} |")
    lines += ["", "## Check (B_fit görüntü ÇD; seçilmiş Look koşullu)", "",
              "| birleştirme | alternatif | J | contr. kesinlik | yanlış yakalama |", "|---|---|---:|---:|---:|"]
    for c in checks:
        s = c["quality"]
        lines.append(f"| {c['pool']} | {c['alternatives']} | {s['J']:.3f} | {s['contradicted_precision']:.3f} | {s['recall_wrong']:.3f} |")
    lines += ["", f"Seçim: Look `{bundle['look']}`; Check `{bundle['pool']}/{bundle['alternatives']}`.",
              f"Bundle: `{bundle['id']}`. B_fit görüntü: {result['n_fit_images']}; B_cal: {result['n_cal_images']}.",
              "Son ölçek ve eşikler yalnız B_cal'dan. Retell B_fit üzerinde eğitilir; sabit epoch kullanır.", "",
              "## Alternatif proxy'si", "", "Yalnız referans desteklemeyen ve contradicted olan claim'lerde alternatif referansta mı?", "",
              "| bölüm | n | referansta alternatif | oran |", "|---|---:|---:|---:|"]
    for name, s in alternative_stats.items():
        lines.append(f"| {name} | {s['n']} | {s['alternative_in_refs']} | {s['rate']} |")
    lines += ["", "B_cal eşik fit verisidir, bağımsız test değildir. İnsan kutuları yalnız Look seçim/ölçümünde kullanıldı.", ""]
    (config.ROOT / "docs/generated/LCR_DEVELOPMENT.md").write_text("\n".join(lines))
    print("Wrote unified bundle and B records", flush=True)


def data_columns():
    return [f"caption_{i}" for i in range(5)]


if __name__ == "__main__":
    run()
