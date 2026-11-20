"""Score human labels from the annotation page and calibrate the verdict thresholds.

    python scripts/10_score_labels.py --name pilot annotation/pilot/labels_pilot_*.csv

Inputs: one CSV per annotator, exported from annotation/<name>/annotate.html.
Outputs (results/labels_<name>.json):
  * label distribution, HICR (incorrect / decided) per annotator and on the adjudicated set
  * Cohen's kappa on the items both annotators labelled
  * for vocabulary claims: a sweep of (t_support, t_contra) that compares the detector verdict with
    the human label, and the operating point that maximises balanced accuracy of
    supported-vs-incorrect while keeping 'contradicted' precision >= 0.5.
    With --write-thresholds the chosen point is written to results/thresholds.json,
    which every later script picks up (lcr/pipeline.thresholds).
  * Look L1 (docs/LOOK_L1_PROTOKOL.md, step 9): error-type distribution/agreement
    (misidentified vs invented) among adjudicated "incorrect" claims, and the human-drawn
    true-referent boxes for "misidentified" claims, written separately to
    results/look_l1_referent_boxes_<name>.json. Silently skipped for CSVs exported before
    these columns existed (annotate.html did not yet write them).
Disagreements are adjudicated by a third annotator's CSV if given with --adjudicator.
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np

import _common  # noqa: F401
from lcr import cache, check, config, metrics, pilot, pipeline

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+")
ap.add_argument("--name", default="pilot")
ap.add_argument("--adjudicator", default=None, help="CSV of the third annotator")
ap.add_argument("--write-thresholds", action="store_true")
a = ap.parse_args()


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {(r["item"], int(r["noun_index"])): r for r in rows}, (rows[0]["annotator"] if rows else path)


sheets = [read(p) for p in a.csvs]
adj = read(a.adjudicator)[0] if a.adjudicator else {}
out = dict(annotators=[w for _, w in sheets])
for sheet, who in sheets:
    labs = [r["label"] for r in sheet.values() if r["label"] and r["label"] != "na"]
    out[f"dist_{who}"] = dict(Counter(labs))
    out[f"hicr_{who}"] = metrics.hicr(labs)

if len(sheets) >= 2:
    (s1, w1), (s2, w2) = sheets[:2]
    both = [k for k in s1 if k in s2 and s1[k]["label"] and s2[k]["label"]]
    l1, l2 = [s1[k]["label"] for k in both], [s2[k]["label"] for k in both]
    out["kappa"] = metrics.cohen_kappa(l1, l2) if both else float("nan")
    out["raw_agreement"] = float(np.mean([x == y for x, y in zip(l1, l2)])) if both else float("nan")
    out["n_double"] = len(both)

# adjudicated label per noun: agreement, else adjudicator, else 'uncertain'
final = {}
keys = set().union(*[set(s) for s, _ in sheets])
for k in keys:
    ls = [s[k]["label"] for s, _ in sheets if k in s and s[k]["label"]]
    if not ls:
        continue
    if len(set(ls)) == 1:
        final[k] = ls[0]
    elif k in adj and adj[k]["label"]:
        final[k] = adj[k]["label"]
    else:
        final[k] = "uncertain"
ref = next(iter(sheets))[0]
labs = [v for v in final.values() if v != "na"]
out["final_dist"] = dict(Counter(labs))
out["final_hicr"] = metrics.hicr(labs)
voc = [(k, v) for k, v in final.items() if v != "na" and ref.get(k, {}).get("in_vocab") == "1"]
oov = [(k, v) for k, v in final.items() if v != "na" and ref.get(k, {}).get("in_vocab") == "0"]
out["hicr_in_vocab"] = metrics.hicr([v for _, v in voc])
out["hicr_out_of_vocab"] = metrics.hicr([v for _, v in oov])

key = cache.load_json(config.ROOT / "annotation" / a.name / "key.json")
item_img = {it["item"]: it["image"] for it in key["items"]}

# ---- Look L1 (docs/LOOK_L1_PROTOKOL.md, step 9): error-type and referent-box fields.
# Older CSVs, exported before these columns existed, simply have no "error_type"/"box" key;
# .get() keeps this script working on them unchanged. Rules (lcr/pilot.py) are CSV-order independent:
# agreement, else the adjudicator, else uncertain / no gold box.
err_by_key: dict[tuple, str] = {}
err_source = Counter()
for k, v in final.items():
    if v != "incorrect":
        continue
    etype, source = pilot.final_error_type([s[k] for s, _ in sheets if k in s], adj.get(k))
    err_by_key[k] = etype
    err_source[source] += 1
out["error_type_dist"] = dict(Counter(err_by_key.values()))
out["error_type_source"] = dict(err_source)

if len(sheets) >= 2:
    both_err = [k for k in both if s1[k].get("error_type") and s2[k].get("error_type")]
    out["n_error_type_double"] = len(both_err)
    if both_err:
        e1_vals = [s1[k]["error_type"] for k in both_err]
        e2_vals = [s2[k]["error_type"] for k in both_err]
        out["error_type_kappa"] = metrics.cohen_kappa(e1_vals, e2_vals)
        out["error_type_raw_agreement"] = float(np.mean([x == y for x, y in zip(e1_vals, e2_vals)]))

# Gold-standard L1 test set: (image, claimed class, human-drawn true-referent box) for every
# adjudicated "misidentified" claim with a box. scripts/46/47 compare Look's own box against
# these once a real pilot has been run; this script only collects them.
referent_boxes, box_source, box_needs_adjudication, box_ious = [], Counter(), [], []
for k, etype in err_by_key.items():
    if etype != "misidentified":
        continue
    box, source, iou = pilot.final_box([s[k] for s, _ in sheets if k in s], adj.get(k))
    box_source[source] += 1
    if iou is not None:
        box_ious.append(iou)
    item, j = k
    if box is None:
        box_needs_adjudication.append(dict(item=item, noun_index=j, reason=source))
        continue
    referent_boxes.append(dict(item=item, noun_index=j, image=item_img.get(item), cls=ref.get(k, {}).get("cls"),
                               box=box, box_source=source, annotator_box_iou=iou))
out["n_misidentified_with_box"] = len(referent_boxes)
out["referent_box_source"] = dict(box_source)
out["referent_box_mean_annotator_iou"] = float(np.mean(box_ious)) if box_ious else None
out["referent_boxes_needing_adjudication"] = box_needs_adjudication
cache.save_json(referent_boxes, config.RESULTS / f"look_l1_referent_boxes_{a.name}.json")

# ---- B human pilot (docs/PILOT_PROTOKOL.md): per-group results on the sampled claims of the S4/S2 records
pilot_mode = key.get("mode") == "lcr_pilot"
if pilot_mode:
    answers, sources, ref_boxes = {}, Counter(), []
    for f in key["focus"]:
        k = (f["item"], f["noun_index"])
        entries = [s[k] for s, _ in sheets if k in s]
        lab = final.get(k)
        if lab is None:
            continue
        ans = dict(label=lab)
        if lab == "incorrect":
            ans["error_type"] = err_by_key.get(k, "uncertain")
        if f["s4_action"] == "replace":
            ans["substitute_ok"], src = pilot.final_answer(entries, adj.get(k), "substitute_ok")
            sources[f"substitute_{src}"] += 1
        answers[(f["id"], f["claim_index"])] = ans
        if lab == "correct" or ans.get("error_type") == "misidentified":
            box, src, iou = pilot.final_box(entries, adj.get(k))
            sources[f"box_{src}"] += 1
            # detector / Look / human box comparison: the human box next to the box Look's S4 record shows
            ref_boxes.append(dict(id=f["id"], claim_index=f["claim_index"], cls=f["cls"], stratum=f["stratum"],
                                  label=lab, error_type=ans.get("error_type"), human_box=box, human_box_source=src,
                                  annotator_box_iou=iou, look_region=f["look_region"], look_box=f["look_box"]))
    out["pilot"] = dict(records=key["records"], s4_bundle_id=key["s4_bundle_id"], population=key["population"],
                        n_focus=len(key["focus"]), n_focus_labelled=len(answers), sources=dict(sources),
                        strata=pilot.summarise_strata(key["focus"], answers),
                        caveat="Only the random group estimates natural rates; the other groups are targeted.")
    cache.save_json(ref_boxes, config.RESULTS / f"pilot_referent_boxes_{a.name}.json")

if not pilot_mode:
    # The v1 threshold sweep below applies to the historical Check, not to the S4/S2 pilot records.
    # ---- threshold calibration on vocabulary claims
    rows = {r["id"]: r for r in cache.load_json(config.CACHE / "pipeline" / f"{key['decoder']}_{key['split']}_sink.json")} \
        if (config.CACHE / "pipeline" / f"{key['decoder']}_{key['split']}_sink.json").exists() else \
        {r["id"]: r for r in cache.load_json(config.CACHE / "pipeline" / f"{key['decoder']}_{key['split']}.json")}
    pairs = []
    for (item, j), lab in voc:
        if lab not in ("correct", "incorrect"):
            continue
        cls = ref[(item, j)]["cls"]
        row = rows.get(item_img[item])
        rec = next((c["rec_att"] for c in (row["claims"] if row else []) if c["cls"] == cls), None)
        if rec is not None:
            pairs.append((rec, lab))
    out["n_calibration_claims"] = len(pairs)
    grid = []
    base = pipeline.thresholds()
    for ts in np.round(np.arange(0.02, 0.41, 0.02), 3):
        for tc in np.round(np.arange(0.005, 0.11, 0.005), 3):
            if tc >= ts:
                continue
            th = {**base, "t_support": float(ts), "t_contra": float(tc)}
            v = [(pipeline.finalise(r, th)["verdict"], lab) for r, lab in pairs]
            sup_ok = np.mean([vv == "supported" for vv, lab in v if lab == "correct"]) if any(l == "correct" for _, l in v) else 0
            con_ok = np.mean([vv == "contradicted" for vv, lab in v if lab == "incorrect"]) if any(l == "incorrect" for _, l in v) else 0
            n_con = sum(vv == "contradicted" for vv, _ in v)
            prec = sum(vv == "contradicted" and lab == "incorrect" for vv, lab in v) / n_con if n_con else float("nan")
            grid.append(dict(t_support=float(ts), t_contra=float(tc), supported_recall=float(sup_ok),
                             contradicted_recall=float(con_ok), contradicted_precision=float(prec),
                             balanced=float((sup_ok + con_ok) / 2)))
    ok = [g for g in grid if not np.isnan(g["contradicted_precision"]) and g["contradicted_precision"] >= 0.5]
    best = max(ok or grid, key=lambda g: g["balanced"]) if grid else None
    out["calibration_best"] = best
    out["calibration_grid"] = grid
cache.save_json(out, config.RESULTS / f"labels_{a.name}.json")
print({k: v for k, v in out.items() if k != "calibration_grid"})
if a.write_thresholds and not pilot_mode and best:
    cache.save_json({**base, "t_support": best["t_support"], "t_contra": best["t_contra"]}, config.RESULTS / "thresholds.json")
    print("thresholds written to results/thresholds.json")
