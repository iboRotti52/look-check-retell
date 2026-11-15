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
Disagreements are adjudicated by a third annotator's CSV if given with --adjudicator.
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np

import _common  # noqa: F401
from lcr import cache, check, config, metrics, pipeline

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

# ---- threshold calibration on vocabulary claims
key = cache.load_json(config.ROOT / "annotation" / a.name / "key.json")
item_img = {it["item"]: it["image"] for it in key["items"]}
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
if a.write_thresholds and best:
    cache.save_json({**base, "t_support": best["t_support"], "t_contra": best["t_contra"]}, config.RESULTS / "thresholds.json")
    print("thresholds written to results/thresholds.json")
