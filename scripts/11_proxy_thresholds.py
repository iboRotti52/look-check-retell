"""Interim verdict thresholds calibrated against the REFERENCE PROXY on training half B.

    python scripts/11_proxy_thresholds.py --attn sink

The provisional thresholds (t_support=0.2, t_contra=0.05) contradict far too many claims.
Until the human pilot labels exist (scripts/10_score_labels.py), this script picks an interim
operating point on half B -- never on validation or test -- using "claim class mentioned in
the five references" as the label. Only thing/stuff claims are used (person claims are almost
always correct). Criterion: maximise informedness of both verdicts,
    J = [P(sup | ok) - P(sup | not ok)] + [P(contra | not ok) - P(contra | ok)],
which, unlike balanced accuracy, does not reward contradicting everything.
--records att uses the attention-selected regions (LCR), det the detector-selected ones (B3).

Writes results/thresholds_proxy_<records><sfx>.json. It is NOT picked up automatically: pass it
explicitly with LCR_THRESHOLDS=..., so that human-calibrated thresholds (results/thresholds.json)
stay the default once they exist.
"""
import argparse

import numpy as np

import _common  # noqa: F401
from lcr import cache, config, pipeline
from lcr.vocab import KIND

ap = argparse.ArgumentParser()
ap.add_argument("--decoder", default="A")
ap.add_argument("--attn", default="sink", help="raw | sink | sink_top3 ... (pipeline-file suffix, see script 12)")
ap.add_argument("--records", default="att", choices=["att", "det"])
a = ap.parse_args()
other = "B" if a.decoder == "A" else "A"
sfx = "" if a.attn == "raw" else f"_{a.attn}"
rows = cache.load_json(config.CACHE / "pipeline" / f"{a.decoder}_{other}{sfx}.json")
pairs = [(c[f"rec_{a.records}"], bool(c["in_refs"])) for r in rows for c in r["claims"] if KIND[c["cls"]] != "person"]
base = pipeline.DEFAULT_THRESHOLDS
print(f"{len(pairs)} thing/stuff claims on half {other}, {100 * np.mean([not l for _, l in pairs]):.1f}% reference-unsupported")

grid = []
for ts in np.round(np.arange(0.02, 0.61, 0.02), 3):
    for tc in np.round(np.arange(0.005, 0.21, 0.005), 3):
        if tc >= ts:
            continue
        th = {**base, "t_support": float(ts), "t_contra": float(tc)}
        v = [(pipeline.finalise(r, th)["verdict"], ok) for r, ok in pairs]
        sup_rec = np.mean([vv == "supported" for vv, ok in v if ok])
        con_rec = np.mean([vv == "contradicted" for vv, ok in v if not ok])
        sup_fa = np.mean([vv == "supported" for vv, ok in v if not ok])
        con_fa = np.mean([vv == "contradicted" for vv, ok in v if ok])
        n_con = sum(vv == "contradicted" for vv, _ in v)
        prec = sum(vv == "contradicted" and not ok for vv, ok in v) / n_con if n_con else float("nan")
        dist = {k: float(np.mean([vv == k for vv, _ in v])) for k in ("supported", "contradicted", "uncertain")}
        grid.append(dict(t_support=float(ts), t_contra=float(tc), supported_recall=float(sup_rec),
                         contradicted_recall=float(con_rec), contradicted_precision=float(prec),
                         informedness=float(sup_rec - sup_fa + con_rec - con_fa),
                         n_contradicted=int(n_con), dist=dist))

best = max(grid, key=lambda g: g["informedness"])
prov = next(g for g in grid if abs(g["t_support"] - base["t_support"]) < 1e-9 and abs(g["t_contra"] - base["t_contra"]) < 1e-9)
tag = f"{a.records}{sfx}"
out = dict(source=f"{a.decoder}_{other}{sfx}", records=a.records, n_claims=len(pairs), best=best, provisional=prov, grid=grid)
cache.save_json(out, config.RESULTS / f"proxy_calibration_{tag}.json")
cache.save_json({**base, "t_support": best["t_support"], "t_contra": best["t_contra"], "calibrated_on": out["source"],
                 "records": a.records, "label": "reference proxy (interim)"}, config.RESULTS / f"thresholds_proxy_{tag}.json")
keys = ("t_support", "t_contra", "informedness", "contradicted_precision", "dist")
print("provisional:", {k: prov[k] for k in keys})
print("chosen     :", {k: best[k] for k in keys})
