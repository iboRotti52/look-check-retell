"""Check v2, learned verifier: which signals tell a wrong object claim from a correct one? (B: choose, val: report)

    python scripts/22_check_verifier.py            # needs scripts/21_caption_signals.py output
    python scripts/22_check_verifier.py --write    # also write <stem>_check2.json (prior+det+cap) and
                                                   # <stem>_check2vis.json (prior+det, visual-only ablation)

Meta-classification of object claims after MetaToken (arXiv 2405.19186) and HaloProbe (ICML'26), with the visual
verification signals of this project added. A logistic regression with shrunk per-class intercepts
(lcr.calibrate.FeatureLogistic), cross-fitted on half B (5 folds by image), refitted on all of B for validation.
Feature groups (each one measured by ablation):
  prior  per-class intercept + co-occurrence PMI of the class with the draft's other claims (from half-A
         references) + number of claims in the draft. Language prior only, no image.
  det    OWLv2 at the Look region: logit score of the claimed class, objectness; and the claim's best score
         anywhere in the image (s_image).
  look   excess attention mass of the region, attention entropy, IoU between the attention region and the
         detector's best box for the claim (attention/detector agreement).
  cap    captioner confidence (scripts/21): log p, entropy, margin, visual gain (log p with vs without the image),
         relative position.
Label: automatic reference proxy ('a reference mentions the class'). Metrics on thing/stuff claims.
Operating points are chosen on out-of-fold B scores: 'contradicted' if P(correct) < tc with tc the largest value
whose B precision for wrong claims is >= target; 'supported' if P(correct) >= ts with B precision >= 0.95
(selective classification, Geifman & El-Yaniv NeurIPS'17). Everything else is 'uncertain'.
"""
import argparse
from collections import Counter
from itertools import combinations

import numpy as np

import _common  # noqa: F401
from lcr import cache, calibrate, check, claims, config, data, grounding, pipeline
from lcr.vocab import CLASS_INDEX, CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--select", default="A_B_look2")
ap.add_argument("--report", default="A_validation_look2")
ap.add_argument("--targets", default="0.5,0.6,0.7")
ap.add_argument("--write", action="store_true")
ap.add_argument("--results", default="check_verifier.json", help="results file name (keep v2 results when re-running on newer Look records)")
ap.add_argument("--overwrite", action="store_true", help="explicitly replace existing learned-verifier pipeline files")
a = ap.parse_args()
assert "test" not in a.select and "test" not in a.report
P = config.CACHE / "pipeline"
Q = pipeline.class_queries()
TARGETS = [float(t) for t in a.targets.split(",")]
if a.write and not a.overwrite:
    existing = [P / f"{stem}{suffix}.json" for stem in (a.select, a.report)
                for suffix in ("_check2", "_check2vis") if (P / f"{stem}{suffix}.json").exists()]
    if existing:
        ap.error("output already exists; use --overwrite to replace: " + ", ".join(str(p) for p in existing))

# co-occurrence PMI from half-A references (other images than B / validation)
cnt, pair, n_img = Counter(), Counter(), 0
for iid in data.ids("A"):
    cl = claims.reference_classes(data.references(iid))
    n_img += 1
    cnt.update(cl)
    pair.update(frozenset(p) for p in combinations(sorted(cl), 2))


def pmi(x, y):
    pxy = (pair[frozenset((x, y))] + 0.5) / n_img
    return float(np.log(pxy / (((cnt[x] + 1) / n_img) * ((cnt[y] + 1) / n_img))))


GROUPS = {
    "prior": ["cooc_mean", "cooc_max", "n_claims"],
    "det": ["s_claim", "objectness", "s_image"],
    "region": ["s_claim", "objectness"],           # det split for H6: only the chosen region ...
    "image": ["s_image"],                          # ... or only the image-level detector score
    "look": ["excess", "att_entropy", "iou_att_det"],
    "cap": ["lp", "entropy", "margin", "vis_gain", "rel_pos"],
}


def load(stem, key):
    rows = cache.load_json(P / f"{stem}.json")
    sig = cache.load_json(P / f"{stem}_capsig.json")
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0])
    xs = []
    for ri, r in enumerate(rows):
        prop = owl[data.parse_id(r["id"])[1]]
        scores = check.score(prop, Q)
        others = [c["cls"] for c in r["claims"]]
        f30k = grounding.class_boxes(r["id"])
        for ci, c in enumerate(r["claims"]):
            rec = c[key]
            reg = rec["region"]
            s = sig.get(r["id"], {}).get(c["cls"])
            o = [pmi(c["cls"], k) for k in others if k != c["cls"]]
            f = dict(cooc_mean=np.mean(o) if o else 0.0, cooc_max=max(o) if o else 0.0, n_claims=len(others),
                     s_claim=float(calibrate.logit(rec["s_claim"])) if reg >= 0 else float(calibrate.logit(0.0)),
                     objectness=float(prop["objectness"][reg]) if reg >= 0 else 0.0,
                     s_image=float(calibrate.logit(rec["s_image"])),
                     excess=c["rec_att"]["excess"], att_entropy=c["att_entropy"], iou_att_det=c["iou_att_det"],
                     **({k: s[k] for k in GROUPS["cap"]} if s else {k: np.nan for k in GROUPS["cap"]}))
            xs.append(dict(ri=ri, ci=ci, id=r["id"], c=CLASS_INDEX[c["cls"]], kind=c["kind"], ok=bool(c["in_refs"]),
                           f30k=c["cls"] in f30k and len(f30k[c["cls"]]) > 0, fold=calibrate.fold_of(r["id"]),
                           feat=f, row=scores[reg] if reg >= 0 else None, cap_alts=c["blip_alts"],
                           refs=claims.reference_classes(data.references(r["id"]))))
    return rows, xs


def matrix(xs, names):
    X = np.array([[x["feat"][n] for n in names] for x in xs], np.float64).reshape(len(xs), len(names))
    return X


def fit_predict(B, V, names, use_class, lam_c=1.0, lam_w=1.0):
    XB, XV = matrix(B, names), matrix(V, names)
    cB, cV = np.array([x["c"] for x in B]), np.array([x["c"] for x in V])
    yB, fB = np.array([x["ok"] for x in B]), np.array([x["fold"] for x in B])
    oof = np.empty(len(B))
    for k in np.unique(fB):
        m = fB == k
        oof[m] = calibrate.FeatureLogistic(lam_c, lam_w, use_class).fit(XB[~m], cB[~m], yB[~m])(XB[m], cB[m])
    full = calibrate.FeatureLogistic(lam_c, lam_w, use_class).fit(XB, cB, yB)
    return oof, full(XV, cV), full


def operating_point(p, ok, target, sup_target=0.95):
    order = np.sort(np.unique(p))
    tc = -np.inf
    for t in order:                                   # largest tc with precision(wrong | p < tc) >= target
        m = p < t
        if m.sum() >= 20 and (~ok[m]).mean() >= target:
            tc = t
    ts = np.inf
    for t in order[::-1]:                             # smallest ts with precision(ok | p >= ts) >= sup_target
        m = p >= t
        if m.sum() >= 20 and ok[m].mean() >= sup_target:
            ts = t
    return float(tc), float(max(ts, tc))


def verdicts(p, tc, ts):
    return np.where(p >= ts, "supported", np.where(p < tc, "contradicted", "uncertain"))


def ap_wrong(p, ok):
    return calibrate.ap_wrong(p, ok)


def report(xs, p, ops):
    m = np.array([x["kind"] != "person" for x in xs])
    ok = np.array([x["ok"] for x in xs])[m]; f30 = np.array([x["f30k"] for x in xs])[m]; pm = p[m]
    out = dict(auroc=calibrate.auroc(pm, ok), auroc_f30k=calibrate.auroc(pm, f30), ap_wrong=ap_wrong(pm, ok),
               ece=calibrate.ece(pm, ok), log_loss=calibrate.log_loss(pm, ok), base_rate_wrong=float((~ok).mean()))
    for t, (tc, ts) in ops.items():
        out[f"op{t}"] = dict(tc=tc, ts=ts, **calibrate.verdict_quality(verdicts(pm, tc, ts), ok))
    return out


CONFIGS = [
    ("prior", ["prior"], True),
    ("det (no class prior)", ["det"], False),
    ("cap (no class prior)", ["cap"], False),
    ("prior+region", ["prior", "region"], True),
    ("prior+image", ["prior", "image"], True),
    ("prior+det", ["prior", "det"], True),
    ("prior+cap", ["prior", "cap"], True),
    ("prior+det+look", ["prior", "det", "look"], True),
    ("prior+det+cap", ["prior", "det", "cap"], True),
    ("det+cap (no class prior)", ["det", "cap"], False),
    ("all", ["prior", "det", "look", "cap"], True),
]
# written verifiers: main (visual + captioner + prior) and a visual-only variant for the Retell ablation
WRITE = {"_check2": "prior+det+cap", "_check2vis": "prior+det"}
results, chosen_out = {}, {}
for key in ("rec_att", "rec_det"):
    rowsB, B = load(a.select, key)
    rowsV, V = load(a.report, key)
    mB = np.array([x["kind"] != "person" for x in B])
    okB = np.array([x["ok"] for x in B])
    # shrinkage strengths: out-of-fold log-loss of the full model on B
    names_all = sum((GROUPS[g] for g in ["prior", "det", "look", "cap"]), [])
    best = None
    for lam_c in (0.3, 1.0, 3.0):
        for lam_w in (0.1, 1.0, 10.0):
            oof, _, _ = fit_predict(B, V, names_all, True, lam_c, lam_w)
            ll = calibrate.log_loss(oof, okB)
            if best is None or ll < best[0]:
                best = (ll, lam_c, lam_w)
    _, lam_c, lam_w = best
    print(f"\n=== {key}: lam_c={lam_c} lam_w={lam_w} (B out-of-fold log-loss {best[0]:.4f})")
    res = []
    for name, groups, use_class in CONFIGS:
        if key == "rec_det" and "look" in groups:
            continue                                   # B3 records must not see attention
        names = sum((GROUPS[g] for g in groups), [])
        oof, pv, model = fit_predict(B, V, names, use_class, lam_c, lam_w)
        ops = {t: operating_point(oof[mB], okB[mB], t) for t in TARGETS}
        r = dict(config=name, features=names, use_class=use_class, B=report(B, oof, ops), val=report(V, pv, ops),
                 weights=model.weights(names) if names else {})
        res.append(r)
        o = r["val"][f"op{TARGETS[0]}"]
        print(f"  {name:26s} B AUROC {r['B']['auroc']:.3f} | val AUROC {r['val']['auroc']:.3f} AP(wrong) "
              f"{r['val']['ap_wrong']:.3f} ECE {r['val']['ece']:.3f} | op{TARGETS[0]}: con {o['contradicted']:.3f} "
              f"cPrec {o['contradicted_precision']:.3f} rec.w {o['recall_wrong']:.3f} sup {o['supported']:.3f} "
              f"sPrec {o['supported_precision']:.3f}")
        for sfx, chosen_name in WRITE.items():
            if a.write and name == chosen_name:
                chosen_out[(key, sfx)] = (oof, pv, ops, name)
    results[key] = dict(lam_c=lam_c, lam_w=lam_w, configs=res)

if a.write:
    for sfx in WRITE:
        for stem in (a.select, a.report):
            rows = None
            for key in ("rec_att", "rec_det"):
                oof, pv, ops, name = chosen_out[(key, sfx)]
                tc, ts = ops[TARGETS[0]]
                rows_k, xs = load(stem, key)
                rows = rows or rows_k
                p = oof if stem == a.select else pv
                v = verdicts(p, tc, ts)
                for x, pp, vv in zip(xs, p, v):
                    rec = rows[x["ri"]]["claims"][x["ci"]][key]
                    alt = ""
                    if x["row"] is not None:          # suggestion: captioner candidate the detector supports most
                        cands = [k for k in x["cap_alts"] if k in CLASS_INDEX and k != CLASSES[x["c"]]]
                        if cands:
                            alt = max(cands, key=lambda k: x["row"][CLASS_INDEX[k]])
                    rec.update(check="v2", verifier=name, p_correct=float(pp), verdict=str(vv), alt=alt,
                               s_alt=float(x["row"][CLASS_INDEX[alt]]) if alt else 0.0, tc=tc, ts=ts)
            for r in rows:
                r["check_metadata"] = dict(script="scripts/22_check_verifier.py", verifier=WRITE[sfx],
                                           target_precision=TARGETS[0], label="reference proxy (automatic)")
            cache.save_json(rows, P / f"{stem}{sfx}.json")
            print("wrote", P / f"{stem}{sfx}.json")
cache.save_json(dict(select=a.select, report=a.report, targets=TARGETS, groups=GROUPS, results=results),
                config.RESULTS / a.results)
