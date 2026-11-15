"""Check v2 (docs/LITERATUR_TARAMASI.md 6-8, 8a; Codex direction C0-C4): calibrate, choose on B, report on validation.

    python scripts/20_check.py                                   # Look v2 records, evaluate everything
    python scripts/20_check.py --write                           # also write *_check2cal.json pipeline files

C0  cross-fitting: B records are scored by calibrators fitted on the other 4 folds of B (folds by image);
    validation records by a calibrator fitted on all of B. Thresholds are chosen on the out-of-fold B scores.
C1  score transform: raw OWLv2 probability / per-class percentile / per-class logistic (offset, offset+slope).
C2  alternative for 'contradicted': detector top-6 + captioner candidates by raw score (round 1) / the same by
    calibrated score / restricted to the claim's kind / captioner candidates only.
C3  decision rule: (a) supported p>=ts, contradicted p<tc AND p_alt>=ta; (b) contradicted p<tc (alternative only
    a suggestion). Thresholds maximise informedness J on B (out-of-fold); a second operating point keeps
    contradicted precision >= 0.5 on B (selective classification, Geifman & El-Yaniv NeurIPS'17).
C4  second proxy: does Flickr30k Entities have a box of the claimed class in the image? (reported, never fitted)
Labels are the automatic reference proxy ('a reference mentions the class'), thing/stuff claims only for metrics.
The same procedure is applied to the detector-region records (B3) for a like-for-like comparison.
"""
import argparse
import itertools

import numpy as np

import _common  # noqa: F401
from lcr import cache, calibrate, check, claims, config, data, grounding, pipeline
from lcr.vocab import CLASS_INDEX, CLASSES, KIND

ap = argparse.ArgumentParser()
ap.add_argument("--select", default="A_B_look2")
ap.add_argument("--report", default="A_validation_look2")
ap.add_argument("--write", action="store_true", help="write <stem>_check2cal.json (separate from script 22's learned verifier)")
ap.add_argument("--overwrite", action="store_true", help="explicitly replace existing calibration-only pipeline files")
a = ap.parse_args()
assert "test" not in a.select and "test" not in a.report, "the test split is not used during development"
Q = pipeline.class_queries()
P = config.CACHE / "pipeline"
if a.write and not a.overwrite:
    existing = [P / f"{stem}_check2cal.json" for stem in (a.select, a.report)
                if (P / f"{stem}_check2cal.json").exists()]
    if existing:
        ap.error("output already exists; use --overwrite to replace: " + ", ".join(str(p) for p in existing))
TRANSFORMS = ["raw", "percentile", "offset", "offset_slope"]
ALTS = ["det6cap_raw", "det6cap_cal", "samekind_cal", "cap_cal"]
LAMS = [0.1, 0.3, 1.0, 3.0, 10.0]


def load(stem, key):
    rows = cache.load_json(P / f"{stem}.json")
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0])
    out = []
    for ri, r in enumerate(rows):
        prop = owl[data.parse_id(r["id"])[1]]
        scores = check.score(prop, Q)
        refs = claims.reference_classes(data.references(r["id"]))
        f30k = grounding.class_boxes(r["id"])
        for ci, c in enumerate(r["claims"]):
            reg = c[key]["region"]
            row = scores[reg].astype(np.float64) if reg >= 0 else None
            det6 = [CLASSES[j] for j in np.argsort(-row)[:7]] if row is not None else []
            out.append(dict(ri=ri, ci=ci, id=r["id"], cls=c["cls"], c=CLASS_INDEX[c["cls"]], kind=c["kind"],
                            ok=bool(c["in_refs"]), f30k=c["cls"] in f30k and len(f30k[c["cls"]]) > 0,
                            refs=refs, row=row, det6=[x for x in det6 if x != c["cls"]][:6],
                            cap=[x for x in c["blip_alts"] if x != c["cls"]], fold=calibrate.fold_of(r["id"])))
    return rows, out


def fit_arrays(xs):
    m = [x for x in xs if x["row"] is not None]
    return (np.array([x["row"][x["c"]] for x in m]), np.array([x["c"] for x in m]),
            np.array([x["ok"] for x in m]), np.array([x["fold"] for x in m]), m)


def calibrators(train, transform, lam):
    s, c, y, f, _ = fit_arrays(train)
    full = calibrate.make(transform, lam).fit(s, c, y)
    per_fold = {k: calibrate.make(transform, lam).fit(s[f != k], c[f != k], y[f != k]) for k in np.unique(f)}
    return full, per_fold


def choose_alt(x, cal, variant):
    if x["row"] is None:
        return "", 0.0, 0.0
    cands = list(dict.fromkeys(x["cap"] if variant == "cap_cal" else x["det6"] + x["cap"]))
    if variant == "samekind_cal":
        cands = [k for k in cands if KIND[k] == x["kind"]]
    if not cands:
        return "", 0.0, 0.0
    raw = np.array([x["row"][CLASS_INDEX[k]] for k in cands])
    cal_v = cal(raw, np.array([CLASS_INDEX[k] for k in cands]))
    j = int(np.argmax(raw if variant == "det6cap_raw" else cal_v))
    return cands[j], float(raw[j]), float(cal_v[j])


def score_all(xs, full, per_fold, alt_variant, oof):
    """p_claim, p_alt (calibrated) and the chosen alternative for every record."""
    out = []
    for x in xs:
        cal = per_fold[x["fold"]] if oof else full
        if x["row"] is None:
            out.append(dict(p=np.nan, alt="", s_alt=0.0, p_alt=0.0))
            continue
        p = float(cal(x["row"][x["c"]], x["c"])[0])
        alt, s_alt, p_alt = choose_alt(x, cal, alt_variant)
        out.append(dict(p=p, alt=alt, s_alt=s_alt, p_alt=p_alt))
    return out


def verdicts(sc, rule, ts, tc, ta):
    v = []
    for s in sc:
        if not np.isfinite(s["p"]):
            v.append("uncertain")
        elif s["p"] >= ts:
            v.append("supported")
        elif s["p"] < tc and (rule == "b" or (s["alt"] and s["p_alt"] >= ta)):
            v.append("contradicted")
        else:
            v.append("uncertain")
    return np.array(v)


def best_thresholds(sc, ok, rule, min_con_precision=None):
    """Max-J thresholds (quantile grids of the calibrated scores). J separates into a supported and a
    contradicted part because the two sets are disjoint whenever tc <= ts."""
    p = np.array([s["p"] for s in sc]); pa = np.array([s["p_alt"] for s in sc])
    fin = np.isfinite(p)
    p, pa, ok = p[fin], pa[fin], np.asarray(ok)[fin]
    q = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    qa = np.unique(np.quantile(pa, np.linspace(0.0, 0.95, 20))) if rule == "a" else np.array([-np.inf])
    r = lambda m, g: m[:, g].mean(axis=1) if g.any() else np.zeros(len(m))
    sup = p[None, :] >= q[:, None]                                    # (G, N)
    j_sup = r(sup, ok) - r(sup, ~ok)
    con = (p[None, None, :] < q[:, None, None]) & (pa[None, None, :] >= qa[None, :, None])   # (G, A, N)
    cm = con.reshape(-1, len(p))
    j_con = (r(cm, ~ok) - r(cm, ok)).reshape(len(q), len(qa))
    prec = np.where(cm.sum(1) > 0, (cm & ~ok[None]).sum(1) / np.maximum(cm.sum(1), 1), 0).reshape(len(q), len(qa))
    best = None
    for i_s, i_c in itertools.product(range(len(q)), range(len(q))):
        if q[i_c] > q[i_s]:
            continue
        cand = j_con[i_c]
        if min_con_precision is not None:
            cand = np.where(prec[i_c] >= min_con_precision, cand, -np.inf)
        i_a = int(np.argmax(cand))
        j = j_sup[i_s] + cand[i_a]
        if best is None or j > best[0]:
            best = (float(j), float(q[i_s]), float(q[i_c]), float(qa[i_a]))
    return dict(J_select=best[0], ts=best[1], tc=best[2], ta=best[3])


def ts_mask(xs):
    return np.array([x["kind"] != "person" for x in xs])


def evaluate(xs, sc, rule, th):
    m = ts_mask(xs)
    p = np.array([s["p"] for s in sc])[m]
    ok = np.array([x["ok"] for x in xs])[m]
    f30 = np.array([x["f30k"] for x in xs])[m]
    fin = np.isfinite(p)
    v = verdicts([s for s, k in zip(sc, m) if k], rule, th["ts"], th["tc"], th["ta"])
    wrong = ~ok
    con = v == "contradicted"
    alts = [s["alt"] for s, k in zip(sc, m) if k]
    refs = [x["refs"] for x, k in zip(xs, m) if k]
    useful = [alt in rf for alt, rf, w, c in zip(alts, refs, wrong, con) if w and c and alt]
    return dict(auroc=calibrate.auroc(p[fin], ok[fin]), auroc_f30k=calibrate.auroc(p[fin], f30[fin]),
                ece=calibrate.ece(p[fin], ok[fin]) if np.nanmax(p) <= 1 else float("nan"),
                log_loss=calibrate.log_loss(p[fin], ok[fin]),
                no_region=float((~fin).mean()),
                **calibrate.verdict_quality(v, ok),
                J_f30k=calibrate.verdict_quality(v, f30)["J"],
                alt_in_refs_among_correct_contradictions=float(np.mean(useful)) if useful else float("nan"))


def run_config(trainB, valx, transform, lam, alt_variant, rule, cals=None):
    full, per_fold = cals or calibrators(trainB, transform, lam)
    scB = score_all(trainB, full, per_fold, alt_variant, oof=True)
    scV = score_all(valx, full, per_fold, alt_variant, oof=False)
    mB = ts_mask(trainB)
    okB = np.array([x["ok"] for x in trainB])[mB]
    th = best_thresholds([s for s, k in zip(scB, mB) if k], okB, rule)
    return dict(th=th, B=evaluate(trainB, scB, rule, th), val=evaluate(valx, scV, rule, th)), scB, scV, (full, per_fold)


def choose_lam(trainB, transform):
    if transform not in ("offset", "offset_slope"):
        return 1.0
    s, c, y, f, _ = fit_arrays(trainB)
    losses = {lam: calibrate.log_loss(calibrate.cross_fit(transform, s, c, y, f, lam), y) for lam in LAMS}
    return min(losses, key=losses.get)


results = {}
records_out = {}
for key in ("rec_att", "rec_det"):
    rowsB, B = load(a.select, key)
    rowsV, V = load(a.report, key)
    print(f"\n=== {key}: B {len(B)} claims, validation {len(V)} claims")
    # round-1 reference: raw score, round-1 alternative, rule (a), thresholds from script 11 on the same records
    tag = "att_look2" if key == "rec_att" else "det"
    th11 = cache.load_json(config.RESULTS / f"thresholds_proxy_{tag}.json")
    base, _, _, cals_raw = run_config(B, V, "raw", 1.0, "det6cap_raw", "a")
    scV_raw = score_all(V, *cals_raw, "det6cap_raw", oof=False)
    th_r1 = dict(ts=th11["t_support"], tc=th11["t_contra"], ta=th11["t_support"])
    round1 = dict(th=th_r1, val=evaluate(V, scV_raw, "a", th_r1), note=f"script 11 thresholds ({tag})")
    grid = []
    for transform in TRANSFORMS:
        lam = choose_lam(B, transform)
        cals = calibrators(B, transform, lam)
        for alt_variant, rule in itertools.product(ALTS, ("a", "b")):
            if rule == "b" and alt_variant != "det6cap_cal" and transform != "raw":
                continue                                   # rule (b) ignores the alternative for the verdict
            res, _, _, _ = run_config(B, V, transform, lam, alt_variant, rule, cals)
            grid.append(dict(transform=transform, lam=lam, alt=alt_variant, rule=rule, **res))
            print(f"  {transform:12s} lam={lam:<4} {alt_variant:13s} rule {rule}: "
                  f"B J {res['B']['J']:.3f} | val J {res['val']['J']:.3f} AUROC {res['val']['auroc']:.3f} "
                  f"con-prec {res['val']['contradicted_precision']:.3f} con {res['val']['contradicted']:.3f} "
                  f"unc {res['val']['uncertain']:.3f}")
    results[key] = dict(round1=round1, grid=grid)

# choose ONE configuration on B (attention records, out-of-fold J); apply the same configuration to B3
att = results["rec_att"]["grid"]
best = max(att, key=lambda g: g["B"]["J"])
cfg = dict(transform=best["transform"], alt=best["alt"], rule=best["rule"])
print(f"\nchosen on B (attention records): {cfg}")
final = {}
for key in ("rec_att", "rec_det"):
    rowsB, B = load(a.select, key)
    rowsV, V = load(a.report, key)
    lam = choose_lam(B, cfg["transform"])
    res, scB, scV, cals = run_config(B, V, cfg["transform"], lam, cfg["alt"], cfg["rule"])
    # second operating point: contradicted precision >= 0.5 on B
    mB = ts_mask(B)
    okB = np.array([x["ok"] for x in B])[mB]
    th_p = best_thresholds([s for s, k in zip(scB, mB) if k], okB, cfg["rule"], min_con_precision=0.5)
    prec_op = dict(th=th_p, B=evaluate(B, scB, cfg["rule"], th_p), val=evaluate(V, scV, cfg["rule"], th_p))
    final[key] = dict(lam=lam, j_opt=res, precision_target_0_5=prec_op, round1=results[key]["round1"])
    if a.write:
        for stem, rows, xs, sc in ((a.select, rowsB, B, scB), (a.report, rowsV, V, scV)):
            # Both record types must be updated on the same rows. Reloading for
            # rec_det previously discarded the rec_att updates from this loop.
            rows = records_out.setdefault(stem, rows)
            v = verdicts(sc, cfg["rule"], **{k: res["th"][k] for k in ("ts", "tc", "ta")})
            for x, s, vv in zip(xs, sc, v):
                rec = rows[x["ri"]]["claims"][x["ci"]][key]
                rec.update(check="v2", p_claim=s["p"] if np.isfinite(s["p"]) else None,
                           alt=s["alt"], s_alt=s["s_alt"], p_alt=s["p_alt"], verdict=str(vv))
if a.write:
    meta = dict(config=cfg, thresholds_att=final["rec_att"]["j_opt"]["th"], thresholds_det=final["rec_det"]["j_opt"]["th"],
                label="reference proxy (automatic)", script="scripts/20_check.py")
    for stem, rows in records_out.items():
        for r in rows:
            r["check_metadata"] = meta
        cache.save_json(rows, P / f"{stem}_check2cal.json")
        print("wrote", P / f"{stem}_check2cal.json")

out = dict(select=a.select, report=a.report, chosen=cfg, final=final,
           grid_att=results["rec_att"]["grid"], grid_det=results["rec_det"]["grid"],
           label="labels are the automatic reference proxy; thing/stuff claims; B scored out-of-fold")
cache.save_json(out, config.RESULTS / "check_v2.json")
print(f"\n{'':34s} {'J':>6s} {'AUROC':>6s} {'ECE':>6s} {'sup':>5s} {'con':>5s} {'unc':>5s} {'sPrec':>6s} {'cPrec':>6s} {'rec.w':>6s} {'J_f30k':>6s} {'altOK':>6s}")
for key in ("rec_att", "rec_det"):
    for lbl, r in (("round 1 (raw, script-11 thr.)", final[key]["round1"]["val"]),
                   ("Check v2, J-optimal", final[key]["j_opt"]["val"]),
                   ("Check v2, con-precision>=0.5", final[key]["precision_target_0_5"]["val"])):
        print(f"{key[4:]:4s} {lbl:29s} {r['J']:6.3f} {r['auroc']:6.3f} {r['ece']:6.3f} {r['supported']:5.2f} "
              f"{r['contradicted']:5.2f} {r['uncertain']:5.2f} {r['supported_precision']:6.3f} "
              f"{r['contradicted_precision']:6.3f} {r['recall_wrong']:6.3f} {r['J_f30k']:6.3f} "
              f"{r['alt_in_refs_among_correct_contradictions']:6.3f}")
