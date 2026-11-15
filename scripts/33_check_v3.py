"""Check v3 selection and report (docs/CHECK_V3_PROTOKOL.md).

    python scripts/33_check_v3.py            # nested CV on half B, lock -> results/check_v3_selection.json
    python scripts/33_check_v3.py --report   # validation once -> results/check_v3.json, writes *_look4_check3.json
Needs scripts 21 (capsig), 31 (vlm) and 32 (cons) outputs for A_B_look4 and A_validation_look4.
Evaluation set as in Check v2: non-person claims (person is almost always correct). Labels are the automatic
reference proxy, not human labels.
"""
import sys
from collections import Counter
from itertools import combinations

import numpy as np
from scipy.special import expit, logit as _logit

import _common  # noqa: F401
from lcr import cache, calibrate, check, claims, config, data, pipeline
from lcr.vocab import CLASS_INDEX

P = config.CACHE / "pipeline"
STEMS = {"B": "A_B_look4", "val": "A_validation_look4"}
SEL = config.RESULTS / "check_v3_selection.json"
TARGETS = (0.5, 0.6, 0.7)
COVERAGES = (0.1, 0.2, 0.3)
LAMS = [(lc, lw) for lc in (0.3, 1.0, 3.0) for lw in (0.1, 1.0, 10.0)]
GBM_GRID = [(d, lr) for d in (2, 3) for lr in (0.05, 0.1)]
TOL = 0.001
CLOTHING = {"shirt", "jacket", "sweater", "hat", "helmet", "dress", "shorts", "pants", "suit", "uniform",
            "costume", "vest", "skirt", "scarf", "bikini", "swimsuit", "glove", "boot", "shoe", "glasses",
            "goggles", "mask", "headband", "bandanna"}
GROUPS = {
    "prior": ["cooc_mean", "cooc_max", "n_claims"],
    "det": ["s_claim", "objectness", "s_image"],
    "cap": ["lp", "entropy", "margin", "vis_gain", "rel_pos"],
    "look": ["excess", "att_entropy", "iou_att_det"],
    "vqa_yes": ["vqa_yes"], "vqa_ens": ["vqa_ens"], "vqa_word": ["vqa_word"],
    "itm": ["itm_full", "itm_drop"],
    "crop": ["itm_crop"],
    "cons": ["cons_sample", "cons_beam"],
}
BASE = ["prior", "det", "cap"]
N_PRIOR = len(GROUPS["prior"])


def lg(p):
    return float(_logit(np.clip(p, 1e-4, 1 - 1e-4)))


# --------------------------------------------------------------------------- features
_cnt, _pair, _n = Counter(), Counter(), 0
for _iid in data.ids("A"):
    _cl = claims.reference_classes(data.references(_iid)); _n += 1
    _cnt.update(_cl); _pair.update(frozenset(p) for p in combinations(sorted(_cl), 2))


def pmi(x, y):
    pxy = (_pair[frozenset((x, y))] + 0.5) / _n
    return float(np.log(pxy / (((_cnt[x] + 1) / _n) * ((_cnt[y] + 1) / _n))))


def load(stem, key="rec_att"):
    rows = cache.load_json(P / f"{stem}.json")
    sig = cache.load_json(P / f"{stem}_capsig.json")
    vlm = cache.load_json(P / f"{stem}_vlm.json")
    cons = cache.load_json(P / f"{stem}_cons.json")
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0]); Q = pipeline.class_queries()
    xs = []
    for ri, r in enumerate(rows):
        if not r["claims"]:
            continue
        prop = owl[data.parse_id(r["id"])[1]]; scores = check.score(prop, Q)
        others = [c["cls"] for c in r["claims"]]
        refs = claims.reference_classes(data.references(r["id"]))
        co = cons[r["id"]]
        for ci, c in enumerate(r["claims"]):
            rec = c[key]; reg = rec["region"]
            s = sig.get(r["id"], {}).get(c["cls"])
            v = vlm[r["id"]]["claims"][ci]
            assert v["cls"] == c["cls"]
            o = [pmi(c["cls"], k) for k in others if k != c["cls"]]
            f = dict(cooc_mean=np.mean(o) if o else 0.0, cooc_max=max(o) if o else 0.0, n_claims=len(others),
                     s_claim=lg(rec["s_claim"]) if reg >= 0 else lg(0.0),
                     objectness=float(prop["objectness"][reg]) if reg >= 0 else 0.0, s_image=lg(rec["s_image"]),
                     excess=c["rec_att"]["excess"], att_entropy=c["att_entropy"], iou_att_det=c["iou_att_det"],
                     vqa_yes=lg(v["vqa_yes"]), vqa_ens=lg(v["vqa_ens"]), vqa_word=lg(v["vqa_word"]),
                     itm_full=lg(v["itm_full"]), itm_drop=v["itm_drop"],
                     itm_crop=lg(v["itm_crop_att"] if key == "rec_att" else v["itm_crop_det"]),
                     cons_sample=co["sample"].get(c["cls"], 0.0), cons_beam=co["beam"].get(c["cls"], 0.0),
                     **({k: s[k] for k in GROUPS["cap"]} if s else {k: np.nan for k in GROUPS["cap"]}))
            alts = [k for k in c["blip_alts"] if k in CLASS_INDEX and k != c["cls"]]
            xs.append(dict(ri=ri, ci=ci, id=r["id"], c=CLASS_INDEX[c["cls"]], kind=c["kind"],
                           group="clothing" if c["cls"] in CLOTHING else c["kind"],
                           ok=bool(c["in_refs"]), fold=calibrate.fold_of(r["id"]), feat=f, alts=alts, refs=refs,
                           alt_owl={k: float(scores[reg][CLASS_INDEX[k]]) for k in alts} if reg >= 0 else {},
                           alt_vqa={k: vlm[r["id"]]["alts"].get(k, 0.0) for k in alts},
                           alt_cons={k: co["sample"].get(k, 0.0) for k in alts}))
    return rows, xs


def matrix(xs, names):
    return np.array([[x["feat"][n] for n in names] for x in xs], np.float64).reshape(len(xs), len(names))


# --------------------------------------------------------------------------- model classes
class Logistic:
    def __init__(self, hp):
        self.m = calibrate.FeatureLogistic(hp[0], hp[1], True)

    def fit(self, X, c, y):
        self.m.fit(X, c, y); return self

    def __call__(self, X, c):
        return self.m(X, c)


class Factorized:
    """HaloProbe-style factorisation: the class prior (shrunk class intercepts + prior features, lam_c) is fitted
    on its own and enters as a fixed logit offset; the remaining signals go into a class-balanced logistic without
    class terms (lam_w), i.e. an approximate likelihood ratio that the prior cannot absorb."""

    def __init__(self, hp):
        self.hp = hp

    def fit(self, X, c, y):
        from sklearn.linear_model import LogisticRegression
        self.prior = calibrate.FeatureLogistic(self.hp[0], 1.0, True).fit(X[:, :N_PRIOR], c, y)
        V = X[:, N_PRIOR:]
        cnt = np.sum(np.isfinite(V), 0)
        self.imp = np.divide(np.nansum(V, 0), cnt, out=np.zeros(V.shape[1]), where=cnt > 0)
        V = np.where(np.isnan(V), self.imp, V)
        self.mu, self.sd = V.mean(0), V.std(0) + 1e-9
        w = np.ones(len(y))
        for k in np.unique(c):
            mk = c == k
            for lab in (False, True):
                m = mk & (y == lab)
                if m.any():
                    w[m] = 0.5 * mk.sum() / m.sum()
        # sklearn: C * sum(loss) + ||w||^2 / 2  <=>  sum(loss) + lam_w ||w||^2 (as FeatureLogistic) with C = 1/(2 lam_w)
        self.vis = LogisticRegression(C=1.0 / (2 * self.hp[1]), max_iter=5000)
        self.vis.fit((V - self.mu) / self.sd, y, sample_weight=w / w.mean())
        return self

    def __call__(self, X, c):
        V = np.where(np.isnan(X[:, N_PRIOR:]), self.imp, X[:, N_PRIOR:])
        zp = _logit(np.clip(self.prior(X[:, :N_PRIOR], c), 1e-6, 1 - 1e-6))
        return expit(zp + self.vis.decision_function((V - self.mu) / self.sd))


class GBM:
    """Gradient boosting with the training-fold class prior (shrunk class intercept, as a logit) as a feature."""

    def __init__(self, hp):
        self.hp = hp

    def _z(self, X, c):
        return np.column_stack([_logit(np.clip(self.prior(np.zeros((len(c), 0)), c), 1e-6, 1 - 1e-6)), X])

    def fit(self, X, c, y):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.prior = calibrate.FeatureLogistic(1.0, 1.0, True).fit(np.zeros((len(y), 0)), c, y)
        self.m = HistGradientBoostingClassifier(max_depth=self.hp[0], learning_rate=self.hp[1], max_iter=200,
                                                early_stopping=False, random_state=config.SEED)
        self.m.fit(self._z(X, c), y)
        return self

    def __call__(self, X, c):
        return self.m.predict_proba(self._z(X, c))[:, 1]


MODELS = {"logistic": (Logistic, LAMS), "factorized": (Factorized, LAMS), "gbm": (GBM, GBM_GRID)}


# --------------------------------------------------------------------------- selection helpers
def operating_point(p, ok, target, sup_target=0.95):
    """As Check v2: largest tc with precision(wrong | p < tc) >= target, smallest ts with precision(ok) >= 0.95."""
    order = np.sort(np.unique(p)); tc = -np.inf; ts = np.inf
    for t in order:
        m = p < t
        if m.sum() >= 20 and (~ok[m]).mean() >= target:
            tc = t
    for t in order[::-1]:
        m = p >= t
        if m.sum() >= 20 and ok[m].mean() >= sup_target:
            ts = t
    return float(tc), float(max(ts, tc))


def cv_scores(X, c, y, folds, kind, hp):
    out = np.empty(len(y)); cls = MODELS[kind][0]
    for k in np.unique(folds):
        m = folds == k
        out[m] = cls(hp).fit(X[~m], c[~m], y[~m])(X[m], c[m])
    return out


def pick_hp(X, c, y, folds, em, kind):
    best = None
    for hp in MODELS[kind][1]:
        s = cv_scores(X, c, y, folds, kind, hp)
        ll = calibrate.log_loss(s[em], y[em])
        if best is None or ll < best[0]:
            best = (ll, hp, s)
    return best[1], best[2]


def nested(X, c, y, folds, em, kind):
    """Outer out-of-fold scores; hyper-parameters (inner 4-fold log-loss) and thresholds (inner out-of-fold
    scores) are chosen inside each outer training set only."""
    oof = np.empty(len(y)); thr = {t: {} for t in TARGETS}; chosen = []
    for f in range(5):
        tr, te = folds != f, folds == f
        hp, inner = pick_hp(X[tr], c[tr], y[tr], folds[tr], em[tr], kind)
        chosen.append(list(hp))
        oof[te] = MODELS[kind][0](hp).fit(X[tr], c[tr], y[tr])(X[te], c[te])
        for t in TARGETS:
            thr[t][f] = operating_point(inner[em[tr]], y[tr][em[tr]], t)
    return oof, thr, chosen


def verdicts(p, tc, ts):
    return np.where(p >= ts, "supported", np.where(p < tc, "contradicted", "uncertain"))


def fold_verdicts(p, folds, thr_t):
    v = np.empty(len(p), object)
    for f in range(5):
        m = folds == f; v[m] = verdicts(p[m], *thr_t[f])
    return v


def quality(p, ok, v=None):
    out = dict(n=int(len(ok)), auroc=calibrate.auroc(p, ok), ap_wrong=calibrate.ap_wrong(p, ok),
               ece=calibrate.ece(p, ok), log_loss=calibrate.log_loss(p, ok), base_rate_wrong=float((~ok).mean()))
    order = np.argsort(p, kind="stable")
    for cov in COVERAGES:
        k = int(round(cov * len(p)))
        out[f"prec_at_cov{int(cov * 100)}"] = float((~ok[order[:k]]).mean()) if k else float("nan")
    if v is not None:
        out.update(calibrate.verdict_quality(v, ok))
    return out


def families():
    fams = [("v2", BASE)]
    for vq in (None, "vqa_yes", "vqa_ens", "vqa_word"):
        for itm in (False, True):
            for cons in (False, True):
                extra = ([vq] if vq else []) + (["itm"] if itm else []) + (["cons"] if cons else [])
                if extra:
                    fams.append(("v2+" + "+".join(extra), BASE + extra))
    return [(n + s, g + a) for n, g in fams for s, a in (("", []), ("+crop", ["crop"]), ("+look", ["look"]))]


ABLATIONS = [("vqa_yes alone", ["vqa_yes"], False), ("itm alone", ["itm"], False), ("cons alone", ["cons"], False),
             ("prior", ["prior"], True), ("prior+vqa_yes", ["prior", "vqa_yes"], True),
             ("prior+itm", ["prior", "itm"], True), ("prior+cons", ["prior", "cons"], True)]


def cols_of(groups, names_all):
    names = [n for g in groups for n in GROUPS[g]]
    return names, [names_all.index(n) for n in names]


def run_family(X, names_all, groups, c, y, f, m, kind="logistic"):
    names, cols = cols_of(groups, names_all)
    oof, thr, chosen = nested(X[:, cols], c, y, f, m, kind)
    return dict(oof=oof, thr=thr, chosen=chosen, oof_log_loss=calibrate.log_loss(oof[m], y[m]))


def arrays(xs):
    return (np.array([x["c"] for x in xs]), np.array([x["ok"] for x in xs]), np.array([x["fold"] for x in xs]),
            np.array([x["kind"] != "person" for x in xs]))


def alt_rates(xs, mask):
    """Among reference-unsupported claims with captioner alternatives: how often the top-ranked alternative is in
    the references (automatic proxy)."""
    out = {}
    for rule in ("owl", "vqa", "cons"):
        n = hit = 0
        for x, mm in zip(xs, mask):
            sc = x[f"alt_{rule}"]
            if not mm or x["ok"] or not x["alts"] or not sc:
                continue
            k = max(x["alts"], key=lambda a: sc.get(a, -1)); n += 1; hit += k in x["refs"]
        out[rule] = dict(n=n, rate=hit / max(n, 1))
    return out


def main():
    if SEL.exists():
        raise FileExistsError(f"{SEL} exists: Check v3 selection is locked")
    _, B = load(STEMS["B"])
    names_all = sorted({n for g in GROUPS.values() for n in g})
    X = matrix(B, names_all); c, y, f, m = arrays(B)
    fams = families()
    print(len(fams), "feature families,", len(B), "B claims,", int(m.sum()), "evaluated", flush=True)
    res = [run_family(X, names_all, g, c, y, f, m) for _, g in fams]
    table = {}
    for (name, groups), r in zip(fams, res):
        q = quality(r["oof"][m], y[m], fold_verdicts(r["oof"], f, r["thr"][0.5])[m])
        table[name] = dict(groups=groups, oof_log_loss=r["oof_log_loss"], chosen_hp=r["chosen"], B=q)
        print(f"{name:26s} ll {r['oof_log_loss']:.4f} auroc {q['auroc']:.3f} ap {q['ap_wrong']:.3f} "
              f"cprec {q['contradicted_precision']:.3f} recw {q['recall_wrong']:.3f} p@20 {q['prec_at_cov20']:.3f}",
              flush=True)
    best_ll = min(v["oof_log_loss"] for v in table.values())
    near = [n for n, v in table.items() if v["oof_log_loss"] <= best_ll + TOL]
    feat_choice = min(near, key=lambda n: (len(table[n]["groups"]), table[n]["oof_log_loss"]))
    groups = table[feat_choice]["groups"]
    models = {"logistic": dict(oof_log_loss=table[feat_choice]["oof_log_loss"], B=table[feat_choice]["B"])}
    for kind in ("factorized", "gbm"):
        r = run_family(X, names_all, groups, c, y, f, m, kind)
        models[kind] = dict(oof_log_loss=r["oof_log_loss"], chosen_hp=r["chosen"],
                            B=quality(r["oof"][m], y[m], fold_verdicts(r["oof"], f, r["thr"][0.5])[m]))
        print(f"model {kind:10s} ll {r['oof_log_loss']:.4f} auroc {models[kind]['B']['auroc']:.3f}", flush=True)
    best_m = min(v["oof_log_loss"] for v in models.values())
    model_choice = next(k for k in ("logistic", "factorized", "gbm") if models[k]["oof_log_loss"] <= best_m + TOL)
    abl = {}
    for name, g, use_class in ABLATIONS:
        _, cols = cols_of(g, names_all); oof = np.empty(len(y))
        for k in range(5):
            mm = f == k
            oof[mm] = calibrate.FeatureLogistic(1.0, 1.0, use_class).fit(X[~mm][:, cols], c[~mm], y[~mm])(
                X[mm][:, cols], c[mm])
        abl[name] = quality(oof[m], y[m])
    alt = alt_rates(B, m)
    alt_choice = max(alt, key=lambda k: alt[k]["rate"])
    cache.save_json(dict(protocol="docs/CHECK_V3_PROTOKOL.md", tolerance=TOL, families=table,
                         feature_choice=feat_choice, models=models, model_choice=model_choice, ablations_B=abl,
                         alternative_B=alt, alternative_choice=alt_choice), SEL)
    print("LOCKED feature family", feat_choice, "| model", model_choice, "| alternative", alt_choice, flush=True)


def bootstrap(ids, fn, n_boot=2000):
    uniq = sorted(set(ids)); pos = {u: [] for u in uniq}
    for i, u in enumerate(ids):
        pos[u].append(i)
    pos = [np.array(pos[u]) for u in uniq]
    rng = np.random.default_rng(config.SEED); d = []
    for _ in range(n_boot):
        d.append(fn(np.concatenate([pos[j] for j in rng.integers(0, len(uniq), len(uniq))])))
    d = np.array(d, float)
    return [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))]


def report():
    sel = cache.load_json(SEL)
    out_path = config.RESULTS / "check_v3.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists: Check v3 validation already reported")
    groups, kind = sel["families"][sel["feature_choice"]]["groups"], sel["model_choice"]
    rowsB, B = load(STEMS["B"]); rowsV, V = load(STEMS["val"])
    names = [n for g in groups for n in GROUPS[g]]
    XB, XV = matrix(B, names), matrix(V, names)
    cB, yB, fB, mB = arrays(B); cV, yV, _, mV = arrays(V)
    oofB, thrB, _ = nested(XB, cB, yB, fB, mB, kind)
    ops = {t: operating_point(oofB[mB], yB[mB], t) for t in TARGETS}
    hp, _ = pick_hp(XB, cB, yB, fB, mB, kind)
    model = MODELS[kind][0](hp).fit(XB, cB, yB)
    pV = model(XV, cV)
    v2rows = cache.load_json(P / f"{STEMS['val']}_check2.json")
    pV2 = np.array([v2rows[x["ri"]]["claims"][x["ci"]]["rec_att"]["p_correct"] for x in V])
    vV2 = np.array([v2rows[x["ri"]]["claims"][x["ci"]]["rec_att"]["verdict"] for x in V])
    gV = np.array([x["group"] for x in V])
    rep = dict(feature_choice=sel["feature_choice"], model=kind, hp=list(hp), features=names,
               ops={str(t): list(o) for t, o in ops.items()}, groups={},
               label="automatic reference proxy (not human labels); non-person claims")
    for g in ("all", "clothing", "thing", "stuff"):
        mm = mV if g == "all" else mV & (gV == g)
        rep["groups"][g] = dict(v3={str(t): quality(pV[mm], yV[mm], verdicts(pV[mm], *ops[t])) for t in TARGETS},
                                v2=quality(pV2[mm], yV[mm], vV2[mm]))
    idx = np.where(mV)[0]; ids = [V[i]["id"] for i in idx]
    p3, p2, yy, v2 = pV[idx], pV2[idx], yV[idx], vV2[idx]
    v3 = verdicts(p3, *ops[0.5]); al = np.arange(len(yy))

    def cprec(v, i):
        s = v[i] == "contradicted"
        return (~yy[i][s]).mean() if s.any() else np.nan

    def recw(v, i):
        w = ~yy[i]
        return ((v[i] == "contradicted") & w).sum() / max(w.sum(), 1)

    def pcov(p, i, cov=0.2):
        o = np.argsort(p[i], kind="stable")[:int(round(cov * len(i)))]
        return (~yy[i][o]).mean()

    diffs = dict(auroc=lambda i: calibrate.auroc(p3[i], yy[i]) - calibrate.auroc(p2[i], yy[i]),
                 ap_wrong=lambda i: calibrate.ap_wrong(p3[i], yy[i]) - calibrate.ap_wrong(p2[i], yy[i]),
                 prec_at_cov20=lambda i: pcov(p3, i) - pcov(p2, i),
                 contradicted_precision=lambda i: cprec(v3, i) - cprec(v2, i),
                 recall_wrong=lambda i: recw(v3, i) - recw(v2, i))
    rep["bootstrap_v3_minus_v2"] = {k: dict(diff=float(fn(al)), ci95=bootstrap(ids, fn)) for k, fn in diffs.items()}
    rep["alternative_val"] = alt_rates(V, mV)
    rep["coefficients"] = model.m.weights(names) if kind == "logistic" else None
    cache.save_json(rep, out_path)
    write_records(rowsB, B, oofB, fold_verdicts(oofB, fB, thrB[0.5]), rowsV, V, pV, verdicts(pV, *ops[0.5]),
                  sel, names, kind, ops)
    g = rep["groups"]["all"]
    keys = ("auroc", "ap_wrong", "prec_at_cov20", "contradicted_precision", "recall_wrong")
    print("v2", {k: round(g["v2"][k], 3) for k in keys})
    print("v3", {k: round(g["v3"]["0.5"][k], 3) for k in keys})
    for k, v in rep["bootstrap_v3_minus_v2"].items():
        print(f"  {k}: {v['diff']:+.4f} CI {v['ci95'][0]:+.4f} {v['ci95'][1]:+.4f}")


def write_records(rowsB, B, pB, vB, rowsV, V, pV, vV, sel, names, kind, ops):
    rule = sel["alternative_choice"]
    for stem, rows, xs, p, v, how in (
            (STEMS["B"], rowsB, B, pB, vB, "nested out-of-fold (hyper-parameters and thresholds per outer fold)"),
            (STEMS["val"], rowsV, V, pV, vV, "model fitted on all of B, thresholds from B nested out-of-fold")):
        for x, pp, vv in zip(xs, p, v):
            rec = rows[x["ri"]]["claims"][x["ci"]]["rec_att"]
            sc = x[f"alt_{rule}"]
            alt = max(x["alts"], key=lambda a: sc.get(a, -1)) if x["alts"] and sc else ""
            rec.update(check="v3", verifier=f"{sel['feature_choice']}:{kind}", p_correct=float(pp), verdict=str(vv),
                       alt=alt, alt_rule=rule, alt_score=float(sc.get(alt, 0.0)) if alt else 0.0)
        for r in rows:
            r["check_metadata"] = dict(script="scripts/33_check_v3.py", features=names, model=kind,
                                       target_precision=0.5, thresholds=list(ops[0.5]), scores=how,
                                       label="reference proxy (automatic)")
        cache.save_json(rows, P / f"{stem}_check3.json")


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
