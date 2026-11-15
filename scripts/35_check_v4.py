"""Check v4: the original region-based rule, improved inside its own logic (docs/CHECK_V4_PROTOKOL.md).

    python scripts/35_check_v4.py            # 144 configurations, 5-fold image CV on half B, lock
    python scripts/35_check_v4.py --report   # validation once -> results/check_v4.json, *_look4_check4.json
Only OWLv2 evidence inside the Look v4 region (and the claim's image-level score, already in the original record).
No language prior, no captioner signals, no second verifier. Labels: automatic reference proxy; non-person claims.
"""
import itertools
import sys

import numpy as np
from scipy.special import logit as _logit

import _common  # noqa: F401
from lcr import cache, calibrate, config, data, look
from lcr.vocab import CLASS_INDEX, CLASSES, KIND, THINGS

P = config.CACHE / "pipeline"
STEMS = {"B": "A_B_look4", "val": "A_validation_look4"}
SEL = config.RESULTS / "check_v4_selection.json"
CON_TARGET, MIN_N = 0.5, 20
REPORT_TARGETS = (0.5, 0.6, 0.7)
GRID = np.linspace(0.01, 0.99, 99)
DEFAULT = dict(Q="Q1", C="C1", A="A1", R="R1", G="G0", E="E0")
OPTIONS = dict(Q=("Q1", "Q2"), C=("C1", "C2"), A=("A1", "A2", "A3"), R=("R1", "R2"), G=("G0", "G1"),
               E=("E0", "E1", "E2"))
E_Q = {"E0": None, "E1": 0.10, "E2": 0.20}
QUERIES = {"Q1": "class_queries_single.npy", "Q2": "class_queries_ensemble.npy"}
CLOTHING = {"shirt", "jacket", "sweater", "hat", "helmet", "dress", "shorts", "pants", "suit", "uniform",
            "costume", "vest", "skirt", "scarf", "bikini", "swimsuit", "glove", "boot", "shoe", "glasses",
            "goggles", "mask", "headband", "bandanna"}

# A3 substitute groups: the comment headings of lcr/vocab.py THINGS, fixed by hand; stuff is one group.
_GROUP_STARTS = [("animal", "dog"), ("vehicle", "bicycle"), ("play", "ball"), ("clothing", "shirt"),
                 ("animal_gear", "collar"), ("handheld", "camera"), ("structure", "fence"), ("plant", "tree")]
GROUP = {}
_names = list(THINGS)
for gi, (g, start) in enumerate(_GROUP_STARTS):
    end = _names.index(_GROUP_STARTS[gi + 1][1]) if gi + 1 < len(_GROUP_STARTS) else len(_names)
    for n in _names[_names.index(start):end]:
        GROUP[n] = g
for c in CLASSES:
    GROUP.setdefault(c, KIND[c])                 # stuff -> "stuff", person -> "person"
CLS_KIND = np.array([KIND[c] for c in CLASSES])
CLS_GROUP = np.array([GROUP[c] for c in CLASSES])


def eligible(claim, a):
    """Boolean mask over classes that may serve as the alternative for `claim`."""
    m = np.ones(len(CLASSES), bool); m[CLASS_INDEX[claim]] = False
    if a == "A1":
        return m
    m &= CLS_KIND != "person"
    if a == "A2":
        return m & (CLS_KIND == KIND[claim])
    return m & (CLS_GROUP == GROUP[claim])


def logits(prop, q):
    e = prop["embeds"].astype(np.float32)
    return (e @ q.T + prop["shift"].astype(np.float32)[:, None]) * prop["scale"].astype(np.float32)[:, None]


def calib_stats(q):
    """Label-free per-class statistics: mean/sd of the per-image max logit over half-B images."""
    owl = cache.OwlCache("train")
    mx = np.array([logits(owl[data.parse_id(i)[1]], q).max(0) for i in data.ids("B")])
    return mx.mean(0), mx.std(0) + 1e-6


def scale(lg, c, stats):
    if c == "C1":
        return 1.0 / (1.0 + np.exp(-lg))
    mu, sd = stats
    return (lg - mu) / sd


def features(stem):
    """Per non-person claim and per (Q, C, R, A) option: claim score, alternative score, image score."""
    rows = cache.load_json(P / f"{stem}.json")
    maps = np.load(P / f"{stem}.maps.npz"); owner = {tuple(o): k for k, o in enumerate(maps["owner"])}
    owl = cache.OwlCache(data.parse_id(rows[0]["id"])[0])
    qs = {k: np.load(config.CACHE / "owl" / f) for k, f in QUERIES.items()}
    stats = {k: calib_stats(q) for k, q in qs.items()}
    xs, mism = [], 0
    for ri, r in enumerate(rows):
        prop = None
        for ci, c in enumerate(r["claims"]):
            if c["kind"] == "person":
                continue
            prop = prop or owl[data.parse_id(r["id"])[1]]
            rec = c["rec_att"]; reg = rec["region"]
            x = dict(ri=ri, ci=ci, id=r["id"], cls=c["cls"], kind=c["kind"], ok=bool(c["in_refs"]),
                     group="clothing" if c["cls"] in CLOTHING else c["kind"], fold=calibrate.fold_of(r["id"]),
                     excess=float(rec["excess"]), has_region=reg >= 0, v=dict(), live=rec["verdict"])
            if reg >= 0:
                att = maps["maps"][owner[(ri, ci)]]
                idx, rs, _ = look.select_frozen(att, prop["boxes"].astype(np.float32),
                                                prop["objectness"].astype(np.float32), c["kind"])
                mism += int(idx != reg)
                top = np.argsort(-np.where(np.isfinite(rs), rs, -np.inf))[:3]
                w = np.clip(rs[top], 0, None); w = w / w.sum() if w.sum() > 0 else np.full(len(top), 1 / len(top))
                k = CLASS_INDEX[c["cls"]]
                for qn, q in qs.items():
                    lg = logits(prop, q)
                    for cn in ("C1", "C2"):
                        S = scale(lg, cn, stats[qn])
                        img = float(S[:, k].max())
                        for rn, row in (("R1", S[reg]), ("R2", w @ S[top])):
                            for an in ("A1", "A2", "A3"):
                                el = eligible(c["cls"], an)
                                j = int(np.argmax(np.where(el, row, -np.inf)))
                                x["v"][(qn, cn, rn, an)] = (float(row[k]), float(row[j]), CLASSES[j], img)
            xs.append(x)
    print(stem, len(xs), "non-person claims; region index mismatches vs records:", mism, flush=True)
    return rows, xs


# --------------------------------------------------------------------------- rule and thresholds
def arrays(xs, cfg):
    key = (cfg["Q"], cfg["C"], cfg["R"], cfg["A"])
    lo = -np.inf
    sc = np.array([x["v"][key][0] if x["has_region"] else lo for x in xs])
    sa = np.array([x["v"][key][1] if x["has_region"] else lo for x in xs])
    si = np.array([x["v"][key][3] if x["has_region"] else lo for x in xs])
    return sc, sa, si


def gate(xs, cfg, train_mask):
    q = E_Q[cfg["E"]]
    ex = np.array([x["excess"] for x in xs]); kinds = np.array([x["kind"] for x in xs])
    reg = np.array([x["has_region"] for x in xs])
    if q is None:
        return reg
    g = reg.copy()
    for k in ("thing", "stuff"):
        m = kinds == k
        t = np.quantile(ex[m & train_mask & reg], q)
        g &= ~(m & (ex < t))
    return g


def rule(sc, sa, si, g, ts, tc, guard, ta=None):
    """Original verdict form. ta (alternative threshold) = ts, except when no supported threshold is feasible
    (ts = inf): then the top grid value is used so that contradiction stays defined."""
    ta = ts if ta is None else ta
    con = g & (sc < tc) & (sa >= ta)
    if guard:
        con &= si < ta
    sup = g & (sc >= ts)
    return np.where(sup, "supported", np.where(con, "contradicted", "uncertain"))


def fit_thresholds(sc, sa, si, g, ok, guard, con_target=CON_TARGET):
    """Amendment 1 (docs/CHECK_V4_PROTOKOL.md §5): joint grid over (tau_s, tau_c) maximising the three-way
    informedness J on the fitting data, subject to contradicted precision >= con_target (>= MIN_N decisions);
    "no contradictions" (tau_c = -inf) is always allowed. tau_s is the shared presence threshold for the claim and
    the alternative, as in the original rule. Returns (tau_s, tau_c, tau_alt = tau_s)."""
    vals = sc[g & np.isfinite(sc)]
    n_ok, n_w = max(ok.sum(), 1), max((~ok).sum(), 1)
    if len(vals) < MIN_N:
        return np.inf, -np.inf, np.inf
    grid = np.unique(np.quantile(vals, GRID))
    best = (-np.inf, np.inf, -np.inf)
    for ts in grid:
        sup = g & (sc >= ts)
        j_s = (sup & ok).sum() / n_ok - (sup & ~ok).sum() / n_w
        if j_s > best[0]:
            best = (j_s, ts, -np.inf)
        cand = g & (sc < ts) & (sa >= ts)
        if guard:
            cand &= si < ts
        if cand.sum() < MIN_N:
            continue
        order = np.argsort(sc[cand]); ss = sc[cand][order]; cw = np.cumsum(~ok[cand][order])
        tcs = grid[grid <= ts]
        k = np.searchsorted(ss, tcs, "left")             # decisions with sc < tc
        valid = k >= MIN_N
        if not valid.any():
            continue
        kk = k[valid]; w = cw[kk - 1]
        feas = w / kk >= con_target
        if not feas.any():
            continue
        j = j_s + w / n_w - (kk - w) / n_ok
        j = np.where(feas, j, -np.inf); i = int(np.argmax(j))
        if j[i] > best[0]:
            best = (j[i], ts, tcs[valid][i])
    return float(best[1]), float(best[2]), float(best[1])


def run(xs, cfg, train, test, con_target=CON_TARGET):
    """Thresholds per kind on `train`, verdicts on `test` (boolean masks)."""
    sc, sa, si = arrays(xs, cfg); ok = np.array([x["ok"] for x in xs]); kinds = np.array([x["kind"] for x in xs])
    g = gate(xs, cfg, train); guard = cfg["G"] == "G1"
    v = np.full(len(xs), "uncertain", object); th = {}
    for k in ("thing", "stuff"):
        m = kinds == k
        ts, tc, ta = fit_thresholds(sc[m & train], sa[m & train], si[m & train], g[m & train], ok[m & train],
                                    guard, con_target)
        tm = m & test
        v[tm] = rule(sc[tm], sa[tm], si[tm], g[tm], ts, tc, guard, ta)
        th[k] = (ts, tc, ta)
    return v, th


def cv(xs, cfg, con_target=CON_TARGET):
    folds = np.array([x["fold"] for x in xs]); v = np.empty(len(xs), object)
    for f in range(5):
        vf, _ = run(xs, cfg, folds != f, folds == f, con_target)
        v[folds == f] = vf[folds == f]
    return v


def quality(v, ok):
    return calibrate.verdict_quality(v, ok)


def n_changed(cfg):
    return sum(cfg[k] != DEFAULT[k] for k in DEFAULT)


def name(cfg):
    return " ".join(cfg[k] for k in DEFAULT)


def main():
    if SEL.exists():
        raise FileExistsError(f"{SEL} exists: Check v4 selection is locked")
    _, B = features(STEMS["B"])
    ok = np.array([x["ok"] for x in B])
    table = {}
    for combo in itertools.product(*OPTIONS.values()):
        cfg = dict(zip(OPTIONS, combo))
        q = quality(cv(B, cfg), ok)
        table[name(cfg)] = dict(cfg=cfg, changed=n_changed(cfg), B_cv=q)
    feas = {k: v for k, v in table.items() if v["B_cv"]["contradicted_precision"] >= 0.45}
    best = max(v["B_cv"]["J"] for v in feas.values())      # amendment 1: primary criterion J
    near = [k for k, v in feas.items() if v["B_cv"]["J"] >= best - 0.01]
    choice = min(near, key=lambda k: (table[k]["changed"], -table[k]["B_cv"]["contradicted_precision"]))
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]["B_cv"]["J"])[:15]:
        q = v["B_cv"]
        print(f"{k:22s} ch {v['changed']} recw {q['recall_wrong']:.3f} cprec {q['contradicted_precision']:.3f} "
              f"sprec {q['supported_precision']:.3f} J {q['J']:.3f} con {q['contradicted']:.3f}", flush=True)
    d = table[name(DEFAULT)]["B_cv"]
    print("default", {k: round(d[k], 3) for k in ("recall_wrong", "contradicted_precision", "J")})
    cache.save_json(dict(protocol="docs/CHECK_V4_PROTOKOL.md", table=table, choice=choice, best_J=best, amendment="§5 amendment 1",
                         default=name(DEFAULT)), SEL)
    print("LOCKED", choice, flush=True)


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
    sel = cache.load_json(SEL); out_path = config.RESULTS / "check_v4.json"
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists: Check v4 validation already reported")
    rowsB, B = features(STEMS["B"]); rowsV, V = features(STEMS["val"])
    cfg = sel["table"][sel["choice"]]["cfg"]
    allB, allV = np.ones(len(B), bool), np.ones(len(V), bool)
    both = B + V; trainB = np.r_[allB, np.zeros(len(V), bool)]; testV = ~trainB
    okV = np.array([x["ok"] for x in V]); gV = np.array([x["group"] for x in V]); ids = [x["id"] for x in V]
    systems = {}
    vs, ths = {}, {}
    for sysname, c in (("v4", cfg), ("original_tuned", DEFAULT)):
        for t in REPORT_TARGETS:
            v, th = run(both, c, trainB, testV, t)
            vs[(sysname, t)] = v[testV]; ths[(sysname, t)] = th
    vs[("original_live", 0.5)] = np.array([x["live"] for x in V])
    # learned references (not the main Check)
    for ref, f in (("ref_check_v2", "check2"), ("ref_check_v3", "check3")):
        recs = cache.load_json(P / f"{STEMS['val']}_{f}.json")
        vs[(ref, 0.5)] = np.array([recs[x["ri"]]["claims"][x["ci"]]["rec_att"]["verdict"] for x in V])
    for (sn, t), v in vs.items():
        systems.setdefault(sn, {})[str(t)] = {g: quality(v[(gV == g) if g != "all" else np.ones(len(V), bool)],
                                                          okV[(gV == g) if g != "all" else np.ones(len(V), bool)])
                                               for g in ("all", "clothing", "thing", "stuff")}
    # continuous region score AUROC (calibrated claim score in the region)
    key = (cfg["Q"], cfg["C"], cfg["R"], cfg["A"])
    sc = np.array([x["v"][key][0] if x["has_region"] else -1e9 for x in V])
    auroc = calibrate.auroc(sc, okV)
    v4 = vs[("v4", 0.5)]

    def cprec(v, i):
        s = v[i] == "contradicted"
        return (~okV[i][s]).mean() if s.any() else np.nan

    def recw(v, i):
        w = ~okV[i]
        return ((v[i] == "contradicted") & w).sum() / max(w.sum(), 1)

    def jj(v, i):
        return calibrate.verdict_quality(v[i], okV[i])["J"]

    boots = {}
    for base in ("original_live", "original_tuned"):
        b = vs[(base, 0.5)]; al = np.arange(len(V))
        boots[f"v4_minus_{base}"] = {
            m: dict(diff=float(fn(v4, al) - fn(b, al)), ci95=bootstrap(ids, lambda i: fn(v4, i) - fn(b, i)))
            for m, fn in (("contradicted_precision", cprec), ("recall_wrong", recw), ("J", jj))}
    rep = dict(choice=sel["choice"], cfg=cfg, thresholds={f"{s}@{t}": th for (s, t), th in ths.items()},
               systems=systems, bootstrap=boots, region_score_auroc=auroc,
               label="automatic reference proxy (not human labels); non-person claims")
    cache.save_json(rep, out_path)
    write_records(rowsB, B, rowsV, V, cfg, sel)
    for sn in ("original_live", "original_tuned", "v4", "ref_check_v2", "ref_check_v3"):
        q = systems[sn]["0.5"]["all"]
        print(f"{sn:16s} cprec {q['contradicted_precision']:.3f} recw {q['recall_wrong']:.3f} "
              f"sprec {q['supported_precision']:.3f} J {q['J']:.3f} con {q['contradicted']:.3f}")
    for k, v in boots.items():
        print(k, {m: (round(d['diff'], 3), [round(c, 3) for c in d['ci95']]) for m, d in v.items()})


def write_records(rowsB, B, rowsV, V, cfg, sel):
    key = (cfg["Q"], cfg["C"], cfg["R"], cfg["A"])
    folds = np.array([x["fold"] for x in B])
    vB = cv(B, cfg)
    both = B + V; trainB = np.r_[np.ones(len(B), bool), np.zeros(len(V), bool)]
    vV, th = run(both, cfg, trainB, ~trainB)
    vV = vV[len(B):]
    for stem, rows, xs, v, how in ((STEMS["B"], rowsB, B, vB, "5-fold image CV on B (thresholds per fold)"),
                                   (STEMS["val"], rowsV, V, vV, "thresholds from all of B")):
        for x, vv in zip(xs, v):
            rec = rows[x["ri"]]["claims"][x["ci"]]["rec_att"]
            s = x["v"].get(key) if x["has_region"] else None
            rec.update(check="v4", check_cfg=sel["choice"], verdict=str(vv),
                       s_claim_cal=s[0] if s else None, s_alt_cal=s[1] if s else None,
                       alt=s[2] if s else rec.get("alt", ""), s_image_cal=s[3] if s else None)
        for r in rows:
            r["check_metadata"] = dict(script="scripts/35_check_v4.py", cfg=cfg, thresholds=th,
                                       verdicts=how, label="reference proxy (automatic)",
                                       note="person claims keep the live record (presence only)")
        cache.save_json(rows, P / f"{stem}_check4.json")


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
