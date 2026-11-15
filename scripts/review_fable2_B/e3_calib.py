"""Calibration of the image-level absence test: shrink targets, two-sided (pos+unmentioned) scales, oracle allocation."""
import collections
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
th = np.array([KIND[c] == "thing" for c in CLASSES])
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
grp = np.array([re.GROUP[c] for c in CLASSES])
distinct = np.isin(grp, ["animal", "vehicle"])
subs = dict(sameset=m & same, av2av=m & same & distinct[cx.k], rare_lbl=m & (cx.npos[cx.kk] < 20))
print("n: sameset", subs["sameset"].sum(), "animal/vehicle same-set", subs["av2av"].sum(), "swap label with <20 positive images", subs["rare_lbl"].sum(), "hibase", (m & cx.hibase).sum())
print("hibase swap labels:", collections.Counter(names[cx.kk[m & cx.hibase]]).most_common(15))
print("hibase true labels:", collections.Counter(names[cx.k[m & cx.hibase]]).most_common(15))
curve_pa(cx, lambda f: IM, name="PA raw, shrink to kind quantile (S3)", subsets=subs)

# generic evaluation with a custom per-fold threshold function tau(f, alpha)
def curve_tau(taufn, name, targets=(0.03, 0.05, 0.08)):
    pos = cx.ok & ~cx.nv & (cx.kind == "thing")
    sub = dict(all=m, hibase=m & cx.hibase, clothing=m & cx.clo, nonclo=m & ~cx.clo); sub.update(subs)
    pts = []
    for alpha in ALPHAS:
        ct = np.zeros(len(cx.n), bool); cs = np.zeros(len(cx.n), bool)
        for f in range(5):
            te = cx.fold == f; tau = taufn(f, alpha)
            ct[te] = cx.has[te] & (IM[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
            cs[te] = cx.has[te] & (IM[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
        pts.append([ct[pos].mean(), ct[pos & cx.clo].mean()] + [cs[s].mean() for s in sub.values()])
    pts = np.array(pts); o = np.argsort(pts[:, 0]); pts = pts[o]
    print(f"{name:40s} " + " | ".join(f"FCR_thing={t}: " + " ".join(f"{k} {np.interp(t, pts[:, 0], pts[:, 2 + i]):.3f}" for i, k in enumerate(sub)) + f" fcr_clo {np.interp(t, pts[:, 0], pts[:, 1]):.3f}" for t in targets), flush=True)

def no_shrink(f, alpha, min_n=3):
    tr = cx.img_fold != f; base = pa_tau(IM[tr], cx.men[tr], alpha)
    tau = base.copy()
    for j in range(C):
        p = IM[tr & cx.men[:, j], j]
        if len(p) >= min_n: tau[j] = np.quantile(p, alpha)
    return tau
curve_tau(no_shrink, "PA raw, no shrink (n>=3)")

def zshrink(f, alpha, mm=20, robust=True, hi=None):
    """Shrink in a class-normalised space: z = (s - loc_c)/scale_c with loc/scale from images NOT mentioning c."""
    tr = cx.img_fold != f; tau = np.zeros(C)
    loc = np.zeros(C); sc = np.ones(C)
    for j in range(C):
        u = IM[tr & ~cx.men[:, j], j]
        if robust:
            loc[j] = np.median(u); sc[j] = (np.quantile(u, 0.84) - np.quantile(u, 0.16)) / 2 + 1e-6
        else:
            loc[j] = u.mean(); sc[j] = u.std() + 1e-6
    for kind in ("thing", "stuff"):
        cols = [j for j, c in enumerate(CLASSES) if KIND[c] == kind]
        pooled = np.concatenate([(IM[tr & cx.men[:, j], j] - loc[j]) / sc[j] for j in cols])
        qk = np.quantile(pooled, alpha)
        for j in cols:
            p = (IM[tr & cx.men[:, j], j] - loc[j]) / sc[j]; n = len(p)
            qz = qk if n < 3 else (n * np.quantile(p, alpha) + mm * qk) / (n + mm)
            tau[j] = loc[j] + sc[j] * qz
    return tau
for mm in (5, 20, 50):
    curve_tau(lambda f, a, mm=mm: zshrink(f, a, mm), f"PA, shrink in robust-z space m={mm}")
curve_tau(lambda f, a: zshrink(f, a, 20, robust=False), "PA, shrink in mean/std-z space m=20")
curve_tau(lambda f, a: zshrink(f, a, 1e9), "pure robust-z global quantile (S2-like scale)")

# ---- two-sided LDA log-likelihood ratio with a global threshold
def llr_scores(shr=20):
    St = np.full(len(cx.n), np.nan); Ss = np.full(len(cx.n), np.nan)
    for f in range(5):
        tr = cx.img_fold != f; te = cx.fold == f
        mu_n = np.zeros(C); mu_p = np.zeros(C); var = np.ones(C); gaps = []
        for j in range(C):
            u = IM[tr & ~cx.men[:, j], j]; p = IM[tr & cx.men[:, j], j]
            mu_n[j] = np.median(u); var[j] = ((np.quantile(u, .84) - np.quantile(u, .16)) / 2) ** 2 + 1e-6
            mu_p[j] = np.median(p) if len(p) else np.nan
        z_gap = (mu_p - mu_n) / np.sqrt(var); g0 = np.nanmedian(z_gap[th])
        npj = (tr[:, None] & cx.men).sum(0)
        z_gap = np.where(npj > 0, (npj * np.nan_to_num(z_gap) + shr * g0) / (npj + shr), g0)
        def llr(s, j):
            z = (s - mu_n[j]) / np.sqrt(var[j]); return z_gap[j] * (z - z_gap[j] / 2)
        St[te] = llr(IM[cx.ri[te], cx.k[te]], cx.k[te]); Ss[te] = llr(IM[cx.ri[te], cx.kk[te]], cx.kk[te])
    return St, Ss
St, Ss = llr_scores(); curve_global(cx, St, Ss, name="two-sided LDA LLR, global thr", subsets=subs)

# ---- oracle: best per-class thresholds for swap detection at total thing-FCR (in-sample, optimistic upper bound for ANY per-class monotone recalibration of image-max)
pos = cx.ok & ~cx.nv & (cx.kind == "thing") & cx.has
def oracle(target, weights=None):
    w = np.ones(len(cx.n)) if weights is None else weights
    lo, hi = 0.0, 1e4; best = None
    cls = [j for j in range(C) if th[j]]
    data_ = {}
    for j in cls:
        p = np.sort(IM[cx.ri[pos & (cx.k == j)], j]); sidx = m & (cx.kk == j); s = IM[cx.ri[sidx], j]; ws = w[sidx]
        cand = np.unique(np.r_[-np.inf, p, s]) ; cand = np.r_[cand, np.inf]
        fc = np.array([(p < t).sum() for t in cand]); dt = np.array([ws[s < t].sum() for t in cand])
        data_[j] = (fc, dt)
    for _ in range(60):
        lam = (lo + hi) / 2; F = D = 0
        for j in cls:
            fc, dt = data_[j]; i = np.argmax(dt - lam * fc); F += fc[i]; D += dt[i]
        if F / pos.sum() > target: lo = lam
        else: hi = lam; best = (F / pos.sum(), D / w[m].sum())
    return best
for t in (0.03, 0.05, 0.08):
    print("ORACLE per-class thresholds (in-sample upper bound) target", t, "-> FCR %.3f det %.3f" % oracle(t))
# oracle restricted to respecting class-symmetric FCR (every class same alpha, in-sample quantile, no shrink)
