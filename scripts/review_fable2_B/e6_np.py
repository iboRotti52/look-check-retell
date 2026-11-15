"""Per-class cost-sensitive (Neyman-Pearson style) thresholds: tau_c = argmax F_unmentioned(tau) - lam * F_pos(tau),
one lam for all classes (no class prior). Positives smoothed with a pooled centred shape. Out-of-fold."""
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
cnt = np.load(config.CACHE / "review_fable2_B_mencount.npy")
thc = np.array([KIND[c] == "thing" for c in CLASSES])
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
grp = np.array([re.GROUP[c] for c in CLASSES]); distinct = np.isin(grp, ["animal", "vehicle"])
pos = cx.ok & ~cx.nv & (cx.kind == "thing") & cx.has
SUB = dict(all=m, hibase=m & cx.hibase, clothing=m & cx.clo, nonclo=m & ~cx.clo, sameset=m & same, av2av=m & same & distinct[cx.k])
sets = [[CLASS_INDEX[c] for c in v] for v in re.CONFUSION.values()]

def rival_X(Lc, lam=2.0):
    X = Lc.max(1).copy()
    for S in sets:
        sub = Lc[:, :, S]
        for a, j in enumerate(S):
            riv = np.delete(sub, a, axis=2).max(2)
            with np.errstate(invalid="ignore"):
                v = sub[:, :, a] - lam * np.maximum(riv - sub[:, :, a], 0)
            X[:, j] = np.nanmax(np.where(np.isfinite(sub[:, :, a]), v, -np.inf), 1)
    return X

def np_tau(X, tr, lam, strict=True, m0=10, cap=None):
    tau = np.full(C, -np.inf)
    P = {}
    for j in range(C):
        pm = tr & ((cnt[:, j] >= 2) if strict and (tr & (cnt[:, j] >= 2)).sum() >= 5 else cx.men[:, j])
        P[j] = np.sort(X[pm, j])
    for kind in ("thing", "stuff"):
        cols = [j for j, c in enumerate(CLASSES) if KIND[c] == kind]
        G = np.sort(np.concatenate([P[j] - np.median(P[j]) for j in cols if len(P[j]) >= 20]))
        for j in cols:
            p = P[j]; u = np.sort(X[tr & ~cx.men[:, j], j])
            if len(p) < 2: continue
            cand = np.unique(np.r_[p, np.quantile(u, np.linspace(0.5, 0.999, 200))])
            Fp = (np.searchsorted(p, cand, side="left") + m0 * np.searchsorted(G, cand - np.median(p), side="left") / len(G)) / (len(p) + m0)
            Fn = np.searchsorted(u, cand, side="left") / len(u)
            if cap is not None:                       # optional: never spend more than `cap` of a class's positives
                okc = Fp <= cap
                if not okc.any(): continue
                cand, Fp, Fn = cand[okc], Fp[okc], Fn[okc]
            tau[j] = cand[np.argmax(Fn - lam * Fp)]
    return tau

def curve(name, Xfn, taufn, grid, targets=(0.03, 0.05, 0.08), ret=False):
    pts = []; Xs = [Xfn(f) for f in range(5)]
    for gpar in grid:
        ct = np.zeros(len(cx.n), bool); cs = np.zeros(len(cx.n), bool)
        for f in range(5):
            te = cx.fold == f; tr = cx.img_fold != f; X = Xs[f]; tau = taufn(X, tr, gpar)
            ct[te] = cx.has[te] & (X[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
            cs[te] = cx.has[te] & (X[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
        pts.append([ct[pos].mean(), ct[pos & cx.clo].mean()] + [cs[s].mean() for s in SUB.values()])
    pts = np.array(pts); pts = pts[np.argsort(pts[:, 0])]
    print(f"{name:44s} range FCR [{pts[0,0]:.3f},{pts[-1,0]:.3f}] " + " | ".join(f"FCR={t}: " + " ".join(f"{k} {np.interp(t, pts[:, 0], pts[:, 2 + i]):.3f}" for i, k in enumerate(SUB)) + f" fcr_clo {np.interp(t, pts[:, 0], pts[:, 1]):.3f}" for t in targets), flush=True)
    return pts

if __name__ == "__main__":
    LAMS = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100)
    XR = rival_X(L, 2.0)
    pa = lambda X, tr, a: pa_tau(X[tr], cx.men[tr], a)
    pa2 = lambda X, tr, a: pa_tau(X[tr], (cnt >= 2)[tr], a)
    curve("PA any-mention, image-max (S3 no-abst)", lambda f: IM, pa, ALPHAS)
    curve("PA >=2-mention, rival lam=2", lambda f: XR, pa2, ALPHAS)
    curve("NP any-mention, image-max", lambda f: IM, lambda X, tr, l: np_tau(X, tr, l, strict=False), LAMS)
    curve("NP strict, image-max", lambda f: IM, lambda X, tr, l: np_tau(X, tr, l), LAMS)
    curve("NP strict, rival lam=2", lambda f: XR, lambda X, tr, l: np_tau(X, tr, l), LAMS)
    curve("NP strict, rival lam=2, m0=30", lambda f: XR, lambda X, tr, l: np_tau(X, tr, l, m0=30), LAMS)
    curve("NP strict, rival lam=2, cap .15", lambda f: XR, lambda X, tr, l: np_tau(X, tr, l, cap=.15), LAMS)
    curve("NP strict, rival lam=2, cap .10", lambda f: XR, lambda X, tr, l: np_tau(X, tr, l, cap=.10), LAMS)
