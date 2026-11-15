"""Rival-aware image-level evidence: the claim only counts at boxes where the detector does not prefer a mutually
exclusive name. Positive-anchored thresholds (no abstention), out-of-fold."""
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES)
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
subs = dict(sameset=cx.m & same, otherset=cx.m & ~same)
sets = [[CLASS_INDEX[c] for c in v] for v in re.CONFUSION.values()]
curve_pa(cx, lambda f: IM, name="baseline image-max raw", subsets=subs)

def med_pos(f):   # per-class median of positives' image-max (training folds), shrunk like tau; used to centre logits
    tr = cx.img_fold != f
    return pa_tau(IM[tr], cx.men[tr], 0.5)

def rival_X(Lc, lam, delta):
    X = Lc.max(1).copy()
    for S in sets:
        sub = Lc[:, :, S]                                   # (N,100,|S|)
        for a, j in enumerate(S):
            riv = np.delete(sub, a, axis=2).max(2)
            pen = np.maximum(riv - sub[:, :, a] - delta, 0)
            with np.errstate(invalid="ignore"):
                v = sub[:, :, a] - lam * pen
            X[:, j] = np.nanmax(np.where(np.isfinite(sub[:, :, a]), v, -np.inf), 1)
    return X

for centred in (False, True):
    for lam, delta in ((0.5, 0), (1, 0), (2, 0), (1, 0.5), (1, 1.0), (4, 0)):
        if centred:
            cache = {}
            def Xf(f, lam=lam, delta=delta):
                c = med_pos(f); X = rival_X(L - c[None, None], lam, delta); return X      # centred score; thresholds re-fit on it
            curve_pa(cx, Xf, name=f"rival centred lam={lam} d={delta}", subsets=subs)
        else:
            X = rival_X(L, lam, delta)
            curve_pa(cx, lambda f: X, name=f"rival raw lam={lam} d={delta}", subsets=subs)
