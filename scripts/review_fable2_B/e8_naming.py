"""Threshold-free separability and 'what is there' naming at the attended box, on reference-confirmed thing claims."""
import collections
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
cnt = np.load(config.CACHE / "review_fable2_B_mencount.npy")
thc = np.array([KIND[c] == "thing" for c in CLASSES])
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
pos = cx.ok & ~cx.nv & (cx.kind == "thing") & cx.has
med = np.zeros((5, C))
for f in range(5):
    tr = cx.img_fold != f; med[f] = pa_tau(IM[tr], (cnt >= 2)[tr], 0.5)
medc = med[cx.fold]                                   # (n, C) class medians (centre), out-of-fold
n = cx.n; t1 = np.maximum(cx.top[:, 0], 0)
B1 = L[cx.ri, t1]                                     # (n, C) logits at Look top-1 box
B3 = np.where((cx.top[:, :3] >= 0)[..., None], L[cx.ri[:, None], np.maximum(cx.top[:, :3], 0)], -np.inf)  # (n,3,C)
bb = L[cx.ri, :, cx.k].argmax(1); BC = L[cx.ri, bb]   # logits at the claim's best box in the image
I = IM[cx.ri]
grp = np.array([re.GROUP[c] for c in CLASSES])
def pair(name, X):   # P(true class scores above swapped class) on swaps
    w = X[n, cx.k] > X[n, cx.kk]
    print(f"{name:34s} pairwise true>swap: all {w[m].mean():.3f} | same-set {w[m&same].mean():.3f} | same-set clothing {w[m&same&cx.clo].mean():.3f} | same-set non-clothing {w[m&same&~cx.clo].mean():.3f} | cross-set {w[m&~same].mean():.3f}")
pair("image-max raw", I); pair("image-max centred", I - medc); pair("Look top-1 box raw", B1); pair("Look top-1 box centred", B1 - medc)
pair("Look top-3 max centred", B3.max(1) - medc); pair("claim-best box centred", BC - medc)
print("n same-set clothing", (m & same & cx.clo).sum(), "same-set non-clothing", (m & same & ~cx.clo).sum())

# ---- naming: detector's top thing class at the attended box vs the reference-confirmed claim
def naming(name, X, cand="all"):
    Xc = np.where(thc[None], X - medc, -np.inf)
    if cand == "set":
        msk = np.array([re.confusion_mask(CLASSES[j]) | (np.arange(C) == j) for j in range(C)])[cx.k]; Xc = np.where(msk, Xc, -np.inf)
    top = Xc.argmax(1); hit = top == cx.k
    inref = np.array([names[t] in cx.refs[cx.xs[i]["id"]] for i, t in enumerate(top)])
    hasset = np.array([names[j] in re.CONFUSION_OF for j in cx.k])
    sel = pos if cand == "all" else pos & hasset
    g = lambda s: f"{hit[sel & s].mean():.3f}"
    print(f"{name:40s} top-name == claim: all {g(np.ones(len(n), bool))} (n={sel.sum()}) animal {g(grp[cx.k]=='animal')} vehicle {g(grp[cx.k]=='vehicle')} clothing {g(cx.clo)} other {g(~cx.clo & ~np.isin(grp[cx.k], ['animal','vehicle']))} | top name in refs {inref[sel].mean():.3f}")
    return top
naming("Look top-1 box, all thing classes", B1)
naming("claim-best box, all thing classes", BC)
naming("Look top-1 box, within confusion set", B1, "set")
naming("Look top-3 max, within confusion set", B3.max(1), "set")
naming("claim-best box, within confusion set", BC, "set")
naming("image-max, within confusion set", I, "set")
# per clothing class within-set naming at claim-best box
Xc = BC - medc
for c in "shirt jacket sweater vest dress suit uniform pants shorts hat helmet dog horse bicycle car ball frisbee bench".split():
    j = CLASS_INDEX[c]; s = pos & (cx.k == j); msk = re.confusion_mask(c) | (np.arange(C) == j)
    if s.sum() < 5: continue
    top = np.where(msk[None], Xc[s], -np.inf).argmax(1)
    print(f"  {c:9s} n={s.sum():4d} within-set argmax==claim {np.mean(top == j):.2f}  confusions: {collections.Counter(names[top[top != j]]).most_common(3)}")
