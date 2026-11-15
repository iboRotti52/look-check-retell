"""Reproduce S3-no-abstention and S2 on my harness; anatomy of the swap test."""
import importlib, collections
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1)
n = cx.n
def gather(L, ri, top, k):       # per-claim logits at boxes `top` (n, K) for label k -> (n,K), -inf where missing
    v = L[ri[:, None], np.maximum(top, 0), k[:, None]]
    return np.where(top >= 0, v, -np.inf)
reg3_t = gather(L, cx.ri, cx.top[:, :3], cx.k).max(1); reg3_s = gather(L, cx.ri, cx.top[:, :3], cx.kk).max(1)
for alpha in (0.05, 0.075, 0.10, 0.15, 0.20):
    ct = np.zeros(len(n), bool); cs = np.zeros(len(n), bool)
    for f in range(5):
        te = cx.fold == f; tau = pa_tau(IM[cx.img_fold != f], cx.men[cx.img_fold != f], alpha)
        ct[te] = cx.has[te] & (IM[cx.ri[te], cx.k[te]] < tau[cx.k[te]]) & ~cx.nv[te]
        cs[te] = cx.has[te] & (IM[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
    report(f"S3 no-abst alpha={alpha}", ct, cs, cx)
# ---- anatomy of the swap set ----
m = cx.m; names = np.array(CLASSES)
src = np.array(["alt" if any(a not in cx.refs[x["id"]] and KIND[a] == "thing" for a in x["blip_alts"]) else "rand" for x in cx.xs])
print("swap n", m.sum(), "source", collections.Counter(src[m]))
same_set = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
print("swap class in same CONFUSION set as true class:", same_set[m].mean(), "| swapped class has a confusion set:", np.mean([names[b] in re.CONFUSION_OF for b in cx.kk[m]]))
print("by source: same_set alt", same_set[m & (src == "alt")].mean(), "rand", same_set[m & (src == "rand")].mean())
pairs = collections.Counter((names[a], names[b]) for a, b in zip(cx.k[m], cx.kk[m]))
print("top pairs (true->swap):", pairs.most_common(40))
print("swap class freq:", collections.Counter(names[cx.kk[m]]).most_common(25))
print("true class freq:", collections.Counter(names[cx.k[m]]).most_common(15))
np.save(ROOT + "/tmp/src.npy", src)
