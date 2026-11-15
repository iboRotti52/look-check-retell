"""Multi-annotator positives: how much of the 'clean positive' lower tail is reference noise? Thresholds from >=2-mention images."""
import pickle
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
CNTF = config.CACHE / "review_fable2_B_mencount.npy"
if CNTF.exists(): cnt = np.load(CNTF)
else:
    cnt = np.zeros((len(cx.ids), C), np.int8)
    for i, iid in enumerate(cx.ids):
        for r in data.references(iid):
            for c in claims.reference_classes([r]): cnt[i, CLASS_INDEX[c]] += 1
    np.save(CNTF, cnt)
assert ((cnt > 0) == cx.men).all()
th = cx.kind == "thing"; pos = cx.ok & ~cx.nv & th & cx.has
c_claim = cnt[cx.ri, cx.k]
print("ref-confirmed thing claims by number of references mentioning the class:", {int(v): int((pos & (c_claim == v)).sum()) for v in range(1, 6)})
for alpha in (0.05, 0.10, 0.15):
    ct = np.zeros(len(cx.n), bool)
    for f in range(5):
        te = cx.fold == f; tr = cx.img_fold != f; tau = pa_tau(IM[tr], cx.men[tr], alpha)
        ct[te] = cx.has[te] & (IM[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
    print(f"S3-noabst alpha={alpha}: FCR by #mentions:", " ".join(f"{v}:{ct[pos & (c_claim == v)].mean():.3f}" for v in range(1, 6)),
          f"| clothing 1:{ct[pos & cx.clo & (c_claim == 1)].mean():.3f} >=2:{ct[pos & cx.clo & (c_claim >= 2)].mean():.3f}",
          f"| non-clothing 1:{ct[pos & ~cx.clo & (c_claim == 1)].mean():.3f} >=2:{ct[pos & ~cx.clo & (c_claim >= 2)].mean():.3f} >=3:{ct[pos & ~cx.clo & (c_claim >= 3)].mean():.3f}")
# image level: lower tail of positives by mention count, a few classes
for c in "horse cow bird ball shirt jacket hat dress bicycle car".split():
    j = CLASS_INDEX[c]
    print(f"{c:9s} image-max q10: 1 mention (n={int((cnt[:, j] == 1).sum())}) {np.quantile(IM[cnt[:, j] == 1, j], .1) if (cnt[:, j] == 1).any() else float('nan'):.2f} | >=2 (n={int((cnt[:, j] >= 2).sum())}) {np.quantile(IM[cnt[:, j] >= 2, j], .1) if (cnt[:, j] >= 2).any() else float('nan'):.2f} | unmentioned median {np.median(IM[cnt[:, j] == 0, j]):.2f}")

def run(name, fit_men, eval_pos, Xfn=lambda f: IM, targets=(0.02, 0.03, 0.05)):
    """thresholds: PA quantile on fit_men positives; FCR on eval_pos claims; detection on swaps."""
    sub = dict(all=m, hibase=m & cx.hibase, clothing=m & cx.clo, nonclo=m & ~cx.clo)
    pts = []
    for alpha in ALPHAS:
        ct = np.zeros(len(cx.n), bool); cs = np.zeros(len(cx.n), bool)
        for f in range(5):
            te = cx.fold == f; tr = cx.img_fold != f; X = Xfn(f)
            tau = pa_tau(X[tr], fit_men[tr], alpha)
            ct[te] = cx.has[te] & (X[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
            cs[te] = cx.has[te] & (X[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
        pts.append([ct[eval_pos].mean(), ct[pos].mean()] + [cs[s].mean() for s in sub.values()])
    pts = np.array(pts)
    print(f"{name:46s} " + " | ".join(f"FCR={t}: " + " ".join(f"{k} {np.interp(t, pts[:, 0], pts[:, 2 + i]):.3f}" for i, k in enumerate(sub)) + f" (std FCR {np.interp(t, pts[:, 0], pts[:, 1]):.3f})" for t in targets), flush=True)
    return pts
strict = pos & (c_claim >= 2)
print("\nFCR measured on claims whose class >=2 references mention (strict FCR), n =", strict.sum())
run("fit: any mention  | eval: strict", cx.men, strict)
run("fit: >=2 mentions | eval: strict", cnt >= 2, strict)
run("fit: >=3 mentions | eval: strict", cnt >= 3, strict)
print("FCR measured on all reference-confirmed claims (standard)")
run("fit: any mention  | eval: standard", cx.men, pos, targets=(0.03, 0.05, 0.08))
run("fit: >=2 mentions | eval: standard", cnt >= 2, pos, targets=(0.03, 0.05, 0.08))
pickle.dump(dict(cnt=cnt), open(ROOT + "/tmp/cnt.pkl", "wb"))
