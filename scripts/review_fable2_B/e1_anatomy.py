"""Where do swap misses come from? Class distribution of real (proxy-wrong) claims vs swap labels; per-class separability."""
import collections
from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); n = cx.n; names = np.array(CLASSES); m = cx.m
alpha = 0.15
ct = np.zeros(len(n), bool); cs = np.zeros(len(n), bool); reg_s = np.zeros(len(n), bool)
top3 = cx.top[:, :3]
g = lambda k: np.where(top3 >= 0, L[cx.ri[:, None], np.maximum(top3, 0), k[:, None]], -np.inf).max(1)
r_t, r_s = g(cx.k), g(cx.kk)
for f in range(5):
    te = cx.fold == f; tau = pa_tau(IM[cx.img_fold != f], cx.men[cx.img_fold != f], alpha)
    ct[te] = cx.has[te] & (IM[cx.ri[te], cx.k[te]] < tau[cx.k[te]]) & ~cx.nv[te]
    cs[te] = cx.has[te] & (IM[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
    reg_s[te] = r_s[te] >= tau[cx.kk[te]]
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
print(f"S3-noabst a=.15: swaps same-confusion-set n={int((m&same).sum())} det {cs[m&same].mean():.3f} | other n={int((m&~same).sum())} det {cs[m&~same].mean():.3f}")
print(f"missed swaps: supported at Look region {reg_s[m&~cs].mean():.3f}, only elsewhere {(~reg_s[m&~cs]).mean():.3f}")
print("\nper swap class (n>=20): n, det, share of all misses")
miss = m & ~cs
for c, cnt in collections.Counter(names[cx.kk[m]]).most_common(30):
    j = CLASS_INDEX[c]; s = m & (cx.kk == j)
    print(f"  {c:11s} n={cnt:4d} det {cs[s].mean():.2f} miss-share {(miss&s).sum()/miss.sum():.3f} npos_img {cx.npos[j]:4d} same-set {same[s].mean():.2f}")
print("\nper (true->swap) pair detection, top pairs")
for (a, b), cnt in collections.Counter(zip(names[cx.k[m]], names[cx.kk[m]])).most_common(30):
    s = m & (cx.k == CLASS_INDEX[a]) & (cx.kk == CLASS_INDEX[b]); print(f"  {a:9s}->{b:10s} n={cnt:3d} det {cs[s].mean():.2f}")
# ---- class distribution of real claims: all, proxy-wrong, swap labels
th = cx.kind == "thing"
tot = collections.Counter(names[cx.k[th]]); wr = collections.Counter(names[cx.k[th & ~cx.ok]]); swc = collections.Counter(names[cx.kk[m]])
print("\nthing claims", th.sum(), "proxy-wrong", (th & ~cx.ok).sum())
print("class: n_claims, n_proxy_wrong, n_swap_label, FCR@.15, within-class AUROC image-max(pos claims vs swaps-to-class), AUROC mention")
for c, cnt in tot.most_common(40):
    j = CLASS_INDEX[c]; p = th & cx.ok & (cx.k == j) & ~cx.nv; s = m & (cx.kk == j)
    au = A(np.r_[IM[cx.ri[p], j], IM[cx.ri[s], j]], np.r_[np.ones(p.sum(), bool), np.zeros(s.sum(), bool)]) if s.sum() >= 5 and p.sum() >= 5 else float("nan")
    print(f"  {c:11s} {cnt:4d} {wr[c]:4d} {swc[c]:4d}  FCR {ct[p].mean() if p.sum() else float('nan'):.2f}  AUswap {au:.2f}  AUmention {A(IM[:, j], cx.men[:, j]):.2f}")
# rank correlation between proxy-wrong class counts and swap label counts
cl = [c for c in CLASSES if KIND[c] == "thing"]
a = np.array([wr[c] for c in cl]); b = np.array([swc[c] for c in cl]); t = np.array([tot[c] for c in cl])
from scipy.stats import spearmanr
print("spearman proxy-wrong vs swap-label counts", spearmanr(a, b)[0], " proxy-wrong vs all-claims", spearmanr(a, t)[0])
print("share of proxy-wrong thing claims whose class is among the 10 most claimed classes:", sum(wr[c] for c, _ in tot.most_common(10)) / a.sum(),
      "| same for swap labels:", sum(swc[c] for c, _ in tot.most_common(10)) / b.sum())
