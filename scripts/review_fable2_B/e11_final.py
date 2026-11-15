"""Candidate S4 = rival-aware evidence + cost-sensitive per-class thresholds (+ exemplar-refined queries), against
S3 (no abstention) and S2, on cleaner swap subsets. Also substitutes and the region-level rival add-on. B, out-of-fold."""
import collections
from common import *
import e6_np as E6
cx = E6.cx; m = cx.m; cnt = E6.cnt; names = np.array(CLASSES); L = E6.L; IM = E6.IM; pos = E6.pos
same = E6.same; setof = np.array([re.CONFUSION_OF.get(c, "") for c in CLASSES])
clean = m & same & ~cx.clo & (setof[cx.k] != "boat")
SUBS = dict(all=m, clean_excl=clean, same_clo=m & same & cx.clo, cross=m & ~same, hibase=m & cx.hibase)
print({k: int(v.sum()) for k, v in SUBS.items()})
S2 = np.load(config.CACHE / "review_fable2_B_s2.npz")
c2 = np.isin(S2["a_true"], ["replace", "remove"]); c2s = np.isin(S2["a_sw"], ["replace", "remove"])
allpos = cx.ok & ~cx.nv
line = lambda name, ct, cs: print(f"{name:42s} FCR all {ct[allpos].mean():.3f} thing {ct[pos].mean():.3f} clo {ct[pos&cx.clo].mean():.3f} dog {ct[pos&(cx.k==CLASS_INDEX['dog'])].mean():.3f} stuff {ct[allpos&(cx.kind=='stuff')].mean():.3f} | det " + " ".join(f"{k} {cs[s].mean():.3f}" for k, s in SUBS.items()), flush=True)
line("S2 five-way z", c2, c2s)

# exemplar-refined queries (disc, beta .5), per fold
def protos(f, min_n=5):
    tr = cx.img_fold != f; Pm = np.zeros((C, 512), np.float32); have = np.zeros(C, bool)
    for j in range(C):
        imgs = np.where(tr & (cnt[:, j] >= 2))[0]
        if len(imgs) < min_n: continue
        b = L[imgs, :, j].argmax(1); e = cx.E[imgs, b].astype(np.float32).mean(0); Pm[j] = e / np.linalg.norm(e); have[j] = True
    return Pm, have
def exq(f, beta=0.5):
    Pm, have = protos(f); q = cx.q.copy()
    for j in np.where(have)[0]:
        riv = [a for a in np.where(re.confusion_mask(CLASSES[j]))[0] if have[a]]
        q[j] = q[j] + beta * (Pm[j] - (Pm[riv].mean(0) if riv else Pm[have].mean(0)))
    return q / np.linalg.norm(q, axis=1, keepdims=True)
XR = E6.rival_X(L, 2.0)
XE = []
for f in range(5):
    Lq = cx.logits(exq(f)); XE.append(E6.rival_X(Lq, 2.0)); del Lq

def flags(Xfn, taufn, par):
    ct = np.zeros(len(cx.n), bool); cs = np.zeros(len(cx.n), bool); TAU = []
    for f in range(5):
        te = cx.fold == f; tr = cx.img_fold != f; X = Xfn(f); tau = taufn(X, tr, par); TAU.append(tau)
        ct[te] = cx.has[te] & ~cx.nv[te] & (X[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
        cs[te] = cx.has[te] & (X[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
    return ct, cs, TAU
def at_fcr(Xfn, taufn, grid, target):
    """largest-FCR grid point with thing-FCR <= target"""
    best = None
    for par in grid:
        ct, cs, TAU = flags(Xfn, taufn, par)
        if ct[pos].mean() <= target and (best is None or ct[pos].mean() > best[0][pos].mean()): best = (ct, cs, TAU, par)
    return best
LAMS = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 100)
AL = np.r_[np.arange(0.005, 0.1, 0.005), np.arange(0.1, 0.26, 0.01)]
pa = lambda X, tr, a: pa_tau(X[tr], cx.men[tr], a); npt = lambda X, tr, l: E6.np_tau(X, tr, l)
SYS = {"S3 no-abst (PA, image-max)": (lambda f: IM, pa, AL), "S4a NP thresholds, image-max": (lambda f: IM, npt, LAMS),
       "S4b NP + rival-aware": (lambda f: XR, npt, LAMS), "S4c NP + rival + exemplar queries": (lambda f: XE[f], npt, LAMS)}
keep = {}
for target in (0.03, 0.05, 0.08):
    print(f"--- thing-FCR <= {target}")
    for name, (Xfn, tf, grid) in SYS.items():
        ct, cs, TAU, par = at_fcr(Xfn, tf, grid, target); line(f"{name} [{par:.3g}]", ct, cs); keep[(name, target)] = (ct, cs, TAU, par)

# ---- what drives the change: where each system spends its false contradictions (FCR<=.05)
for name in ("S3 no-abst (PA, image-max)", "S4b NP + rival-aware"):
    ct = keep[(name, 0.05)][0]; c = collections.Counter(names[cx.k[pos & ct]])
    print(name, "false contradictions by class:", c.most_common(8), "total", int((pos & ct).sum()))

# ---- substitutes on detected swaps (S4b at FCR<=.05)
ct, cs, TAU, par = keep[("S4b NP + rival-aware", 0.05)]
med = np.stack([pa_tau(IM[cx.img_fold != f], (cnt >= 2)[cx.img_fold != f], 0.5) for f in range(5)])[cx.fold]
tauc = np.stack(TAU)[cx.fold]                       # per-claim OOF thresholds (n, C)
t3 = cx.top[:, :3]; B3 = np.where((t3 >= 0)[..., None], L[cx.ri[:, None], np.maximum(t3, 0)], -np.inf)
B1 = B3[:, 0]; n = cx.n
cmask = np.array([re.confusion_mask(c) for c in CLASSES])[cx.kk]
bbs = L[cx.ri, :, cx.kk].argmax(1); BS = L[cx.ri, bbs]      # logits at the (swapped) claim's best box
def subst(X, need_tau=True, beat=True, centred=False):
    S = (X - med) if centred else (X - tauc)
    okm = cmask & (X >= tauc if need_tau else True) & ((X > X[n, cx.kk][:, None]) if beat else True)
    S = np.where(okm, S, -np.inf); j = S.argmax(1); return np.where(np.isfinite(S.max(1)), j, -1)
det = m & cs
print(f"S4b FCR<=.05: detected {det.sum()} of {m.sum()}; detected & same-set (substitute recoverable by construction) {int((det & same).sum())}")
for name, j in [("S3 rule: Look top-1, >= own tau, > claim", subst(B1)), ("Look top-3 max, >= own tau, > claim", subst(B3.max(1))),
                ("Look top-1, centred argmax, >= own tau", subst(B1, centred=True)),
                ("claim-best box, >= own tau, > claim", subst(BS)), ("image-max, >= own tau", subst(IM[cx.ri], beat=False))]:
    rep = det & (j >= 0); hit = rep & (j == cx.k)
    inref = np.mean([names[j[i]] in cx.refs[cx.xs[i]["id"]] for i in np.where(rep)[0]])
    print(f"  {name:42s} replace {rep.sum():4d} ({rep.sum()/det.sum():.2f} of detected) | substitute == true class {hit.sum()/max(rep.sum(),1):.3f} | in refs {inref:.3f} | true-class recovery among detected same-set {hit.sum()/(det&same).sum():.3f} | of all swaps {hit.sum()/m.sum():.3f}")
# fallback chain: Look top-1 rule, else image-level
j1 = subst(B1); j2 = subst(IM[cx.ri], beat=False); j = np.where(j1 >= 0, j1, j2); rep = det & (j >= 0); hit = rep & (j == cx.k)
print(f"  chain Look top-1 -> image-max: replace {rep.sum()} substitute==true {hit.sum()/rep.sum():.3f} recovery among detected same-set {hit.sum()/(det&same).sum():.3f} of all swaps {hit.sum()/m.sum():.3f}")

# ---- region-level rival add-on: also contradict claims that exist 'elsewhere' when the attended box is confidently a rival
Xt = XR; reg_t = B3[n, :, cx.k].max(1) if False else np.take_along_axis(B3, cx.k[:, None, None].repeat(3, 1), 2)[..., 0].max(1)
reg_s = np.take_along_axis(B3, cx.kk[:, None, None].repeat(3, 1), 2)[..., 0].max(1)
cm_t = np.array([re.confusion_mask(c) for c in CLASSES])[cx.k]
def region_rival(kk, cmk, reg, delta):
    Sc = B1 - med; mar = np.where(cmk & (B1 >= tauc), Sc, -np.inf).max(1) - Sc[n, kk]
    return cx.has & (reg < tauc[n, kk]) & (mar >= delta)
for delta in (0.0, 1.0, 2.0, 3.0):
    at = region_rival(cx.k, cm_t, reg_t, delta) & ~cx.nv; as_ = region_rival(cx.kk, cmask, reg_s, delta)
    line(f"S4b@.05 + region rival delta={delta}", ct | at, cs | as_)
