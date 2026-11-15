"""Query quality: text prompt variants and image-exemplar (prototype) queries from the same frozen OWLv2 embeddings."""
from common import *
import e6_np as E6
cx = E6.cx; m = cx.m; cnt = E6.cnt; names = np.array(CLASSES)
qs = dict(single=cx.q, grammatical=np.load(config.CACHE / "owl" / "class_queries_grammatical.npy"), ensemble=np.load(config.CACHE / "owl" / "class_queries_ensemble.npy"))
qs.update(dict(np.load(config.CACHE / "review_fable2_B_queries.npz")))
LAMS = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100)
swc = np.bincount(cx.kk[m], minlength=C)
def sep(IMf, tag):
    """within-class AUROC: reference-confirmed claims of class c vs swaps relabelled c (image-max of c). IMf(f) per fold."""
    S_t = np.empty(len(cx.n)); S_s = np.empty(len(cx.n))
    for f in range(5):
        te = cx.fold == f; X = IMf(f); S_t[te] = X[cx.ri[te], cx.k[te]]; S_s[te] = X[cx.ri[te], cx.kk[te]]
    aus = {}
    for j in np.where(swc >= 10)[0]:
        p = E6.pos & (cx.k == j); s = m & (cx.kk == j)
        if p.sum() >= 5: aus[names[j]] = A(np.r_[S_t[p], S_s[s]], np.r_[np.ones(p.sum(), bool), np.zeros(s.sum(), bool)])
    w = np.array([swc[CLASS_INDEX[c]] for c in aus]); v = np.array(list(aus.values()))
    clo = np.array([re.GROUP[c] == "clothing" for c in aus])
    print(f"[{tag}] within-class swap AUROC: weighted mean {np.sum(w*v)/w.sum():.3f} clothing {np.sum((w*v)[clo])/w[clo].sum():.3f} non-clothing {np.sum((w*v)[~clo])/w[~clo].sum():.3f} | "
          + " ".join(f"{c} {aus[c]:.2f}" for c in ["shirt", "jacket", "sweater", "suit", "dress", "hat", "uniform", "toy", "wall", "ball", "horse"] if c in aus), flush=True)
for nm, q in qs.items():
    Lq = cx.logits(q.astype(np.float32)); IMq = Lq.max(1); sep(lambda f: IMq, nm)
    XR = E6.rival_X(Lq, 2.0)
    E6.curve(f"  {nm}: NP strict + rival", lambda f: XR, lambda X, tr, l: E6.np_tau(X, tr, l), LAMS)
    del Lq
# ---- exemplar queries (per fold, training images only)
L0 = cx.logits()
def protos(f, min_n=5):
    tr = cx.img_fold != f; Pm = np.zeros((C, 512), np.float32); have = np.zeros(C, bool)
    for j in range(C):
        imgs = np.where(tr & (cnt[:, j] >= 2))[0]
        if len(imgs) < min_n: continue
        b = L0[imgs, :, j].argmax(1); e = cx.E[imgs, b].astype(np.float32).mean(0); Pm[j] = e / np.linalg.norm(e); have[j] = True
    return Pm, have
PR = [protos(f) for f in range(5)]
def exq(f, mode, beta):
    Pm, have = PR[f]; q = cx.q.copy()
    if mode == "pure":
        q[have] = Pm[have]
    elif mode == "mix":
        d = Pm - Pm[have].mean(0); q[have] = q[have] + beta * d[have]
    elif mode == "disc":
        for j in np.where(have)[0]:
            riv = [a for a in np.where(re.confusion_mask(CLASSES[j]))[0] if have[a]]
            d = Pm[j] - (Pm[riv].mean(0) if riv else Pm[have].mean(0)); q[j] = q[j] + beta * d
    return q / np.linalg.norm(q, axis=1, keepdims=True)
for mode, beta in (("pure", 0), ("mix", 0.25), ("mix", 0.5), ("mix", 1.0), ("disc", 0.25), ("disc", 0.5), ("disc", 1.0)):
    Ls = [cx.logits(exq(f, mode, beta)) for f in range(5)] if False else None
    IMs, XRs = [], []
    for f in range(5):
        Lq = cx.logits(exq(f, mode, beta)); IMs.append(Lq.max(1)); XRs.append(E6.rival_X(Lq, 2.0)); del Lq
    sep(lambda f: IMs[f], f"exemplar {mode} b={beta}")
    E6.curve(f"  exemplar {mode} b={beta}: NP strict + rival", lambda f: XRs[f], lambda X, tr, l: E6.np_tau(X, tr, l), LAMS)
