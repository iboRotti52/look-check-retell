"""Shared harness for Fable's second review (B only; no validation/test). Exploratory, not report numbers.
Builds per-image full proposal logits and per-claim arrays; swap set identical to scripts/43_check_fix.py."""
import sys, pickle, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/scripts"); sys.path.insert(0, ROOT)
import _common  # noqa
import numpy as np
from lcr import cache, config, claims, data, calibrate
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX, KIND
C = len(CLASSES); A = calibrate.auroc
P = config.CACHE / "pipeline"
FEAT = config.CACHE / "review_fable2_B_feat.pkl"


def build():
    rows = cache.load_json(P / "A_B_look4.json")
    assert all(r["id"].startswith("train_") for r in rows)
    maps = np.load(P / "A_B_look4.maps.npz"); owner = {tuple(o): i for i, o in enumerate(maps["owner"])}
    owl = cache.OwlCache("train")
    rule = dict(min_patches=0, alpha=0.5)
    xs, E, SH, SC, BX, OB, CNT = [], [], [], [], [], [], []
    for ri, row in enumerate(rows):
        prop = owl[int(row["id"].rsplit("_", 1)[1])]
        n = len(prop["objectness"]); CNT.append(n)
        pad = lambda a: np.concatenate([a, np.zeros((100 - n, *a.shape[1:]), a.dtype)]) if n < 100 else a
        E.append(pad(prop["embeds"])); SH.append(pad(prop["shift"])); SC.append(pad(prop["scale"]))
        BX.append(pad(prop["boxes"])); OB.append(pad(prop["objectness"]))
        for ci, c in enumerate(row["claims"]):
            if c["kind"] == "person":
                continue
            top, w = re.candidates(maps["maps"][owner[(ri, ci)]].astype(np.float32), prop, c["kind"], rule, n=10)
            xs.append(dict(ri=ri, ci=ci, id=row["id"], cls=c["cls"], kind=c["kind"], ok=bool(c["in_refs"]),
                           top=top.tolist(), blip_alts=c["blip_alts"], nv=claims.non_visual(row["draft"], c["cls"]),
                           draft=row["draft"]))
    ids = [r["id"] for r in rows]
    refs = {i: claims.reference_classes(data.references(i)) for i in ids}
    pickle.dump(dict(xs=xs, E=np.stack(E), SH=np.stack(SH), SC=np.stack(SC), BX=np.stack(BX), OB=np.stack(OB),
                     CNT=np.array(CNT), ids=ids, refs=refs), open(FEAT, "wb"), protocol=4)


class Ctx:
    def __init__(self):
        if not FEAT.exists():
            build()
        D = pickle.load(open(FEAT, "rb"))
        self.D = D; self.xs = xs = D["xs"]; self.ids = ids = D["ids"]; self.refs = refs = D["refs"]
        self.E, self.SH, self.SC, self.BX, self.OB, self.CNT = (D[k] for k in ("E", "SH", "SC", "BX", "OB", "CNT"))
        self.valid = np.arange(100)[None] < self.CNT[:, None]
        self.q = np.load(config.CACHE / "owl" / "class_queries_single.npy").astype(np.float32)
        self.men = np.array([[c in refs[i] for c in CLASSES] for i in ids])
        self.img_fold = np.array([calibrate.fold_of(i) for i in ids])
        self.ri = np.array([x["ri"] for x in xs]); self.k = np.array([CLASS_INDEX[x["cls"]] for x in xs])
        self.ok = np.array([x["ok"] for x in xs]); self.nv = np.array([x["nv"] for x in xs])
        self.kind = np.array([x["kind"] for x in xs]); self.fold = self.img_fold[self.ri]
        self.top = np.array([(x["top"] + [-1] * 10)[:10] for x in xs]); self.has = self.top[:, 0] >= 0
        self.clo = np.array([re.GROUP[x["cls"]] == "clothing" for x in xs])
        self.sw = self.swap(); self.m = self.sw >= 0; self.kk = np.where(self.m, self.sw, self.k)
        npos = self.men.sum(0); self.npos = npos
        self.hibase = self.m & (npos[np.maximum(self.sw, 0)] > npos[self.k])
        self.n = np.arange(len(xs))

    def swap(self):
        rng = np.random.default_rng(0); by = {}
        for c in CLASSES:
            by.setdefault(re.GROUP[c], []).append(c)
        sw = []
        for x in self.xs:
            if not (x["ok"] and x["kind"] == "thing"):
                sw.append(-1); continue
            r = self.refs[x["id"]]
            cand = [a for a in x["blip_alts"] if a not in r and KIND[a] == "thing"]
            if not cand:
                cand = [c for c in by[re.GROUP[x["cls"]]] if c not in r and c != x["cls"]]
                cand = [cand[rng.integers(len(cand))]] if cand else []
            sw.append(CLASS_INDEX[cand[0]] if cand else -1)
        return np.array(sw)

    def logits(self, q=None):
        """(N_img, 100, Q) float32 logits; invalid proposals = -inf."""
        q = self.q if q is None else q
        out = np.empty((len(self.ids), 100, q.shape[0]), np.float32)
        for s in range(0, len(self.ids), 250):
            e = self.E[s:s + 250].astype(np.float32)
            out[s:s + 250] = (e @ q.T + self.SH[s:s + 250].astype(np.float32)[..., None]) * self.SC[s:s + 250].astype(np.float32)[..., None]
        out[~self.valid] = -np.inf
        return out


def pa_tau(IM, men, alpha, m=20, min_n=5):
    """positive-anchored thresholds without abstention (same as lcr.region_evidence.fit_positive_anchored)."""
    tau, _ = re.fit_positive_anchored(IM, men, alpha, m=m, min_n=min_n, abstain_auroc=None, abstain_min=0)
    return tau


def report(name, con_true, con_sw, cx, extra=""):
    """con_true: contradicted flags with the true label (all claims); con_sw: with swapped label (valid on cx.m)."""
    pos = cx.ok & ~cx.nv; th = cx.kind == "thing"; m = cx.m
    f = lambda v: f"{v:.3f}"
    print(f"{name:34s} FCR {f(con_true[pos].mean())} thing {f(con_true[pos&th].mean())} clo {f(con_true[pos&cx.clo].mean())} "
          f"oth {f(con_true[pos&th&~cx.clo].mean())} | det {f(con_sw[m].mean())} hi {f(con_sw[m&cx.hibase].mean())} "
          f"clo {f(con_sw[m&cx.clo].mean())} oth {f(con_sw[m&~cx.clo].mean())} {extra}", flush=True)
    return dict(FCR=con_true[pos].mean(), FCR_thing=con_true[pos & th].mean(), det=con_sw[m].mean(), det_hi=con_sw[m & cx.hibase].mean())


ALPHAS = np.r_[0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
GROUPS = dict(animal=lambda c: re.GROUP[c] == "animal", vehicle=lambda c: re.GROUP[c] == "vehicle", clothing=lambda c: re.GROUP[c] == "clothing")


def curve_pa(cx, Xfold, targets=(0.03, 0.05, 0.08), name="", verbose=True, subsets=None):
    """Xfold(f) -> (N_img, C) image-level support score computed with training images of folds != f only.
    Positive-anchored per-class alpha-quantile thresholds, out-of-fold. Returns detection interpolated at thing-FCR targets."""
    Xs = [Xfold(f) for f in range(5)]
    pos = cx.ok & ~cx.nv & (cx.kind == "thing"); m = cx.m
    sub = dict(all=m, hibase=m & cx.hibase, clothing=m & cx.clo, nonclo=m & ~cx.clo)
    sub.update(subsets or {})
    pts = []
    for alpha in ALPHAS:
        ct = np.zeros(len(cx.n), bool); cs = np.zeros(len(cx.n), bool)
        for f in range(5):
            te = cx.fold == f; X = Xs[f]; tr = cx.img_fold != f
            tau = pa_tau(X[tr], cx.men[tr], alpha)
            ct[te] = cx.has[te] & (X[cx.ri[te], cx.k[te]] < tau[cx.k[te]])
            cs[te] = cx.has[te] & (X[cx.ri[te], cx.kk[te]] < tau[cx.kk[te]])
        pts.append([ct[pos].mean(), ct[pos & cx.clo].mean()] + [cs[s].mean() for s in sub.values()])
    pts = np.array(pts); out = {}
    for t in targets:
        out[t] = {k: float(np.interp(t, pts[:, 0], pts[:, 2 + i])) for i, k in enumerate(sub)}
        out[t]["fcr_clo"] = float(np.interp(t, pts[:, 0], pts[:, 1]))
    if verbose:
        print(f"{name:40s} " + " | ".join(f"FCR_thing={t}: " + " ".join(f"{k} {v:.3f}" for k, v in out[t].items()) for t in targets), flush=True)
    return out, pts


def curve_global(cx, S_true, S_sw, targets=(0.03, 0.05, 0.08), name="", subsets=None):
    """One global threshold on a per-claim support score (already out-of-fold). Lower = contradicted."""
    pos = cx.ok & ~cx.nv & (cx.kind == "thing") & cx.has; m = cx.m & cx.has
    sub = dict(all=m, hibase=m & cx.hibase, clothing=m & cx.clo, nonclo=m & ~cx.clo); sub.update(subsets or {})
    out = {}
    for t in targets:
        thr = np.quantile(S_true[pos], t)
        out[t] = {k: float((S_sw[s] < thr).mean()) for k, s in sub.items()}
        out[t]["fcr_clo"] = float((S_true[pos & cx.clo] < thr).mean())
    print(f"{name:40s} " + " | ".join(f"FCR_thing={t}: " + " ".join(f"{k} {v:.3f}" for k, v in out[t].items()) for t in targets), flush=True)
    return out
