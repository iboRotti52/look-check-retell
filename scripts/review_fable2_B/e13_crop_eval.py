"""Does re-encoding the attended box (same frozen OWLv2) separate true from swapped labels better than the cached
full-image proposals? Same-confusion-set swaps with a finished crop. Threshold-free comparisons."""
import pickle, shutil, collections
from common import *
src = config.CACHE / "review_fable2_B_crop.pkl"; tmp = str(src) + ".read"; shutil.copyfile(src, tmp); done = pickle.load(open(tmp, "rb"))
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
cnt = np.load(config.CACHE / "review_fable2_B_mencount.npy")
same = np.array([re.CONFUSION_OF.get(names[a]) is not None and re.CONFUSION_OF.get(names[a]) == re.CONFUSION_OF.get(names[b]) for a, b in zip(cx.k, cx.kk)])
idx = np.array(sorted(i for i in done if same[i])); print("crops:", len(done), "same-set:", len(idx), "clothing:", int(cx.clo[idx].sum()))
CM = np.stack([done[i]["max"] for i in idx]); CB = np.stack([done[i]["big"] for i in idx])
B1 = L[cx.ri[idx], cx.top[idx, 0]]; I = IM[cx.ri[idx]]; k = cx.k[idx]; kk = cx.kk[idx]; n = np.arange(len(idx)); clo = cx.clo[idx]
area = np.array([(done[i]["crop"][2] - done[i]["crop"][0]) * (done[i]["crop"][3] - done[i]["crop"][1]) / (done[i]["size"][0] * done[i]["size"][1]) for i in idx])
def se(p): return np.sqrt(p * (1 - p) / max(len(n), 1))
def pair(name, X):
    w = X[n, k] > X[n, kk]
    small = area < 0.15
    print(f"{name:36s} pairwise true>swap: clothing {w[clo].mean():.3f} (n={clo.sum()}) | non-clothing {w[~clo].mean():.3f} (n={(~clo).sum()}) | attended box <15% of image: {w[small].mean():.3f} (n={small.sum()}) | >=15%: {w[~small].mean():.3f}")
    return w
w_img = pair("full image: image-max", I); w_b1 = pair("full image: Look top-1 box", B1); w_c = pair("crop: max over crop proposals", CM); pair("crop: largest crop proposal", CB)
w_mx = pair("max(Look top-1 box, crop max)", np.maximum(B1, CM)); pair("mean(Look top-1 box, crop max)", (B1 + CM) / 2); pair("mean(image-max, crop max)", (I + CM) / 2)
# paired comparison crop vs top-1 box (McNemar counts)
for g, s in (("clothing", clo), ("non-clothing", ~clo)):
    print(f"{g}: crop right & box wrong {int((w_c & ~w_b1 & s).sum())} | box right & crop wrong {int((~w_c & w_b1 & s).sum())}")
# margin-based detection at a fixed false-contradiction rate, within-sample (no thresholds from elsewhere):
# score = claim logit - best confusion-set rival logit (same evidence source). positives: true label; negatives: swapped label.
cm_t = np.array([re.confusion_mask(c) for c in CLASSES])
def margin(X, lab): return X[n, lab] - np.where(cm_t[lab], X, -np.inf).max(1)
for name, X in (("Look top-1 box", B1), ("image-max", I), ("crop max", CM), ("mean(box, crop)", (B1 + CM) / 2), ("mean(image-max, crop)", (I + CM) / 2)):
    for g, s in (("clothing", clo), ("non-clothing", ~clo)):
        mt, ms = margin(X, k)[s], margin(X, kk)[s]
        print(f"  margin vs rivals [{name:22s}] {g:12s} AUROC {A(np.r_[mt, ms], np.r_[np.ones(len(mt), bool), np.zeros(len(ms), bool)]):.3f} | detection at 5% false contradictions {np.mean(ms < np.quantile(mt, .05)):.3f} at 10% {np.mean(ms < np.quantile(mt, .10)):.3f}")
# raw claim score (absence-style) for the same sample
for name, X in (("image-max", I), ("crop max", CM)):
    for g, s in (("clothing", clo), ("non-clothing", ~clo)):
        mt, ms = X[n, k][s], X[n, kk][s]
        print(f"  claim logit    [{name:22s}] {g:12s} pooled AUROC {A(np.r_[mt, ms], np.r_[np.ones(len(mt), bool), np.zeros(len(ms), bool)]):.3f}")
for c in "shirt jacket sweater vest dress suit uniform".split():
    j = CLASS_INDEX[c]; p = k == j; s = kk == j
    if p.sum() >= 8 and s.sum() >= 8:
        f = lambda X: A(np.r_[X[p, j], X[s, j]], np.r_[np.ones(p.sum(), bool), np.zeros(s.sum(), bool)])
        print(f"  within-class {c:8s} (pos {p.sum()}, swaps {s.sum()}): image-max {f(I):.2f} | Look box {f(B1):.2f} | crop max {f(CM):.2f} | mean(box,crop) {f((B1+CM)/2):.2f}")
