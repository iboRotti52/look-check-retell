from common import *
cx = Ctx(); L = cx.logits(); IM = L.max(1); names = np.array(CLASSES); m = cx.m
tau = pa_tau(IM, cx.men, 0.09)
q = lambda a, ps: " ".join(f"{v:6.2f}" for v in np.quantile(a, ps)) if len(a) else "-"
print("class       npos | pos q05 q10 q25 q50 | unmentioned q50 q90 q95 q99 | swaps-to-class q10 q50 q75 q90 | tau(.09) | det")
for c in "cat horse sheep cow goat duck bird motorcycle truck dog shirt jacket sweater suit dress hat helmet toy ball wall leash uniform bench ramp".split():
    j = CLASS_INDEX[c]; p = IM[cx.men[:, j], j]; u = IM[~cx.men[:, j], j]; s = IM[cx.ri[m & (cx.kk == j)], j]
    print(f"{c:11s} {len(p):4d} | {q(p, [.05,.1,.25,.5])} | {q(u, [.5,.9,.95,.99])} | {q(s, [.1,.5,.75,.9])} | {tau[j]:6.2f} | {(s < tau[j]).mean() if len(s) else float('nan'):.2f}  n_sw={len(s)}")
# for dog->animal swaps: at the box where the swapped class peaks, what does the detector prefer?
sel = m & (cx.k == CLASS_INDEX["dog"]) & np.isin(names[cx.kk], ["cat", "horse", "sheep", "cow", "goat"])
ri = cx.ri[sel]; kk = cx.kk[sel]
b = L[ri, :, kk].argmax(1)
dog = L[ri, b, CLASS_INDEX["dog"]]; sw = L[ri, b, kk]
print("dog->animal swaps n", sel.sum(), ": at swapped class's best box, dog logit - swapped logit quantiles", q(dog - sw, [.05, .1, .25, .5, .75]))
print(" share where dog > swapped at that box:", (dog > sw).mean(), " share missed by tau:", (sw >= tau[kk]).mean(), " missed AND dog>swapped+1:", ((sw >= tau[kk]) & (dog > sw + 1)).mean())
# same margin for true positives of those animals: at claim's best box, claim - best rival (animal set)
an = [CLASS_INDEX[c] for c in re.CONFUSION["animal"]]
for c in ["horse", "dog", "bird", "cow"]:
    j = CLASS_INDEX[c]; imgs = np.where(cx.men[:, j])[0]; bb = L[imgs, :, j].argmax(1)
    riv = np.max([L[imgs, bb, a] for a in an if a != j], 0)
    print(f" positives {c}: claim - best animal rival at claim's best box:", q(L[imgs, bb, j] - riv, [.02, .05, .1, .25, .5]))
