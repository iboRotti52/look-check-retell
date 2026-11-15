import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, pickle
sys.path.insert(0, ROOT + "/scripts")
import _common
import numpy as np
from scipy.special import expit
from lcr import calibrate
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX, KIND
S=ROOT + "/tmp"
D=pickle.load(open(ROOT + "/data/cache/review_fable_B_feat.pkl","rb")); xs=[x for x in D["xs"] if x["kind"]!="person"]; IM=D["image_max"]; refs=D["refs"]; ids=D["ids"]
C=len(CLASSES); GROUP=re.GROUP; A=calibrate.auroc
k=np.array([CLASS_INDEX[x["cls"]] for x in xs]); ri=np.array([x["ri"] for x in xs]); n=np.arange(len(xs)); ok=np.array([x["ok"] for x in xs]); th=np.array([x["kind"]=="thing" for x in xs])
men=np.array([[c in refs[i] for c in CLASSES] for i in ids]); fold_img=np.array([calibrate.fold_of(i) for i in ids])
rng=np.random.default_rng(0)
cls_by_group={}
for c in CLASSES: cls_by_group.setdefault(GROUP[c],[]).append(c)
sw=[]
for i,x in enumerate(xs):
    if not (x["ok"] and x["kind"]=="thing"): sw.append(-1); continue
    r=refs[x["id"]]
    cand=[a for a in x["blip_alts"] if a not in r and KIND[a]=="thing"]
    if not cand:
        cand=[c for c in cls_by_group[GROUP[x["cls"]]] if c not in r and c!=x["cls"]]
        cand=[cand[rng.integers(len(cand))]] if cand else []
    sw.append(CLASS_INDEX[cand[0]] if cand else -1)
sw=np.array(sw); m=sw>=0; grp=np.array([GROUP[x["cls"]] for x in xs])
mu,sd=IM.mean(0),IM.std(0)+1e-6; med=np.median(IM,0); mad=1.4826*np.median(np.abs(IM-med),0)+1e-6
scales={"raw":IM,"z mean/std (current)":(IM-mu)/sd,"z median/MAD":(IM-med)/mad,"shift by class median":IM-med,"shift by class q25":IM-np.quantile(IM,.25,0)}
# null-referenced: mean/sd over images NOT mentioning the class
mun=np.array([IM[~men[:,j],j].mean() for j in range(C)]); sdn=np.array([IM[~men[:,j],j].std() for j in range(C)])+1e-6
scales["z on non-mention images"]=(IM-mun)/sdn; scales["shift by non-mention mean"]=IM-mun
# Platt with per-class offset on image-level mention, cross-fitted by image
for name in ("offset","offset_slope"):
    out=np.zeros_like(IM)
    cc=np.tile(np.arange(C),(len(ids),1))
    for f in range(5):
        tr=fold_img!=f
        cal=calibrate.make(name,1.0).fit(expit(IM[tr]).ravel(),cc[tr].ravel(),men[tr].ravel())
        out[~tr]=calibrate.logit(cal(expit(IM[~tr]).ravel(),cc[~tr].ravel())).reshape(-1,C)
    scales[f"Platt {name} on mention (OOF)"]=out
def auc(M,mm): return A(np.r_[M[n,k][mm],M[n[mm],sw[mm]]],np.r_[np.ones(mm.sum()),np.zeros(mm.sum())].astype(bool))
def tpr(M,mm,f=.05): return (M[n[mm],sw[mm]]<np.quantile(M[n,k][mm],f)).mean()
hi=m&(mu[np.maximum(sw,0)]>=mu[k])
print(f"{'scale (image-level)':34s} swapAUROC TPR@5% TPR@10% | hi-base-swap AUROC TPR@5% | animal clothing | proxyAUROC thing | elephant-FPR-like: frac images above FCR5% thr")
for nm,Mi in scales.items():
    M=Mi[ri]; t=np.quantile(M[n,k][m],.05)
    j=CLASS_INDEX["elephant"]; s=CLASS_INDEX["shirt"]
    print(f"{nm:34s} {auc(M,m):.3f} {tpr(M,m):.3f} {tpr(M,m,.1):.3f} | {auc(M,hi):.3f} {tpr(M,hi):.3f} | {auc(M,m&(grp=='animal')):.3f} {auc(M,m&(grp=='clothing')):.3f} | {A(M[n,k][th],ok[th]):.3f} | eleph {(Mi[:,j]>=t).mean():.3f} shirt-mentioned-recall {(Mi[men[:,s],s]>=t).mean():.3f}")
