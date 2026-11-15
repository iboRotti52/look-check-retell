import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, pickle
sys.path.insert(0, ROOT + "/scripts")
import _common
import numpy as np
from lcr import calibrate, cache, config
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX, KIND
S=ROOT + "/tmp"
D=pickle.load(open(ROOT + "/data/cache/review_fable_B_feat.pkl","rb")); xs=[x for x in D["xs"] if x["kind"]!="person"]; IM=D["image_max"]; refs=D["refs"]; ids=D["ids"]
C=len(CLASSES); GROUP=re.GROUP; A=calibrate.auroc
k=np.array([CLASS_INDEX[x["cls"]] for x in xs]); ri=np.array([x["ri"] for x in xs]); n=np.arange(len(xs))
pooled=np.stack([x["lg3"].max(0) for x in xs]); top1=np.stack([x["lg3"][0] for x in xs]); mu,sd=IM.mean(0),IM.std(0)+1e-6
rng=np.random.default_rng(0)
cls_by_group={}
for c in CLASSES: cls_by_group.setdefault(GROUP[c],[]).append(c)
sw=[];src=[]
for i,x in enumerate(xs):
    if not (x["ok"] and x["kind"]=="thing"): sw.append(-1);src.append(0); continue
    r=refs[x["id"]]
    cand=[a for a in x["blip_alts"] if a not in r and KIND[a]=="thing"]; s=1
    if not cand:
        s=2; cand=[c for c in cls_by_group[GROUP[x["cls"]]] if c not in r and c!=x["cls"]]
        cand=[cand[rng.integers(len(cand))]] if cand else []
    sw.append(CLASS_INDEX[cand[0]] if cand else -1); src.append(s)
sw=np.array(sw); src=np.array(src); m=sw>=0
grp=np.array([GROUP[x["cls"]] for x in xs])
def auc(M,mm): return A(np.r_[M[n,k][mm],M[n[mm],sw[mm]]],np.r_[np.ones(mm.sum()),np.zeros(mm.sum())].astype(bool))
def tpr(M,mm,f=.05): return (M[n[mm],sw[mm]]<np.quantile(M[n,k][mm],f)).mean()
Zi=(IM[ri]-mu)/sd; Ri=IM[ri]
print("controls for class-prior advantage (image-level): AUROC / TPR@FCR5%")
for nm,mm in [("all",m),("swap=captioner alt",m&(src==1)),("swap=random same-group",m&(src==2)),("swap class base logit >= true class (mu)",m&(mu[np.maximum(sw,0)]>=mu[k])),("swap class base logit < true",m&(mu[np.maximum(sw,0)]<mu[k]))]:
    print(f"{nm:42s} n={mm.sum():4d} z {auc(Zi,mm):.3f}/{tpr(Zi,mm):.3f}   raw {auc(Ri,mm):.3f}/{tpr(Ri,mm):.3f}")
print("by group (image-level raw / z AUROC; region raw):")
for g in ["animal","vehicle","play","clothing","handheld","structure","plant"]:
    mm=m&(grp==g); print(f"  {g:10s} n={mm.sum():4d} raw {auc(Ri,mm):.3f} z {auc(Zi,mm):.3f} region-raw {auc(pooled,mm):.3f} top1box-raw {auc(top1,mm):.3f}")
# ensemble queries, image-level
owl=cache.OwlCache("train")
for qf in ["class_queries_single.npy","class_queries_ensemble.npy"]:
    try: q=np.load(config.CACHE/"owl"/qf).astype(np.float32)
    except Exception as e: print(qf,"missing"); continue
    IMq=np.stack([re.proposal_logits(owl[int(i.rsplit('_',1)[1])],q).max(0) for i in ids]); M=IMq[ri]
    print(qf, "image raw AUROC all %.3f clothing %.3f non-clothing %.3f | TPR@5%% all %.3f clothing %.3f"%(auc(M,m),auc(M,m&(grp=="clothing")),auc(M,m&(grp!="clothing")),tpr(M,m),tpr(M,m&(grp=="clothing"))))
# end-to-end proposed rule on raw logits
print("\nproposed raw rule: contradicted iff image-max raw < a (and region-max < a); replace iff same-group class raw >= b on Look top-1 box else remove")
subm=np.array([[GROUP[c]==GROUP[CLASSES[j]] and c!=CLASSES[j] and KIND[c]!="person" for c in CLASSES] for j in range(C)])
for f in (.02,.05,.10):
    a=np.quantile(Ri[n,k][m],f)
    con_sw=Ri[n[m],sw[m]]<a; 
    for b in (-1.5,-.85,0.0):
        v=np.where(subm[sw[m]],top1[m],-np.inf); sj=v.argmax(1); ss=v.max(1)
        rep=con_sw&(ss>=b)
        # false replace on true label: contradicted & proposes something
        v0=np.where(subm[k[m]],top1[m],-np.inf)
        print(f"FCR {f:.2f} (a={a:.2f}) b={b:5.2f}: detect {con_sw.mean():.3f} replace {rep.mean():.3f} replace==true class {(rep&(sj==k[m])).mean():.3f} precision of substitute {((rep&(sj==k[m])).sum()/max(rep.sum(),1)):.3f}")
print("current bundle for reference: FCR 0.119, detect 0.561, replace 0.419, replace==true 0.291 (substitute precision 0.69)")
