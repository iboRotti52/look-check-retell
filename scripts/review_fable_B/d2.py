import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, pickle
sys.path.insert(0, ROOT + "/scripts")
import _common
import numpy as np
from lcr import calibrate
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX, KIND
S=ROOT + "/tmp"
D=pickle.load(open(ROOT + "/data/cache/review_fable_B_feat.pkl","rb")); xs=[x for x in D["xs"] if x["kind"]!="person"]; IM=D["image_max"]; refs=D["refs"]; ids=D["ids"]
C=len(CLASSES); A=calibrate.auroc
men=np.array([[c in refs[i] for c in CLASSES] for i in ids])
fold_img=np.array([calibrate.fold_of(i) for i in ids])
k=np.array([CLASS_INDEX[x["cls"]] for x in xs]); ok=np.array([x["ok"] for x in xs]); kind=np.array([x["kind"] for x in xs]); th=kind=="thing"
ri=np.array([x["ri"] for x in xs]); n=np.arange(len(xs)); fold=fold_img[ri]
pooled=np.stack([x["lg3"].max(0) for x in xs]); top1=np.stack([x["lg3"][0] for x in xs])
mu,sd=IM.mean(0),IM.std(0)+1e-6
# positive-quantile scale: rank of logit within class positives (image-max of images mentioning class), cross-fitted; shrink to global for rare
def posrank(train):
    glob=np.sort(np.concatenate([IM[train&men[:,j],j]-np.median(IM[train&men[:,j],j]) for j in range(C) if (train&men[:,j]).sum()>=20]))
    tabs={}
    for j in range(C):
        p=IM[train&men[:,j],j]
        tabs[j]=np.sort(p) if len(p)>=20 else None
    return tabs
def apply(tabs,v,j):
    t=tabs[j]
    if t is None: return np.nan
    return np.searchsorted(t,v,side="right")/len(t)
def pr_matrix(M):   # M (N,C) logits for claims -> pos-rank with OOF tables
    out=np.full(M.shape,np.nan)
    for f in range(5):
        tabs=posrank(fold_img!=f)
        for i in np.where(fold==f)[0]:
            for j in range(C):
                t=tabs[j]
                if t is not None: out[i,j]=np.searchsorted(t,M[i,j],side="right")/len(t)
    return out
PR_img=pr_matrix(IM[ri]); PR_reg=pr_matrix(pooled); PR_top1=pr_matrix(top1)
have=~np.isnan(PR_img[n,k])
print("claims with class having >=20 positive images:",have.mean())
print("== AUROC vs proxy on those claims (thing/stuff) ==")
for nm,v in [("image raw",IM[ri][n,k]),("image z",((IM[ri]-mu)/sd)[n,k]),("image posrank",PR_img[n,k]),("region raw",pooled[n,k]),("region posrank",PR_reg[n,k])]:
    print(f"{nm:15s} thing {A(v[th&have],ok[th&have]):.3f} stuff {A(v[~th&have],ok[~th&have]):.3f}")
# ---------- synthetic swap: clean negatives -------------
rng=np.random.default_rng(0)
GROUP=re.GROUP
FINE={}
for g,names in dict(upper="shirt jacket sweater dress suit uniform costume vest bikini swimsuit",lower="shorts pants skirt",head="hat helmet headband bandanna",
    face="glasses goggles mask",foot="boot shoe",carry="backpack bag",animal="dog horse bird duck cat cow sheep goat fish camel elephant",
    wheeled="bicycle motorcycle scooter unicycle",road="car truck bus train",boatg="boat canoe kayak raft",board="skateboard snowboard surfboard ski sled",
    throw="ball frisbee toy stick bone balloon",seat="bench chair couch bed",built="fence wall railing",bldg="building house tent",plant="tree bush flower").items():
    for c in names.split(): FINE[c]=g
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
sw=np.array(sw); m=sw>=0
print("\nswap set n=",m.sum(), "(positives: in-refs thing claims; negative: same region relabelled with captioner alt / same-group class not in refs)")
def Z(M): return (M-mu)/sd
for nm,pos,neg in [("region z",Z(pooled)[n,k][m],Z(pooled)[n[m],sw[m]]),("region raw",pooled[n,k][m],pooled[n[m],sw[m]]),("image raw",IM[ri][n,k][m],IM[ri][n[m],sw[m]]),
                   ("image z",Z(IM[ri])[n,k][m],Z(IM[ri])[n[m],sw[m]]),("region posrank",PR_reg[n,k][m],PR_reg[n[m],sw[m]]),("image posrank",PR_img[n,k][m],PR_img[n[m],sw[m]]),("top1box raw",top1[n,k][m],top1[n[m],sw[m]])]:
    g=~np.isnan(pos)&~np.isnan(neg)
    print(f"{nm:15s} AUROC true-vs-swapped {A(np.r_[pos[g],neg[g]],np.r_[np.ones(g.sum()),np.zeros(g.sum())].astype(bool)):.3f} (n={g.sum()})")
# bundle rule on swapped claims
TS,TC=1.3388,1.1280
z=Z(pooled); zi=Z(IM[ri])
def actions(kk):
    sc=z[n,kk]; si=zi[n,kk]
    allm=np.ones((len(xs),C),bool); allm[n,kk]=False
    sa=np.where(allm,z,-np.inf).max(1)
    subm=np.array([[GROUP[c]==GROUP[CLASSES[j]] and c!=CLASSES[j] and KIND[c]!="person" for c in CLASSES] for j in range(C)])[kk]
    zs=np.where(subm,z,-np.inf); ss=zs.max(1); sj=zs.argmax(1)
    keep=sc>=TS; els=~keep&(si>=TS); con=~keep&~els&(sc<TC)&(sa>=TS); rep=con&(ss>=TS)
    return keep,els,con,rep,sj
keep,els,con,rep,sj=actions(np.where(m,sw,k))
print(f"\nBUNDLE on swapped (wrong by construction): keep {keep[m].mean():.3f} elsewhere {els[m].mean():.3f} contradicted {con[m].mean():.3f} (replace {rep[m].mean():.3f}; replace with TRUE class {(rep&(sj==k))[m].mean():.3f}) skip {(~keep&~els&~con)[m].mean():.3f}")
keep0,els0,con0,rep0,sj0=actions(k)
print(f"BUNDLE on the same claims, true label: keep {keep0[m].mean():.3f} elsewhere {els0[m].mean():.3f} contradicted {con0[m].mean():.3f} skip {(~keep0&~els0&~con0)[m].mean():.3f}")
# per group breakdown
for g in ["animal","vehicle","play","clothing","handheld","structure"]:
    gm=m&np.array([GROUP[x["cls"]]==g for x in xs])
    print(f"  {g:10s} n={gm.sum():4d} swapped: keep {keep[gm].mean():.2f} contr {con[gm].mean():.2f} replace-true {(rep&(sj==k))[gm].mean():.2f} | true label: keep {keep0[gm].mean():.2f} contr {con0[gm].mean():.2f}")
# alternative rule: raw logits, substitute = argmax same-group (or FINE) on a single box
def variant(M3fn,name,fine):
    hit=[];tot=0
    for i in np.where(m)[0]:
        x=xs[i]; lg=x["lg3"]; kk=sw[i]
        g=GROUP[CLASSES[kk]]
        cand=[j for j,c in enumerate(CLASSES) if j!=kk and KIND[c]!="person" and (FINE.get(c,"-"+c)==FINE.get(CLASSES[kk],"?") if fine else GROUP[c]==g)]
        if not cand: continue
        v=M3fn(lg,i)
        hit.append(cand[int(np.argmax(v[cand]))]==k[i])
    print(f"{name:45s} top substitute == true class: {np.mean(hit):.3f} (n={len(hit)})")
variant(lambda lg,i:Z(lg.max(0)),"z, max over 3 boxes, same GROUP (current)",False)
variant(lambda lg,i:lg.max(0),"raw, max over 3 boxes, same GROUP",False)
variant(lambda lg,i:lg[0],"raw, top-1 Look box, same GROUP",False)
variant(lambda lg,i:Z(lg[0]),"z, top-1 Look box, same GROUP",False)
variant(lambda lg,i:PR_top1[i],"posrank, top-1 box, same GROUP",False)
variant(lambda lg,i:PR_reg[i],"posrank, max3, same GROUP",False)
variant(lambda lg,i:lg[0],"raw, top-1 Look box, FINE set",True)
variant(lambda lg,i:Z(lg.max(0)),"z, max3, FINE set",True)
# ---------- mixture proportion of truly wrong among proxy-wrong (thing) ----------
print("\n== mixture estimate: share of proxy-wrong thing claims that behave like swapped negatives (image raw logit) ==")
pos=IM[ri][n,k][m]; neg=IM[ri][n[m],sw[m]]; unk=IM[ri][n,k][th&~ok]
for t in [-4,-3,-2,-1,0]:
    a,b,c=(pos<t).mean(),(neg<t).mean(),(unk<t).mean(); print(f"t={t}: P(s<t) pos {a:.3f} neg {b:.3f} proxy-wrong {c:.3f} -> pi={(c-a)/(b-a):.2f}")
