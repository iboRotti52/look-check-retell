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
print("nonperson claims",len(xs), "has region", sum(len(x["top"])>0 for x in xs))
mu,sd=IM.mean(0),IM.std(0)+1e-6
med=np.median(IM,0); mad=1.4826*np.median(np.abs(IM-med),0)+1e-6
TS,TC=1.3388,1.1280
k=np.array([CLASS_INDEX[x["cls"]] for x in xs]); ok=np.array([x["ok"] for x in xs]); kind=np.array([x["kind"] for x in xs])
n=np.arange(len(xs))
pooled=np.stack([x["lg3"].max(0) if len(x["top"]) else np.full(len(CLASSES),-np.inf) for x in xs])
top1=np.stack([x["lg3"][0] if len(x["top"]) else np.full(len(CLASSES),-np.inf) for x in xs])
ri=np.array([x["ri"] for x in xs])
z=(pooled-mu)/sd; zi=(IM[ri]-mu)/sd
sc=z[n,k]; si=zi[n,k]
allm=np.array([re.alternative_mask(c,"all") for c in CLASSES])[k]
sa=np.where(allm,z,-np.inf).max(1)
A=calibrate.auroc
th=kind=="thing"
print("\n== AUROC vs proxy (thing / stuff) ==")
for name,v in [("region max3 z",sc),("region top1 z",((top1-mu)/sd)[n,k]),("image max z",si),("region raw logit",pooled[n,k]),("image raw logit",IM[ri][n,k]),
               ("region robust z",((pooled-med)/mad)[n,k]),("image robust z",((IM[ri]-med)/mad)[n,k])]:
    print(f"{name:20s} thing {A(v[th],ok[th]):.3f}  stuff {A(v[~th],ok[~th]):.3f}")
print("\n== vacuity of alt condition (thing) ==")
print("P(sa_any>=ts) all:",(sa[th]>=TS).mean(), " | given sc<tc:",(sa[th&(sc<TC)]>=TS).mean())
nalt=(np.where(allm,z,-np.inf)>=TS).sum(1); print("median #classes>=ts in pooled region:",np.median(nalt[th]), "mean",nalt[th].mean())
person=CLASS_INDEX["person"]; print("P(person z>=ts in region):",(z[th,person]>=TS).mean())
print("\n== Look inertness: decisions ==")
keep=sc>=TS; els=~keep&(si>=TS); con=~keep&~els&(sc<TC)&(sa>=TS)
imgonly=(si<TC)
for nm,m in [("bundle contradicted",con&th),("image-only si<tc",imgonly&th),("image-only si<ts",(si<TS)&th)]:
    print(f"{nm:22s} n={m.sum()} proxy-wrong precision={(~ok[m]).mean():.3f} recall_wrong={m[th&~ok].mean():.3f}")
print("overlap: con within image-only(si<ts):",(con&th&(si<TS)).sum()/max((con&th).sum(),1)," image-only(si<ts) that are con:",(con&th).sum()/((si<TS)&th).sum())
print("keep:",(keep&th).sum(),"elsewhere:",(els&th).sum(), "band tc<=sc<ts & si<ts:",(th&~keep&~els&(sc>=TC)).sum())
# box sizes
ar=[ [ (b[2]-b[0])*(b[3]-b[1]) for b in x["boxes"]] for x in xs]
big=np.array([any(a>0.5 for a in r) for r in ar]); big1=np.array([len(r)>0 and r[0]>0.5 for r in ar])
print("\n== box sizes ==  any top3 area>0.5:",big[th].mean()," top1 area>0.5:",big1[th].mean(), "median top1 area", np.median([r[0] for r,t in zip(ar,th) if t and r]))
# where does support come from: which of 3 boxes
wh=np.array([int(x["lg3"][:,kk].argmax()) if len(x["top"]) else -1 for x,kk in zip(xs,k)])
print("supporting box rank among keep (thing):",np.bincount(wh[keep&th],minlength=3)/ (keep&th).sum())
# is image argmax box among top3
inn=np.array([x["argimg"] in x["top"] for x in xs]); print("image-argmax box is in Look top3 (thing, si>=ts):",inn[th&(si>=TS)].mean(), " is top1:",np.array([len(x['top'])>0 and x['argimg']==x['top'][0] for x in xs])[th&(si>=TS)].mean())
# per class frequency effect
print("\n== class scale: mu, sd, robust; frac images with z_img>=ts; frac refs mention ==")
men=np.array([[c in refs[i] for c in CLASSES] for i in ids])
ZI=(IM-mu)/sd; ZR=(IM-med)/mad
for c in ["dog","ball","shirt","frisbee","hat","elephant","grass","water","pants","jacket","camera","fence"]:
    j=CLASS_INDEX[c]; m=men[:,j]
    print(f"{c:9s} mu={mu[j]:6.2f} sd={sd[j]:5.2f} med={med[j]:6.2f} mad={mad[j]:5.2f} mention={m.mean():.3f} P(z>=ts)={ (ZI[:,j]>=TS).mean():.3f} recall(z>=ts|mention)={(ZI[m,j]>=TS).mean():.3f} FPR(z>=ts|no mention)={(ZI[~m,j]>=TS).mean():.3f} rawP>=.3|mention={(IM[m,j]>=np.log(.3/.7)).mean():.3f} |nomention={(IM[~m,j]>=np.log(.3/.7)).mean():.3f}")
print("\n== image-level detector vs mention, pooled over classes with >=20 mentions ==")
cs=[j for j in range(len(CLASSES)) if men[:,j].sum()>=20 and CLASSES[j]!="person"]
for nm,M in [("raw logit",IM),("mean/std z",ZI),("median/MAD z",ZR)]:
    print(f"{nm:14s} pooled AUROC {A(M[:,cs].ravel(),men[:,cs].ravel()):.4f}  mean per-class AUROC {np.mean([A(M[:,j],men[:,j]) for j in cs]):.4f}")
tk=[j for j in cs if KIND[CLASSES[j]]=="thing"]; st=[j for j in cs if KIND[CLASSES[j]]=="stuff"]
print("per-class AUROC thing mean",np.mean([A(IM[:,j],men[:,j]) for j in tk]),"stuff mean",np.mean([A(IM[:,j],men[:,j]) for j in st]))
worst=sorted([(A(IM[:,j],men[:,j]),CLASSES[j],int(men[:,j].sum())) for j in cs])[:25]; print("worst classes:",[(round(a,2),c,m) for a,c,m in worst])
# recall at threshold for mentioned classes by class
rec=sorted([((ZI[men[:,j],j]>=TS).mean(),CLASSES[j],int(men[:,j].sum())) for j in cs]); print("lowest image-level recall(z>=ts|mentioned):",[(round(a,2),c,m) for a,c,m in rec[:25]])
