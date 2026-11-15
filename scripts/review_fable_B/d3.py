import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, pickle
sys.path.insert(0, ROOT + "/scripts")
import _common
import numpy as np
from lcr import calibrate, cache, config, grounding, look
from lcr import region_evidence as re
from lcr.vocab import CLASSES, CLASS_INDEX, KIND
S=ROOT + "/tmp"
D=pickle.load(open(ROOT + "/data/cache/review_fable_B_feat.pkl","rb")); xs=[x for x in D["xs"] if x["kind"]!="person"]; IM=D["image_max"]; refs=D["refs"]; ids=D["ids"]
C=len(CLASSES); GROUP=re.GROUP
k=np.array([CLASS_INDEX[x["cls"]] for x in xs]); ok=np.array([x["ok"] for x in xs]); th=np.array([x["kind"]=="thing" for x in xs])
ri=np.array([x["ri"] for x in xs]); n=np.arange(len(xs))
pooled=np.stack([x["lg3"].max(0) for x in xs]); mu,sd=IM.mean(0),IM.std(0)+1e-6
rng=np.random.default_rng(0)
cls_by_group={}
for c in CLASSES: cls_by_group.setdefault(GROUP[c],[]).append(c)
sw=[]
for i,x in enumerate(xs):
    if not (x["ok"] and x["kind"]=="thing"): sw.append(-1); continue
    r=refs[x["id"]]
    cand=[a for a in x["blip_alts"] if a not in r and KIND[a]=="thing"]
    src=1
    if not cand:
        cand=[c for c in cls_by_group[GROUP[x["cls"]]] if c not in r and c!=x["cls"]]
        cand=[cand[rng.integers(len(cand))]] if cand else []
    sw.append(CLASS_INDEX[cand[0]] if cand else -1)
sw=np.array(sw); m=sw>=0
cloth=np.array([GROUP[x["cls"]]=="clothing" for x in xs])
print("== clean operating points: false-contradiction on reference-confirmed claims (FCR) vs detection of swapped labels (TPR); score = image-level max (veto makes rule image-level) ==")
def tpr_at(pos,neg,f):
    t=np.quantile(pos,f); return (neg<t).mean(), t
for nm,M in [("z (current scale)",(IM[ri]-mu)/sd),("raw logit",IM[ri])]:
    for sub,mm in [("all things",m),("clothing",m&cloth),("non-clothing",m&~cloth)]:
        pos=M[n,k][mm]; neg=M[n[mm],sw[mm]]
        print(f"{nm:18s} {sub:12s} n={mm.sum():4d} TPR@FCR2% {tpr_at(pos,neg,.02)[0]:.3f}  @5% {tpr_at(pos,neg,.05)[0]:.3f}  @10% {tpr_at(pos,neg,.10)[0]:.3f} (raw thr@5%: {tpr_at(pos,neg,.05)[1]:.2f})")
# per-class threshold = 5th percentile of class positives (image-level mention positives), raw
men=np.array([[c in refs[i] for c in CLASSES] for i in ids]); fold_img=np.array([calibrate.fold_of(i) for i in ids]); fold=fold_img[ri]
for alpha in (.02,.05,.10):
    fc=[];tp=[]
    for f in range(5):
        tr=fold_img!=f
        allpos=np.concatenate([IM[tr&men[:,j],j] for j in range(C) if KIND[CLASSES[j]]=="thing"])
        gq=np.quantile(allpos,alpha)
        thr=np.array([ (lambda p: (len(p)*np.quantile(p,alpha)+20*gq)/(len(p)+20) if len(p)>=5 else gq)(IM[tr&men[:,j],j]) for j in range(C)])
        mm=m&(fold==f)
        fc+= list(IM[ri][n,k][mm]<thr[k[mm]]); tp+=list(IM[ri][n[mm],sw[mm]]<thr[sw[mm]])
    print(f"per-class positive-quantile threshold alpha={alpha}: FCR {np.mean(fc):.3f} TPR {np.mean(tp):.3f} | clothing FCR {np.mean(np.array(fc)[cloth[m][np.argsort(np.argsort(np.where(m)[0]))]] if False else 0):.0f}")
# ---------- Look with human boxes ----------
print("\n== Look vs Flickr30k-Entities boxes (B, thing claims with GT) ==")
owl=cache.OwlCache("train"); q=np.load(config.CACHE/"owl"/"class_queries_single.npy").astype(np.float32)
rows=cache.load_json(config.CACHE/"pipeline"/"A_B_look4.json"); maps=np.load(config.CACHE/"pipeline"/"A_B_look4.maps.npz"); owner={tuple(o):i for i,o in enumerate(maps["owner"])}
R=[]
for i,x in enumerate(xs):
    if x["kind"]!="thing": continue
    gt=grounding.class_boxes(x["id"]).get(x["cls"])
    if gt is None or not len(gt): continue
    prop=owl[int(x["id"].rsplit("_",1)[1])]; boxes=prop["boxes"].astype(np.float32); obj=prop["objectness"].astype(np.float32)
    giou=np.max([look.iou(b,boxes) for b in gt],axis=0)
    att=maps["maps"][owner[(x["ri"],x["ci"])]].astype(np.float32)
    lgc=re.proposal_logits(prop,q[k[i]][None])[:,0]
    y,xx=np.unravel_index(att.argmax(),att.shape); py,px=(y+.5)/24,(xx+.5)/24
    inpk=(boxes[:,0]<=px)&(px<=boxes[:,2])&(boxes[:,1]<=py)&(py<=boxes[:,3])
    rs=look.region_score(att,boxes,obj,alpha=.5,gamma=.5)
    rs_pk=np.where(inpk,rs,-1)
    mass=look.attention_mass(att,boxes)
    top=x["top"]
    # hybrid: among top-K by r, detector picks claim-best
    def hyb(K):
        o=np.argsort(-rs)[:K]; return giou[o[np.argmax(lgc[o])]]>=.5
    garea=float(np.max((gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1])))
    R.append(dict(top1=giou[top[0]]>=.5,top3=(giou[top]>=.5).any(),oracle=giou.max()>=.5,det=giou[lgc.argmax()]>=.5,point=grounding.point_in_boxes(py,px,gt),
                  pk1=giou[rs_pk.argmax()]>=.5,h3=hyb(3),h5=hyb(5),h10=hyb(10),garea=garea,group=GROUP[x["cls"]],
                  top5=(giou[np.argsort(-rs)[:5]]>=.5).any(),top10=(giou[np.argsort(-rs)[:10]]>=.5).any(),
                  energy=float(mass[giou.argmax()]) , small=garea<0.02))
def show(name,rs_):
    f=lambda key: np.mean([r[key] for r in rs_])
    print(f"{name:14s} n={len(rs_):4d} point {f('point'):.3f} top1 {f('top1'):.3f} top1+peak-in-box {f('pk1'):.3f} top3 {f('top3'):.3f} top5 {f('top5'):.3f} top10 {f('top10'):.3f} | det-pick within top3 {f('h3'):.3f} top5 {f('h5'):.3f} top10 {f('h10'):.3f} | detector-argmax(all 100) {f('det'):.3f} proposal-oracle {f('oracle'):.3f}")
show("all",R)
for g in ["animal","vehicle","play","clothing","handheld","structure","plant","animal_gear"]:
    show(g,[r for r in R if r["group"]==g])
show("GT<2% area",[r for r in R if r["small"]]); show("GT 2-10%",[r for r in R if .02<=r["garea"]<.1]); show("GT>=10%",[r for r in R if r["garea"]>=.1])
