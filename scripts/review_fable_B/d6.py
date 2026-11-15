import os; ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
S=ROOT + "/tmp"
src=open(S+"/d5.py").read().split("mu,sd=IM.mean(0)")[0]
exec(src)
prior=np.log(men.mean(0)+1e-4)
M=np.tile(prior,(len(ids),1))[ri]
def auc(M,mm): return A(np.r_[M[n,k][mm],M[n[mm],sw[mm]]],np.r_[np.ones(mm.sum()),np.zeros(mm.sum())].astype(bool))
print("class-prior-only (no image) swap AUROC:",auc(M,m), " proxy AUROC thing:",A(M[n,k][th],ok[th]))
# within-class AUROC (scale-invariant): positives = in-refs claims of class c; negatives = other claims swapped to c
Ri=IM[ri]; res=[]
for c in range(C):
    pos=Ri[n,k][m&(k==c)]; neg=Ri[n[m&(sw==c)],c]
    if len(pos)>=15 and len(neg)>=15: res.append((A(np.r_[pos,neg],np.r_[np.ones(len(pos)),np.zeros(len(neg))].astype(bool)),CLASSES[c],len(pos),len(neg)))
print("within-class AUROC (true c vs swapped-to-c), classes with >=15 each:",[(round(a,2),c,p,q) for a,c,p,q in sorted(res)])
print("weighted mean:",np.average([r[0] for r in res],weights=[min(r[2],r[3]) for r in res]))
