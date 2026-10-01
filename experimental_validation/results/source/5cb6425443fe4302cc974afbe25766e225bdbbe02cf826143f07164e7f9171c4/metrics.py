import bootstrap
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve

def threshold(y,s):
    p,r,t=precision_recall_curve(y,s)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15)
    return float(t[np.flatnonzero(f==f.max())[-1]])

def measure(y,s,t,probability=False):
    y=np.asarray(y); s=np.asarray(s); pred=s>=t
    P=int(y.sum()); N=int(len(y)-P)
    tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum())
    out=dict(support=len(y),positive=P,negative=N,threshold=float(t),
        f1=2*tp/max(P+tp+fp,1),precision=tp/max(tp+fp,1),recall=tp/P if P else None,
        fpr=fp/N if N else None,auroc=None,ap=None,fpr95=None,balanced_accuracy=None)
    if P and N:
        fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
        out.update(auroc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)),
                   fpr95=float(fpr[np.flatnonzero(tpr>=.95)[0]]),balanced_accuracy=.5*(tp/P+1-fp/N))
    if probability:
        out['brier']=float(np.mean((s-y)**2)); ece=0.
        for j in range(10):
            m=(s>=j/10)&(s<((j+1)/10) if j<9 else s<=1)
            if m.any(): ece+=m.mean()*abs(s[m].mean()-y[m].mean())
        out['ece']=float(ece)
    return out
