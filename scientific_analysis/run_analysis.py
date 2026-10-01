"""Reproduce the synthetic temporal baseline study; never overwrite historical runs."""
from pathlib import Path
import hashlib, json, platform, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'CausalFed-TGNN-V4/runs/v4_scale'
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def curve(y,s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    assert len(y)==len(s) and np.isfinite(s).all() and set(np.unique(y))<={0,1}
    order=np.argsort(-s,kind='stable'); yy=y[order]; ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(yy)[ends]; fp=ends+1-tp
    return ss[ends],tp,fp,int(y.sum()),int(len(y)-y.sum())

def metrics(y,s,threshold):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    _,tp,fp,P,N=curve(y,s); assert P>0 and N>0
    tpr=np.r_[0,tp/P]; fpr=np.r_[0,fp/N]
    auc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
    ap=float(np.sum(np.diff(np.r_[0,tp/P])*tp/(tp+fp)))
    pred=s>=threshold; a=int(np.sum(pred & (y==1))); b=int(np.sum(pred & (y==0))); c=P-a
    return dict(auroc=auc,ap=ap,fpr95=float(fp[np.flatnonzero(tp/P>=.95)[0]]/N),
        f1=2*a/max(2*a+b+c,1),precision=a/max(a+b,1),recall=a/P,fpr=b/N,
        balanced_accuracy=(a/P+1-b/N)/2, support=len(y),positive=P,negative=N,
        tp=a,fp=b,fn=c,tn=N-b,threshold=float(threshold))

def threshold_f1(y,s):
    thresholds,tp,fp,P,N=curve(y,s)
    f=2*tp/np.maximum(P+tp+fp,1)
    return float(thresholds[np.argmax(f)])  # descending scores: highest threshold wins ties

def self_test():
    for y,s,auc,ap in [([0,1],[0,1],1,1),([0,1],[1,0],0,.5),([0,1],[1,1],.5,.5),([0,0,1,1],[.1,.4,.35,.8],.75,5/6)]:
        m=metrics(y,s,.5); assert abs(m['auroc']-auc)<1e-12 and abs(m['ap']-ap)<1e-12
    rng=np.random.default_rng(2026)
    for _ in range(100):
        y=np.r_[0,1,rng.integers(0,2,18)]; s=rng.integers(0,5,20)
        pos=s[y==1,None]; neg=s[y==0][None,:]
        expected=float(np.mean((pos>neg)+.5*(pos==neg)))
        assert abs(metrics(y,s,2)['auroc']-expected)<1e-12
        perm=rng.permutation(len(y)); assert abs(metrics(y,s,2)['ap']-metrics(y[perm],s[perm],2)['ap'])<1e-12
    x=np.column_stack([np.ones(200),np.linspace(-2,2,200)])
    y=(x[:,1]>0).astype(int); w,info=fit(x,y)
    assert w[1]>0 and info['gradient_max']<1e-7
    return 'PASS: known-answer AUROC/AP, 100 tied-score pairwise AUROC checks, AP permutation invariance, logistic convergence'

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-40,40)))

def fit(x,y):
    reg=np.eye(x.shape[1])*1e-3; reg[0,0]=0
    w=np.zeros(x.shape[1]); w[0]=np.log(y.mean()/(1-y.mean()))
    def objective(w):
        z=x@w
        return float(np.mean(np.logaddexp(0,z)-y*z)+.5*w@reg@w)
    for it in range(60):
        p=sigmoid(x@w); g=x.T@(p-y)/len(y)+reg@w
        if np.max(np.abs(g))<1e-8: break
        h=(x.T*(p*(1-p)))@x/len(y)+reg+np.eye(x.shape[1])*1e-12
        step=np.linalg.solve(h,g); rate=1.; old=objective(w)
        while objective(w-rate*step)>old and rate>1e-8: rate*=.5
        w-=rate*step
    g=x.T@(sigmoid(x@w)-y)/len(y)+reg@w
    assert np.max(np.abs(g))<1e-7
    return w,dict(iterations=it+1,gradient_max=float(np.max(np.abs(g))),objective=objective(w))

def run():
    test_status=self_test(); source=json.loads((DATA/'RESULTS_MANIFEST.json').read_text())
    records=[]; baseline=[]; diagnostic=[]; prevalence=[]; bins=[]; bank_metrics=[]; calibration=[]; curves=[]
    start=time.perf_counter()
    for old in sorted(source['results'],key=lambda r:(r['scale'],r['seed'])):
        n,seed=old['scale'],old['seed']; path=DATA/f'synthetic_{n}_seed{seed}.csv'
        digest=sha(path); assert digest==old['csv_sha256']
        df=pd.read_csv(path); assert len(df)==n and df.transaction_id.is_unique
        globally_sorted=bool(df.timestamp.is_monotonic_increasing)
        df=df.sort_values(['bank_id','timestamp','transaction_id'],kind='stable').reset_index(drop=True)
        rank=df.groupby('bank_id').cumcount(); size=df.groupby('bank_id').bank_id.transform('size')
        phase=rank/(size-1); train=(phase<.5).to_numpy(); val=((phase>=.5)&(phase<.7)).to_numpy(); test=(phase>=.7).to_numpy()
        unknown=df.fraud_type.isin(['D','E']).to_numpy(); y=df.label.to_numpy()
        assert not unknown[train|val].any() and np.all(train.astype(int)+val+test==1)
        assert df.loc[train,'timestamp'].max()<df.loc[val,'timestamp'].min()<df.loc[test,'timestamp'].min()
        assert df.loc[val,'timestamp'].max()<df.loc[test,'timestamp'].min()
        gaps=[]
        for _,b in df.groupby('bank_id'):
            tt=[datetime.fromisoformat(v) for v in b.timestamp]
            gaps.extend({(v-u).total_seconds()/60 for u,v in zip(tt,tt[1:])})
        amount=np.log1p(df.amount.to_numpy())
        hist=[]
        for key in ['merchant_id','device_id']:
            past=df.groupby(['bank_id',key],sort=False).cumcount().to_numpy()
            hist.append(np.log1p(1000*past/(rank.to_numpy()+1)))
        banks=sorted(df.bank_id.unique()); bankone=np.column_stack([(df.bank_id==b).to_numpy(dtype=float) for b in banks[1:]])
        features={'Amount LR':amount[:,None], 'Context LR':np.column_stack([amount,*hist,bankone])}
        split=np.select([train,val],['train','validation'],default='test')
        for name,mask in [('train',train),('validation',val),('test',test)]:
            for b in banks:
                m=mask & (df.bank_id==b).to_numpy()
                prevalence.append(dict(scale=n,seed=seed,split=name,bank=b,rows=int(m.sum()),fraud_rate=float(y[m].mean()),unknown_rate=float(unknown[m].mean())))
        for j in range(10):
            m=((phase>=j/10)&(phase<(j+1)/10 if j<9 else phase<=1)).to_numpy()
            bins.append(dict(scale=n,seed=seed,bin=j+1,rows=int(m.sum()),fraud_rate=float(y[m].mean()),unknown_rate=float(unknown[m].mean())))
        raw=df.amount.to_numpy(); threshold=float(np.quantile(raw[val],.95,method='linear'))
        diag=metrics(unknown[test],raw[test],threshold)
        full=metrics(unknown,raw,threshold)
        diagnostic.append(dict(scale=n,seed=seed,**diag,legacy_full_auroc=old['open_set_diagnostic']['unknown_auroc'],corrected_full_auroc=full['auroc'],validation_false_alarm=float(np.mean(raw[val]>=threshold))))
        audit=dict(scale=n,seed=seed,csv_sha256=digest,train=int(train.sum()),validation=int(val.sum()),test=int(test.sum()),
            known_test=int(np.sum(test & ~unknown)),unknown_test=int(np.sum(test&unknown)),fraud_rate=float(y.mean()),
            globally_sorted_in_source=globally_sorted,within_bank_gap_minutes=sorted(set(gaps)),
            duplicate_transaction_ids=0,unknown_in_train_validation=0,models={})
        for name,xx in features.items():
            mu=xx[train].mean(axis=0); sd=xx[train].std(axis=0); sd[sd<1e-12]=1
            x=np.column_stack([np.ones(n),(xx-mu)/sd]); fit_start=time.perf_counter()
            w,info=fit(x[train],y[train]); seconds=time.perf_counter()-fit_start
            p=sigmoid(x@w); t=threshold_f1(y[val],p[val]); known=test&~unknown
            for pop,mask in [('known',known),('all_late',test)]:
                met=metrics(y[mask],p[mask],t)
                baseline.append(dict(scale=n,seed=seed,model=name,population=pop,**met,brier=float(np.mean((p[mask]-y[mask])**2)),fit_seconds=seconds))
            for b in banks:
                m=known&(df.bank_id==b).to_numpy(); bank_metrics.append(dict(scale=n,seed=seed,model=name,bank=b,**metrics(y[m],p[m],t)))
            modeldata=dict(weights=w.tolist(),training_mean=mu.tolist(),training_sd=sd.tolist(),threshold=t,fit_seconds=seconds,**info)
            modelpath=OUT/f'model_{n}_{seed}_{name.replace(" ","_")}.json'; modelpath.write_text(json.dumps(modeldata,indent=2))
            audit['models'][name]=dict(file=modelpath.name,sha256=sha(modelpath),**info)
            predfile=OUT/f'predictions_{n}_{seed}_{name.replace(" ","_")}.csv'
            pd.DataFrame(dict(transaction_id=df.loc[test,'transaction_id'],bank=df.loc[test,'bank_id'],label=y[test],unknown=unknown[test].astype(int),probability=p[test])).to_csv(predfile,index=False)
            audit['models'][name]['predictions_sha256']=sha(predfile)
            if n==500000:
                scores,tp,fp,P,N=curve(y[known],p[known]); ids=np.unique(np.r_[0,np.linspace(0,len(tp)-1,180).astype(int),len(tp)-1])
                for j in ids: curves.append(dict(scale=n,seed=seed,model=name,fpr=float(fp[j]/N),tpr=float(tp[j]/P),precision=float(tp[j]/(tp[j]+fp[j]))))
                # Equal-width probability bins, retain counts and sums for weighted pooling.
                for j in range(10):
                    m=known&(p>=j/10)&(p<((j+1)/10) if j<9 else p<=1)
                    calibration.append(dict(scale=n,seed=seed,model=name,bin=j,rows=int(m.sum()),sum_probability=float(p[m].sum()),positives=int(y[m].sum())))
        records.append(audit)
        print(f'PASS {n:,} seed {seed}: two trained baselines, split/hash checks, diagnostics',flush=True)
    frames={'baselines':baseline,'diagnostics':diagnostic,'prevalence':prevalence,'temporal_bins':bins,'bank_metrics':bank_metrics,'calibration':calibration,'curves':curves}
    for name,rows in frames.items(): pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),source_manifest_sha256=sha(DATA/'RESULTS_MANIFEST.json'),
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),
        scope='synthetic pooled logistic baselines and metric/data diagnostics; not CausalFed-TGNN or federated training',
        protocol=dict(split='per-bank chronological 50/20/30 percent; global non-overlap asserted',seeds=[42,43,44,45,46],l2=1e-3,regularization='mean log loss plus lambda/2 times squared non-intercept coefficients',
            max_iterations=60,tolerance=1e-8,known_threshold='validation maximum F1, largest threshold on ties',novelty='raw amount high means unknown; validation 95th percentile',
            features={'Amount LR':['log1p(amount)'],'Context LR':['log1p(amount)','log1p(1000 * prior client merchant count / (prior client events + 1))','log1p(1000 * prior client device count / (prior client events + 1))','Bank_B','Bank_C','Bank_D']},
            history='prequential unlabeled history includes earlier validation/test events; excludes current and future events',seed_summary='mean and sample standard deviation ddof=1; no confidence intervals or significance claims'),
        tests=test_status,records=records,wall_seconds=time.perf_counter()-start)
    manifest['result_hashes']={p.name:sha(p) for p in OUT.glob('*.csv')}
    (OUT/'ANALYSIS_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(test_status); print('Completed',len(records),'datasets,',len(baseline)//2,'model fits',flush=True)

if __name__=='__main__': run()
