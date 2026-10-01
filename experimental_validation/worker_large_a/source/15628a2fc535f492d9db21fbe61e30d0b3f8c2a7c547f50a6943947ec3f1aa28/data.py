"""Causal, client-local, typed entity snapshots from immutable event streams."""
import bootstrap
import hashlib
import json
import time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'CausalFed-TGNN-V4/runs/v4_scale'
ENTITIES = ['account_id', 'merchant_id', 'device_id', 'location']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def snapshots(df, window=8):
    """Read history before appending current event. No labels or future degrees."""
    amount = np.log1p(df.amount.to_numpy()).astype('float32')
    times = pd.to_datetime(df.timestamp, utc=True).astype('int64').to_numpy() / 86400e9
    banks = df.bank_id.to_numpy()
    history = np.zeros((len(df), len(ENTITIES), 4), dtype='float32')
    previous = np.full((len(df), len(ENTITIES)), -1, dtype='int64')
    bank_counts = {}
    pools = [{} for _ in ENTITIES]
    for i, values in enumerate(df[ENTITIES].itertuples(index=False, name=None)):
        b = banks[i]; rank = bank_counts.get(b, 0)
        for r, value in enumerate(values):
            key = (b, value)
            q, count = pools[r].get(key, (deque(maxlen=window), 0))
            if q:
                ids = list(q)
                previous[i, r] = ids[-1]
                history[i,r] = [float(amount[ids].mean()), np.log1p(1000*count/(rank+1)),
                                np.log1p(times[i]-times[ids[-1]]), 1.0]
            q.append(i); pools[r][key] = (q, count+1)
        bank_counts[b] = rank+1
    return history, previous

def prepare(scale, seed):
    start = time.perf_counter()
    path = SOURCE / f'synthetic_{scale}_seed{seed}.csv'
    old = json.loads((SOURCE/'RESULTS_MANIFEST.json').read_text())
    reference = next(r for r in old['results'] if r['scale']==scale and r['seed']==seed)
    checksum = sha(path)
    if checksum != reference['csv_sha256']: raise ValueError('Dataset checksum mismatch')
    df = pd.read_csv(path)
    assert len(df)==scale and df.transaction_id.is_unique and df.bank_id.nunique()==4
    source_sorted = bool(df.timestamp.is_monotonic_increasing)
    df = df.sort_values(['bank_id','timestamp','transaction_id'], kind='stable').reset_index(drop=True)
    rank = df.groupby('bank_id').cumcount().to_numpy()
    size = df.groupby('bank_id').bank_id.transform('size').to_numpy()
    phase = rank/(size-1)
    split = np.where(phase<.5, 0, np.where(phase<.7,1,2))
    unknown = df.fraud_type.isin(['D','E']).to_numpy()
    assert not unknown[split<2].any()
    for a,b in [(0,1),(1,2)]:
        assert df.loc[split==a,'timestamp'].max()<df.loc[split==b,'timestamp'].min()
    history, previous = snapshots(df)
    tr = split==0
    # Exact existing Context LR feature set; equivalent inputs for MLP and trees.
    ctx=[]
    for key in ['merchant_id','device_id']:
        counts=df.groupby(['bank_id',key],sort=False).cumcount().to_numpy()
        ctx.append(np.log1p(1000*counts/(rank+1)))
    bank = pd.Categorical(df.bank_id, categories=sorted(df.bank_id.unique())).codes.astype('int64')
    x = np.column_stack([np.log1p(df.amount),*ctx,*[(bank==j).astype(float) for j in [1,2,3]]]).astype('float32')
    mu=x[tr].mean(0); sd=x[tr].std(0); sd[sd<1e-6]=1
    hm=history[tr].mean(0); hs=history[tr].std(0); hs[hs<1e-6]=1
    gaps = pd.to_datetime(df.timestamp,utc=True).groupby(df.bank_id).diff().dt.total_seconds().dropna()/60
    audit=dict(scale=scale,seed=seed,dataset_hash=checksum,rows=len(df),banks=4,
               source_globally_sorted=source_sorted,duplicate_ids=0,unknown_train=int(unknown[tr].sum()),
               unknown_validation=int(unknown[split==1].sum()),bank_gap_minutes=sorted(gaps.unique().tolist()),
               split_hash=digest(list(zip(df.transaction_id.tolist(),split.tolist()))),
               supports={name:dict(rows=int((split==j).sum()),fraud=int(df.loc[split==j,'label'].sum()),
                         unknown=int(unknown[split==j].sum())) for j,name in enumerate(['train','validation','test'])},
               graph_seconds=time.perf_counter()-start)
    return dict(df=df,x=(x-mu)/sd,h=(history-hm)/hs,raw_history=history,previous=previous,
                y=df.label.to_numpy('int64'),bank=bank,split=split,unknown=unknown,audit=audit,
                normalization=dict(x_mean=mu.tolist(),x_sd=sd.tolist(),h_mean=hm.tolist(),h_sd=hs.tolist()))

def audit_all():
    records=[]; clients=[]; supports=[]
    for scale in [50000,100000,500000]:
        for seed in range(42,47):
            d=prepare(scale,seed); a=d['audit']; records.append(a)
            for bank,g in d['df'].groupby('bank_id'):
                clients.append(dict(scale=scale,seed=seed,bank=bank,rows=len(g),fraud_prevalence=g.label.mean(),
                   amount_mean=g.amount.mean(),amount_sd=g.amount.std(),merchant_top_share=g.merchant_id.value_counts(normalize=True).max(),
                   device_top_share=g.device_id.value_counts(normalize=True).max(),
                   **{f'mechanism_{m}':int((g.fraud_type==m).sum()) for m in ['legitimate','A','B','C','D','E']}))
                for j,s in enumerate(['train','validation','test']):
                    z=g.loc[d['split'][g.index]==j]
                    for m,n in z.fraud_type.value_counts().items():
                        supports.append(dict(scale=scale,seed=seed,bank=bank,split=s,mechanism=m,rows=int(n)))
            print(f'AUDIT PASS {scale} {seed}',flush=True)
    (ROOT/'DATA_AUDIT.json').write_text(json.dumps(records,indent=2))
    pd.json_normalize(records).to_csv(ROOT/'DATA_AUDIT.csv',index=False)
    pd.DataFrame(clients).to_csv(ROOT/'CLIENT_HETEROGENEITY.csv',index=False)
    pd.DataFrame(supports).to_csv(ROOT/'DATA_SUPPORT.csv',index=False)
    (ROOT/'DATA_AUDIT.md').write_text('# Data audit\n\nAll 15 original hashes, row counts, four banks, unique IDs, chronological split separation, and D/E exclusion passed. See DATA_AUDIT.json and DATA_SUPPORT.csv for exact populations. Source is bank-major; snapshots are bank-local and strictly historical. Original per-bank gaps are 1440 minutes. No raw data were modified.\n')

if __name__=='__main__': audit_all()
