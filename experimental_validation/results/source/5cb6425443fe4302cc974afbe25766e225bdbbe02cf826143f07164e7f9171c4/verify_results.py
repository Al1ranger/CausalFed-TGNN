"""Independent artifact hashes and metric reconstruction from saved predictions."""
import bootstrap
import json
from pathlib import Path
import pandas as pd
from data import ROOT,sha,digest
from metrics import measure

def verify():
    runs=[json.loads(p.read_text()) for p in (ROOT/'results/runs').glob('*.json')]
    checks=[]
    for r in runs:
        if r['status']!='COMPLETE': continue
        for name,key in [('checkpoint','checkpoint_hash'),('predictions','predictions_hash'),('log','log_hash')]:
            assert sha(ROOT/r[name])==r[key],(r['experiment_id'],name)
        snapshot=ROOT/r['source_snapshot']
        assert digest({p.name:sha(p) for p in sorted(snapshot.glob('*.py'))})==r['code_hash']
        assert digest(r['configuration'])==r['configuration_hash']
        df=pd.read_csv(ROOT/r['predictions']);known=df.unknown==0
        rebuilt=measure(df.loc[known,'label'],df.loc[known,'probability'],r['metrics']['known']['threshold'],True)
        for k in ['auroc','ap','f1','brier','ece']:
            assert abs(rebuilt[k]-r['metrics']['known'][k])<1e-6,(r['experiment_id'],k)
        for score,m in r['metrics']['openset'].items():
            rebuilt=measure(df.unknown,df[score],m['threshold'])
            for k in ['auroc','ap','fpr95','f1']: assert abs(rebuilt[k]-m[k])<1e-6,(r['experiment_id'],score,k)
        checks.append(r['experiment_id'])
    result=dict(status='PASS',verified_completed_runs=len(checks),checks=checks,
                scope='source/config/checkpoint/prediction/log hashes; known and open-set metrics reconstructed from saved predictions')
    (ROOT/'RESULT_VERIFICATION.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

if __name__=='__main__':verify()
