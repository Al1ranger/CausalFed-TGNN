"""Validate historical logistic evidence without refitting or overwriting it."""
import bootstrap
import json
import pandas as pd
from data import ROOT,sha
from metrics import measure

def main():
    folder=ROOT.parent/'scientific_analysis/results';path=folder/'ANALYSIS_MANIFEST.json'
    m=json.loads(path.read_text())
    assert sha(folder.parent/'run_analysis.py')==m['script_sha256']
    assert sha(ROOT.parent/'CausalFed-TGNN-V4/runs/v4_scale/RESULTS_MANIFEST.json')==m['source_manifest_sha256']
    for name,checksum in m['result_hashes'].items():assert sha(folder/name)==checksum,name
    table=pd.read_csv(folder/'baselines.csv');verified=[]
    for r in m['records']:
        for name,entry in r['models'].items():
            assert sha(folder/entry['file'])==entry['sha256']
            p=folder/f'predictions_{r["scale"]}_{r["seed"]}_{name.replace(" ","_")}.csv'
            assert sha(p)==entry['predictions_sha256'];df=pd.read_csv(p);known=df.unknown==0
            model=json.loads((folder/entry['file']).read_text());z=measure(df.loc[known,'label'],df.loc[known,'probability'],model['threshold'],True)
            row=table[(table.scale==r['scale'])&(table.seed==r['seed'])&(table.model==name)&(table.population=='known')].iloc[0]
            for metric in ['auroc','ap','f1','brier']:assert abs(z[metric]-row[metric])<1e-10
            verified.append(dict(scale=r['scale'],seed=r['seed'],model=name,**z))
    pd.DataFrame(verified).to_csv(ROOT/'results/tables/historical_baselines_verified.csv',index=False)
    result=dict(status='PASS',scope='historical evidence independently hash checked and metrics reconstructed; not new fits',
       manifest_path=str(path),manifest_sha256=sha(path),verified_fits=len(verified),
       original_protocol=m['protocol'],source_environment=m['environment'])
    (ROOT/'HISTORICAL_VERIFICATION.json').write_text(json.dumps(result,indent=2))
    print('PASS historical logistic evidence:',len(verified),'fits')

if __name__=='__main__':main()
