"""Generate all numerical tables and figures from the canonical run records."""
import bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import ROOT
from run import manifest

TABLES=ROOT/'results/tables'; FIGS=ROOT/'results/figures'
METRICS=['auroc','ap','f1','precision','recall','balanced_accuracy','brier','ece']
LABELS={'mlp':'MLP','history_mlp':'History MLP','sage':'GraphSAGE','temporal_sage':'Temporal SAGE',
 'hetero_static':'Static HGT','hgt':'HGT','full':'CausalFed variant','no_inv':'No IRM','no_env':'No env branch',
 'no_time':'No elapsed time','no_con':'No contrastive','homogeneous':'Collapsed HGT'}

def group(frame,keys,metrics):
    rows=[]
    for identity,g in frame.groupby(keys,dropna=False):
        if not isinstance(identity,tuple):identity=(identity,)
        r=dict(zip(keys,identity));r['seeds_completed']=int(g.seed.nunique());r['evidence_status']='COMPLETE' if r['seeds_completed']==5 else 'PARTIAL'
        r['positive_class']='fraud A/B/C' if 'score' not in keys else 'withheld mechanisms D/E'
        r['population']='known test events' if 'score' not in keys else 'all late test events'
        r['threshold_policy']='validation maximum F1' if 'score' not in keys else 'known validation 95th percentile'
        r['summary_policy']='mean and sample SD only for five distinct seeds'
        for m in metrics:
            r[m+'_mean']=float(g[m].mean()) if len(g)==5 and g[m].notna().all() else None
            r[m+'_sd']=float(g[m].std(ddof=1)) if len(g)==5 and g[m].notna().all() else None
        rows.append(r)
    return pd.DataFrame(rows)

def savefig(fig,name):
    fig.tight_layout()
    for ext in ['svg','pdf','png']:fig.savefig(FIGS/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def plot_models(summary,metric,name):
    clean=summary[(summary.method=='centralized')&(summary.attack=='clean')]
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,scale in zip(axes,[50000,100000,500000]):
        s=clean[(clean.scale==scale)&clean[metric+'_mean'].notna()]
        ax.bar(np.arange(len(s)),s[metric+'_mean'],yerr=s[metric+'_sd'],color='#346a89',capsize=2)
        ax.set_xticks(np.arange(len(s)),[LABELS.get(v,v) for v in s.model],rotation=75,ha='right')
        ax.set_title(f'{scale:,} events');ax.set_ylabel(metric.upper());ax.grid(axis='y',alpha=.2)
    fig.suptitle('Known-class test performance · mean ± sample SD · five seeds')
    savefig(fig,name)

def main():
    manifest(); m=json.loads((ROOT/'RESULTS_MANIFEST.json').read_text());runs=m['runs'];valid=[r for r in runs if r['status']=='COMPLETE']
    rows=[];opens=[];banks=[];cost=[];curves=[]
    for r in valid:
        base=dict(experiment_id=r['experiment_id'],scale=r['scale'],seed=r['seed'],model=r['model'],
                   method=r['configuration']['method'],attack=r['configuration']['attack'])
        rows.append(dict(**base,**r['metrics']['known']))
        for score,v in r['metrics']['openset'].items():opens.append(dict(**base,score=score,**v))
        for bank,v in r['metrics']['per_bank'].items():banks.append(dict(**base,bank=bank,**v))
        cost.append(dict(**base,**r['runtime'],parameter_count=r['parameter_count'],checkpoint_bytes=r['checkpoint_bytes']))
        for ep in json.loads((ROOT/r['log']).read_text()):
            curves.append(dict(**base,step=ep['epoch_or_round'],train_loss=ep['train_loss'],validation_loss=ep['validation_loss'],
                               validation_auroc=ep['validation_metrics']['auroc'],validation_ap=ep['validation_metrics']['ap']))
    raw=pd.DataFrame(rows);op=pd.DataFrame(opens);bk=pd.DataFrame(banks);co=pd.DataFrame(cost);cu=pd.DataFrame(curves)
    if not len(raw):return
    for name,df in [('known_runs',raw),('openset_runs',op),('per_bank_runs',bk),('computational_cost_runs',co),('convergence_runs',cu)]:df.to_csv(TABLES/(name+'.csv'),index=False)
    keys=['scale','model','method','attack']; summary=group(raw,keys,METRICS); summary.to_csv(TABLES/'centralized_and_federated_summary.csv',index=False)
    ops=group(op,keys+['score'],['auroc','ap','f1','fpr95','recall','fpr']);ops.to_csv(TABLES/'Table_E_openset.csv',index=False)
    bks=group(bk,keys+['bank'],METRICS);bks.to_csv(TABLES/'per_bank_summary.csv',index=False)
    for name,mask in [('Table_B_baselines',summary.model.isin(['mlp','history_mlp'])),
       ('Table_C_graph_models',summary.model.isin(['sage','temporal_sage','hetero_static','hgt'])),
       ('Table_D_full_model',summary.model=='full'),('Table_G_federated',summary.method!='centralized'),('Table_H_attacks',summary.attack!='clean')]:
        summary[mask].to_csv(TABLES/(name+'.csv'),index=False)
    group(co,keys,['train_seconds','inference_and_prototype_seconds','graph_seconds','parameter_count','checkpoint_bytes',
        'peak_process_memory_bytes']).to_csv(TABLES/'Table_I_cost.csv',index=False)
    if (ROOT/'DATA_AUDIT.csv').exists():pd.read_csv(ROOT/'DATA_AUDIT.csv').to_csv(TABLES/'Table_A_data.csv',index=False)
    deltas=[]
    for _,r in raw[(raw.method=='centralized')&(raw.attack=='clean')].iterrows():
        if r.model not in ['no_inv','no_env','no_time','no_con','homogeneous']:continue
        match=raw[(raw.scale==r.scale)&(raw.seed==r.seed)&(raw.model=='full')&(raw.method=='centralized')&(raw.attack=='clean')]
        if len(match)!=1:continue
        base=match.iloc[0]
        d={k:r[k] for k in keys+['seed']}
        for metric in ['auroc','ap','f1']:
            d[metric+'_delta']=r[metric]-base[metric]
            d[metric+'_relative_delta']=d[metric+'_delta']/base[metric] if base[metric] else None
        deltas.append(d)
    if deltas:
        dd=pd.DataFrame(deltas);dd.to_csv(TABLES/'ablation_paired_runs.csv',index=False)
        group(dd,keys,['auroc_delta','ap_delta','f1_delta']).to_csv(TABLES/'Table_F_ablations.csv',index=False)
    attack_deltas=[]
    for _,r in raw[raw.attack!='clean'].iterrows():
        clean=raw[(raw.scale==r.scale)&(raw.seed==r.seed)&(raw.model==r.model)&(raw.method==r.method)&(raw.attack=='clean')]
        if len(clean)==1:attack_deltas.append({**{k:r[k] for k in keys+['seed']},**{v+'_degradation':float(clean.iloc[0][v]-r[v]) for v in ['auroc','ap','f1']}})
    if attack_deltas:pd.DataFrame(attack_deltas).to_csv(TABLES/'attack_paired_degradation.csv',index=False)
    for metric,name in [('auroc','known_auroc'),('ap','known_ap'),('f1','known_f1'),('ece','calibration_ece')]:plot_models(summary,metric,name)
    for metric in ['auroc','fpr95']:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,scale in zip(axes,[50000,100000,500000]):
            s=ops[(ops.scale==scale)&(ops.model=='full')&(ops.method=='centralized')&(ops.attack=='clean')]
            ax.bar(s.score,s[metric+'_mean'],yerr=s[metric+'_sd'],capsize=3,color='#627c4e');ax.tick_params(axis='x',rotation=35);ax.set_title(f'{scale:,}');ax.set_ylabel(metric.upper())
        fig.suptitle('Unknown-event detection · full variant · five-seed mean ± sample SD');savefig(fig,'openset_'+metric)
    fig,ax=plt.subplots(figsize=(8,4))
    s=bks[(bks.model=='full')&(bks.method=='centralized')&(bks.attack=='clean')]
    for scale,g in s.groupby('scale'):ax.errorbar(g.bank.astype(int),g.auroc_mean,yerr=g.auroc_sd,marker='o',label=f'{scale:,}')
    ax.set(xlabel='Bank index',ylabel='Known AUROC',xticks=range(4));ax.legend();savefig(fig,'per_bank')
    fig,ax=plt.subplots(figsize=(8,4))
    for key,g in cu[(cu.model=='full')&(cu.attack=='clean')].groupby(['scale','method']):
        z=g.groupby('step').validation_loss.agg(['mean','std']);ax.plot(z.index,z['mean'],label=f'{key[0]:,} {key[1]}')
    ax.set(xlabel='Epoch or communication round',ylabel='Validation cross entropy');ax.legend(fontsize=7);savefig(fig,'convergence')
    fig,ax=plt.subplots(figsize=(8,4))
    for model,g in co[(co.method=='centralized')&(co.attack=='clean')&co.model.isin(['mlp','sage','hgt','full'])].groupby('model'):
        z=g.groupby('scale').train_seconds.agg(['mean','std']);ax.errorbar(z.index,z['mean'],yerr=z['std'],marker='o',label=LABELS[model])
    ax.set(xlabel='Events',ylabel='Training seconds including validation');ax.legend();savefig(fig,'runtime_scalability')
    fed=summary[(summary.method!='centralized')&(summary.attack=='clean')]
    if len(fed):
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(fed.method,fed.auroc_mean,yerr=fed.auroc_sd,capsize=3);ax.set_ylabel('Known AUROC');ax.set_title('50k federated simulation');savefig(fig,'federated_methods')
        fig,ax=plt.subplots(figsize=(7,4))
        for method,g in cu[(cu.method!='centralized')&(cu.attack=='clean')].groupby('method'):
            z=g.groupby('step').validation_auroc.agg(['mean','std']);ax.plot(z.index,z['mean'],label=method)
        ax.set(xlabel='Communication round',ylabel='Validation AUROC');ax.legend();savefig(fig,'federated_rounds')
    if deltas:
        fig,ax=plt.subplots(figsize=(8,4));z=pd.DataFrame(deltas).groupby('model').auroc_delta.agg(['mean','std'])
        ax.bar(z.index,z['mean'],color='#8c6256');ax.axhline(0,color='black',lw=.6);ax.set_ylabel('AUROC ablation minus full');ax.set_title('Descriptive pooled-scale deltas; use Table F for scale-specific SD');savefig(fig,'ablation_deltas')
    status=['# Experiment status','','| Experiment | Status | Seeds completed | Dataset | Result files | Notes |','|---|---|---|---|---|---|']
    for _,r in summary.iterrows():status.append(f'| {r.model}/{r.method}/{r.attack} | {r.evidence_status} | {r.seeds_completed}/5 | {r.scale} original synthetic | results/runs | Eight-epoch/round screening |')
    for name,why in [('Adaptive aggregation','No defined project algorithm'),('Continual replay and drift','Not implemented/executed'),('Temporal stress v2','Not executed'),('Public PaySim','No validated complete input'),('IEEE-CIS','Authentication and terms required'),('Explainability quality','Not executed'),('Unseen-bank transfer','Not executed')]:
        status.append(f'| {name} | PENDING | 0 | — | — | {why} |')
    for r in runs:
        if r['status']!='COMPLETE':status.append(f'| {r["experiment_id"]} | {r["status"]} | 0 | {r["scale"]} | results/runs | See run error or active status |')
    (ROOT/'EXPERIMENT_STATUS.md').write_text('\n'.join(status),encoding='utf-8')
    lines=['# Observed results','',f'{len(valid)} completed recorded fits. No statistical significance claim. Mean ± sample SD requires five valid seeds.','']
    for _,r in summary[(summary.model=='full')&(summary.method=='centralized')&(summary.attack=='clean')].iterrows():
        if r.seeds_completed==5:lines.append(f'- {r.scale:,} events: full variant known AUROC {r.auroc_mean:.4f} ± {r.auroc_sd:.4f}, AP {r.ap_mean:.4f} ± {r.ap_sd:.4f}, F1 {r.f1_mean:.4f} ± {r.f1_sd:.4f} (n=5).')
    lines+=['','The study remains partial: finite-history architecture, fixed training budget, no unseen-bank transfer, no public-data validation, and no continual or realistic burst-timing experiment. Interpret comparisons using the history-matched MLP and scale-specific paired ablations.']
    (ROOT/'RESULTS_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
    env=json.loads((ROOT/'environment.json').read_text())
    (ROOT/'ENVIRONMENT_REPORT.md').write_text('# Execution environment\n\n```json\n'+json.dumps(env,indent=2)+'\n```\n\nCPU-only; two threads per process. Concurrent worker workloads affect wall time. Deterministic algorithms enabled; exact cross-platform reproducibility is not guaranteed.\n')
    print(f'Generated reports for {len(valid)} completed fits',flush=True)

if __name__=='__main__':main()
