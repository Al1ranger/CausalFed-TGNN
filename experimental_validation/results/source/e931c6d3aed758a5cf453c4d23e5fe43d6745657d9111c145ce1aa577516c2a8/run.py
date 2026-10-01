"""Recorded compact neural experiments. Test data never selects a checkpoint."""
import bootstrap
import argparse
import copy
import importlib.metadata
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from torch.nn import functional as F
from data import ROOT,prepare,sha,digest
from models import Model,objective,MODELS
from metrics import measure,threshold

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
OUT=ROOT/'results'
for directory in ['runs','checkpoints','predictions','logs','tables','figures']:
    (OUT/directory).mkdir(parents=True,exist_ok=True)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def hardware():
    return dict(python=sys.version,os=platform.platform(),cpu=platform.processor(),logical_cpus=psutil.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,device='cpu',cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,packages={m:importlib.metadata.version(m) for m in
        ['torch','torch-geometric','numpy','pandas','scipy','scikit-learn','pytest','psutil']},
        deterministic_algorithms=True,torch_threads=torch.get_num_threads())

def code_hash():
    return digest({p.name:sha(p) for p in sorted(ROOT.glob('*.py'))})

def tensor_data(d):
    return {k:torch.from_numpy(d[k].copy()) for k in ['x','h','y','bank']}

def batches(ids,rng,batch=2048):
    ix=rng.permutation(ids)
    return [ix[i:i+batch] for i in range(0,len(ix),batch)]

def predict(model,t,ids,batch=8192):
    model.eval(); ls=[]; zs=[]
    with torch.no_grad():
        for i in range(0,len(ids),batch):
            ix=ids[i:i+batch]; l,z,_=model(t['x'][ix],t['h'][ix]); ls.append(l); zs.append(z)
    return torch.cat(ls),torch.cat(zs)

def train_epoch(model,t,ids,optimizer,rng,global_state=None,prox=0.,control=None,flip=0.):
    model.train(); total=0.; last={}; count=0
    for ix in batches(ids,rng):
        y=t['y'][ix].clone()
        if flip:
            mask=torch.from_numpy(rng.random(len(ix))<flip); y[mask]=1-y[mask]
        optimizer.zero_grad()
        loss,last=objective(model,t['x'][ix],t['h'][ix],y,t['bank'][ix])
        if prox:
            loss=loss+prox/2*sum((p-global_state[n]).square().sum() for n,p in model.named_parameters())
        loss.backward()
        if control:
            for n,p in model.named_parameters():
                if p.grad is not None: p.grad.add_(control[n])
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient')
        optimizer.step(); total+=float(loss.detach())*len(ix); count+=1
    return total/len(ids),last,count

def evaluate(model,t,d,train_ids,val_ids,test_ids):
    tic=time.perf_counter()
    vl,vz=predict(model,t,val_ids); tl,tz=predict(model,t,test_ids); _,rz=predict(model,t,train_ids)
    vp=vl.softmax(1)[:,1].numpy(); tp=tl.softmax(1)[:,1].numpy()
    cut=threshold(d['y'][val_ids],vp)
    known=~d['unknown'][test_ids]; test_y=d['y'][test_ids]
    result={'known':measure(test_y[known],tp[known],cut,True),'all_late':measure(test_y,tp,cut,True)}
    result['per_bank']={str(b):measure(test_y[known&(d['bank'][test_ids]==b)],tp[known&(d['bank'][test_ids]==b)],cut,True) for b in range(4)}
    proto=torch.stack([rz[d['y'][train_ids]==c].mean(0) for c in [0,1]])
    scores={
      'energy':(-torch.logsumexp(vl,1).numpy(),-torch.logsumexp(tl,1).numpy()),
      'negative_margin':(-torch.abs(vl[:,1]-vl[:,0]).numpy(),-torch.abs(tl[:,1]-tl[:,0]).numpy()),
      'prototype_distance':(torch.cdist(vz,proto).min(1).values.numpy(),torch.cdist(tz,proto).min(1).values.numpy()),
      'amount':(d['df'].amount.to_numpy()[val_ids],d['df'].amount.to_numpy()[test_ids]),
    }
    result['openset']={}; pred={'transaction_id':d['df'].transaction_id.to_numpy()[test_ids],
         'bank':d['bank'][test_ids],'label':test_y,'unknown':d['unknown'][test_ids].astype(int),'probability':tp}
    for name,(v,s) in scores.items():
        q=float(np.quantile(v,.95,method='linear'))
        result['openset'][name]=measure(d['unknown'][test_ids].astype(int),s,q)
        result['openset'][name].update(direction='higher_is_unknown',normalization='none',
          calibration_population='known validation events only',threshold_rule='linear 95th percentile',
          validation_false_positive_rate=float((v>=q).mean()))
        pred[name]=s
    return result,pred,proto,time.perf_counter()-tic

def one(d,kind,seed,epochs=8,method='centralized',attack='clean'):
    scale=d['audit']['scale']; identity=f'{scale}_{seed}_{kind}_{method}_{attack}'
    dest=OUT/'runs'/f'{identity}.json'
    if dest.exists():
        existing=json.loads(dest.read_text())
        if existing['configuration']['epochs_or_rounds']!=epochs: raise ValueError('Existing run has a different budget; use a separate output directory')
        return existing
    seed_all(seed); rng=np.random.default_rng(seed); tic=time.perf_counter()
    config=dict(kind=kind,width=24,epochs_or_rounds=epochs,batch_size=2048,learning_rate=.001 if method=='centralized' else .03,
       optimizer='Adam' if method=='centralized' else 'SGD',method=method,attack=attack,
       local_epochs=1,prox_mu=.01 if method=='fedprox' else 0.,history_window=8,
       lambda_inv=.1 if kind not in ['no_inv','mlp','history_mlp','sage','temporal_sage','hetero_static','hgt'] and method=='centralized' else 0.,
       lambda_con=.01 if kind not in ['no_con','mlp','history_mlp','sage','temporal_sage','hetero_static','hgt'] and method=='centralized' else 0.,
       lambda_env=.1 if kind in ['full','no_inv','no_time','no_con','homogeneous'] else 0.,lambda_replay=0.,lambda_KL=0.,
       checkpoint_selection='lowest validation cross entropy',known_threshold='maximum validation F1; highest threshold on ties',
       attack_spec=None if attack=='clean' else dict(malicious_clients=[0],fraction=.25,start_round=1,
           flip_fraction=.5 if attack=='label_flip' else 0.,update_multiplier=-5 if attack=='poison' else 1,
           sybil_copies=2 if attack=='sybil' else 0,knowledge='own data and current broadcast weights',objective='untargeted degradation'))
    source_hash=code_hash(); source_dir=OUT/'source'/source_hash; source_dir.mkdir(parents=True,exist_ok=True)
    for source in ROOT.glob('*.py'):
        target=source_dir/source.name
        if not target.exists(): target.write_bytes(source.read_bytes())
    record=dict(experiment_id=identity,timestamp=datetime.now(timezone.utc).isoformat(),git_commit=None,
       code_hash=source_hash,source_snapshot=str(source_dir.relative_to(ROOT)),dataset_hash=d['audit']['dataset_hash'],split_hash=d['audit']['split_hash'],
       scale=scale,seed=seed,model=kind,configuration=config,configuration_hash=digest(config),status='RUNNING',
       supports=d['audit']['supports'],hardware=hardware(),architecture_scope='one-layer finite-history causal entity-snapshot graph',
       cross_environment_scope='same four banks in train and test; unseen-bank transfer not evaluated')
    dest.write_text(json.dumps(record,indent=2))
    try:
        model=Model(kind); t=tensor_data(d)
        train=np.flatnonzero(d['split']==0); val=np.flatnonzero(d['split']==1); test=np.flatnonzero(d['split']==2)
        best=float('inf'); best_state=None; logs=[]; aggregate_seconds=0.; local_seconds=0.; bytes_tx=0
        optimizer=torch.optim.Adam(model.parameters(),lr=.001) if method=='centralized' else None
        controls={b:{n:torch.zeros_like(p) for n,p in model.named_parameters()} for b in range(4)}
        server_control={n:torch.zeros_like(p) for n,p in model.named_parameters()}
        for epoch in range(epochs):
            ts=time.perf_counter(); detail={}
            if method=='centralized':
                loss,detail,_=train_epoch(model,t,train,optimizer,rng)
            else:
                global_state=copy.deepcopy(model.state_dict()); states=[]; counts=[]; losses=[]; client_log=[]
                for b in range(4):
                    local=copy.deepcopy(model); ids=train[d['bank'][train]==b]
                    opt=torch.optim.SGD(local.parameters(),lr=.03)
                    correction={n:server_control[n]-controls[b][n] for n in server_control} if method=='scaffold' else None
                    ll,ds,steps=train_epoch(local,t,ids,opt,rng,global_state,config['prox_mu'],correction,
                       .5 if b==0 and attack=='label_flip' else 0.)
                    state=copy.deepcopy(local.state_dict())
                    if method=='scaffold':
                        controls[b]={n:controls[b][n]-server_control[n]+(global_state[n]-state[n])/(steps*.03) for n in server_control}
                    if b==0 and attack in ['poison','sybil']:
                        state={n:global_state[n]-5*(v-global_state[n]) for n,v in state.items()}
                    states.append(state); counts.append(len(ids)); losses.append(ll)
                    client_log.append(dict(client=b,samples=len(ids),loss=ll,active_environments=ds['active_environments'],
                      irm_disabled_single_environment=True,contrastive_disabled_single_environment=True))
                local_seconds+=time.perf_counter()-ts; ag=time.perf_counter()
                if attack=='sybil': states+= [copy.deepcopy(states[0]),copy.deepcopy(states[0])]; counts += [counts[0],counts[0]]
                weights=np.array(counts)/sum(counts)
                merged={}
                for n in global_state:
                    stack=torch.stack([s[n] for s in states])
                    if method=='median': merged[n]=torch.quantile(stack,.5,dim=0)
                    else: merged[n]=sum(float(w)*s[n] for w,s in zip(weights,states))
                model.load_state_dict(merged)
                if method=='scaffold': server_control={n:sum(controls[b][n] for b in range(4))/4 for n in server_control}
                aggregate_seconds+=time.perf_counter()-ag
                payload=sum(p.numel()*p.element_size() for p in model.state_dict().values())
                bytes_tx+=payload*(4+len(states))
                if method=='scaffold': bytes_tx+=payload*8
                loss=float(np.mean(losses)); detail=dict(clients=client_log,aggregation_weights=weights.tolist() if method!='median' else None,
                    aggregation='coordinate median (average middle two)' if method=='median' else 'sample-weighted mean')
            vl,_=predict(model,t,val); vloss=float(F.cross_entropy(vl,t['y'][val]))
            vp=vl.softmax(1)[:,1].numpy(); vm=measure(d['y'][val],vp,.5,True)
            logs.append(dict(epoch_or_round=epoch+1,train_loss=loss,validation_loss=vloss,validation_metrics=vm,
                seconds=time.perf_counter()-ts,**detail))
            if vloss<best: best=vloss; best_state=copy.deepcopy(model.state_dict()); selected=epoch+1
        train_seconds=time.perf_counter()-tic
        model.load_state_dict(best_state)
        met,preds,proto,infer_seconds=evaluate(model,t,d,train,val,test)
        checkpoint=OUT/'checkpoints'/f'{identity}.pt'
        torch.save(dict(state_dict=best_state,configuration=config,normalization=d['normalization'],
           prototypes=proto,known_threshold=met['known']['threshold'],
           novelty_thresholds={k:v['threshold'] for k,v in met['openset'].items()}),checkpoint)
        reload=Model(kind); reload.load_state_dict(torch.load(checkpoint,weights_only=False)['state_dict'])
        l1,_=predict(model,t,test[:128]); l2,_=predict(reload,t,test[:128]); assert torch.equal(l1,l2)
        prediction=OUT/'predictions'/f'{identity}.csv'; pd.DataFrame(preds).to_csv(prediction,index=False)
        logpath=OUT/'logs'/f'{identity}.json'; logpath.write_text(json.dumps(logs,indent=2))
        record.update(status='COMPLETE',metrics=met,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_hash=sha(checkpoint),
          checkpoint_bytes=checkpoint.stat().st_size,parameter_count=sum(p.numel() for p in model.parameters()),
          predictions=str(prediction.relative_to(ROOT)),predictions_hash=sha(prediction),log=str(logpath.relative_to(ROOT)),log_hash=sha(logpath),
          selected_epoch_or_round=selected,completed_epochs_or_rounds=epochs,
          convergence_status='fixed budget completed; convergence not established',checkpoint_reload_exact=True,
          runtime=dict(train_seconds=train_seconds,inference_and_prototype_seconds=infer_seconds,graph_seconds=d['audit']['graph_seconds'],
             peak_process_memory_bytes=getattr(psutil.Process().memory_info(),'peak_wset',None),
             memory_scope='process lifetime high-water mark, includes preceding runs',peak_gpu_memory_bytes=None,
             local_training_seconds=local_seconds if method!='centralized' else None,
             aggregation_seconds=aggregate_seconds if method!='centralized' else None,
             communication_payload_bytes=bytes_tx if method!='centralized' else None))
        print(f'COMPLETE {identity} AUROC={met["known"]["auroc"]:.4f} AP={met["known"]["ap"]:.4f} train={train_seconds:.1f}s',flush=True)
    except Exception:
        record.update(status='FAILED',error=traceback.format_exc()); print(record['error'],flush=True)
    dest.write_text(json.dumps(record,indent=2,allow_nan=False))
    return record

def manifest():
    records=[json.loads(p.read_text()) for p in sorted((OUT/'runs').glob('*.json'))]
    obj=dict(created_utc=datetime.now(timezone.utc).isoformat(),study_status='PARTIAL',
      historical_manifests_preserved=True,runs=records,completed=sum(r['status']=='COMPLETE' for r in records),
      failed=sum(r['status']=='FAILED' for r in records),claims=dict(causal_identification=False,privacy_guarantees=False,
      unseen_bank_generalization=False,public_data_validation=False))
    (ROOT/'RESULTS_MANIFEST.json').write_text(json.dumps(obj,indent=2,allow_nan=False))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scales',nargs='+',type=int,default=[50000,100000,500000]);p.add_argument('--seeds',nargs='+',type=int,default=list(range(42,47)))
    p.add_argument('--models',nargs='+',default=MODELS);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--methods',nargs='+',default=['centralized']);p.add_argument('--attacks',nargs='+',default=['clean']);args=p.parse_args()
    env=hardware(); (ROOT/'environment.json').write_text(json.dumps(env,indent=2))
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env['packages'].items())+'\n')
    for scale in args.scales:
        for seed in args.seeds:
            d=prepare(scale,seed)
            for method in args.methods:
                for kind in args.models:
                    for attack in args.attacks: one(d,kind,seed,args.epochs,method,attack);manifest()

if __name__=='__main__': main()
