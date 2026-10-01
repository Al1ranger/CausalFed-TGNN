"""Disjoint-scale worker with atomic publication of completed records."""
import bootstrap
import argparse
import json
import os
import run
from data import ROOT,prepare
from models import MODELS

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scale',type=int,required=True);p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--methods',nargs='+',default=['centralized']);p.add_argument('--models',nargs='+',default=MODELS)
    p.add_argument('--attacks',nargs='+',default=['clean']);p.add_argument('--name',required=True);a=p.parse_args()
    run.OUT=ROOT/('worker_'+a.name)
    for sub in ['runs','checkpoints','predictions','logs','tables','figures']:(run.OUT/sub).mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        d=prepare(a.scale,seed)
        for method in a.methods:
            for model in a.models:
                for attack in a.attacks:
                    identity=f'{a.scale}_{seed}_{model}_{method}_{attack}'
                    target=ROOT/'results/runs'/f'{identity}.json'
                    if target.exists(): continue
                    record=run.one(d,model,seed,8,method,attack)
                    temp=target.with_suffix('.publishing')
                    temp.write_text(json.dumps(record,indent=2,allow_nan=False));os.replace(temp,target)
