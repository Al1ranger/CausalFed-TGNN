import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
import pandas as pd
import torch
from data import snapshots
from models import Model,objective,MODELS
from metrics import measure,threshold

def frame():
    return pd.DataFrame(dict(amount=[1.,2.,3.,4.],timestamp=pd.date_range('2024-01-01',periods=4,tz='UTC').astype(str),
        bank_id=['A']*4,account_id=['a']*4,merchant_id=['m']*4,device_id=['d']*4,location=['l']*4))

def test_temporal_leakage():
    d=frame(); h,p=snapshots(d); altered=d.copy();altered.loc[3,'amount']=1e6
    h2,_=snapshots(altered)
    np.testing.assert_array_equal(h[:3],h2[:3]);np.testing.assert_array_equal(h,snapshots(d.assign(label=[1,0,1,0]))[0])
    assert (p[1:]<np.arange(1,4)[:,None]).all()
    for n in range(1,5): np.testing.assert_array_equal(h[:n],snapshots(d.iloc[:n])[0])

def test_bank_isolation():
    d=frame(); d.loc[2:,'bank_id']='B';h,p=snapshots(d)
    assert (p[2]==-1).all() and (h[2]==0).all()

def test_ties_and_direction():
    assert measure([0,1],[1,1],1)['auroc']==.5
    assert measure([0,1],[0,1],.5)['auroc']==1
    assert measure([0,1],[1,0],.5)['auroc']==0
    assert measure([0,1],[1,1],1)['ap']==.5
    assert measure([0,1],[1,1],1)['fpr95']==1
    assert threshold([0,1,1],[.1,.6,.8])==.6

def test_models_gradients_and_determinism():
    torch.set_num_threads(2)
    x=torch.randn(16,6); h=torch.randn(16,4,4); y=torch.arange(16)%2; b=torch.arange(16)%4
    for kind in MODELS:
        torch.manual_seed(42);m=Model(kind);loss,stats=objective(m,x,h,y,b);loss.backward()
        assert torch.isfinite(loss) and m.head.weight.grad.abs().sum()>0
        if kind!='mlp':
            altered=h.clone(); altered[:,0,0]+=10
            assert not torch.equal(m(x,h)[0],m(x,altered)[0])
        state=m.state_dict(); fresh=Model(kind);fresh.load_state_dict(state)
        assert torch.equal(m(x,h)[0],fresh(x,h)[0])
        torch.manual_seed(42); m2=Model(kind)
        assert torch.equal(m(x,h)[0],m2(x,h)[0])

def test_single_environment_disables_irm():
    m=Model('full');_,s=objective(m,torch.randn(8,6),torch.randn(8,4,4),torch.arange(8)%2,torch.zeros(8,dtype=torch.long))
    assert s['irm'] is None and s['active_environments']==1

def test_time_ablation():
    m=Model('no_time');x=torch.randn(8,6);h=torch.randn(8,4,4);h2=h.clone();h2[:,:,2]+=100
    assert torch.equal(m(x,h)[0],m(x,h2)[0])
