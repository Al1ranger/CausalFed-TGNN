import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
import torch
from models import Model,objective
from run import train_epoch

def test_scaffold_gradient_correction_matches_one_step_algebra():
    torch.manual_seed(12);model=Model('mlp');other=Model('mlp');other.load_state_dict(model.state_dict())
    t=dict(x=torch.randn(12,6),h=torch.randn(12,4,4),y=torch.arange(12)%2,bank=torch.zeros(12,dtype=torch.long))
    correction={n:torch.ones_like(p)*.03 for n,p in model.named_parameters()}
    loss,_=objective(other,t['x'],t['h'],t['y'],t['bank']);loss.backward()
    expected={n:p.detach()-.03*(p.grad+correction[n]) for n,p in other.named_parameters()}
    optimizer=torch.optim.SGD(model.parameters(),lr=.03)
    train_epoch(model,t,np.arange(12),optimizer,np.random.default_rng(1),control=correction)
    for n,p in model.named_parameters():torch.testing.assert_close(p,expected[n],atol=1e-7,rtol=1e-6)

def test_four_client_median_uses_middle_pair():
    updates=torch.tensor([[1.,4.],[2.,3.],[3.,2.],[1000.,-1000.]])
    torch.testing.assert_close(torch.quantile(updates,.5,dim=0),torch.tensor([2.5,2.5]))
