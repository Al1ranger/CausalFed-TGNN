import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import torch
from models import Model

def test_snapshot_batch_has_no_cross_prediction_edges():
    torch.manual_seed(19)
    x=torch.randn(8,6);h=torch.randn(8,4,4)
    for kind in ['sage','temporal_sage','hgt','full','homogeneous']:
        model=Model(kind).eval()
        batch=model(x,h)[0]
        singles=torch.cat([model(x[i:i+1],h[i:i+1])[0] for i in range(8)])
        torch.testing.assert_close(batch,singles,atol=1e-6,rtol=1e-5)

def test_collapsed_relations_are_permutation_invariant():
    torch.manual_seed(19);x=torch.randn(8,6);h=torch.randn(8,4,4)
    model=Model('homogeneous').eval()
    torch.testing.assert_close(model(x,h)[0],model(x,h[:,[2,0,3,1]])[0],atol=1e-6,rtol=1e-5)
    typed=Model('full').eval()
    assert not torch.allclose(typed(x,h)[0],typed(x,h[:,[2,0,3,1]])[0])
