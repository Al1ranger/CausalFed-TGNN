"""Finite-history snapshot implementation; conceptual modules remain explicit."""
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv, SAGEConv

TYPES=['account','merchant','device','location']
EDGES=[(t,'context_for','transaction') for t in TYPES]+[('transaction','uses',t) for t in TYPES]
MODELS=['mlp','history_mlp','sage','temporal_sage','hetero_static','hgt','full','no_inv','no_env','no_time','no_con','homogeneous']

class Model(nn.Module):
    def __init__(self,kind='full',width=24):
        super().__init__(); self.kind=kind; self.width=width
        self.temporal=kind not in ['mlp','sage','hetero_static','no_time']
        self.decomposed=kind in ['full','no_inv','no_env','no_time','no_con','homogeneous']
        self.hetero=kind not in ['mlp','sage','temporal_sage','homogeneous']
        self.tx=nn.Linear(22 if kind=='history_mlp' else 6,width)
        if kind in ['mlp','history_mlp']: self.layer=nn.Linear(width,width)
        elif kind=='homogeneous':
            self.entity=nn.Linear(4,width)
            self.layer=HGTConv(width,width,(['entity','transaction'],[('entity','context_for','transaction'),('transaction','uses','entity')]),heads=2)
        elif self.hetero:
            self.entity=nn.ModuleDict({t:nn.Linear(4,width) for t in TYPES})
            self.layer=HGTConv(width,width,(TYPES+['transaction'],EDGES),heads=2)
        else:
            self.entity=nn.Linear(4,width); self.layer=SAGEConv(width,width)
        if self.decomposed:
            self.inv=nn.Linear(width,width)
            if kind!='no_env':
                self.env=nn.Linear(width,width); self.bankhead=nn.Linear(width,4)
        self.head=nn.Linear(width,2)

    def forward(self,x,h):
        h=h.clone()
        if not self.temporal: h[:,:,2]=0
        tx=F.relu(self.tx(torch.cat([x,h.flatten(1)],1) if self.kind=='history_mlp' else x)); n=len(x)
        if self.kind in ['mlp','history_mlp']: z=F.relu(self.layer(tx))
        elif self.kind=='homogeneous':
            nodes={'entity':F.relu(self.entity(h)).reshape(-1,self.width),'transaction':tx}
            src=torch.arange(4*n,device=x.device);dst=torch.arange(n,device=x.device).repeat_interleave(4)
            z=F.relu(self.layer(nodes,{('entity','context_for','transaction'):torch.stack([src,dst]),
                      ('transaction','uses','entity'):torch.stack([dst,src])})['transaction']+tx)
        elif self.hetero:
            nodes={t:F.relu(self.entity[t](h[:,i])) for i,t in enumerate(TYPES)}
            nodes['transaction']=tx
            ids=torch.arange(n,device=x.device); edge=torch.stack([ids,ids])
            z=F.relu(self.layer(nodes,{e:edge for e in EDGES})['transaction']+tx)
        else:
            entities=F.relu(self.entity(h)).reshape(-1,self.width)
            nodes=torch.cat([tx,entities])
            src=torch.arange(4*n,device=x.device)+n
            dst=torch.arange(n,device=x.device).repeat_interleave(4)
            edge=torch.stack([torch.cat([src,dst]),torch.cat([dst,src])])
            z=F.relu(self.layer(nodes,edge)[:n]+tx)
        env=None
        if self.decomposed:
            if self.kind!='no_env': env=self.bankhead(F.relu(self.env(z)))
            z=F.relu(self.inv(z))
        return self.head(z),z,env

def objective(model,x,h,y,bank,inv_weight=.1,con_weight=.01,env_weight=.1):
    logits,z,env=model(x,h)
    supervised=F.cross_entropy(logits,y)
    loss=supervised; stats={'supervised':float(supervised.detach()),'active_environments':int(bank.unique().numel())}
    active=model.decomposed and bank.unique().numel()>=2
    # IRMv1 scale-gradient squared, each environment weighted equally.
    if active and model.kind!='no_inv' and inv_weight:
        scale=torch.ones((),requires_grad=True,device=x.device)
        penalties=[]
        for b in bank.unique():
            risk=F.cross_entropy(logits[bank==b]*scale,y[bank==b])
            penalties.append(torch.autograd.grad(risk,scale,create_graph=True)[0].square())
        irm=torch.stack(penalties).mean(); loss=loss+inv_weight*irm
        stats['irm']=float(irm.detach())
    else: stats['irm']=None
    # Supervised contrastive alignment: anchors use positives of same class in other banks.
    if active and model.kind!='no_con' and con_weight:
        ids=torch.arange(min(len(y),128),device=y.device)
        zz=F.normalize(z[ids],dim=1); sim=zz@zz.T/.2
        off=~torch.eye(len(ids),dtype=torch.bool,device=y.device)
        pos=(y[ids,None]==y[None,ids])&(bank[ids,None]!=bank[None,ids])&off
        valid=pos.any(1)
        if valid.any():
            logp=sim-torch.logsumexp(sim.masked_fill(~off,-1e9),1,keepdim=True)
            con=(-(logp*pos).sum(1)/pos.sum(1).clamp(min=1))[valid].mean()
            loss=loss+con_weight*con; stats['contrastive']=float(con.detach())
    if env is not None:
        el=F.cross_entropy(env,bank); loss=loss+env_weight*el; stats['environment']=float(el.detach())
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    return loss,stats
