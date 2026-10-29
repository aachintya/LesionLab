"""Train small downstream heads on cached features; image encoders are absent."""
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

SEED=1729


def specs():
    result=[]
    for feature,widths,balances in [('pca64',[128,256],['none','sqrt','soft_sampling']),
                                   ('pca128',[128],['sqrt','soft_sampling']),
                                   ('pca64_plus_scores',[128],['sqrt','soft_sampling'])]:
        for width in widths:
            for balance in balances:
                result.append(dict(id=f'{feature}_mlp{width}_{balance}',feature=feature,family='mlp',width=width,balance=balance,epochs=40))
    for balance in ['none','sqrt','soft_sampling']:
        result.append(dict(id=f'pca64_gated64_{balance}',feature='pca64',family='gated',width=64,balance=balance,epochs=40))
    return result


class GatedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(size,64),nn.LayerNorm(64),nn.GELU()) for size in [64,64,12]])
        self.gate=nn.Sequential(nn.Linear(140,32),nn.GELU(),nn.Linear(32,3))
        self.classifier=nn.Sequential(nn.Linear(204,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,11))

    def forward(self,x):
        parts=[x[:,:64],x[:,64:128],x[:,128:140]]
        projected=torch.stack([layer(value) for layer,value in zip(self.projections,parts)],dim=1)
        weights=torch.softmax(self.gate(x),dim=1)
        joint=(projected*weights[:,:,None]).flatten(1)
        return self.classifier(torch.cat([joint,x[:,128:140]],dim=1))


def build(spec,dimension):
    if spec['family']=='gated':
        if dimension!=140:raise ValueError('Gated head expects two 64-D images and 12 metadata features.')
        return GatedHead()
    width=spec['width']
    return nn.Sequential(nn.Linear(dimension,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(.3),
                         nn.Linear(width,width//2),nn.LayerNorm(width//2),nn.GELU(),nn.Dropout(.3),nn.Linear(width//2,11))


def fit(spec,features,labels):
    if not np.isfinite(features).all():raise ValueError('Nonfinite neural-head features.')
    torch.set_num_threads(2);torch.manual_seed(SEED)
    scaler=StandardScaler().fit(features)
    x=torch.from_numpy(scaler.transform(features).astype(np.float32));y=torch.as_tensor(labels,dtype=torch.long)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite neural-head features.')
    model=build(spec,x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=spec['epochs'])
    counts=torch.bincount(y,minlength=11).float().clamp_min(1)
    power=0 if spec['balance']=='none' else .5
    weights=counts.pow(-power);weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.05)
    for _ in range(spec['epochs']):
        model.train()
        if spec['balance']=='soft_sampling':order=torch.multinomial(counts[y].pow(-.5),len(y),replacement=True)
        else:order=torch.randperm(len(y))
        for rows in order.split(128):
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(x[rows]),y[rows])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite head-training loss.')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        scheduler.step()
    model.eval()
    return model,scaler


def predict(model,scaler,features):
    x=torch.from_numpy(scaler.transform(features).astype(np.float32))
    model.eval()
    with torch.inference_mode():result=torch.softmax(model(x),dim=1).numpy().astype(float)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1),1,atol=1e-5):raise ValueError('Invalid neural-head scores.')
    return result
