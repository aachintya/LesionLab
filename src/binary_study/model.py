"""Small binary heads on frozen image representations; no pretrained weight mutation."""
from __future__ import annotations
import torch
from torch import nn

VARIANTS=('linear','concat','concat_subtype','reliability','reliability_subtype')


class BinaryFusion(nn.Module):
    def __init__(self, dimension, metadata_width, quality_width, variant='reliability_subtype',width=64):
        super().__init__()
        if variant not in VARIANTS or variant=='linear':raise ValueError('Use a neural variant')
        self.variant=variant
        self.projections=nn.ModuleList([nn.Sequential(nn.Linear(dimension,width),nn.LayerNorm(width),nn.GELU()) for _ in range(2)])
        self.metadata=nn.Sequential(nn.Linear(metadata_width,width),nn.LayerNorm(width),nn.GELU())
        self.experts=nn.ModuleList([nn.Linear(width,1) for _ in range(3)])
        self.gated=variant.startswith('reliability');self.auxiliary=variant.endswith('subtype')
        self.gate=nn.Sequential(nn.Linear(2*quality_width+3+3,32),nn.GELU(),nn.Linear(32,3)) if self.gated else None
        self.fusion=nn.Sequential(nn.Linear(3*width+3+2*quality_width,width),nn.GELU(),nn.Dropout(.15))
        self.binary=nn.Linear(width,1)
        self.subtype=nn.Linear(width,4) if self.auxiliary else None

    def forward(self, features, metadata, quality, available):
        if features.ndim!=3 or features.shape[1]!=2 or available.shape!=(len(features),3):
            raise ValueError('Expected paired feature slots and three availability flags')
        if (available.sum(1)==0).any():raise ValueError('At least one input modality is required')
        # Mask before and after projection: unavailable content, NaNs and biases cannot enter.
        features=torch.where(available[:,:2,None].bool(),features,torch.zeros_like(features))
        metadata=torch.where(available[:,2,None].bool(),metadata,torch.zeros_like(metadata))
        quality=torch.where(available[:,:2,None].bool(),quality,torch.zeros_like(quality))
        h=torch.stack([self.projections[i](features[:,i]) for i in range(2)]+[self.metadata(metadata)],1)
        h=h*available[:,:,None]
        expert=torch.cat([head(h[:,i]) for i,head in enumerate(self.experts)],1)
        if self.gated:
            p=expert.detach().sigmoid().clamp(1e-6,1-1e-6)
            entropy=-(p*p.log()+(1-p)*(1-p).log())*available
            logits=self.gate(torch.cat([quality.flatten(1),available,entropy],1)).masked_fill(~available.bool(),-torch.inf)
            weights=logits.softmax(1)
        else:weights=available/available.sum(1,keepdim=True)
        fused=self.fusion(torch.cat([(h*weights[:,:,None]*3).flatten(1),available,quality.flatten(1)],1))
        return {'logit':self.binary(fused).squeeze(1),'expert_logits':expert,'weights':weights,
                'subtype_logits':self.subtype(fused) if self.subtype is not None else None}


def training_loss(result, target, subtype, available, pos_weight, subtype_weights, auxiliary_weight=.2):
    binary=nn.functional.binary_cross_entropy_with_logits(result['logit'],target.float(),pos_weight=pos_weight)
    experts=nn.functional.binary_cross_entropy_with_logits(result['expert_logits'],target[:,None].float().expand(-1,3),
                                                           pos_weight=pos_weight,reduction='none')
    loss=binary+.1*(experts*available).sum()/available.sum().clamp_min(1)
    if result['subtype_logits'] is not None:
        valid=(target==1)&(subtype>=0)
        if valid.any():loss=loss+auxiliary_weight*nn.functional.cross_entropy(result['subtype_logits'][valid],subtype[valid],weight=subtype_weights)
    return loss
