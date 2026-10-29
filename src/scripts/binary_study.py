"""Binary multimodal study CLI. All new artifacts stay outside the saved baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from binary_study.common import initialize,verify_reference,package


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','smoke-gpu','benchmark','extract','extract-view','train','lock','evaluate','external','plots','package','verify'])
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--milk-images',type=Path);p.add_argument('--isic-ground-truth',type=Path);p.add_argument('--isic-metadata',type=Path);p.add_argument('--isic-images',type=Path)
    p.add_argument('--pad-metadata',type=Path);p.add_argument('--pad-images',type=Path)
    p.add_argument('--precision',choices=['auto','fp32'],default='auto');p.add_argument('--device',default='auto');p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--phash-radius',type=int,default=3);p.add_argument('--pca-dimension',type=int,default=128)
    p.add_argument('--conditions',nargs='+',default=['clean','blur','jpeg','dark']);p.add_argument('--isic-train-cap',type=int,default=0)
    p.add_argument('--view',choices=['clinical','dermoscopic']);p.add_argument('--scope',choices=['development','external'],default='development')
    p.add_argument('--variants',nargs='+',default=['linear','concat','concat_subtype','reliability','reliability_subtype'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,2026,3407]);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=7);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--width',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--target-sensitivity',type=float,default=.95)
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--destination',type=Path)
    args=p.parse_args()
    if args.workers<1 or args.epochs<1 or args.patience<1 or args.batch_size<1 or args.bootstrap<20:p.error('Positive settings required; bootstrap >=20')
    if not 0<args.target_sensitivity<=1:p.error('Sensitivity target must be in (0,1]')
    root=initialize(args.output,args.reference)
    if args.command=='prepare':
        if not all([args.milk_images,args.isic_ground_truth,args.isic_metadata,args.isic_images]):p.error('Supply MILK images and all three ISIC 2019 inputs')
        from binary_study.workflow import prepare
        prepare(root,args.reference,args.milk_images,args.isic_ground_truth,args.isic_metadata,args.isic_images,args.seed,args.phash_radius,args.workers,args.pca_dimension,args.conditions,args.isic_train_cap)
    elif args.command=='smoke-gpu':
        import torch
        from binary_study.features import encoder,tensor,forward
        from binary_study.model import BinaryFusion,training_loss
        from binary_study.common import choose_device
        import numpy as np
        device=choose_device(args.device)
        if device.type!='cuda':raise RuntimeError('The notebook GPU smoke requires CUDA')
        model=encoder(args.reference,'clinical',device)
        images=[np.full((224,224,3),128,np.uint8)]*2
        z,_=forward(model,images,device,False)
        head=BinaryFusion(z.shape[1],15,7,width=8).to(device)
        f=torch.as_tensor(np.stack([z,z],1),device=device);meta=torch.zeros(2,15,device=device);q=torch.zeros(2,2,7,device=device);m=torch.tensor([[1,1,0]]*2,dtype=torch.float32,device=device)
        result=head(f,meta,q,m);loss=training_loss(result,torch.tensor([0,1],device=device),torch.tensor([-1,1],device=device),m,torch.tensor(1.,device=device),torch.ones(4,device=device))
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('GPU backward smoke failed')
        verify_reference(args.reference)
        print('GPU encoder forward and binary-head backward smoke passed. Synthetic numerical check only; no checkpoints or metrics saved.',flush=True)
    elif args.command=='benchmark':
        from binary_study.features import benchmark
        benchmark(root,args.reference,args.device,precision=args.precision)
    elif args.command=='extract':
        from binary_study.workflow import extract
        extract(root,args.reference,args.scope,args.device,args.workers)
    elif args.command=='extract-view':
        from binary_study.features import extract_view
        from binary_study.common import verify_lock
        if args.scope=='external':verify_lock(root)
        if not args.view:p.error('--view required')
        protocol=json.loads((root/'protocol.json').read_text())
        extract_view(root,args.reference,args.view,args.scope,args.device,protocol['conditions'],args.workers)
    elif args.command=='train':
        from binary_study.training import train
        train(root,args.variants,tuple(args.seeds),args.device,epochs=args.epochs,patience=args.patience,
              batch_size=args.batch_size,width=args.width,learning_rate=args.learning_rate)
    elif args.command=='lock':
        from binary_study.evaluation import calibrate_and_lock
        calibrate_and_lock(root,args.target_sensitivity)
    elif args.command=='evaluate':
        from binary_study.evaluation import evaluate
        evaluate(root,args.scope,args.bootstrap)
    elif args.command=='external':
        if not args.pad_metadata or not args.pad_images:p.error('Supply PAD metadata and extracted image directory')
        from binary_study.workflow import prepare_external,extract
        from binary_study.evaluation import evaluate
        prepare_external(root,args.pad_metadata,args.pad_images,args.workers)
        extract(root,args.reference,'external',args.device,args.workers)
        evaluate(root,'external',args.bootstrap)
    elif args.command=='plots':
        from binary_study.common import verify_lock
        from binary_study.plots import draw
        verify_lock(root);draw(root)
    elif args.command=='package':
        package(root,args.destination or root.parent/'binary_study_artifacts.zip')
    elif args.command=='verify':
        verify_reference(args.reference)
        print('Frozen binary reference verified. No app or baseline files modified.',flush=True)

if __name__=='__main__':main()
