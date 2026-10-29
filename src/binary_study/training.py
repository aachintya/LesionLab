"""Matched binary controls, finite training, deterministic epoch resume and no external selection."""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
from .common import sha,digest,write_json,seed_all,choose_device,verify_lock
from .data import mask_metadata
from .model import BinaryFusion,VARIANTS,training_loss

PATTERNS={'available':(1,1,1),'full':(1,1,1),'clinical_metadata':(1,0,1),'dermoscopic_metadata':(0,1,1),
          'clinical_only':(1,0,0),'dermoscopic_only':(0,1,0),'images_only':(1,1,0),'metadata_only':(0,0,1)}


def load_features(path):
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    n=len(data['targets'])
    if not np.isin(data['targets'],[0,1]).all() or len(set(data['case_ids']))!=n:raise ValueError('Invalid binary feature cohort')
    for key in ['features','metadata','quality','available']:
        if not np.isfinite(data[key]).all():raise ValueError(f'Nonfinite cache: {key}')
    return data


def pattern_rows(data, indices, pattern):
    if pattern not in PATTERNS:raise ValueError('Unknown input pattern')
    indices=np.asarray(indices,dtype=int);m=data['available'][indices]
    if pattern in ('full','images_only'):valid=(m[:,:2]>0).all(1)
    elif pattern.startswith('clinical'):valid=m[:,0]>0
    elif pattern.startswith('dermoscopic'):valid=m[:,1]>0
    elif pattern=='metadata_only':valid=m[:,2]>0
    else:valid=m.sum(1)>0
    return indices[valid]


def inputs(data, indices, pattern='available', condition='clean'):
    rows=pattern_rows(data,indices,pattern);conditions=data['conditions'].tolist()
    if condition not in conditions:raise ValueError(f'Feature condition unavailable: {condition}')
    c=conditions.index(condition);m=data['available'][rows].copy()*np.array(PATTERNS[pattern],np.float32)
    f=data['features'][c,rows].copy()*m[:,:2,None];q=data['quality'][c,rows].copy()*m[:,:2,None]
    meta=data['metadata'][rows].copy()*m[:,2,None]
    return rows,(f,meta,q,m)


def augmented(data, indices, rng):
    n=len(indices);v=rng.integers(0,len(data['conditions']),size=n)
    # Half clean; other views are predeclared encoder-space acquisition simulations.
    v[rng.random(n)<.5]=0
    f=data['features'][v,indices].copy();q=data['quality'][v,indices].copy();m=data['available'][indices].copy()
    keep=(rng.random(m.shape)>.25).astype(np.float32);m*=keep
    flags=data['observed'][indices].copy()*(rng.random((n,3))>.15)
    meta=mask_metadata(data['metadata'][indices],flags)
    m[:,2]*=(flags.sum(1)>0)
    for i in np.flatnonzero(m[:,:2].sum(1)==0):
        choices=np.flatnonzero(data['available'][indices[i],:2]);m[i,rng.choice(choices)]=1
    return f*m[:,:2,None],meta*m[:,2,None],q*m[:,:2,None],m


def flatten(parts):
    f,meta,q,m=parts
    return np.c_[f.reshape(len(f),-1),meta,q.reshape(len(q),-1),m]


def torch_parts(parts,device):return [torch.as_tensor(a,dtype=torch.float32,device=device) for a in parts]


def predict(model, data, indices, pattern='available', condition='clean', device='cpu', batch_size=1024):
    rows,parts=inputs(data,indices,pattern,condition)
    if not len(rows):return rows,np.empty(0)
    if isinstance(model,dict):return rows,expit(model['estimator'].decision_function(flatten(parts)))
    model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            p=[a[start:start+batch_size] for a in parts]
            values.append(model(*torch_parts(p,device))['logit'].sigmoid().cpu().numpy())
    return rows,np.concatenate(values).astype(float)


def validation_score(model,data,device):
    val=np.flatnonzero(data['splits']=='val');scores=[]
    for source,pattern in [('MILK','full'),('MILK','clinical_metadata'),('MILK','dermoscopic_metadata'),('ISIC2019','dermoscopic_metadata')]:
        idx=val[data['sources'][val]==source];rows,p=predict(model,data,idx,pattern,device=device)
        if len(rows) and len(np.unique(data['targets'][rows]))==2:scores.append(roc_auc_score(data['targets'][rows],p))
    if not scores:raise ValueError('Validation requires positive and negative reference cases')
    return float(np.mean(scores))


def atomic_torch(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def train_one(root, variant, seed, config, data, device):
    folder=Path(root)/'models'/variant/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
    identity=digest({'config':config,'variant':variant,'seed':seed,'features':sha(Path(root)/'development_features.npz')})
    summary_path=folder/'summary.json'
    if summary_path.exists():
        saved=json.loads(summary_path.read_text())
        if saved['identity']!=identity:raise ValueError('Training configuration differs from saved run')
        if saved['complete']:
            if sha(folder/'best.pt' if variant!='linear' else folder/'model.joblib')!=saved['model_sha256']:raise ValueError('Completed model changed')
            print(f'{variant}/seed-{seed}: already complete',flush=True);return
    if (Path(root)/'lock.json').exists():raise ValueError('Calibration locked: no further training permitted')
    train=np.flatnonzero(data['splits']=='train');y=data['targets'][train]
    if len(np.unique(y))!=2:raise ValueError('Binary training requires both classes')
    seed_all(seed)
    if variant=='linear':
        chunks=[flatten(inputs(data,train)[1])];targets=[y]
        for j in range(2):chunks.append(flatten(augmented(data,train,np.random.default_rng(seed+j))));targets.append(y)
        estimator=LogisticRegression(C=config['linear_c'],class_weight='balanced',max_iter=1000,random_state=seed)
        estimator.fit(np.concatenate(chunks),np.concatenate(targets))
        model={'estimator':estimator,'identity':identity};joblib.dump(model,folder/'model.joblib')
        score=validation_score(model,data,'cpu')
        write_json(summary_path,{'identity':identity,'complete':True,'epochs_completed':1,'best_validation_mean_auc':score,
                                'model_sha256':sha(folder/'model.joblib'),'variant':variant,'seed':seed})
        print(f'{variant}/seed-{seed}: fitted; validation mean AUC {score:.4f}',flush=True);return
    dimension=data['features'].shape[-1]
    init={'dimension':dimension,'metadata_width':data['metadata'].shape[-1],'quality_width':data['quality'].shape[-1],
          'variant':variant,'width':config['width']}
    model=BinaryFusion(**init).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3,factor=.5)
    resume=folder/'resume.pt';start=0;best=-1.;bad=0
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['identity']!=identity:raise ValueError('Resume identity differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch']+1;best=state['best'];bad=state['bad']
    pos_weight=torch.tensor(float((y==0).sum()/max((y==1).sum(),1)),device=device)
    counts=np.bincount(data['subtypes'][train][data['subtypes'][train]>=0],minlength=4)
    sw=1/np.sqrt(np.maximum(counts,1));sw=sw/sw.mean();sw=np.minimum(sw,3.)
    subtype_weights=torch.tensor(sw,dtype=torch.float32,device=device)
    completed=start
    for epoch in range(start,config['epochs']):
        if bad>=config['patience']:break
        seed_all(seed+epoch*100003);rng=np.random.default_rng(seed+epoch*100003)
        order=rng.permutation(train);total=0.;started=time.monotonic();model.train()
        for step,offset in enumerate(range(0,len(order),config['batch_size'])):
            idx=order[offset:offset+config['batch_size']]
            parts=augmented(data,idx,rng);target=torch.tensor(data['targets'][idx],device=device);subtype=torch.tensor(data['subtypes'][idx],device=device)
            tensors=torch_parts(parts,device);result=model(*tensors);loss=training_loss(result,target,subtype,tensors[3],pos_weight,subtype_weights,config['auxiliary_weight'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();total+=float(loss.detach())*len(idx)
            if step and step%50==0:print(f'{variant}/{seed} epoch {epoch+1} batch {step}: loss {float(loss):.4f}',flush=True)
        score=validation_score(model,data,device);scheduler.step(score);completed=epoch+1
        if score>best+1e-6:
            best=score;bad=0
            atomic_torch(folder/'best.pt',{'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'init':init,'identity':identity})
        else:bad+=1
        atomic_torch(resume,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                            'epoch':epoch,'best':best,'bad':bad,'identity':identity})
        record={'identity':identity,'complete':False,'epochs_completed':completed,'best_validation_mean_auc':best,'variant':variant,'seed':seed}
        write_json(summary_path,record)
        with (folder/'history.jsonl').open('a') as log:log.write(json.dumps({'epoch':completed,'loss':total/len(train),'validation_mean_auc':score,'seconds':time.monotonic()-started})+'\n')
        print(f'{variant}/seed-{seed} EPOCH {completed}: loss={total/len(train):.4f}, validation mean AUC={score:.4f}, {time.monotonic()-started:.1f}s',flush=True)
        if bad>=config['patience']:break
    record={'identity':identity,'complete':True,'epochs_completed':completed,'best_validation_mean_auc':best,
            'variant':variant,'seed':seed,'model_sha256':sha(folder/'best.pt'),'parameters':sum(p.numel() for p in model.parameters())}
    write_json(summary_path,record)


def train(root,variants=VARIANTS,seeds=(42,2026,3407),device_name='auto',**kwargs):
    root=Path(root);torch.set_num_threads(2);device=choose_device(device_name)
    data=load_features(root/'development_features.npz')
    identity=json.loads((root/'development_feature_identity.json').read_text())
    if identity['sha256']!=sha(root/'development_features.npz'):raise ValueError('Development cache changed')
    if set(data['sources'])-{'MILK','ISIC2019'} or 'external' in data['splits']:raise ValueError('External test data cannot train/select heads')
    if not {'MILK','ISIC2019'}<=set(data['sources'][data['splits']=='train']):raise ValueError('Both MILK and ISIC 2019 must supply training cases')
    config={'variants':list(variants),'seeds':list(seeds),'epochs':40,'patience':7,'batch_size':256,'width':64,
            'learning_rate':1e-3,'linear_c':1.,'auxiliary_weight':.2,'selection':'mean validation ROC-AUC across fixed source/view tracks'}
    config.update(kwargs)
    path=root/'training_config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Training config changed; use a new run')
    if not path.exists():write_json(path,config)
    if (root/'lock.json').exists():verify_lock(root)
    for variant in variants:
        if variant not in VARIANTS:raise ValueError('Unknown variant')
        # Linear baseline is deterministic; never count duplicated fits as seeds.
        for seed in (seeds[:1] if variant=='linear' else seeds):train_one(root,variant,seed,config,data,device)
    return config


def load_model(folder,device='cpu'):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
    if not summary['complete']:raise ValueError('Model training is incomplete')
    if summary['variant']=='linear':
        if sha(folder/'model.joblib')!=summary['model_sha256']:raise ValueError('Saved model changed')
        return joblib.load(folder/'model.joblib')
    if sha(folder/'best.pt')!=summary['model_sha256']:raise ValueError('Saved model changed')
    payload=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model=BinaryFusion(**payload['init']);model.load_state_dict(payload['model']);return model.eval().to(device)
