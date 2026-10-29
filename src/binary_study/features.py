"""Frozen encoder extraction, chunk resume and train-only PCA/metadata fitting."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import io
import json
from pathlib import Path
import time
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from sklearn.decomposition import PCA
import torch
from .common import sha,digest,write_json,choose_device,verify_reference,source_identity
from .data import SharedMetadata

CONDITIONS=('clean','blur','jpeg','dark')
QUALITY_NAMES=('log_laplacian_variance','brightness','contrast','underexposure','overexposure','flip_js','head_entropy')


def encoder(reference, view, device):
    from models.efficientnet_v2_image import build_image_model
    model=build_image_model(11,pretrained=False)
    model.load_state_dict(torch.load(Path(reference)/f'{view}.pt',map_location='cpu',weights_only=True))
    return model.requires_grad_(False).eval().to(device)


def prepare_image(record, cache):
    from components.image_cache import load_preprocessed_image
    from components import preprocessing
    if sha(record['path'])!=record['sha256']:raise ValueError(f"Input image changed: {record['path']}")
    if record['source']=='MILK':
        return load_preprocessed_image(record['path'],modality=record['view'],roi_mode='foreground',cache_dir=cache)
    # Match the saved encoders' 600px source scale before costly inpainting.
    # All methods share this fixed external-data preparation; MILK is byte-compatible.
    key=digest([record['sha256'],record['view'],600,sha(preprocessing.__file__)])
    target=Path(cache)/'external'/f'{key}.npy'
    if target.exists():return np.load(target,allow_pickle=False)
    with Image.open(record['path']) as im:
        rgb=ImageOps.exif_transpose(im).convert('RGB');rgb.thumbnail((600,600),Image.Resampling.BICUBIC)
        image=np.array(rgb)
    image=preprocessing.preprocess_for_modality(image,record['view'])
    image,_=preprocessing.extract_lesion_roi_foreground(image)
    image=preprocessing.preprocess_final_steps(image)
    target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp')
    with temp.open('wb') as f:np.save(f,image,allow_pickle=False)
    temp.replace(target);return image


def perturb(image, condition):
    if condition=='clean':return image
    if condition=='blur':return cv2.GaussianBlur(image,(9,9),2.)
    if condition=='dark':return np.clip(image.astype(float)*.45,0,255).astype(np.uint8)
    if condition=='jpeg':
        out=io.BytesIO();Image.fromarray(image).save(out,format='JPEG',quality=30)
        with Image.open(io.BytesIO(out.getvalue())) as im:return np.array(im.convert('RGB'))
    raise ValueError(f'Unknown corruption condition {condition}')


def pixels_quality(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    return [np.log1p(cv2.Laplacian(gray,cv2.CV_64F).var()),gray.mean()/255.,gray.std()/255.,
            float((gray<12).mean()),float((gray>243).mean())]


def tensor(images, device):
    a=np.stack(images).astype(np.float32)/255.
    a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return torch.from_numpy(a.transpose(0,3,1,2)).to(device).contiguous(memory_format=torch.channels_last)


def forward(model, images, device, amp):
    context=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' and amp else nullcontext()
    with torch.inference_mode(),context:
        x=tensor(images,device);z=model.backbone(x);p=model.head(z).float().softmax(1)
        other=model(torch.flip(x,[3])).float().softmax(1)
        middle=((p+other)/2).clamp_min(1e-8)
        js=.5*((p*(p.clamp_min(1e-8).log()-middle.log())).sum(1)+(other*(other.clamp_min(1e-8).log()-middle.log())).sum(1))
        entropy=-(p*p.clamp_min(1e-8).log()).sum(1)
    z=z.float().cpu().numpy();scores=torch.stack([js,entropy],1).cpu().numpy()
    if not np.isfinite(z).all() or not np.isfinite(scores).all():raise FloatingPointError('Nonfinite frozen encoder outputs')
    return z,scores


def benchmark(root, reference, device_name='auto', batch_sizes=(8,16,32), steps=3, precision='auto'):
    root=Path(root);device=choose_device(device_name);torch.set_num_threads(2);cv2.setNumThreads(1)
    if (root/'benchmark.json').exists():
        existing=json.loads((root/'benchmark.json').read_text())
        if existing['reference_sha256']!=sha(Path(reference)/'manifest.json') or existing['source_sha256']!=source_identity() or existing.get('precision','auto')!=precision:
            raise ValueError('Benchmark identity changed; use a new run')
        print('Reusing the verified benchmark profile.',flush=True);return existing
    if device.type!='cuda' and device_name=='auto':raise RuntimeError('Enable GPU; CPU benchmarking requires explicit --device cpu')
    records=json.loads((root/'development_fingerprints.json').read_text())
    frame=pd.read_csv(root/'development.csv');train=set(frame.loc[frame['split']=='train','case_id'])
    sample=[]
    for source in ('MILK','ISIC2019'):
        sample.extend([r for r in records if r['case_id'] in train and r['source']==source][:2])
    if not sample:raise ValueError('No training images for benchmark')
    started=time.monotonic();images=[prepare_image(r,root/'cache/rgb') for r in sample]
    preproc=(time.monotonic()-started)/len(images)
    model=encoder(reference,'clinical',device);results=[]
    for amp in ([False,True] if device.type=='cuda' and precision=='auto' else [False]):
        for size in batch_sizes:
            try:
                batch=[images[i%len(images)] for i in range(size)]
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                forward(model,batch,device,amp)
                if device.type=='cuda':torch.cuda.synchronize(device)
                start=time.monotonic()
                for _ in range(steps):forward(model,batch,device,amp)
                if device.type=='cuda':torch.cuda.synchronize(device)
                speed=steps*size/(time.monotonic()-start)
                results.append({'batch_size':size,'amp':amp,'images_per_second_with_flip':speed,
                                'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None})
                print(f'GPU benchmark batch={size} AMP={amp}: {speed:.1f} images/s (includes flip check)',flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();print(f'Benchmark batch={size} AMP={amp}: out of memory',flush=True)
            except FloatingPointError:
                if not amp:raise
                print(f'Benchmark AMP={amp}: nonfinite outputs; retaining FP32 candidates.',flush=True)
    if not results:raise RuntimeError('No GPU profile passed')
    chosen=max(results,key=lambda r:r['images_per_second_with_flip'])
    result={'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'torch':str(torch.__version__),'precision':precision,'profiles':results,'selected':chosen,'preprocessing_seconds_per_image':preproc,
            'reference_sha256':sha(Path(reference)/'manifest.json'),'source_sha256':source_identity(),
            'warning':'Throughput estimate only; original preprocessing, disk and overlap checks also contribute.'}
    write_json(root/'benchmark.json',result);return result


def extract_view(root, reference, view, scope='development', device_name='auto', conditions=CONDITIONS,
                 workers=2, model_factory=None):
    root=Path(root);device=choose_device(device_name);torch.set_num_threads(2);cv2.setNumThreads(1)
    benchmark_data=json.loads((root/'benchmark.json').read_text())
    verify_reference(reference)
    if benchmark_data['reference_sha256']!=sha(Path(reference)/'manifest.json') or benchmark_data['source_sha256']!=source_identity():
        raise ValueError('Benchmark source/reference changed')
    profile=benchmark_data['selected'];batch_size=profile['batch_size'];amp=bool(profile['amp'] and device.type=='cuda')
    frame=pd.read_csv(root/f'{scope}.csv');wanted=set(frame.case_id)
    all_records=json.loads((root/f'{scope}_fingerprints.json').read_text())
    records=sorted([r for r in all_records if r['case_id'] in wanted and r['view']==view],key=lambda r:r['key'])
    if not records:return
    identity=digest({'reference':sha(Path(reference)/f'{view}.pt'),'source':source_identity(),'conditions':list(conditions),
                     'records':[(r['key'],r['sha256']) for r in records],'amp':amp})
    directory=root/'cache/features'/scope/view;directory.mkdir(parents=True,exist_ok=True)
    model=None;start=time.monotonic();chunk_size=256
    for chunk,start_index in enumerate(range(0,len(records),chunk_size)):
        part=directory/f'{chunk:05d}.npz';rows=records[start_index:start_index+chunk_size]
        if part.exists():
            with np.load(part,allow_pickle=False) as z:
                if str(z['identity'])!=identity or z['keys'].tolist()!=[r['key'] for r in rows]:raise ValueError('Feature chunk identity changed')
            print(f'{scope}/{view}: reuse {min(start_index+len(rows),len(records))}/{len(records)}',flush=True);continue
        if model is None:model=(model_factory(reference,view,device) if model_factory else encoder(reference,view,device))
        with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
            base=[]
            for done,image in enumerate(pool.map(lambda r:prepare_image(r,root/'cache/rgb'),rows),1):
                base.append(image)
                if done%32==0 or done==len(rows):
                    print(f'{scope}/{view}: prepared {start_index+done}/{len(records)} images for extraction',flush=True)
        zs=[];qs=[]
        for condition in conditions:
            print(f'{scope}/{view}: extracting {condition}, images {start_index+1}-{start_index+len(rows)}',flush=True)
            feats=[];quality=[]
            for offset in range(0,len(rows),batch_size):
                images=[perturb(a,condition) for a in base[offset:offset+batch_size]]
                f,stability=forward(model,images,device,amp)
                feats.append(f);quality.append(np.c_[np.array([pixels_quality(a) for a in images]),stability])
            zs.append(np.concatenate(feats));qs.append(np.concatenate(quality))
        temp=part.with_suffix('.tmp')
        with temp.open('wb') as f:np.savez_compressed(f,identity=np.array(identity),keys=np.array([r['key'] for r in rows]),
                                                   features=np.stack(zs),quality=np.stack(qs).astype(np.float32))
        temp.replace(part)
        print(f'{scope}/{view}: cached {min(start_index+len(rows),len(records))}/{len(records)} images; {time.monotonic()-start:.1f}s',flush=True)
    write_json(directory/'complete.json',{'identity':identity,'images':len(records),'conditions':list(conditions),
               'sha256':{p.name:sha(p) for p in sorted(directory.glob('*.npz'))}})


def assemble(root, scope='development', pca_dimension=128, conditions=CONDITIONS):
    root=Path(root);frame=pd.read_csv(root/f'{scope}.csv');count=len(frame)
    if scope=='development' and set(frame.source)-{'MILK','ISIC2019'}:raise ValueError('External source in development cache')
    if scope!='development' and not (root/'lock.json').exists():raise ValueError('External assembly requires calibration lock')
    frame_hash=sha(root/f'{scope}.csv');destination=root/f'{scope}_features.npz'
    identity={'cases_sha256':frame_hash,'source':source_identity(),'conditions':list(conditions)}
    identity_path=root/f'{scope}_feature_identity.json'
    if destination.exists():
        saved=json.loads(identity_path.read_text())
        if saved['identity']!=identity or saved['sha256']!=sha(destination):raise ValueError('Assembled feature cache changed')
        return destination
    all_records=json.loads((root/f'{scope}_fingerprints.json').read_text());lookup={r['key']:r for r in all_records}
    case_index={key:i for i,key in enumerate(frame.case_id)};raw=None
    quality=np.zeros((len(conditions),count,2,len(QUALITY_NAMES)),np.float32);n=np.zeros((count,2),np.float32)
    for vi,view in enumerate(('clinical','dermoscopic')):
        directory=root/'cache/features'/scope/view
        if not directory.exists():continue
        complete=json.loads((directory/'complete.json').read_text())
        if complete['conditions']!=list(conditions):raise ValueError('Extraction conditions differ')
        for name,expected in complete['sha256'].items():
            path=directory/name
            if sha(path)!=expected:raise ValueError('Feature chunk bytes changed')
            with np.load(path,allow_pickle=False) as z:
                if raw is None:raw=np.zeros((len(conditions),count,2,z['features'].shape[-1]),np.float32)
                for j,key in enumerate(z['keys']):
                    r=lookup[str(key)];i=case_index[r['case_id']]
                    raw[:,i,vi]+=z['features'][:,j];quality[:,i,vi]+=z['quality'][:,j];n[i,vi]+=1
    if raw is None:raise ValueError('No image features extracted')
    raw/=np.maximum(n[None,:,:,None],1);quality/=np.maximum(n[None,:,:,None],1)
    available=n>0
    if scope=='development':
        train=frame['split'].eq('train').to_numpy()
        metadata=SharedMetadata().fit(frame[train]);metadata.save(root/'metadata.json')
        dim=min(pca_dimension,raw.shape[-1],*(int((train&available[:,i]).sum())-1 for i in range(2)))
        if dim<2:raise ValueError('Need at least three training cases per image modality')
        reducers=[];means=[];stds=[];qmeans=[];qstds=[]
        for i in range(2):
            x=raw[0,train&available[:,i],i]
            pca=PCA(n_components=dim,svd_solver='randomized',random_state=42).fit(x)
            transformed=pca.transform(x);reducers.append(pca);means.append(transformed.mean(0));stds.append(np.maximum(transformed.std(0),1e-5))
            q=quality[0,train&available[:,i],i];qmeans.append(q.mean(0));qstds.append(np.maximum(q.std(0),1e-5))
        transforms={'pca':reducers,'mean':means,'std':stds,'quality_mean':qmeans,'quality_std':qstds,'dimension':dim,
                    'fit_case_ids':frame.loc[train,'case_id'].tolist(),'fit_split':'train','source_sha256':source_identity()}
        joblib.dump(transforms,root/'transforms.joblib')
    else:metadata=SharedMetadata.load(root/'metadata.json');transforms=joblib.load(root/'transforms.joblib');dim=transforms['dimension']
    reduced=np.zeros((*raw.shape[:-1],dim),np.float32)
    for i in range(2):
        present=available[:,i]
        for c in range(len(conditions)):
            if present.any():
                reduced[c,present,i]=(transforms['pca'][i].transform(raw[c,present,i])-transforms['mean'][i])/transforms['std'][i]
                quality[c,present,i]=(quality[c,present,i]-transforms['quality_mean'][i])/transforms['quality_std'][i]
    meta,observed=metadata.transform(frame)
    available=np.c_[available,observed.sum(1)>0].astype(np.float32)
    if not np.isfinite(reduced).all() or not np.isfinite(quality).all():raise FloatingPointError('Nonfinite reduced features')
    with destination.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,features=reduced,quality=quality,metadata=meta,observed=observed,available=available,
         case_ids=frame.case_id.to_numpy(str),groups=frame.group_id.to_numpy(str),sources=frame.source.to_numpy(str),
         splits=frame['split'].to_numpy(str),targets=frame.binary.to_numpy(int),subtypes=frame.subtype.to_numpy(int),
         patient_grouping=frame.patient_grouping.to_numpy(bool),
         confirmations=frame.get('diagnosis_confirm_type',pd.Series('unknown',index=frame.index)).fillna('unknown').to_numpy(str),conditions=np.array(conditions),original_target=frame.original_target.to_numpy(int))
    destination.with_suffix('.tmp').replace(destination)
    write_json(identity_path,{'identity':identity,'sha256':sha(destination)})
    print(f'{scope} feature cache ready: {count} lesions, dimension {dim}; train-only transforms',flush=True)
    return destination
