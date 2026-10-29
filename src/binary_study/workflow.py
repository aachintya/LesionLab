"""Pipeline boundaries: development first, sealed calibration, then external testing."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .common import sha,write_json,source_identity,verify_lock
from .data import milk_cases,isic_cases,pad_cases,assert_groups,LABEL_POLICY
from .overlap import fingerprints,quarantine_development,quarantine_external
from .features import CONDITIONS,extract_view,assemble


def prepare(root,reference,milk_images,isic_ground_truth,isic_metadata,isic_images,seed=42,radius=3,workers=2,
            pca_dimension=128,conditions=CONDITIONS,isic_train_cap=0):
    root,reference=Path(root),Path(reference)
    conditions=tuple(conditions)
    if not conditions or conditions[0]!='clean' or set(conditions)-set(CONDITIONS):raise ValueError('Conditions must start with clean and use known corruptions')
    protocol={'study':'binary-study-v1','label_policy':LABEL_POLICY,'primary_endpoint':'Source malignancy including SCC in situ; AK and KA excluded',
              'source_sha256':source_identity(),'reference_manifest_sha256':sha(reference/'manifest.json'),
              'source_csv_sha256':{'MILK':sha(reference/'split.csv'),'ISIC2019_ground_truth':sha(isic_ground_truth),'ISIC2019_metadata':sha(isic_metadata)},
              'roots':{'MILK':str(Path(milk_images).resolve()),'ISIC2019':str(Path(isic_images).resolve())},
              'isic_split_seed':seed,'isic_split_fractions':{'train':.70,'val':.15,'fusion':.15},'phash_radius':radius,
              'external_preprocessing_max_side':600,'MILK_preprocessing':'unchanged original pipeline',
              'pca_dimension':pca_dimension,'conditions':list(conditions),'isic_train_cap':isic_train_cap,
              'training_sources':['MILK','ISIC2019'],'external_test_source':'PAD','PAD_used_for_selection':False,
              'MILK_internal_results_are_exploratory':True,'encoder_training_history':'Original MILK train split; never relabel a random re-split as unseen',
              'corruption_scope':'Fixed blur/JPEG/dark perturbations after original deterministic preprocessing; not new independent patients',
              'metadata_fields':['age','sex','coarse_anatomical_site','observed_flags'],'prediction_excludes_reference_diagnoses_and_source':True}
    destination=root/'protocol.json'
    if destination.exists() and json.loads(destination.read_text())!=protocol:raise ValueError('Prepared protocol changed; use a new run')
    if (root/'prepared.json').exists():
        saved=json.loads((root/'prepared.json').read_text())
        for name,h in saved['sha256'].items():
            if sha(root/name)!=h:raise ValueError('Prepared cohort changed')
        print('Verified existing development manifests; PAD remains unopened.',flush=True);return
    if (root/'lock.json').exists():raise ValueError('Cannot prepare a new development cohort after lock')
    write_json(destination,protocol)
    milk=milk_cases(reference/'split.csv',milk_images);isic=isic_cases(isic_ground_truth,isic_metadata,isic_images,seed)
    frame=pd.concat([milk,isic],ignore_index=True);assert_groups(frame)
    records=fingerprints(frame,root/'development_fingerprints.json',workers)
    eligible,audit=quarantine_development(frame,records,radius)
    if isic_train_cap:
        if isic_train_cap<32:raise ValueError('ISIC training cap must be zero or >=32 cases')
        candidates=eligible[(eligible.source=='ISIC2019')&(eligible['split']=='train')]
        sizes=candidates.groupby('group_id').size();order=np.random.default_rng(seed).permutation(sizes.index);kept=[];n=0
        for key in order:
            if n+sizes[key]<=isic_train_cap:kept.append(key);n+=int(sizes[key])
        remove=set(candidates.group_id)-set(kept)
        eligible=eligible[~eligible.group_id.isin(remove)].reset_index(drop=True)
        audit['train_cap_removed_groups']=sorted(remove)
    assert_groups(eligible)
    for source in ('MILK','ISIC2019'):
        for split in ('train','val','fusion'):
            subset=eligible[(eligible.source==source)&(eligible['split']==split)]
            if set(subset.binary)!={0,1}:raise ValueError(f'{source}/{split} lacks both classes after audit')
    frame.to_csv(root/'all_development_cases.csv',index=False);eligible.to_csv(root/'development.csv',index=False)
    audit['counts']=[{'source':s,'split':sp,'target':int(y),'lesions':int(n)} for (s,sp,y),n in eligible.groupby(['source','split','binary']).size().items()]
    audit['patient_grouping_by_source']={s:bool(g.patient_grouping.all()) for s,g in eligible.groupby('source')}
    audit['reference_labels_are_not_predictor_inputs']=True
    write_json(root/'development_audit.json',audit)
    if (reference/'legacy_holdout.csv').exists():shutil.copyfile(reference/'legacy_holdout.csv',root/'legacy_holdout.csv')
    # Plain source copies avoid nested-ZIP auto-extraction changing reference bytes.
    src=Path(__file__).resolve().parents[1]
    for p in [*Path(__file__).parent.glob('*.py'),src/'scripts/binary_study.py',src/'components/preprocessing.py',
              src/'components/image_cache.py',src/'models/efficientnet_v2_image.py']:
        target=root/'source_files'/p.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write_json(root/'prepared.json',{'sha256':{n:sha(root/n) for n in ['development.csv','all_development_cases.csv',
                            'development_fingerprints.json','development_audit.json']}})
    print('Development data prepared. Eligible lesion counts:',flush=True)
    print(eligible.groupby(['source','split','binary']).size().to_string(),flush=True)
    print('PAD-UFES-20 has not been opened, fitted or selected on.',flush=True)


def extract(root,reference,scope='development',device='auto',workers=2):
    import torch
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    if scope=='external':verify_lock(root)
    output=root/f'{scope}_features.npz'
    if output.exists():return assemble(root,scope,protocol['pca_dimension'],protocol['conditions'])
    # Independent Python processes, not forked PyTorch DataLoader workers.
    parallel=device=='auto' and torch.cuda.device_count()>=2
    if parallel:
        running=[];logs=root/'logs';logs.mkdir(exist_ok=True)
        try:
            for i,view in enumerate(('clinical','dermoscopic')):
                path=logs/f'{scope}_{view}.log';handle=path.open('a')
                command=[sys.executable,'-u',str(Path(__file__).resolve().parents[1]/'scripts/binary_study.py'),
                         'extract-view','--output',str(root),'--reference',str(reference),'--scope',scope,
                         '--view',view,'--device',f'cuda:{i}','--workers',str(workers)]
                process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
                running.append((process,handle,path,view));print(f'Started {scope}/{view} on GPU {i}; log {path}',flush=True)
            while any(p.poll() is None for p,_,_,_ in running):
                for p,_,path,view in running:
                    if p.poll() not in (None,0):raise RuntimeError(f'Feature worker {view} failed. Inspect {path}')
                    tail=path.read_text(errors='replace').splitlines()[-1:]
                    print(f'{view}: {tail[0] if tail else "initializing"}',flush=True)
                time.sleep(10)
            for p,_,path,view in running:
                if p.returncode:raise RuntimeError(f'Feature worker {view} failed. Inspect {path}')
        finally:
            for p,h,_path,_view in running:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                h.close()
    else:
        for view in ('clinical','dermoscopic'):
            extract_view(root,reference,view,scope,device,protocol['conditions'],workers)
    return assemble(root,scope,protocol['pca_dimension'],protocol['conditions'])


def prepare_external(root,pad_metadata,pad_images,workers=2):
    root=Path(root);lock=verify_lock(root)
    identity={'metadata_sha256':sha(pad_metadata),'image_root':str(Path(pad_images).resolve()),'lock_sha256':sha(root/'lock.json')}
    receipt=root/'external_input.json'
    if receipt.exists() and json.loads(receipt.read_text())!=identity:raise ValueError('External source/lock changed after first access')
    if (root/'external_prepared.json').exists():
        saved=json.loads((root/'external_prepared.json').read_text())
        for name,h in saved['sha256'].items():
            if sha(root/name)!=h:raise ValueError('External cohort artifacts changed')
        return
    write_json(receipt,identity)
    frame=pad_cases(pad_metadata,pad_images)
    records=fingerprints(frame,root/'external_fingerprints.json',workers)
    protocol=json.loads((root/'protocol.json').read_text())
    eligible,audit=quarantine_external(frame,records,json.loads((root/'development_fingerprints.json').read_text()),protocol['phash_radius'])
    if set(eligible.binary)!={0,1}:raise ValueError('External test lacks both classes after declared exclusions')
    eligible.to_csv(root/'external.csv',index=False);write_json(root/'external_audit.json',audit)
    write_json(root/'external_prepared.json',{'sha256':{n:sha(root/n) for n in ['external.csv','external_audit.json','external_fingerprints.json','external_input.json']}})
    verify_lock(root)
    print(f'Locked external cohort: {len(eligible)} lesions, {eligible.group_id.nunique()} patients. No fitting on PAD.',flush=True)
