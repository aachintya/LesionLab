"""Exploratory feature/head/TTA study with preserved encoders and baseline.

Select on fusion OOF only; examine two locked profiles on previously seen
validation data. No holdout evaluation or automatic app/checkpoint promotion.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import torch

from components.augmentations import get_transforms
from components.dataset import MILK10kDataset
from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS
from models.efficientnet_v2_image import build_image_model
from models.metadata_mlp import build_metadata_model
from scripts.study_frozen_fusion import sha,write_json,protected_hashes,check_protected,BASELINE,CHECKPOINTS
from utils import feature_study as methods
from utils.fusion_study import historical_recipe,out_of_fold,metrics,class_rows,confusion,confusion_metrics,predict_candidate
from utils.training import pick_device

OUTPUT=ROOT/'experiments/frozen-feature-study-v2'
OLD=ROOT/'experiments/frozen-fusion-study-v1'
TTA_NAMES=['identity','horizontal_flip','vertical_flip','rotate90','rotate180','rotate270']


def views(inputs):
    return [inputs,torch.flip(inputs,[3]),torch.flip(inputs,[2]),
            torch.rot90(inputs,1,[2,3]),torch.rot90(inputs,2,[2,3]),torch.rot90(inputs,3,[2,3])]


def extract(frame):
    path=OUTPUT/'frozen_embeddings.npz';manifest=OUTPUT/'feature_identity.json'
    identity={'split_csv':sha(BASELINE/'payload/final_metadata_with_split.csv'),
              'source_probability_cache':sha(OLD/'frozen_probabilities.npz'),
              'source_identity':sha(OLD/'feature_identity.json'),
              'encoders':{name:sha(CHECKPOINTS/name) for name in ['best_clinical.pt','best_dermoscopic.pt','best_metadata.pt','metadata_encoder.json']},
              'preprocessing':{name:sha(ROOT/name) for name in ['src/components/preprocessing.py','src/components/image_cache.py','src/components/augmentations.py']},
              'tta':TTA_NAMES,'torch':str(torch.__version__)}
    if path.exists():
        stored=json.loads(manifest.read_text())
        if stored['identity']!=identity or stored['sha256']!=sha(path):raise ValueError('Feature cache identity differs.')
        print('Verified embedding cache loaded.',flush=True);return
    clinical=frame[frame.image_type=='clinical: close-up'].sort_values('lesion_id').reset_index(drop=True)
    dermoscopic=frame[frame.image_type=='dermoscopic'].sort_values('lesion_id').reset_index(drop=True)
    assert clinical.lesion_id.tolist()==dermoscopic.lesion_id.tolist()
    assert clinical.target.tolist()==dermoscopic.target.tolist()
    torch.set_num_threads(2);cv2.setNumThreads(1);device=pick_device()
    available=clinical.split.isin(['fusion','val']).to_numpy();embeddings=[];probs=[];tta=[]
    for name,rows in [('clinical',clinical),('dermoscopic',dermoscopic)]:
        model=build_image_model(11,dropout=.5,pretrained=False)
        model.load_state_dict(torch.load(CHECKPOINTS/f'best_{name}.pt',map_location='cpu',weights_only=True))
        model.requires_grad_(False).eval().to(device)
        data=MILK10kDataset(rows,str(ROOT/'data/raw/MILK10k_Training_Input'),
                           transform=get_transforms(modality=name)[1],roi_mode='foreground',modality=name)
        zs=[];ps=[];ts=[]
        print(f'{name}: extracting {len(rows)} frozen embeddings; six-view averaging only for fusion/validation.',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool,torch.inference_mode():
            for start in range(0,len(rows),16):
                end=min(start+16,len(rows));batch=list(pool.map(data.__getitem__,range(start,end)))
                inputs=torch.stack([row[0] for row in batch]).to(device)
                z=model.backbone(inputs);p=torch.softmax(model.head(z),dim=1)
                average=p.clone();indices=np.flatnonzero(available[start:end])
                if len(indices):
                    selected=inputs[torch.as_tensor(indices,device=device)]
                    total=p[torch.as_tensor(indices,device=device)].clone()
                    for view in views(selected)[1:]:total+=torch.softmax(model(view),dim=1)
                    average[torch.as_tensor(indices,device=device)]=total/6
                zs.append(z.cpu().numpy());ps.append(p.cpu().numpy());ts.append(average.cpu().numpy())
                if start==0 or (start//16+1)%16==0 or end==len(rows):print(f'{name}: {end}/{len(rows)}',flush=True)
        assert not model.training and all(not value.requires_grad and value.grad is None for value in model.parameters())
        embeddings.append(np.concatenate(zs));probs.append(np.concatenate(ps));tta.append(np.concatenate(ts))
        del model
        if device.type=='mps':torch.mps.empty_cache()
    encoder=MetadataEncoder.load(CHECKPOINTS/'metadata_encoder.json');raw=encoder.transform(clinical)
    metadata=build_metadata_model(encoder.input_dim_,11)
    metadata.load_state_dict(torch.load(CHECKPOINTS/'best_metadata.pt',map_location='cpu',weights_only=True))
    metadata.requires_grad_(False).eval()
    with torch.inference_mode():mp=torch.softmax(metadata(torch.from_numpy(raw)),dim=1).numpy()
    p=np.stack([*probs,mp],axis=1).astype(np.float64);ta=np.stack([*tta,mp],axis=1).astype(np.float64)
    old=np.load(OLD/'frozen_probabilities.npz',allow_pickle=False)
    lookup={lid:i for i,lid in enumerate(clinical.lesion_id)}
    rows=np.asarray([lookup[str(lid)] for lid in old['lesion_ids']])
    difference=float(np.max(np.abs(p[rows]-old['probabilities'])))
    np.testing.assert_allclose(p[rows],old['probabilities'],atol=2e-5,rtol=1e-4)
    # Preserve the exact v1 comparator probability cache for identity views.
    p[rows]=old['probabilities']
    temporary=OUTPUT/'extracting.tmp.npz'
    np.savez_compressed(temporary,embeddings=np.stack(embeddings,axis=1),probabilities=p,tta_probabilities=ta,
                        metadata=raw,labels=clinical.target.to_numpy(dtype=int),
                        lesion_ids=clinical.lesion_id.to_numpy(dtype=str),splits=clinical.split.to_numpy(dtype=str),tta_available=available)
    temporary.replace(path)
    write_json(manifest,{'identity':identity,'sha256':sha(path),'embedding_shape':[len(clinical),2,1280],
                         'identity_probability_max_difference_from_v1':difference,'device':str(device)})


def representations(cache):
    destination=OUTPUT/'representations.joblib'
    train=np.flatnonzero(cache['splits']=='train');z=cache['embeddings'];reduced=[];reducers={}
    for index,name in enumerate(['clinical','dermoscopic']):
        reducer=PCA(n_components=128,whiten=True,svd_solver='randomized',iterated_power=4,random_state=methods.SEED)
        reducer.fit(z[train,index]);reduced.append(reducer.transform(z[:,index]));reducers[name]=reducer
    joblib.dump({'reducers':reducers,'fit_split':'train','fit_lesions':len(train),'metadata_features':12},destination)
    p=cache['probabilities'].reshape(len(z),33)
    result={f'pca{k}':np.c_[reduced[0][:,:k],reduced[1][:,:k],cache['metadata']] for k in [64,128]}
    result['pca64_plus_scores']=np.c_[result['pca64'],p]
    result['scores']=p;result['tta_scores']=cache['tta_probabilities'].reshape(len(z),33)
    return result


def intervals(y,baseline,candidate,draws=5000):
    a=baseline.argmax(1);b=candidate.argmax(1);keys=['accuracy','balanced_accuracy','macro_f1']
    point_a=confusion_metrics(confusion(y,a));point_b=confusion_metrics(confusion(y,b))
    samples=np.empty((draws,3));rng=np.random.default_rng(methods.SEED)
    for i in range(draws):
        rows=rng.integers(0,len(y),len(y));left=confusion_metrics(confusion(y[rows],a[rows]));right=confusion_metrics(confusion(y[rows],b[rows]))
        samples[i]=[right[key]-left[key] for key in keys]
    # Bonferroni allocation for at most two locked candidate profiles.
    return {key:{'difference':point_b[key]-point_a[key],
                 'lower_97_5':float(np.quantile(samples[:,i],.0125)),
                 'upper_97_5':float(np.quantile(samples[:,i],.9875))} for i,key in enumerate(keys)}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    before_path=OUTPUT/'protected_files_before.json'
    protected=json.loads(before_path.read_text()) if before_path.exists() else protected_hashes()
    check_protected(protected)
    if not before_path.exists():write_json(before_path,protected)
    frame=pd.read_csv(BASELINE/'payload/final_metadata_with_split.csv')
    selected=frame[frame.split.isin(['train','fusion','val'])].copy()
    for name,expected in [('train',3406),('fusion',524),('val',786)]:assert selected[selected.split==name].lesion_id.nunique()==expected
    assert selected.groupby('lesion_id').split.nunique().eq(1).all()
    assert not set(selected.lesion_id)&set(frame[frame.split=='holdout'].lesion_id)
    specs=methods.candidate_specs()
    protocol={'study':'frozen-feature-study-v2','seed':methods.SEED,'candidates':specs,'candidate_count':len(specs),
              'encoders_frozen':True,'holdout_used':False,'validation_previously_inspected':True,
              'selection':'Three-fold fusion OOF, with original training lesions as auxiliary fitting rows only for embedding classifiers.',
              'downstream_training_label_use':'Embedding classifiers reuse 3406 training labels plus fusion-fold fitting labels; score-only methods fit fusion-fold labels only. These are not isolated fusion-mechanism comparisons.',
              'pca_fit_split':'train only; fixed before OOF; never fusion OOF evaluation or validation.',
              'profiles':['best_macro','class_stability'],
              'class_stability_constraint':{'NV_max_oof_recall_drop':.03,'SCCKA_max_oof_recall_drop':.03},
              'large_gain_target':{'validation_macro_f1_delta':.10,'NV_max_validation_recall_drop':.03,
                                   'SCCKA_max_validation_recall_drop':.03,'accuracy_drop_allowed':False,
                                   'macro_f1_adjusted_interval_lower_above_zero':True},
              'uncertainty':'Conditional paired lesion bootstrap, 5000 draws, 97.5% intervals for up to two profiles; exploratory after prior validation inspection.',
              'tta_views':TTA_NAMES,'tta_available_on':['fusion','val'],'class_order':list(DIAGNOSIS_COLS),
              'input_reference_sha256':sha(OLD/'frozen_probabilities.npz'),
              'limitations':['Validation already selected encoder checkpoints and informed the previous research round.',
                             'Fusion includes a single MAL_OTH lesion; patient grouping is unverified.',
                             'No test can guarantee future improvement; independent evaluation is required before promotion.']}
    protocol_path=OUTPUT/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise ValueError('Protocol changed; use a new study directory.')
    else:write_json(protocol_path,protocol)
    print(f'Recorded {len(specs)} broader alternatives; all encoder weights frozen; holdout excluded.',flush=True)
    extract(selected)
    cache=np.load(OUTPUT/'frozen_embeddings.npz',allow_pickle=False);x=representations(cache)
    train=np.flatnonzero(cache['splits']=='train');fusion=np.flatnonzero(cache['splits']=='fusion');validation=np.flatnonzero(cache['splits']=='val')
    y=cache['labels'];assert not len(np.intersect1d(train,fusion)) and not len(np.intersect1d(fusion,validation))
    with warnings.catch_warnings(record=True):folds=list(StratifiedKFold(3,shuffle=True,random_state=methods.SEED).split(np.zeros(len(fusion)),y[fusion]))
    base_oof=out_of_fold(historical_recipe(),cache['probabilities'][fusion],y[fusion],folds)
    base_class=class_rows(y[fusion],base_oof.argmax(1),DIAGNOSIS_COLS);baseline_oof=metrics(y[fusion],base_oof)
    rankings=[];failed=[];predictions={}
    for index,spec in enumerate(specs):
        try:
            pred=methods.oof(spec,x,y,train,fusion,folds);scores=metrics(y[fusion],pred)
            detail=class_rows(y[fusion],pred.argmax(1),DIAGNOSIS_COLS)
            stable=all(detail[k]['recall']>=base_class[k]['recall']-.03 for k in [8,9])
            rankings.append({'id':spec['id'],**scores,'nv_recall':detail[8]['recall'],'sccka_recall':detail[9]['recall'],'stable':stable})
            predictions[spec['id']]=pred
            print(f"OOF {index+1}/{len(specs)} {spec['id']}: F1={scores['macro_f1']:.4f}; NV={detail[8]['recall']:.3f}; SCCKA={detail[9]['recall']:.3f}; stable={stable}",flush=True)
        except RuntimeError as exc:
            failed.append({'id':spec['id'],'reason':str(exc)});print(f"Excluded {spec['id']}: {exc}",flush=True)
    rankings.sort(key=lambda row:(-row['macro_f1'],-row['balanced_accuracy'],row['id']))
    if not rankings:raise RuntimeError('No completed alternatives.')
    pd.DataFrame(rankings).to_csv(OUTPUT/'fusion_oof_leaderboard.csv',index=False)
    stable=[row for row in rankings if row['stable']]
    chosen={'best_macro':rankings[0]['id'],'class_stability':stable[0]['id'] if stable else None}
    lock={'profiles':chosen,'protocol_sha256':sha(protocol_path),'selection_data':'fusion OOF only','baseline_oof':baseline_oof,
          'baseline_oof_class_recall':{DIAGNOSIS_COLS[k]:base_class[k]['recall'] for k in [8,9]},'failed':failed}
    lock_path=OUTPUT/'selection_locked.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text())!=lock:raise ValueError('Locked selection changed; validation reuse is forbidden.')
    else:write_json(lock_path,lock)
    print('LOCKED '+json.dumps(chosen)+'; beginning exploratory validation comparison.',flush=True)
    baseline=joblib.load(CHECKPOINTS/'best_fusion_lr.joblib').predict_proba(x['scores'][validation])
    prior=predict_candidate(joblib.load(OLD/'candidate_fusion.joblib'),cache['probabilities'][validation])
    references={'saved_baseline':metrics(y[validation],baseline),'previous_fusion_candidate':metrics(y[validation],prior)}
    reference_classes=class_rows(y[validation],baseline.argmax(1),DIAGNOSIS_COLS)
    results={};seen={}
    all_scores={'baseline':baseline,'prior':prior}
    for profile,name in chosen.items():
        if name is None:continue
        if name in seen:
            results[profile]={'same_as':seen[name],'id':name};continue
        spec=next(item for item in specs if item['id']==name)
        rows=methods.fitting_rows(spec,train,fusion,np.arange(len(fusion)))
        model=methods.fit(spec,x[spec['feature']][rows],y[rows]);pred=methods.predict(spec,model,x[spec['feature']][validation])
        score=metrics(y[validation],pred);detail=class_rows(y[validation],pred.argmax(1),DIAGNOSIS_COLS)
        interval=intervals(y[validation],baseline,pred)
        large=(score['macro_f1']-references['saved_baseline']['macro_f1']>=.10
               and score['accuracy']>=references['saved_baseline']['accuracy']
               and all(detail[k]['recall']>=reference_classes[k]['recall']-.03 for k in [8,9])
               and interval['macro_f1']['lower_97_5']>0)
        payload={'format_version':1,'spec':spec,'estimator':model,'representations_file':'representations.joblib',
                 'protocol_sha256':sha(protocol_path),'encoder_files':json.loads((OUTPUT/'feature_identity.json').read_text())['identity']['encoders'],
                 'class_order':list(DIAGNOSIS_COLS),'calibrated':False}
        artifact=OUTPUT/f'candidate_{profile}.joblib';joblib.dump(payload,artifact)
        results[profile]={'id':name,'spec':spec,'validation_metrics':score,'class_details':detail,
                          'paired_differences':interval,'large_gain_target_passed':bool(large),'fitted_lesions':len(rows),'artifact_sha256':sha(artifact)}
        seen[name]=profile;all_scores[profile]=pred
        print(json.dumps({'profile':profile,'method':name,'metrics':score,'large_gain_target':bool(large),'delta_macro_f1':interval['macro_f1']},indent=2),flush=True)
    np.savez_compressed(OUTPUT/'validation_scores.npz',targets=y[validation],lesion_ids=cache['lesion_ids'][validation],**all_scores)
    result={'profiles':results,'references':references,'baseline_class_details':reference_classes,'baseline_oof':baseline_oof,
            'validation_lesions':len(validation),'fusion_lesions':len(fusion),'train_lesions':len(train),
            'holdout_evaluated':False,'validation_is_exploratory':True,'promoted_to_demo':False,
            'protected_files_unchanged':len(protected),'failed':failed,'protocol_sha256':sha(protocol_path),'limitations':protocol['limitations']}
    check_protected(protected);write_json(OUTPUT/'results.json',result)
    print('Completed; saved models and baseline unchanged.',flush=True)


if __name__=='__main__':main()
