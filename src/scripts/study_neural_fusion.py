"""Exploratory nonlinear-head follow-up; original encoders and holdout untouched."""
import json
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from scripts.study_frozen_fusion import sha,write_json,protected_hashes,check_protected,CHECKPOINTS
from scripts.study_frozen_features import intervals
from utils import frozen_neural_head as heads
from utils.fusion_study import metrics,class_rows,historical_recipe,out_of_fold
from components.paths import DIAGNOSIS_COLS

SOURCE=ROOT/'experiments/frozen-feature-study-v2';OUTPUT=ROOT/'experiments/frozen-neural-study-v3'


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    before_path=OUTPUT/'protected_files_before.json'
    if before_path.exists():
        before=json.loads(before_path.read_text());check_protected(before)
    else:
        before=protected_hashes()
        for name in ['frozen_embeddings.npz','representations.joblib','protocol.json','results.json']:
            before[str((SOURCE/name).relative_to(ROOT))]=sha(SOURCE/name)
        write_json(before_path,before)
    methods=heads.specs()
    protocol={'study':'frozen-neural-study-v3','candidate_count':len(methods),'candidates':methods,'seed':heads.SEED,
              'original_encoders_frozen':True,'holdout_used':False,'validation_previously_inspected':True,
              'feature_cache_sha256':sha(SOURCE/'frozen_embeddings.npz'),'reducers_sha256':sha(SOURCE/'representations.joblib'),
              'fitting':'3406 original training rows plus only the fitting fusion-fold rows. No early stopping or epoch selection on OOF/validation.',
              'selection':'Three-fold fusion OOF; macro F1 then balanced accuracy then ID.',
              'profiles':['best_macro','class_stability'],'stability_oof_max_recall_drop':.03,
              'validation_large_gain_target':{'macro_f1_delta':.10,'maximum_NV_and_SCCKA_recall_drop':.03,'accuracy_drop_allowed':False},
              'intervals':'Conditional 97.5% paired bootstrap intervals, 5000 draws; exploratory, with prior validation reuse.'}
    path=OUTPUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('Neural study protocol changed.')
    else:write_json(path,protocol)
    print(f'Recorded {len(methods)} nonlinear heads on cached features; no convolutional training.',flush=True)
    d=np.load(SOURCE/'frozen_embeddings.npz',allow_pickle=False);reducers=joblib.load(SOURCE/'representations.joblib')['reducers']
    z=d['embeddings'];parts=[reducers[name].transform(z[:,i]) for i,name in enumerate(['clinical','dermoscopic'])]
    x={f'pca{k}':np.c_[parts[0][:,:k],parts[1][:,:k],d['metadata']] for k in [64,128]}
    x['pca64_plus_scores']=np.c_[x['pca64'],d['probabilities'].reshape(len(z),33)]
    y=d['labels'];train=np.flatnonzero(d['splits']=='train');fusion=np.flatnonzero(d['splits']=='fusion');val=np.flatnonzero(d['splits']=='val')
    with warnings.catch_warnings(record=True):folds=list(StratifiedKFold(3,shuffle=True,random_state=heads.SEED).split(np.zeros(len(fusion)),y[fusion]))
    baseline_oof=out_of_fold(historical_recipe(),d['probabilities'][fusion],y[fusion],folds)
    base_classes=class_rows(y[fusion],baseline_oof.argmax(1),DIAGNOSIS_COLS)
    rankings=[];failed=[]
    for index,spec in enumerate(methods):
        destination=OUTPUT/f"oof_{spec['id']}.npz"
        if destination.exists():p=np.load(destination,allow_pickle=False)['probabilities']
        else:
            p=np.zeros((len(fusion),11))
            for fitting,evaluation in folds:
                rows=np.r_[train,fusion[fitting]];test=fusion[evaluation]
                assert not len(np.intersect1d(rows,test))
                model,scaler=heads.fit(spec,x[spec['feature']][rows],y[rows])
                p[evaluation]=heads.predict(model,scaler,x[spec['feature']][test])
            np.savez_compressed(destination,probabilities=p)
        score=metrics(y[fusion],p);detail=class_rows(y[fusion],p.argmax(1),DIAGNOSIS_COLS)
        stable=all(detail[k]['recall']>=base_classes[k]['recall']-.03 for k in [8,9])
        rankings.append({'id':spec['id'],**score,'nv_recall':detail[8]['recall'],'sccka_recall':detail[9]['recall'],'stable':stable})
        print(f"OOF {index+1}/{len(methods)} {spec['id']}: F1={score['macro_f1']:.4f}; NV={detail[8]['recall']:.3f}; SCCKA={detail[9]['recall']:.3f}; stable={stable}",flush=True)
    rankings.sort(key=lambda row:(-row['macro_f1'],-row['balanced_accuracy'],row['id']))
    pd.DataFrame(rankings).to_csv(OUTPUT/'fusion_oof_leaderboard.csv',index=False)
    stable=[row for row in rankings if row['stable']]
    profiles={'best_macro':rankings[0]['id'],'class_stability':stable[0]['id'] if stable else None}
    lock={'profiles':profiles,'protocol_sha256':sha(path),'selection_data':'fusion OOF only'}
    locked=OUTPUT/'selection_locked.json'
    if locked.exists():
        if json.loads(locked.read_text())!=lock:raise ValueError('Locked neural choices changed.')
    else:write_json(locked,lock)
    print('LOCKED '+json.dumps(profiles)+'; validation examination follows.',flush=True)
    baseline=joblib.load(CHECKPOINTS/'best_fusion_lr.joblib').predict_proba(d['probabilities'][val].reshape(len(val),33))
    baseline_classes=class_rows(y[val],baseline.argmax(1),DIAGNOSIS_COLS)
    results={};seen={};scores={'baseline':baseline}
    for profile,name in profiles.items():
        if name is None:continue
        if name in seen:results[profile]={'id':name,'same_as':seen[name]};continue
        spec=next(item for item in methods if item['id']==name);rows=np.r_[train,fusion]
        model,scaler=heads.fit(spec,x[spec['feature']][rows],y[rows]);p=heads.predict(model,scaler,x[spec['feature']][val])
        values=metrics(y[val],p);detail=class_rows(y[val],p.argmax(1),DIAGNOSIS_COLS);confidence=intervals(y[val],baseline,p)
        large=(confidence['macro_f1']['difference']>=.10 and confidence['macro_f1']['lower_97_5']>0
               and values['accuracy']>=metrics(y[val],baseline)['accuracy']
               and all(detail[k]['recall']>=baseline_classes[k]['recall']-.03 for k in [8,9]))
        torch.save({'spec':spec,'dimension':x[spec['feature']].shape[1],'state_dict':model.state_dict()},OUTPUT/f'candidate_{profile}.pt')
        joblib.dump(scaler,OUTPUT/f'candidate_{profile}_scaler.joblib')
        results[profile]={'id':name,'spec':spec,'validation_metrics':values,'class_details':detail,'paired_differences':confidence,
                          'large_gain_target_passed':bool(large),'fitted_lesions':len(rows)}
        scores[profile]=p;seen[name]=profile
        print(json.dumps({'profile':profile,'id':name,'metrics':values,'large_gain_target':bool(large)},indent=2),flush=True)
    check_protected(before)
    write_json(OUTPUT/'results.json',{'profiles':results,'baseline_validation':metrics(y[val],baseline),'baseline_class_details':baseline_classes,
                                    'validation_lesions':len(val),'train_lesions':len(train),'fusion_lesions':len(fusion),
                                    'holdout_evaluated':False,'validation_is_exploratory':True,'promoted_to_demo':False,
                                    'protected_files_unchanged':len(before)})
    np.savez_compressed(OUTPUT/'validation_scores.npz',targets=y[val],lesion_ids=d['lesion_ids'][val],**scores)
    print('Neural study completed; all original models and earlier artifacts unchanged.',flush=True)


if __name__=='__main__':main()
