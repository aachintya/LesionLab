"""Development-only calibration lock; untouched-source evaluation and paired uncertainty."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,
                             average_precision_score,brier_score_loss,matthews_corrcoef)
from .common import sha,write_json,source_identity,verify_lock
from .training import PATTERNS,load_features,load_model,predict,pattern_rows
from .data import SUBTYPES


def choose_threshold(y,p,target=.95):
    y=np.asarray(y);p=np.asarray(p,float)
    if not 0<target<=1 or not np.isfinite(p).all():raise ValueError('Invalid sensitivity target or scores')
    positives=np.sort(p[y==1])
    if not len(positives):raise ValueError('Calibration needs positive cases')
    required=int(np.ceil(target*len(positives)))
    return float(positives[len(positives)-required])


def fit_calibration(y,p,target):
    if len(np.unique(y))!=2:raise ValueError('Calibration requires both classes')
    lr=LogisticRegression(C=10.,solver='lbfgs').fit(logit(np.clip(p,1e-6,1-1e-6))[:,None],y)
    slope=float(lr.coef_[0,0]);intercept=float(lr.intercept_[0])
    calibrated=expit(slope*logit(np.clip(p,1e-6,1-1e-6))+intercept)
    return {'method':'Platt logistic calibration','slope':slope,'intercept':intercept,
            'threshold':choose_threshold(y,calibrated,target),'target_calibration_sensitivity':target,
            'calibration_cases':len(y),'calibration_positive_cases':int(np.sum(y)),
            'calibration_realized_sensitivity':float(np.mean(calibrated[y==1]>=choose_threshold(y,calibrated,target)))}


def apply_calibration(p,cal):return expit(cal['slope']*logit(np.clip(p,1e-6,1-1e-6))+cal['intercept'])


def calibrate_and_lock(root,target=.95):
    root=Path(root)
    if (root/'lock.json').exists():
        lock=verify_lock(root)
        old=next(iter(next(iter(lock['models'].values()))['calibration'].values()))
        if old['target_calibration_sensitivity']!=target:raise ValueError('Cannot change a locked sensitivity target')
        return lock
    if (root/'external.csv').exists():raise ValueError('External cohort was opened before selection lock; use a fresh experiment')
    config=json.loads((root/'training_config.json').read_text());data=load_features(root/'development_features.npz')
    if 'PAD' in data['sources']:raise ValueError('External labels cannot enter calibration')
    if not {'concat','reliability_subtype'}<=set(config['variants']):raise ValueError('Train the predeclared primary candidate and comparator')
    indices=np.flatnonzero(data['splits']=='fusion');models={};sealed={}
    for variant in config['variants']:
        seeds=config['seeds'][:1] if variant=='linear' else config['seeds'];members=[];per_pattern={}
        for seed in seeds:
            folder=root/'models'/variant/f'seed-{seed}';model=load_model(folder);member={'seed':seed,'folder':str(folder.relative_to(root))}
            members.append(member)
            for p in folder.iterdir():
                if p.name in ('summary.json','best.pt','model.joblib'):sealed[str(p.relative_to(root))]=sha(p)
            for pattern in PATTERNS:
                rows,p=predict(model,data,indices,pattern)
                if len(rows) and len(np.unique(data['targets'][rows]))==2:
                    per_pattern.setdefault(pattern,[]).append((rows,p))
        calibrators={};seed_calibrators={}
        for pattern,values in per_pattern.items():
            for rows,_p in values[1:]:np.testing.assert_array_equal(rows,values[0][0])
            rows=values[0][0];y=data['targets'][rows]
            calibrators[pattern]=fit_calibration(y,np.mean([p for _,p in values],0),target)
            calibrators[pattern]['case_ids_sha256']=__import__('hashlib').sha256('\n'.join(data['case_ids'][rows]).encode()).hexdigest()
            seed_calibrators[pattern]={str(seed):fit_calibration(y,p,target) for seed,(_r,p) in zip(seeds,values)}
        models[variant]={'members':members,'calibration':calibrators,'seed_calibration':seed_calibrators}
    for name in ['identity.json','protocol.json','development.csv','development_feature_identity.json','metadata.json',
                 'transforms.joblib','training_config.json','benchmark.json','development_fingerprints.json','development_features.npz']:
        sealed[name]=sha(root/name)
    lock={'format':'binary-calibration-lock-v1','source_sha256':source_identity(),'sha256':sealed,
          'models':models,'selection_and_calibration_sources':['MILK','ISIC2019'],'calibration_split':'fusion',
          'primary_candidate':'reliability_subtype','primary_comparator':'concat','external_pattern':'clinical_metadata',
          'external_source':'PAD','primary_metric':'Specificity at independently locked sensitivity-targeting thresholds',
          'sensitivity_noninferiority_margin':.02,'subtype_noninferiority_margin':.02,
          'interpretation':'Targets are calibration targets, not guarantees on external sensitivity; all comparators locked before PAD access.'}
    write_json(root/'lock.json',lock)
    print('MODEL AND THRESHOLD LOCK WRITTEN. PAD has not been used for fitting or selection.',flush=True)
    return lock


def wilson(success,total):
    if not total:return None
    p=success/total;z=1.959963984540054;d=1+z*z/total
    c=(p+z*z/(2*total))/d;h=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [float(max(0,c-h)),float(min(1,c+h))]


def metrics(y,p,threshold=.5,subtypes=None):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=threshold
    if not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all():raise ValueError('Invalid evaluation rows')
    tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
    positive=tp+fn;negative=tn+fp
    result={'lesions':len(y),'positive_lesions':positive,'negative_lesions':negative,'threshold':float(threshold),
     'true_positive':tp,'true_negative':tn,'false_positive':fp,'false_negative':fn,
     'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
     'sensitivity':tp/positive if positive else None,'specificity':tn/negative if negative else None,
     'precision':tp/(tp+fp) if tp+fp else 0.,'npv':tn/(tn+fn) if tn+fn else None,
     'positive_f1_dice':float(f1_score(y,pred,pos_label=1,zero_division=0)),
     'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
     'mcc':float(matthews_corrcoef(y,pred)),
     'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
     'pr_average_precision':float(average_precision_score(y,p)) if positive else None,
     'brier_score':float(brier_score_loss(y,p)),
     'sensitivity_wilson95':wilson(tp,positive),'specificity_wilson95':wilson(tn,negative)}
    bins=np.minimum((p*10).astype(int),9);ece=0.
    for b in range(10):
        selected=bins==b
        if selected.any():ece+=float(selected.mean()*abs(y[selected].mean()-p[selected].mean()))
    result['ece_10_bins']=ece
    if subtypes is not None:
        subtypes=np.asarray(subtypes);result['malignant_subtypes']={}
        for k,name in enumerate(SUBTYPES):
            selected=(subtypes==k)&(y==1);n=int(selected.sum());found=int(pred[selected].sum())
            result['malignant_subtypes'][name]={'lesions':n,'detected':found,'missed':n-found,
                                               'sensitivity':found/n if n else None,'wilson95':wilson(found,n)}
    return result


def paired_uncertainty(y,left,right,groups,subtypes,left_threshold,right_threshold,draws=1000,seed=2026):
    rng=np.random.default_rng(seed);groups=np.asarray(groups);unique=np.unique(groups)
    members=[np.flatnonzero(groups==g) for g in unique];samples={k:[] for k in ['specificity','sensitivity','melanoma_sensitivity','balanced_accuracy','macro_f1']}
    def simple(idx,p,t):
        pred=p[idx]>=t;yy=y[idx];pos=yy==1;neg=~pos;mel=(subtypes[idx]==1)&pos
        return {'sensitivity':float(pred[pos].mean()) if pos.any() else None,
                'specificity':float((~pred[neg]).mean()) if neg.any() else None,
                'melanoma_sensitivity':float(pred[mel].mean()) if mel.any() else None,
                'balanced_accuracy':float((pred[pos].mean()+(~pred[neg]).mean())/2) if pos.any() and neg.any() else None,
                'macro_f1':float(f1_score(yy,pred,labels=[0,1],average='macro',zero_division=0))}
    all_idx=np.arange(len(y));a=simple(all_idx,left,left_threshold);b=simple(all_idx,right,right_threshold)
    for _ in range(draws):
        idx=np.concatenate([members[i] for i in rng.integers(0,len(members),len(members))])
        aa=simple(idx,left,left_threshold);bb=simple(idx,right,right_threshold)
        for key in samples:
            if aa[key] is not None and bb[key] is not None:samples[key].append(bb[key]-aa[key])
    return {'method':'Paired group bootstrap; candidate minus comparator, conditional on trained models',
            'groups':len(unique),'draws':draws,'seed':seed,'deltas':{k:{'point':b[k]-a[k] if a[k] is not None and b[k] is not None else None,
               'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'valid_draws':len(v)} for k,v in samples.items()}}


def evaluate(root,scope='development',draws=1000):
    root=Path(root);lock=verify_lock(root);data=load_features(root/f'{scope}_features.npz')
    identity={'lock_sha256':sha(root/'lock.json'),'features_sha256':sha(root/f'{scope}_features.npz'),'bootstrap_draws':draws}
    destination=root/f'{scope}_results.json'
    if destination.exists():
        result=json.loads(destination.read_text())
        if result['identity']!=identity:raise ValueError('Evaluation identity changed; do not retune an inspected test')
        print(f'{scope}: returning identical saved evaluation',flush=True);return result
    if scope=='external':
        if set(data['sources'])!={'PAD'} or set(data['splits'])!={'external'}:raise ValueError('Invalid external cohort')
        tracks=[('PAD','external','clinical_metadata'),('PAD','external','clinical_only')]
    else:
        if set(data['sources'])-{'MILK','ISIC2019'}:raise ValueError('External source in development evaluation')
        tracks=[('MILK','val','full'),('MILK','holdout','full'),('MILK','val','clinical_metadata'),
                ('MILK','holdout','clinical_metadata'),('MILK','holdout','images_only'),
                ('MILK','holdout','dermoscopic_metadata'),('MILK','holdout','clinical_only'),
                ('MILK','holdout','dermoscopic_only'),('ISIC2019','val','dermoscopic_metadata')]
    result={'identity':identity,'scope':scope,'models':{},'primary_comparisons':{},
            'internal_is_exploratory':scope=='development','thresholds_frozen_before_external_access':True,
            'no_test_calibration_or_selection':True,'label_policy':json.loads((root/'protocol.json').read_text())['label_policy']}
    predictions=[];paired={}
    for source,split,pattern in tracks:
        idx=np.flatnonzero((data['sources']==source)&(data['splits']==split));track=f'{source}/{split}/{pattern}'
        if not len(idx):continue
        result['models'][track]={}
        for variant,spec in lock['models'].items():
            members=[load_model(root/m['folder']) for m in spec['members']]
            # All missing-input patterns use only actually available features.
            for condition in data['conditions']:
                outputs=[predict(m,data,idx,pattern,str(condition)) for m in members]
                rows=outputs[0][0]
                for rr,_p in outputs[1:]:np.testing.assert_array_equal(rr,rows)
                if not len(rows):continue
                if pattern not in spec['calibration']:raise ValueError('No locked calibration for this input pattern')
                raw=np.mean([p for _,p in outputs],0);cal=spec['calibration'][pattern];p=apply_calibration(raw,cal)
                name=f'{variant}/{condition}';result['models'][track][name]=metrics(data['targets'][rows],p,cal['threshold'],data['subtypes'][rows])
                result['models'][track][name]['member_count']=len(members)
                for i,row in enumerate(rows):predictions.append({'track':track,'variant':variant,'condition':str(condition),'case_id':str(data['case_ids'][row]),
                   'group_id':str(data['groups'][row]),'target':int(data['targets'][row]),'subtype':int(data['subtypes'][row]),
                   'probability':float(p[i]),'threshold':cal['threshold'],'predicted':int(p[i]>=cal['threshold'])})
                if condition=='clean':
                    paired[(track,variant)]=(rows,p,cal['threshold'])
                    confirmations=data.get('confirmations',np.full(len(data['targets']),'unknown'))[rows]
                    hist=confirmations=='histopathology'
                    if hist.any():result['models'][track][name]['histopathology_only']=metrics(data['targets'][rows][hist],p[hist],cal['threshold'],data['subtypes'][rows][hist])
                    seed_scores={}
                    for member,(_rows,pp) in zip(spec['members'],outputs):
                        cc=spec['seed_calibration'][pattern][str(member['seed'])]
                        seed_scores[str(member['seed'])]=metrics(data['targets'][rows],apply_calibration(pp,cc),cc['threshold'],data['subtypes'][rows])
                    result['models'][track][name]['per_seed']=seed_scores
        # Predeclared primary contrast; no picking a favorable external winner.
        left=paired.get((track,lock['primary_comparator']));right=paired.get((track,lock['primary_candidate']))
        if left and right:
            rows,p,t=left;rr,q,u=right;np.testing.assert_array_equal(rows,rr)
            uncertainty=paired_uncertainty(data['targets'][rows],p,q,data['groups'][rows],data['subtypes'][rows],t,u,draws)
            uncertainty['grouping']='patient where known, otherwise lesion'
            uncertainty['all_groups_are_known_patients']=bool(data['patient_grouping'][rows].all())
            delta=uncertainty['deltas'];spec=delta['specificity']['ci95'];sens=delta['sensitivity']['ci95'];mel=delta['melanoma_sensitivity']['ci95']
            uncertainty['predeclared_signal_supported']=bool(spec and sens and mel and spec[0]>0 and sens[0]>=-.02 and mel[0]>=-.02)
            uncertainty['interpretation']='Failure of an interval guardrail is not proof of harm; it means the planned improvement is not established.'
            result['primary_comparisons'][track]=uncertainty
    if scope=='development' and (root/'legacy_holdout.csv').exists():
        legacy=pd.read_csv(root/'legacy_holdout.csv').set_index('lesion_id');idx=np.flatnonzero((data['sources']=='MILK')&(data['splits']=='holdout'))
        ids=[s.split(':',1)[1] for s in data['case_ids'][idx]]
        old=legacy.loc[ids];np.testing.assert_array_equal(old.target.to_numpy(),data['original_target'][idx])
        hard=old.predicted.isin([0,1,6,7,9]).to_numpy(float)
        baseline=metrics(data['targets'][idx],hard,.5,data['subtypes'][idx])
        for key in ('roc_auc','pr_average_precision','brier_score','ece_10_bins'):baseline[key]=None
        baseline['rule']='Archived original eleven-class argmax mapped to its positive family, evaluated only on new eligible cases'
        baseline['historical_only_not_a_recalibrated_binary_control']=True
        result['archived_original_same_cases']=baseline
    pd.DataFrame(predictions).to_csv(root/f'{scope}_predictions.csv',index=False)
    write_json(destination,result);write_summary(root,scope,result)
    print(f'{scope} evaluation saved; candidate and regressions retained without automatic promotion.',flush=True)
    return result


def write_summary(root,scope,result):
    lines=[f'# Binary study: {scope}', '', 'All thresholds were fitted on development calibration cases only. No test threshold tuning.', '',
           '| Cohort / input pattern | Method | Accuracy | Sensitivity | Specificity | Balanced accuracy | Macro F1 | PR AP |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for track,methods in result['models'].items():
        for name,m in methods.items():
            if not name.endswith('/clean'):continue
            values=[m[k] for k in ('accuracy','sensitivity','specificity','balanced_accuracy','macro_f1','pr_average_precision')]
            lines.append('| '+track+' | '+name+' | '+' | '.join(f'{v:.4f}' if v is not None else 'N/A' for v in values)+' |')
    lines+=['','See the JSON for confusion counts, per-cancer sensitivity, calibration, corruption conditions, seed variation and paired group-bootstrap intervals.',
            'Internal MILK validation/holdout were inspected in previous studies and remain exploratory. Binary labels exclude actinic keratosis and keratoacanthoma.',
            'No comparison with an unrelated published accuracy or eleven-class Dice is claimed. No model is automatically promoted into the demo.']
    (Path(root)/f'{scope}_comparison.md').write_text('\n'.join(lines)+'\n')
