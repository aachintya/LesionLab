"""Binary study invariants and a synthetic end-to-end run. No legacy fork tests."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from binary_study.common import sha,write_json,initialize,verify_lock
from binary_study.data import (milk_label,source_label,SharedMetadata,mask_metadata,isic_cases,pad_cases,ImageIndex)
from binary_study.model import BinaryFusion,training_loss
from binary_study.training import train,load_features,load_model,predict
from binary_study.evaluation import calibrate_and_lock,choose_threshold,evaluate


class BinaryStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def reference(self):
        ref=self.root/'reference';ref.mkdir(exist_ok=True);(ref/'clinical.pt').write_bytes(b'fixture checkpoint')
        (ref/'dermoscopic.pt').write_bytes(b'fixture dermoscopic checkpoint')
        write_json(ref/'manifest.json',{'format':'binary-reference-v1','sha256':{name:sha(ref/name) for name in ['clinical.pt','dermoscopic.pt']}})
        return ref
    def fixture(self,name='run'):
        ref=self.reference();out=initialize(self.root/name,ref);rng=np.random.default_rng(7)
        rows=[]
        for source in ['MILK','ISIC2019']:
            for split,n in [('train',24),('val',8),('fusion',12),('holdout',8)]:
                if source=='ISIC2019' and split=='holdout':continue
                for i in range(n):
                    y=i%2;idx=len(rows)
                    rows.append({'case_id':f'{source}:{idx}','source':source,'group_id':f'{source}:group{idx}',
                       'split':split,'binary':y,'subtype':i%3 if y else -1,'age':40+20*y,'sex':'male' if i%3 else 'female',
                       'site':'back','clinical':'[]','dermoscopic':'[]','patient_grouping':source=='ISIC2019','original_target':1 if y else 8})
        frame=pd.DataFrame(rows);n=len(frame)
        f=rng.normal(size=(2,n,2,6)).astype(np.float32);f[:,:,:,0]+=np.array(frame.binary)[None,:,None]*2
        avail=np.ones((n,3),np.float32);avail[frame.source.eq('ISIC2019'),0]=0;f*=avail[None,:,:2,None]
        meta,obs=SharedMetadata().fit(frame[frame['split']=='train']).transform(frame)
        q=rng.normal(size=(2,n,2,7)).astype(np.float32)*avail[None,:,:2,None]
        frame.to_csv(out/'development.csv',index=False)
        with (out/'development_features.npz').open('wb') as handle:np.savez_compressed(handle,features=f,quality=q,metadata=meta,observed=obs,
           available=avail,case_ids=frame.case_id.to_numpy(str),sources=frame.source.to_numpy(str),groups=frame.group_id.to_numpy(str),
           splits=frame['split'].to_numpy(str),targets=frame.binary.to_numpy(int),subtypes=frame.subtype.to_numpy(int),conditions=np.array(['clean','blur']),
           patient_grouping=frame.patient_grouping.to_numpy(bool),original_target=frame.original_target.to_numpy(int))
        write_json(out/'development_feature_identity.json',{'sha256':sha(out/'development_features.npz')})
        for name in ['protocol.json','metadata.json','benchmark.json']:
            write_json(out/name,{'label_policy':'fixture'})
        write_json(out/'development_fingerprints.json',[]);joblib.dump({'fixture':True},out/'transforms.joblib')
        return out,ref
    def do_train(self,out):return train(out,seeds=(42,),device_name='cpu',epochs=2,patience=3,batch_size=16,width=8)

    def test_granular_binary_labels_not_broad_grouping(self):
        self.assertEqual(milk_label(0,'Solar or actinic keratosis'),(-1,-1))
        self.assertEqual(milk_label(0,'Squamous cell carcinoma in situ, Bowens disease'),(1,2))
        self.assertEqual(milk_label(9,'Squamous cell carcinoma, Invasive'),(1,2))
        self.assertEqual(milk_label(9,'Keratoacanthoma'),(-1,-1))
        self.assertEqual(source_label('ACK','PAD'),(-1,-1))
        with self.assertRaises(ValueError):milk_label(9,'unknown subtype')
        with self.assertRaises(ValueError):source_label('AKIEC','PAD')

    def test_metadata_fit_boundary_and_no_outcome_features(self):
        frame=pd.DataFrame({'split':['train']*3,'age':[20,40,np.nan],'sex':['male','female',None],'site':['back','ARM',None],
                            'diagnosis':['melanoma']*3,'binary':[1]*3,'source':['PAD']*3})
        enc=SharedMetadata().fit(frame);a,flags=enc.transform(frame)
        other=frame.assign(binary=0,diagnosis='BCC',source='ISIC2019');b,_=enc.transform(other)
        np.testing.assert_array_equal(a,b);self.assertAlmostEqual(enc.mean,30.)
        with self.assertRaises(ValueError):enc.fit(frame.assign(split='external'))
        masked=mask_metadata(a,np.zeros_like(flags));self.assertTrue((masked==0).all())

    def test_missing_modality_is_exactly_ignored_even_nan(self):
        torch.manual_seed(1);m=BinaryFusion(6,15,7,variant='reliability_subtype',width=8).eval()
        features=torch.randn(4,2,6,requires_grad=True);metadata=torch.randn(4,15);quality=torch.randn(4,2,7)
        available=torch.tensor([[1,0,1]]*4,dtype=torch.float32)
        first=m(features,metadata,quality,available)
        replacement=features.detach().clone();replacement[:,1]=float('nan');qq=quality.clone();qq[:,1]=float('nan')
        second=m(replacement,metadata,qq,available)
        torch.testing.assert_close(first['logit'],second['logit']);self.assertTrue(torch.equal(first['weights'][:,1],torch.zeros(4)))
        loss=training_loss(first,torch.tensor([0,1,0,1]),torch.tensor([-1,0,-1,1]),available,torch.tensor(1.),torch.ones(4))
        loss.backward();self.assertTrue((features.grad[:,1]==0).all())
        with self.assertRaises(ValueError):m(features,metadata,quality,torch.zeros_like(available))

    def test_threshold_uses_calibration_and_handles_ties(self):
        y=np.array([1,1,1,1,0,0]);p=np.array([.1,.2,.2,.9,.3,.8])
        t=choose_threshold(y,p,.75);self.assertAlmostEqual(t,.2);self.assertGreaterEqual((p[y==1]>=t).mean(),.75)
        with self.assertRaises(ValueError):choose_threshold(np.zeros(3),np.ones(3),.95)

    def test_source_grouping_and_lesion_aggregation(self):
        images=self.root/'images';images.mkdir();rng=np.random.default_rng(9)
        rows=[]
        for i in range(30):
            diag='BCC' if i%2 else 'NV'
            for j in range(2):
                name=f'ISIC_{i:04d}_{j}';Image.fromarray(rng.integers(0,255,(24,24,3),dtype=np.uint8)).save(images/f'{name}.jpg')
                rows.append({'image':name,**{d:int(d==diag) for d in ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']},'lesion_id':f'lesion_{i}',
                             'age_approx':50,'sex':'male','anatom_site_general':'posterior torso'})
        f=pd.DataFrame(rows);gt=self.root/'gt.csv';meta=self.root/'meta.csv'
        f[['image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC']].to_csv(gt,index=False)
        f[['image','lesion_id','age_approx','sex','anatom_site_general']].to_csv(meta,index=False)
        result=isic_cases(gt,meta,images);self.assertEqual(len(result),30)
        self.assertEqual(set(result['split']),{'train','val','fusion'});self.assertTrue((result.groupby('group_id')['split'].nunique()==1).all())
        self.assertTrue(all(len(json.loads(x))==2 for x in result.dermoscopic))
        f[['image','age_approx','sex','anatom_site_general']].to_csv(meta,index=False)
        with self.assertRaisesRegex(ValueError,'official ISIC 2019'):
            isic_cases(gt,meta,images)
        with self.assertRaises(ValueError):ImageIndex(images).get('../../ISIC_0000_0.jpg')
        pad=pd.DataFrame([{'patient_id':'PAT_1','lesion_id':'1','img_id':'ISIC_0000_0.jpg','diagnostic':'NEV','age':55,'gender':'MALE','region':'ARM'},
                          {'patient_id':'PAT_1','lesion_id':'1','img_id':'ISIC_0000_1.jpg','diagnostic':'NEV','age':55,'gender':'MALE','region':'ARM'}])
        path=self.root/'pad.csv';pad.to_csv(path,index=False);external=pad_cases(path,images)
        self.assertEqual(len(external),1);self.assertEqual(len(json.loads(external.clinical.iloc[0])),2)
        self.assertEqual(external.group_id.iloc[0],'PAD:PAT_1')

    def test_exact_and_transformed_overlap_quarantine(self):
        from binary_study.overlap import fingerprint,quarantine_external
        rng=np.random.default_rng(11);a=rng.integers(0,255,(72,72,3),dtype=np.uint8)
        p=self.root/'a.png';q=self.root/'b.png';Image.fromarray(a).save(p);Image.fromarray(np.rot90(a)).save(q)
        base={'key':'a','case_id':'MILK:a','group_id':'MILK:a','source':'MILK','split':'train','view':'clinical','path':str(p),'image_id':'a'}
        new={**base,'key':'b','case_id':'PAD:b','group_id':'PAD:patient','source':'PAD','split':'external','path':str(q),'image_id':'b'}
        row=pd.DataFrame([{'case_id':'PAD:b','group_id':'PAD:patient','binary':1}])
        retained,audit=quarantine_external(row,[fingerprint(new)],[fingerprint(base)],radius=3)
        self.assertTrue(retained.empty);self.assertEqual(audit['quarantined_patient_groups'],['PAD:patient'])

    def test_complete_training_calibration_and_evaluation(self):
        out,ref=self.fixture();before=sha(ref/'clinical.pt');self.do_train(out)
        lock=calibrate_and_lock(out,.95);self.assertEqual(lock['calibration_split'],'fusion')
        self.assertEqual(set(lock['models']),{'linear','concat','concat_subtype','reliability','reliability_subtype'})
        result=evaluate(out,draws=30);self.assertIn('MILK/holdout/full',result['models'])
        self.assertEqual(result['models']['MILK/holdout/full']['concat/clean']['lesions'],8)
        self.assertEqual(evaluate(out,draws=30),result)
        self.assertEqual(before,sha(ref/'clinical.pt'))
        with self.assertRaisesRegex(ValueError,'locked sensitivity'):calibrate_and_lock(out,.9)
        with self.assertRaisesRegex(ValueError,'config changed'):train(out,seeds=(42,),device_name='cpu',epochs=3)
        target=out/'models/concat/seed-42/best.pt';target.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Locked artifact changed'):verify_lock(out)

    def test_external_source_is_rejected_by_training(self):
        out,_ref=self.fixture();data=load_features(out/'development_features.npz');data['sources'][0]='PAD'
        with (out/'development_features.npz').open('wb') as f:np.savez_compressed(f,**data)
        write_json(out/'development_feature_identity.json',{'sha256':sha(out/'development_features.npz')})
        with self.assertRaisesRegex(ValueError,'External test'):self.do_train(out)

    def test_holdout_labels_do_not_affect_fitted_head(self):
        out,_ref=self.fixture();data=load_features(out/'development_features.npz')
        train(out,variants=['concat'],seeds=(42,),device_name='cpu',epochs=2,patience=3,batch_size=16,width=8)
        left=load_model(out/'models/concat/seed-42').state_dict()
        second=self.root/'second';second.mkdir()
        for p in out.iterdir():
            if p.is_file() and p.name not in ['training_config.json']:(second/p.name).write_bytes(p.read_bytes())
        mask=data['splits']=='holdout';data['targets'][mask]=1-data['targets'][mask]
        with (second/'development_features.npz').open('wb') as f:np.savez_compressed(f,**data)
        write_json(second/'development_feature_identity.json',{'sha256':sha(second/'development_features.npz')})
        train(second,variants=['concat'],seeds=(42,),device_name='cpu',epochs=2,patience=3,batch_size=16,width=8)
        right=load_model(second/'models/concat/seed-42').state_dict()
        for k in left:torch.testing.assert_close(left[k],right[k],rtol=0,atol=0)


    def test_raw_images_through_train_only_features_and_external_lock(self):
        from binary_study.features import extract_view,assemble
        from binary_study.overlap import fingerprints
        from binary_study.workflow import prepare_external
        class TinyBackbone(torch.nn.Module):
            def forward(self,x):return torch.cat([x.mean((2,3)),x.std((2,3))],1)
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=TinyBackbone();self.head=torch.nn.Linear(6,11)
            def forward(self,x):return self.head(self.backbone(x))
        def factory(_ref,_view,device):
            torch.manual_seed(2);return TinyEncoder().requires_grad_(False).eval().to(device)
        out,ref=self.fixture();frame=pd.read_csv(out/'development.csv');images=self.root/'raw';images.mkdir()
        rng=np.random.default_rng(22)
        for index,row in frame.iterrows():
            for view in ['clinical','dermoscopic']:
                files=[]
                if view=='dermoscopic' or row.source=='MILK':
                    path=images/f'{index}_{view}.png'
                    Image.fromarray(rng.integers(0,255,(40,48,3),dtype=np.uint8)).save(path);files=[str(path)]
                frame.loc[index,view]=json.dumps(files)
        frame.to_csv(out/'development.csv',index=False)
        fingerprints(frame,out/'development_fingerprints.json',workers=1)
        (out/'development_features.npz').unlink();(out/'development_feature_identity.json').unlink()
        write_json(out/'benchmark.json',{'reference_sha256':sha(ref/'manifest.json'),
                   'source_sha256':__import__('binary_study.common',fromlist=['source_identity']).source_identity(),
                   'selected':{'batch_size':32,'amp':False}})
        write_json(out/'protocol.json',{'label_policy':'fixture','phash_radius':3,'pca_dimension':4,'conditions':['clean','blur']})
        for view in ['clinical','dermoscopic']:
            extract_view(out,ref,view,device_name='cpu',conditions=['clean','blur'],workers=1,model_factory=factory)
        assemble(out,pca_dimension=4,conditions=['clean','blur'])
        transforms=joblib.load(out/'transforms.joblib')
        self.assertEqual(set(transforms['fit_case_ids']),set(frame.loc[frame['split']=='train','case_id']))
        enc=SharedMetadata.load(out/'metadata.json');self.assertAlmostEqual(enc.mean,50.)
        data=load_features(out/'development_features.npz');missing=data['sources']=='ISIC2019'
        self.assertTrue((data['features'][:,missing,0]==0).all())
        # Resume reuses verified encoder chunks, not a second extraction with altered outputs.
        for view in ['clinical','dermoscopic']:
            extract_view(out,ref,view,device_name='cpu',conditions=['clean','blur'],workers=1,model_factory=lambda *_:self.fail('cached encoder rebuilt'))
        self.do_train(out);calibrate_and_lock(out)
        before=sha(out/'transforms.joblib');meta_before=sha(out/'metadata.json')
        pad_images=self.root/'pad_images';pad_images.mkdir();rows=[]
        for i in range(4):
            for j in range(2):
                name=f'PAT_{i}_{i}_{j}.png';Image.fromarray(rng.integers(0,255,(40,48,3),dtype=np.uint8)).save(pad_images/name)
                rows.append({'patient_id':f'PAT_{i}','lesion_id':str(i),'img_id':name,'diagnostic':'BCC' if i%2 else 'NEV',
                             'age':60,'gender':'MALE','region':'ARM'})
        metadata=self.root/'pad_metadata.csv';pd.DataFrame(rows).to_csv(metadata,index=False)
        prepare_external(out,metadata,pad_images,workers=1)
        for view in ['clinical','dermoscopic']:
            extract_view(out,ref,view,'external','cpu',['clean','blur'],1,model_factory=factory)
        assemble(out,'external',4,['clean','blur']);result=evaluate(out,'external',draws=30)
        self.assertEqual(result['models']['PAD/external/clinical_metadata']['concat/clean']['lesions'],4)
        self.assertEqual(before,sha(out/'transforms.joblib'));self.assertEqual(meta_before,sha(out/'metadata.json'))
        self.assertTrue(result['primary_comparisons']['PAD/external/clinical_metadata']['all_groups_are_known_patients'])
        with self.assertRaises(ValueError):calibrate_and_lock(out,.9)

    def test_epoch_resume_matches_uninterrupted_training(self):
        import binary_study.training as training
        first,_ref=self.fixture('continuous')
        config=dict(variants=['concat'],seeds=(42,),device_name='cpu',epochs=3,patience=4,batch_size=16,width=8)
        train(first,**config);expected=load_model(first/'models/concat/seed-42').state_dict()
        second=self.root/'interrupted';second.mkdir()
        for path in first.iterdir():
            if path.is_file():(second/path.name).write_bytes(path.read_bytes())
        original=training.atomic_torch;interrupted=[False]
        def stop_after_resume(path,value):
            original(path,value)
            if Path(path).name=='resume.pt' and not interrupted[0]:
                interrupted[0]=True;raise KeyboardInterrupt()
        with patch('binary_study.training.atomic_torch',side_effect=stop_after_resume):
            with self.assertRaises(KeyboardInterrupt):train(second,**config)
        train(second,**config);actual=load_model(second/'models/concat/seed-42').state_dict()
        for key in expected:torch.testing.assert_close(expected[key],actual[key],atol=0,rtol=0)

    def test_external_data_cannot_be_opened_before_lock(self):
        from binary_study.workflow import prepare_external
        out,_ref=self.fixture()
        with self.assertRaises(FileNotFoundError):prepare_external(out,self.root/'not-opened.csv',self.root/'not-opened-images')


    def test_output_isolation_and_reference_tampering(self):
        ref=self.reference()
        for out in [ref/'run',self.root,self.root/'baselines/new',self.root/'src/checkpoints/new']:
            with self.assertRaises(ValueError):initialize(out,ref)
        (ref/'clinical.pt').write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError,'Immutable reference'):initialize(self.root/'safe-run',ref)

if __name__=='__main__':unittest.main()
