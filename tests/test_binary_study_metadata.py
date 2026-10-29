"""Regression coverage for official ISIC repeated-image metadata conflicts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from binary_study.data import isic_cases,isic_metadata
from binary_study.overlap import image_records,quarantine_development


class ISICMetadataTests(unittest.TestCase):
    def test_age_median_is_label_blind_order_independent_and_audited(self):
        frame=pd.DataFrame({'age_approx':[50,55,np.nan,999], 'sex':['Male','male',None,'unknown'],
                            'anatom_site_general':['posterior torso','anterior torso',None,None],
                            'diagnosis':['NV']*4,'binary':[0]*4})
        age,sex,site,audit=isic_metadata(frame)
        self.assertEqual((age,sex,site),(52.5,'male','trunk'))
        self.assertFalse(audit['metadata_quarantine']);self.assertTrue(audit['metadata_age_varied'])
        self.assertEqual(json.loads(audit['metadata_observations'])['missing_or_invalid_age_rows'],2)
        self.assertEqual(isic_metadata(frame),isic_metadata(frame.iloc[::-1].assign(diagnosis='MEL',binary=1)))

    def test_sex_and_coarse_site_conflicts_are_flagged(self):
        frame=pd.DataFrame({'age_approx':[55,55],'sex':['male','female'],
                            'anatom_site_general':['head/neck','posterior torso']})
        _age,sex,site,audit=isic_metadata(frame)
        self.assertIsNone(sex);self.assertIsNone(site)
        self.assertEqual(json.loads(audit['metadata_quarantine_reasons']),['conflicting_sex','conflicting_coarse_site'])

    def test_unknown_fields_remain_missing(self):
        frame=pd.DataFrame({'age_approx':['unknown',-1],'sex':[None,'unknown'],
                            'anatom_site_general':['unknown',None]})
        age,sex,site,audit=isic_metadata(frame)
        self.assertEqual((age,sex,site),(None,None,None));self.assertFalse(audit['metadata_quarantine'])

    def fixture(self,root):
        images=root/'images';images.mkdir();rows=[]
        for i in range(30):
            diag='BCC' if i%2 else 'NV'
            for j in range(2):
                name=f'ISIC_{i:04d}_{j}'
                Image.new('RGB',(8,8),(i,j,50)).save(images/f'{name}.jpg')
                rows.append({'image':name,**{d:int(d==diag) for d in ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']},
                             'lesion_id':f'lesion_{i}','age_approx':50+5*j if i==0 else 50,
                             'sex':'female' if i==1 and j==1 else 'male',
                             'anatom_site_general':'head/neck' if i==2 and j==1 else 'posterior torso'})
        frame=pd.DataFrame(rows);gt=root/'gt.csv';meta=root/'meta.csv'
        frame[['image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC']].to_csv(gt,index=False)
        frame[['image','lesion_id','age_approx','sex','anatom_site_general']].to_csv(meta,index=False)
        return frame,gt,meta,images

    def test_real_adapter_retains_images_and_quarantines_whole_group(self):
        with tempfile.TemporaryDirectory() as temp:
            frame,gt,meta,images=self.fixture(Path(temp));cases=isic_cases(gt,meta,images)
            self.assertEqual(len(cases),30);self.assertEqual(len(image_records(cases)),60)
            first=cases[cases.case_id.eq('ISIC2019:lesion_0')].iloc[0]
            self.assertEqual(first.age,52.5)
            self.assertEqual(int(cases.metadata_quarantine.sum()),2)
            # Another case belonging to the same known patient is also quarantined.
            cases.loc[cases.case_id.eq('ISIC2019:lesion_3'),'group_id']='ISIC2019:lesion_1'
            retained,audit=quarantine_development(cases,[])
            self.assertEqual(len(retained),27)
            self.assertEqual(audit['metadata_conflict_groups'],['ISIC2019:lesion_1','ISIC2019:lesion_2'])
            self.assertEqual(audit['isic_cases_with_varying_age'],1)
            self.assertTrue(cases.binary.isin([0,1]).all())
            self.assertEqual(set(retained['split']),{'train','val','fusion'})

    def test_diagnosis_and_patient_conflicts_still_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            frame,gt,meta,images=self.fixture(Path(temp))
            labels=pd.read_csv(gt);labels.loc[1,['NV','BCC']]=[0,1];labels.to_csv(gt,index=False)
            with self.assertRaisesRegex(ValueError,'conflicting labels'):isic_cases(gt,meta,images)
            frame[['image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC']].to_csv(gt,index=False)
            metadata=pd.read_csv(meta);metadata['patient_id']='patient_1';metadata.loc[1,'patient_id']='patient_2';metadata.to_csv(meta,index=False)
            with self.assertRaisesRegex(ValueError,'patient_id'):isic_cases(gt,meta,images)


if __name__=='__main__':unittest.main()
