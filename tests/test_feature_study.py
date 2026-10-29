"""Check head fitting isolation, score columns, and deterministic TTA views."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from utils import feature_study as methods
from scripts.study_frozen_features import views


class FeatureStudyTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(23)
        self.x=rng.normal(size=(44,12))
        self.y=np.tile([0,2,6,10],11)

    def test_auxiliary_training_rows_never_include_fusion_evaluation_rows(self):
        spec=next(item for item in methods.candidate_specs() if item['family']=='lr')
        auxiliary=np.arange(12);fusion=np.arange(12,44)
        folds=[(np.arange(16,32),np.arange(16)),(np.arange(16),np.arange(16,32))]
        result=methods.oof(spec,{spec['feature']:self.x},self.y,auxiliary,fusion,folds)
        self.assertEqual(result.shape,(32,11))
        self.assertTrue(np.allclose(result.sum(1),1))
        with self.assertRaises(ValueError):
            methods.oof(spec,{spec['feature']:self.x},self.y,np.array([12]),fusion,folds)

    def test_score_only_methods_exclude_auxiliary_train_rows(self):
        spec=next(item for item in methods.candidate_specs() if item['family']=='hist')
        np.testing.assert_array_equal(methods.fitting_rows(spec,np.arange(12),np.arange(12,44),np.array([1,3])),[13,15])

    def test_all_classifier_families_preserve_global_class_positions(self):
        original=self.x.copy()
        for family in ['lr','ridge','rbf','hist','extra']:
            spec=next(item for item in methods.candidate_specs() if item['family']==family)
            model=methods.fit(spec,self.x,self.y);result=methods.predict(spec,model,self.x)
            self.assertEqual(result.shape,(44,11))
            self.assertTrue((result[:,[1,3,4,5,7,8,9]]==0).all())
            self.assertTrue(np.allclose(result.sum(1),1))
            np.testing.assert_array_equal(result.argmax(1),model.predict(self.x))
        np.testing.assert_array_equal(self.x,original)

    def test_bad_features_and_pooling_weights_are_rejected(self):
        spec=next(item for item in methods.candidate_specs() if item['family']=='pool')
        with self.assertRaises(ValueError):methods.predict(spec,None,np.full((2,33),np.nan))
        p=np.ones((2,33))/11
        with self.assertRaises(ValueError):methods.predict({**spec,'weights':[-1,1,1]},None,p)

    def test_pooling_matches_its_declared_views_and_ignores_metadata_when_zero(self):
        spec=next(item for item in methods.candidate_specs() if item['id']=='tta_mean_images')
        p=np.random.default_rng(5).dirichlet(np.ones(11),size=(4,3))
        expected=p[:,:2].mean(1)
        np.testing.assert_allclose(methods.predict(spec,None,p.reshape(4,33)),expected)

    def test_tta_preserves_pixels_and_does_not_mutate_input(self):
        source=torch.arange(12).reshape(1,3,2,2);original=source.clone();result=views(source)
        self.assertEqual(len(result),6)
        for value in result:
            self.assertEqual(value.shape,source.shape)
            torch.testing.assert_close(value.flatten().sort().values,source.flatten().sort().values)
        torch.testing.assert_close(result[0],source)
        self.assertEqual(result[1][0,0].tolist(),[[1,0],[3,2]])
        self.assertEqual(result[3][0,0].tolist(),[[1,3],[0,2]])
        torch.testing.assert_close(source,original)


if __name__=='__main__':unittest.main()
