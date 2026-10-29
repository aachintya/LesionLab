"""Guard the experiment's feature order, fold isolation, classes and metrics."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.fusion_study import (candidate_specs, historical_recipe, feature_matrix, fit_candidate,
                               predict_candidate, validate_probabilities, out_of_fold, metrics,
                               paired_bootstrap)


class FusionStudyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.p = rng.dirichlet(np.ones(11), size=(33, 3))
        self.y = np.tile(np.arange(11), 3)

    def test_feature_order_and_image_only_excludes_metadata(self):
        spec = historical_recipe()
        x = feature_matrix(self.p, spec)
        for index in range(3):np.testing.assert_array_equal(x[:,index*11:(index+1)*11], self.p[:,index])
        spec = {**spec, "images_only":True}
        altered = self.p.copy();altered[:,2] = 1/11
        np.testing.assert_array_equal(feature_matrix(self.p,spec),feature_matrix(altered,spec))

    def test_fitted_probability_columns_keep_canonical_positions_when_a_class_is_absent(self):
        keep = self.y != 6
        spec = next(item for item in candidate_specs() if item["family"] == "lr")
        candidate = fit_candidate(spec, self.p[keep], self.y[keep])
        predicted = predict_candidate(candidate, self.p)
        self.assertEqual(predicted.shape,(33,11))
        self.assertTrue((predicted[:,6] == 0).all())
        self.assertTrue(np.allclose(predicted.sum(1),1))
        observed = candidate["estimator"].predict_proba(feature_matrix(self.p,spec))
        np.testing.assert_array_equal(predicted[:,candidate["estimator"].classes_],observed)

    def test_out_of_fold_rejects_overlap_or_missing_coverage(self):
        spec = candidate_specs()[0]
        with self.assertRaises(ValueError):out_of_fold(spec,self.p,self.y,[(np.arange(20),np.arange(5))])
        with self.assertRaises(ValueError):out_of_fold(spec,self.p,self.y,[(np.arange(20),np.arange(20,30))])
        folds = [(np.arange(11,33),np.arange(11)),(np.r_[0:11,22:33],np.arange(11,22)),(np.arange(22),np.arange(22,33))]
        np.testing.assert_allclose(out_of_fold(spec,self.p,self.y,folds),self.p.mean(axis=1))

    def test_metrics_match_sklearn_with_fixed_classes(self):
        p = self.p[:,0];pred = p.argmax(1)
        result = metrics(self.y,p)
        self.assertAlmostEqual(result["accuracy"],accuracy_score(self.y,pred))
        self.assertAlmostEqual(result["macro_f1"],f1_score(self.y,pred,labels=np.arange(11),average="macro",zero_division=0))
        self.assertAlmostEqual(result["balanced_accuracy"],recall_score(self.y,pred,labels=np.arange(11),average="macro",zero_division=0))
        # Absent classes remain in the metric denominator.
        self.assertAlmostEqual(metrics(self.y[:2],np.eye(11)[self.y[:2]])["macro_f1"],2/11)

    def test_identical_predictions_have_zero_paired_interval(self):
        result = paired_bootstrap(self.y,self.p[:,0],self.p[:,0],draws=100)
        for values in result.values():self.assertEqual(values,{"difference":0.,"lower_95":0.,"upper_95":0.})

    def test_pooling_is_normalized_and_does_not_modify_inputs(self):
        before = self.p.copy()
        for spec in candidate_specs():
            if spec["family"] == "pool":
                fitted = fit_candidate(spec,self.p,self.y)
                prediction = predict_candidate(fitted,self.p)
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.allclose(prediction.sum(1),1))
        spec = next(item for item in candidate_specs() if item["id"] == "mean_images")
        np.testing.assert_allclose(predict_candidate(fit_candidate(spec,self.p,self.y),self.p), self.p[:,:2].mean(1))
        np.testing.assert_array_equal(self.p,before)

    def test_invalid_probability_inputs_are_rejected(self):
        for values in [np.zeros((2,3,11)), np.ones((2,2,11))/11, np.full((2,3,11),np.nan)]:
            with self.assertRaises(ValueError):validate_probabilities(values)


if __name__ == "__main__":unittest.main()
