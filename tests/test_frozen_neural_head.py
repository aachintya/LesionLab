"""Check downstream-head isolation, replay and finite equal-weight ensembles."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import frozen_neural_head as heads, frozen_ensemble as ensemble


class FrozenHeadTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.x = rng.normal(size=(48, 140))
        self.y = np.tile(np.arange(8), 6)

    def test_scaler_only_fits_supplied_training_rows_and_inputs_unchanged(self):
        original = self.x.copy()
        spec = {**heads.specs()[0], "epochs": 1}
        model, scaler = heads.fit(spec, self.x[:32], self.y[:32])
        np.testing.assert_allclose(scaler.mean_, self.x[:32].mean(0))
        self.assertEqual(scaler.n_samples_seen_, 32)
        p = heads.predict(model, scaler, self.x[32:])
        self.assertEqual(p.shape, (16, 11))
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(p).all())
        self.assertFalse(model.training)
        np.testing.assert_array_equal(self.x, original)

    def test_state_dict_replay_and_deterministic_fit(self):
        spec = {**heads.specs()[0], "epochs": 1}
        model, scaler = heads.fit(spec, self.x, self.y)
        state = deepcopy(model.state_dict())
        replay = heads.build(spec, 140)
        replay.load_state_dict(state)
        expected = heads.predict(model, scaler, self.x)
        np.testing.assert_array_equal(heads.predict(replay, scaler, self.x), expected)
        fitted_again, second_scaler = heads.fit(spec, self.x, self.y)
        np.testing.assert_array_equal(heads.predict(fitted_again, second_scaler, self.x), expected)

    def test_gated_head_geometry_and_sampling_fit(self):
        spec = {**next(s for s in heads.specs() if s["family"] == "gated" and s["balance"] == "soft_sampling"), "epochs": 1}
        with self.assertRaises(ValueError):
            heads.build(spec, 139)
        model, scaler = heads.fit(spec, self.x, self.y)
        result = heads.predict(model, scaler, self.x)
        self.assertEqual(result.shape, (48, 11))
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaises(ValueError):
            heads.fit(spec, np.full((48, 140), np.nan), self.y)

    def test_all_fixed_specifications_have_unique_names(self):
        for methods, count in [(heads.specs(), 13), (ensemble.specs(), 22)]:
            self.assertEqual(len(methods), count)
            self.assertEqual(len({item["id"] for item in methods}), count)

    def test_ensemble_formulas_and_no_input_mutation(self):
        rng = np.random.default_rng(9)
        p = {name: rng.dirichlet(np.ones(11), size=8) for name in ensemble.MEMBERS}
        original = deepcopy(p)
        spec = {"members": ["scaled_scores", "feature_mlp"], "rule": "arithmetic"}
        result = ensemble.combine(spec, p)
        np.testing.assert_allclose(result, (p["scaled_scores"] + p["feature_mlp"]) / 2)
        result = ensemble.combine({**spec, "rule": "geometric"}, p)
        expected = np.sqrt(p["scaled_scores"] * p["feature_mlp"])
        expected /= expected.sum(1, keepdims=True)
        np.testing.assert_allclose(result, expected)
        for name in p:
            np.testing.assert_array_equal(p[name], original[name])

    def test_ensemble_rejects_invalid_scores_and_duplicate_members(self):
        p = {name: np.ones((3, 11)) / 11 for name in ensemble.MEMBERS}
        spec = ensemble.specs()[0]
        for invalid in [np.full((3, 11), np.nan), np.ones((3, 11)), np.full((3, 11), -1 / 11)]:
            with self.assertRaises(ValueError):
                ensemble.combine(spec, {**p, spec["members"][0]: invalid})
        with self.assertRaises(ValueError):
            ensemble.combine({**spec, "members": ["scaled_scores", "scaled_scores"]}, p)
        with self.assertRaises(ValueError):
            ensemble.combine({**spec, "rule": "unknown"}, p)


if __name__ == "__main__":
    unittest.main()
