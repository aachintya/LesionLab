"""Fixed screening policy, replay against archived metrics, and chart schemas."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.paths import DIAGNOSIS_COLS
from utils.screening import POSITIVE_CLASSES, screening_from_result, screening_scores, presentation_screening
from utils import demo_charts


class ScreeningTests(unittest.TestCase):
    def test_declared_families_and_boundary(self):
        for i, code in enumerate(DIAGNOSIS_COLS):
            vector = np.eye(11)[i]
            self.assertEqual(screening_scores(vector)["positive"], code in POSITIVE_CLASSES)
        vector = np.zeros(11); vector[1] = .5; vector[8] = .5
        self.assertTrue(screening_scores(vector)["positive"])
        vector[1] -= 1e-6; vector[8] += 1e-6
        self.assertFalse(screening_scores(vector)["positive"])

    def test_pooled_rule_can_differ_from_leading_diagnosis_without_altering_it(self):
        vector = np.zeros(11);vector[8] = .4;vector[[0,1,7]] = .2
        result = {"mode": "fusion", "class_code": "NV",
                  "scores": [{"class": code, "score": float(value)} for code,value in zip(DIAGNOSIS_COLS, vector)]}
        original = copy.deepcopy(result)
        screening = screening_from_result(result)
        self.assertTrue(screening["positive"])
        self.assertFalse(screening["mapped_top_diagnosis"]["positive"])
        self.assertEqual(result, original)
        self.assertFalse(screening["calibrated"])
        displayed=presentation_screening(result)
        self.assertFalse(displayed["positive"])
        self.assertEqual(displayed["rule"],"group_leading_diagnosis")
        self.assertNotIn("threshold",displayed)
        self.assertEqual(result,original)
        result["scores"].reverse()
        self.assertEqual(screening_from_result(result), screening)

    def test_invalid_scores_and_validator_results_are_refused(self):
        for vector in (np.full(11,np.nan),np.ones(10)/10,np.zeros(11),np.r_[-.1,np.ones(10)*.11]):
            with self.assertRaises(ValueError):screening_scores(vector)
        with self.assertRaises(ValueError):screening_from_result({"mode":"validator"})

    def test_replay_both_rules_and_both_models_on_all_786_lesions(self):
        with np.load(ROOT/'demo/evaluation/validation_scores.npz') as archive:
            targets=archive['targets'];scores={name:archive[key] for name,key in
                [('saved_original','baseline'),('improved_ensemble','best_macro')]}
        reference=json.loads((ROOT/'demo/evaluation/screening.json').read_text())
        evaluation=reference['evaluations']['risk_screening_all']
        self.assertEqual(reference['protocol']['groups']['risk_screening_all']['positive'],list(POSITIVE_CLASSES))
        y=np.isin(targets,[DIAGNOSIS_COLS.index(code) for code in POSITIVE_CLASSES])
        for name,values in scores.items():
            output=[screening_scores(row) for row in values]
            for rule in ['pooled','mapped']:
                prediction=np.array([row['positive'] if rule=='pooled' else row['mapped_top_diagnosis']['positive'] for row in output])
                counts={'true_positive':int(np.sum(y&prediction)),'false_negative':int(np.sum(y&~prediction)),
                        'true_negative':int(np.sum(~y&~prediction)),'false_positive':int(np.sum(~y&prediction))}
                expected=evaluation['models'][name]
                if rule=='mapped':expected=expected['mapped_11class_argmax']
                self.assertEqual(counts,expected['confusion'])
                self.assertAlmostEqual(float(np.mean(y==prediction)),expected['accuracy'])

    def test_chart_schemas_accept_scores_and_recorded_evaluation_data(self):
        values=np.arange(1,12,dtype=float);values/=values.sum()
        result={'mode':'fusion','class_code':DIAGNOSIS_COLS[int(values.argmax())],
                'scores':[{'class':code,'score':float(value)} for code,value in zip(DIAGNOSIS_COLS,values)],
                'branch_scores':{'clinical':values,'dermoscopic':np.roll(values,1),'metadata':np.roll(values,2)}}
        screening=screening_from_result(result)
        frame=pd.read_csv(ROOT/'demo/evaluation/screening_predictions.csv')
        charts=[demo_charts.group_balance(screening),demo_charts.diagnosis_bars(result['scores']),
                demo_charts.modality_heatmap(result['branch_scores'],DIAGNOSIS_COLS),
                demo_charts.comparison_bars([{'pipeline':'Original','score':.4},{'pipeline':'Improved','score':.7}]),
                demo_charts.evaluation_curves(frame),demo_charts.evaluation_curves(frame,precision_recall=True),
                demo_charts.confusion_chart({'true_negative':152,'false_positive':72,'false_negative':56,'true_positive':506})]
        for chart in charts:self.assertIn('$schema',chart.to_dict())

    def test_historical_ji_adf_comparison_aligns_all_524_archived_decisions(self):
        from scripts.evaluate_historical_screening import evaluate
        result, cases = evaluate()
        self.assertEqual(len(cases), 524)
        self.assertEqual(result['positive_lesions'], 376)
        for name, counts in [('JI-ADF adaptation',(54,94,22,354)),
                             ('Concatenation control',(73,75,20,356)),
                             ('Original late fusion',(79,69,24,352))]:
            self.assertEqual(tuple(result['models'][name]['confusion'].values()), counts)


if __name__=='__main__':unittest.main()
