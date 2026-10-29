"""Regroup archived holdout decisions; preserve all original scores and files."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from components.paths import DIAGNOSIS_COLS
from utils.screening import POSITIVE_CLASSES


def evaluate():
    evidence = ROOT / "demo/evaluation"
    sources = {"JI-ADF adaptation": evidence / "ji_adf.holdout.csv",
               "Concatenation control": evidence / "concat_fixed.holdout.csv",
               "Original late fusion": evidence / "baseline_holdout_predictions.csv"}
    original = pd.read_csv(sources["Original late fusion"]).set_index("lesion_id")
    assert len(original) == 524 and original.index.is_unique
    positive = [DIAGNOSIS_COLS.index(code) for code in POSITIVE_CLASSES]
    reference = np.isin(original.target, positive)
    result = {"split": "historical_holdout", "lesions": len(original),
              "positive_lesions": int(reference.sum()), "negative_lesions": int((~reference).sum()),
              "rule": "group_existing_11class_argmax", "positive_classes": list(POSITIVE_CLASSES),
              "no_training_or_threshold_fitting": True, "models": {}, "source_sha256": {},
              "scope": "Our JI-ADF adaptation, not the published paper's reported performance. "
                       "The new ensemble's validation result uses a separate cohort."}
    cases = pd.DataFrame({"lesion_id": original.index, "positive_reference": reference})
    for name, path in sources.items():
        frame = pd.read_csv(path).set_index("lesion_id")
        assert frame.index.is_unique and set(frame.index) == set(original.index)
        frame = frame.loc[original.index]
        np.testing.assert_array_equal(frame.target, original.target)
        if "prob_AKIEC" in frame:
            scores = frame[["prob_" + code for code in DIAGNOSIS_COLS]].to_numpy()
            np.testing.assert_array_equal(scores.argmax(1), frame.predicted.to_numpy())
        prediction = np.isin(frame.predicted, positive)
        tn, fp, fn, tp = map(int, confusion_matrix(reference, prediction, labels=[0, 1]).ravel())
        result["models"][name] = {"accuracy": float(accuracy_score(reference, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(reference, prediction)),
            "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
            "macro_f1": float(f1_score(reference, prediction, average="macro", zero_division=0)),
            "confusion": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp}}
        result["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases[name] = prediction
    return result, cases


if __name__ == "__main__":
    result, cases = evaluate()
    out = ROOT / "demo/evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "historical_holdout.json").write_text(json.dumps(result, indent=2) + "\n")
    cases.to_csv(out / "historical_holdout_groups.csv", index=False)
    print(json.dumps(result, indent=2))
