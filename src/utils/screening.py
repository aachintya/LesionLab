"""Fixed broader screening derived from unchanged eleven-class model scores."""
from __future__ import annotations

import numpy as np

from components.paths import DIAGNOSIS_COLS

SCREENING_VERSION = "1.0"
POSITIVE_CLASSES = ("AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA")
NEGATIVE_CLASSES = tuple(code for code in DIAGNOSIS_COLS if code not in POSITIVE_CLASSES)
POSITIVE_LABEL = "Malignant / precancerous group"
NEGATIVE_LABEL = "Benign / inflammatory group"
THRESHOLD = .5
PRESENTATION_RULE = "group_leading_diagnosis"


def screening_scores(values):
    """Return two declared rules without changing the original class prediction."""
    vector = np.asarray(values, dtype=float)
    if (vector.shape != (11,) or not np.isfinite(vector).all() or (vector < 0).any()
            or not np.isclose(vector.sum(), 1, atol=1e-4)):
        raise ValueError("Screening requires finite scores for all eleven classes.")
    score = float(np.clip(vector[[DIAGNOSIS_COLS.index(code) for code in POSITIVE_CLASSES]].sum(), 0, 1))
    positive = score >= THRESHOLD
    top_code = DIAGNOSIS_COLS[int(np.argmax(vector))]
    mapped = top_code in POSITIVE_CLASSES
    return {"version": SCREENING_VERSION, "positive_score": score, "negative_score": 1-score,
            "threshold": THRESHOLD, "positive": positive,
            "label": POSITIVE_LABEL if positive else NEGATIVE_LABEL,
            "rule": "sum_positive_class_scores_ge_0.5", "calibrated": False,
            "near_boundary": bool(abs(score-THRESHOLD) < .05),
            "mapped_top_diagnosis": {"class_code": top_code, "positive": mapped,
                                     "label": POSITIVE_LABEL if mapped else NEGATIVE_LABEL},
            "rules_agree": positive == mapped,
            "groups": {"positive": list(POSITIVE_CLASSES), "negative": list(NEGATIVE_CLASSES)},
            "scope": "Derived grouping of eleven-class outputs; includes precancerous and mixed diagnostic families."}


def screening_from_result(result):
    if result.get("mode") == "validator":
        raise ValueError("Input validation has no lesion screening output.")
    rows = result["scores"]
    by_code = {row["class"]: row["score"] for row in rows}
    if len(rows) != 11 or set(by_code) != set(DIAGNOSIS_COLS):
        raise ValueError("Screening needs every canonical diagnosis exactly once.")
    return screening_scores([by_code[code] for code in DIAGNOSIS_COLS])


def presentation_screening(result):
    """One fixed demo policy; retain pooled scores separately for research replay."""
    screening = screening_from_result(result)
    leading = screening["mapped_top_diagnosis"]
    if result.get("class_code", leading["class_code"]) != leading["class_code"]:
        raise ValueError("The displayed diagnosis must match the original class-score ranking.")
    return {"rule": PRESENTATION_RULE, **leading, "groups": screening["groups"],
            "description": "Map the existing leading 11-class diagnosis to its diagnostic family.",
            "calibrated": False, "scope": screening["scope"]}
