"""Grounded, local summaries of existing predictions; no diagnostic text generation.

The ambiguity policy is a transparent demo display rule, not a fitted confidence
estimator. It does not change the underlying classifier or its argmax prediction.
"""
from __future__ import annotations

import numpy as np

from components.paths import DIAGNOSIS_COLS, NUM_CLASSES

INTERPRETATION_VERSION = "1.1"
MIN_SCORE_MARGIN = 0.05
MODALITY_NAMES = {"clinical": "Clinical photograph", "dermoscopic": "Dermoscopic image",
                  "metadata": "Patient metadata"}
METADATA_NAMES = {"age_approx": "age", "skin_tone_class": "skin tone", "sex": "sex", "site": "body site"}


def _vector(values):
    vector = np.asarray(values, dtype=np.float64)
    if (vector.shape != (NUM_CLASSES,) or not np.isfinite(vector).all()
            or (vector < 0).any() or not np.isclose(vector.sum(), 1, atol=1e-4)):
        raise ValueError("Cannot interpret invalid model scores.")
    return vector


def _is_provided(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "nan", "none", "not provided"}
    return not (isinstance(value, (float, np.floating)) and not np.isfinite(value))


def interpret_result(result, *, metadata=None):
    """Summarize recorded scores without inventing image findings or disease risks."""
    if result.get("mode") == "validator":
        raise ValueError("Input validation does not produce a disease interpretation.")
    rows = result.get("scores", [])
    by_code = {row["class"]: row for row in rows}
    if len(rows) != NUM_CLASSES or set(by_code) != set(DIAGNOSIS_COLS):
        raise ValueError("An interpretation needs one score per canonical class.")
    values = _vector([by_code[code]["score"] for code in DIAGNOSIS_COLS])
    order = np.argsort(-values, kind="stable")
    leading, alternative = map(int, order[:2])
    margin = float(values[leading] - values[alternative])
    code = DIAGNOSIS_COLS[leading]
    if result.get("class_code") != code:
        raise ValueError("The recorded prediction and model scores disagree.")
    names = {key: row["name"] for key, row in by_code.items()}
    ranked = [{"class": DIAGNOSIS_COLS[int(i)], "name": names[DIAGNOSIS_COLS[int(i)]],
               "score": float(values[i])} for i in order[:3]]
    branches = result.get("branch_scores", {})
    expected = {"clinical", "dermoscopic", "metadata"} if result.get("mode") == "fusion" else {result.get("mode")}
    if set(branches) != expected:
        raise ValueError("Missing modality predictions for this analysis mode.")
    modalities = []
    for key in MODALITY_NAMES:
        if key not in branches:
            continue
        branch = _vector(branches[key])
        winner = int(np.argmax(branch))
        branch_code = DIAGNOSIS_COLS[winner]
        modalities.append({"key": key, "label": MODALITY_NAMES[key], "class": branch_code,
                           "name": names[branch_code], "score": float(branch[winner]),
                           "matches_combined": branch_code == code})
    image_modalities = [row for row in modalities if row["key"] != "metadata"]
    image_codes = {row["class"] for row in image_modalities}
    views_disagree = len(image_codes) > 1
    fusion_differs = result["mode"] == "fusion" and code not in image_codes
    reasons = []
    if margin + 1e-12 < MIN_SCORE_MARGIN:
        reasons.append({"code": "close_scores", "text":
                        "The two leading categories have closely spaced model scores."})
    if views_disagree:
        reasons.append({"code": "views_disagree", "text":
                        "The clinical and dermoscopic image models favour different categories."})
    if fusion_differs:
        reasons.append({"code": "fusion_differs", "text":
                        "The combined result differs from the leading category of both image models."})
    components = []
    ensemble = result.get("ensemble_members", {})
    if ensemble:
        if set(ensemble) != {"feature_lr", "tta_images"}:
            raise ValueError("Unexpected ensemble components.")
        for key, label in (("feature_lr", "Image features + metadata classifier"),
                           ("tta_images", "Six-view clinical + dermoscopic average")):
            vector = _vector(ensemble[key])
            winner = int(np.argmax(vector))
            component_code = DIAGNOSIS_COLS[winner]
            components.append({"key": key, "label": label, "class": component_code,
                               "name": names[component_code], "score": float(vector[winner]),
                               "matches_combined": component_code == code})
        if components[0]["class"] != components[1]["class"]:
            reasons.append({"code": "ensemble_disagreement", "text":
                            "The feature classifier and six-view image average favour different categories."})
    inconclusive = bool(reasons)
    if inconclusive:
        summary = (f"The model ranks {names[code]} first, with {ranked[1]['name']} as the closest alternative. "
                   "The available model outputs do not support a clear single-category interpretation for this input.")
    else:
        summary = (f"The model favours {names[code]} over the other categories for the supplied input. "
                   f"The closest alternative is {ranked[1]['name']}.")
    if result["mode"] == "fusion":
        if views_disagree:
            agreement = "The two image views disagree. Review their separate findings below alongside the combined result."
        elif fusion_differs:
            agreement = "The image views agree with each other, while the combined model favours a different category."
        else:
            agreement = f"Both image models and the combined model favour {names[code]}."
        agreement += " Agreement describes model outputs and does not establish that they are correct."
    else:
        agreement = (f"This summary uses the {MODALITY_NAMES[result['mode']].lower()} model alone. "
                     "A comparison between image modalities is unavailable in this mode.")
    supplied = [label for key, label in METADATA_NAMES.items() if _is_provided((metadata or {}).get(key))]
    if result["mode"] == "fusion":
        metadata_note = ("No patient metadata were entered; the metadata branch uses its trained handling of unknown values."
                         if not supplied else f"Metadata entered: {', '.join(supplied)}. Other fields use trained unknown-value handling."
                         if len(supplied) < len(METADATA_NAMES) else "All four patient metadata fields were entered.")
    else:
        metadata_note = "Patient metadata were not used in this mode."
    return {
        "version": INTERPRETATION_VERSION,
        "status": "inconclusive" if inconclusive else "model_preference",
        "title": "Inconclusive result" if inconclusive else "Model preference",
        "summary": summary, "reasons": reasons, "leading_categories": ranked,
        "ensemble_components": components,
        "ensemble_agreement": ("Both ensemble components favour the same category."
                               if len(components) == 2 and components[0]["class"] == components[1]["class"] else
                               "The ensemble components have different preferences; their combined scores determine the final ranking."
                               if components else None),
        "modalities": modalities, "image_views_agree": not views_disagree if len(image_modalities) == 2 else None,
        "agreement_summary": agreement, "metadata_note": metadata_note,
        "provided_metadata_fields": supplied,
        "score_margin": margin,
        "policy": {"kind": "demo_ambiguity_rules", "minimum_score_margin": MIN_SCORE_MARGIN,
                   "image_disagreement_is_inconclusive": True,
                   "fusion_outside_image_choices_is_inconclusive": True,
                   "ensemble_disagreement_is_inconclusive": bool(ensemble),
                   "calibrated": False,
                   "description": "Inconclusive flags a top-two score gap below 5 percentage points, disagreement between image models, "
                                  "a combined category favoured by neither image model, or disagreement between ensemble components when present. These are unvalidated display rules, "
                                  "not a clinical confidence estimate. Scores and the original argmax prediction remain unchanged."},
        "scope": "Research interpretation of model outputs; it does not establish a diagnosis or estimate individual disease risk.",
    }
