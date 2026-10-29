"""Fusion-only experiments on frozen class-probability vectors.

Search accepts fusion-fit data only. Validation is reserved for the locked choice.
The historical classifier is never fitted or overwritten by this module.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

K = 11
SEED = 20261028


def validate_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if (p.ndim != 3 or p.shape[1:] != (3, K) or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=2), 1, atol=1e-5)):
        raise ValueError("Expected finite, normalized N×3×11 modality probabilities.")
    return p


def candidate_specs():
    candidates = []
    for name, weights in [
        ("mean_all", [1/3, 1/3, 1/3]), ("mean_images", [.5, .5, 0]),
        ("weighted_derm_metadata", [.25, .65, .10]),
        ("weighted_clinical_images", [.6, .4, 0]),
        ("weighted_derm_images", [.4, .6, 0]),
        ("weighted_metadata_30", [.35, .35, .30]),
    ]:
        candidates.append({"id": name, "family": "pool", "weights": weights, "geometric": False})
    for name, weights in [("geometric_all", [1/3, 1/3, 1/3]), ("geometric_images", [.5, .5, 0])]:
        candidates.append({"id": name, "family": "pool", "weights": weights, "geometric": True})
    for feature in ["probability", "centered_log_probability"]:
        for images_only in [False, True]:
            suffix = "images" if images_only else "all"
            for c in [.1, 1., 10.]:
                candidates.append({"id": f"lr_{feature}_{suffix}_scaled_balanced_C{c:g}",
                                   "family": "lr", "feature": feature, "images_only": images_only,
                                   "scale": True, "C": c, "class_weight": "balanced"})
            candidates.append({"id": f"lr_{feature}_{suffix}_scaled_unweighted_C1", "family": "lr",
                               "feature": feature, "images_only": images_only,
                               "scale": True, "C": 1., "class_weight": None})
    for c in [.5, 5., 50.]:
        candidates.append({"id": f"lr_probability_all_unscaled_balanced_C{c:g}", "family": "lr",
                           "feature": "probability", "images_only": False,
                           "scale": False, "C": c, "class_weight": "balanced"})
    return candidates


def historical_recipe():
    return {"id": "historical_recipe_refit", "family": "lr", "feature": "probability",
            "images_only": False, "scale": False, "C": .05, "class_weight": "balanced"}


def feature_matrix(p, spec):
    p = validate_probabilities(p)
    p = p[:, :2] if spec["images_only"] else p
    if spec["feature"] == "centered_log_probability":
        p = np.log(np.clip(p, 1e-6, 1))
        p = p - p.mean(axis=2, keepdims=True)
    elif spec["feature"] != "probability":
        raise ValueError("Unknown fusion feature representation.")
    return p.reshape(len(p), -1)


def fit_candidate(spec, p, y):
    p = validate_probabilities(p)
    y = np.asarray(y, dtype=int)
    if len(y) != len(p) or not np.isin(y, np.arange(K)).all():
        raise ValueError("Fusion labels do not match probability rows.")
    if spec["family"] == "pool":
        weights = np.asarray(spec["weights"])
        if weights.shape != (3,) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError("Invalid pooling weights.")
        return {"spec": spec, "estimator": None}
    estimator = LogisticRegression(C=spec["C"], class_weight=spec["class_weight"],
                                   max_iter=2000, tol=1e-5, solver="lbfgs", random_state=SEED)
    if spec["scale"]:
        estimator = make_pipeline(StandardScaler(), estimator)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(feature_matrix(p, spec), y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Fusion optimizer did not converge.")
    return {"spec": spec, "estimator": estimator}


def predict_candidate(candidate, p):
    p = validate_probabilities(p)
    spec = candidate["spec"]
    if spec["family"] == "pool":
        weights = np.asarray(spec["weights"])
        if spec["geometric"]:
            logits = np.einsum("nmk,m->nk", np.log(np.clip(p, 1e-6, 1)), weights)
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
        else:
            result = np.einsum("nmk,m->nk", p, weights)
        return result / result.sum(axis=1, keepdims=True)
    estimator = candidate["estimator"]
    observed = estimator.predict_proba(feature_matrix(p, spec))
    result = np.zeros((len(p), K))
    # A class with one fusion lesion is absent from one fold's fitting data.
    # Preserve canonical class positions instead of shifting the output columns.
    result[:, np.asarray(estimator.classes_, dtype=int)] = observed
    return result


def confusion(y, prediction):
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    if y.shape != prediction.shape or not np.isin(y, np.arange(K)).all() or not np.isin(prediction, np.arange(K)).all():
        raise ValueError("Invalid canonical class indices.")
    return np.bincount(y * K + prediction, minlength=K*K).reshape(K, K)


def confusion_metrics(cm):
    correct = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(correct, support, out=np.zeros(K), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(K), where=(support + predicted) > 0)
    return {"accuracy": float(correct.sum() / cm.sum()), "balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean())}


def metrics(y, p):
    p = np.asarray(p, dtype=float)
    if p.shape != (len(y), K) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid fused probabilities.")
    y = np.asarray(y, dtype=int)
    result = confusion_metrics(confusion(y, p.argmax(1)))
    result["log_loss"] = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean())
    result["brier_score"] = float(np.square(p - np.eye(K)[y]).sum(axis=1).mean())
    return result


def out_of_fold(spec, p, y, folds):
    result, assigned = np.zeros((len(y), K)), np.zeros(len(y), dtype=int)
    for train, test in folds:
        if len(np.intersect1d(train, test)):
            raise ValueError("Fusion fold leaks fitting rows into evaluation.")
        fitted = fit_candidate(spec, p[train], y[train])
        result[test] = predict_candidate(fitted, p[test])
        assigned[test] += 1
    if not (assigned == 1).all():
        raise ValueError("Every fusion row must have exactly one out-of-fold prediction.")
    return result


def paired_bootstrap(y, baseline, candidate, *, draws=5000, seed=SEED):
    y = np.asarray(y, dtype=int)
    a, b = np.asarray(baseline).argmax(1), np.asarray(candidate).argmax(1)
    names = ["accuracy", "balanced_accuracy", "macro_f1"]
    observed_a, observed_b = confusion_metrics(confusion(y, a)), confusion_metrics(confusion(y, b))
    deltas = np.empty((draws, 3))
    rng = np.random.default_rng(seed)
    for index in range(draws):
        rows = rng.integers(0, len(y), len(y))
        left, right = confusion_metrics(confusion(y[rows], a[rows])), confusion_metrics(confusion(y[rows], b[rows]))
        deltas[index] = [right[name] - left[name] for name in names]
    return {name: {"difference": observed_b[name] - observed_a[name],
                   "lower_95": float(np.quantile(deltas[:, j], .025)),
                   "upper_95": float(np.quantile(deltas[:, j], .975))} for j, name in enumerate(names)}


def class_rows(y, prediction, classes):
    cm = confusion(y, prediction)
    result = []
    for i, name in enumerate(classes):
        support, predicted, correct = int(cm[i].sum()), int(cm[:, i].sum()), int(cm[i, i])
        alternatives = sorted([(int(cm[i, j]), classes[j]) for j in range(K) if i != j and cm[i, j]], reverse=True)
        result.append({"class": name, "support": support, "correct": correct,
                       "recall": correct/support if support else 0.,
                       "precision": correct/predicted if predicted else 0.,
                       "f1": 2*correct/(support+predicted) if support+predicted else 0.,
                       "top_confusions": [{"predicted": other, "count": count} for count, other in alternatives[:3]]})
    return result
