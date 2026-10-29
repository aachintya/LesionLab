"""Fusion-only selection, fixed holdout comparison and paired uncertainty."""
from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report, confusion_matrix

from components.paths import DIAGNOSIS_COLS, NUM_CLASSES
from improved.training import metrics


def ensemble_probabilities(logits, counts, alpha):
    """Scalar prior restoration after Balanced Softmax; no class-specific threshold tuning."""
    logits = np.asarray(logits)
    counts = np.asarray(counts)
    if logits.ndim != 3 or logits.shape[2] != NUM_CLASSES or counts.shape != (NUM_CLASSES,) or (counts <= 0).any():
        raise ValueError("Expected seed x lesion x 11 logits and positive train class counts")
    if not np.isfinite(logits).all() or not np.isfinite(counts).all() or not 0 <= alpha <= 1:
        raise ValueError("Invalid logits, counts or prior coefficient")
    return softmax(logits + alpha * np.log(counts), axis=2).mean(axis=0)


def select_prior(logits, labels, counts, *, split):
    if split != "fusion":
        raise ValueError("Prior selection is restricted to the fusion split; holdout cannot tune the model")
    records = []
    for alpha in (0.0, 0.5, 1.0):
        scores = metrics(labels, ensemble_probabilities(logits, counts, alpha).argmax(1))
        records.append({"alpha": alpha, **scores})
    # Fixed grid, deterministic tie break. Equal seed weights are never tuned.
    best = max(records, key=lambda row: (row["macro_f1"], row["balanced_accuracy"], row["accuracy"], -row["alpha"]))
    return {"selection_split": "fusion", "criterion": "macro_f1", "alpha": best["alpha"],
            "seed_weights": "equal", "candidates": records}


def detailed_metrics(labels, probabilities):
    predictions = probabilities.argmax(1)
    return {**metrics(labels, predictions),
            "per_class": classification_report(labels, predictions, labels=list(range(NUM_CLASSES)),
                                                target_names=list(DIAGNOSIS_COLS), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(NUM_CLASSES))).tolist()}


def _from_matrix(labels, predictions):
    matrix = np.bincount(labels * NUM_CLASSES + predictions, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    diagonal = matrix.diagonal()
    row_sum, col_sum = matrix.sum(1), matrix.sum(0)
    recall = np.divide(diagonal, row_sum, out=np.zeros(NUM_CLASSES), where=row_sum != 0)
    f1 = np.divide(2 * diagonal, row_sum + col_sum, out=np.zeros(NUM_CLASSES), where=(row_sum + col_sum) != 0)
    return np.array([diagonal.sum() / matrix.sum(), recall.mean(), f1.mean()])


def paired_bootstrap(labels, baseline_pred, candidate_pred, *, repetitions=2000, seed=2026):
    labels, baseline_pred, candidate_pred = map(np.asarray, (labels, baseline_pred, candidate_pred))
    if len(labels) == 0 or not (labels.shape == baseline_pred.shape == candidate_pred.shape) or repetitions < 20:
        raise ValueError("Paired bootstrap requires aligned nonempty predictions and >=20 resamples")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), len(labels))
        deltas.append(_from_matrix(labels[indices], candidate_pred[indices]) - _from_matrix(labels[indices], baseline_pred[indices]))
    deltas = np.asarray(deltas)
    point = _from_matrix(labels, candidate_pred) - _from_matrix(labels, baseline_pred)
    return {"method": "paired lesion bootstrap, percentile 95% CI", "resamples": repetitions, "seed": seed,
            "limitations": "Conditional on this internal split. Rare classes with one lesion have very uncertain performance; "
                           "this interval does not establish external or clinical superiority. Repeated holdout use biases comparisons.",
            "deltas": {name: {"point": float(point[index]), "ci95": np.quantile(deltas[:, index], [0.025, 0.975]).tolist()}
                       for index, name in enumerate(("accuracy", "balanced_accuracy", "macro_f1"))}}
