from typing import cast

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Runtime: sklearn accepts 0.0 for zero_division; stubs type it as str only.
ZERO_DIVISION = cast(str, 0.0)


def calculate_metrics(y_true, y_pred, y_prob=None, average="binary"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=ZERO_DIVISION),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=ZERO_DIVISION),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    if y_prob is not None:
        try:
            if average == "binary":
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics