"""Fixed equal-weight ensembles of independently cross-fitted frozen-feature heads."""
from itertools import combinations

import numpy as np

MEMBERS = ("scaled_scores", "feature_lr", "feature_mlp", "tta_images")


def specs():
    result = []
    for count in [2, 3, 4]:
        for members in combinations(MEMBERS, count):
            for rule in ["arithmetic", "geometric"]:
                result.append({"id": f"{rule}_" + "_".join(members),
                               "members": list(members), "rule": rule})
    return result


def combine(spec, predictions):
    members = spec["members"]
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("An ensemble needs distinct members.")
    p = np.stack([np.asarray(predictions[name], dtype=float) for name in members])
    if p.ndim != 3 or p.shape[2] != 11 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite, nonnegative N by 11 member scores.")
    if not np.allclose(p.sum(2), 1, atol=1e-5):
        raise ValueError("Member scores must sum to one.")
    if spec["rule"] == "arithmetic":
        result = p.mean(0)
    elif spec["rule"] == "geometric":
        score = np.log(np.clip(p, 1e-12, 1)).mean(0)
        result = np.exp(score - score.max(1, keepdims=True))
    else:
        raise ValueError("Unknown ensemble rule.")
    return result / result.sum(1, keepdims=True)
