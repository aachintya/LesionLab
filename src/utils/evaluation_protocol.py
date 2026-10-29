"""Identify the held-out examples and inputs independently of the training rows."""
import csv
import hashlib
import json

from components.paths import DIAGNOSIS_COLS


def evaluation_fingerprint(csv_path, split):
    columns = ("lesion_id", "isic_id", "image_type", "target", "age_approx", "sex", "site", "skin_tone_class")
    with open(csv_path, newline="") as handle:
        rows = [{name: row.get(name, "") for name in columns}
                for row in csv.DictReader(handle) if row.get("split") == split]
    if not rows:
        raise ValueError(f"No examples found for evaluation split {split!r}.")
    rows.sort(key=lambda row: (row["lesion_id"], row["isic_id"], row["image_type"]))
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metric_protocol(csv_path, split):
    return {"decision_rule": "argmax", "class_order": list(DIAGNOSIS_COLS),
            "evaluation_fingerprint": evaluation_fingerprint(csv_path, split)}
