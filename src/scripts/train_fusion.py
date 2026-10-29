"""Train fusion LogisticRegression on stacked modality softmax (train split)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
if str(PROJECT_SRC / "components") not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC / "components"))


def _default_csv() -> Path:
    from components.paths import FINAL_METADATA_WITH_SPLIT_CSV

    return FINAL_METADATA_WITH_SPLIT_CSV


def train_fusion(
    csv_path: Path,
    *,
    checkpoint_path: Path | None = None,
    tta: bool = True,
    fit_split: str = "train",
    eval_split: str = "val",
    validate_inputs: bool = True,
) -> Path:
    from models.fusion_lr import (
        FUSION_CHECKPOINT_NAME,
        build_fusion_feature_matrix,
        labels_for_lesions,
        train_fusion_lr,
    )
    from utils.eval_inference import collect_lesion_modality_probs, require_checkpoints

    require_checkpoints(PROJECT_SRC)
    ckpt = checkpoint_path or (PROJECT_SRC / "checkpoints" / FUSION_CHECKPOINT_NAME)

    if fit_split == "train":
        print("WARNING: in-sample stacking can overfit; prefer --fit-split fusion with a dedicated split.")
    print(f"Collecting {fit_split}-split base model predictions...")
    train_inf = collect_lesion_modality_probs(
        fit_split,
        csv_path,
        project_src=PROJECT_SRC,
        tta=tta, validate_inputs=validate_inputs,
    )
    if not train_inf.has_gt:
        raise ValueError("Train split must have ground-truth labels for fusion training.")

    X_train = build_fusion_feature_matrix(
        train_inf.lesion_ids,
        train_inf.clinical_probs,
        train_inf.dermoscopic_probs,
        train_inf.metadata_probs,
    )
    y_train = labels_for_lesions(train_inf.eval_metadata, train_inf.lesion_ids)
    from components.paths import NUM_CLASSES
    if set(y_train) != set(range(NUM_CLASSES)):
        raise ValueError("Fusion fitting split must contain all 11 classes. Increase its size or revise the grouped split.")
    print(f"Train lesions: {len(y_train)}, feature dim: {X_train.shape[1]}")

    print(f"\nCollecting {eval_split}-split predictions for fusion evaluation...")
    val_inf = collect_lesion_modality_probs(
        eval_split,
        csv_path,
        project_src=PROJECT_SRC,
        tta=tta, validate_inputs=validate_inputs,
    )
    X_val = build_fusion_feature_matrix(
        val_inf.lesion_ids,
        val_inf.clinical_probs,
        val_inf.dermoscopic_probs,
        val_inf.metadata_probs,
    )
    y_val = (
        labels_for_lesions(val_inf.eval_metadata, val_inf.lesion_ids)
        if val_inf.has_gt
        else None
    )

    model = train_fusion_lr(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        checkpoint_path=ckpt,
    )
    import json
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from utils.checkpointing import dataset_hash
    report = dict(fit_split=fit_split, evaluation_split=eval_split, tta=tta,
                  fit_lesions=len(y_train), evaluation_lesions=len(val_inf.lesion_ids),
                  csv_sha256=dataset_hash(csv_path), upload_gate_enabled=validate_inputs)
    from utils.evaluation_protocol import metric_protocol
    report.update(metric_protocol(csv_path, eval_split))
    if y_val is not None:
        predicted = model.predict(X_val)
        report.update(accuracy=float(accuracy_score(y_val, predicted)),
                      balanced_accuracy=float(balanced_accuracy_score(y_val, predicted)),
                      macro_f1=float(f1_score(y_val, predicted, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)))
    ckpt.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train fusion LogisticRegression on concat(clinical, dermoscopic, metadata) "
            "softmax vectors (33 features for 11 classes)."
        )
    )
    parser.add_argument("--fit-split", choices=["train", "fusion"], default="train")
    parser.add_argument("--eval-split", choices=["val", "holdout"], default="val")
    parser.add_argument("--trusted-benchmark", action="store_true", help="Evaluate curated benchmark inputs without the upload gate; never use for uploads")
    parser.add_argument("--csv", type=Path, default=_default_csv())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fusion checkpoint path (default: checkpoints/best_fusion_lr.joblib).",
    )
    parser.add_argument(
        "--tta", dest="tta", action="store_true", default=True,
        help="Use test-time augmentation when collecting modality probs (default ON).",
    )
    parser.add_argument(
        "--no-tta", dest="tta", action="store_false",
        help="Disable test-time augmentation.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    train_fusion(args.csv, checkpoint_path=args.output, tta=args.tta,
                 fit_split=args.fit_split, eval_split=args.eval_split, validate_inputs=not args.trusted_benchmark)


if __name__ == "__main__":
    main()
