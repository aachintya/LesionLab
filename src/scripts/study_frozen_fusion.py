"""Preregister, cache frozen encoders, select on fusion OOF, confirm on validation.

No holdout evaluation, encoder training, production checkpoint writes, or promotion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold
import torch

from components.augmentations import get_transforms
from components.dataset import MILK10kDataset
from components.image_cache import _VERSION
from components.metadata_encoder import MetadataEncoder
from components.paths import DIAGNOSIS_COLS
from models.efficientnet_v2_image import build_image_model
from models.metadata_mlp import build_metadata_model
from utils.fusion_study import (SEED, candidate_specs, historical_recipe, fit_candidate,
                               predict_candidate, metrics, out_of_fold, paired_bootstrap,
                               class_rows, confusion, validate_probabilities)
from utils.training import pick_device

BASELINE = ROOT / "baselines/baseline-v1-2026-10-27"
CHECKPOINTS = ROOT / "src/checkpoints"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def protected_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in [ROOT / "baselines", CHECKPOINTS]
            for p in folder.rglob("*") if p.is_file()}


def check_protected(before):
    changed = [name for name, value in before.items() if sha(ROOT / name) != value]
    if changed:
        raise RuntimeError(f"A protected model/baseline file changed: {changed}")


def extract_features(frame, output, identity, batch_size=16):
    destination = output / "frozen_probabilities.npz"
    identity_path = output / "feature_identity.json"
    if destination.exists():
        saved = json.loads(identity_path.read_text())
        if saved["identity"] != identity or saved["npz_sha256"] != sha(destination):
            raise ValueError("Cached frozen features have a different identity or checksum.")
        print("Using verified frozen feature cache.", flush=True)
        return
    device = pick_device()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    clinical = frame[frame.image_type == "clinical: close-up"].sort_values("lesion_id").reset_index(drop=True)
    dermoscopic = frame[frame.image_type == "dermoscopic"].sort_values("lesion_id").reset_index(drop=True)
    assert clinical.lesion_id.tolist() == dermoscopic.lesion_id.tolist()
    assert clinical.target.tolist() == dermoscopic.target.tolist()
    arrays = {}
    for name, rows in [("clinical", clinical), ("dermoscopic", dermoscopic)]:
        model = build_image_model(11, dropout=.5, pretrained=False)
        model.load_state_dict(torch.load(CHECKPOINTS / f"best_{name}.pt", map_location="cpu", weights_only=True))
        model.requires_grad_(False).eval().to(device)
        dataset = MILK10kDataset(rows, str(ROOT / "data/raw/MILK10k_Training_Input"),
                                transform=get_transforms(modality=name)[1], roi_mode="foreground", modality=name)
        batches = []
        print(f"Extracting {name}: {len(rows)} lesions on {device}; encoder frozen.", flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool, torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                items = list(pool.map(dataset.__getitem__, range(start, min(len(rows), start+batch_size))))
                inputs = torch.stack([item[0] for item in items]).to(device)
                batches.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
                if start == 0 or (start // batch_size + 1) % 8 == 0 or start+batch_size >= len(rows):
                    print(f"{name}: {min(start+batch_size, len(rows))}/{len(rows)}", flush=True)
        assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        arrays[name] = np.concatenate(batches)
        del model
        if device.type == "mps":torch.mps.empty_cache()
    encoder = MetadataEncoder.load(CHECKPOINTS / "metadata_encoder.json")
    metadata = build_metadata_model(encoder.input_dim_, 11)
    metadata.load_state_dict(torch.load(CHECKPOINTS / "best_metadata.pt", map_location="cpu", weights_only=True))
    metadata.requires_grad_(False).eval()
    # The saved encoder accepts only its predefined age, skin tone, sex and site.
    features = encoder.transform(clinical)
    with torch.inference_mode():
        arrays["metadata"] = torch.softmax(metadata(torch.from_numpy(features)), dim=1).numpy()
    p = validate_probabilities(np.stack([arrays[name] for name in ["clinical", "dermoscopic", "metadata"]], axis=1))
    np.savez_compressed(destination, probabilities=p, labels=clinical.target.to_numpy(dtype=int),
                        lesion_ids=clinical.lesion_id.to_numpy(dtype=str), splits=clinical.split.to_numpy(dtype=str))
    write_json(identity_path, {"identity": identity, "npz_sha256": sha(destination), "device": str(device),
                               "shape": list(p.shape), "encoder_gradients_disabled": True})
    print("Frozen features saved; no training performed on the encoders.", flush=True)


def run(output):
    output = output.resolve()
    for protected in [ROOT / "baselines", ROOT / "src", ROOT / "data"]:
        if output.is_relative_to(protected.resolve()):
            raise ValueError("Experiment output must be outside saved models, baseline, source and data directories.")
    output.mkdir(parents=True, exist_ok=True)
    before_path = output / "protected_files_before.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else protected_hashes()
    check_protected(before)
    if not before_path.exists():write_json(before_path, before)
    csv = BASELINE / "payload/final_metadata_with_split.csv"
    frame = pd.read_csv(csv)
    groups = {split: set(frame[frame.split == split].lesion_id) for split in ["train", "val", "fusion", "holdout"]}
    for a in groups:
        for b in groups:
            if a != b:assert not groups[a] & groups[b], (a,b)
    selected = frame[frame.split.isin(["fusion", "val"])].copy()
    assert set(selected.split) == {"fusion", "val"} and not set(selected.lesion_id) & groups["holdout"]
    assert len(selected) == 2 * selected.lesion_id.nunique()
    assert selected.groupby("lesion_id").target.nunique().eq(1).all()
    assert selected.groupby("lesion_id").image_type.nunique().eq(2).all()
    source_files = [ROOT / name for name in ["src/components/preprocessing.py", "src/components/image_cache.py",
                    "src/components/dataset.py", "src/components/augmentations.py", "src/components/metadata_encoder.py",
                    "src/models/efficientnet_v2_image.py", "src/models/metadata_mlp.py"]]
    image_identity = []
    for row in selected.sort_values(["lesion_id", "image_type"]).itertuples():
        path = ROOT / "data/raw/MILK10k_Training_Input" / row.lesion_id / f"{row.isic_id}.jpg"
        image_identity.append([row.lesion_id, row.image_type, row.isic_id, sha(path)])
    identity = {"split_csv_sha256": sha(csv), "preprocessing_version": _VERSION,
                "source_sha256": {str(p.relative_to(ROOT)):sha(p) for p in source_files},
                "checkpoint_sha256": {name:sha(CHECKPOINTS / name) for name in
                    ["best_clinical.pt", "best_dermoscopic.pt", "best_metadata.pt", "metadata_encoder.json", "best_fusion_lr.joblib"]},
                "image_identity_sha256": hashlib.sha256(json.dumps(image_identity).encode()).hexdigest(),
                "torch": torch.__version__, "numpy":np.__version__, "opencv":cv2.__version__}
    specs = candidate_specs()
    protocol = {"study": "frozen-fusion-study-v1", "selection_seed": SEED,
                "fit_split": "fusion", "confirmation_split": "val", "holdout_used": False,
                "lesions": {name:len(groups[name]) for name in ["fusion", "val"]},
                "class_order": list(DIAGNOSIS_COLS), "tta": False, "upload_gate_enabled": False,
                "selection": "3-fold stratified lesion OOF on fusion split; macro F1, then balanced accuracy, then ID",
                "singleton_policy": "An absent fitting class gets a zero probability column, keeping all 11 class positions and fixed-class metrics.",
                "locked_validation_comparisons": ["selected_alternative", "unchanged_saved_baseline"],
                "candidate_count": len(specs), "candidates": specs, "baseline_recipe":historical_recipe(),
                "criterion": {"positive_fusion_oof_macro_f1_delta":True,
                              "minimum_validation_macro_f1_delta":.02,
                              "validation_paired_95_macro_f1_lower_bound_above_zero":True,
                              "nonnegative_validation_balanced_accuracy_delta":True},
                "bootstrap_draws":5000, "fixed_metric_labels":list(range(11)),
                "limitations": ["Validation previously selected encoder checkpoints; it is not a new independent test.",
                                "Very small rare-class supports and unavailable patient grouping limit certainty.",
                                "Bootstrap intervals condition on these examples and frozen weights, not training-seed or distribution-shift uncertainty."],
                "feature_identity":identity, "sklearn":sklearn.__version__}
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:raise ValueError("Preregistered protocol differs; start a new study instead of changing it.")
    else:write_json(protocol_path, protocol)
    print(f"Protocol recorded: {len(specs)} alternatives; fusion OOF selection, validation confirmation; holdout excluded.", flush=True)
    extract_features(selected, output, identity)
    cache = np.load(output / "frozen_probabilities.npz", allow_pickle=False)
    mask = cache["splits"] == "fusion"
    p, y, ids = cache["probabilities"][mask], cache["labels"][mask], cache["lesion_ids"][mask]
    assert len(set(ids)) == len(ids) == len(groups["fusion"]) and set(y) == set(range(11))
    with warnings.catch_warnings(record=True):
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    fold_record = [{"fold":i, "train_lesions":ids[train].tolist(), "test_lesions":ids[test].tolist(),
                    "absent_fitting_classes": sorted(set(range(11))-set(y[train]))} for i,(train,test) in enumerate(folds)]
    write_json(output / "selection_folds.json", fold_record)
    baseline_oof = out_of_fold(historical_recipe(), p, y, folds)
    baseline_oof_metrics = metrics(y, baseline_oof)
    rankings, failures, oof_predictions = [], [], {}
    for index, spec in enumerate(specs):
        try:
            predicted = out_of_fold(spec, p, y, folds)
            score = metrics(y, predicted)
            rankings.append({"id":spec["id"], **score})
            oof_predictions[spec["id"]] = predicted
            print(f"OOF {index+1}/{len(specs)} {spec['id']}: macro F1={score['macro_f1']:.4f}", flush=True)
        except RuntimeError as exc:
            failures.append({"id":spec["id"], "reason":str(exc)})
    if not rankings:raise RuntimeError("No eligible fusion method completed OOF evaluation.")
    rankings.sort(key=lambda row:(-row["macro_f1"],-row["balanced_accuracy"],row["id"]))
    pd.DataFrame(rankings).to_csv(output / "fusion_oof_leaderboard.csv", index=False)
    best_spec = next(spec for spec in specs if spec["id"] == rankings[0]["id"])
    lock = {"selected_id":best_spec["id"], "selected_spec":best_spec, "selection_source":"fusion OOF only",
            "protocol_sha256":sha(protocol_path), "baseline_oof":baseline_oof_metrics,
            "selected_oof":rankings[0], "failed_candidates":failures}
    lock_path = output / "selection_locked.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:raise ValueError("The locked selection changed; confirmation reuse is forbidden.")
    else:write_json(lock_path, lock)
    fitted = fit_candidate(best_spec, p, y)
    artifact = output / "candidate_fusion.joblib"
    joblib.dump({"format_version":1, **fitted, "class_order":list(DIAGNOSIS_COLS),
                 "protocol_sha256":sha(protocol_path), "encoder_checkpoint_sha256":identity["checkpoint_sha256"]}, artifact)
    print(f"LOCKED {best_spec['id']}. Beginning the single validation comparison.", flush=True)
    # Validation labels enter scoring only after the candidate identity is locked.
    confirm = cache["splits"] == "val"
    val_p, val_y, val_ids = cache["probabilities"][confirm], cache["labels"][confirm], cache["lesion_ids"][confirm]
    assert not set(ids) & set(val_ids) and len(val_ids) == len(groups["val"])
    baseline = joblib.load(CHECKPOINTS / "best_fusion_lr.joblib")
    assert list(baseline.classes_) == list(range(11)) and baseline.n_features_in_ == 33
    baseline_prob = baseline.predict_proba(val_p.reshape(len(val_p),33))
    candidate_prob = predict_candidate(fitted, val_p)
    comparison = paired_bootstrap(val_y, baseline_prob, candidate_prob)
    passed = (rankings[0]["macro_f1"] > baseline_oof_metrics["macro_f1"]
              and comparison["macro_f1"]["difference"] >= .02
              and comparison["macro_f1"]["lower_95"] > 0
              and comparison["balanced_accuracy"]["difference"] >= 0)
    named = {"saved_fusion":baseline_prob, "selected_fusion":candidate_prob,
             "clinical":val_p[:,0], "dermoscopic":val_p[:,1], "metadata":val_p[:,2]}
    all_metrics = {name:metrics(val_y, values) for name,values in named.items()}
    all_classes = {name:class_rows(val_y, values.argmax(1), DIAGNOSIS_COLS) for name,values in named.items()}
    for name, values in named.items():
        np.save(output / f"{name}_validation_confusion.npy", confusion(val_y, values.argmax(1)), allow_pickle=False)
        pd.DataFrame(all_classes[name]).to_csv(output / f"{name}_validation_classes.csv", index=False)
    pd.DataFrame({"lesion_id":val_ids, "target":val_y, **{name:values.argmax(1) for name,values in named.items()}}).to_csv(output / "validation_predictions.csv", index=False)
    np.savez_compressed(output / "validation_scores.npz", lesion_ids=val_ids, targets=val_y, **named)
    image_any_correct = (val_p[:,:2].argmax(2) == val_y[:,None]).any(1)
    metadata_correct = val_p[:,2].argmax(1) == val_y
    fusion_wrong = baseline_prob.argmax(1) != val_y
    conflicts = {"at_least_one_image_correct_but_saved_fusion_wrong":int((image_any_correct & fusion_wrong).sum()),
                 "both_images_wrong":int((~image_any_correct).sum()),
                 "metadata_correct_but_saved_fusion_wrong":int((metadata_correct & fusion_wrong).sum()),
                 "image_models_disagree":int((val_p[:,0].argmax(1) != val_p[:,1].argmax(1)).sum())}
    check_protected(before)
    result = {"selected_id":best_spec["id"], "protocol_sha256":sha(protocol_path),
              "candidate_sha256":sha(artifact), "validation_lesions":len(val_y), "fusion_lesions":len(y),
              "baseline_oof":baseline_oof_metrics, "selected_oof":rankings[0],
              "validation_metrics":all_metrics, "paired_validation_differences":comparison,
              "validation_class_details":all_classes, "branch_diagnostics":conflicts,
              "passes_preregistered_promising_criterion":bool(passed), "protected_files_unchanged":len(before),
              "promoted_to_demo":False, "holdout_evaluated":False, "calibrated":False,
              "limitations":protocol["limitations"], "failed_candidates":failures}
    write_json(output / "results.json", result)
    print(json.dumps({key:result[key] for key in ["selected_id","validation_metrics","paired_validation_differences",
                        "passes_preregistered_promising_criterion","branch_diagnostics","protected_files_unchanged"]}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/frozen-fusion-study-v1")
    args = parser.parse_args()
    run(args.output)
