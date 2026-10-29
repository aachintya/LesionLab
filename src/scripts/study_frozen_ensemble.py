"""Final fixed ensemble check; no validation-driven weights or encoder training."""
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from components.paths import DIAGNOSIS_COLS
from scripts.study_frozen_fusion import sha, write_json, protected_hashes, check_protected
from scripts.study_frozen_features import intervals
from utils import frozen_ensemble as ensemble, feature_study as feature, frozen_neural_head as neural
from utils.fusion_study import metrics, class_rows, historical_recipe, out_of_fold, predict_candidate

SOURCE = ROOT / "experiments/frozen-feature-study-v2"
NEURAL = ROOT / "experiments/frozen-neural-study-v3"
FIRST = ROOT / "experiments/frozen-fusion-study-v1"
OUTPUT = ROOT / "experiments/frozen-ensemble-study-v4"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before_path = OUTPUT / "protected_files_before.json"
    if before_path.exists():
        before = json.loads(before_path.read_text())
        check_protected(before)
    else:
        before = protected_hashes()
        for folder in [FIRST, SOURCE, NEURAL]:
            for path in folder.iterdir():
                if path.is_file() and path.suffix in {".npz", ".joblib", ".pt", ".json"}:
                    before[str(path.relative_to(ROOT))] = sha(path)
        write_json(before_path, before)
    protocol = {
        "study": "frozen-ensemble-study-v4", "seed": neural.SEED,
        "members": list(ensemble.MEMBERS), "candidates": ensemble.specs(),
        "candidate_count": len(ensemble.specs()),
        "selection": "Three-fold fusion OOF; fixed equal weights; macro F1, balanced accuracy, then ID.",
        "member_selection": "Members come from earlier exploratory rounds; their selection is not nested in these folds.",
        "validation_previously_inspected": True, "holdout_used": False, "encoders_frozen": True,
        "feature_cache_sha256": sha(SOURCE / "frozen_embeddings.npz"),
        "profiles": ["best_macro", "class_stability"], "maximum_NV_SCCKA_oof_recall_drop": .03,
        "large_gain_target": {"macro_f1_delta": .10, "no_accuracy_drop": True, "maximum_NV_SCCKA_recall_drop": .03},
        "uncertainty": "Exploratory conditional paired bootstrap, 5000 draws, 97.5% per profile; no search-wide significance claim."
    }
    path = OUTPUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Ensemble protocol changed.")
    else:
        write_json(path, protocol)
    print(f"Recorded {len(ensemble.specs())} fixed equal-weight ensembles; holdout excluded.", flush=True)
    data = np.load(SOURCE / "frozen_embeddings.npz", allow_pickle=False)
    train = np.flatnonzero(data["splits"] == "train")
    fusion = np.flatnonzero(data["splits"] == "fusion")
    val = np.flatnonzero(data["splits"] == "val")
    y = data["labels"]
    reducers = joblib.load(SOURCE / "representations.joblib")["reducers"]
    reduced = [reducers[name].transform(data["embeddings"][:, i]) for i, name in enumerate(["clinical", "dermoscopic"])]
    x = {f"pca{k}": np.c_[reduced[0][:, :k], reduced[1][:, :k], data["metadata"]] for k in [64, 128]}
    x["pca64_plus_scores"] = np.c_[x["pca64"], data["probabilities"].reshape(len(y), 33)]
    with warnings.catch_warnings(record=True):
        folds = list(StratifiedKFold(3, shuffle=True, random_state=neural.SEED).split(np.zeros(len(fusion)), y[fusion]))
    np.savez_compressed(OUTPUT / "selection_folds.npz", lesion_ids=data["lesion_ids"][fusion],
                        fold=np.asarray([next(i for i, (_, test) in enumerate(folds) if row in test) for row in range(len(fusion))]))
    score_spec = joblib.load(FIRST / "candidate_fusion.joblib")["spec"]
    lr_spec = next(spec for spec in feature.candidate_specs() if spec["id"] == "pca64_plus_scores_lr_balanced_C0.1")
    mlp_spec = next(spec for spec in neural.specs() if spec["id"] == "pca128_mlp128_sqrt")
    oof = {
        "scaled_scores": out_of_fold(score_spec, data["probabilities"][fusion], y[fusion], folds),
        "feature_lr": feature.oof(lr_spec, x, y, train, fusion, folds),
        "feature_mlp": np.load(NEURAL / f"oof_{mlp_spec['id']}.npz", allow_pickle=False)["probabilities"],
        "tta_images": data["tta_probabilities"][fusion, :2].mean(1),
    }
    if json.loads((NEURAL / "protocol.json").read_text())["seed"] != neural.SEED:
        raise ValueError("Neural cached OOF folds differ.")
    base_oof = out_of_fold(historical_recipe(), data["probabilities"][fusion], y[fusion], folds)
    base_details = class_rows(y[fusion], base_oof.argmax(1), DIAGNOSIS_COLS)
    rankings = []
    for spec in ensemble.specs():
        p = ensemble.combine(spec, oof)
        detail = class_rows(y[fusion], p.argmax(1), DIAGNOSIS_COLS)
        stable = all(detail[k]["recall"] >= base_details[k]["recall"] - .03 for k in [8, 9])
        rankings.append({"id": spec["id"], **metrics(y[fusion], p), "stable": stable,
                         "nv_recall": detail[8]["recall"], "sccka_recall": detail[9]["recall"]})
    rankings.sort(key=lambda row: (-row["macro_f1"], -row["balanced_accuracy"], row["id"]))
    pd.DataFrame(rankings).to_csv(OUTPUT / "fusion_oof_leaderboard.csv", index=False)
    stable = [row for row in rankings if row["stable"]]
    profiles = {"best_macro": rankings[0]["id"], "class_stability": stable[0]["id"] if stable else None}
    locked = {"protocol_sha256": sha(path), "profiles": profiles, "selection_data": "fusion OOF only",
              "baseline_oof": metrics(y[fusion], base_oof), "baseline_oof_class_details": base_details}
    lock_path = OUTPUT / "selection_locked.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != locked:
            raise ValueError("Locked choices differ.")
    else:
        write_json(lock_path, locked)
    print("LOCKED " + json.dumps(profiles), flush=True)
    baseline_data = np.load(SOURCE / "validation_scores.npz", allow_pickle=False)
    if not np.array_equal(baseline_data["lesion_ids"], data["lesion_ids"][val]):
        raise ValueError("Validation row identity differs.")
    baseline = baseline_data["baseline"]
    # Reuse unchanged full-fit member artifacts. Only ensemble arithmetic is new.
    lr_payload = joblib.load(SOURCE / "candidate_best_macro.joblib")
    mlp_payload = torch.load(NEURAL / "candidate_best_macro.pt", map_location="cpu", weights_only=False)
    if lr_payload["spec"] != lr_spec or mlp_payload["spec"] != mlp_spec:
        raise ValueError("Full-fit member identity differs.")
    model = neural.build(mlp_spec, mlp_payload["dimension"])
    model.load_state_dict(mlp_payload["state_dict"])
    val_members = {
        "scaled_scores": predict_candidate(joblib.load(FIRST / "candidate_fusion.joblib"), data["probabilities"][val]),
        "feature_lr": feature.predict(lr_spec, lr_payload["estimator"], x[lr_spec["feature"]][val]),
        "feature_mlp": neural.predict(model, joblib.load(NEURAL / "candidate_best_macro_scaler.joblib"), x[mlp_spec["feature"]][val]),
        "tta_images": data["tta_probabilities"][val, :2].mean(1),
    }
    base_details = class_rows(y[val], baseline.argmax(1), DIAGNOSIS_COLS)
    base_metrics = metrics(y[val], baseline)
    results = {}; scores = {"baseline": baseline}; seen = {}
    for profile, name in profiles.items():
        if name is None:
            continue
        if name in seen:
            results[profile] = {"id": name, "same_as": seen[name]}
            continue
        spec = next(item for item in ensemble.specs() if item["id"] == name)
        p = ensemble.combine(spec, val_members)
        values = metrics(y[val], p); detail = class_rows(y[val], p.argmax(1), DIAGNOSIS_COLS)
        confidence = intervals(y[val], baseline, p)
        large = (values["macro_f1"] - base_metrics["macro_f1"] >= .10
                 and confidence["macro_f1"]["lower_97_5"] > 0 and values["accuracy"] >= base_metrics["accuracy"]
                 and all(detail[k]["recall"] >= base_details[k]["recall"] - .03 for k in [8, 9]))
        results[profile] = {"id": name, "spec": spec, "validation_metrics": values, "class_details": detail,
                            "paired_differences": confidence, "large_gain_target_passed": bool(large)}
        write_json(OUTPUT / f"candidate_{profile}.json", {"spec": spec, "members": {
            "scaled_scores": str(FIRST.relative_to(ROOT) / "candidate_fusion.joblib"),
            "feature_lr": str(SOURCE.relative_to(ROOT) / "candidate_best_macro.joblib"),
            "feature_mlp": str(NEURAL.relative_to(ROOT) / "candidate_best_macro.pt"),
            "tta_images": "Equal clinical/dermoscopic mean over the same six views; metadata omitted for this member."},
            "class_order": list(DIAGNOSIS_COLS), "protocol_sha256": sha(path), "promoted_to_demo": False})
        scores[profile] = p; seen[name] = profile
        print(json.dumps({"profile": profile, "metrics": values, "large_gain_target": bool(large)}, indent=2), flush=True)
    check_protected(before)
    write_json(OUTPUT / "results.json", {"profiles": results, "baseline_validation": base_metrics,
               "baseline_class_details": base_details, "validation_lesions": len(val), "holdout_evaluated": False,
               "validation_is_exploratory": True, "promoted_to_demo": False, "protected_files_unchanged": len(before)})
    np.savez_compressed(OUTPUT / "validation_scores.npz", targets=y[val], lesion_ids=data["lesion_ids"][val], **scores)
    print("Fixed ensemble check completed; protected artifacts unchanged.", flush=True)


if __name__ == "__main__":
    main()
