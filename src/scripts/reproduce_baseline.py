"""Run the actual archived v1 source/weights in a separate subprocess/runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile


def reproduce(reference, images, output):
    reference, images, output = Path(reference).resolve(), Path(images).resolve(), Path(output).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from improved.runtime import output_directory, verify_reference
    verify_reference(reference)
    output = output_directory(output, reference)
    runtime = output / "baseline_runtime"
    runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(reference / "source_snapshot.zip") as archive:
        for info in archive.infolist():
            target = (runtime / info.filename).resolve()
            if not target.is_relative_to(runtime) or info.is_dir():
                raise ValueError("Unexpected baseline source archive entry")
        archive.extractall(runtime)
    checkpoints = runtime / "src/checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for path in (reference / "checkpoints").iterdir():
        shutil.copy2(path, checkpoints / path.name)
    # This is a separate process: load archived source without any live-module cache.
    worker = runtime / "evaluate_frozen.py"
    worker.write_text('''import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import numpy as np
import pandas as pd
import torch
from utils.eval_inference import collect_lesion_modality_probs
from models.fusion_lr import build_fusion_feature_matrix, labels_for_lesions, load_fusion_lr
torch.set_num_threads(2)
reference, output = map(Path, sys.argv[1:3])
result = collect_lesion_modality_probs('holdout', reference / 'split.csv',
    project_src=Path(__file__).parent / 'src', tta=False, validate_inputs=False)
features = build_fusion_feature_matrix(result.lesion_ids, result.clinical_probs, result.dermoscopic_probs, result.metadata_probs)
model = load_fusion_lr(reference / 'checkpoints/best_fusion_lr.joblib')
prob = model.predict_proba(features)
labels = labels_for_lesions(result.eval_metadata, result.lesion_ids)
pd.DataFrame({'lesion_id': result.lesion_ids, 'target': labels, 'predicted': prob.argmax(1)}).to_csv(output / 'baseline_holdout_predictions.csv', index=False)
''')
    import subprocess
    env = os.environ.copy()
    # components.paths expects the official directory name under RAW_DIR.
    raw = runtime / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    link = raw / "MILK10k_Training_Input"
    if link.is_symlink() and link.resolve() != images:
        raise ValueError("Baseline runtime points to different images")
    if not link.exists():
        link.symlink_to(images, target_is_directory=True)
    env["MILK10K_RAW_DIR"] = str(raw)
    env.pop("MILK10K_CACHE_DIR", None)  # v2 RGB caches have a different preprocessing contract.
    print("Reproducing v1 with its archived code. Its original preprocessing can take several minutes.", flush=True)
    subprocess.run([sys.executable, str(worker), str(reference), str(output)], env=env, check=True)
    import pandas as pd
    from improved.training import metrics
    predictions = pd.read_csv(output / "baseline_holdout_predictions.csv")
    observed = metrics(predictions.target.values, predictions.predicted.values)
    expected = json.loads((reference / "metrics.json").read_text())
    if any(abs(observed[name] - expected[name]) > 1e-10 for name in observed):
        raise ValueError(f"Reproduced baseline differs from saved metrics: {observed}; paired comparison stopped")
    print("Archived baseline reproduced exactly.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reproduce(args.reference, args.images, args.output)
