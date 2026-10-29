"""Import your own Kaggle training_artifacts.zip into the local app."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.app_inference import CHECKPOINT_DIR
from utils.model_artifacts import install_artifacts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    print(json.dumps(install_artifacts(args.zip, args.checkpoint_dir), indent=2))
