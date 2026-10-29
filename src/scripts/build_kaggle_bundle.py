"""Package source, notebook, and the trained input validator for private Kaggle upload."""
import argparse
from pathlib import Path
import zipfile


def build_bundle(root, output):
    root, output = Path(root), Path(output)
    files = list((root / "src").rglob("*.py")) + list((root / "tests").glob("*.py"))
    files += [root / "README.md", root / "notebooks/kaggle_training.ipynb"]
    for name in ("input_validator.pt", "input_validator.metrics.json"):
        path = root / "src/checkpoints" / name
        if path.exists():
            files.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root))
    print(f"Built {output} ({output.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "dist/skin_cancer_kaggle.zip")
    args = parser.parse_args()
    build_bundle(Path(__file__).resolve().parents[2], args.output)
