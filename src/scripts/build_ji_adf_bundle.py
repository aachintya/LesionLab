"""Create an offline-ready JI-ADF Kaggle upload without modifying v1/v2 bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGENET_SHA256 = "dd5fe13b1d60ec15317ccc8ca158186e134d3366c3dde9cb9a4e301f2dc66c74"
IMAGENET_URL = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"


def build(root, baseline, pretrained, output):
    from utils.baselines import sha256_file, verify_baseline
    from utils.model_artifacts import REQUIRED_FILES
    root, baseline, pretrained, output = map(Path, (root, baseline, pretrained, output))
    verify_baseline(baseline)
    if sha256_file(pretrained) != IMAGENET_SHA256:
        raise ValueError("The ImageNet initialization file has an unexpected checksum")
    members = {}
    with zipfile.ZipFile(baseline / "payload/training_artifacts.zip") as original:
        for name in REQUIRED_FILES:
            members["checkpoints/" + name] = original.read("src/checkpoints/" + name)
    for name, source in {"split.csv": "payload/final_metadata_with_split.csv", "metrics.json": "metrics.json",
                         "source_snapshot.zip": "payload/source_snapshot.zip"}.items():
        members[name] = (baseline / source).read_bytes()
    members["imagenet_efficientnet_v2_s.pt"] = pretrained.read_bytes()
    old = json.loads((baseline / "manifest.json").read_text())
    manifest = {"format_version": 3, "historical_baseline": baseline.name,
                "baseline_manifest_sha256": sha256_file(baseline / "manifest.json"),
                "raw_release_sha256": old["raw_release_sha256"], "imagenet_source": IMAGENET_URL,
                "sha256": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}}
    members["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    files = [*root.glob("src/**/*.py"), *root.glob("tests/test_ji_adf*.py"), root / "README.md", root / "notebooks/kaggle_ji_adf.ipynb"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root))
        for name, data in members.items():
            # Kaggle unpacks nested .zip inputs. Preserve the original ZIP bytes
            # under an opaque suffix; setup restores its manifest name locally.
            uploaded_name = name + ".bin" if name == "source_snapshot.zip" else name
            archive.writestr("reference/" + uploaded_name, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError("JI-ADF package failed its ZIP integrity check")
    temporary.replace(output)
    print(f"Built {output} ({output.stat().st_size/1e6:.1f} MB); SHA256 {sha256_file(output)}")
    return output


def build_repair(root, baseline, output):
    """Small additive upload for notebooks that already attached the old bundle."""
    from utils.baselines import sha256_file, verify_baseline
    root, baseline, output = map(Path, (root, baseline, output))
    verify_baseline(baseline)
    notebook = json.loads((root / "notebooks/kaggle_ji_adf.ipynb").read_text())
    setup_source = "".join(notebook["cells"][5]["source"])
    compile(setup_source, "ji_adf_setup_cell.py", "exec")
    cell_path = root / "dist/ji_adf_setup_cell.py"
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    cell_path.write_text(setup_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(root / "src/scripts/prepare_ji_adf_reference.py", "repair_ji_adf_reference.py")
        archive.write(baseline / "payload/source_snapshot.zip", "source_snapshot.zip.bin")
        archive.writestr("README.txt", "Attach this as an additional private Kaggle dataset.\n"
                         "Keep your existing code and MILK10k datasets attached.\n"
                         "Replace the failed setup cell with ji_adf_setup_cell.py from this upload.\n"
                         "It restores the original source archive into a separate writable reference\n"
                         "and verifies every member against the unchanged baseline manifest.\n")
        archive.writestr("ji_adf_setup_cell.py", setup_source)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():
            raise ValueError("JI-ADF repair package failed its ZIP integrity check")
    temporary.replace(output)
    print(f"Built {output} ({output.stat().st_size/1e6:.1f} MB); SHA256 {sha256_file(output)}")
    return output


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=root / "baselines/baseline-v1-2026-10-27")
    p.add_argument("--pretrained", type=Path, default=root / "dist/ji_adf_pretrained/efficientnet_v2_s-dd5fe13b.pth")
    p.add_argument("--output", type=Path, default=root / "dist/skin_cancer_ji_adf.zip")
    p.add_argument("--repair-only", action="store_true", help="Build only the small reference-repair upload")
    args = p.parse_args()
    if args.repair_only:
        output = args.output if "--output" in sys.argv else root / "dist/ji_adf_reference_fix.zip"
        build_repair(root, args.baseline, output)
    else:
        build(root, args.baseline, args.pretrained, args.output)
