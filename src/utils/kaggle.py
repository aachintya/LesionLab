"""Kaggle filesystem setup and artifact export; no GPU/account dependencies."""
import json
from pathlib import Path
import shutil
import stat
import zipfile


def safe_extract(archive_path, destination, *, keep_weights=False):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
        for member in archive.infolist():
            target = destination / member.filename
            if keep_weights and target.is_file() and target.suffix in {".pt", ".joblib"}:
                continue
            archive.extract(member, destination)


def link_raw_inputs(input_root, raw_dir, overrides=None):
    input_root, raw_dir = Path(input_root), Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    overrides = overrides or {}
    required = {"MILK10k_Training_Metadata.csv", "MILK10k_Training_GroundTruth.csv", "MILK10k_Training_Input"}
    names = sorted(required | {"MILK10k_Training_Supplement.csv", "MILK10k_Test_Metadata.csv", "MILK10k_Test_Input"})
    linked = {}
    for name in names:
        matches = [Path(overrides[name])] if name in overrides else sorted(input_root.rglob(name))
        matches = [p for p in matches if p.exists() and (p.is_dir() if name.endswith("_Input") else p.is_file())]
        # Deduplicate aliases, but require a choice between different releases.
        matches = list({p.resolve(): p for p in matches}.values())
        if not matches:
            if name in required:
                raise FileNotFoundError(f"Attach the official MILK10k files as a Kaggle input. Missing: {name}. Unzip image archives first.")
            continue
        if len(matches) != 1:
            raise ValueError(f"Multiple matches for {name}: {matches}. Set RAW_OVERRIDES in the notebook.")
        source, target = matches[0].resolve(), raw_dir / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"Existing raw-data link differs: {target}. Use a new work directory for a different dataset.")
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())
        linked[name] = str(source)
    return linked


def export_artifacts(project_root, output_path):
    project_root, output_path = Path(project_root), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in ("src/checkpoints", "data/processed", "logs"):
        directory = project_root / relative
        if directory.exists():
            files.extend(p for p in directory.rglob("*") if p.is_file()
                         and not any(part.startswith(".") for part in p.relative_to(directory).parts)
                         and p.suffix in {".pt", ".json", ".csv", ".joblib", ".log"})
    temporary = output_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(project_root))
    temporary.replace(output_path)
    print(f"Saved {len(files)} artifacts to {output_path}", flush=True)
    return output_path


def restore_artifacts(source, project_root):
    source, project_root = Path(source), Path(project_root)
    if any((project_root / "src/checkpoints").glob("*_resume.pt")):
        print("Existing local resume state found; not overwriting it from an older output.")
        return
    if source.is_file():
        safe_extract(source, project_root)
    else:
        for name in ("src/checkpoints", "data/processed", "logs"):
            if (source / name).exists():
                shutil.copytree(source / name, project_root / name, dirs_exist_ok=True)


def verify_split(csv_path):
    import pandas as pd
    from components.paths import NUM_CLASSES
    frame = pd.read_csv(csv_path)
    expected = {"train", "val", "fusion", "holdout"}
    if set(frame.split) != expected:
        raise ValueError("Kaggle workflow requires train/val/fusion/holdout splits; restore the matching processed CSV or regenerate splits.")
    if (frame.groupby("lesion_id").split.nunique() > 1).any():
        raise ValueError("A lesion occurs in multiple splits")
    for column in ("patient_id", "case_id", "subject_id", "patient"):
        if column in frame and frame[column].notna().any():
            valid = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
            if (frame[valid].groupby(column).split.nunique() > 1).any():
                raise ValueError(f"{column} occurs in multiple splits")
            break
    table = pd.crosstab(frame.drop_duplicates("lesion_id").split, frame.drop_duplicates("lesion_id").target)
    if set(table.columns) != set(range(NUM_CLASSES)) or (table == 0).any().any():
        raise ValueError(f"Each split needs all 11 classes. Adjust fractions or SPLIT_SEED before training.\n{table}")
    print("Lesion counts by split/class:\n", table)
    return table
