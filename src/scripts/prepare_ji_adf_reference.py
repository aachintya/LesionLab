"""Restore an upload-safe source archive into a fully verified writable reference.

This file uses only the standard library so the small repair dataset can also
repair notebooks using the original JI-ADF package. Neither the input dataset nor
the manifest is modified. The .bin payload is the original ZIP's exact bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError(f"Unsafe reference manifest path: {name}")
    return path


def verify_reference(reference):
    reference = Path(reference).resolve()
    manifest = json.loads((reference / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = _member(reference, name)
        if not path.is_file():
            kind = "directory instead of a file" if path.is_dir() else "missing file"
            raise ValueError(f"Frozen reference failed integrity check: {name} ({kind})")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen reference failed integrity check: {name} "
                             f"(expected SHA256 {expected}, got {actual})")
    return manifest


def prepare_reference(source, destination, *, snapshot=None):
    """Copy verified reference members, restoring only a missing source ZIP.

    An existing but corrupted ZIP is rejected. The optional replacement must
    match the existing manifest; checksums are never rewritten. The destination
    is published only after every copied file passes verification.
    """
    source, destination = Path(source).resolve(), Path(destination).absolute()
    resolved_destination = destination.resolve()
    if (resolved_destination.is_relative_to(source) or source.is_relative_to(resolved_destination)
            or destination.is_symlink()):
        raise ValueError("The prepared reference must be separate from the input reference")
    if any((parent / "SHA256SUMS").exists() for parent in (resolved_destination, *resolved_destination.parents)):
        raise ValueError("Cannot prepare a reference inside a frozen baseline")
    manifest_path = source / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    members = {}
    for name, expected in manifest["sha256"].items():
        path = _member(source, name)
        if name == "source_snapshot.zip" and not path.is_file():
            path = Path(snapshot).resolve() if snapshot is not None else source / "source_snapshot.zip.bin"
        if not path.is_file():
            raise FileNotFoundError(f"Reference member {name} is missing. Attach the JI-ADF reference repair "
                                    "dataset with source_snapshot.zip.bin, or use the corrected code upload.")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen reference failed integrity check: {name} "
                             f"(expected SHA256 {expected}, got {actual}); no files were changed")
        members[name] = path
    if destination.exists():
        if not destination.is_dir() or (destination / "manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("Prepared reference belongs to another upload; choose a new destination")
        verify_reference(destination)
        print(f"Reference verified: {len(members)} files; reusing {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name, path in members.items():
            target = _member(staging, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (staging / "manifest.json").write_bytes(manifest_bytes)
        verify_reference(staging)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Reference verified: {len(members)} files; ready at {destination}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    prepare_reference(args.source, args.destination, snapshot=args.snapshot)
