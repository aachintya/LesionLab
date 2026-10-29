"""Download a reproducible research starter dataset for supported-image validation.

Sources: PAD-UFES-20 clinical photographs, HAM10000 dermoscopy, Caltech101
objects, COIL100 household objects, and Enrico application screenshots.
Images stay local and are not committed. See docs/input_validator.md for terms.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import zipfile

import pandas as pd
from PIL import Image, ImageOps
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PAD_REPO = "SalmaneExploring/pad-ufes-20"
PAD_REV = "086dd94836d7b2094132a21406a75dcb6bf7ec2c"
HAM_META = "https://huggingface.co/datasets/jarvisit/ham10000-skin-lesions/resolve/5004e634101c4773c2d59c83939fa57eb0ab1f76/HAM10000_metadata.csv"
SOURCES = {
    "pad": {"url": "https://data.mendeley.com/datasets/zr7vgbcyr2/1", "terms": "CC BY 4.0", "citation": "Pacheco et al., PAD-UFES-20 (2020), doi:10.17632/zr7vgbcyr2.1"},
    "ham": {"url": "https://doi.org/10.7910/DVN/DBW86T", "terms": "CC BY-NC 4.0; research/noncommercial", "citation": "Tschandl, Rosendahl & Kittler, HAM10000 (2018), doi:10.1038/sdata.2018.161"},
    "caltech": {"url": "https://data.caltech.edu/records/mzrjq-6wc02", "terms": "Public research dataset; no explicit image license on landing page; verify before redistribution/commercial use", "citation": "Fei-Fei et al., Caltech101, doi:10.22002/D1.20086"},
    "coil": {"url": "https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php", "terms": "Public research dataset; no explicit image license on landing page; verify before redistribution/commercial use", "citation": "Nene, Nayar & Murase, COIL100, CUCS-006-96 (1996)"},
    "enrico": {"url": "https://github.com/luileito/enrico", "terms": "Repository MIT; screenshots inherit third-party/Rico copyright terms: https://www.interactionmining.org/legal/copyright", "citation": "Leiva, Hota & Oulasvirta, Enrico, MobileHCI (2020)"},
}


def fetch(url, path):
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    temporary = path.with_suffix(path.suffix + ".partial")
    with session.get(url, timeout=(15, 90), stream=True) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 256):
                handle.write(chunk)
    temporary.replace(path)
    session.close()
    return path


def split_for(group):
    # Stable grouping; each source has independent groups in all three splits.
    bucket = int(hashlib.sha256(("validator-v1:" + group).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data/validator")
    parser.add_argument("--lesions-per-source", type=int, default=600)
    args = parser.parse_args()
    root = args.output.resolve()
    cache = root / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    rows, errors = [], []

    def save_image(raw, source, identity, label, group, category, url):
        destination = root / "images" / source / f"{identity}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(io.BytesIO(raw)) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            if min(image.size) < 64:
                raise ValueError("Image below 64-pixel minimum")
            image.thumbnail((512, 512))
            # Normalize formats across sources to reduce encoding shortcuts.
            image.save(destination, quality=92)
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            gray = image.convert("L").resize((9, 8))
            pixels = list(gray.getdata())
            dhash = sum(int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                        for y in range(8) for x in range(8))
        return {"path": str(destination.relative_to(root)), "label": label,
                "split": split_for(group), "group_id": group, "source": source,
                "category": category, "source_url": url,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "pixel_hash": pixel_hash,
                "dhash": f"{dhash:016x}"}

    print("Preparing public-source metadata...", flush=True)
    api_path = fetch(f"https://huggingface.co/api/datasets/{PAD_REPO}/revision/{PAD_REV}", cache / "pad_repository.json")
    files = {Path(x["rfilename"]).name: x["rfilename"] for x in json.loads(api_path.read_text())["siblings"]}
    pad = pd.read_csv(fetch(f"https://huggingface.co/datasets/{PAD_REPO}/resolve/{PAD_REV}/metadata.csv", cache / "pad_metadata.csv"))
    ham = pd.read_csv(fetch(HAM_META, cache / "ham_metadata.csv"))
    tasks = []
    # One photograph per patient/lesion, sampled across diagnoses.
    for source, frame, group_col, id_col, diagnosis in [
        ("pad", pad, "patient_id", "img_id", "diagnostic"),
        ("ham", ham, "lesion_id", "image_id", "dx"),
    ]:
        candidates = frame.sample(frac=1, random_state=42).drop_duplicates(group_col)
        # Round-robin diagnosis categories so the starter set is not mostly nevi.
        pools = [list(g.index) for _, g in candidates.groupby(diagnosis)]
        chosen = []
        while len(chosen) < args.lesions_per_source and any(pools):
            for pool in pools:
                if pool and len(chosen) < args.lesions_per_source:
                    chosen.append(pool.pop())
        for index in chosen:
            row = frame.loc[index]
            identity = str(row[id_col])
            if source == "pad":
                url = f"https://huggingface.co/datasets/{PAD_REPO}/resolve/{PAD_REV}/{files[identity]}"
            else:
                url = f"https://isic-archive.s3.amazonaws.com/images/{identity}.jpg"
            tasks.append((source, identity, f"{source}:{row[group_col]}", str(row[diagnosis]), url))

    def get_positive(task):
        source, identity, group, category, url = task
        raw = fetch(url, cache / source / identity).read_bytes()
        return save_image(raw, source, Path(identity).stem, 1, group, category, url)

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(get_positive, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append({"task": futures[future], "error": str(exc)})
            if i % 100 == 0:
                print(f"Lesion downloads: {i}/{len(tasks)} ({len(errors)} failures)", flush=True)

    print("Downloading object and screenshot archives...", flush=True)
    archives = {
        "caltech": ("https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1", cache / "caltech.zip"),
        "coil": ("https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip", cache / "coil.zip"),
        "enrico": ("https://userinterfaces.aalto.fi/enrico/resources/screenshots.zip", cache / "enrico.zip"),
        "enrico_metadata": ("https://userinterfaces.aalto.fi/enrico/resources/metadata.zip", cache / "enrico_metadata.zip"),
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, *params): source for source, params in archives.items()}
        for future in as_completed(futures):
            source = futures[future]
            print(f"Downloaded {source}: {future.result().stat().st_size / 1e6:.1f} MB", flush=True)

    with zipfile.ZipFile(cache / "caltech.zip") as archive:
        tar_name = next(n for n in archive.namelist() if n.endswith("101_ObjectCategories.tar.gz"))
        with tarfile.open(fileobj=io.BytesIO(archive.read(tar_name)), mode="r:gz") as tar:
            groups = {}
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".jpg") and not Path(member.name).name.startswith("."):
                    groups.setdefault(Path(member.name).parent.name, []).append(member)
            for category, members in sorted(groups.items()):
                rng.shuffle(members)
                for member in members[:12]:
                    identity = f"{category}_{Path(member.name).stem}"
                    try:
                        rows.append(save_image(tar.extractfile(member).read(), "caltech", identity, 0, f"caltech:{category}", category, archives["caltech"][0]))
                    except ValueError:
                        pass

    with zipfile.ZipFile(cache / "coil.zip") as archive:
        for name in sorted(archive.namelist()):
            filename = Path(name).name
            if not filename.endswith(".png") or not filename.startswith("obj"):
                continue
            identity = Path(filename).stem
            object_id, angle = identity.split("__")
            if int(angle) % 90 == 0:  # Four views per physical object; group all views.
                rows.append(save_image(archive.read(name), "coil", identity, 0, f"coil:{object_id}", "household_object", archives["coil"][0]))

    with zipfile.ZipFile(cache / "enrico_metadata.zip") as archive:
        app_ids = {}
        for name in archive.namelist():
            if name.endswith(".json") and not name.startswith("__MACOSX"):
                data = json.loads(archive.read(name))
                if isinstance(data, dict) and data:
                    app = data.get("appId") or data.get("app_id") or data.get("title")
                    if app:
                        app_ids[Path(name).stem.split("-")[0]] = str(app)
    with zipfile.ZipFile(cache / "enrico.zip") as archive:
        members = [n for n in archive.namelist() if n.endswith(".jpg") and not n.startswith("__MACOSX")]
        rng.shuffle(members)
        seen_apps = set()
        for name in members:
            identity = Path(name).stem.split("-")[0]
            app = app_ids.get(identity)
            if app is None or app in seen_apps:
                continue
            seen_apps.add(app)
            rows.append(save_image(archive.read(name), "enrico", identity, 0, f"enrico:{app}", "screenshot", archives["enrico"][0]))
            if len(seen_apps) >= 400:
                break
    if not any(r["source"] == "enrico" for r in rows):
        raise ValueError("No screenshot application identities resolved; inspect Enrico metadata format")

    manifest = pd.DataFrame(rows).sort_values(["source", "path"]).drop_duplicates("pixel_hash")
    # Conservatively merge near-duplicate image groups before assigning splits.
    parents = {g: g for g in manifest.group_id}
    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    hashed = [(int(r.dhash, 16), r.group_id) for r in manifest.itertuples()]
    merges = 0
    for index, (value, group) in enumerate(hashed):
        for previous, other in hashed[:index]:
            if (value ^ previous).bit_count() <= 4:
                a, b = find(group), find(other)
                if a != b:
                    parents[max(a, b)] = min(a, b)
                    merges += 1
    manifest["original_group_id"] = manifest.group_id
    manifest["group_id"] = manifest.group_id.map(find)
    manifest["split"] = manifest.group_id.map(split_for)
    manifest.to_csv(root / "manifest.csv", index=False)
    (root / "duplicate_audit.json").write_text(json.dumps({"dhash_max_distance": 4, "group_merges": merges}, indent=2))
    (root / "sources.json").write_text(json.dumps(SOURCES, indent=2))
    (root / "download_errors.json").write_text(json.dumps(errors, indent=2))
    summary = manifest.groupby(["source", "label", "split"]).size().rename("images").reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {len(manifest)} images: {root / 'manifest.csv'}", flush=True)
    if errors:
        print(f"WARNING: {len(errors)} downloads failed; see download_errors.json", flush=True)


if __name__ == "__main__":
    main()
