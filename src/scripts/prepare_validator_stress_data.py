"""Independent COCO bottle stress set; never included in validator training.

Each source photograph contributes a full scene and a bottle crop. This is a
negative-only stress test, not another training/validation split.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import pandas as pd
from PIL import Image, ImageOps
from prepare_validator_data import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data/validator")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    root = args.output.resolve()
    archive = fetch("https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip", root / "downloads/coco_annotations.zip")
    with zipfile.ZipFile(archive) as z:
        data = json.loads(z.read("annotations/instances_val2017.json"))
    bottle_id = next(x["id"] for x in data["categories"] if x["name"] == "bottle")
    images = {x["id"]: x for x in data["images"]}
    licenses = {x["id"]: x for x in data["licenses"]}
    boxes = {}
    for annotation in data["annotations"]:
        if annotation["category_id"] != bottle_id or annotation.get("iscrowd"):
            continue
        image = images[annotation["image_id"]]
        box = annotation["bbox"]
        # Require a visible, reasonably large bottle, not an incidental tiny one.
        if box[2] * box[3] < 0.1 * image["width"] * image["height"] or min(box[2:]) < 64:
            continue
        previous = boxes.get(image["id"])
        if previous is None or box[2] * box[3] > previous[2] * previous[3]:
            boxes[image["id"]] = box
    ids = sorted(boxes)
    random.Random(42).shuffle(ids)
    ids = ids[:args.count]

    def download(identity):
        meta = images[identity]
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{meta['file_name']}"
        original = fetch(url, root / "downloads/coco" / meta["file_name"])
        with Image.open(original) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        x, y, w, h = boxes[identity]
        crop = image.crop((max(0, int(x - w * 0.15)), max(0, int(y - h * 0.15)),
                           min(image.width, int(x + w * 1.15)), min(image.height, int(y + h * 1.15))))
        rows = []
        for kind, variant in [("bottle_scene", image), ("bottle_crop", crop)]:
            destination = root / "images/coco_stress" / f"{identity}_{kind}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            variant.thumbnail((512, 512))
            variant.save(destination, quality=92)
            rows.append({"path": str(destination.relative_to(root)), "label": 0, "split": "test",
                         "group_id": f"coco:{identity}", "source": "coco_stress", "category": kind,
                         "source_url": url, "original_url": meta.get("flickr_url"),
                         "license": licenses[meta["license"]]["name"], "license_url": licenses[meta["license"]]["url"],
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download, identity) for identity in ids]
        for future in as_completed(futures):
            rows.extend(future.result())
    pd.DataFrame(rows).sort_values("path").to_csv(root / "stress_manifest.csv", index=False)
    print(f"Independent test: {len(rows)} images from {len(ids)} bottle photographs. No training use.")


if __name__ == "__main__":
    main()
