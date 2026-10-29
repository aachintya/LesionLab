"""Warm deterministic preprocessing on CPU before allocating a training GPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cache_one(task):
    import cv2
    from components.image_cache import load_preprocessed_image
    cv2.setNumThreads(1)
    image_path, modality, cache_dir = task
    load_preprocessed_image(image_path, modality=modality, roi_mode="foreground", cache_dir=cache_dir)
    return 1


def main():
    import pandas as pd
    from components.paths import FINAL_METADATA_WITH_SPLIT_CSV, training_image_dir
    from components.data_utils import filter_by_modality
    from components.dataset import MILK10kDataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=FINAL_METADATA_WITH_SPLIT_CSV)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    tasks = []
    df = pd.read_csv(args.csv)
    for modality in ("clinical", "dermoscopic"):
        dataset = MILK10kDataset(filter_by_modality(df, modality), str(training_image_dir()))
        for i in range(len(dataset)):
            path, _ = dataset._resolve_image_path(i)
            tasks.append((path, modality, str(args.cache_dir)))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, _ in enumerate(pool.map(cache_one, tasks, chunksize=8), 1):
            if count % 500 == 0:
                print(f"Cached {count}/{len(tasks)} images", flush=True)
    print(f"Cache ready: {len(tasks)} images in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
