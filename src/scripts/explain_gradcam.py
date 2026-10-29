from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from components.augmentations import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from components.data_utils import filter_by_modality
from components.dataset import MILK10kDataset
from models.efficientnet_v2_image import build_image_model
from utils.gradcam import GradCAM


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _denorm_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.detach().to("cpu").float().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = (x * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def _overlay_cam(rgb: np.ndarray, cam01: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam01, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((cam * 255.0).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha).clip(0, 255)
    return out.astype(np.uint8)


def _run_modality(
    df: pd.DataFrame,
    modality: str,
    image_dir: str,
    roi_mode: str,
    ckpt_path: str,
    num_classes: int,
    device: torch.device,
    out_dir: Path,
) -> int:
    _, tfm = get_transforms(modality=modality)
    dataset = MILK10kDataset(df, image_dir, transform=tfm, roi_mode=roi_mode, modality=modality)

    model = build_image_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    cam = GradCAM(model, model.backbone.features[-1], device=device)

    for i in range(len(dataset)):
        image, label = dataset[i]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))[0]
            pred = int(torch.argmax(logits).item())

        res = cam(images=image.unsqueeze(0), target_class=pred)
        overlay = _overlay_cam(_denorm_to_uint8(image), res.cam)

        lesion_id = str(df.iloc[i]["lesion_id"])
        fname = f"{lesion_id}_{modality}_true{int(label)}_pred{pred}.png"
        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    cam.close()
    return len(dataset)


def main() -> None:
    from components.paths import FINAL_METADATA_WITH_SPLIT_CSV, training_image_dir

    p = argparse.ArgumentParser(description="Generate Grad-CAM overlays for image modality models.")
    p.add_argument("--csv", type=str, default=str(FINAL_METADATA_WITH_SPLIT_CSV))
    p.add_argument("--image_dir", type=str, default=str(training_image_dir()))
    p.add_argument(
        "--modality",
        type=str,
        default="both",
        choices=["clinical", "dermoscopic", "both"],
        help="Modality to process. 'both' processes clinical then dermoscopic.",
    )
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint path (ignored when --modality=both).")
    p.add_argument("--out_dir", type=str, default="../outputs/gradcam_overlays")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--per-class", action="store_true", help="One lesion per class, same lesion_id across modalities.")
    p.add_argument("--extra", type=int, default=0, help="Extra random lesions appended after --per-class selection.")
    p.add_argument("--roi_mode", type=str, default="foreground", choices=["none", "foreground"])
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    device = _pick_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_all = pd.read_csv(args.csv)
    df_all = df_all[df_all["split"] == args.split].copy()
    num_classes = int(df_all["target"].nunique())

    # select lesion_ids (modality-agnostic so both share the same set)
    if args.per_class:
        rng = np.random.default_rng(args.seed)
        lesion_df = df_all.drop_duplicates("lesion_id")
        selected_ids = []
        for cls in sorted(lesion_df["target"].unique()):
            cls_df = lesion_df[lesion_df["target"] == cls]
            picked = cls_df.sample(n=1, random_state=int(rng.integers(1 << 31)))
            selected_ids.append(picked["lesion_id"].iloc[0])
        if args.extra > 0:
            pool = lesion_df[~lesion_df["lesion_id"].isin(selected_ids)]
            extras = pool.sample(n=min(args.extra, len(pool)), random_state=int(rng.integers(1 << 31)))
            selected_ids.extend(extras["lesion_id"].tolist())
    else:
        lesion_ids = df_all["lesion_id"].drop_duplicates()
        selected_ids = lesion_ids.sample(n=min(args.n, len(lesion_ids)), random_state=args.seed).tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["clinical", "dermoscopic"] if args.modality == "both" else [args.modality]

    for modality in modalities:
        df = filter_by_modality(df_all, modality)
        df = df[df["lesion_id"].isin(selected_ids)].copy()
        df = df.set_index("lesion_id").loc[selected_ids].reset_index()

        if len(df) == 0:
            print(f"No rows for modality={modality}, skipping.")
            continue

        ckpt = args.ckpt if args.modality != "both" else f"checkpoints/best_{modality}.pt"
        n = _run_modality(df, modality, args.image_dir, args.roi_mode, ckpt, num_classes, device, out_dir)
        print(f"Wrote {n} Grad-CAM overlays [{modality}] → {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
