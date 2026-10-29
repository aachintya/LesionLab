"""Train all three modality models sequentially."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.train_model import CLINICAL_TRAIN_DEFAULTS, _default_csv, train_modality  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clinical, dermoscopic, and metadata models.")
    parser.add_argument("--csv", type=Path, default=_default_csv())
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs for dermoscopic/metadata (clinical default: 30).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for dermoscopic/metadata image models.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=None,
        help="Freeze epochs for dermoscopic (clinical default: 6).",
    )
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument(
        "--select-metric",
        choices=["f1", "recall", "accuracy"],
        default="f1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    for modality in ("clinical", "dermoscopic", "metadata"):
        print("\n" + "=" * 60)
        print(f"Starting training: {modality}")
        print("=" * 60)
        kwargs: dict = {
            "csv_path": args.csv,
            "device_name": args.device, "num_workers": args.num_workers, "cache_dir": args.cache_dir,
            "amp": args.amp, "channels_last": args.channels_last, "resume": args.resume,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "use_wandb": args.wandb,
            "head_lr": args.head_lr,
            "backbone_lr": args.backbone_lr,
            "sampler_power": args.sampler_power,
            "class_weight_power": args.class_weight_power,
            "select_metric": args.select_metric,
        }
        if modality == "clinical":
            kwargs["epochs"] = int(
                args.epochs if args.epochs is not None else CLINICAL_TRAIN_DEFAULTS["epochs"]
            )
            kwargs["freeze_epochs"] = int(
                args.freeze_epochs
                if args.freeze_epochs is not None
                else CLINICAL_TRAIN_DEFAULTS["freeze_epochs"]
            )
        else:
            kwargs["epochs"] = 20 if args.epochs is None else args.epochs
            kwargs["freeze_epochs"] = 4 if args.freeze_epochs is None else args.freeze_epochs

        train_modality(modality, **kwargs)


if __name__ == "__main__":
    main()
