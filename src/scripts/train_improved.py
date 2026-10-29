"""Train one isolated v2 seed. Use run_improved.py for the two-GPU workflow."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if not (0 <= args.freeze_epochs < args.epochs and args.batch_size > 0 and args.accumulate > 0
            and args.patience > 0 and args.size >= 64 and args.cpu_threads > 0):
        raise ValueError("Invalid training settings")
    from improved.training import train
    train(args)
