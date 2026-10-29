"""JI-ADF research baseline: smoke, train, run, evaluate and freeze."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parser():
    from ji_adf.model import VARIANTS
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)

    def inputs(name, output=True):
        command = commands.add_parser(name)
        command.add_argument("--reference", type=Path, required=True)
        if output:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--images", type=Path, required=True)
            command.add_argument("--cache", type=Path)
        return command

    def training_options(command):
        command.add_argument("--initialization", choices=["imagenet", "v1"], default="imagenet")
        command.add_argument("--epochs", type=int, default=50)
        command.add_argument("--freeze-epochs", type=int, default=0)
        command.add_argument("--patience", type=int, default=10)
        command.add_argument("--size", type=int, default=384)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--accumulate", type=int, default=4)
        command.add_argument("--head-lr", type=float, default=1e-4)
        command.add_argument("--backbone-lr", type=float, default=1e-4)
        command.add_argument("--weight-decay", type=float, default=1e-5)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--cpu-threads", type=int, default=2)

    training = inputs("train")
    training_options(training)
    training.add_argument("--variant", choices=VARIANTS, default="ji_adf")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--resume", action="store_true")
    run = inputs("run")
    training_options(run)
    run.add_argument("--variants", nargs="+", choices=VARIANTS, default=["ji_adf", "concat_fixed"])
    run.add_argument("--seeds", nargs="+", type=int, default=[42])
    smoke = inputs("smoke", False)
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--variant", choices=VARIANTS, default="ji_adf")
    smoke.add_argument("--initialization", choices=["imagenet", "v1"], default="imagenet")
    smoke.add_argument("--size", type=int, default=384)
    smoke.add_argument("--batch-size", type=int, default=4)
    smoke.add_argument("--accumulate", type=int, default=4)
    evaluation = inputs("evaluate")
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.add_argument("--reproduce-baseline", action="store_true")
    evaluation.add_argument("--bootstrap", type=int, default=2000)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--destination", type=Path, required=True)
    freeze.add_argument("--name", default="ji-adf-v1")
    verify = commands.add_parser("verify")
    verify.add_argument("path", type=Path)
    return p


def main():
    args = parser().parse_args()
    for key in ("size", "batch_size", "accumulate", "epochs", "patience", "cpu_threads"):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if hasattr(args, "freeze_epochs") and not 0 <= args.freeze_epochs < args.epochs:
        raise ValueError("freeze_epochs must be smaller than epochs")
    if hasattr(args, "size") and args.size < 64:
        raise ValueError("Image size must be at least 64")
    if hasattr(args, "workers") and args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.command == "train":
        from ji_adf.training import train
        train(args)
    elif args.command in ("run", "smoke"):
        from ji_adf import workflow
        getattr(workflow, args.command)(args)
    elif args.command == "evaluate":
        from ji_adf.evaluation import evaluate
        evaluate(args)
    elif args.command == "freeze":
        from ji_adf.baseline import freeze_run
        print(freeze_run(args.run, args.destination, args.name))
    elif args.command == "verify":
        from utils.baselines import verify_baseline
        print(json.dumps(verify_baseline(args.path), indent=2))


if __name__ == "__main__":
    main()
