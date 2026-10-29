"""Freeze, verify, or compare a named baseline without modifying active models."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils.baselines import compare_metrics, freeze_baseline, verify_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="Save a new read-only snapshot; existing names cannot be overwritten")
    freeze.add_argument("--artifacts", type=Path, required=True)
    freeze.add_argument("--name", required=True)
    verify = commands.add_parser("verify", help="Check every saved file against its SHA-256")
    verify.add_argument("baseline", type=Path)
    compare = commands.add_parser("compare", help="Compare results only when evaluation protocols match")
    compare.add_argument("--baseline", type=Path, default=ROOT / "baselines/baseline-v1-2026-10-27")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            directory = freeze_baseline(ROOT, args.artifacts, args.name)
            result = verify_baseline(directory)
        elif args.command == "verify":
            result = verify_baseline(args.baseline)
        else:
            verify_baseline(args.baseline)
            result = compare_metrics(json.loads((args.baseline / "metrics.json").read_text()),
                                     json.loads(args.candidate.read_text()))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
