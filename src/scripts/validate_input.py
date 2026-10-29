"""JSON input-validation endpoint for upload clients; never returns a diagnosis."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.input_validation import DEFAULT_VALIDATOR, InputValidator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_VALIDATOR)
    args = parser.parse_args()
    result = InputValidator(args.checkpoint).validate(args.image)
    print(json.dumps(result.to_dict()))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
