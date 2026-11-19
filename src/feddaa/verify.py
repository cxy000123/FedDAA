"""Check the packaged files and compare input hashes with the paper's data."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from feddaa import CONFIG_DIR


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(paths, expected):
    problems = []
    for name, path in paths.items():
        if not path.is_file():
            problems.append(f"Missing: {path}")
        elif digest(path) != expected[name]:
            problems.append(f"SHA-256 differs: {path}")
    return problems


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", action="store_true", help="Require the exact inputs used in the paper")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fixed-scalers", type=Path, default=Path("data/scalers.json"))
    args = parser.parse_args()
    if args.paper:
        expected = read(CONFIG_DIR / "input_hashes.json")
        paths = {name: args.data_dir / name for name in expected}
        paths["scalers.json"] = args.fixed_scalers
    else:
        root = Path.cwd()
        expected = read(root / "SHA256.json")
        paths = {name: root / name for name in expected}
    errors = check(paths, expected)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print(f"Verified {len(paths)} files.")


if __name__ == "__main__":
    main()
