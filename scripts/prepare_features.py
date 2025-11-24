import argparse
from pathlib import Path
from src.large_scale_features import run_large_scale_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunksize", type=int, default=100000)
    p.add_argument("--id-cols", nargs="*", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_large_scale_features(args.input, args.output_dir, args.chunksize, args.id_cols)


if __name__ == "__main__":
    main()



