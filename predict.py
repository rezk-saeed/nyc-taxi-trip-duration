"""Load the saved model and score it on a CSV file."""
"""Load the saved model and score it on a CSV file.

The saved pipeline already contains every preprocessing step (features, zones,
routes, encoding), so the CSV only needs the RAW trip columns.
No cleaning is applied here: validation/test data is scored as it comes.

Usage (from the project root):
    python predict.py --data data/split/val.csv
    python predict.py --data data/split/test.csv --model-path models/ridge_pipeline.joblib
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences the joblib core-count warning on Windows

import argparse
from pathlib import Path

import joblib

from src import config as cfg
from src.data import load_csv
from src.evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to the CSV file to score.")
    parser.add_argument("--model-path", default=cfg.MODEL_PATH,
                        help="Path to the saved model (default: models/ridge_pipeline.joblib).")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model_path)

    for path in (data_path, model_path):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

    model = joblib.load(model_path)
    df = load_csv(data_path)
    print(f"[predict] loaded {len(df)} rows from {data_path.name}")
    evaluate(model, df, data_path.name)


if __name__ == "__main__":
    main()