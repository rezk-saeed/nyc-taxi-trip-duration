import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")   # silences the joblib core-count warning on Windows


"""Run a grid of route settings in one go and print a results table.

Speed trick: the steps that do not depend on the route settings (feature
engineering + the 500 fine zones) are fitted ONCE, then only the cheap
remaining steps are re-fitted for every grid row.

Usage (from the project root):
    python experiments.py
    python experiments.py --train data/split/train.csv --val data/split/val.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src import config as cfg
from src.cleaning import clean_train
from src.data import load_csv
from src.pipeline import build_pipeline

# (ROUTE_ZONE_COUNT, ROUTE_MIN_COUNT) combinations to try
GRID = [(100, 100), (100, 50), (100, 30), (100, 10), (150, 30)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=cfg.SAMPLE_DIR / "train.csv")
    parser.add_argument("--val", default=cfg.SAMPLE_DIR / "val.csv")
    parser.add_argument("--out", default=cfg.ROOT / "reports" / "route_experiments.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    train = clean_train(load_csv(args.train))
    val = load_csv(args.val)
    y_train = np.log1p(train[cfg.TARGET])
    y_val = np.log1p(val[cfg.TARGET])

    # Fit the route-independent steps once (features + fine zones)
    base = build_pipeline()[:2]
    x_train = base.fit_transform(train)
    x_val = base.transform(val)

    rows = []
    for zones, min_count in GRID:
        cfg.ROUTE_ZONE_COUNT, cfg.ROUTE_MIN_COUNT = zones, min_count
        rest = build_pipeline()[2:]               # route zones, route id, encoding, Ridge
        start = time.time()
        rest.fit(x_train, y_train)
        seconds = time.time() - start

        row = {
            "route_zones": zones,
            "min_count": min_count,
            "val_r2": r2_score(y_val, rest.predict(x_val)),
            "val_rmse": float(np.sqrt(mean_squared_error(y_val, rest.predict(x_val)))),
            "train_r2": r2_score(y_train, rest.predict(x_train)),
            "fit_seconds": round(seconds),
        }
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    results = pd.DataFrame(rows).round(4)
    results["gap"] = (results["train_r2"] - results["val_r2"]).round(4)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print("\n", results.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()