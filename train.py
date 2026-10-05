"""Train the Ridge pipeline, evaluate on validation, and save it."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src import config as cfg
from src.cleaning import clean_train
from src.data import load_csv
from src.evaluate import evaluate
from src.pipeline import build_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=cfg.SAMPLE_DIR / "train.csv")
    parser.add_argument("--val", default=None,
                        help="Validation CSV. If omitted, 20%% of train is held out.")
    parser.add_argument("--model-path", default=cfg.MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    train = load_csv(args.train)

    if args.val:
        val = load_csv(args.val)
    else:
        print("[train] no --val given: holding out 20% of train")
        train, val = train_test_split(train, test_size=0.2, random_state=cfg.RANDOM_STATE)

    train = clean_train(train)               # clean TRAIN only; val stays raw
    y_train = np.log1p(train[cfg.TARGET])    # the model predicts log1p(duration)

    model = build_pipeline()
    model.fit(train, y_train)

    evaluate(model, train, "train")
    evaluate(model, val, "val")

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[train] model saved to {model_path}")
    pass


if __name__ == "__main__":
    main()



