# NYC Taxi Trip Duration (Ridge, alpha=1)

> **Note:** This is an educational machine-learning project. It uses the
> NYC Taxi Trip Duration data (originally from Kaggle, with a train/validation/test
> split provided by the course). The data files are **not** included in this repository.

Predicts the duration of New York City taxi trips. The model is fixed to
`Ridge(alpha=1)`, so every gain comes from data cleaning and feature engineering.
The target is `log1p(trip_duration)` and the metric is R² (computed on that log target).

## Results

| Metric | Value |
|---|---|
| Val R² (229,319 trips) | 0.7107 |
| Val RMSE (log scale) | 0.4303 |
| Train R² (cleaned train set) | 0.7795 |
| **Test R²** (229,322 trips, evaluated once after freezing the model) | **0.7151** |
| Test RMSE (log scale) | 0.4248 |

Progress on the validation set as features were added (full data):

| Step | Val R² |
|---|---|
| Baseline: log distance + hour + day of week | 0.6005 |
| + airport indicators (JFK, LGA) | 0.6122 |
| + month | 0.6134 |
| + KMeans pickup/dropoff zones (20 zones) | 0.6408 |
| Zones: 500 | 0.6569 |
| Day of year replaces month | 0.6601 |
| + route feature (30 coarse zones, min count 100) | 0.6900 |
| Routes: 100 zones, min count 100 | 0.6983 |
| Routes: 150 zones, min count 30 | 0.7063 |
| Routes: 200 zones, min count 10 (final) | **0.7107** |

The full write-up is in `reports/NYC_Taxi_Trip_Duration_Report.pdf`, and the
exploratory analysis is in `notebooks/NYC_Taxi_Trip_Duration_EDA.ipynb`.

## Data

The project expects the course split of the dataset:

- `data/split/train.csv`, `data/split/val.csv` (about 1M and 229K trips)
- `data/split_sample/` (small sample for quick checks)

These files, the test set and the trained model are excluded from git (see `.gitignore`).

## Setup

```bash
pip install -r requirements.txt
```

## Usage (run from the project root)

```bash
# Train on the full data, evaluate on validation, save the model
python train.py --train data/split/train.csv --val data/split/val.csv

# Load the saved model and score any CSV with the raw trip columns
python predict.py --data data/split/val.csv

# Quick run on the small sample
python train.py --val data/split_sample/val.csv

# Re-run the route-setting grid used for tuning
python experiments.py --train data/split/train.csv --val data/split/val.csv
```

## How it works

1. **Cleaning (train only):** `src/cleaning.py` removes trips with durations under
   10 s or over 6 h, coordinates outside NYC, speeds over 100 km/h, and trips longer
   than 10 min that start and end within 50 m. Validation and test data are never cleaned.
2. **Features** (all inside one sklearn `Pipeline`, so the saved model accepts raw rows):
   - Haversine distance (log-transformed)
   - Hour, day of week, day of year (one-hot)
   - JFK / LGA indicators (trip starts or ends within 2 km)
   - KMeans zones for pickup and dropoff (500 fine-grained zones)
   - Route: pickup zone x dropoff zone on a coarser 200-zone map; routes with fewer
     than 10 training trips are merged into one shared group
3. **Model:** `Ridge(alpha=1)` on the one-hot and scaled features.

## Project layout

```
src/config.py      paths, thresholds, feature settings
src/data.py        CSV loading
src/cleaning.py    train-only outlier rules
src/features.py    feature functions and the ZoneEncoder
src/pipeline.py    builds the full pipeline
src/evaluate.py    RMSE / R2 on the log target
train.py           train, evaluate, save
predict.py         load the saved model and score a CSV
experiments.py     grid runner for the route settings
notebooks/         EDA notebook (no modeling)
reports/           experiment results and the final report
```

## Limitations

- Day of year only works for the period covered by the data (January to June 2016)
  and would not generalize to another year.
- About 25 configurations were compared on the validation set, so the validation score
  could be slightly optimistic; the single test evaluation is the unbiased estimate.
- Zones and routes are learned from the training data; locations far from any
  training trip fall back to the nearest zone.