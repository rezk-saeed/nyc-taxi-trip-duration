# NYC Taxi Trip Duration (Ridge, alpha=1)

Predicts the duration of New York City taxi trips. The model is fixed to
`Ridge(alpha=1)`, so every gain comes from data cleaning and feature engineering.
The target is `log1p(trip_duration)` and the metric is R².

## Final result (validation set, 229,319 trips)

| Metric | Value |
|---|---|
| Val R² | 0.7107 |
| Val RMSE (log scale) | 0.4303 |
| Train R² (cleaned train set) | 0.7795 |

Progress on the validation set while features were added (full data):

| Step | Val R² |
|---|---|
| Baseline: log distance + hour + day of week | 0.6005 |
| + airport indicators (JFK, LGA) | 0.6122 |
| + KMeans pickup/dropoff zones (20 zones, month also used) | 0.6408 |
| 500 zones | 0.6569 |
| day of year replaces month | 0.6601 |
| + route feature (30 coarse zones, min count 100) | 0.6900 |
| route tuned: 200 zones, min count 10 | **0.7107** |

## Setup

```bash
pip install -r requirements.txt
```

Place the course data under `data/split/` (`train.csv`, `val.csv`) and, for quick
checks, `data/split_sample/`. Data and model files are not tracked by git.

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
   10 s or over 6 h, coordinates outside NYC, speeds over 100 km/h, and long trips
   that start and end at the same place. Validation and test data are never cleaned.
2. **Features** (all inside one sklearn `Pipeline`, so the saved model accepts raw rows):
   - Haversine distance (log-transformed)
   - Hour, day of week, day of year (one-hot)
   - JFK / LGA indicators (trip starts or ends within 2 km)
   - KMeans zones for pickup and dropoff (500 fine-grained zones)
   - Route: pickup zone x dropoff zone on a coarse zoning, with rare routes merged
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
- Zones and routes are learned from the training data; locations far from any
  training trip fall back to the nearest zone.