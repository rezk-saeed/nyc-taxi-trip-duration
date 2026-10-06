"""Central configuration: paths, constants, thresholds."""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "split_sample"
SPLIT_DIR = ROOT / "data" / "split"

# ---- Columns ---------------------------------------------------------------
TARGET = "trip_duration"

# ---- Cleaning thresholds (train only; justified in the EDA notebook) -------
MIN_DURATION_SEC = 10
MAX_DURATION_SEC = 6 * 3600
NYC_LON_RANGE = (-74.3, -73.7)
NYC_LAT_RANGE = (40.5, 41.0)
MAX_SPEED_KMH = 100
STATIONARY_KM = 0.05               # "same place" if pickup and dropoff are closer than this
STATIONARY_MAX_DURATION_SEC = 600  # ... yet the trip lasted longer than this

# ---- Model (fixed by the project rules) ------------------------------------
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
MODEL_PATH = ROOT / "models" / "ridge_pipeline.joblib"

# ---- Features fed to the model (grow this list as we add features) ---------
CATEGORICAL_FEATURES = ["hour", "dayofweek", "month"]
NUMERIC_FEATURES = ["log_distance_km", "is_jfk", "is_lga"]

# ---- Airports ---------------------------------------------------------------
AIRPORTS = {
    "jfk": (40.6413, -73.7781),   # (latitude, longitude)
    "lga": (40.7769, -73.8740),
}
AIRPORT_RADIUS_KM = 2