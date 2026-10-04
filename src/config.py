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