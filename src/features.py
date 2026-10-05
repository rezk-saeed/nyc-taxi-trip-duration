"""Feature engineering helpers."""
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (works on arrays/Series)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def add_features(df):
    """Return a copy of `df` with the engineered columns added.

    Takes RAW trip rows, so the exact same function runs at train time and at
    predict time (it lives inside the Pipeline).
    """
    out = df.copy()
    pickup = pd.to_datetime(out["pickup_datetime"])

    # --- time ---
    out["hour"] = pickup.dt.hour
    out["dayofweek"] = pickup.dt.dayofweek

    # --- distance ---
    out["distance_km"] = haversine_km(
        out["pickup_latitude"], out["pickup_longitude"],
        out["dropoff_latitude"], out["dropoff_longitude"],
    )
    out["log_distance_km"] = np.log1p(out["distance_km"])
    return out