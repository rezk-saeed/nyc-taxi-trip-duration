"""Feature engineering helpers."""
import numpy as np
import pandas as pd
from src import config as cfg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

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
    out["dayofyear"] = pickup.dt.dayofyear
    # --- distance ---
    out["distance_km"] = haversine_km(
        out["pickup_latitude"], out["pickup_longitude"],
        out["dropoff_latitude"], out["dropoff_longitude"],
    )
    out["log_distance_km"] = np.log1p(out["distance_km"])

        # --- airports: trip starts OR ends near the airport ---
    for name, (air_lat, air_lon) in cfg.AIRPORTS.items():
        pickup_km = haversine_km(out["pickup_latitude"], out["pickup_longitude"], air_lat, air_lon)
        dropoff_km = haversine_km(out["dropoff_latitude"], out["dropoff_longitude"], air_lat, air_lon)
        out[f"is_{name}"] = (( pickup_km < cfg.AIRPORT_RADIUS_KM ) | ( dropoff_km < cfg.AIRPORT_RADIUS_KM )).astype(int)

    return out

class ZoneEncoder(BaseEstimator, TransformerMixin):
    """Assign every pickup and dropoff point to a KMeans zone (fit on train only)."""

    def __init__(self, n_zones=20, fit_sample_size=200_000, random_state=42):
        self.n_zones = n_zones
        self.fit_sample_size = fit_sample_size
        self.random_state = random_state

    def fit(self, X, y=None):
        points = np.vstack([
            X[["pickup_latitude", "pickup_longitude"]].to_numpy(),
            X[["dropoff_latitude", "dropoff_longitude"]].to_numpy(),
        ])
        # KMeans on millions of points is slow, so learn the zones from a random sample
        if len(points) > self.fit_sample_size:
            rng = np.random.default_rng(self.random_state)
            points = points[rng.choice(len(points), self.fit_sample_size, replace=False)]
        self.kmeans_ = KMeans(n_clusters=self.n_zones, n_init=3,random_state=self.random_state).fit(points)
        return self

    def transform(self, X):
        out = X.copy()
        out["pickup_zone"] = self.kmeans_.predict(X[["pickup_latitude", "pickup_longitude"]].to_numpy())
        out["dropoff_zone"] = self.kmeans_.predict(X[["dropoff_latitude", "dropoff_longitude"]].to_numpy())
        return out