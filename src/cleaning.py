"""Outlier removal. Applied to the TRAINING set only, never to val/test."""
import pandas as pd

from src import config as cfg
from src.features import haversine_km


def outlier_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True for trips that break at least one rule."""
    duration = df[cfg.TARGET]
    distance_km = haversine_km(df["pickup_latitude"], df["pickup_longitude"],
                               df["dropoff_latitude"], df["dropoff_longitude"])
    speed_kmh = distance_km / (duration / 3600)

    lon_lo, lon_hi = cfg.NYC_LON_RANGE
    lat_lo, lat_hi = cfg.NYC_LAT_RANGE
    in_nyc = (
        df["pickup_longitude"].between(lon_lo, lon_hi)
        & df["dropoff_longitude"].between(lon_lo, lon_hi)
        & df["pickup_latitude"].between(lat_lo, lat_hi)
        & df["dropoff_latitude"].between(lat_lo, lat_hi)
    )

    return (
        (duration < cfg.MIN_DURATION_SEC)
        | (duration > cfg.MAX_DURATION_SEC)
        | ~in_nyc
        | (speed_kmh > cfg.MAX_SPEED_KMH)
        | ((distance_km < cfg.STATIONARY_KM) & (duration > cfg.STATIONARY_MAX_DURATION_SEC))
    )


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    """Drop outlier trips from the training set and report how many were removed."""
    bad = outlier_mask(df)
    print(f"[cleaning] removed {bad.sum()} of {len(df)} trips ({100 * bad.mean():.2f}%)")
    return df.loc[~bad].reset_index(drop=True)