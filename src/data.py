"""Data loading helpers."""
import pandas as pd


def load_csv(path) -> pd.DataFrame:
    """Read a trip CSV and parse the pickup timestamp."""
    return pd.read_csv(path, parse_dates=["pickup_datetime"])