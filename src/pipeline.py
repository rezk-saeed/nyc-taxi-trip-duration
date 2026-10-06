"""Builds the full sklearn Pipeline: raw dataframe -> prediction."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config as cfg
from src.features import add_features
from src.features import ZoneEncoder, add_features


def build_pipeline() -> Pipeline:
    """Feature engineering -> encoding/scaling -> Ridge(alpha=1)."""
    preprocess = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), cfg.CATEGORICAL_FEATURES),
        ("scale", StandardScaler(), cfg.NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("zones", ZoneEncoder(cfg.ZONE_COUNT, cfg.ZONE_FIT_SAMPLE_SIZE, cfg.RANDOM_STATE)),
        ("preprocess", preprocess),
        ("model", Ridge(alpha=cfg.RIDGE_ALPHA)),
    ])