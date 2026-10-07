"""Builds the full sklearn Pipeline: raw dataframe -> prediction."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config as cfg
from src.features import ZoneEncoder, add_features, add_route


def build_pipeline() -> Pipeline:
    """Feature engineering -> zones -> routes -> encoding/scaling -> Ridge(alpha=1)."""
    preprocess = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), cfg.CATEGORICAL_FEATURES),
        # rare and unseen routes are merged into one shared "infrequent" group
        ("ohe_route", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=cfg.ROUTE_MIN_COUNT), ["route"]),
        ("scale", StandardScaler(), cfg.NUMERIC_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("zones", ZoneEncoder(cfg.ZONE_COUNT, cfg.ZONE_FIT_SAMPLE_SIZE, cfg.RANDOM_STATE)),
        ("route_zones", ZoneEncoder(cfg.ROUTE_ZONE_COUNT, cfg.ZONE_FIT_SAMPLE_SIZE,
                                    cfg.RANDOM_STATE, prefix="route_")),
        ("route", FunctionTransformer(add_route)),
        ("preprocess", preprocess),
        ("model", Ridge(alpha=cfg.RIDGE_ALPHA)),
    ])