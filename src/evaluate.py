"""Metrics, always computed on the log1p target."""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from src import config as cfg
def evaluate(model,df,name:str):
    """Predict on raw `df` and report RMSE / R2 against log1p(trip_duration)."""
    y_true = np.log1p(df[cfg.TARGET])
    y_pred = model.predict(df)
    metrics = {"rmse": float(np.sqrt(mean_squared_error(y_true,y_pred))),
                "r2": float(r2_score(y_true,y_pred)),}

    print(f"[{name}] RMSE = {metrics['rmse']:.4f} | R2 = {metrics['r2']:.4f}")
    return metrics


