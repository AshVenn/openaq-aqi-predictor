from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    # Avoid sklearn version differences (older versions don't support squared=)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def summarize_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"MAE={metrics['mae']:.2f}, "
        f"RMSE={metrics['rmse']:.2f}, "
        f"R2={metrics['r2']:.3f}"
    )
