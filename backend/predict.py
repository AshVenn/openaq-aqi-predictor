from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.aqi import aqi_category, compute_aqi_row, convert_to_standard
from src.features import add_time_features

from .schemas import PredictRequest

INPUT_POLLUTANTS = ["pm25", "pm10", "no2", "o3", "co", "so2"]


def _standardize_pollutants(
    pollutant_values: Dict[str, Optional[float]],
    unit_values: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[float]]:
    standardized: Dict[str, Optional[float]] = {}
    for pollutant in INPUT_POLLUTANTS:
        value = pollutant_values.get(pollutant)
        unit = unit_values.get(pollutant) if unit_values else None
        if value is None or (isinstance(value, float) and np.isnan(value)):
            standardized[pollutant] = None
            continue
        if unit:
            converted, _ = convert_to_standard(pollutant, value, unit)
            standardized[pollutant] = converted
        else:
            standardized[pollutant] = float(value)
    return standardized


def build_feature_frame(
    request: PredictRequest, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]], List[str]]:
    pollutant_values = request.pollutants.dict()
    unit_values = request.units.dict() if request.units else {}

    standardized = _standardize_pollutants(pollutant_values, unit_values)

    row = {
        "latitude": request.latitude,
        "longitude": request.longitude,
        "timestamp": request.timestamp,
    }
    row.update({p: standardized.get(p) for p in INPUT_POLLUTANTS})

    df = pd.DataFrame([row])
    df = add_time_features(df, time_col="timestamp")

    for pollutant in INPUT_POLLUTANTS:
        df[f"{pollutant}_is_missing"] = df[pollutant].isna().astype(int)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    provided = [p for p, v in pollutant_values.items() if v is not None]

    return df[feature_cols], standardized, provided


def compute_exact_aqi(standardized: Dict[str, Optional[float]]) -> Tuple[Optional[float], Optional[str]]:
    row_values = {
        pollutant: (None if value is None else float(value))
        for pollutant, value in standardized.items()
    }
    has_any = any(value is not None for value in row_values.values())
    if not has_any:
        return None, None

    aqi_exact = compute_aqi_row(row_values)
    return aqi_exact, aqi_category(aqi_exact)
