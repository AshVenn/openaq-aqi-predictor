from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = df[time_col].dt.hour
    df["day_of_week"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_cols: Sequence[str],
    lags: Iterable[int] = (1,),
    time_col: str = "timestamp",
) -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(list(group_cols) + [time_col])
    for lag in lags:
        for col in target_cols:
            if col not in df.columns:
                continue
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df.groupby(list(group_cols))[col].shift(lag)
    return df


def build_feature_columns(
    pollutant_cols: Sequence[str], include_lags: bool = True
) -> List[str]:
    base_features = list(pollutant_cols) + ["hour", "day_of_week", "month", "latitude", "longitude"]
    if not include_lags:
        return base_features
    lag_features = [f"{c}_lag1" for c in pollutant_cols]
    return base_features + lag_features
