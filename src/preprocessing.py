from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.aqi import convert_to_standard


POLLUTANTS = {"pm25", "pm10", "no2", "o3", "co", "so2"}


def _normalize_col_name(name: str) -> str:
    name = name.strip().lower()
    name = name.replace(" ", "_").replace("-", "_")
    return name


def normalize_pollutant_name(p: Optional[str]) -> str:
    """
    Normalize pollutant names so PM2.5 variants map to 'pm25' to avoid dropping data.
    """
    if p is None:
        return ""
    text = str(p).strip().lower()
    compact = text.replace(" ", "").replace("_", "").replace("-", "")
    if compact == "pm2.5" or compact == "pm25":
        return "pm25"
    return compact


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalized = {_normalize_col_name(c): c for c in df.columns}
    rename_map = {}
    for norm, original in normalized.items():
        if norm in ("country", "city", "location", "coordinates", "pollutant", "value"):
            rename_map[original] = norm
        elif norm in ("unit",):
            rename_map[original] = "unit"
        elif norm in ("source_name", "source"):
            rename_map[original] = "source_name"
        elif norm in ("last_updated", "last_updated_utc", "datetime"):
            rename_map[original] = "last_updated"
    df = df.rename(columns=rename_map)
    return df


def parse_coordinates(value: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    text = str(value)
    numbers = re.findall(r"-?\d+\.\d+|-?\d+", text)
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None, None


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    unit = str(unit).strip().lower()
    unit = unit.replace("\u00b5", "u")
    unit = unit.replace("ug/m^3", "ug/m3")
    unit = unit.replace("ug/m\u00b3", "ug/m3")
    unit = unit.replace("mg/m\u00b3", "mg/m3")
    return unit


def load_raw_data(path: str) -> pd.DataFrame:
    # OpenAQ exports often use semicolon delimiters; allow fallback sniffing.
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8")
        if df.shape[1] <= 1:
            raise ValueError("Unexpected delimiter; falling back to sniffing.")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    df = standardize_columns(df)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    raw_pollutants = df["pollutant"].copy()
    df["pollutant"] = df["pollutant"].apply(normalize_pollutant_name)
    df = df[df["pollutant"].isin(POLLUTANTS)]
    raw_has_pm25 = raw_pollutants.apply(normalize_pollutant_name).eq("pm25").any()

    df["unit"] = df["unit"].apply(normalize_unit)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    lat_lon = df["coordinates"].apply(parse_coordinates)
    df["latitude"] = lat_lon.apply(lambda x: x[0])
    df["longitude"] = lat_lon.apply(lambda x: x[1])

    converted = df.apply(
        lambda row: convert_to_standard(row["pollutant"], row["value"], row["unit"]),
        axis=1,
        result_type="expand",
    )
    converted.columns = ["value_std", "unit_std"]
    df = pd.concat([df, converted], axis=1)

    df = df.dropna(subset=["timestamp", "value_std"])

    df = df.drop_duplicates(
        subset=[
            "timestamp",
            "location",
            "pollutant",
            "value_std",
            "latitude",
            "longitude",
        ]
    )

    keep_cols = [
        "country",
        "city",
        "location",
        "latitude",
        "longitude",
        "timestamp",
        "pollutant",
        "value_std",
        "unit_std",
        "source_name",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    print("Pollutant counts after cleaning:")
    print(df["pollutant"].value_counts(dropna=False))
    if raw_has_pm25:
        assert df["pollutant"].eq("pm25").any(), "pm25 present in raw data but missing after cleaning"
    return df


def aggregate_and_pivot(df_long: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    df = df_long.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.floor(freq)

    preferred_keys = ["source_name", "location", "latitude", "longitude", "timestamp"]
    fallback_keys = ["location", "latitude", "longitude", "timestamp"]
    if all(k in df.columns for k in preferred_keys):
        stable_keys = preferred_keys
    else:
        stable_keys = [k for k in fallback_keys if k in df.columns]
    if not stable_keys:
        raise ValueError("No stable keys found for aggregation.")

    group_cols = stable_keys + ["pollutant"]
    grouped = df.groupby(group_cols, dropna=False)["value_std"].mean().reset_index()

    wide = grouped.pivot_table(
        index=stable_keys,
        columns="pollutant",
        values="value_std",
        aggfunc="mean",
    ).reset_index()

    extra_cols = [c for c in ["country", "city"] if c in df.columns]
    if extra_cols:
        def _first_non_null(series: pd.Series):
            series = series.dropna()
            return series.iloc[0] if not series.empty else np.nan

        meta = (
            df.groupby(stable_keys, dropna=False)[extra_cols]
            .agg(_first_non_null)
            .reset_index()
        )
        wide = wide.merge(meta, on=stable_keys, how="left")

    wide.columns.name = None
    return wide
