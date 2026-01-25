from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Breakpoint:
    bp_low: float
    bp_high: float
    i_low: int
    i_high: int


BREAKPOINTS: Dict[str, Dict[str, Iterable[Breakpoint]]] = {
    # US EPA AQI breakpoints
    "pm25": {
        "unit": "ug/m3",
        "table": [
            Breakpoint(0.0, 12.0, 0, 50),
            Breakpoint(12.1, 35.4, 51, 100),
            Breakpoint(35.5, 55.4, 101, 150),
            Breakpoint(55.5, 150.4, 151, 200),
            Breakpoint(150.5, 250.4, 201, 300),
            Breakpoint(250.5, 350.4, 301, 400),
            Breakpoint(350.5, 500.4, 401, 500),
        ],
    },
    "pm10": {
        "unit": "ug/m3",
        "table": [
            Breakpoint(0, 54, 0, 50),
            Breakpoint(55, 154, 51, 100),
            Breakpoint(155, 254, 101, 150),
            Breakpoint(255, 354, 151, 200),
            Breakpoint(355, 424, 201, 300),
            Breakpoint(425, 504, 301, 400),
            Breakpoint(505, 604, 401, 500),
        ],
    },
    "o3": {
        "unit": "ppm",
        "table": [
            Breakpoint(0.000, 0.054, 0, 50),
            Breakpoint(0.055, 0.070, 51, 100),
            Breakpoint(0.071, 0.085, 101, 150),
            Breakpoint(0.086, 0.105, 151, 200),
            Breakpoint(0.106, 0.200, 201, 300),
            Breakpoint(0.201, 0.604, 301, 500),
        ],
    },
    "co": {
        "unit": "ppm",
        "table": [
            Breakpoint(0.0, 4.4, 0, 50),
            Breakpoint(4.5, 9.4, 51, 100),
            Breakpoint(9.5, 12.4, 101, 150),
            Breakpoint(12.5, 15.4, 151, 200),
            Breakpoint(15.5, 30.4, 201, 300),
            Breakpoint(30.5, 40.4, 301, 400),
            Breakpoint(40.5, 50.4, 401, 500),
        ],
    },
    "so2": {
        "unit": "ppb",
        "table": [
            Breakpoint(0, 35, 0, 50),
            Breakpoint(36, 75, 51, 100),
            Breakpoint(76, 185, 101, 150),
            Breakpoint(186, 304, 151, 200),
            Breakpoint(305, 604, 201, 300),
            Breakpoint(605, 804, 301, 400),
            Breakpoint(805, 1004, 401, 500),
        ],
    },
    "no2": {
        "unit": "ppb",
        "table": [
            Breakpoint(0, 53, 0, 50),
            Breakpoint(54, 100, 51, 100),
            Breakpoint(101, 360, 101, 150),
            Breakpoint(361, 649, 151, 200),
            Breakpoint(650, 1249, 201, 300),
            Breakpoint(1250, 1649, 301, 400),
            Breakpoint(1650, 2049, 401, 500),
        ],
    },
}

MOLECULAR_WEIGHTS = {
    "o3": 48.00,
    "no2": 46.01,
    "so2": 64.07,
    "co": 28.01,
}


def _ugm3_to_ppm(value_ugm3: float, mw: float) -> float:
    # 25C and 1 atm; ppm = (ug/m3) * 24.45 / (MW * 1000)
    return (value_ugm3 * 24.45) / (mw * 1000.0)


def _ppm_to_ugm3(value_ppm: float, mw: float) -> float:
    return (value_ppm * mw * 1000.0) / 24.45


def _normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    unit = unit.strip().lower()
    unit = unit.replace("\u00b5", "u")
    unit = unit.replace("ug/m^3", "ug/m3")
    unit = unit.replace("ug/m\u00b3", "ug/m3")
    unit = unit.replace("mg/m\u00b3", "mg/m3")
    return unit


def convert_to_standard(
    pollutant: str, value: Optional[float], unit: Optional[str]
) -> Tuple[Optional[float], Optional[str]]:
    """
    Convert a concentration to the standard unit expected by BREAKPOINTS.
    Returns (value, unit) in standard units or (None, None) if conversion fails.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    if pollutant not in BREAKPOINTS:
        return None, None

    unit = _normalize_unit(unit)
    target_unit = BREAKPOINTS[pollutant]["unit"]

    if unit == target_unit:
        return float(value), target_unit

    if pollutant in ("pm25", "pm10"):
        if unit == "mg/m3":
            return float(value) * 1000.0, target_unit
        if unit == "ug/m3":
            return float(value), target_unit
        return None, None

    mw = MOLECULAR_WEIGHTS.get(pollutant)
    if mw is None:
        return None, None

    if target_unit == "ppm":
        if unit == "ppb":
            return float(value) / 1000.0, target_unit
        if unit == "ug/m3":
            return _ugm3_to_ppm(float(value), mw), target_unit
        if unit == "mg/m3":
            return _ugm3_to_ppm(float(value) * 1000.0, mw), target_unit
        if unit == "ppm":
            return float(value), target_unit

    if target_unit == "ppb":
        if unit == "ppm":
            return float(value) * 1000.0, target_unit
        if unit == "ug/m3":
            return _ugm3_to_ppm(float(value), mw) * 1000.0, target_unit
        if unit == "mg/m3":
            return _ugm3_to_ppm(float(value) * 1000.0, mw) * 1000.0, target_unit
        if unit == "ppb":
            return float(value), target_unit

    return None, None


def compute_iaqi(
    pollutant: str, concentration: Optional[float], unit: Optional[str] = None
) -> Optional[float]:
    if pollutant not in BREAKPOINTS:
        return None

    if unit is None:
        unit = BREAKPOINTS[pollutant]["unit"]

    converted, converted_unit = convert_to_standard(pollutant, concentration, unit)
    if converted is None or converted_unit is None:
        return None

    for bp in BREAKPOINTS[pollutant]["table"]:
        if bp.bp_low <= converted <= bp.bp_high:
            return ((bp.i_high - bp.i_low) / (bp.bp_high - bp.bp_low)) * (
                converted - bp.bp_low
            ) + bp.i_low

    return None


def compute_aqi_row(
    row_values: Dict[str, Optional[float]],
    units: Optional[Dict[str, Optional[str]]] = None,
) -> Optional[float]:
    iaqis: List[float] = []
    for pollutant in BREAKPOINTS.keys():
        if pollutant not in row_values:
            continue
        unit = None
        if units:
            unit = units.get(pollutant)
        iaqi = compute_iaqi(pollutant, row_values.get(pollutant), unit)
        if iaqi is not None:
            iaqis.append(iaqi)
    if not iaqis:
        return None
    return float(np.max(iaqis))


def aqi_category(aqi: Optional[float]) -> Optional[str]:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return None
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    if aqi <= 500:
        return "Hazardous"
    return "Hazardous"


def compute_aqi_dataframe(df, pollutant_cols: Optional[List[str]] = None):
    if pollutant_cols is None:
        pollutant_cols = list(BREAKPOINTS.keys())

    aqi_values = []
    for _, row in df.iterrows():
        row_values = {p: row.get(p) for p in pollutant_cols}
        aqi = compute_aqi_row(row_values)
        aqi_values.append(aqi)

    df = df.copy()
    df["aqi"] = aqi_values
    df["aqi_category"] = df["aqi"].apply(aqi_category)
    return df
