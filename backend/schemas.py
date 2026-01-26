from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Pollutants(BaseModel):
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    o3: Optional[float] = None
    co: Optional[float] = None
    so2: Optional[float] = None


class Units(BaseModel):
    pm25: Optional[str] = None
    pm10: Optional[str] = None
    no2: Optional[str] = None
    o3: Optional[str] = None
    co: Optional[str] = None
    so2: Optional[str] = None


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: datetime
    pollutants: Pollutants
    units: Optional[Units] = None


class ModelInfo(BaseModel):
    best_model_name: Optional[str] = None
    input_pollutants: List[str] = []
    features: List[str] = []


class InputSummary(BaseModel):
    latitude: float
    longitude: float
    timestamp: str
    provided_pollutants: List[str]


class PredictResponse(BaseModel):
    aqi_pred: Optional[float]
    aqi_category_pred: Optional[str]
    aqi_exact: Optional[float]
    aqi_category_exact: Optional[str]
    used_model: bool
    used_exact: bool
    model_info: ModelInfo
    input_summary: InputSummary
