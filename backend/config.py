from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

load_dotenv(BACKEND_DIR / ".env")

MODEL_DIR = BACKEND_DIR / "models"

MODEL_PATH = Path(os.getenv("AQI_MODEL_PATH", MODEL_DIR / "aqi_estimator.joblib"))
FEATURE_COLS_PATH = Path(
    os.getenv("AQI_FEATURE_COLS_PATH", MODEL_DIR / "feature_cols.json")
)
MODEL_META_PATH = Path(os.getenv("AQI_MODEL_META_PATH", MODEL_DIR / "model_meta.json"))

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

API_BEARER_TOKEN = os.getenv("AQI_API_BEARER_TOKEN", "").strip()
REQUIRE_API_AUTH = os.getenv("AQI_REQUIRE_API_AUTH", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
