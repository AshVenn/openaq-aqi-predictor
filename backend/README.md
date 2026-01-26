# AQI Estimation API

FastAPI service that loads the exported AQI estimator and serves predictions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## Model artifacts

Place the exported artifacts in `backend/models/`:
- `aqi_estimator.joblib`
- `feature_cols.json`
- `model_meta.json`

Alternatively, set environment variables:
- `AQI_MODEL_PATH`
- `AQI_FEATURE_COLS_PATH`
- `AQI_MODEL_META_PATH`

## Run

From the project root:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 37.7749,
    "longitude": -122.4194,
    "timestamp": "2025-01-01T12:00:00",
    "pollutants": {
      "pm25": 12.5,
      "pm10": null,
      "no2": 8.2,
      "o3": null,
      "co": null,
      "so2": null
    },
    "units": {
      "pm25": "ug/m3",
      "no2": "ppb"
    }
  }'
```
