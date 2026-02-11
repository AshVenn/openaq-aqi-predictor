# AQI Estimation API

FastAPI service that loads the exported AQI estimator and serves predictions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

## API auth (Bearer token)

Create `backend/.env` (auto-loaded on startup):

```bash
AQI_API_BEARER_TOKEN=replace-with-a-long-random-token
AQI_REQUIRE_API_AUTH=true
```

You can start from `backend/.env.example`.

Optional:
- `AQI_REQUIRE_API_AUTH=true` (default) requires bearer auth on all endpoints.
- `AQI_REQUIRE_API_AUTH=false` disables auth (local dev only).

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

## Run with Docker

Build from the project root:

```bash
docker build -f backend/Dockerfile -t aqi-api:latest .
```

Run:

```bash
docker run -d --name aqi-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e AQI_API_BEARER_TOKEN=replace-with-a-long-random-token \
  -e AQI_REQUIRE_API_AUTH=true \
  aqi-api:latest
```

## Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer replace-with-a-long-random-token" \
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

## Frontend API call example

```js
const API_BASE_URL = "http://localhost:8000";
const API_TOKEN = import.meta.env.VITE_AQI_API_BEARER_TOKEN;

async function predictAqi(payload) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_TOKEN}`,
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Predict failed: ${response.status}`);
  }

  return response.json();
}
```
