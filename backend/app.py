from __future__ import annotations

from pathlib import Path
from secrets import compare_digest
import sys

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from .config import ALLOWED_ORIGINS, API_BEARER_TOKEN, REQUIRE_API_AUTH
from .model_loader import get_artifacts
from .predict import POLLUTANTS_ALL, build_feature_frame, compute_exact_aqi
from .schemas import InputSummary, ModelInfo, PredictRequest, PredictResponse
from src.aqi import aqi_category

if REQUIRE_API_AUTH and not API_BEARER_TOKEN:
    raise RuntimeError(
        "AQI_REQUIRE_API_AUTH is enabled, but AQI_API_BEARER_TOKEN is not configured."
    )

bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_bearer_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    if not REQUIRE_API_AUTH:
        return

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not compare_digest(credentials.credentials, API_BEARER_TOKEN):
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(
    title="AQI Estimation API",
    version="1.0.0",
    dependencies=[Depends(verify_api_bearer_token)],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    artifacts = get_artifacts()
    model_loaded = artifacts.model is not None
    model_name = artifacts.meta.get("best_model_name") if artifacts.meta else None
    return {"ok": True, "model_loaded": model_loaded, "model_name": model_name}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    artifacts = get_artifacts()
    feature_cols = artifacts.feature_cols or artifacts.meta.get("features", [])
    if not feature_cols:
        raise HTTPException(status_code=500, detail="Feature columns not available.")

    input_pollutants = artifacts.meta.get("input_pollutants", POLLUTANTS_ALL)

    X, standardized, provided = build_feature_frame(request, feature_cols, input_pollutants)
    aqi_exact, aqi_category_exact = compute_exact_aqi(standardized)
    all_pollutants_provided = all(
        standardized.get(pollutant) is not None for pollutant in input_pollutants
    )

    if all_pollutants_provided and aqi_exact is not None:
        # When all pollutant inputs are present, use direct AQI calculation (max IAQI).
        aqi_pred = aqi_exact
        aqi_category_pred = aqi_category_exact
        used_model = False
    else:
        if artifacts.model is None:
            aqi_pred = None
            used_model = False
        else:
            aqi_pred = float(artifacts.model.predict(X)[0])
            used_model = True

        aqi_category_pred = aqi_category(aqi_pred) if aqi_pred is not None else None

    model_info = ModelInfo(
        best_model_name=artifacts.meta.get("best_model_name"),
        input_pollutants=artifacts.meta.get("input_pollutants", input_pollutants),
        features=artifacts.meta.get("features", feature_cols),
    )

    input_summary = InputSummary(
        latitude=request.latitude,
        longitude=request.longitude,
        timestamp=request.timestamp.isoformat(),
        provided_pollutants=provided,
    )

    return PredictResponse(
        aqi_pred=aqi_pred,
        aqi_category_pred=aqi_category_pred,
        aqi_exact=aqi_exact,
        aqi_category_exact=aqi_category_exact,
        used_model=used_model,
        used_exact=aqi_exact is not None,
        model_info=model_info,
        input_summary=input_summary,
    )
