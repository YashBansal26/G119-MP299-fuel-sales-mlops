from contextlib import asynccontextmanager
import math
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from mp299_ops.prediction_logging import prediction_logger


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
LOGGER = prediction_logger()


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    highway_distance_km: float = Field(ge=0)
    nearby_pumps: int
    price_rs: float
    trucks_passing: int
    temp_c: float = Field(le=75)
    staff: int
    ownership: str
    zone: str


class PredictionResponse(BaseModel):
    prediction: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="MP299 Fuel Station Daily Sales", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    started = perf_counter()
    try:
        features = pd.DataFrame([payload.model_dump()])
        litres_per_day = float(request.app.state.model.predict(features)[0])
        if not math.isfinite(litres_per_day):
            raise ValueError("Model prediction must be finite")
        response = PredictionResponse(prediction=litres_per_day)
    except Exception as exc:
        # Exception messages may include model input; keep only the class name.
        LOGGER.error("prediction_failed", extra={
            "status": "error", "status_code": 500,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "error_type": type(exc).__name__,
        })
        raise HTTPException(status_code=500, detail="Prediction failed") from None
    LOGGER.info("prediction_served", extra={
        "status": "ok", "status_code": 200,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
    })
    return response
