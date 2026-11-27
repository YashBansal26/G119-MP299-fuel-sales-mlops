from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


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
    features = pd.DataFrame([payload.model_dump()])
    litres_per_day = request.app.state.model.predict(features)[0]
    return PredictionResponse(prediction=float(litres_per_day))
