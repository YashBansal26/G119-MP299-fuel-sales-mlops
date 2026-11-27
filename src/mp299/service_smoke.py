"""Verify one API prediction against both the shipped pipeline and v1 champion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from app.main import app
from mp299.config import ROOT

PAYLOAD = {
    "highway_distance_km": 2.5, "nearby_pumps": 4, "price_rs": 96.5,
    "trucks_passing": 350, "temp_c": 32, "staff": 6,
    "ownership": "private", "zone": "Highway",
}


def smoke_test(base_url: str | None = None) -> dict:
    """Raise on HTTP, model identity or prediction mismatch; never retrain."""
    summary = json.loads((ROOT / "reports" / "training_results.json").read_text())
    model_path = ROOT / "models" / "model.joblib"
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != summary["model_sha256"]:
        raise AssertionError("Service model file differs from the verified v1 artifact")
    features = pd.DataFrame([PAYLOAD])
    file_prediction = float(joblib.load(model_path).predict(features)[0])
    old_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
        client = MlflowClient()
        champion = client.get_model_version_by_alias(summary["registered_model_name"], "champion")
        if int(champion.version) != 1 or champion.run_id != summary["best"]["run_id"]:
            raise AssertionError("champion must refer to the verified Stage 2 v1 run")
        pipeline = mlflow.sklearn.load_model(f"models:/{champion.name}@champion")
        champion_prediction = float(pipeline.predict(features)[0])
    finally:
        mlflow.set_tracking_uri(old_uri)
    transport = (httpx.Client(base_url=base_url, timeout=15, trust_env=False)
                 if base_url else TestClient(app))
    with transport as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json() != {"status": "ok"}:
            raise AssertionError("Unexpected health response")
        response = client.post("/predict", json=PAYLOAD)
        response.raise_for_status()
        prediction = response.json()["prediction"]
    if not isinstance(prediction, (int, float)) or not math.isfinite(prediction):
        raise AssertionError("API must return a finite numeric prediction")
    for expected in (file_prediction, champion_prediction):
        if not math.isclose(prediction, expected, rel_tol=1e-10, abs_tol=1e-6):
            raise AssertionError("API prediction differs from the v1 model")
    return {
        "status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "http" if base_url else "TestClient", "request": PAYLOAD,
        "health_status": health.status_code, "health_response": health.json(),
        "prediction_status": response.status_code, "api_prediction": prediction,
        "joblib_prediction": file_prediction, "champion_prediction": champion_prediction,
        "predictions_match": True, "registered_model_name": champion.name,
        "champion_version": int(champion.version), "champion_run_id": champion.run_id,
        "model_sha256": model_hash,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Optional running API URL; otherwise use TestClient")
    parser.add_argument("--output", type=Path, help="Save smoke-test evidence as JSON")
    args = parser.parse_args()
    result = smoke_test(args.url)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
