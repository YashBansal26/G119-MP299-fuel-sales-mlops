import math

import pytest
from fastapi.testclient import TestClient

from app.main import app


PAYLOAD = {
    "highway_distance_km": 2.5,
    "nearby_pumps": 4,
    "price_rs": 96.5,
    "trucks_passing": 350,
    "temp_c": 32,
    "staff": 6,
    "ownership": "private",
    "zone": "Highway",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("payload", [PAYLOAD, {**PAYLOAD, "temp_c": 75, "highway_distance_km": 0}])
def test_predict_with_shipped_model(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    assert isinstance(prediction, float)
    assert math.isfinite(prediction)


@pytest.mark.parametrize(
    ("field", "value"),
    [("temp_c", 75.1), ("highway_distance_km", -0.1), ("record_id", "extra")],
)
def test_rejects_invalid_payload(client, field, value):
    response = client.post("/predict", json={**PAYLOAD, field: value})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", field] for error in response.json()["detail"])


def test_rejects_missing_field(client):
    response = client.post("/predict", json={field: value for field, value in PAYLOAD.items() if field != "zone"})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", "zone"] for error in response.json()["detail"])
