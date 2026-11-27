from datetime import datetime
from io import StringIO
import json
import logging
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

import app.main as service
from mp299_ops.prediction_logging import PredictionJsonFormatter, prediction_logger


PAYLOAD = {
    "highway_distance_km": 2.5, "nearby_pumps": 4, "price_rs": 96.5,
    "trucks_passing": 350, "temp_c": 32, "staff": 6,
    "ownership": "private", "zone": "Highway",
}


@pytest.fixture
def observed_service(monkeypatch):
    model = Mock(spec=["predict"])
    model.predict.return_value = [16220.0]
    monkeypatch.setattr(service.joblib, "load", lambda path: model)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(PredictionJsonFormatter())
    logger = logging.getLogger("mp299.test.predictions")
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    logger.setLevel(logging.INFO)
    monkeypatch.setattr(service, "LOGGER", logger)
    with TestClient(service.app, raise_server_exceptions=False) as client:
        yield client, model, stream


def test_each_prediction_produces_one_json_event(observed_service):
    client, model, stream = observed_service
    for _ in range(2):
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 200
        assert response.json() == {"prediction": 16220.0}
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(events) == model.predict.call_count == 2
    for event in events:
        assert event["event"] == "prediction_served"
        assert event["status"] == "ok" and event["status_code"] == 200
        assert event["duration_ms"] >= 0
        assert datetime.fromisoformat(event["timestamp"]).utcoffset().total_seconds() == 0
        assert set(event) == {"event", "timestamp", "level", "status", "status_code", "duration_ms"}
    assert not set(PAYLOAD).intersection(events[0])


@pytest.mark.parametrize("change", [
    {"temp_c": 75.01}, {"highway_distance_km": -0.01},
    {"customer_phone": "private-contact-value"},
])
def test_rejected_requests_do_not_invoke_or_log_predictions(observed_service, change):
    client, model, stream = observed_service
    assert client.post("/predict", json=PAYLOAD | change).status_code == 422
    model.predict.assert_not_called()
    assert stream.getvalue() == ""


def test_missing_input_and_health_have_no_prediction_log(observed_service):
    client, model, stream = observed_service
    missing = {key: value for key, value in PAYLOAD.items() if key != "staff"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.get("/health").status_code == 200
    model.predict.assert_not_called()
    assert stream.getvalue() == ""


def test_failure_logs_no_payload_or_exception_message(observed_service):
    client, model, stream = observed_service
    model.predict.side_effect = RuntimeError("private customer information")
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed"}
    event = json.loads(stream.getvalue())
    assert event["event"] == "prediction_failed"
    assert event["status_code"] == 500 and event["error_type"] == "RuntimeError"
    assert "private customer information" not in stream.getvalue()
    assert "Highway" not in stream.getvalue()


@pytest.mark.parametrize("prediction", [float("nan"), float("inf")])
def test_non_finite_prediction_cannot_be_logged_as_success(observed_service, prediction):
    client, model, stream = observed_service
    model.predict.return_value = [prediction]
    assert client.post("/predict", json=PAYLOAD).status_code == 500
    assert json.loads(stream.getvalue())["event"] == "prediction_failed"


def test_logger_writes_json_to_stdout_once_when_configured_twice(monkeypatch):
    stream = StringIO()
    monkeypatch.setattr("sys.stdout", stream)
    logger = logging.getLogger("mp299.predictions")
    monkeypatch.setattr(logger, "handlers", [])
    configured = prediction_logger()
    assert prediction_logger() is configured
    assert len(configured.handlers) == 1
    assert configured.handlers[0].stream is stream
    configured.info("prediction_served", extra={"status": "ok", "status_code": 200, "duration_ms": 1.0})
    assert len(stream.getvalue().splitlines()) == 1
    assert json.loads(stream.getvalue())["event"] == "prediction_served"
