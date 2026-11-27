# Stage 3 Model Lead integration check

Owner: Sakshi (S24CSEU0449). Verified 27 November 2026.
Prepared with AI assistance.

The Service lead's `service-lead/stage-3` implementation was combined with
the current `main`, including the Data lead's validation changes. The v1
model was preserved and no retraining was performed.

## Verified result

| Check | Observed result |
| --- | --- |
| `GET /health` | HTTP 200, `{"status":"ok"}` |
| `POST /predict` | HTTP 200, finite numeric prediction |
| API prediction | 16220.040503070559 L/day |
| Direct `models/model.joblib` prediction | 16220.040503070559 L/day |
| MLflow `mp299-fuel-sales@champion` prediction | 16220.040503070559 L/day |
| Champion version | 1 |
| Champion run | `a27d339a2edd44a4b2ccb441cec32afb` |
| Combined local test suite | 31 passed |

The model file's SHA256 matches `reports/training_results.json`. The smoke
check also checks the alias version and source run, so matching a sample
prediction alone is not the model identity check.

Request used for both TestClient and a real HTTP request to a locally running
Uvicorn service:

```json
{
  "highway_distance_km": 2.5,
  "nearby_pumps": 4,
  "price_rs": 96.5,
  "trucks_passing": 350,
  "temp_c": 32,
  "staff": 6,
  "ownership": "private",
  "zone": "Highway"
}
```

Machine-readable evidence:

- `reports/stage3_model_smoke.json`: TestClient request and model comparison.
- `reports/stage3_model_smoke_http.json`: actual HTTP request and model comparison,
  with the two source commit IDs used for the integration.

## Repeat the acceptance check

From the repository root with the pinned requirements installed:

```powershell
$env:PYTHONPATH="src"
python -m mp299.service_smoke
python -m pytest -q
```

To check an already running API, pass its URL:

```powershell
python -m mp299.service_smoke --url http://127.0.0.1:8000
```

`--output reports/stage3_model_smoke.json` optionally saves fresh evidence.
The command exits with an error for failed HTTP requests, a changed v1 model
hash, an incorrect champion version/run, or prediction mismatches. The new
`tests/test_service_model_integration.py` runs the same acceptance check in CI.

This confirms model integration through the prediction service. The recorded
HTTP check used local Uvicorn; it does not claim a Docker or Kubernetes deployment.
