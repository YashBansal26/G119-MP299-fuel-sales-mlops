# G119 MP299 — Stage 3 Ops integration

Aryan Bansal (`Aryan-bnsl`, S24CSEU1801), 27 November 2026.
Prepared with AI coding assistance; review the implementation before viva.

## Scope and team handoff

This contribution adds operational checks around the existing service.
Yash's data contract/cleaning, Aaryan's API and Dockerfile, and Sakshi's v1
champion/registry evidence retain their existing ownership and history.
The trained artifact, raw data, MLflow runs and findings are not regenerated.
Stage 4 drift monitoring, retraining and holdout delivery remain separate work.

## Prediction logging

`src/mp299_ops/prediction_logging.py` configures an isolated Python logger
with one stdout handler and a JSON formatter. Repeated configuration does
not duplicate the handler. It does not reconfigure Uvicorn or the root logger.

`app/main.py` records `prediction_served` only after inference has produced
a finite numeric prediction. Each event contains UTC `timestamp`, `level`,
`status`, `status_code` and `duration_ms`, measured with a monotonic clock.
It contains no request body, feature values, customer names or phone numbers.

Pydantic rejects invalid requests before inference; they have no prediction
event. Exceptions and non-finite outputs produce `prediction_failed` with
the exception class name, not its potentially sensitive message. The client
receives a generic HTTP 500 response, preventing Uvicorn from logging the
original inference exception as an unhandled traceback.

`tests/test_prediction_logging.py` uses a model double to verify event count,
JSON parsing, timestamp/latency, stdout destination, duplicate-handler
prevention, failed inference, non-finite predictions, missing/extra fields,
and both MP299 client-rule rejections. Existing integration tests still use
the real saved champion and MLflow registry.

## Reproduce local checks

Use Python 3.12 from the repository root:

```sh
python -m pip install -r requirements.txt
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest -q
PYTHONPATH=src python -m mp299_ops.deployment
PYTHONPATH=src python -m mp299.service_smoke
```

No raw train/later CSVs are needed for these tests. The existing real-model
integration tests need the committed `models/model.joblib`, `mlflow.db` and
`tracking_artifacts/`; keep those files together in a checkout.

For a running API, use `PYTHONPATH=src python -m mp299.service_smoke --url
http://127.0.0.1:8000`. This reuses Sakshi's artifact fingerprint and champion
identity checks and does not retrain or promote any model.

## CI and container check

The push-triggered workflow preserves the existing pinned action revisions
and dependency pins. It runs these steps in order:

1. Install requirements and run the combined pytest suite.
2. Check the Kubernetes YAML names, image, replicas, selectors, probes and ports.
3. Build `mp299-api:1.0` from the Service lead's Dockerfile.
4. Start it with port 8000 bound only to the runner's loopback address.
5. Retry `/health` for startup, then compare a real HTTP prediction with the
   shipped joblib model and MLflow v1 champion using `mp299.service_smoke`.
6. Print container logs and remove the CI container, including after failure.

To repeat the container check on a machine with Docker:

```sh
docker build -t mp299-api:1.0 .
docker run --detach --name mp299-api-local --publish 127.0.0.1:8000:8000 mp299-api:1.0
curl --fail --retry 30 --retry-connrefused --retry-delay 2 --retry-max-time 90 http://127.0.0.1:8000/health
PYTHONPATH=src python -m mp299.service_smoke --url http://127.0.0.1:8000
docker logs mp299-api-local
docker rm --force mp299-api-local
```

The local editing environment has no Docker executable. Container build and
HTTP smoke-test evidence comes from the GitHub Actions run for the published
commit; inspect its individual steps. A local pytest pass alone does not
establish that the image built. No Kubernetes cluster application is claimed.

## Viva in Hinglish

- Logger Python ka `logging` module use karta hai. Formatter event ko JSON
  line banata hai, aur handler stdout par likhta hai.
- Successful prediction ke baad hi success event aata hai. 422 request model
  tak nahi jaati. Error par raw exception message log nahi karte.
- `perf_counter()` elapsed time measure karta hai; UTC timestamp event ka
  clock time batata hai. Dono ka purpose alag hai.
- Dockerfile mein requirements code se pehle install hote hain, isliye code
  edit hone par dependency layer reuse ho sakti hai.
- Readiness traffic ke liye hai; liveness unhealthy container restart ke liye.
  Dono `/health` use karte hain, lekin liveness ka initial delay zyada hai.
- CI mein pytest ke baad actual image build aur HTTP prediction bhi check
  hoti hai. Isse code import hone aur container chalne, dono ka evidence milta hai.
