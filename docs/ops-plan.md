# G119 MP299 operations preparation

Owner: Aryan Bansal (`Aryan-bnsl`, S24CSEU1801)  
Date: 20 November 2026

## Work prepared for Stage 1

- Added the push-triggered GitHub Actions workflow. It installs the pinned
  requirements and runs pytest on Python 3.12 without requiring raw CSV files.
- Prepared Kubernetes Deployment and Service manifests using the exact MP299
  names, image, replica count, ports and health-check requirements.
- Added a reusable deployment-contract checker and seven tests. The negative
  tests catch mismatched selectors, wrong image/replicas, incorrect routing,
  missing health probes and premature liveness checks.

This is Ops preparation. The team's Stage 1 completion also depends on the
Data lead committing DVC metadata, the seven data-check implementations and
the completed `data_issues` section of `reports/findings.yaml`.

## Deployment contract

| Setting | Required value |
| --- | --- |
| Image | `mp299-api:1.0` |
| Deployment | `mp299-api` |
| Service | `mp299-svc` |
| Replicas | 4 |
| Container port | 8000 |
| Service routing | 80 to 8000 |
| Readiness | HTTP `/health`, initial delay 10 seconds |
| Liveness | HTTP `/health`, initial delay 30 seconds |

The Service selects the Deployment's Pod labels. The longer liveness delay
allows startup time before Kubernetes starts restarting an unhealthy process.
The manifests are configuration artifacts; the API image and a live deployment
have not been built or tested as part of this preparation. The handbook does
not require running a Kubernetes cluster.

## Run the checks

From the repository root:

```sh
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m mp299_ops.deployment
```

These checks validate the configuration. They do not test a trained model,
API requests, a container or a live Kubernetes deployment.

## Handoff to the other roles

- **Data lead:** preserve the original `train.csv` and `later.csv` bytes, track
  them with DVC, keep their raw CSVs out of Git, and commit the findings and
  investigation code. These Ops tests do not access the data.
- **Model lead:** add pinned training dependencies to `requirements.txt` as
  needed. Keep the original `data_sha256`, MLflow runs, registry and champion
  model consistent with the handbook.
- **Service lead:** provide `GET /health` and `POST /predict`, the request
  schema rules, TestClient tests, Dockerfile and the `mp299-api:1.0` image.
  Listen on `0.0.0.0:8000`. The current Ops tests do not replace API tests.
- **Ops lead:** keep CI passing as the team adds tests and dependencies; then
  add JSON prediction logging and the Stage 4 monitoring/retraining checks.

## Later Ops work

Stage 3 requires a green GitHub Actions run with the team's full test suite,
API integration and JSON logs to stdout. Stage 4 requires 10-bin PSI on the
six numeric input features, v1 MAE for each zone, and a time-based retraining
comparison. Use the first 70 percent of date-sorted `later.csv` for updating
v2 and the last 30 percent for comparing v1 and v2. Promote v2 only when its
MAE is at least 15 percent lower, then record the actual results in
`reports/findings.yaml` with the Model lead.

Future monitoring results and deployment outcomes are not claimed here.
Implementation and documentation were prepared with AI coding assistance;
the owner should review and understand the code before the individual viva.
