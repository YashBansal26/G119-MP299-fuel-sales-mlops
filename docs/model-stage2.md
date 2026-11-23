# Stage 2 model handoff

Model lead: Sakshi (S24CSEU0449). Completed 23 November 2026.
Implementation and documentation were prepared with AI assistance.

## Result

Target: `litres_per_day`. Metric: validation mean absolute error, in L/day.

| Setting | Validation MAE |
| --- | ---: |
| Ridge, alpha 0.1 | 1010.404 |
| Ridge, alpha 1 | 1010.103 |
| **Ridge, alpha 10** | **1008.179** |
| Random forest, 100 trees, depth 8 | 1424.913 |
| Random forest, 200 trees, depth 12 | 1403.170 |
| Gradient boosting, 100 trees, depth 2 | 1158.358 |
| Median baseline | 4471.195 |

All six settings use the same 800 training and 200 validation rows, selected
with `random_state=42`, a 20% validation fraction, and stratification by zone.
The saved model is the exact pipeline evaluated on validation; it is not
refitted on validation rows. This shuffled split is an initial development
comparison. Evaluation on later dates remains Stage 4 work.

Original `train.csv` SHA256:
`7d5e7fb87a385182fc2a2372e6cd4a5f8b48220316b37160299e895cc8e86711`.

## Inputs and cleaning

The pipeline selects exactly these eight features:

`highway_distance_km`, `nearby_pumps`, `price_rs`, `trucks_passing`,
`temp_c`, `staff`, `ownership`, `zone`.

Record ID, date and target are excluded from model features. Training applies
the configured fixed stuck-sensor and target-typo repairs before splitting.
The pipeline converts `staff=999` to missing, strips category whitespace,
imputes numeric medians and categorical modes, scales numeric features and
one-hot encodes categories. All learned statistics come only from training
rows. Unknown categories are accepted by the encoder. The original input
DataFrame and raw CSV bytes are preserved.

For data containing dates in the known stuck-sensor window, apply that fixed
date-based repair before selecting the eight model inputs. The standalone
pipeline has no date input and cannot infer a sensor outage from a count alone.
Do not use the globally imputed exploration CSV for split evaluation.

## MLflow and artifacts

| Item | Value |
| --- | --- |
| Tracking backend | `sqlite:///mlflow.db` |
| Experiment | `G119-MP299-fuel-sales` |
| Registered model | `mp299-fuel-sales` |
| Version / alias | `1` / `champion` |
| Best run | `a27d339a2edd44a4b2ccb441cec32afb` |
| Service artifact | `models/model.joblib` |
| Exact run results | `reports/training_results.json` |

Each run logs settings, `data_sha256`, `split_sha256`, `mae`, baseline MAE,
package versions and split record IDs. The winning run also contains an
MLflow model, signature, source snapshot and dependency metadata.
`tracking_artifacts/` contains the referenced artifacts and is committed
alongside the SQLite backend. Its relative paths support relocation.

Run from the repository root after installing the pinned requirements:

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Load the service artifact with:

```python
import joblib
model = joblib.load("models/model.joblib")
predictions = model.predict(feature_dataframe)
```

Alternatively, load the registry model from the repository root:

```python
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/mp299-fuel-sales@champion")
```

Keep `src` on `PYTHONPATH` when using the joblib artifact. The registry model
also bundles its custom pipeline source.

## Verification and next handoff

The automated tests check training-only imputation, exclusion of metadata and
target columns, unseen categories, missing-feature rejection, serialization,
logged scores and hashes, findings consistency, and registry loading after
relocation. Existing Data and Ops tests remain part of the combined suite.

The Service lead can now load the pipeline for the API. Raw training and later
CSVs remain DVC-managed. No later-data evaluation, retraining, holdout
predictions, API deployment or live Kubernetes result is claimed here.
